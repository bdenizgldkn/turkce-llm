#include "gpu_train.h"
#include "rope.h"
#include "../runtime/prng.h"
#include "../runtime/mathlib.h"
#include "../tensor/tensor.h"
#include "../runtime/strutil.h"

/* lm_collect_params sirasi: [0]=embed_table, katman l icin 1+10*l+k:
 * k=0 attn_norm_w, 1 w_qkv, 2 b_qkv, 3 wo, 4 bo, 5 ffn_norm_w,
 * 6 w_gate_up, 7 b_gate_up, 8 w_down, 9 b_down; son: final_norm_w. */
enum { PK_ATTN_NORM = 0, PK_WQKV, PK_BQKV, PK_WO, PK_BO, PK_FFN_NORM, PK_WGU, PK_BGU, PK_WDOWN, PK_BDOWN };

static u32 pidx(u32 layer, u32 k) { return 1u + 10u * layer + k; }

#define F4 ((u64)sizeof(f32))

static CUdeviceptr P_(const GpuTrainer* g, u32 i) { return g->params + g->offsets[i] * F4; }
static CUdeviceptr G_(const GpuTrainer* g, u32 i) { return g->grads + g->offsets[i] * F4; }

static CUdeviceptr dalloc(u64 floats) { return cuda_alloc(floats * F4); }

/* ---------------- baslatma yardimcilari ---------------- */

#define THREADS_1D 256u

static void launch_1d(CUfunction f, u64 n, void** args) {
    u32 grid = (u32)((n + THREADS_1D - 1) / THREADS_1D);
    cuda_launch(f, grid, 1, 1, THREADS_1D, 1, 1, 0, args);
}

/* Satir basina bir blok; blok boyutu 2'nin kuvveti (block_sum sozlesmesi). */
static void launch_rows(CUfunction f, u32 rows, u32 threads, void** args) {
    cuda_launch(f, rows, 1, 1, threads, 1, 1, 0, args);
}

static u32 pow2_threads(u32 n) {
    u32 t = 32;
    while (t < n && t < 256) t <<= 1;
    return t;
}

typedef enum { GEMM_NN, GEMM_NT, GEMM_TN } GemmKind;

/* C[M,N] = alpha*op(A)@op(B) + beta*C, batch adet (bkz. train_kernels.cu). */
static void gemm(GpuTrainer* g, GemmKind kind, CUdeviceptr A, CUdeviceptr B, CUdeviceptr C,
                 i32 M, i32 N, i32 K, i32 lda, i32 ldb, i32 ldc,
                 u32 batch, i64 sA, i64 sB, i64 sC, f32 alpha, f32 beta) {
    CUfunction f = (kind == GEMM_NN) ? g->k_gemm_nn : (kind == GEMM_NT) ? g->k_gemm_nt : g->k_gemm_tn;
    void* args[] = { &A, &B, &C, &M, &N, &K, &lda, &ldb, &ldc, &sA, &sB, &sC, &alpha, &beta };
    cuda_launch(f, (u32)((N + 63) / 64), (u32)((M + 63) / 64), batch, 16, 16, 1, 0, args);
}

/* Tek (batch'siz) GEMM kisayolu, bitisik matrisler icin. */
static void gemm1(GpuTrainer* g, GemmKind kind, CUdeviceptr A, CUdeviceptr B, CUdeviceptr C,
                  i32 M, i32 N, i32 K, f32 beta) {
    i32 lda, ldb;
    switch (kind) {
        case GEMM_NN: lda = K; ldb = N; break;
        case GEMM_NT: lda = K; ldb = K; break;
        default:      lda = M; ldb = N; break; /* TN: A saklanis [K,M] */
    }
    gemm(g, kind, A, B, C, M, N, K, lda, ldb, N, 1, 0, 0, 0, 1.0f, beta);
}

static void add_bias(GpuTrainer* g, CUdeviceptr x, CUdeviceptr b, i32 rows, i32 cols) {
    void* args[] = { &x, &b, &rows, &cols };
    launch_1d(g->k_add_bias_rows, (u64)rows * (u64)cols, args);
}

static void add(GpuTrainer* g, CUdeviceptr a, CUdeviceptr b, CUdeviceptr out, i64 n) {
    void* args[] = { &a, &b, &out, &n };
    launch_1d(g->k_add, (u64)n, args);
}

static void colsum_acc(GpuTrainer* g, CUdeviceptr in, CUdeviceptr out, i32 rows, i32 cols) {
    void* args[] = { &in, &out, &rows, &cols };
    cuda_launch(g->k_colsum_acc, (u32)((cols + 31) / 32), 1, 1, 32, 8, 1, 0, args);
}

static void rmsnorm_fwd(GpuTrainer* g, CUdeviceptr x, CUdeviceptr w, CUdeviceptr out, CUdeviceptr rinv) {
    i32 cols = (i32)g->D;
    f32 eps = g->eps;
    void* args[] = { &x, &w, &out, &rinv, &cols, &eps };
    launch_rows(g->k_rmsnorm_fwd, g->N, pow2_threads(g->D), args);
}

/* dx += rmsnorm geri; dw += sum_n (dy*x*r) */
static void rmsnorm_bwd(GpuTrainer* g, CUdeviceptr x, CUdeviceptr w, CUdeviceptr rinv, CUdeviceptr dy,
                        CUdeviceptr dx, CUdeviceptr dw) {
    i32 cols = (i32)g->D;
    CUdeviceptr wtmp = g->wtmp;
    void* args[] = { &x, &w, &rinv, &dy, &dx, &wtmp, &cols };
    launch_rows(g->k_rmsnorm_bwd, g->N, pow2_threads(g->D), args);
    colsum_acc(g, g->wtmp, dw, (i32)g->N, (i32)g->D);
}

/* ---------------- olusturma / yok etme ---------------- */

GpuTrainer gpu_trainer_create(Allocator* alloc, const char* ptx_path, const LMModel* model,
                              Node** params, u32 num_params, u32 batch_size, u32 seq_len) {
    GpuTrainer g;
    mem_set(&g, 0, sizeof(g));

    g.V = model->vocab_size; g.D = model->d_model; g.H = model->num_heads; g.L = model->num_layers;
    g.F = (u32)model->blocks[0].ffn.w_down->value.shape[0];
    g.T = seq_len; g.B = batch_size; g.N = batch_size * seq_len;
    g.hd = g.D / g.H;
    g.eps = model->eps;

    g.cuda = cuda_init(alloc, ptx_path);
    g.k_gemm_nn = cuda_get_kernel(&g.cuda, "k_gemm_nn");
    g.k_gemm_nt = cuda_get_kernel(&g.cuda, "k_gemm_nt");
    g.k_gemm_tn = cuda_get_kernel(&g.cuda, "k_gemm_tn");
    g.k_add_bias_rows = cuda_get_kernel(&g.cuda, "k_add_bias_rows");
    g.k_add = cuda_get_kernel(&g.cuda, "k_add");
    g.k_colsum_acc = cuda_get_kernel(&g.cuda, "k_colsum_acc");
    g.k_rmsnorm_fwd = cuda_get_kernel(&g.cuda, "k_rmsnorm_fwd");
    g.k_rmsnorm_bwd = cuda_get_kernel(&g.cuda, "k_rmsnorm_bwd");
    g.k_qkv_rope_split = cuda_get_kernel(&g.cuda, "k_qkv_rope_split");
    g.k_qkv_rope_merge_bwd = cuda_get_kernel(&g.cuda, "k_qkv_rope_merge_bwd");
    g.k_attn_softmax_fwd = cuda_get_kernel(&g.cuda, "k_attn_softmax_fwd");
    g.k_attn_softmax_bwd = cuda_get_kernel(&g.cuda, "k_attn_softmax_bwd");
    g.k_heads_merge = cuda_get_kernel(&g.cuda, "k_heads_merge");
    g.k_heads_split = cuda_get_kernel(&g.cuda, "k_heads_split");
    g.k_swiglu_fwd = cuda_get_kernel(&g.cuda, "k_swiglu_fwd");
    g.k_swiglu_bwd = cuda_get_kernel(&g.cuda, "k_swiglu_bwd");
    g.k_embed_fwd = cuda_get_kernel(&g.cuda, "k_embed_fwd");
    g.k_embed_bwd = cuda_get_kernel(&g.cuda, "k_embed_bwd");
    g.k_cross_entropy = cuda_get_kernel(&g.cuda, "k_cross_entropy");
    g.k_adam = cuda_get_kernel(&g.cuda, "k_adam");

    /* Parametre yerlesimi: lm_collect_params sirasiyla tek duz tampon. */
    g.num_params = num_params;
    u64 off = 0;
    for (u32 i = 0; i < num_params; i++) {
        g.offsets[i] = off;
        g.numels[i] = params[i]->value.numel;
        off += params[i]->value.numel;
    }
    g.total_scalars = off;
    g.params = dalloc(off);
    g.grads = dalloc(off);
    g.adam_m = dalloc(off);
    g.adam_v = dalloc(off);
    cuda_memset_zero(g.adam_m, off * F4);
    cuda_memset_zero(g.adam_v, off * F4);

    u64 N = g.N, D = g.D, F = g.F, T = g.T;
    u64 BHT = (u64)g.B * g.H * T;
    for (u32 l = 0; l < g.L; l++) {
        GpuLayerActs* a = &g.layers[l];
        a->x_in = dalloc(N * D);
        a->n1 = dalloc(N * D);
        a->r1 = dalloc(N);
        a->Q = dalloc(N * D); a->K = dalloc(N * D); a->V = dalloc(N * D);
        a->P = dalloc(BHT * T);
        a->att = dalloc(N * D);
        a->x2 = dalloc(N * D);
        a->n2 = dalloc(N * D);
        a->r2 = dalloc(N);
        a->gu = dalloc(N * 2 * F);
        a->h = dalloc(N * F);
    }
    g.x_final = dalloc(N * D);
    g.nf = dalloc(N * D);
    g.rf = dalloc(N);
    g.logits = dalloc(N * (u64)g.V);

    g.dres = dalloc(N * D);
    g.dn = dalloc(N * D);
    g.wtmp = dalloc(N * D);
    g.qkv = dalloc(N * 3 * D);
    g.dqkv = dalloc(N * 3 * D);
    g.proj = dalloc(N * D);
    g.datt = dalloc(N * D);
    g.dQ = dalloc(N * D); g.dK = dalloc(N * D); g.dV = dalloc(N * D); g.dO = dalloc(N * D);
    g.Obuf = dalloc(N * D);
    g.dP = dalloc(BHT * T);
    g.dh = dalloc(N * F);
    g.dgu = dalloc(N * 2 * F);

    g.ids = cuda_alloc(N * sizeof(u32));
    g.targets = cuda_alloc(N * sizeof(u32));
    g.loss_rows = dalloc(N);
    g.host_loss_rows = (f32*)allocator_alloc(alloc, N * F4);

    /* RoPE tablolari (CPU'daki rope_build_tables ile ayni) */
    u64 cshape[2] = { T, g.hd / 2 };
    Tensor cos_t = tensor_create(alloc, cshape, 2);
    Tensor sin_t = tensor_create(alloc, cshape, 2);
    rope_build_tables(&cos_t, &sin_t, T, g.hd, 10000.0f);
    g.cos_t = dalloc(cos_t.numel);
    g.sin_t = dalloc(sin_t.numel);
    cuda_h2d(g.cos_t, cos_t.data, cos_t.numel * F4);
    cuda_h2d(g.sin_t, sin_t.data, sin_t.numel * F4);

    gpu_trainer_upload(&g, params, NULL_PTR);
    return g;
}

void gpu_trainer_destroy(GpuTrainer* g) {
    /* Baglami yok etmek, o baglamdaki tum cihaz bellegini serbest birakir. */
    cuda_shutdown(&g->cuda);
}

/* ---------------- CPU aynasi <-> GPU ---------------- */

void gpu_trainer_upload(GpuTrainer* g, Node** params, const AdamOptimizer* opt) {
    cuda_set_current(&g->cuda);
    for (u32 i = 0; i < g->num_params; i++) {
        cuda_h2d(P_(g, i), params[i]->value.data, g->numels[i] * F4);
        if (opt != NULL_PTR) {
            cuda_h2d(g->adam_m + g->offsets[i] * F4, opt->m[i].data, g->numels[i] * F4);
            cuda_h2d(g->adam_v + g->offsets[i] * F4, opt->v[i].data, g->numels[i] * F4);
        }
    }
}

void gpu_trainer_download(GpuTrainer* g, Node** params, AdamOptimizer* opt) {
    cuda_set_current(&g->cuda);
    for (u32 i = 0; i < g->num_params; i++) {
        cuda_d2h(params[i]->value.data, P_(g, i), g->numels[i] * F4);
        if (opt != NULL_PTR) {
            cuda_d2h(opt->m[i].data, g->adam_m + g->offsets[i] * F4, g->numels[i] * F4);
            cuda_d2h(opt->v[i].data, g->adam_v + g->offsets[i] * F4, g->numels[i] * F4);
        }
    }
}

void gpu_trainer_download_grads(GpuTrainer* g, Node** params) {
    cuda_set_current(&g->cuda);
    for (u32 i = 0; i < g->num_params; i++) {
        cuda_d2h(params[i]->grad.data, G_(g, i), g->numels[i] * F4);
    }
}

void gpu_trainer_upload_grads(GpuTrainer* g, Node** params) {
    cuda_set_current(&g->cuda);
    for (u32 i = 0; i < g->num_params; i++) {
        cuda_h2d(G_(g, i), params[i]->grad.data, g->numels[i] * F4);
    }
}

void gpu_trainer_sample_batch(const u32* tokens, u64 num_tokens, u32 seq_len, u32 batch_size,
                              u64 seed_base, u32* ids_out, u32* targets_out) {
    i64 max_start = (i64)(num_tokens - seq_len - 1);
    for (u32 w = 0; w < batch_size; w++) {
        PCGState rng = pcg_seed(seed_base * 1000003ull + w, 99ull + w);
        i64 start = pcg_range_i64(&rng, 0, max_start);
        for (u32 t = 0; t < seq_len; t++) {
            ids_out[(u64)w * seq_len + t] = tokens[start + t];
            targets_out[(u64)w * seq_len + t] = tokens[start + t + 1];
        }
    }
}

/* ---------------- ileri + geri ---------------- */

f32 gpu_trainer_forward_backward(GpuTrainer* g, const u32* ids, const u32* targets) {
    cuda_set_current(&g->cuda);
    i32 N = (i32)g->N, D = (i32)g->D, F = (i32)g->F, T = (i32)g->T, V = (i32)g->V;
    i32 Bi = (i32)g->B, Hi = (i32)g->H, hd = (i32)g->hd;
    u32 BH = g->B * g->H;
    i64 sTT = (i64)T * T, sThd = (i64)T * hd;
    f32 scale = 1.0f / m_sqrtf((f32)g->hd); /* attention.c ile ayni */
    u32 att_threads = pow2_threads(g->T);

    cuda_h2d(g->ids, ids, (u64)N * sizeof(u32));
    cuda_h2d(g->targets, targets, (u64)N * sizeof(u32));
    cuda_memset_zero(g->grads, g->total_scalars * F4);

    /* ===== ileri ===== */
    {
        CUdeviceptr E = P_(g, 0), ids_d = g->ids, x0 = g->layers[0].x_in;
        void* args[] = { &E, &ids_d, &x0, &N, &D };
        launch_1d(g->k_embed_fwd, (u64)N * D, args);
    }

    for (u32 l = 0; l < g->L; l++) {
        GpuLayerActs* a = &g->layers[l];
        CUdeviceptr x_out = (l + 1 < g->L) ? g->layers[l + 1].x_in : g->x_final;

        rmsnorm_fwd(g, a->x_in, P_(g, pidx(l, PK_ATTN_NORM)), a->n1, a->r1);
        gemm1(g, GEMM_NN, a->n1, P_(g, pidx(l, PK_WQKV)), g->qkv, N, 3 * D, D, 0.0f);
        add_bias(g, g->qkv, P_(g, pidx(l, PK_BQKV)), N, 3 * D);
        {
            CUdeviceptr qkv = g->qkv, c = g->cos_t, s = g->sin_t, Q = a->Q, K = a->K, Vv = a->V;
            void* args[] = { &qkv, &c, &s, &Q, &K, &Vv, &Bi, &T, &Hi, &hd };
            launch_1d(g->k_qkv_rope_split, (u64)N * Hi * (hd / 2), args);
        }
        /* S = Q K^T (baslik basina), sonra yerinde softmax -> P */
        gemm(g, GEMM_NT, a->Q, a->K, a->P, T, T, hd, hd, hd, T, BH, sThd, sThd, sTT, 1.0f, 0.0f);
        {
            CUdeviceptr Pp = a->P;
            void* args[] = { &Pp, &T, &scale };
            launch_rows(g->k_attn_softmax_fwd, BH * g->T, att_threads, args);
        }
        gemm(g, GEMM_NN, a->P, a->V, g->Obuf, T, hd, T, T, hd, hd, BH, sTT, sThd, sThd, 1.0f, 0.0f);
        {
            CUdeviceptr O = g->Obuf, out = a->att;
            void* args[] = { &O, &out, &Bi, &T, &Hi, &hd };
            launch_1d(g->k_heads_merge, (u64)N * D, args);
        }
        gemm1(g, GEMM_NN, a->att, P_(g, pidx(l, PK_WO)), g->proj, N, D, D, 0.0f);
        add_bias(g, g->proj, P_(g, pidx(l, PK_BO)), N, D);
        add(g, a->x_in, g->proj, a->x2, (i64)N * D);

        rmsnorm_fwd(g, a->x2, P_(g, pidx(l, PK_FFN_NORM)), a->n2, a->r2);
        gemm1(g, GEMM_NN, a->n2, P_(g, pidx(l, PK_WGU)), a->gu, N, 2 * F, D, 0.0f);
        add_bias(g, a->gu, P_(g, pidx(l, PK_BGU)), N, 2 * F);
        {
            CUdeviceptr gu = a->gu, h = a->h;
            void* args[] = { &gu, &h, &N, &F };
            launch_1d(g->k_swiglu_fwd, (u64)N * F, args);
        }
        gemm1(g, GEMM_NN, a->h, P_(g, pidx(l, PK_WDOWN)), g->proj, N, D, F, 0.0f);
        add_bias(g, g->proj, P_(g, pidx(l, PK_BDOWN)), N, D);
        add(g, a->x2, g->proj, x_out, (i64)N * D);
    }

    u32 p_final = pidx(g->L, 0); /* = 1 + 10*L */
    rmsnorm_fwd(g, g->x_final, P_(g, p_final), g->nf, g->rf);
    /* logits = nf @ E^T (bagli cikis projeksiyonu) */
    gemm1(g, GEMM_NT, g->nf, P_(g, 0), g->logits, N, V, D, 0.0f);
    {
        CUdeviceptr lg = g->logits, tg = g->targets, lr = g->loss_rows;
        f32 grad_scale = 1.0f / (f32)N;
        void* args[] = { &lg, &tg, &lr, &V, &grad_scale };
        launch_rows(g->k_cross_entropy, (u32)N, 256, args);
    }

    /* ===== geri ===== */
    /* logits artik dL/dlogits. dE += dlogits^T @ nf ; dnf = dlogits @ E */
    gemm1(g, GEMM_TN, g->logits, g->nf, G_(g, 0), V, D, N, 1.0f);
    gemm1(g, GEMM_NN, g->logits, P_(g, 0), g->dn, N, D, V, 0.0f);
    cuda_memset_zero(g->dres, (u64)N * D * F4);
    rmsnorm_bwd(g, g->x_final, P_(g, p_final), g->rf, g->dn, g->dres, G_(g, p_final));

    for (i32 l = (i32)g->L - 1; l >= 0; l--) {
        GpuLayerActs* a = &g->layers[l];
        u32 ul = (u32)l;

        /* --- FFN: x3 = x2 + (h @ Wd + bd); dres = dx3 --- */
        gemm1(g, GEMM_TN, a->h, g->dres, G_(g, pidx(ul, PK_WDOWN)), F, D, N, 1.0f);
        colsum_acc(g, g->dres, G_(g, pidx(ul, PK_BDOWN)), N, D);
        gemm1(g, GEMM_NT, g->dres, P_(g, pidx(ul, PK_WDOWN)), g->dh, N, F, D, 0.0f);
        {
            CUdeviceptr gu = a->gu, dh = g->dh, dgu = g->dgu;
            void* args[] = { &gu, &dh, &dgu, &N, &F };
            launch_1d(g->k_swiglu_bwd, (u64)N * F, args);
        }
        gemm1(g, GEMM_TN, a->n2, g->dgu, G_(g, pidx(ul, PK_WGU)), D, 2 * F, N, 1.0f);
        colsum_acc(g, g->dgu, G_(g, pidx(ul, PK_BGU)), N, 2 * F);
        gemm1(g, GEMM_NT, g->dgu, P_(g, pidx(ul, PK_WGU)), g->dn, N, D, 2 * F, 0.0f);
        rmsnorm_bwd(g, a->x2, P_(g, pidx(ul, PK_FFN_NORM)), a->r2, g->dn, g->dres, G_(g, pidx(ul, PK_FFN_NORM)));
        /* dres artik dx2 */

        /* --- Dikkat: x2 = x + (att @ Wo + bo) --- */
        gemm1(g, GEMM_TN, a->att, g->dres, G_(g, pidx(ul, PK_WO)), D, D, N, 1.0f);
        colsum_acc(g, g->dres, G_(g, pidx(ul, PK_BO)), N, D);
        gemm1(g, GEMM_NT, g->dres, P_(g, pidx(ul, PK_WO)), g->datt, N, D, D, 0.0f);
        {
            CUdeviceptr dout = g->datt, dO = g->dO;
            void* args[] = { &dout, &dO, &Bi, &T, &Hi, &hd };
            launch_1d(g->k_heads_split, (u64)N * D, args);
        }
        /* O = P V:  dV = P^T dO ; dP = dO V^T */
        gemm(g, GEMM_TN, a->P, g->dO, g->dV, T, hd, T, T, hd, hd, BH, sTT, sThd, sThd, 1.0f, 0.0f);
        gemm(g, GEMM_NT, g->dO, a->V, g->dP, T, T, hd, hd, hd, T, BH, sThd, sThd, sTT, 1.0f, 0.0f);
        {
            CUdeviceptr Pp = a->P, dP = g->dP;
            void* args[] = { &Pp, &dP, &T, &scale };
            launch_rows(g->k_attn_softmax_bwd, BH * g->T, att_threads, args);
        }
        /* S = Q K^T:  dQ = dS K ; dK = dS^T Q */
        gemm(g, GEMM_NN, g->dP, a->K, g->dQ, T, hd, T, T, hd, hd, BH, sTT, sThd, sThd, 1.0f, 0.0f);
        gemm(g, GEMM_TN, g->dP, a->Q, g->dK, T, hd, T, T, hd, hd, BH, sTT, sThd, sThd, 1.0f, 0.0f);
        {
            CUdeviceptr dQ = g->dQ, dK = g->dK, dV = g->dV, c = g->cos_t, s = g->sin_t, dqkv = g->dqkv;
            void* args[] = { &dQ, &dK, &dV, &c, &s, &dqkv, &Bi, &T, &Hi, &hd };
            launch_1d(g->k_qkv_rope_merge_bwd, (u64)N * Hi * (hd / 2), args);
        }
        gemm1(g, GEMM_TN, a->n1, g->dqkv, G_(g, pidx(ul, PK_WQKV)), D, 3 * D, N, 1.0f);
        colsum_acc(g, g->dqkv, G_(g, pidx(ul, PK_BQKV)), N, 3 * D);
        gemm1(g, GEMM_NT, g->dqkv, P_(g, pidx(ul, PK_WQKV)), g->dn, N, D, 3 * D, 0.0f);
        rmsnorm_bwd(g, a->x_in, P_(g, pidx(ul, PK_ATTN_NORM)), a->r1, g->dn, g->dres, G_(g, pidx(ul, PK_ATTN_NORM)));
        /* dres artik bu katmanin girdisinin gradyani */
    }

    {
        CUdeviceptr dx = g->dres, ids_d = g->ids, dE = G_(g, 0);
        void* args[] = { &dx, &ids_d, &dE, &N, &D };
        launch_1d(g->k_embed_bwd, (u64)D, args);
    }

    /* Kayip: dizi-ici ortalama, sonra diziler uzerinden ortalama
     * (data_parallel_step ile ayni tanim). */
    cuda_d2h(g->host_loss_rows, g->loss_rows, (u64)N * F4); /* senkron -> tum cekirdekler bitti */
    f32 loss_sum = 0.0f;
    for (u32 b = 0; b < g->B; b++) {
        f32 s = 0.0f;
        for (u32 t = 0; t < g->T; t++) s += g->host_loss_rows[(u64)b * g->T + t];
        loss_sum += s / (f32)g->T;
    }
    return loss_sum / (f32)g->B;
}

void gpu_trainer_adam_step(GpuTrainer* g, AdamOptimizer* opt) {
    cuda_set_current(&g->cuda);
    opt->t++;
    f32 bc1 = 1.0f - m_powf(opt->beta1, (f32)opt->t);
    f32 bc2 = 1.0f - m_powf(opt->beta2, (f32)opt->t);
    CUdeviceptr p = g->params, gr = g->grads, m = g->adam_m, v = g->adam_v;
    i64 n = (i64)g->total_scalars;
    f32 lr = opt->lr, b1 = opt->beta1, b2 = opt->beta2, eps = opt->eps;
    void* args[] = { &p, &gr, &m, &v, &n, &lr, &b1, &b2, &eps, &bc1, &bc2 };
    launch_1d(g->k_adam, (u64)n, args);
    cuda_sync();
}
