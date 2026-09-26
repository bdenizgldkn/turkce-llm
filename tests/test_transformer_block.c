/* Katman 7 NIHAI dogrulama: kucuk olcekli ama TAM bir Transformer
 * blogu (RMSNorm + coklu-baslikli RoPE'li nedensel dikkat + SwiGLU +
 * artik/residual baglantilar) uzerinde SAYISAL GRADYAN KONTROLU.
 * Bu, Katman 3-5 uzerine insa edilen tum mimarinin dogru bilesim
 * oldugunu tek seferde dogrular.
 *
 * KATMAN 17: wq/wk/wv artik TEK bir fuzyonlu w_qkv matrisi (bkz.
 * model/attention.h), w_gate/w_up de TEK bir w_gate_up matrisi (bkz.
 * model/feedforward.h) -- bu yuzden testler artik ayri isimli alt-
 * matrisler yerine bu fuzyonlu buyuk tensorlerin FARKLI BOLGELERINDEN
 * (Q/K/V ya da gate/up) ornek duz (flat) indeksler kontrol ediyor. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/prng.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/model_ops.h"
#include "../model/rope.h"
#include "../model/attention.h"
#include "../model/feedforward.h"
#include "../model/transformer_block.h"

#define D_MODEL 8
#define NUM_HEADS 2
#define HEAD_DIM (D_MODEL / NUM_HEADS)
#define D_FF 16
#define SEQ 4

static u32 g_pass = 0;
static u32 g_fail = 0;

static f32 g_x[SEQ * D_MODEL];
static f32 g_wqkv[D_MODEL * 3 * D_MODEL], g_bqkv[3 * D_MODEL];
static f32 g_wo[D_MODEL * D_MODEL], g_bo[D_MODEL];
static f32 g_attn_norm[D_MODEL];
static f32 g_ffn_norm[D_MODEL];
static f32 g_wgateup[D_MODEL * 2 * D_FF], g_bgateup[2 * D_FF];
static f32 g_wdown[D_FF * D_MODEL], g_bdown[D_MODEL];
static f32 g_lossW[SEQ * D_MODEL];

static Node* leaf1(Allocator* a, const f32* data, u64 n) {
    u64 shape[1] = { n };
    Tensor t = tensor_create(a, shape, 1);
    for (u64 i = 0; i < n; i++) t.data[i] = data[i];
    return node_leaf(a, t, TRUE);
}
static Node* leaf2(Allocator* a, const f32* data, u64 rows, u64 cols) {
    u64 shape[2] = { rows, cols };
    Tensor t = tensor_create(a, shape, 2);
    for (u64 i = 0; i < rows * cols; i++) t.data[i] = data[i];
    return node_leaf(a, t, TRUE);
}

typedef struct BuiltGraph {
    Node* x;
    Node* wqkv; Node* wo;
    Node* attn_norm; Node* ffn_norm;
    Node* wgateup; Node* wdown;
    Node* loss;
} BuiltGraph;

static BuiltGraph build_graph(Allocator* alloc) {
    BuiltGraph g;
    g.x = leaf2(alloc, g_x, SEQ, D_MODEL);

    g.wqkv = leaf2(alloc, g_wqkv, D_MODEL, 3 * D_MODEL); Node* bqkv = leaf1(alloc, g_bqkv, 3 * D_MODEL);
    g.wo = leaf2(alloc, g_wo, D_MODEL, D_MODEL); Node* bo = leaf1(alloc, g_bo, D_MODEL);
    g.attn_norm = leaf1(alloc, g_attn_norm, D_MODEL);
    g.ffn_norm = leaf1(alloc, g_ffn_norm, D_MODEL);
    g.wgateup = leaf2(alloc, g_wgateup, D_MODEL, 2 * D_FF); Node* bgateup = leaf1(alloc, g_bgateup, 2 * D_FF);
    g.wdown = leaf2(alloc, g_wdown, D_FF, D_MODEL); Node* bdown = leaf1(alloc, g_bdown, D_MODEL);

    BlockWeights bw;
    bw.attn_norm_w = g.attn_norm;
    bw.attn.w_qkv = g.wqkv; bw.attn.b_qkv = bqkv;
    bw.attn.wo = g.wo; bw.attn.bo = bo;
    bw.ffn_norm_w = g.ffn_norm;
    bw.ffn.w_gate_up = g.wgateup; bw.ffn.b_gate_up = bgateup;
    bw.ffn.w_down = g.wdown; bw.ffn.b_down = bdown;

    u64 cshape[2] = { SEQ, HEAD_DIM / 2 };
    Tensor cos_t = tensor_create(alloc, cshape, 2);
    Tensor sin_t = tensor_create(alloc, cshape, 2);
    rope_build_tables(&cos_t, &sin_t, SEQ, HEAD_DIM, 10000.0f);
    Node* cos_leaf = node_leaf(alloc, cos_t, FALSE);
    Node* sin_leaf = node_leaf(alloc, sin_t, FALSE);

    u64 mshape[2] = { SEQ, SEQ };
    Tensor mask_t = tensor_create(alloc, mshape, 2);
    build_causal_mask(&mask_t);
    Node* mask_leaf = node_leaf(alloc, mask_t, FALSE);

    Node* y = transformer_block(alloc, g.x, &bw, NUM_HEADS, 1, SEQ, cos_leaf, sin_leaf, mask_leaf, 1e-5f, 0);

    Node* lossW = leaf2(alloc, g_lossW, SEQ, D_MODEL);
    Node* prod = node_mul(alloc, y, lossW);
    g.loss = node_sum_all(alloc, prod);

    return g;
}

static f32 forward_loss(void) {
    Allocator alloc = allocator_create(64ull * 1024 * 1024);
    BuiltGraph g = build_graph(&alloc);
    f32 v = g.loss->value.data[0];
    allocator_destroy(&alloc);
    return v;
}

static f32 fabsf_(f32 v) { return (v < 0) ? -v : v; }

static void check_one(f32* array, u64 idx, f32 analytic, const char* label) {
    f32 eps = 1e-2f;
    f32 orig = array[idx];
    array[idx] = orig + eps; f32 lp = forward_loss();
    array[idx] = orig - eps; f32 lm = forward_loss();
    array[idx] = orig;
    f32 numeric = (lp - lm) / (2.0f * eps);

    f32 tol = 0.03f + 0.05f * fabsf_(numeric);
    if (fabsf_(numeric - analytic) <= tol) { g_pass++; }
    else {
        g_fail++;
        console_write("  [FAIL] "); console_write(label);
        console_write(" numeric*1e4="); console_write_u64((u64)(i64)(numeric * 10000.0f));
        console_write(" analytic*1e4="); console_write_u64((u64)(i64)(analytic * 10000.0f));
        console_write_line("");
    }
}

static void randn_fill(PCGState* rng, f32* arr, u64 n, f32 std) {
    for (u64 i = 0; i < n; i++) arr[i] = (f32)pcg_gaussian(rng, 0.0, (f64)std);
}

int main(void) {
    console_write_line("=== Katman 7 NIHAI: Tam Transformer Blogu - Sayisal Gradyan Kontrolu ===");

    PCGState rng = pcg_seed(1234, 1);
    randn_fill(&rng, g_x, SEQ * D_MODEL, 0.3f);
    randn_fill(&rng, g_wqkv, D_MODEL * 3 * D_MODEL, 0.1f); randn_fill(&rng, g_bqkv, 3 * D_MODEL, 0.05f);
    randn_fill(&rng, g_wo, D_MODEL * D_MODEL, 0.1f); randn_fill(&rng, g_bo, D_MODEL, 0.05f);
    for (u64 i = 0; i < D_MODEL; i++) { g_attn_norm[i] = 1.0f; g_ffn_norm[i] = 1.0f; }
    randn_fill(&rng, g_wgateup, D_MODEL * 2 * D_FF, 0.1f); randn_fill(&rng, g_bgateup, 2 * D_FF, 0.05f);
    randn_fill(&rng, g_wdown, D_FF * D_MODEL, 0.1f); randn_fill(&rng, g_bdown, D_MODEL, 0.05f);
    randn_fill(&rng, g_lossW, SEQ * D_MODEL, 1.0f);

    Allocator alloc = allocator_create(64ull * 1024 * 1024);
    BuiltGraph g = build_graph(&alloc);
    backward(&alloc, g.loss);

    /* Girdi x: birkac ornek eleman */
    for (u64 i = 0; i < 6; i++) check_one(g_x, i, g.x->grad.data[i], "x");

    /* Fuzyonlu w_qkv'nin UC BOLGESINDEN de (Q, K, V) ornekler --
     * satir 0'da sutun 0(Q), D_MODEL(K), 2*D_MODEL(V). */
    check_one(g_wqkv, 0, g.wqkv->grad.data[0], "wqkv[Q]");
    check_one(g_wqkv, 1, g.wqkv->grad.data[1], "wqkv[Q]");
    check_one(g_wqkv, D_MODEL, g.wqkv->grad.data[D_MODEL], "wqkv[K]");
    check_one(g_wqkv, D_MODEL + 1, g.wqkv->grad.data[D_MODEL + 1], "wqkv[K]");
    check_one(g_wqkv, 2 * D_MODEL, g.wqkv->grad.data[2 * D_MODEL], "wqkv[V]");
    check_one(g_wqkv, 2 * D_MODEL + 1, g.wqkv->grad.data[2 * D_MODEL + 1], "wqkv[V]");
    for (u64 i = 0; i < 4; i++) check_one(g_wo, i, g.wo->grad.data[i], "wo");
    for (u64 i = 0; i < 3; i++) check_one(g_attn_norm, i, g.attn_norm->grad.data[i], "attn_norm_w");
    for (u64 i = 0; i < 3; i++) check_one(g_ffn_norm, i, g.ffn_norm->grad.data[i], "ffn_norm_w");

    /* Fuzyonlu w_gate_up'in HER IKI bolgesinden de (gate, up) ornekler. */
    check_one(g_wgateup, 0, g.wgateup->grad.data[0], "wgateup[gate]");
    check_one(g_wgateup, 1, g.wgateup->grad.data[1], "wgateup[gate]");
    check_one(g_wgateup, D_FF, g.wgateup->grad.data[D_FF], "wgateup[up]");
    check_one(g_wgateup, D_FF + 1, g.wgateup->grad.data[D_FF + 1], "wgateup[up]");
    for (u64 i = 0; i < 4; i++) check_one(g_wdown, i, g.wdown->grad.data[i], "wdown");

    allocator_destroy(&alloc);

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
