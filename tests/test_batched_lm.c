/* Katman 18 dogrulama: GERCEK tensor-batching'in (bkz. model/attention.c,
 * model/lm_model.c: yeni batch_size parametresi) matematiksel olarak
 * DOGRU oldugunu kanitlar.
 *
 * Yontem: 3 BAGIMSIZ diziyi (1) TEK TEK (batch_size=1, ust uste 3 kez
 * geri-yayilim biriktirerek) ve (2) TEK SEFERDE yiginlanmis
 * (batch_size=3) isleyip, elde edilen gradyanlarin (uygun olceklemeyle)
 * BIREBIR AYNI oldugunu gosterir. Boylece "batch_size=1 ile ayni
 * matematigi, sadece daha az GPU cagrisiyla urettigini" kanitlar. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/prng.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/model_ops.h"
#include "../model/rope.h"
#include "../model/attention.h"
#include "../model/lm_model.h"
#include "../model/loss.h"
#include "../model/gpu_ops.h"

#define VOCAB 20
#define D_MODEL 16
#define NUM_HEADS 2
#define NUM_LAYERS 2
#define D_FF 32
#define SEQ 5
#define NBATCH 3

static u32 g_pass = 0;
static u32 g_fail = 0;
static f32 fabsf_(f32 v) { return (v < 0) ? -v : v; }

static u32 g_seqs[NBATCH][SEQ] = {
    { 1, 4, 7, 2, 9 },
    { 3, 8, 0, 5, 11 },
    { 6, 2, 14, 1, 17 }
};
static u32 g_targets[NBATCH][SEQ] = {
    { 4, 7, 2, 9, 1 },
    { 8, 0, 5, 11, 3 },
    { 2, 14, 1, 17, 6 }
};

static void build_tables(Allocator* alloc, Node** cos_leaf, Node** sin_leaf, Node** mask_leaf) {
    u64 head_dim = D_MODEL / NUM_HEADS;
    u64 cshape[2] = { SEQ, head_dim / 2 };
    Tensor cos_t = tensor_create(alloc, cshape, 2);
    Tensor sin_t = tensor_create(alloc, cshape, 2);
    rope_build_tables(&cos_t, &sin_t, SEQ, head_dim, 10000.0f);
    *cos_leaf = node_leaf(alloc, cos_t, FALSE);
    *sin_leaf = node_leaf(alloc, sin_t, FALSE);

    u64 mshape[2] = { SEQ, SEQ };
    Tensor mask_t = tensor_create(alloc, mshape, 2);
    build_causal_mask(&mask_t);
    *mask_leaf = node_leaf(alloc, mask_t, FALSE);
}

int main(void) {
    console_write_line("=== Katman 18: Gercek Tensor-Batching Dogrulugu (batch=1x3 vs batch=3) ===");

    Allocator alloc = allocator_create(256ull * 1024 * 1024);
    gpu_ops_init(&alloc, "cuda/kernels.ptx");

    PCGState rng = pcg_seed(99, 4);
    LMModel model = lm_init(&alloc, &rng, VOCAB, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, 1e-5f);

    Node* params[LM_MAX_LAYERS * 12 + 2];
    u32 num_params = lm_collect_params(&model, params);

    /* --- REFERANS: 3 diziyi TEK TEK isle, gradyanlari biriktir. --- */
    for (u32 i = 0; i < num_params; i++) node_zero_grad(params[i]);

    for (u32 b = 0; b < NBATCH; b++) {
        Allocator step_alloc = allocator_create(64ull * 1024 * 1024);
        Node *cos_leaf, *sin_leaf, *mask_leaf;
        build_tables(&step_alloc, &cos_leaf, &sin_leaf, &mask_leaf);

        Node* logits = lm_forward(&step_alloc, &model, g_seqs[b], 1, SEQ, cos_leaf, sin_leaf, mask_leaf, 1, 1);
        Node* loss = node_cross_entropy_loss(&step_alloc, logits, g_targets[b]);
        backward(&step_alloc, loss);

        allocator_destroy(&step_alloc);
    }

    Tensor ref_grad[LM_MAX_LAYERS * 12 + 2];
    for (u32 i = 0; i < num_params; i++) {
        ref_grad[i] = tensor_create(&alloc, params[i]->value.shape, params[i]->value.ndim);
        for (u64 k = 0; k < params[i]->grad.numel; k++) {
            ref_grad[i].data[k] = params[i]->grad.data[k] / (f32)NBATCH;
        }
    }

    /* --- BATCHED: 3 diziyi TEK SEFERDE (batch_size=3) isle. --- */
    for (u32 i = 0; i < num_params; i++) node_zero_grad(params[i]);

    Allocator step_alloc = allocator_create(64ull * 1024 * 1024);
    Node *cos_leaf, *sin_leaf, *mask_leaf;
    build_tables(&step_alloc, &cos_leaf, &sin_leaf, &mask_leaf);

    u32 cat_ids[NBATCH * SEQ];
    u32 cat_targets[NBATCH * SEQ];
    for (u32 b = 0; b < NBATCH; b++) {
        for (u32 t = 0; t < SEQ; t++) {
            cat_ids[b * SEQ + t] = g_seqs[b][t];
            cat_targets[b * SEQ + t] = g_targets[b][t];
        }
    }

    Node* logits = lm_forward(&step_alloc, &model, cat_ids, NBATCH, SEQ, cos_leaf, sin_leaf, mask_leaf, 1, 1);
    console_write("Batched logits shape: ["); console_write_u64(logits->value.shape[0]);
    console_write(","); console_write_u64(logits->value.shape[1]); console_write_line("]");
    Node* loss = node_cross_entropy_loss(&step_alloc, logits, cat_targets);
    backward(&step_alloc, loss);

    for (u32 i = 0; i < num_params; i++) {
        for (u64 k = 0; k < params[i]->grad.numel; k++) {
            f32 got = params[i]->grad.data[k];
            f32 want = ref_grad[i].data[k];
            f32 tol = 1e-4f + 1e-3f * fabsf_(want);
            if (fabsf_(got - want) <= tol) g_pass++;
            else {
                g_fail++;
                console_write("  [FAIL] param="); console_write_u64(i);
                console_write(" idx="); console_write_u64(k);
                console_write(" got*1e6="); console_write_u64((u64)(i64)(got * 1e6f));
                console_write(" want*1e6="); console_write_u64((u64)(i64)(want * 1e6f));
                console_write_line("");
            }
        }
    }

    allocator_destroy(&step_alloc);
    gpu_ops_shutdown();

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
