/* Katman 7 (devam) dogrulama: batch destegi -- 2 bagimsiz dizi, AYNI
 * paylasilan agirliklarla, tek bir ortalama kayipta birlestirilir.
 * Sayisal gradyan kontrolu hem dizi girdilerinde hem de PAYLASILAN
 * agirliklarda (gradyanlarin batch uzerinden dogru toplandigini/
 * ortalandigini kanitlar) yapilir. */
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
#include "../model/batch.h"

#define D_MODEL 8
#define NUM_HEADS 2
#define HEAD_DIM (D_MODEL / NUM_HEADS)
#define D_FF 16
#define SEQ 4
#define BATCH 2

static u32 g_pass = 0;
static u32 g_fail = 0;

static f32 g_x[BATCH][SEQ * D_MODEL];
static f32 g_wqkv[D_MODEL * 3 * D_MODEL], g_bqkv[3 * D_MODEL];
static f32 g_wo[D_MODEL * D_MODEL], g_bo[D_MODEL];
static f32 g_attn_norm[D_MODEL];
static f32 g_ffn_norm[D_MODEL];
static f32 g_wgateup[D_MODEL * 2 * D_FF], g_bgateup[2 * D_FF];
static f32 g_wdown[D_FF * D_MODEL], g_bdown[D_MODEL];
static f32 g_lossW[BATCH][SEQ * D_MODEL];

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
    Node* x[BATCH];
    Node* wqkv; Node* wdown;
    Node* loss;
} BuiltGraph;

static BuiltGraph build_graph(Allocator* alloc) {
    BuiltGraph g;

    Node* wqkv = leaf2(alloc, g_wqkv, D_MODEL, 3 * D_MODEL); Node* bqkv = leaf1(alloc, g_bqkv, 3 * D_MODEL);
    Node* wo = leaf2(alloc, g_wo, D_MODEL, D_MODEL); Node* bo = leaf1(alloc, g_bo, D_MODEL);
    Node* attn_norm = leaf1(alloc, g_attn_norm, D_MODEL);
    Node* ffn_norm = leaf1(alloc, g_ffn_norm, D_MODEL);
    Node* wgateup = leaf2(alloc, g_wgateup, D_MODEL, 2 * D_FF); Node* bgateup = leaf1(alloc, g_bgateup, 2 * D_FF);
    Node* wdown = leaf2(alloc, g_wdown, D_FF, D_MODEL); Node* bdown = leaf1(alloc, g_bdown, D_MODEL);

    g.wqkv = wqkv; g.wdown = wdown;

    BlockWeights bw;
    bw.attn_norm_w = attn_norm;
    bw.attn.w_qkv = wqkv; bw.attn.b_qkv = bqkv;
    bw.attn.wo = wo; bw.attn.bo = bo;
    bw.ffn_norm_w = ffn_norm;
    bw.ffn.w_gate_up = wgateup; bw.ffn.b_gate_up = bgateup;
    bw.ffn.w_down = wdown; bw.ffn.b_down = bdown;

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

    Node* x_batch[BATCH];
    for (u32 b = 0; b < BATCH; b++) {
        x_batch[b] = leaf2(alloc, g_x[b], SEQ, D_MODEL);
        g.x[b] = x_batch[b];
    }

    Node* out_batch[BATCH];
    transformer_block_batch(alloc, x_batch, BATCH, &bw, NUM_HEADS, cos_leaf, sin_leaf, mask_leaf, 1e-5f, 0, out_batch);

    Node* per_item_loss[BATCH];
    for (u32 b = 0; b < BATCH; b++) {
        Node* lw = leaf2(alloc, g_lossW[b], SEQ, D_MODEL);
        Node* prod = node_mul(alloc, out_batch[b], lw);
        per_item_loss[b] = node_sum_all(alloc, prod);
    }

    g.loss = batch_mean_loss(alloc, per_item_loss, BATCH);
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
    console_write_line("=== Katman 7 (Batch Destegi) - Sayisal Gradyan Kontrolu ===");

    PCGState rng = pcg_seed(777, 3);
    randn_fill(&rng, g_x[0], SEQ * D_MODEL, 0.3f);
    randn_fill(&rng, g_x[1], SEQ * D_MODEL, 0.3f);
    randn_fill(&rng, g_wqkv, D_MODEL * 3 * D_MODEL, 0.1f); randn_fill(&rng, g_bqkv, 3 * D_MODEL, 0.05f);
    randn_fill(&rng, g_wo, D_MODEL * D_MODEL, 0.1f); randn_fill(&rng, g_bo, D_MODEL, 0.05f);
    for (u64 i = 0; i < D_MODEL; i++) { g_attn_norm[i] = 1.0f; g_ffn_norm[i] = 1.0f; }
    randn_fill(&rng, g_wgateup, D_MODEL * 2 * D_FF, 0.1f); randn_fill(&rng, g_bgateup, 2 * D_FF, 0.05f);
    randn_fill(&rng, g_wdown, D_FF * D_MODEL, 0.1f); randn_fill(&rng, g_bdown, D_MODEL, 0.05f);
    randn_fill(&rng, g_lossW[0], SEQ * D_MODEL, 1.0f);
    randn_fill(&rng, g_lossW[1], SEQ * D_MODEL, 1.0f);

    Allocator alloc = allocator_create(64ull * 1024 * 1024);
    BuiltGraph g = build_graph(&alloc);
    backward(&alloc, g.loss);

    for (u64 i = 0; i < 4; i++) check_one(g_x[0], i, g.x[0]->grad.data[i], "x[0]");
    for (u64 i = 0; i < 4; i++) check_one(g_x[1], i, g.x[1]->grad.data[i], "x[1]");
    /* Paylasilan agirliklar: gradyan HER IKI batch ogesinden de katki
     * almali -- finite-difference dogrudan kombine (ortalama) kayipla
     * karsilastirildigi icin bu dogal olarak test edilir. */
    for (u64 i = 0; i < 4; i++) check_one(g_wqkv, i, g.wqkv->grad.data[i], "wqkv(paylasilan)");
    for (u64 i = 0; i < 4; i++) check_one(g_wdown, i, g.wdown->grad.data[i], "wdown(paylasilan)");

    allocator_destroy(&alloc);

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
