/* Katman 7-8 NIHAI: tam dil modeli (embedding+transformer+cross-entropy)
 * kucuk olcekte uctan uca sayisal gradyan kontrolu. */
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

#define VOCAB 12
#define D_MODEL 8
#define NUM_HEADS 2
#define NUM_LAYERS 2
#define D_FF 16
#define SEQ 5

static u32 g_pass = 0;
static u32 g_fail = 0;
static f32 fabsf_(f32 v) { return (v < 0) ? -v : v; }

static u32 g_ids[SEQ] = { 1, 4, 7, 2, 9 };
static u32 g_targets[SEQ] = { 4, 7, 2, 9, 1 };

static f32 forward_loss(LMModel* m) {
    Allocator alloc = allocator_create(16ull * 1024 * 1024);
    u64 cshape[2] = { SEQ, (D_MODEL / NUM_HEADS) / 2 };
    Tensor cos_t = tensor_create(&alloc, cshape, 2);
    Tensor sin_t = tensor_create(&alloc, cshape, 2);
    rope_build_tables(&cos_t, &sin_t, SEQ, D_MODEL / NUM_HEADS, 10000.0f);
    Node* cos_leaf = node_leaf(&alloc, cos_t, FALSE);
    Node* sin_leaf = node_leaf(&alloc, sin_t, FALSE);

    u64 mshape[2] = { SEQ, SEQ };
    Tensor mask_t = tensor_create(&alloc, mshape, 2);
    build_causal_mask(&mask_t);
    Node* mask_leaf = node_leaf(&alloc, mask_t, FALSE);

    /* Model agirliklarinin DEGERLERINI (ayni Allocator icinde YENIDEN
     * OLUSTURULMUS node'lara) kopyalayarak taze bir graf kurariz --
     * boylece finite-difference her seferinde ayni parametrelerle
     * ama taze bir hesaplama grafiyle calisir. */
    Node* logits = lm_forward(&alloc, m, g_ids, 1, SEQ, cos_leaf, sin_leaf, mask_leaf, 0, 0);
    Node* loss = node_cross_entropy_loss(&alloc, logits, g_targets);
    f32 v = loss->value.data[0];
    allocator_destroy(&alloc);
    return v;
}

static void check_grad(const char* name, f32 numeric, f32 analytic) {
    f32 tol = 0.03f + 0.08f * fabsf_(numeric);
    if (fabsf_(numeric - analytic) <= tol) { g_pass++; }
    else {
        g_fail++;
        console_write("  [FAIL] "); console_write(name);
        console_write(" numeric*1e4="); console_write_u64((u64)(i64)(numeric * 10000.0f));
        console_write(" analytic*1e4="); console_write_u64((u64)(i64)(analytic * 10000.0f));
        console_write_line("");
    }
}

int main(void) {
    console_write_line("=== Katman 7-8 NIHAI: Tam Dil Modeli Gradyan Kontrolu ===");

    Allocator alloc = allocator_create(64ull * 1024 * 1024);
    PCGState rng = pcg_seed(2026, 25);
    LMModel m = lm_init(&alloc, &rng, VOCAB, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, 1e-5f);

    u64 cshape[2] = { SEQ, (D_MODEL / NUM_HEADS) / 2 };
    Tensor cos_t = tensor_create(&alloc, cshape, 2);
    Tensor sin_t = tensor_create(&alloc, cshape, 2);
    rope_build_tables(&cos_t, &sin_t, SEQ, D_MODEL / NUM_HEADS, 10000.0f);
    Node* cos_leaf = node_leaf(&alloc, cos_t, FALSE);
    Node* sin_leaf = node_leaf(&alloc, sin_t, FALSE);

    u64 mshape[2] = { SEQ, SEQ };
    Tensor mask_t = tensor_create(&alloc, mshape, 2);
    build_causal_mask(&mask_t);
    Node* mask_leaf = node_leaf(&alloc, mask_t, FALSE);

    Node* logits = lm_forward(&alloc, &m, g_ids, 1, SEQ, cos_leaf, sin_leaf, mask_leaf, 0, 0);
    Node* loss = node_cross_entropy_loss(&alloc, logits, g_targets);
    console_write("Baslangic kaybi (rastgele agirliklar): ");
    console_write_u64((u64)(loss->value.data[0] * 1000.0f));
    console_write_line(" (x1000)");

    backward(&alloc, loss);

    Node* params[LM_MAX_LAYERS * 12 + 2];
    u32 nparams = lm_collect_params(&m, params);
    console_write("Toplam parametre TENSORU sayisi: "); console_write_u64(nparams); console_write_line("");

    u64 total_scalars = 0;
    for (u32 i = 0; i < nparams; i++) total_scalars += params[i]->value.numel;
    console_write("Toplam parametre (skaler) sayisi: "); console_write_u64(total_scalars); console_write_line("");

    /* Birkac parametreden ornek gradyan kontrolu (embedding, ilk katman
     * dikkat agirligi, son katman ffn agirligi, embed_table -- tum
     * mimarinin dogru bagli oldugunu kanitlamak icin). */
    f32 eps = 1e-2f;
    Node* to_check[4] = { m.embed_table, m.blocks[0].attn.w_qkv, m.blocks[1].ffn.w_down, m.final_norm_w };
    const char* names[4] = { "embed_table", "block0.w_qkv", "block1.w_down", "final_norm_w" };

    for (i32 pi = 0; pi < 4; pi++) {
        Node* p = to_check[pi];
        u64 ncheck = (p->value.numel < 3) ? p->value.numel : 3;
        for (u64 i = 0; i < ncheck; i++) {
            f32 orig = p->value.data[i];
            p->value.data[i] = orig + eps; f32 lp = forward_loss(&m);
            p->value.data[i] = orig - eps; f32 lm_ = forward_loss(&m);
            p->value.data[i] = orig;
            f32 num = (lp - lm_) / (2.0f * eps);
            check_grad(names[pi], num, p->grad.data[i]);
        }
    }

    allocator_destroy(&alloc);

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
