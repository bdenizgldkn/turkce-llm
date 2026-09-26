/* Katman 8 dogrulama: embedding lookup + cross-entropy kaybi icin
 * sayisal gradyan kontrolu. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/prng.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/embedding.h"
#include "../model/loss.h"

static u32 g_pass = 0;
static u32 g_fail = 0;
static f32 fabsf_(f32 v) { return (v < 0) ? -v : v; }

static void check_grad(const char* name, f32 numeric, f32 analytic) {
    f32 tol = 0.03f + 0.05f * fabsf_(numeric);
    if (fabsf_(numeric - analytic) <= tol) { g_pass++; }
    else {
        g_fail++;
        console_write("  [FAIL] "); console_write(name);
        console_write(" numeric*1e4="); console_write_u64((u64)(i64)(numeric * 10000.0f));
        console_write(" analytic*1e4="); console_write_u64((u64)(i64)(analytic * 10000.0f));
        console_write_line("");
    }
}

/* ================= embedding ================= */

#define EMB_VOCAB 5
#define EMB_DMODEL 3
#define EMB_SEQ 4

static f32 g_table[EMB_VOCAB * EMB_DMODEL];
static f32 g_lossW[EMB_SEQ * EMB_DMODEL];
static u32 g_ids[EMB_SEQ] = { 2, 0, 3, 2 }; /* id=2 tekrar eder: gradyan biriktirmeyi test eder */

static f32 loss_embedding(void) {
    Allocator alloc = allocator_create(4ull * 1024 * 1024);
    u64 tshape[2] = { EMB_VOCAB, EMB_DMODEL };
    Tensor table = tensor_create(&alloc, tshape, 2);
    for (u64 i = 0; i < EMB_VOCAB * EMB_DMODEL; i++) table.data[i] = g_table[i];
    Node* nt = node_leaf(&alloc, table, TRUE);

    Node* emb = node_embedding_lookup(&alloc, nt, g_ids, EMB_SEQ);

    u64 wshape[2] = { EMB_SEQ, EMB_DMODEL };
    Tensor w = tensor_create(&alloc, wshape, 2);
    for (u64 i = 0; i < EMB_SEQ * EMB_DMODEL; i++) w.data[i] = g_lossW[i];
    Node* nw = node_leaf(&alloc, w, TRUE);

    Node* prod = node_mul(&alloc, emb, nw);
    Node* loss = node_sum_all(&alloc, prod);
    f32 v = loss->value.data[0];
    allocator_destroy(&alloc);
    return v;
}

static void test_embedding_grad(void) {
    Allocator alloc = allocator_create(4ull * 1024 * 1024);
    u64 tshape[2] = { EMB_VOCAB, EMB_DMODEL };
    Tensor table = tensor_create(&alloc, tshape, 2);
    for (u64 i = 0; i < EMB_VOCAB * EMB_DMODEL; i++) table.data[i] = g_table[i];
    Node* nt = node_leaf(&alloc, table, TRUE);

    Node* emb = node_embedding_lookup(&alloc, nt, g_ids, EMB_SEQ);

    u64 wshape[2] = { EMB_SEQ, EMB_DMODEL };
    Tensor w = tensor_create(&alloc, wshape, 2);
    for (u64 i = 0; i < EMB_SEQ * EMB_DMODEL; i++) w.data[i] = g_lossW[i];
    Node* nw = node_leaf(&alloc, w, TRUE);

    Node* prod = node_mul(&alloc, emb, nw);
    Node* loss = node_sum_all(&alloc, prod);
    backward(&alloc, loss);

    f32 eps = 1e-2f;
    for (u64 i = 0; i < EMB_VOCAB * EMB_DMODEL; i++) {
        f32 orig = g_table[i];
        g_table[i] = orig + eps; f32 lp = loss_embedding();
        g_table[i] = orig - eps; f32 lm = loss_embedding();
        g_table[i] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        check_grad("embedding.table", num, nt->grad.data[i]);
    }
    allocator_destroy(&alloc);
}

/* ================= cross entropy ================= */

#define CE_SEQ 3
#define CE_VOCAB 5

static f32 g_logits[CE_SEQ * CE_VOCAB];
static u32 g_targets[CE_SEQ] = { 1, 3, 0 };

static f32 loss_ce(void) {
    Allocator alloc = allocator_create(4ull * 1024 * 1024);
    u64 shape[2] = { CE_SEQ, CE_VOCAB };
    Tensor logits = tensor_create(&alloc, shape, 2);
    for (u64 i = 0; i < CE_SEQ * CE_VOCAB; i++) logits.data[i] = g_logits[i];
    Node* nl = node_leaf(&alloc, logits, TRUE);
    Node* loss = node_cross_entropy_loss(&alloc, nl, g_targets);
    f32 v = loss->value.data[0];
    allocator_destroy(&alloc);
    return v;
}

static void test_cross_entropy_grad(void) {
    Allocator alloc = allocator_create(4ull * 1024 * 1024);
    u64 shape[2] = { CE_SEQ, CE_VOCAB };
    Tensor logits = tensor_create(&alloc, shape, 2);
    for (u64 i = 0; i < CE_SEQ * CE_VOCAB; i++) logits.data[i] = g_logits[i];
    Node* nl = node_leaf(&alloc, logits, TRUE);
    Node* loss = node_cross_entropy_loss(&alloc, nl, g_targets);
    backward(&alloc, loss);

    f32 eps = 1e-2f;
    for (u64 i = 0; i < CE_SEQ * CE_VOCAB; i++) {
        f32 orig = g_logits[i];
        g_logits[i] = orig + eps; f32 lp = loss_ce();
        g_logits[i] = orig - eps; f32 lm = loss_ce();
        g_logits[i] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        check_grad("cross_entropy.logits", num, nl->grad.data[i]);
    }

    /* Ozellik kontrolu: dogru sinif icin tek-sicak (one-hot) gibi
     * cok emin bir tahminde kayip ~0'a yakin olmali. */
    Allocator alloc2 = allocator_create(1024 * 1024);
    u64 s2[2] = { 1, 3 };
    Tensor confident = tensor_create(&alloc2, s2, 2);
    confident.data[0] = -100.0f; confident.data[1] = -100.0f; confident.data[2] = 100.0f;
    u32 t2[1] = { 2 };
    Node* nc = node_leaf(&alloc2, confident, TRUE);
    Node* l2 = node_cross_entropy_loss(&alloc2, nc, t2);
    if (l2->value.data[0] < 0.01f) g_pass++; else { g_fail++; console_write_line("  [FAIL] cross_entropy: emin dogru tahminde kayip dusuk olmali"); }
    allocator_destroy(&alloc2);

    allocator_destroy(&alloc);
}

int main(void) {
    console_write_line("=== Katman 8 (Embedding + Cross-Entropy) - Sayisal Gradyan Kontrolu ===");

    PCGState rng = pcg_seed(42, 7);
    for (u64 i = 0; i < EMB_VOCAB * EMB_DMODEL; i++) g_table[i] = (f32)pcg_gaussian(&rng, 0.0, 1.0);
    for (u64 i = 0; i < EMB_SEQ * EMB_DMODEL; i++) g_lossW[i] = (f32)pcg_gaussian(&rng, 0.0, 1.0);
    for (u64 i = 0; i < CE_SEQ * CE_VOCAB; i++) g_logits[i] = (f32)pcg_gaussian(&rng, 0.0, 1.0);

    test_embedding_grad();
    test_cross_entropy_grad();

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
