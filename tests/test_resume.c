/* Egitime checkpoint'ten devam etme dogrulamasi (Linux tasimasi,
 * bkz. training/train_lm.c 4b).
 *
 * A) Kesintisiz: model N adim egitilir.
 * B) Kesintili: AYNI baslangic modeli K adim egitilir, checkpoint_save
 *    ile kaydedilir; sonra FARKLI tohumla baslatilmis YEPYENI bir model
 *    + optimizer checkpoint_load ile yuklenir ve opt.t'den N'e devam eder.
 *
 * train_lm.c'deki dongunun AYNISI kullanilir (data_parallel_step, GPU
 * acik, coklu worker, tohum 2026+step, adam_step_parallel). Devam
 * dogruysa A ve B'nin nihai parametreleri ve Adam durumu BIT BIT ayni
 * olmalidir (GPU cekirdeklerinde atomik islem yok -- deterministik). */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/prng.h"
#include "../runtime/file_io.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/lm_model.h"
#include "../model/gpu_ops.h"
#include "../training/adam.h"
#include "../training/checkpoint.h"
#include "../training/data_parallel.h"

#define VOCAB 20
#define D_MODEL 16
#define NUM_HEADS 2
#define NUM_LAYERS 2
#define D_FF 32
#define SEQ 6
#define NUM_TOKENS 5000
#define NUM_WORKERS 8
#define TOTAL_STEPS 6u
#define SPLIT_STEP 3u
#define STEP_ARENA (64ull * 1024 * 1024)
#define CKPT_PATH "test_resume_ckpt.bin"

static u32 g_pass = 0;
static u32 g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; console_write("  [FAIL] "); console_write_line(msg); } \
} while (0)

static u32 g_tokens[NUM_TOKENS];

typedef struct Run {
    LMModel model;
    Node* params[LM_MAX_LAYERS * 12 + 2];
    u32 num_params;
    AdamOptimizer opt;
} Run;

static void run_init(Run* r, Allocator* alloc, u64 init_seed) {
    PCGState rng = pcg_seed(init_seed, 1);
    r->model = lm_init(alloc, &rng, VOCAB, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, 1e-5f);
    r->num_params = lm_collect_params(&r->model, r->params);
    r->opt = adam_create(alloc, r->params, r->num_params, 3e-3f, 0.9f, 0.999f, 1e-8f);
}

/* training/train_lm.c'deki egitim dongusuyle ayni adim. */
static void run_steps(Run* r, u32 from, u32 to) {
    for (u32 step = from; step < to; step++) {
        adam_zero_grad(&r->opt);
        data_parallel_step(&r->model, r->params, r->num_params, g_tokens, NUM_TOKENS, SEQ,
                           STEP_ARENA, NUM_WORKERS, 2026ull + step, 1, 1);
        adam_step_parallel(&r->opt, NUM_WORKERS);
    }
}

static bool32 tensors_bit_equal(const Tensor* a, const Tensor* b) {
    if (a->numel != b->numel) return FALSE;
    const u32* x = (const u32*)a->data;
    const u32* y = (const u32*)b->data;
    for (u64 k = 0; k < a->numel; k++) if (x[k] != y[k]) return FALSE;
    return TRUE;
}

int main(void) {
    console_write_line("=== Checkpoint'ten Devam: kesintisiz vs kaydet+yukle+devam (bit esitligi) ===");

    for (u32 i = 0; i < NUM_TOKENS; i++) g_tokens[i] = (i * 7u + i / 13u) % VOCAB;

    Allocator alloc = allocator_create(512ull * 1024 * 1024);
    gpu_ops_init(&alloc, "cuda/kernels.ptx");
    gpu_ops_init_workers(&alloc, "cuda/kernels.ptx", NUM_WORKERS);

    /* A) Kesintisiz */
    Run a;
    run_init(&a, &alloc, 55);
    run_steps(&a, 0, TOTAL_STEPS);

    /* B1) Ayni baslangic, SPLIT_STEP adim, kaydet */
    Run b1;
    run_init(&b1, &alloc, 55);
    run_steps(&b1, 0, SPLIT_STEP);
    CHECK(checkpoint_save(CKPT_PATH, b1.params, b1.num_params, &b1.opt), "resume: checkpoint kaydedilemedi");

    /* B2) FARKLI tohumla yepyeni model: yukleme tum durumu gercekten
     * uzerine yazmazsa sonuc A'dan farkli cikar. */
    Run b2;
    run_init(&b2, &alloc, 999);
    Allocator scratch = allocator_create(64ull * 1024 * 1024);
    bool32 loaded = checkpoint_load(&scratch, CKPT_PATH, b2.params, b2.num_params, &b2.opt);
    allocator_destroy(&scratch);
    CHECK(loaded, "resume: checkpoint yuklenemedi");
    CHECK(b2.opt.t == SPLIT_STEP, "resume: yuklenen opt.t, kaydedilen adim sayisina esit degil");

    u32 start_step = (u32)b2.opt.t;
    run_steps(&b2, start_step, TOTAL_STEPS);

    CHECK(a.opt.t == b2.opt.t, "resume: nihai Adam adim sayaci (t) farkli");
    CHECK(a.num_params == b2.num_params, "resume: parametre sayisi farkli");
    for (u32 i = 0; i < a.num_params; i++) {
        CHECK(tensors_bit_equal(&a.params[i]->value, &b2.params[i]->value), "resume: parametre degerleri bit bit ayni degil");
        CHECK(tensors_bit_equal(&a.opt.m[i], &b2.opt.m[i]), "resume: Adam momentum (m) bit bit ayni degil");
        CHECK(tensors_bit_equal(&a.opt.v[i], &b2.opt.v[i]), "resume: Adam ikinci moment (v) bit bit ayni degil");
    }

    /* Kontrol: devam etmeden once b1 != a olmali (test gercekten bir sey
     * olcuyor mu -- SPLIT_STEP..TOTAL_STEPS adimlari parametreleri degistirmeli). */
    CHECK(!tensors_bit_equal(&a.params[0]->value, &b1.params[0]->value),
          "resume: kontrol -- ara durum nihai durumla ayni (test anlamsiz)");

    file_delete(CKPT_PATH);
    gpu_ops_shutdown();

    console_write("Sonuc: "); console_write_u64(g_pass); console_write(" basarili, ");
    console_write_u64(g_fail); console_write_line(" basarisiz.");
    return g_fail == 0 ? 0 : 1;
}
