/* Katman 16 dogrulama: data_parallel_step'in (coklu thread, "govde/
 * shadow" modelli) urettigi ORTALAMA gradyanin, AYNI rastgele
 * pencerelerin TEK THREAD'DE, DOGRUDAN orijinal model uzerinde ardi
 * ardina (biriktirerek) hesaplanip ortalamasi alinan gradyanla
 * BIREBIR AYNI oldugunu kanitlar. Bu, "deger paylasan/gradyani taze"
 * govde mekanizmasinin (build_shadow_model) ve thread'ler arasi
 * indirgemenin dogrulugunu, gercek bir egitim koşusuna guvenmeden
 * once kanitlar. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/prng.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/rope.h"
#include "../model/attention.h"
#include "../model/lm_model.h"
#include "../model/loss.h"
#include "../model/gpu_ops.h"
#include "../training/data_parallel.h"

#define VOCAB 20
#define D_MODEL 16
#define NUM_HEADS 2
#define NUM_LAYERS 2
#define D_FF 32
#define SEQ 6
#define NUM_TOKENS 5000
#define NUM_WORKERS 8
#define SEED_BASE 777ull

static u32 g_pass = 0;
static u32 g_fail = 0;
static f32 fabsf_(f32 v) { return (v < 0) ? -v : v; }

static u32 g_tokens[NUM_TOKENS];

int main(void) {
    console_write_line("=== Katman 16: data_parallel_step Dogrulugu (coklu thread vs tek thread referans) ===");

    for (u32 i = 0; i < NUM_TOKENS; i++) g_tokens[i] = i % VOCAB;

    Allocator alloc = allocator_create(256ull * 1024 * 1024);
    gpu_ops_init(&alloc, "cuda/kernels.ptx");
    gpu_ops_init_workers(&alloc, "cuda/kernels.ptx", NUM_WORKERS); /* Aday 1 (bkz. PROJE_PLANI.md Bolum 17) yolunu da dogrula */

    PCGState rng = pcg_seed(55, 2);
    LMModel model = lm_init(&alloc, &rng, VOCAB, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, 1e-5f);

    Node* params[LM_MAX_LAYERS * 12 + 2];
    u32 num_params = lm_collect_params(&model, params);

    /* --- REFERANS: NUM_WORKERS pencereyi (AYNI tohum formulu ile) TEK
     * THREAD'DE, dogrudan orijinal modelin dugumleri uzerinde, ardi
     * ardina biriktirerek (zero_grad'siz) hesapla. --- */
    for (u32 i = 0; i < num_params; i++) node_zero_grad(params[i]);

    u64 head_dim = D_MODEL / NUM_HEADS;
    for (u32 w = 0; w < NUM_WORKERS; w++) {
        Allocator step_alloc = allocator_create(64ull * 1024 * 1024);

        u64 cshape[2] = { SEQ, head_dim / 2 };
        Tensor cos_t = tensor_create(&step_alloc, cshape, 2);
        Tensor sin_t = tensor_create(&step_alloc, cshape, 2);
        rope_build_tables(&cos_t, &sin_t, SEQ, head_dim, 10000.0f);
        Node* cos_leaf = node_leaf(&step_alloc, cos_t, FALSE);
        Node* sin_leaf = node_leaf(&step_alloc, sin_t, FALSE);

        u64 mshape[2] = { SEQ, SEQ };
        Tensor mask_t = tensor_create(&step_alloc, mshape, 2);
        build_causal_mask(&mask_t);
        Node* mask_leaf = node_leaf(&step_alloc, mask_t, FALSE);

        /* data_parallel.c'nin dp_worker_run'iyla AYNI tohum formulu. */
        PCGState wrng = pcg_seed(SEED_BASE * 1000003ull + w, 99ull + w);
        i64 max_start = (i64)(NUM_TOKENS - SEQ - 1);
        i64 start = pcg_range_i64(&wrng, 0, max_start);
        const u32* input_ids = g_tokens + start;
        const u32* target_ids = g_tokens + start + 1;

        Node* logits = lm_forward(&step_alloc, &model, input_ids, 1, SEQ, cos_leaf, sin_leaf, mask_leaf, 1, 1);
        Node* loss = node_cross_entropy_loss(&step_alloc, logits, target_ids);
        backward(&step_alloc, loss);

        allocator_destroy(&step_alloc);
    }

    /* Referans gradyani ayri bir tampona kopyala (params[i]->grad su an
     * NUM_WORKERS pencerenin TOPLAMI -- data_parallel_step'in ic
     * ortalamasiyla karsilastirmak icin NUM_WORKERS'a bolunmus halini
     * saklayacagiz). */
    Tensor ref_grad[LM_MAX_LAYERS * 12 + 2];
    for (u32 i = 0; i < num_params; i++) {
        ref_grad[i] = tensor_create(&alloc, params[i]->value.shape, params[i]->value.ndim);
        for (u64 k = 0; k < params[i]->grad.numel; k++) {
            ref_grad[i].data[k] = params[i]->grad.data[k] / (f32)NUM_WORKERS;
        }
    }

    /* --- SIMDI: ayni isi data_parallel_step ile (coklu thread, govde
     * modeliyle) yap ve karsilastir. --- */
    for (u32 i = 0; i < num_params; i++) node_zero_grad(params[i]);

    f32 avg_loss = data_parallel_step(&model, params, num_params, g_tokens, NUM_TOKENS, SEQ,
                                       64ull * 1024 * 1024, NUM_WORKERS, SEED_BASE, 1, 1);
    console_write("Ortalama kayip: "); console_write_u64((u64)(avg_loss * 1000.0f)); console_write_line(" (x1000)");

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

    gpu_ops_shutdown();

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
