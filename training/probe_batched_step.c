/* Katman 18 - TANILAMA/OLCUM ARACI (kalici egitim koduna DAHIL DEGIL).
 *
 * Gercek tensor-batching'in (batch=8, TEK thread, TEK graf, katman
 * basina 32 GPU cagrisi yerine 4) gercek duvar-saati hizini, mevcut
 * (train_lm.c'nin kullandigi) 8-thread veri-paralel yaklasimiyla
 * karsilastirmak icin. train_lm.c/data_parallel.c'ye HICBIR sekilde
 * dokunmaz -- sadece OLCUM amaclidir (bkz. PROJE_PLANI.md Bolum 18). */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/file_io.h"
#include "../runtime/prng.h"
#include "../runtime/timer.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/rope.h"
#include "../model/attention.h"
#include "../model/lm_model.h"
#include "../model/loss.h"
#include "../model/gpu_ops.h"

#define VOCAB_SIZE 31769u
#define D_MODEL    384u
#define NUM_HEADS  6u
#define NUM_LAYERS 12u
#define D_FF       1024u
#define SEQ_LEN    128u
#define BATCH_SIZE 8u
#define NUM_STEPS  20u

static void write_fixed3(f64 v) {
    if (v < 0) { console_write("-"); v = -v; }
    u64 scaled = (u64)(v * 1000.0 + 0.5);
    console_write_u64(scaled / 1000); console_write(".");
    if ((scaled % 1000) < 100) console_write("0");
    if ((scaled % 1000) < 10) console_write("0");
    console_write_u64(scaled % 1000);
}

int main(void) {
    console_write_line("=== Katman 18 OLCUM: Tek-Thread Gercek Batching (batch=8) ===");

    Allocator persist = allocator_create(4ull * 1024 * 1024 * 1024);
    u64 data_size = 0;
    void* raw = file_read_entire("data/raw/wikipedia_tokens.bin", &persist, &data_size);
    if (!raw) { console_write_line("[HATA] token dosyasi okunamadi"); return 1; }
    const u32* tokens = (const u32*)raw;
    u64 num_tokens = data_size / sizeof(u32);

    gpu_ops_init(&persist, "cuda/kernels.ptx");

    PCGState model_rng = pcg_seed(1337, 1);
    LMModel model = lm_init(&persist, &model_rng, VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, 1e-5f);

    u64 head_dim = D_MODEL / NUM_HEADS;
    u64 cshape[2] = { SEQ_LEN, head_dim / 2 };
    Tensor cos_t = tensor_create(&persist, cshape, 2);
    Tensor sin_t = tensor_create(&persist, cshape, 2);
    rope_build_tables(&cos_t, &sin_t, SEQ_LEN, head_dim, 10000.0f);
    Node* cos_leaf = node_leaf(&persist, cos_t, FALSE);
    Node* sin_leaf = node_leaf(&persist, sin_t, FALSE);

    u64 mshape[2] = { SEQ_LEN, SEQ_LEN };
    Tensor mask_t = tensor_create(&persist, mshape, 2);
    build_causal_mask(&mask_t);
    Node* mask_leaf = node_leaf(&persist, mask_t, FALSE);

    PCGState data_rng = pcg_seed(2026, 99);
    u32 cat_ids[BATCH_SIZE * SEQ_LEN];
    u32 cat_targets[BATCH_SIZE * SEQ_LEN];

    f64 t0 = timer_now_seconds();
    for (u32 step = 0; step < NUM_STEPS; step++) {
        for (u32 b = 0; b < BATCH_SIZE; b++) {
            i64 max_start = (i64)(num_tokens - SEQ_LEN - 1);
            i64 start = pcg_range_i64(&data_rng, 0, max_start);
            for (u32 t = 0; t < SEQ_LEN; t++) {
                cat_ids[b * SEQ_LEN + t] = tokens[start + t];
                cat_targets[b * SEQ_LEN + t] = tokens[start + t + 1];
            }
        }

        Allocator step_alloc = allocator_create(8192ull * 1024 * 1024);

        Node* logits = lm_forward(&step_alloc, &model, cat_ids, BATCH_SIZE, SEQ_LEN, cos_leaf, sin_leaf, mask_leaf, 1, 1);
        Node* loss = node_cross_entropy_loss(&step_alloc, logits, cat_targets);
        backward(&step_alloc, loss);

        f32 loss_val = loss->value.data[0];
        allocator_destroy(&step_alloc);

        f64 elapsed = timer_now_seconds() - t0;
        f64 gpu_seconds; u32 gpu_calls;
        gpu_ops_debug_stats(&gpu_seconds, &gpu_calls);
        console_write("adim "); console_write_u64(step);
        console_write(" | kayip="); write_fixed3((f64)loss_val);
        console_write(" | toplam_gecen="); write_fixed3(elapsed); console_write(" sn");
        console_write(" | gpu_cagri="); console_write_u64(gpu_calls);
        console_write_line("");
    }

    gpu_ops_shutdown();
    allocator_destroy(&persist);
    return 0;
}
