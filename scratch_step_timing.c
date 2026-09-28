#include "runtime/types.h"
#include "runtime/memory.h"
#include "runtime/console.h"
#include "runtime/timer.h"
#include "runtime/prng.h"
#include "model/lm_model.h"
#include "model/gpu_train.h"
#include "training/adam.h"

#define VOCAB_SIZE 31769u
#define D_MODEL 384u
#define NUM_HEADS 6u
#define NUM_LAYERS 24u
#define D_FF 1024u
#define SEQ_LEN 1024u
#define BATCH_SEQS 14u
#define EPS 1e-5f

int main(void) {
    console_write_line("=== Faz5 gercek adim suresi olcumu ===");
    Allocator persist = allocator_create(32ull * 1024 * 1024 * 1024);

    PCGState rng = pcg_seed(1, 1);
    LMModel model = lm_init(&persist, &rng, VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, EPS);
    Node* params[LM_MAX_LAYERS * 12 + 2];
    u32 num_params = lm_collect_params(&model, params);
    AdamOptimizer opt = adam_create(&persist, params, num_params, 3e-4f, 0.9f, 0.999f, 1e-8f);

    GpuTrainer gt = gpu_trainer_create(&persist, "cuda/train_kernels.ptx", &model, params, num_params,
                                       BATCH_SEQS, SEQ_LEN);
    gpu_trainer_upload(&gt, params, &opt);

    u32* ids = (u32*)allocator_alloc(&persist, (u64)BATCH_SEQS * SEQ_LEN * sizeof(u32));
    u32* tgt = (u32*)allocator_alloc(&persist, (u64)BATCH_SEQS * SEQ_LEN * sizeof(u32));
    PCGState drng = pcg_seed(5, 5);
    for (u64 i = 0; i < (u64)BATCH_SEQS * SEQ_LEN; i++) {
        ids[i] = (u32)pcg_range_i64(&drng, 0, (i64)VOCAB_SIZE - 1);
        tgt[i] = (u32)pcg_range_i64(&drng, 0, (i64)VOCAB_SIZE - 1);
    }

    for (int i = 0; i < 3; i++) {
        f32 loss = gpu_trainer_forward_backward(&gt, ids, tgt);
        gpu_trainer_adam_step(&gt, &opt);
        console_write("  isinma kayip: "); console_write_u64((u64)(loss * 1000)); console_write_line("");
    }

    const int REPEAT = 20;
    f64 t0 = timer_now_seconds();
    for (int i = 0; i < REPEAT; i++) {
        f32 loss = gpu_trainer_forward_backward(&gt, ids, tgt);
        gpu_trainer_adam_step(&gt, &opt);
        (void)loss;
    }
    f64 t1 = timer_now_seconds();

    f64 per_step = (t1 - t0) / REPEAT;
    console_write("Adim basina sure: "); console_write_u64((u64)(per_step * 1000.0)); console_write_line(" ms");
    console_write("70.000 adim tahmini: "); console_write_u64((u64)(per_step * 70000.0 / 3600.0)); console_write_line(" saat");

    return 0;
}
