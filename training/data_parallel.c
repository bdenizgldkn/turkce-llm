#include "data_parallel.h"
#include "../runtime/prng.h"
#include "../runtime/thread.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/rope.h"
#include "../model/attention.h"
#include "../model/loss.h"
#include "../model/gpu_ops.h"

void build_shadow_model(Allocator* alloc, const LMModel* orig, LMModel* shadow) {
    *shadow = *orig;
    shadow->embed_table = node_leaf(alloc, orig->embed_table->value, TRUE);
    for (u32 l = 0; l < orig->num_layers; l++) {
        const BlockWeights* ob = &orig->blocks[l];
        BlockWeights* sb = &shadow->blocks[l];
        sb->attn_norm_w = node_leaf(alloc, ob->attn_norm_w->value, TRUE);
        sb->attn.w_qkv = node_leaf(alloc, ob->attn.w_qkv->value, TRUE);
        sb->attn.b_qkv = node_leaf(alloc, ob->attn.b_qkv->value, TRUE);
        sb->attn.wo = node_leaf(alloc, ob->attn.wo->value, TRUE);
        sb->attn.bo = node_leaf(alloc, ob->attn.bo->value, TRUE);
        sb->ffn_norm_w = node_leaf(alloc, ob->ffn_norm_w->value, TRUE);
        sb->ffn.w_gate_up = node_leaf(alloc, ob->ffn.w_gate_up->value, TRUE);
        sb->ffn.b_gate_up = node_leaf(alloc, ob->ffn.b_gate_up->value, TRUE);
        sb->ffn.w_down = node_leaf(alloc, ob->ffn.w_down->value, TRUE);
        sb->ffn.b_down = node_leaf(alloc, ob->ffn.b_down->value, TRUE);
    }
    shadow->final_norm_w = node_leaf(alloc, orig->final_norm_w->value, TRUE);
}

typedef struct DPWorkerCtx {
    const LMModel* orig_model;
    const u32* tokens;
    u64 num_tokens;
    u64 seq_len;
    u64 seed_a, seed_b;
    u64 step_arena_size;
    i32 use_gpu_layers, use_gpu_output;
    i32 worker_id; /* bkz. model/gpu_ops.h: gpu_ops_set_worker_id */

    Allocator step_alloc;
    LMModel shadow;
    Node* shadow_params[LM_MAX_LAYERS * 12 + 2];
    u32 num_params;
    f32 loss_value;
} DPWorkerCtx;

static u32 dp_worker_run(void* arg) {
    DPWorkerCtx* ctx = (DPWorkerCtx*)arg;
    gpu_ops_set_worker_id(ctx->worker_id); /* bu OS thread'inin KENDI bagimsiz CUDA baglamini kullansin */
    ctx->step_alloc = allocator_create(ctx->step_arena_size);

    build_shadow_model(&ctx->step_alloc, ctx->orig_model, &ctx->shadow);
    ctx->num_params = lm_collect_params(&ctx->shadow, ctx->shadow_params);

    u64 d_model = ctx->orig_model->d_model;
    u64 num_heads = ctx->orig_model->num_heads;
    u64 head_dim = d_model / num_heads;
    u64 seq_len = ctx->seq_len;

    u64 cshape[2] = { seq_len, head_dim / 2 };
    Tensor cos_t = tensor_create(&ctx->step_alloc, cshape, 2);
    Tensor sin_t = tensor_create(&ctx->step_alloc, cshape, 2);
    rope_build_tables(&cos_t, &sin_t, seq_len, head_dim, 10000.0f);
    Node* cos_leaf = node_leaf(&ctx->step_alloc, cos_t, FALSE);
    Node* sin_leaf = node_leaf(&ctx->step_alloc, sin_t, FALSE);

    u64 mshape[2] = { seq_len, seq_len };
    Tensor mask_t = tensor_create(&ctx->step_alloc, mshape, 2);
    build_causal_mask(&mask_t);
    Node* mask_leaf = node_leaf(&ctx->step_alloc, mask_t, FALSE);

    PCGState rng = pcg_seed(ctx->seed_a, ctx->seed_b);
    i64 max_start = (i64)(ctx->num_tokens - seq_len - 1);
    i64 start = pcg_range_i64(&rng, 0, max_start);
    const u32* input_ids = ctx->tokens + start;
    const u32* target_ids = ctx->tokens + start + 1;

    Node* logits = lm_forward(&ctx->step_alloc, &ctx->shadow, input_ids, 1, seq_len, cos_leaf, sin_leaf, mask_leaf,
                               ctx->use_gpu_layers, ctx->use_gpu_output);
    Node* loss = node_cross_entropy_loss(&ctx->step_alloc, logits, target_ids);
    backward(&ctx->step_alloc, loss);

    ctx->loss_value = loss->value.data[0];
    return 0;
}

/* Paralel gradyan indirgemesi: her thread, HER parametrenin eleman
 * araliginin [numel*t/T, numel*(t+1)/T) dilimini isler (boylece yuk,
 * parametre boyutlarindan bagimsiz olarak esit dagilir). Her eleman icin
 * toplama sirasi tek-thread'li surumle AYNI (grad + w0 + w1 + ... sonra
 * * inv_workers) -- sonuc BIT BIT aynidir. */
typedef struct DPReduceCtx {
    Node** params;
    u32 num_params;
    DPWorkerCtx* workers;
    u32 num_workers;
    u32 thread_index;
    u32 num_threads;
} DPReduceCtx;

static u32 dp_reduce_run(void* arg) {
    DPReduceCtx* r = (DPReduceCtx*)arg;
    f32 inv_workers = 1.0f / (f32)r->num_workers;
    for (u32 i = 0; i < r->num_params; i++) {
        u64 n = r->params[i]->grad.numel;
        u64 lo = n * r->thread_index / r->num_threads;
        u64 hi = n * (r->thread_index + 1) / r->num_threads;
        f32* dst = r->params[i]->grad.data;
        for (u32 w = 0; w < r->num_workers; w++) {
            const f32* src = r->workers[w].shadow_params[i]->grad.data;
            for (u64 k = lo; k < hi; k++) dst[k] += src[k];
        }
        for (u64 k = lo; k < hi; k++) dst[k] *= inv_workers;
    }
    return 0;
}

f32 data_parallel_step(const LMModel* orig_model, Node** params, u32 num_params,
                        const u32* tokens, u64 num_tokens, u64 seq_len,
                        u64 step_arena_size, u32 num_workers, u64 seed_base,
                        i32 use_gpu_layers, i32 use_gpu_output) {
    static DPWorkerCtx workers[DATA_PARALLEL_MAX_WORKERS];
    ThreadHandle threads[DATA_PARALLEL_MAX_WORKERS];

    for (u32 w = 0; w < num_workers; w++) {
        workers[w].orig_model = orig_model;
        workers[w].tokens = tokens;
        workers[w].num_tokens = num_tokens;
        workers[w].seq_len = seq_len;
        workers[w].step_arena_size = step_arena_size;
        workers[w].seed_a = seed_base * 1000003ull + w;
        workers[w].seed_b = 99ull + w;
        workers[w].use_gpu_layers = use_gpu_layers;
        workers[w].use_gpu_output = use_gpu_output;
        workers[w].worker_id = (i32)w;
    }

    for (u32 w = 0; w < num_workers; w++) thread_create(&threads[w], dp_worker_run, &workers[w]);
    thread_join_all(threads, num_workers);

    /* Gradyanlar (orijinal params ve shadow'lar) node_leaf ile taze
     * ayrilir -- bitisik ve ayni sekilde oldugu varsayimi burada
     * dogrudan ham dizi erisimi icin kullanilir. */
    static DPReduceCtx reducers[DATA_PARALLEL_MAX_WORKERS];
    for (u32 t = 0; t < num_workers; t++) {
        reducers[t].params = params;
        reducers[t].num_params = num_params;
        reducers[t].workers = workers;
        reducers[t].num_workers = num_workers;
        reducers[t].thread_index = t;
        reducers[t].num_threads = num_workers;
    }
    for (u32 t = 0; t < num_workers; t++) thread_create(&threads[t], dp_reduce_run, &reducers[t]);
    thread_join_all(threads, num_workers);

    f32 loss_sum = 0.0f;
    for (u32 w = 0; w < num_workers; w++) loss_sum += workers[w].loss_value;

    for (u32 w = 0; w < num_workers; w++) allocator_destroy(&workers[w].step_alloc);

    return loss_sum / (f32)num_workers;
}
