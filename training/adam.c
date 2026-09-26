#include "adam.h"
#include "../runtime/mathlib.h"
#include "../runtime/thread.h"

#define ADAM_MAX_PARAMS_PER_THREAD 512

AdamOptimizer adam_create(Allocator* alloc, Node** params, u32 num_params,
                          f32 lr, f32 beta1, f32 beta2, f32 eps) {
    AdamOptimizer opt;
    opt.params = params;
    opt.num_params = num_params;
    opt.t = 0;
    opt.lr = lr; opt.beta1 = beta1; opt.beta2 = beta2; opt.eps = eps;

    opt.m = (Tensor*)allocator_alloc(alloc, (u64)num_params * sizeof(Tensor));
    opt.v = (Tensor*)allocator_alloc(alloc, (u64)num_params * sizeof(Tensor));
    for (u32 i = 0; i < num_params; i++) {
        opt.m[i] = tensor_create(alloc, params[i]->value.shape, params[i]->value.ndim);
        tensor_fill(&opt.m[i], 0.0f);
        opt.v[i] = tensor_create(alloc, params[i]->value.shape, params[i]->value.ndim);
        tensor_fill(&opt.v[i], 0.0f);
    }
    return opt;
}

static void adam_update_one(AdamOptimizer* opt, u32 i, f32 bc1, f32 bc2);

void adam_step(AdamOptimizer* opt) {
    opt->t++;
    f32 bc1 = 1.0f - m_powf(opt->beta1, (f32)opt->t);
    f32 bc2 = 1.0f - m_powf(opt->beta2, (f32)opt->t);

    /* Parametreler her zaman leaf/bitisik tensorlerdir (view degil),
     * bu yuzden duz (flat) indeksleme guvenlidir. */
    for (u32 i = 0; i < opt->num_params; i++) adam_update_one(opt, i, bc1, bc2);
}

static void adam_update_one(AdamOptimizer* opt, u32 i, f32 bc1, f32 bc2) {
    Node* p = opt->params[i];
    Tensor* m = &opt->m[i];
    Tensor* v = &opt->v[i];

    for (u64 j = 0; j < p->value.numel; j++) {
        f32 g = p->grad.data[j];
        m->data[j] = opt->beta1 * m->data[j] + (1.0f - opt->beta1) * g;
        v->data[j] = opt->beta2 * v->data[j] + (1.0f - opt->beta2) * g * g;

        f32 m_hat = m->data[j] / bc1;
        f32 v_hat = v->data[j] / bc2;

        p->value.data[j] -= opt->lr * m_hat / (m_sqrtf(v_hat) + opt->eps);
    }
}

typedef struct AdamRangeCtx {
    AdamOptimizer* opt;
    u32 indices[ADAM_MAX_PARAMS_PER_THREAD];
    u32 count;
    f32 bc1, bc2;
} AdamRangeCtx;

static u32 adam_worker(void* arg) {
    AdamRangeCtx* ctx = (AdamRangeCtx*)arg;
    for (u32 k = 0; k < ctx->count; k++) {
        adam_update_one(ctx->opt, ctx->indices[k], ctx->bc1, ctx->bc2);
    }
    return 0;
}

void adam_step_parallel(AdamOptimizer* opt, u32 num_threads) {
    opt->t++;
    f32 bc1 = 1.0f - m_powf(opt->beta1, (f32)opt->t);
    f32 bc2 = 1.0f - m_powf(opt->beta2, (f32)opt->t);

    if (num_threads <= 1 || opt->num_params < num_threads || num_threads > THREAD_MAX_HANDLES) {
        for (u32 i = 0; i < opt->num_params; i++) adam_update_one(opt, i, bc1, bc2);
        return;
    }

    /* Acgozlu (greedy) yuk dengesi: her parametre tensoru, o ana kadar
     * en az skaler-eleman yuku biriken thread'e atanir. Buyukten kucuge
     * onceden siralamaya gerek yok -- parametreler zaten cok farkli
     * boyutlarda (embed_table tek basina ~%36) oldugu icin acgozlu
     * atama pratikte iyi dengeler. */
    static AdamRangeCtx ctxs[THREAD_MAX_HANDLES];
    ThreadHandle threads[THREAD_MAX_HANDLES];
    u64 loads[THREAD_MAX_HANDLES];

    for (u32 t = 0; t < num_threads; t++) { ctxs[t].count = 0; loads[t] = 0; ctxs[t].opt = opt; ctxs[t].bc1 = bc1; ctxs[t].bc2 = bc2; }

    for (u32 i = 0; i < opt->num_params; i++) {
        u32 best = 0;
        for (u32 t = 1; t < num_threads; t++) if (loads[t] < loads[best]) best = t;
        if (ctxs[best].count < ADAM_MAX_PARAMS_PER_THREAD) {
            ctxs[best].indices[ctxs[best].count++] = i;
        }
        loads[best] += opt->params[i]->value.numel;
    }

    for (u32 t = 0; t < num_threads; t++) thread_create(&threads[t], adam_worker, &ctxs[t]);
    thread_join_all(threads, num_threads);
}

void adam_zero_grad(AdamOptimizer* opt) {
    for (u32 i = 0; i < opt->num_params; i++) node_zero_grad(opt->params[i]);
}
