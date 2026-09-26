#include "loss.h"
#include "../tensor/tensor.h"
#include "../runtime/mathlib.h"

typedef struct CrossEntropyAux {
    Tensor probs; /* [seq_len, vocab_size], softmax olasiliklari (bitisik) */
    const u32* targets;
} CrossEntropyAux;

static void backward_cross_entropy(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* logits = self->parents[0];
    const CrossEntropyAux* aux = (const CrossEntropyAux*)self->aux_ptr;
    u64 seq_len = aux->probs.shape[0];
    u64 vocab = aux->probs.shape[1];
    f32 dy = self->grad.data[0]; /* skaler kayip gradyani (genelde 1.0) */
    f32 scale = dy / (f32)seq_len;

    for (u64 i = 0; i < seq_len; i++) {
        const f32* prow = aux->probs.data + i * vocab;
        f32* grow = logits->grad.data + i * vocab;
        u32 t = aux->targets[i];
        for (u64 j = 0; j < vocab; j++) {
            f32 g = prow[j];
            if (j == t) g -= 1.0f;
            grow[j] += g * scale;
        }
    }
}

Node* node_cross_entropy_loss(Allocator* alloc, Node* logits, const u32* targets) {
    u64 seq_len = logits->value.shape[0];
    u64 vocab = logits->value.shape[1];

    Tensor probs = tensor_create(alloc, logits->value.shape, 2);
    f32 total_loss = 0.0f;

    for (u64 i = 0; i < seq_len; i++) {
        const f32* row = logits->value.data + i * vocab;
        f32* prow = probs.data + i * vocab;

        f32 max_v = row[0];
        for (u64 j = 1; j < vocab; j++) if (row[j] > max_v) max_v = row[j];

        f32 sum_exp = 0.0f;
        for (u64 j = 0; j < vocab; j++) {
            f32 e = m_expf(row[j] - max_v);
            prow[j] = e;
            sum_exp += e;
        }
        for (u64 j = 0; j < vocab; j++) prow[j] /= sum_exp;

        u32 t = targets[i];
        f32 log_sum_exp = m_logf(sum_exp) + max_v;
        f32 loss_i = log_sum_exp - row[t];
        total_loss += loss_i;
    }
    total_loss /= (f32)seq_len;

    u64 out_shape[1] = { 1 };
    Tensor out = tensor_create(alloc, out_shape, 1);
    out.data[0] = total_loss;

    Node* n = node_make_custom(alloc, out, TRUE);
    n->parents[0] = logits; n->num_parents = 1;

    CrossEntropyAux* aux = (CrossEntropyAux*)allocator_alloc(alloc, sizeof(CrossEntropyAux));
    aux->probs = probs;
    aux->targets = targets;
    n->aux_ptr = aux;
    n->backward_fn = backward_cross_entropy;
    return n;
}
