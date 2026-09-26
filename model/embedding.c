#include "embedding.h"
#include "../tensor/tensor.h"

static void backward_embedding(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* table = self->parents[0];
    const u32* ids = (const u32*)self->aux_ptr;
    u64 d_model = self->value.shape[1];

    for (u64 i = 0; i < self->value.shape[0]; i++) {
        f32* grad_row = table->grad.data + (u64)ids[i] * d_model;
        const f32* dy_row = self->grad.data + i * d_model;
        for (u64 j = 0; j < d_model; j++) grad_row[j] += dy_row[j];
    }
}

Node* node_embedding_lookup(Allocator* alloc, Node* embed_table, const u32* token_ids, u64 seq_len) {
    u64 d_model = embed_table->value.shape[1];
    u64 shape[2] = { seq_len, d_model };
    Tensor v = tensor_create(alloc, shape, 2);

    for (u64 i = 0; i < seq_len; i++) {
        const f32* src_row = embed_table->value.data + (u64)token_ids[i] * d_model;
        f32* dst_row = v.data + i * d_model;
        for (u64 j = 0; j < d_model; j++) dst_row[j] = src_row[j];
    }

    Node* n = node_make_custom(alloc, v, TRUE);
    n->parents[0] = embed_table; n->num_parents = 1;
    n->aux_ptr = token_ids;
    n->backward_fn = backward_embedding;
    return n;
}
