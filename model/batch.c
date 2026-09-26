#include "batch.h"

void transformer_block_batch(Allocator* alloc, Node** x_batch, u32 batch_size,
                              const BlockWeights* w, u32 num_heads,
                              Node* cos_leaf, Node* sin_leaf, Node* causal_mask, f32 eps,
                              i32 use_gpu, Node** out_batch) {
    for (u32 b = 0; b < batch_size; b++) {
        u64 seq_len = x_batch[b]->value.shape[0];
        /* Burada transformer_block'un YENI (Katman 18) tensor-batching
         * batch_size parametresine 1 veriyoruz -- bu fonksiyon (Katman 7)
         * KENDI agirlik-paylasimli batch mekanizmasini kullanir (her
         * x_batch[b] ayri, YIGINLANMAMIS bir dizidir). */
        out_batch[b] = transformer_block(alloc, x_batch[b], w, num_heads, 1, seq_len, cos_leaf, sin_leaf, causal_mask, eps, use_gpu);
    }
}

Node* batch_mean_loss(Allocator* alloc, Node** losses, u32 batch_size) {
    Node* acc = losses[0];
    for (u32 b = 1; b < batch_size; b++) {
        acc = node_add(alloc, acc, losses[b]);
    }
    return node_scale(alloc, acc, 1.0f / (f32)batch_size);
}
