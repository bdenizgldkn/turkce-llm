/* Katman 7 - Tam Transformer bloğu (LLaMA tarzi, pre-norm):
 *   x = x + Attention(RMSNorm(x))
 *   x = x + SwiGLU(RMSNorm(x))
 */
#ifndef MODEL_TRANSFORMER_BLOCK_H
#define MODEL_TRANSFORMER_BLOCK_H

#include "../autograd/node.h"
#include "attention.h"
#include "feedforward.h"

typedef struct BlockWeights {
    Node* attn_norm_w; /* RMSNorm agirligi [d_model] */
    AttentionWeights attn;
    Node* ffn_norm_w;   /* RMSNorm agirligi [d_model] */
    FeedForwardWeights ffn;
} BlockWeights;

/* x: [batch_size*seq_len, d_model] (bkz. attention.h: gercek tensor-
 * batching, batch_size=1 -> orijinal davranisla birebir ayni). */
Node* transformer_block(Allocator* alloc, Node* x, const BlockWeights* w, u32 num_heads,
                         u32 batch_size, u64 seq_len,
                         Node* cos_leaf, Node* sin_leaf, Node* causal_mask, f32 eps, i32 use_gpu);

#endif /* MODEL_TRANSFORMER_BLOCK_H */
