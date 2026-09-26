#include "transformer_block.h"
#include "model_ops.h"

Node* transformer_block(Allocator* alloc, Node* x, const BlockWeights* w, u32 num_heads,
                         u32 batch_size, u64 seq_len,
                         Node* cos_leaf, Node* sin_leaf, Node* causal_mask, f32 eps, i32 use_gpu) {
    Node* normed1 = node_rmsnorm(alloc, x, w->attn_norm_w, eps);
    Node* attn_out = causal_self_attention(alloc, normed1, &w->attn, num_heads, batch_size, seq_len, cos_leaf, sin_leaf, causal_mask, use_gpu);
    Node* x2 = node_add(alloc, x, attn_out);

    Node* normed2 = node_rmsnorm(alloc, x2, w->ffn_norm_w, eps);
    Node* ffn_out = swiglu_feedforward(alloc, normed2, &w->ffn, use_gpu);
    Node* x3 = node_add(alloc, x2, ffn_out);

    return x3;
}
