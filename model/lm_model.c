#include "lm_model.h"
#include "embedding.h"
#include "model_ops.h"
#include "gpu_ops.h"
#include "../tensor/tensor.h"

static Node* rand_leaf2(Allocator* alloc, PCGState* rng, u64 rows, u64 cols, f32 std) {
    u64 shape[2] = { rows, cols };
    Tensor t = tensor_create(alloc, shape, 2);
    for (u64 i = 0; i < rows * cols; i++) t.data[i] = (f32)pcg_gaussian(rng, 0.0, (f64)std);
    return node_leaf(alloc, t, TRUE);
}

static Node* const_leaf1(Allocator* alloc, u64 n, f32 value) {
    u64 shape[1] = { n };
    Tensor t = tensor_create(alloc, shape, 1);
    tensor_fill(&t, value);
    return node_leaf(alloc, t, TRUE);
}

static AttentionWeights init_attention(Allocator* alloc, PCGState* rng, u32 d_model) {
    AttentionWeights w;
    f32 std = 0.02f;
    w.w_qkv = rand_leaf2(alloc, rng, d_model, 3u * d_model, std); w.b_qkv = const_leaf1(alloc, 3u * d_model, 0.0f);
    w.wo = rand_leaf2(alloc, rng, d_model, d_model, std); w.bo = const_leaf1(alloc, d_model, 0.0f);
    return w;
}

static FeedForwardWeights init_feedforward(Allocator* alloc, PCGState* rng, u32 d_model, u32 d_ff) {
    FeedForwardWeights w;
    f32 std = 0.02f;
    w.w_gate_up = rand_leaf2(alloc, rng, d_model, 2u * d_ff, std); w.b_gate_up = const_leaf1(alloc, 2u * d_ff, 0.0f);
    w.w_down = rand_leaf2(alloc, rng, d_ff, d_model, std); w.b_down = const_leaf1(alloc, d_model, 0.0f);
    return w;
}

LMModel lm_init(Allocator* alloc, PCGState* rng, u32 vocab_size, u32 d_model,
                 u32 num_heads, u32 num_layers, u32 d_ff, f32 eps) {
    LMModel m;
    m.vocab_size = vocab_size;
    m.d_model = d_model;
    m.num_heads = num_heads;
    m.num_layers = num_layers;
    m.eps = eps;

    /* Gomme tablosu: kucuk std ile baslatilir (buyuk vocab ile
     * toplam parametrenin buyuk kismini olusturur -- bkz. PROJE_PLANI.md). */
    m.embed_table = rand_leaf2(alloc, rng, vocab_size, d_model, 0.02f);

    for (u32 l = 0; l < num_layers; l++) {
        m.blocks[l].attn_norm_w = const_leaf1(alloc, d_model, 1.0f);
        m.blocks[l].attn = init_attention(alloc, rng, d_model);
        m.blocks[l].ffn_norm_w = const_leaf1(alloc, d_model, 1.0f);
        m.blocks[l].ffn = init_feedforward(alloc, rng, d_model, d_ff);
    }

    m.final_norm_w = const_leaf1(alloc, d_model, 1.0f);

    return m;
}

Node* lm_forward(Allocator* alloc, LMModel* m, const u32* token_ids,
                  u32 batch_size, u64 seq_len,
                  Node* cos_leaf, Node* sin_leaf, Node* causal_mask,
                  i32 use_gpu_layers, i32 use_gpu_output) {
    Node* x = node_embedding_lookup(alloc, m->embed_table, token_ids, (u64)batch_size * seq_len);

    for (u32 l = 0; l < m->num_layers; l++) {
        x = transformer_block(alloc, x, &m->blocks[l], m->num_heads, batch_size, seq_len, cos_leaf, sin_leaf, causal_mask, m->eps, use_gpu_layers);
    }

    x = node_rmsnorm(alloc, x, m->final_norm_w, m->eps);

    /* Agirlik baglantili (tied) cikis projeksiyonu: logits = x @ embed_table^T.
     * Bu tek islem, buyuk vocab_size yuzunden toplam FLOP'larin ezici
     * cogunlugunu olusturur -- bu yuzden istege bagli olarak GPU'ya
     * tasinabilir (bkz. model/gpu_ops.h, ~200x hizlanma olculdu). */
    Node* embed_t = node_transpose(alloc, m->embed_table, 0, 1);
    Node* logits = use_gpu_output ? node_matmul2d_gpu(alloc, x, embed_t)
                                   : node_matmul2d(alloc, x, embed_t);
    return logits;
}

u32 lm_collect_params(LMModel* m, Node** out_params) {
    u32 n = 0;
    out_params[n++] = m->embed_table;
    for (u32 l = 0; l < m->num_layers; l++) {
        BlockWeights* b = &m->blocks[l];
        out_params[n++] = b->attn_norm_w;
        out_params[n++] = b->attn.w_qkv; out_params[n++] = b->attn.b_qkv;
        out_params[n++] = b->attn.wo; out_params[n++] = b->attn.bo;
        out_params[n++] = b->ffn_norm_w;
        out_params[n++] = b->ffn.w_gate_up; out_params[n++] = b->ffn.b_gate_up;
        out_params[n++] = b->ffn.w_down; out_params[n++] = b->ffn.b_down;
    }
    out_params[n++] = m->final_norm_w;
    return n;
}
