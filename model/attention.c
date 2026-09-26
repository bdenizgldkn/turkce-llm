#include "attention.h"
#include "model_ops.h"
#include "gpu_ops.h"
#include "../runtime/mathlib.h"

void build_causal_mask(Tensor* mask_out) {
    u64 seq = mask_out->shape[0];
    for (u64 i = 0; i < seq; i++) {
        for (u64 j = 0; j < seq; j++) {
            u64 idx[2] = { i, j };
            f32 v = (j <= i) ? 0.0f : -1e9f; /* gercek -inf yerine sayisal olarak guvenli buyuk negatif */
            mask_out->data[tensor_compute_offset(mask_out->strides, idx, 2)] = v;
        }
    }
}

Node* causal_self_attention(Allocator* alloc, Node* x, const AttentionWeights* w,
                             u32 num_heads, u32 batch_size, u64 seq_len,
                             Node* cos_leaf, Node* sin_leaf, Node* causal_mask,
                             i32 use_gpu) {
    u64 d_model = x->value.shape[1];

    /* Fuzyonlu Q/K/V: TUM batch icin TEK matris carpimi, sonra uc
     * parcaya bolunur -- matematiksel olarak ayri wq/wk/wv carpimlarinin
     * AYNISI, ama tek bir GPU gidis-gelisi (bkz. attention.h). */
    Node* QKV = node_add_bias(alloc, use_gpu ? node_matmul2d_gpu(alloc, x, w->w_qkv) : node_matmul2d(alloc, x, w->w_qkv), w->b_qkv);
    Node* Q = node_slice_cols(alloc, QKV, 0, d_model);
    Node* K = node_slice_cols(alloc, QKV, d_model, d_model);
    Node* V = node_slice_cols(alloc, QKV, 2 * d_model, d_model);

    u64 head_dim = d_model / (u64)num_heads;
    f32 scale = 1.0f / m_sqrtf((f32)head_dim);

    /* Dikkat (attention) skoru MUTLAKA dizi-bazinda kalmalidir -- bir
     * dizi ASLA baska bir dizinin token'ina bakmamali. Bu yuzden buyuk
     * projeksiyonlar TUM batch icin birlikte hesaplansa da, skor/softmax/
     * agirlikli-toplam kismi HER batch ogesi icin ayri ayri (satir
     * dilimi, bkz. node_slice_rows) yapilir. batch_size=1 durumunda bu
     * dis dongu tek bir kez calisir ve ORIJINAL davranisla BIREBIR
     * AYNIDIR. */
    Node* batch_out = NULL_PTR;
    for (u32 b = 0; b < batch_size; b++) {
        u64 row_start = (u64)b * seq_len;
        Node* Qb = (batch_size == 1) ? Q : node_slice_rows(alloc, Q, row_start, seq_len);
        Node* Kb = (batch_size == 1) ? K : node_slice_rows(alloc, K, row_start, seq_len);
        Node* Vb = (batch_size == 1) ? V : node_slice_rows(alloc, V, row_start, seq_len);

        Node* concat_out = NULL_PTR;
        for (u32 h = 0; h < num_heads; h++) {
            u64 col_start = (u64)h * head_dim;
            Node* Qh = node_rope(alloc, node_slice_cols(alloc, Qb, col_start, head_dim), cos_leaf, sin_leaf);
            Node* Kh = node_rope(alloc, node_slice_cols(alloc, Kb, col_start, head_dim), cos_leaf, sin_leaf);
            Node* Vh = node_slice_cols(alloc, Vb, col_start, head_dim);

            Node* scores = node_matmul2d(alloc, Qh, node_transpose(alloc, Kh, 0, 1));
            scores = node_scale(alloc, scores, scale);
            scores = node_add(alloc, scores, causal_mask);

            Node* attn = node_softmax_lastdim(alloc, scores);
            Node* head_out = node_matmul2d(alloc, attn, Vh);

            concat_out = (h == 0) ? head_out : node_concat_cols2(alloc, concat_out, head_out);
        }
        batch_out = (b == 0) ? concat_out : node_concat_rows2(alloc, batch_out, concat_out);
    }

    return node_add_bias(alloc, use_gpu ? node_matmul2d_gpu(alloc, batch_out, w->wo) : node_matmul2d(alloc, batch_out, w->wo), w->bo);
}
