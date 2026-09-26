#include "feedforward.h"
#include "model_ops.h"
#include "gpu_ops.h"

Node* swiglu_feedforward(Allocator* alloc, Node* x, const FeedForwardWeights* w, i32 use_gpu) {
    u64 d_ff = w->w_down->value.shape[0];

    /* Fuzyonlu gate/up: TEK matris carpimi, sonra ikiye bolunur. */
    Node* GATE_UP = node_add_bias(alloc, use_gpu ? node_matmul2d_gpu(alloc, x, w->w_gate_up) : node_matmul2d(alloc, x, w->w_gate_up), w->b_gate_up);
    Node* gate = node_slice_cols(alloc, GATE_UP, 0, d_ff);
    Node* up   = node_slice_cols(alloc, GATE_UP, d_ff, d_ff);

    Node* silu_gate = node_mul(alloc, gate, node_sigmoid(alloc, gate)); /* silu(x) = x*sigmoid(x) */
    Node* hidden = node_mul(alloc, silu_gate, up);

    return node_add_bias(alloc, use_gpu ? node_matmul2d_gpu(alloc, hidden, w->w_down) : node_matmul2d(alloc, hidden, w->w_down), w->b_down);
}
