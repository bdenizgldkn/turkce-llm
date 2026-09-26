/* Katman 7 - SwiGLU ileri besleme (feed-forward) katmani.
 * LLaMA tarzi: out = W_down( silu(W_gate(x)) * W_up(x) )
 * silu(x) = x*sigmoid(x); mevcut node_sigmoid/node_mul'dan bilesim. */
#ifndef MODEL_FEEDFORWARD_H
#define MODEL_FEEDFORWARD_H

#include "../autograd/node.h"

/* KATMAN 17 - FUZYONLU (fused) gate/up projeksiyonu: w_gate/w_up tek
 * bir [d_model, 2*d_ff] matriste birlestirilir, tek matris carpimiyla
 * hesaplanip sonra ikiye bolunur -- ayni GPU-cagri-azaltma gerekcesi
 * (bkz. model/attention.h). */
typedef struct FeedForwardWeights {
    Node* w_gate_up; Node* b_gate_up; /* [d_model, 2*d_ff], [2*d_ff] */
    Node* w_down; Node* b_down;       /* [d_ff, d_model] */
} FeedForwardWeights;

/* use_gpu != 0 ise gate/up/down matris carpimlari GPU'da hesaplanir
 * (bkz. model/gpu_ops.h). */
Node* swiglu_feedforward(Allocator* alloc, Node* x, const FeedForwardWeights* w, i32 use_gpu);

#endif /* MODEL_FEEDFORWARD_H */
