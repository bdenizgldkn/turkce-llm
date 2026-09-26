#include "model_ops.h"
#include "rope.h"
#include "../runtime/mathlib.h"

/* ================= add_bias ================= */

static void backward_add_bias(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* x = self->parents[0];
    Node* b = self->parents[1];

    tensor_add_inplace(&x->grad, &self->grad);

    for (u64 i = 0; i < self->grad.shape[0]; i++) {
        for (u64 j = 0; j < self->grad.shape[1]; j++) {
            u64 idx[2] = { i, j };
            f32 g = self->grad.data[tensor_compute_offset(self->grad.strides, idx, 2)];
            u64 bidx[1] = { j };
            b->grad.data[tensor_compute_offset(b->grad.strides, bidx, 1)] += g;
        }
    }
}

Node* node_add_bias(Allocator* alloc, Node* x, Node* bias) {
    Tensor v = tensor_add_bias_2d(alloc, &x->value, &bias->value);
    Node* n = node_make_custom(alloc, v, TRUE);
    n->parents[0] = x; n->parents[1] = bias; n->num_parents = 2;
    n->backward_fn = backward_add_bias;
    return n;
}

/* ================= slice_cols ================= */

static void backward_slice_cols(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* a = self->parents[0];
    Tensor view = tensor_slice_cols(&a->grad, self->aux_dim0, self->value.shape[1]);
    tensor_add_inplace(&view, &self->grad);
}

Node* node_slice_cols(Allocator* alloc, Node* a, u64 col_start, u64 num_cols) {
    Tensor v = tensor_slice_cols(&a->value, col_start, num_cols);
    Node* n = node_make_custom(alloc, v, TRUE);
    n->parents[0] = a; n->num_parents = 1;
    n->aux_dim0 = (u32)col_start;
    n->backward_fn = backward_slice_cols;
    return n;
}

/* ================= rmsnorm ================= */

static void backward_rmsnorm(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* x = self->parents[0];
    Node* w = self->parents[1];
    f32 eps = self->aux_scalar;

    u64 last = self->value.shape[self->value.ndim - 1];
    u64 outer = self->value.numel / last;

    for (u64 o = 0; o < outer; o++) {
        const f32* xr = x->value.data + o * last;
        const f32* dy = self->grad.data + o * last;
        f32* dxr = x->grad.data + o * last;

        f32 ss = 0.0f;
        for (u64 i = 0; i < last; i++) ss += xr[i] * xr[i];
        f32 ms = ss / (f32)last;
        f32 r = 1.0f / m_sqrtf(ms + eps);
        f32 r3 = r * r * r;

        f32 S = 0.0f;
        for (u64 i = 0; i < last; i++) S += dy[i] * w->value.data[i] * xr[i];

        for (u64 j = 0; j < last; j++) {
            dxr[j] += w->value.data[j] * r * dy[j] - (r3 / (f32)last) * xr[j] * S;
            w->grad.data[j] += dy[j] * xr[j] * r;
        }
    }
}

Node* node_rmsnorm(Allocator* alloc, Node* x, Node* weight, f32 eps) {
    Tensor v = tensor_rmsnorm(alloc, &x->value, &weight->value, eps);
    Node* n = node_make_custom(alloc, v, TRUE);
    n->parents[0] = x; n->parents[1] = weight; n->num_parents = 2;
    n->aux_scalar = eps;
    n->backward_fn = backward_rmsnorm;
    return n;
}

/* ================= RoPE ================= */

static void backward_rope(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* x = self->parents[0];
    Node* cos_n = self->parents[1];
    Node* sin_n = self->parents[2];

    u64 seq = self->value.shape[0];
    u64 hd = self->value.shape[1];
    u64 half = hd / 2;

    for (u64 p = 0; p < seq; p++) {
        const f32* dy = self->grad.data + p * hd;
        f32* dx = x->grad.data + p * hd;
        const f32* crow = cos_n->value.data + p * half;
        const f32* srow = sin_n->value.data + p * half;

        for (u64 k = 0; k < half; k++) {
            f32 dy0 = dy[2 * k];
            f32 dy1 = dy[2 * k + 1];
            f32 c = crow[k];
            f32 s = srow[k];
            dx[2 * k]     += dy0 * c + dy1 * s;
            dx[2 * k + 1] += -dy0 * s + dy1 * c;
        }
    }
}

Node* node_rope(Allocator* alloc, Node* x, Node* cos_leaf, Node* sin_leaf) {
    Tensor v = tensor_rope(alloc, &x->value, &cos_leaf->value, &sin_leaf->value);
    Node* n = node_make_custom(alloc, v, TRUE);
    n->parents[0] = x; n->parents[1] = cos_leaf; n->parents[2] = sin_leaf; n->num_parents = 3;
    n->backward_fn = backward_rope;
    return n;
}

/* ================= concat_cols2 ================= */

static void backward_concat_cols2(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* a = self->parents[0];
    Node* b = self->parents[1];
    u64 ca = a->value.shape[1];
    u64 cb = b->value.shape[1];

    Tensor ga = tensor_slice_cols(&self->grad, 0, ca);
    Tensor gb = tensor_slice_cols(&self->grad, ca, cb);
    tensor_add_inplace(&a->grad, &ga);
    tensor_add_inplace(&b->grad, &gb);
}

Node* node_concat_cols2(Allocator* alloc, Node* a, Node* b) {
    Tensor v = tensor_concat_cols2(alloc, &a->value, &b->value);
    Node* n = node_make_custom(alloc, v, TRUE);
    n->parents[0] = a; n->parents[1] = b; n->num_parents = 2;
    n->backward_fn = backward_concat_cols2;
    return n;
}

/* ================= slice_rows / concat_rows2 (Katman 18) ================= */

static void backward_slice_rows(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* a = self->parents[0];
    Tensor view = tensor_slice_rows(&a->grad, self->aux_dim0, self->value.shape[0]);
    tensor_add_inplace(&view, &self->grad);
}

Node* node_slice_rows(Allocator* alloc, Node* a, u64 row_start, u64 num_rows) {
    Tensor v = tensor_slice_rows(&a->value, row_start, num_rows);
    Node* n = node_make_custom(alloc, v, TRUE);
    n->parents[0] = a; n->num_parents = 1;
    n->aux_dim0 = (u32)row_start;
    n->backward_fn = backward_slice_rows;
    return n;
}

static void backward_concat_rows2(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* a = self->parents[0];
    Node* b = self->parents[1];
    u64 ra = a->value.shape[0];
    u64 rb = b->value.shape[0];

    Tensor ga = tensor_slice_rows(&self->grad, 0, ra);
    Tensor gb = tensor_slice_rows(&self->grad, ra, rb);
    tensor_add_inplace(&a->grad, &ga);
    tensor_add_inplace(&b->grad, &gb);
}

Node* node_concat_rows2(Allocator* alloc, Node* a, Node* b) {
    Tensor v = tensor_concat_rows2(alloc, &a->value, &b->value);
    Node* n = node_make_custom(alloc, v, TRUE);
    n->parents[0] = a; n->parents[1] = b; n->num_parents = 2;
    n->backward_fn = backward_concat_rows2;
    return n;
}
