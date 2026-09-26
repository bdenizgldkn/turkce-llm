#include "rope.h"
#include "../runtime/mathlib.h"

void rope_build_tables(Tensor* cos_out, Tensor* sin_out, u64 seq_len, u64 head_dim, f32 base) {
    u64 half = head_dim / 2;
    for (u64 pos = 0; pos < seq_len; pos++) {
        for (u64 k = 0; k < half; k++) {
            f32 exponent = -(2.0f * (f32)k) / (f32)head_dim;
            f32 theta = m_powf(base, exponent);
            f32 angle = (f32)pos * theta;

            u64 idx[2] = { pos, k };
            u64 off = tensor_compute_offset(cos_out->strides, idx, 2);
            cos_out->data[off] = m_cosf(angle);
            sin_out->data[off] = m_sinf(angle);
        }
    }
}

Tensor tensor_rope(Allocator* alloc, const Tensor* x, const Tensor* cos_t, const Tensor* sin_t) {
    /* x bitisik olmayabilir (orn. multi-head sutun dilimlemesinden gelen
     * bir view) -- bu yuzden x'i kendi stride'lariyla okuyoruz; out ve
     * cos_t/sin_t her zaman bitisiktir (varsayim: cagiran taraf bunlari
     * boyle uretir). */
    Tensor out = tensor_create(alloc, x->shape, x->ndim);
    u64 seq = x->shape[0];
    u64 hd = x->shape[1];
    u64 half = hd / 2;

    for (u64 p = 0; p < seq; p++) {
        f32* orow = out.data + p * hd;
        const f32* crow = cos_t->data + p * half;
        const f32* srow = sin_t->data + p * half;

        for (u64 k = 0; k < half; k++) {
            u64 i0[2] = { p, 2 * k };
            u64 i1[2] = { p, 2 * k + 1 };
            f32 x0 = x->data[tensor_compute_offset(x->strides, i0, 2)];
            f32 x1 = x->data[tensor_compute_offset(x->strides, i1, 2)];
            f32 c = crow[k];
            f32 s = srow[k];
            orow[2 * k]     = x0 * c - x1 * s;
            orow[2 * k + 1] = x0 * s + x1 * c;
        }
    }
    return out;
}
