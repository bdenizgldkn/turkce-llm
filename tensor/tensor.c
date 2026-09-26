#include "tensor.h"
#include "../runtime/mathlib.h"

static void compute_contiguous_strides(const u64* shape, u32 ndim, i64* out_strides) {
    i64 acc = 1;
    for (i32 d = (i32)ndim - 1; d >= 0; d--) {
        out_strides[d] = acc;
        acc *= (i64)shape[d];
    }
}

static u64 numel_of(const u64* shape, u32 ndim) {
    u64 n = 1;
    for (u32 d = 0; d < ndim; d++) n *= shape[d];
    return n;
}

u64 tensor_compute_offset(const i64* strides, const u64* idx, u32 ndim) {
    i64 off = 0;
    for (u32 d = 0; d < ndim; d++) off += (i64)idx[d] * strides[d];
    return (u64)off;
}

void tensor_flat_to_multi(u64 flat, const u64* shape, u32 ndim, u64* out_idx) {
    for (i32 d = (i32)ndim - 1; d >= 0; d--) {
        out_idx[d] = flat % shape[d];
        flat /= shape[d];
    }
}

/* Dosya-ici kisa takma adlar (asagidaki mevcut cagrilari degistirmeden
 * okunabilirligi korumak icin). */
#define compute_offset tensor_compute_offset
#define flat_to_multi tensor_flat_to_multi

Tensor tensor_create(Allocator* alloc, const u64* shape, u32 ndim) {
    Tensor t;
    t.ndim = ndim;
    for (u32 d = 0; d < ndim; d++) t.shape[d] = shape[d];
    compute_contiguous_strides(t.shape, ndim, t.strides);
    t.numel = numel_of(shape, ndim);
    t.data = (f32*)allocator_alloc(alloc, t.numel * sizeof(f32));
    t.owns_data = TRUE;
    return t;
}

void tensor_free(Allocator* alloc, Tensor* t) {
    if (t->owns_data && alloc && t->data) {
        allocator_free(alloc, t->data);
    }
    t->data = NULL_PTR;
    t->owns_data = FALSE;
    t->numel = 0;
}

bool32 tensor_is_contiguous(const Tensor* t) {
    i64 expected[TENSOR_MAX_DIMS];
    compute_contiguous_strides(t->shape, t->ndim, expected);
    for (u32 d = 0; d < t->ndim; d++) {
        if (t->strides[d] != expected[d]) return FALSE;
    }
    return TRUE;
}

void tensor_fill(Tensor* t, f32 value) {
    u64 idx[TENSOR_MAX_DIMS];
    for (u64 flat = 0; flat < t->numel; flat++) {
        flat_to_multi(flat, t->shape, t->ndim, idx);
        t->data[compute_offset(t->strides, idx, t->ndim)] = value;
    }
}

void tensor_copy_from(Tensor* dst, const Tensor* src) {
    u64 idx[TENSOR_MAX_DIMS];
    for (u64 flat = 0; flat < dst->numel; flat++) {
        flat_to_multi(flat, dst->shape, dst->ndim, idx);
        f32 v = src->data[compute_offset(src->strides, idx, src->ndim)];
        dst->data[compute_offset(dst->strides, idx, dst->ndim)] = v;
    }
}

f32 tensor_get(const Tensor* t, const u64* idx) {
    return t->data[compute_offset(t->strides, idx, t->ndim)];
}

void tensor_set(Tensor* t, const u64* idx, f32 value) {
    t->data[compute_offset(t->strides, idx, t->ndim)] = value;
}

Tensor tensor_reshape(const Tensor* src, const u64* new_shape, u32 new_ndim) {
    /* NOT: basitlik icin sadece bitisik (contiguous) kaynaklar desteklenir;
     * cagiran taraf gerekirse once tensor_contiguous ile somutlastirmalidir. */
    Tensor out;
    out.ndim = new_ndim;
    for (u32 d = 0; d < new_ndim; d++) out.shape[d] = new_shape[d];
    out.numel = numel_of(new_shape, new_ndim);
    compute_contiguous_strides(out.shape, new_ndim, out.strides);
    out.data = src->data;
    out.owns_data = FALSE;
    return out;
}

Tensor tensor_transpose(const Tensor* src, u32 dim0, u32 dim1) {
    Tensor out = *src;
    out.owns_data = FALSE;

    u64 tmp_shape = out.shape[dim0];
    out.shape[dim0] = out.shape[dim1];
    out.shape[dim1] = tmp_shape;

    i64 tmp_stride = out.strides[dim0];
    out.strides[dim0] = out.strides[dim1];
    out.strides[dim1] = tmp_stride;

    return out;
}

Tensor tensor_contiguous(Allocator* alloc, const Tensor* src) {
    Tensor out = tensor_create(alloc, src->shape, src->ndim);
    tensor_copy_from(&out, src);
    return out;
}

static Tensor elementwise_binop(Allocator* alloc, const Tensor* a, const Tensor* b, i32 op) {
    /* op: 0=add, 1=sub, 2=mul. V1: shape'ler tam olarak esit olmalidir
     * (broadcasting yok -- bkz. tensor.h basligindaki not). */
    Tensor out = tensor_create(alloc, a->shape, a->ndim);
    u64 idx[TENSOR_MAX_DIMS];

    for (u64 flat = 0; flat < out.numel; flat++) {
        flat_to_multi(flat, out.shape, out.ndim, idx);
        f32 av = a->data[compute_offset(a->strides, idx, a->ndim)];
        f32 bv = b->data[compute_offset(b->strides, idx, b->ndim)];
        f32 r;
        if (op == 0) r = av + bv;
        else if (op == 1) r = av - bv;
        else r = av * bv;
        out.data[compute_offset(out.strides, idx, out.ndim)] = r;
    }
    return out;
}

Tensor tensor_add(Allocator* alloc, const Tensor* a, const Tensor* b) { return elementwise_binop(alloc, a, b, 0); }
Tensor tensor_sub(Allocator* alloc, const Tensor* a, const Tensor* b) { return elementwise_binop(alloc, a, b, 1); }
Tensor tensor_mul(Allocator* alloc, const Tensor* a, const Tensor* b) { return elementwise_binop(alloc, a, b, 2); }

Tensor tensor_scale(Allocator* alloc, const Tensor* a, f32 scalar) {
    Tensor out = tensor_create(alloc, a->shape, a->ndim);
    u64 idx[TENSOR_MAX_DIMS];
    for (u64 flat = 0; flat < out.numel; flat++) {
        flat_to_multi(flat, out.shape, out.ndim, idx);
        f32 av = a->data[compute_offset(a->strides, idx, a->ndim)];
        out.data[compute_offset(out.strides, idx, out.ndim)] = av * scalar;
    }
    return out;
}

Tensor tensor_relu(Allocator* alloc, const Tensor* a) {
    Tensor out = tensor_create(alloc, a->shape, a->ndim);
    u64 idx[TENSOR_MAX_DIMS];
    for (u64 flat = 0; flat < out.numel; flat++) {
        flat_to_multi(flat, out.shape, out.ndim, idx);
        f32 v = a->data[compute_offset(a->strides, idx, a->ndim)];
        out.data[compute_offset(out.strides, idx, out.ndim)] = (v > 0.0f) ? v : 0.0f;
    }
    return out;
}

Tensor tensor_sigmoid(Allocator* alloc, const Tensor* a) {
    Tensor out = tensor_create(alloc, a->shape, a->ndim);
    u64 idx[TENSOR_MAX_DIMS];
    for (u64 flat = 0; flat < out.numel; flat++) {
        flat_to_multi(flat, out.shape, out.ndim, idx);
        f32 v = a->data[compute_offset(a->strides, idx, a->ndim)];
        f32 y = 1.0f / (1.0f + m_expf(-v));
        out.data[compute_offset(out.strides, idx, out.ndim)] = y;
    }
    return out;
}

Tensor tensor_tanh_op(Allocator* alloc, const Tensor* a) {
    Tensor out = tensor_create(alloc, a->shape, a->ndim);
    u64 idx[TENSOR_MAX_DIMS];
    for (u64 flat = 0; flat < out.numel; flat++) {
        flat_to_multi(flat, out.shape, out.ndim, idx);
        f32 v = a->data[compute_offset(a->strides, idx, a->ndim)];
        out.data[compute_offset(out.strides, idx, out.ndim)] = m_tanhf(v);
    }
    return out;
}

Tensor tensor_softmax_lastdim(Allocator* alloc, const Tensor* a) {
    /* Sozlesme: a bitisiktir (contiguous) -- cagiran taraf gerekirse
     * once tensor_contiguous ile somutlastirmalidir. */
    Tensor out = tensor_create(alloc, a->shape, a->ndim);
    u64 last = a->shape[a->ndim - 1];
    u64 outer = a->numel / last;

    for (u64 o = 0; o < outer; o++) {
        const f32* row_in = a->data + o * last;
        f32* row_out = out.data + o * last;

        f32 max_v = row_in[0];
        for (u64 i = 1; i < last; i++) if (row_in[i] > max_v) max_v = row_in[i];

        f32 sum = 0.0f;
        for (u64 i = 0; i < last; i++) {
            f32 e = m_expf(row_in[i] - max_v);
            row_out[i] = e;
            sum += e;
        }
        for (u64 i = 0; i < last; i++) row_out[i] /= sum;
    }
    return out;
}

Tensor tensor_rmsnorm(Allocator* alloc, const Tensor* a, const Tensor* weight, f32 eps) {
    Tensor out = tensor_create(alloc, a->shape, a->ndim);
    u64 last = a->shape[a->ndim - 1];
    u64 outer = a->numel / last;

    for (u64 o = 0; o < outer; o++) {
        const f32* row = a->data + o * last;
        f32* orow = out.data + o * last;

        f32 ss = 0.0f;
        for (u64 i = 0; i < last; i++) ss += row[i] * row[i];
        f32 ms = ss / (f32)last;
        f32 r = 1.0f / m_sqrtf(ms + eps);

        for (u64 i = 0; i < last; i++) orow[i] = row[i] * r * weight->data[i];
    }
    return out;
}

Tensor tensor_sum_all(Allocator* alloc, const Tensor* a) {
    u64 shape1[1] = { 1 };
    Tensor out = tensor_create(alloc, shape1, 1);
    u64 idx[TENSOR_MAX_DIMS];
    f32 sum = 0.0f;
    for (u64 flat = 0; flat < a->numel; flat++) {
        flat_to_multi(flat, a->shape, a->ndim, idx);
        sum += a->data[compute_offset(a->strides, idx, a->ndim)];
    }
    out.data[0] = sum;
    return out;
}

Tensor tensor_slice_cols(const Tensor* src, u64 col_start, u64 num_cols) {
    Tensor out = *src;
    out.owns_data = FALSE;
    out.shape[1] = num_cols;
    out.numel = out.shape[0] * num_cols;
    out.data = src->data + (i64)col_start * src->strides[1];
    return out;
}

Tensor tensor_add_bias_2d(Allocator* alloc, const Tensor* x, const Tensor* bias) {
    Tensor out = tensor_create(alloc, x->shape, x->ndim);
    for (u64 i = 0; i < x->shape[0]; i++) {
        for (u64 j = 0; j < x->shape[1]; j++) {
            u64 xi[2] = { i, j };
            u64 bi[1] = { j };
            f32 v = x->data[compute_offset(x->strides, xi, 2)] +
                    bias->data[compute_offset(bias->strides, bi, 1)];
            out.data[compute_offset(out.strides, xi, 2)] = v;
        }
    }
    return out;
}

Tensor tensor_concat_cols2(Allocator* alloc, const Tensor* a, const Tensor* b) {
    u64 rows = a->shape[0];
    u64 ca = a->shape[1];
    u64 cb = b->shape[1];
    u64 shape[2] = { rows, ca + cb };
    Tensor out = tensor_create(alloc, shape, 2);

    for (u64 i = 0; i < rows; i++) {
        for (u64 j = 0; j < ca; j++) {
            u64 ai[2] = { i, j }, oi[2] = { i, j };
            out.data[compute_offset(out.strides, oi, 2)] = a->data[compute_offset(a->strides, ai, 2)];
        }
        for (u64 j = 0; j < cb; j++) {
            u64 bi[2] = { i, j }, oi[2] = { i, ca + j };
            out.data[compute_offset(out.strides, oi, 2)] = b->data[compute_offset(b->strides, bi, 2)];
        }
    }
    return out;
}

Tensor tensor_slice_rows(const Tensor* src, u64 row_start, u64 num_rows) {
    Tensor out = *src;
    out.owns_data = FALSE;
    out.shape[0] = num_rows;
    out.numel = num_rows * src->shape[1];
    out.data = src->data + (i64)row_start * src->strides[0];
    return out;
}

Tensor tensor_concat_rows2(Allocator* alloc, const Tensor* a, const Tensor* b) {
    u64 cols = a->shape[1];
    u64 ra = a->shape[0];
    u64 rb = b->shape[0];
    u64 shape[2] = { ra + rb, cols };
    Tensor out = tensor_create(alloc, shape, 2);

    for (u64 i = 0; i < ra; i++) {
        for (u64 j = 0; j < cols; j++) {
            u64 ai[2] = { i, j }, oi[2] = { i, j };
            out.data[compute_offset(out.strides, oi, 2)] = a->data[compute_offset(a->strides, ai, 2)];
        }
    }
    for (u64 i = 0; i < rb; i++) {
        for (u64 j = 0; j < cols; j++) {
            u64 bi[2] = { i, j }, oi[2] = { ra + i, j };
            out.data[compute_offset(out.strides, oi, 2)] = b->data[compute_offset(b->strides, bi, 2)];
        }
    }
    return out;
}

static bool32 same_shape(const Tensor* a, const Tensor* b) {
    if (a->ndim != b->ndim) return FALSE;
    for (u32 d = 0; d < a->ndim; d++) if (a->shape[d] != b->shape[d]) return FALSE;
    return TRUE;
}

void tensor_add_inplace(Tensor* dst, const Tensor* src) {
    /* Hizli yol: ikisi de bitisik ve ayni sekilde ise eleman basina
     * cok-boyutlu indeks hesabina (bolme/mod) gerek yok. Ayni toplama
     * islemleri ayni sirayla yapilir -- sonuc genel yolla BIT BIT ayni.
     * (Linux tasimasinda gradyan indirgemesi/geri yayilimda olculen
     * darbogaz, bkz. PROJE_PLANI.md Bolum 20.) */
    if (same_shape(dst, src) && tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
        f32* d = dst->data;
        const f32* s = src->data;
        for (u64 k = 0; k < dst->numel; k++) d[k] += s[k];
        return;
    }

    u64 idx[TENSOR_MAX_DIMS];
    for (u64 flat = 0; flat < dst->numel; flat++) {
        flat_to_multi(flat, dst->shape, dst->ndim, idx);
        u64 doff = compute_offset(dst->strides, idx, dst->ndim);
        u64 soff = compute_offset(src->strides, idx, src->ndim);
        dst->data[doff] += src->data[soff];
    }
}

Tensor tensor_matmul2d(Allocator* alloc, const Tensor* a, const Tensor* b) {
    u64 M = a->shape[0];
    u64 K = a->shape[1];
    u64 N = b->shape[1];

    u64 out_shape[2] = { M, N };
    Tensor out = tensor_create(alloc, out_shape, 2);

    for (u64 i = 0; i < M; i++) {
        for (u64 j = 0; j < N; j++) {
            f32 sum = 0.0f;
            for (u64 k = 0; k < K; k++) {
                u64 a_idx[2] = { i, k };
                u64 b_idx[2] = { k, j };
                sum += a->data[compute_offset(a->strides, a_idx, 2)] *
                       b->data[compute_offset(b->strides, b_idx, 2)];
            }
            u64 o_idx[2] = { i, j };
            out.data[compute_offset(out.strides, o_idx, 2)] = sum;
        }
    }
    return out;
}
