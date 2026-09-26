/* Katman 7 dogrulama: yeni fused otograd islemleri (add_bias, slice_cols,
 * rmsnorm, RoPE) icin sayisal gradyan kontrolu. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/model_ops.h"
#include "../model/rope.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static f32 fabsf_(f32 x) { return (x < 0) ? -x : x; }

static void check_grad(const char* name, f32 numeric, f32 analytic) {
    f32 tol = 0.02f + 0.05f * fabsf_(numeric);
    if (fabsf_(numeric - analytic) <= tol) { g_pass++; }
    else {
        g_fail++;
        console_write("  [FAIL] "); console_write(name);
        console_write(" numeric*1e4="); console_write_u64((u64)(i64)(numeric * 10000.0f));
        console_write(" analytic*1e4="); console_write_u64((u64)(i64)(analytic * 10000.0f));
        console_write_line("");
    }
}

/* ================= add_bias ================= */

static f32 g_bias_data[3];

static f32 loss_add_bias(const f32* xdata, u64 rows, u64 cols) {
    Allocator alloc = allocator_create(1024 * 1024);
    u64 shape[2] = { rows, cols };
    Tensor x = tensor_create(&alloc, shape, 2);
    u64 bshape[1] = { cols };
    Tensor b = tensor_create(&alloc, bshape, 1);
    for (u64 i = 0; i < rows * cols; i++) x.data[i] = xdata[i];
    for (u64 i = 0; i < cols; i++) b.data[i] = g_bias_data[i];

    Node* nx = node_leaf(&alloc, x, TRUE);
    Node* nb = node_leaf(&alloc, b, TRUE);
    Node* y = node_add_bias(&alloc, nx, nb);
    Node* loss = node_sum_all(&alloc, y);
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_add_bias_grad(void) {
    u64 rows = 3, cols = 3;
    f32 xdata[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    g_bias_data[0] = 0.5f; g_bias_data[1] = -0.3f; g_bias_data[2] = 1.2f;

    Allocator alloc = allocator_create(1024 * 1024);
    u64 shape[2] = { rows, cols };
    Tensor x = tensor_create(&alloc, shape, 2);
    u64 bshape[1] = { cols };
    Tensor b = tensor_create(&alloc, bshape, 1);
    for (u64 i = 0; i < rows * cols; i++) x.data[i] = xdata[i];
    for (u64 i = 0; i < cols; i++) b.data[i] = g_bias_data[i];

    Node* nx = node_leaf(&alloc, x, TRUE);
    Node* nb = node_leaf(&alloc, b, TRUE);
    Node* y = node_add_bias(&alloc, nx, nb);
    Node* loss = node_sum_all(&alloc, y);
    backward(&alloc, loss);

    f32 eps = 1e-2f;
    for (u64 i = 0; i < rows * cols; i++) {
        f32 orig = xdata[i];
        xdata[i] = orig + eps; f32 lp = loss_add_bias(xdata, rows, cols);
        xdata[i] = orig - eps; f32 lm = loss_add_bias(xdata, rows, cols);
        xdata[i] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        check_grad("add_bias.x", num, nx->grad.data[i]);
    }
    for (u64 j = 0; j < cols; j++) {
        f32 orig = g_bias_data[j];
        g_bias_data[j] = orig + eps; f32 lp = loss_add_bias(xdata, rows, cols);
        g_bias_data[j] = orig - eps; f32 lm = loss_add_bias(xdata, rows, cols);
        g_bias_data[j] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        check_grad("add_bias.b", num, nb->grad.data[j]);
    }
    allocator_destroy(&alloc);
}

/* ================= slice_cols ================= */

static void test_slice_cols_grad(void) {
    Allocator alloc = allocator_create(1024 * 1024);
    u64 shape[2] = { 2, 4 };
    Tensor x = tensor_create(&alloc, shape, 2);
    for (u64 i = 0; i < 8; i++) x.data[i] = (f32)i;

    Node* nx = node_leaf(&alloc, x, TRUE);
    Node* s = node_slice_cols(&alloc, nx, 1, 2); /* kolonlar 1,2 */
    Node* loss = node_sum_all(&alloc, s);
    backward(&alloc, loss);

    /* d(sum of cols[1:3])/dx should be 1 for cols 1,2 and 0 for cols 0,3 */
    bool32 ok = TRUE;
    for (u64 r = 0; r < 2; r++) {
        for (u64 c = 0; c < 4; c++) {
            f32 expected = (c == 1 || c == 2) ? 1.0f : 0.0f;
            f32 got = nx->grad.data[r * 4 + c];
            if (fabsf_(got - expected) > 1e-5f) ok = FALSE;
        }
    }
    if (ok) g_pass++; else { g_fail++; console_write_line("  [FAIL] slice_cols.x gradyan maskesi yanlis"); }

    allocator_destroy(&alloc);
}

/* ================= rmsnorm ================= */

static f32 g_rms_w[4];

static f32 loss_rmsnorm(const f32* xdata, u64 n) {
    Allocator alloc = allocator_create(1024 * 1024);
    u64 shape[1] = { n };
    Tensor x = tensor_create(&alloc, shape, 1);
    Tensor w = tensor_create(&alloc, shape, 1);
    for (u64 i = 0; i < n; i++) { x.data[i] = xdata[i]; w.data[i] = g_rms_w[i]; }

    Node* nx = node_leaf(&alloc, x, TRUE);
    Node* nw = node_leaf(&alloc, w, TRUE);
    Node* y = node_rmsnorm(&alloc, nx, nw, 1e-5f);
    Node* loss = node_sum_all(&alloc, y);
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_rmsnorm_grad(void) {
    u64 n = 4;
    f32 xdata[4] = { 1.0f, -2.0f, 0.5f, 3.0f };
    g_rms_w[0] = 1.0f; g_rms_w[1] = 0.8f; g_rms_w[2] = -1.2f; g_rms_w[3] = 0.3f;

    Allocator alloc = allocator_create(1024 * 1024);
    u64 shape[1] = { n };
    Tensor x = tensor_create(&alloc, shape, 1);
    Tensor w = tensor_create(&alloc, shape, 1);
    for (u64 i = 0; i < n; i++) { x.data[i] = xdata[i]; w.data[i] = g_rms_w[i]; }

    Node* nx = node_leaf(&alloc, x, TRUE);
    Node* nw = node_leaf(&alloc, w, TRUE);
    Node* y = node_rmsnorm(&alloc, nx, nw, 1e-5f);
    Node* loss = node_sum_all(&alloc, y);
    backward(&alloc, loss);

    f32 eps = 1e-2f;
    for (u64 i = 0; i < n; i++) {
        f32 orig = xdata[i];
        xdata[i] = orig + eps; f32 lp = loss_rmsnorm(xdata, n);
        xdata[i] = orig - eps; f32 lm = loss_rmsnorm(xdata, n);
        xdata[i] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        check_grad("rmsnorm.x", num, nx->grad.data[i]);
    }
    for (u64 j = 0; j < n; j++) {
        f32 orig = g_rms_w[j];
        g_rms_w[j] = orig + eps; f32 lp = loss_rmsnorm(xdata, n);
        g_rms_w[j] = orig - eps; f32 lm = loss_rmsnorm(xdata, n);
        g_rms_w[j] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        check_grad("rmsnorm.w", num, nw->grad.data[j]);
    }
    allocator_destroy(&alloc);
}

/* ================= RoPE ================= */

static f32 g_rope_w[8]; /* seq=2, head_dim=4 */

static f32 loss_rope(const f32* xdata, const Tensor* cos_t, const Tensor* sin_t) {
    Allocator alloc = allocator_create(1024 * 1024);
    u64 shape[2] = { 2, 4 };
    Tensor x = tensor_create(&alloc, shape, 2);
    Tensor w = tensor_create(&alloc, shape, 2);
    for (u64 i = 0; i < 8; i++) { x.data[i] = xdata[i]; w.data[i] = g_rope_w[i]; }

    Node* nx = node_leaf(&alloc, x, TRUE);
    Node* nw = node_leaf(&alloc, w, TRUE);
    Node* ncos = node_leaf(&alloc, *cos_t, FALSE);
    Node* nsin = node_leaf(&alloc, *sin_t, FALSE);
    Node* r = node_rope(&alloc, nx, ncos, nsin);
    Node* prod = node_mul(&alloc, r, nw);
    Node* loss = node_sum_all(&alloc, prod);
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_rope_grad(void) {
    Allocator alloc = allocator_create(1024 * 1024);
    u64 cshape[2] = { 2, 2 }; /* seq=2, head_dim/2=2 */
    Tensor cos_t = tensor_create(&alloc, cshape, 2);
    Tensor sin_t = tensor_create(&alloc, cshape, 2);
    rope_build_tables(&cos_t, &sin_t, 2, 4, 10000.0f);

    f32 xdata[8] = { 1.0f, 2.0f, -1.0f, 0.5f, 0.3f, -0.7f, 1.5f, 2.2f };
    for (u64 i = 0; i < 8; i++) g_rope_w[i] = (f32)(i + 1) * 0.1f - 0.3f;

    u64 shape[2] = { 2, 4 };
    Tensor x = tensor_create(&alloc, shape, 2);
    Tensor w = tensor_create(&alloc, shape, 2);
    for (u64 i = 0; i < 8; i++) { x.data[i] = xdata[i]; w.data[i] = g_rope_w[i]; }

    Node* nx = node_leaf(&alloc, x, TRUE);
    Node* nw = node_leaf(&alloc, w, TRUE);
    Node* ncos = node_leaf(&alloc, cos_t, FALSE);
    Node* nsin = node_leaf(&alloc, sin_t, FALSE);
    Node* r = node_rope(&alloc, nx, ncos, nsin);
    Node* prod = node_mul(&alloc, r, nw);
    Node* loss = node_sum_all(&alloc, prod);
    backward(&alloc, loss);

    f32 eps = 1e-2f;
    for (u64 i = 0; i < 8; i++) {
        f32 orig = xdata[i];
        xdata[i] = orig + eps; f32 lp = loss_rope(xdata, &cos_t, &sin_t);
        xdata[i] = orig - eps; f32 lm = loss_rope(xdata, &cos_t, &sin_t);
        xdata[i] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        check_grad("rope.x", num, nx->grad.data[i]);
    }

    /* Ozellik kontrolu: RoPE, her (x0,x1) ciftinin normunu korumali
     * (bir dondurme oldugu icin) -- x^2+y^2 degismemeli. */
    bool32 norm_ok = TRUE;
    for (u64 p = 0; p < 2; p++) {
        for (u64 k = 0; k < 2; k++) {
            f32 x0 = xdata[p * 4 + 2 * k], x1 = xdata[p * 4 + 2 * k + 1];
            f32 y0 = r->value.data[p * 4 + 2 * k], y1 = r->value.data[p * 4 + 2 * k + 1];
            f32 n0 = x0 * x0 + x1 * x1;
            f32 n1 = y0 * y0 + y1 * y1;
            if (fabsf_(n0 - n1) > 1e-3f) norm_ok = FALSE;
        }
    }
    if (norm_ok) g_pass++; else { g_fail++; console_write_line("  [FAIL] rope: dondurme normu korumadi"); }

    allocator_destroy(&alloc);
}

/* ================= slice_rows / concat_rows2 (Katman 18) ================= */

static void test_slice_rows_grad(void) {
    Allocator alloc = allocator_create(1024 * 1024);
    u64 shape[2] = { 4, 3 };
    Tensor x = tensor_create(&alloc, shape, 2);
    for (u64 i = 0; i < 12; i++) x.data[i] = (f32)i;

    Node* nx = node_leaf(&alloc, x, TRUE);
    Node* s = node_slice_rows(&alloc, nx, 1, 2); /* satirlar 1,2 */
    Node* loss = node_sum_all(&alloc, s);
    backward(&alloc, loss);

    bool32 ok = TRUE;
    for (u64 r = 0; r < 4; r++) {
        for (u64 c = 0; c < 3; c++) {
            f32 expected = (r == 1 || r == 2) ? 1.0f : 0.0f;
            f32 got = nx->grad.data[r * 3 + c];
            if (fabsf_(got - expected) > 1e-5f) ok = FALSE;
        }
    }
    if (ok) g_pass++; else { g_fail++; console_write_line("  [FAIL] slice_rows.x gradyan maskesi yanlis"); }

    allocator_destroy(&alloc);
}

static f32 g_lossw_rows[12];

static f32 loss_concat_rows2(const f32* adata, const f32* bdata) {
    Allocator alloc = allocator_create(1024 * 1024);
    u64 ash[2] = { 2, 3 }, bsh[2] = { 3, 3 };
    Tensor a = tensor_create(&alloc, ash, 2);
    Tensor b = tensor_create(&alloc, bsh, 2);
    for (u64 i = 0; i < 6; i++) a.data[i] = adata[i];
    for (u64 i = 0; i < 9; i++) b.data[i] = bdata[i];

    Node* na = node_leaf(&alloc, a, TRUE);
    Node* nb = node_leaf(&alloc, b, TRUE);
    Node* cat = node_concat_rows2(&alloc, na, nb);
    u64 wsh[2] = { 5, 3 };
    Tensor w = tensor_create(&alloc, wsh, 2);
    for (u64 i = 0; i < 15; i++) w.data[i] = g_lossw_rows[i % 12];
    Node* nw = node_leaf(&alloc, w, TRUE);
    Node* prod = node_mul(&alloc, cat, nw);
    Node* loss = node_sum_all(&alloc, prod);
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_concat_rows2_grad(void) {
    f32 adata[6] = { 1, 2, 3, 4, 5, 6 };
    f32 bdata[9] = { 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    for (u64 i = 0; i < 12; i++) g_lossw_rows[i] = (f32)(i + 1) * 0.1f - 0.5f;

    Allocator alloc = allocator_create(1024 * 1024);
    u64 ash[2] = { 2, 3 }, bsh[2] = { 3, 3 };
    Tensor a = tensor_create(&alloc, ash, 2);
    Tensor b = tensor_create(&alloc, bsh, 2);
    for (u64 i = 0; i < 6; i++) a.data[i] = adata[i];
    for (u64 i = 0; i < 9; i++) b.data[i] = bdata[i];

    Node* na = node_leaf(&alloc, a, TRUE);
    Node* nb = node_leaf(&alloc, b, TRUE);
    Node* cat = node_concat_rows2(&alloc, na, nb);
    u64 wsh[2] = { 5, 3 };
    Tensor w = tensor_create(&alloc, wsh, 2);
    for (u64 i = 0; i < 15; i++) w.data[i] = g_lossw_rows[i % 12];
    Node* nw = node_leaf(&alloc, w, TRUE);
    Node* prod = node_mul(&alloc, cat, nw);
    Node* loss = node_sum_all(&alloc, prod);
    backward(&alloc, loss);

    f32 eps = 1e-2f;
    for (u64 i = 0; i < 6; i++) {
        f32 orig = adata[i];
        adata[i] = orig + eps; f32 lp = loss_concat_rows2(adata, bdata);
        adata[i] = orig - eps; f32 lm = loss_concat_rows2(adata, bdata);
        adata[i] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        check_grad("concat_rows2.a", num, na->grad.data[i]);
    }
    for (u64 i = 0; i < 9; i++) {
        f32 orig = bdata[i];
        bdata[i] = orig + eps; f32 lp = loss_concat_rows2(adata, bdata);
        bdata[i] = orig - eps; f32 lm = loss_concat_rows2(adata, bdata);
        bdata[i] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        check_grad("concat_rows2.b", num, nb->grad.data[i]);
    }
    allocator_destroy(&alloc);
}

int main(void) {
    console_write_line("=== Katman 7 (Model Op'lari) - Sayisal Gradyan Kontrolu ===");

    test_add_bias_grad();
    test_slice_cols_grad();
    test_rmsnorm_grad();
    test_rope_grad();
    test_slice_rows_grad();
    test_concat_rows2_grad();

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
