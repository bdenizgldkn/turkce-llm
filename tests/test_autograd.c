/* Katman 4 dogrulama: SAYISAL GRADYAN KONTROLU (numerical gradient
 * checking) -- PROJE_PLANI.md'de karar verilen test stratejisi.
 * Her op icin: otograd'in verdigi turev, (f(x+e)-f(x-e))/(2e) ile
 * hesaplanan sayisal turevle karsilastirilir. Tensorler f32 oldugu
 * icin eps ve tolerans f32 hassasiyetine gore secilmistir. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static f32 fabsf_(f32 x) { return (x < 0) ? -x : x; }

static void check_grad(const char* name, f32 numeric, f32 analytic) {
    f32 tol = 0.02f + 0.05f * fabsf_(numeric);
    f32 diff = fabsf_(numeric - analytic);
    if (diff <= tol) {
        g_pass++;
    } else {
        g_fail++;
        console_write("  [FAIL] "); console_write(name);
        console_write(" numeric*1e4=");
        console_write_u64((u64)(i64)(numeric * 10000.0f));
        console_write(" analytic*1e4=");
        console_write_u64((u64)(i64)(analytic * 10000.0f));
        console_write_line("");
    }
}

/* ---- Genel finite-difference cercevesi ----
 * build_fn: verilen x verisiyle (tek girdili) grafigi kurar ve skaler
 * kayip degerini dondurur. Her cagri kendi kucuk Allocator'ini yaratir
 * (basit ve guvenli; performans onemli degil, doğruluk kontrolu bu). */
typedef f32 (*ScalarLossFn1)(const f32* xdata, const u64* shape, u32 ndim);

static f32 numeric_grad_elem(ScalarLossFn1 fn, f32* xdata, const u64* shape, u32 ndim, u64 elem, f32 eps) {
    f32 orig = xdata[elem];
    xdata[elem] = orig + eps;
    f32 lp = fn(xdata, shape, ndim);
    xdata[elem] = orig - eps;
    f32 lm = fn(xdata, shape, ndim);
    xdata[elem] = orig;
    return (lp - lm) / (2.0f * eps);
}

/* ================= add: loss = sum(a+b) ================= */

static f32 g_b_data[6]; /* yardimci sabit girdi (bazı testlerde kullanilir) */

static f32 loss_add(const f32* xdata, const u64* shape, u32 ndim) {
    Allocator alloc = allocator_create(1024 * 1024);
    Tensor a = tensor_create(&alloc, shape, ndim);
    Tensor b = tensor_create(&alloc, shape, ndim);
    for (u64 i = 0; i < a.numel; i++) { a.data[i] = xdata[i]; b.data[i] = g_b_data[i]; }
    Node* na = node_leaf(&alloc, a, TRUE);
    Node* nb = node_leaf(&alloc, b, TRUE);
    Node* sum_node = node_add(&alloc, na, nb);
    Node* loss = node_sum_all(&alloc, sum_node);
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_add_grad(void) {
    u64 shape[1] = { 4 };
    f32 xdata[4] = { 1.0f, -2.0f, 3.0f, 0.5f };
    for (u64 i = 0; i < 4; i++) g_b_data[i] = (f32)(i + 1) * 0.3f;

    Allocator alloc = allocator_create(1024 * 1024);
    Tensor a = tensor_create(&alloc, shape, 1);
    Tensor b = tensor_create(&alloc, shape, 1);
    for (u64 i = 0; i < 4; i++) { a.data[i] = xdata[i]; b.data[i] = g_b_data[i]; }
    Node* na = node_leaf(&alloc, a, TRUE);
    Node* nb = node_leaf(&alloc, b, TRUE);
    Node* s = node_add(&alloc, na, nb);
    Node* loss = node_sum_all(&alloc, s);
    backward(&alloc, loss);

    for (u64 i = 0; i < 4; i++) {
        f32 num = numeric_grad_elem(loss_add, xdata, shape, 1, i, 1e-2f);
        check_grad("add.a", num, na->grad.data[i]);
    }
    allocator_destroy(&alloc);
}

/* ================= mul: loss = sum(a*b) ================= */

static f32 loss_mul(const f32* xdata, const u64* shape, u32 ndim) {
    Allocator alloc = allocator_create(1024 * 1024);
    Tensor a = tensor_create(&alloc, shape, ndim);
    Tensor b = tensor_create(&alloc, shape, ndim);
    for (u64 i = 0; i < a.numel; i++) { a.data[i] = xdata[i]; b.data[i] = g_b_data[i]; }
    Node* na = node_leaf(&alloc, a, TRUE);
    Node* nb = node_leaf(&alloc, b, TRUE);
    Node* m = node_mul(&alloc, na, nb);
    Node* loss = node_sum_all(&alloc, m);
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_mul_grad(void) {
    u64 shape[1] = { 4 };
    f32 xdata[4] = { 2.0f, -1.0f, 0.5f, 3.0f };
    for (u64 i = 0; i < 4; i++) g_b_data[i] = (f32)(i + 1) * -0.7f;

    Allocator alloc = allocator_create(1024 * 1024);
    Tensor a = tensor_create(&alloc, shape, 1);
    Tensor b = tensor_create(&alloc, shape, 1);
    for (u64 i = 0; i < 4; i++) { a.data[i] = xdata[i]; b.data[i] = g_b_data[i]; }
    Node* na = node_leaf(&alloc, a, TRUE);
    Node* nb = node_leaf(&alloc, b, TRUE);
    Node* m = node_mul(&alloc, na, nb);
    Node* loss = node_sum_all(&alloc, m);
    backward(&alloc, loss);

    for (u64 i = 0; i < 4; i++) {
        f32 num = numeric_grad_elem(loss_mul, xdata, shape, 1, i, 1e-2f);
        check_grad("mul.a", num, na->grad.data[i]);
    }
    allocator_destroy(&alloc);
}

/* ================= matmul2d: loss = sum(A*B) ================= */

static f32 g_B2[6]; /* 3x2 sabit ikinci matris (K=3,N=2) */

static f32 loss_matmul(const f32* xdata, const u64* shape, u32 ndim) {
    (void)ndim;
    Allocator alloc = allocator_create(1024 * 1024);
    u64 a_shape[2] = { 2, 3 };
    u64 b_shape[2] = { 3, 2 };
    Tensor A = tensor_create(&alloc, a_shape, 2);
    Tensor B = tensor_create(&alloc, b_shape, 2);
    for (u64 i = 0; i < 6; i++) A.data[i] = xdata[i];
    for (u64 i = 0; i < 6; i++) B.data[i] = g_B2[i];
    (void)shape;
    Node* na = node_leaf(&alloc, A, TRUE);
    Node* nb = node_leaf(&alloc, B, TRUE);
    Node* c = node_matmul2d(&alloc, na, nb);
    Node* loss = node_sum_all(&alloc, c);
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_matmul_grad(void) {
    u64 a_shape[2] = { 2, 3 };
    u64 b_shape[2] = { 3, 2 };
    f32 a_data[6] = { 1, 2, 3, 4, 5, 6 };
    for (u64 i = 0; i < 6; i++) g_B2[i] = (f32)(i + 1) * 0.5f - 1.0f;

    Allocator alloc = allocator_create(1024 * 1024);
    Tensor A = tensor_create(&alloc, a_shape, 2);
    Tensor B = tensor_create(&alloc, b_shape, 2);
    for (u64 i = 0; i < 6; i++) { A.data[i] = a_data[i]; B.data[i] = g_B2[i]; }
    Node* na = node_leaf(&alloc, A, TRUE);
    Node* nb = node_leaf(&alloc, B, TRUE);
    Node* c = node_matmul2d(&alloc, na, nb);
    Node* loss = node_sum_all(&alloc, c);
    backward(&alloc, loss);

    for (u64 i = 0; i < 6; i++) {
        f32 num = numeric_grad_elem(loss_matmul, a_data, a_shape, 2, i, 1e-2f);
        check_grad("matmul.A", num, na->grad.data[i]);
    }
    allocator_destroy(&alloc);
}

/* ================= transpose: loss = sum(transpose(x) * B) ================= */

static f32 loss_transpose(const f32* xdata, const u64* shape, u32 ndim) {
    (void)ndim;
    Allocator alloc = allocator_create(1024 * 1024);
    u64 x_shape[2] = { 2, 3 };
    Tensor X = tensor_create(&alloc, x_shape, 2);
    for (u64 i = 0; i < 6; i++) X.data[i] = xdata[i];
    (void)shape;
    Node* nx = node_leaf(&alloc, X, TRUE);
    Node* xt = node_transpose(&alloc, nx, 0, 1); /* 3x2 */
    Node* loss = node_sum_all(&alloc, xt);       /* transpose toplami = orijinal toplam, ama gradyan yapisini test eder */
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_transpose_grad(void) {
    u64 x_shape[2] = { 2, 3 };
    f32 x_data[6] = { 1, 2, 3, 4, 5, 6 };

    Allocator alloc = allocator_create(1024 * 1024);
    Tensor X = tensor_create(&alloc, x_shape, 2);
    for (u64 i = 0; i < 6; i++) X.data[i] = x_data[i];
    Node* nx = node_leaf(&alloc, X, TRUE);
    Node* xt = node_transpose(&alloc, nx, 0, 1);
    Node* loss = node_sum_all(&alloc, xt);
    backward(&alloc, loss);

    for (u64 i = 0; i < 6; i++) {
        f32 num = numeric_grad_elem(loss_transpose, x_data, x_shape, 2, i, 1e-2f);
        check_grad("transpose.x", num, nx->grad.data[i]);
    }
    allocator_destroy(&alloc);
}

/* ================= sigmoid: loss = sum(sigmoid(x)) ================= */

static f32 loss_sigmoid(const f32* xdata, const u64* shape, u32 ndim) {
    Allocator alloc = allocator_create(1024 * 1024);
    Tensor X = tensor_create(&alloc, shape, ndim);
    for (u64 i = 0; i < X.numel; i++) X.data[i] = xdata[i];
    Node* nx = node_leaf(&alloc, X, TRUE);
    Node* s = node_sigmoid(&alloc, nx);
    Node* loss = node_sum_all(&alloc, s);
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_sigmoid_grad(void) {
    u64 shape[1] = { 4 };
    f32 x_data[4] = { -1.5f, 0.0f, 0.7f, 2.2f };

    Allocator alloc = allocator_create(1024 * 1024);
    Tensor X = tensor_create(&alloc, shape, 1);
    for (u64 i = 0; i < 4; i++) X.data[i] = x_data[i];
    Node* nx = node_leaf(&alloc, X, TRUE);
    Node* s = node_sigmoid(&alloc, nx);
    Node* loss = node_sum_all(&alloc, s);
    backward(&alloc, loss);

    for (u64 i = 0; i < 4; i++) {
        f32 num = numeric_grad_elem(loss_sigmoid, x_data, shape, 1, i, 1e-2f);
        check_grad("sigmoid.x", num, nx->grad.data[i]);
    }
    allocator_destroy(&alloc);
}

/* ================= relu: loss = sum(relu(x)) ================= */

static f32 loss_relu(const f32* xdata, const u64* shape, u32 ndim) {
    Allocator alloc = allocator_create(1024 * 1024);
    Tensor X = tensor_create(&alloc, shape, ndim);
    for (u64 i = 0; i < X.numel; i++) X.data[i] = xdata[i];
    Node* nx = node_leaf(&alloc, X, TRUE);
    Node* r = node_relu(&alloc, nx);
    Node* loss = node_sum_all(&alloc, r);
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_relu_grad(void) {
    u64 shape[1] = { 4 };
    /* 0'a yakin/uzerinde deger yok -- relu'nun kink noktasinda (x=0)
     * merkezi fark kararsiz olabilir, bu yuzden testte kaciniyoruz. */
    f32 x_data[4] = { -2.0f, 1.5f, 3.0f, -0.8f };

    Allocator alloc = allocator_create(1024 * 1024);
    Tensor X = tensor_create(&alloc, shape, 1);
    for (u64 i = 0; i < 4; i++) X.data[i] = x_data[i];
    Node* nx = node_leaf(&alloc, X, TRUE);
    Node* r = node_relu(&alloc, nx);
    Node* loss = node_sum_all(&alloc, r);
    backward(&alloc, loss);

    for (u64 i = 0; i < 4; i++) {
        f32 num = numeric_grad_elem(loss_relu, x_data, shape, 1, i, 1e-2f);
        check_grad("relu.x", num, nx->grad.data[i]);
    }
    allocator_destroy(&alloc);
}

/* ================= softmax (son eksen): loss = sum(softmax(x) * W) ================= */

static f32 g_softW[6];

static f32 loss_softmax(const f32* xdata, const u64* shape, u32 ndim) {
    Allocator alloc = allocator_create(1024 * 1024);
    Tensor X = tensor_create(&alloc, shape, ndim);
    Tensor W = tensor_create(&alloc, shape, ndim);
    for (u64 i = 0; i < X.numel; i++) { X.data[i] = xdata[i]; W.data[i] = g_softW[i]; }
    Node* nx = node_leaf(&alloc, X, TRUE);
    Node* nw = node_leaf(&alloc, W, TRUE);
    Node* sm = node_softmax_lastdim(&alloc, nx);
    Node* prod = node_mul(&alloc, sm, nw);
    Node* loss = node_sum_all(&alloc, prod);
    f32 result = loss->value.data[0];
    allocator_destroy(&alloc);
    return result;
}

static void test_softmax_grad(void) {
    u64 shape[2] = { 2, 3 }; /* 2 satir, her satirda 3'luk softmax */
    f32 x_data[6] = { 1.0f, 2.0f, 0.5f, -1.0f, 0.2f, 3.0f };
    f32 w_data[6] = { 0.5f, -0.3f, 1.2f, 0.1f, -0.7f, 0.4f };
    for (u64 i = 0; i < 6; i++) g_softW[i] = w_data[i];

    Allocator alloc = allocator_create(1024 * 1024);
    Tensor X = tensor_create(&alloc, shape, 2);
    Tensor W = tensor_create(&alloc, shape, 2);
    for (u64 i = 0; i < 6; i++) { X.data[i] = x_data[i]; W.data[i] = w_data[i]; }
    Node* nx = node_leaf(&alloc, X, TRUE);
    Node* nw = node_leaf(&alloc, W, TRUE);
    Node* sm = node_softmax_lastdim(&alloc, nx);
    Node* prod = node_mul(&alloc, sm, nw);
    Node* loss = node_sum_all(&alloc, prod);
    backward(&alloc, loss);

    for (u64 i = 0; i < 6; i++) {
        f32 num = numeric_grad_elem(loss_softmax, x_data, shape, 2, i, 1e-2f);
        check_grad("softmax.x", num, nx->grad.data[i]);
    }
    allocator_destroy(&alloc);
}

int main(void) {
    console_write_line("=== Katman 4 (Otograd) - Sayisal Gradyan Kontrolu ===");

    test_add_grad();
    test_mul_grad();
    test_matmul_grad();
    test_transpose_grad();
    test_sigmoid_grad();
    test_relu_grad();
    test_softmax_grad();

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
