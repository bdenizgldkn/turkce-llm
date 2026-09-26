/* Katman 3 dogrulama testleri: Tensor (shape/stride, view'ler,
 * elemanter islemler, matmul). */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../tensor/tensor.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; console_write("  [FAIL] "); console_write_line(msg); } \
} while (0)

static bool32 f32_close(f32 a, f32 b) {
    f32 d = a - b;
    if (d < 0) d = -d;
    return d < 1e-5f;
}

static void test_create_fill_getset(Allocator* alloc) {
    u64 shape[2] = { 3, 4 };
    Tensor t = tensor_create(alloc, shape, 2);
    tensor_fill(&t, 7.0f);

    bool32 all_seven = TRUE;
    for (u64 i = 0; i < 3; i++)
        for (u64 j = 0; j < 4; j++) {
            u64 idx[2] = { i, j };
            if (!f32_close(tensor_get(&t, idx), 7.0f)) all_seven = FALSE;
        }
    CHECK(all_seven, "tensor: fill/get tutarsiz");

    u64 idx2[2] = { 1, 2 };
    tensor_set(&t, idx2, 42.0f);
    CHECK(f32_close(tensor_get(&t, idx2), 42.0f), "tensor: set/get tutarsiz");
    CHECK(t.numel == 12, "tensor: numel yanlis");
    CHECK(tensor_is_contiguous(&t), "tensor: yeni olusturulan tensor bitisik olmali");

    tensor_free(alloc, &t);
}

static void test_reshape(Allocator* alloc) {
    u64 shape[2] = { 2, 6 };
    Tensor t = tensor_create(alloc, shape, 2);
    for (u64 i = 0; i < t.numel; i++) t.data[i] = (f32)i;

    u64 new_shape[3] = { 2, 2, 3 };
    Tensor r = tensor_reshape(&t, new_shape, 3);
    CHECK(r.numel == 12, "reshape: numel korunmadi");

    u64 idx[3] = { 1, 1, 0 }; /* flat index = 1*6 + 1*3 + 0 = 9 */
    CHECK(f32_close(tensor_get(&r, idx), 9.0f), "reshape: veri duzeni bozuldu");

    tensor_free(alloc, &t); /* r bir view, veriyi paylasir; r icin free cagirmaya gerek yok */
}

static void test_transpose(Allocator* alloc) {
    u64 shape[2] = { 2, 3 };
    Tensor t = tensor_create(alloc, shape, 2);
    /* [[0,1,2],[3,4,5]] */
    for (u64 i = 0; i < 6; i++) t.data[i] = (f32)i;

    Tensor tt = tensor_transpose(&t, 0, 1); /* mantiken [[0,3],[1,4],[2,5]] */
    CHECK(tt.shape[0] == 3 && tt.shape[1] == 2, "transpose: shape yanlis");
    CHECK(!tensor_is_contiguous(&tt), "transpose: view bitisik olmamali");

    u64 idx[2] = { 1, 0 };
    CHECK(f32_close(tensor_get(&tt, idx), 1.0f), "transpose: [1,0] beklenen 1.0");
    u64 idx2[2] = { 2, 1 };
    CHECK(f32_close(tensor_get(&tt, idx2), 5.0f), "transpose: [2,1] beklenen 5.0");

    Tensor cont = tensor_contiguous(alloc, &tt);
    CHECK(tensor_is_contiguous(&cont), "contiguous: sonuc bitisik degil");
    /* cont fiziksel olarak [[0,3],[1,4],[2,5]] duzeninde olmali -> flat: 0,3,1,4,2,5 */
    f32 expected[6] = { 0, 3, 1, 4, 2, 5 };
    bool32 match = TRUE;
    for (u64 i = 0; i < 6; i++) if (!f32_close(cont.data[i], expected[i])) match = FALSE;
    CHECK(match, "contiguous: fiziksel veri duzeni yanlis");

    tensor_free(alloc, &cont);
    tensor_free(alloc, &t);
}

static void test_elementwise(Allocator* alloc) {
    u64 shape[1] = { 4 };
    Tensor a = tensor_create(alloc, shape, 1);
    Tensor b = tensor_create(alloc, shape, 1);
    f32 av[4] = { 1, 2, 3, 4 };
    f32 bv[4] = { 10, 20, 30, 40 };
    for (u64 i = 0; i < 4; i++) { a.data[i] = av[i]; b.data[i] = bv[i]; }

    Tensor sum = tensor_add(alloc, &a, &b);
    Tensor diff = tensor_sub(alloc, &a, &b);
    Tensor prod = tensor_mul(alloc, &a, &b);
    Tensor scaled = tensor_scale(alloc, &a, 2.0f);

    f32 exp_sum[4] = { 11, 22, 33, 44 };
    f32 exp_diff[4] = { -9, -18, -27, -36 };
    f32 exp_prod[4] = { 10, 40, 90, 160 };
    f32 exp_scaled[4] = { 2, 4, 6, 8 };

    bool32 ok = TRUE;
    for (u64 i = 0; i < 4; i++) {
        if (!f32_close(sum.data[i], exp_sum[i])) ok = FALSE;
        if (!f32_close(diff.data[i], exp_diff[i])) ok = FALSE;
        if (!f32_close(prod.data[i], exp_prod[i])) ok = FALSE;
        if (!f32_close(scaled.data[i], exp_scaled[i])) ok = FALSE;
    }
    CHECK(ok, "elementwise: add/sub/mul/scale sonuclari yanlis");

    tensor_free(alloc, &a); tensor_free(alloc, &b);
    tensor_free(alloc, &sum); tensor_free(alloc, &diff);
    tensor_free(alloc, &prod); tensor_free(alloc, &scaled);
}

static void test_matmul(Allocator* alloc) {
    /* A = [[1,2],[3,4]] (2x2), B = [[5,6],[7,8]] (2x2)
     * A*B = [[1*5+2*7, 1*6+2*8],[3*5+4*7,3*6+4*8]] = [[19,22],[43,50]] */
    u64 shape[2] = { 2, 2 };
    Tensor A = tensor_create(alloc, shape, 2);
    Tensor B = tensor_create(alloc, shape, 2);
    f32 av[4] = { 1, 2, 3, 4 };
    f32 bv[4] = { 5, 6, 7, 8 };
    for (u64 i = 0; i < 4; i++) { A.data[i] = av[i]; B.data[i] = bv[i]; }

    Tensor C = tensor_matmul2d(alloc, &A, &B);
    f32 expected[4] = { 19, 22, 43, 50 };
    bool32 ok = TRUE;
    for (u64 i = 0; i < 4; i++) if (!f32_close(C.data[i], expected[i])) ok = FALSE;
    CHECK(ok, "matmul: 2x2 sonucu yanlis");

    /* Transpoze edilmis (bitisik olmayan) bir girdi ile de dogru calismali. */
    Tensor At = tensor_transpose(&A, 0, 1); /* [[1,3],[2,4]] */
    Tensor C2 = tensor_matmul2d(alloc, &At, &B);
    /* At*B = [[1*5+3*7,1*6+3*8],[2*5+4*7,2*6+4*8]] = [[26,30],[38,44]] */
    f32 expected2[4] = { 26, 30, 38, 44 };
    bool32 ok2 = TRUE;
    for (u64 i = 0; i < 4; i++) if (!f32_close(C2.data[i], expected2[i])) ok2 = FALSE;
    CHECK(ok2, "matmul: bitisik olmayan (transpoze) girdiyle sonuc yanlis");

    tensor_free(alloc, &A); tensor_free(alloc, &B);
    tensor_free(alloc, &C); tensor_free(alloc, &C2);
}

int main(void) {
    console_write_line("=== Katman 3 (Tensor) Testleri ===");

    Allocator alloc = allocator_create(16ull * 1024 * 1024);

    test_create_fill_getset(&alloc);
    test_reshape(&alloc);
    test_transpose(&alloc);
    test_elementwise(&alloc);
    test_matmul(&alloc);

    allocator_destroy(&alloc);

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
