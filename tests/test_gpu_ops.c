/* Katman 7/8 (devam) dogrulama: GPU matmul entegrasyonu.
 * 1) Dogruluk: GPU sonucu CPU sonucuyla (tensor_matmul2d) karsilastirilir.
 * 2) Sayisal gradyan kontrolu: node_matmul2d_gpu otogradda dogru mu.
 * 3) Zamanlama: buyuk bir matris icin CPU'ya karsi GPU hizi (gercek
 *   RTX 4050 uzerinde) -- hizlanmanin somut kaniti. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/prng.h"
#include "../runtime/timer.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/gpu_ops.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static f32 fabsf_(f32 v) { return (v < 0) ? -v : v; }

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; console_write("  [FAIL] "); console_write_line(msg); } \
} while (0)

static void test_correctness(Allocator* alloc, PCGState* rng) {
    u64 M = 64, K = 48, N = 32;
    u64 ashape[2] = { M, K }, bshape[2] = { K, N };
    Tensor A = tensor_create(alloc, ashape, 2);
    Tensor B = tensor_create(alloc, bshape, 2);
    for (u64 i = 0; i < M * K; i++) A.data[i] = (f32)pcg_gaussian(rng, 0.0, 1.0);
    for (u64 i = 0; i < K * N; i++) B.data[i] = (f32)pcg_gaussian(rng, 0.0, 1.0);

    Tensor cpu_result = tensor_matmul2d(alloc, &A, &B);

    Node* na = node_leaf(alloc, A, TRUE);
    Node* nb = node_leaf(alloc, B, TRUE);
    Node* gpu_node = node_matmul2d_gpu(alloc, na, nb);

    bool32 match = TRUE;
    f32 max_diff = 0.0f;
    for (u64 i = 0; i < M * N; i++) {
        f32 d = fabsf_(cpu_result.data[i] - gpu_node->value.data[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1e-2f) match = FALSE;
    }
    console_write("  CPU/GPU matmul en buyuk fark*1e6: ");
    console_write_u64((u64)(max_diff * 1e6f));
    console_write_line("");
    CHECK(match, "gpu matmul: CPU sonucundan farkli");
}

static f32 g_a[6], g_b[6];

static f32 loss_gpu_matmul(void) {
    Allocator alloc = allocator_create(4ull * 1024 * 1024);
    u64 ashape[2] = { 2, 3 }, bshape[2] = { 3, 2 };
    Tensor A = tensor_create(&alloc, ashape, 2);
    Tensor B = tensor_create(&alloc, bshape, 2);
    for (u64 i = 0; i < 6; i++) { A.data[i] = g_a[i]; B.data[i] = g_b[i]; }
    Node* na = node_leaf(&alloc, A, TRUE);
    Node* nb = node_leaf(&alloc, B, TRUE);
    Node* c = node_matmul2d_gpu(&alloc, na, nb);
    Node* loss = node_sum_all(&alloc, c);
    f32 v = loss->value.data[0];
    allocator_destroy(&alloc);
    return v;
}

static void test_gradient(PCGState* rng) {
    for (u64 i = 0; i < 6; i++) g_a[i] = (f32)pcg_gaussian(rng, 0.0, 1.0);
    for (u64 i = 0; i < 6; i++) g_b[i] = (f32)pcg_gaussian(rng, 0.0, 1.0);

    Allocator alloc = allocator_create(4ull * 1024 * 1024);
    u64 ashape[2] = { 2, 3 }, bshape[2] = { 3, 2 };
    Tensor A = tensor_create(&alloc, ashape, 2);
    Tensor B = tensor_create(&alloc, bshape, 2);
    for (u64 i = 0; i < 6; i++) { A.data[i] = g_a[i]; B.data[i] = g_b[i]; }
    Node* na = node_leaf(&alloc, A, TRUE);
    Node* nb = node_leaf(&alloc, B, TRUE);
    Node* c = node_matmul2d_gpu(&alloc, na, nb);
    Node* loss = node_sum_all(&alloc, c);
    backward(&alloc, loss);

    f32 eps = 1e-2f;
    for (u64 i = 0; i < 6; i++) {
        f32 orig = g_a[i];
        g_a[i] = orig + eps; f32 lp = loss_gpu_matmul();
        g_a[i] = orig - eps; f32 lm = loss_gpu_matmul();
        g_a[i] = orig;
        f32 numeric = (lp - lm) / (2.0f * eps);
        f32 tol = 0.03f + 0.05f * fabsf_(numeric);
        CHECK(fabsf_(numeric - na->grad.data[i]) <= tol, "gpu matmul: A gradyani yanlis");
    }
    for (u64 i = 0; i < 6; i++) {
        f32 orig = g_b[i];
        g_b[i] = orig + eps; f32 lp = loss_gpu_matmul();
        g_b[i] = orig - eps; f32 lm = loss_gpu_matmul();
        g_b[i] = orig;
        f32 numeric = (lp - lm) / (2.0f * eps);
        f32 tol = 0.03f + 0.05f * fabsf_(numeric);
        CHECK(fabsf_(numeric - nb->grad.data[i]) <= tol, "gpu matmul: B gradyani yanlis");
    }
    allocator_destroy(&alloc);
}

static void test_timing(Allocator* alloc, PCGState* rng) {
    u64 N = 1024;
    u64 shape[2] = { N, N };
    Tensor A = tensor_create(alloc, shape, 2);
    Tensor B = tensor_create(alloc, shape, 2);
    for (u64 i = 0; i < N * N; i++) A.data[i] = (f32)pcg_gaussian(rng, 0.0, 1.0);
    for (u64 i = 0; i < N * N; i++) B.data[i] = (f32)pcg_gaussian(rng, 0.0, 1.0);

    Node* na = node_leaf(alloc, A, TRUE);
    Node* nb = node_leaf(alloc, B, TRUE);
    node_matmul2d_gpu(alloc, na, nb); /* isinma turu (JIT/context maliyetini disla) */

    f64 t0 = timer_now_seconds();
    Tensor cpu_r = tensor_matmul2d(alloc, &A, &B);
    f64 t1 = timer_now_seconds();
    (void)cpu_r;

    f64 t2 = timer_now_seconds();
    node_matmul2d_gpu(alloc, na, nb);
    f64 t3 = timer_now_seconds();

    f64 cpu_ms = (t1 - t0) * 1000.0;
    f64 gpu_ms = (t3 - t2) * 1000.0;

    console_write("  1024x1024 matmul -- CPU: "); console_write_u64((u64)cpu_ms); console_write(" ms, GPU: ");
    console_write_u64((u64)gpu_ms); console_write_line(" ms (H2D+kernel+D2H dahil)");

    if (gpu_ms > 0.0) {
        console_write("  Hizlanma: "); console_write_u64((u64)(cpu_ms / gpu_ms)); console_write_line("x");
    }
}

int main(void) {
    console_write_line("=== CUDA Entegrasyonu: GPU Matmul Testleri ===");

    Allocator scratch = allocator_create(64ull * 1024 * 1024);
    gpu_ops_init(&scratch, "cuda/kernels.ptx");

    Allocator alloc = allocator_create(200ull * 1024 * 1024);
    PCGState rng = pcg_seed(2026, 9);

    test_correctness(&alloc, &rng);
    test_gradient(&rng);
    test_timing(&alloc, &rng);

    gpu_ops_shutdown();
    allocator_destroy(&alloc);
    allocator_destroy(&scratch);

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
