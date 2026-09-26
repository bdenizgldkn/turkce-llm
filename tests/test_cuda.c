/* Katman 5 dogrulama: CUDA cekirdeklerini GERCEK GPU (RTX 4050) uzerinde
 * calistirip, CPU'da hesaplanan beklenen sonuclarla karsilastirir.
 * Hicbir cuBLAS/cuDNN/Thrust kullanilmaz -- cekirdekler kernels.cu'da
 * elle yazilmis, CUDA Driver API (cuda_backend.h) ile yuklenip calistirilir. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../cuda/cuda_backend.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static bool32 f32_close(f32 a, f32 b, f32 tol) {
    f32 d = a - b;
    if (d < 0) d = -d;
    return d < tol;
}

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; console_write("  [FAIL] "); console_write_line(msg); } \
} while (0)

static void test_add(CudaCtx* ctx, Allocator* alloc) {
    const u32 N = 1000;
    f32* h_a = (f32*)allocator_alloc(alloc, N * sizeof(f32));
    f32* h_b = (f32*)allocator_alloc(alloc, N * sizeof(f32));
    f32* h_out = (f32*)allocator_alloc(alloc, N * sizeof(f32));
    for (u32 i = 0; i < N; i++) { h_a[i] = (f32)i; h_b[i] = (f32)(N - i); }

    CUdeviceptr d_a = cuda_alloc(N * sizeof(f32));
    CUdeviceptr d_b = cuda_alloc(N * sizeof(f32));
    CUdeviceptr d_out = cuda_alloc(N * sizeof(f32));

    cuda_h2d(d_a, h_a, N * sizeof(f32));
    cuda_h2d(d_b, h_b, N * sizeof(f32));

    CUfunction k = cuda_get_kernel(ctx, "k_add");
    i32 n_i = (i32)N;
    void* params[] = { &d_a, &d_b, &d_out, &n_i };
    cuda_launch_1d(k, N, params);
    cuda_sync();

    cuda_d2h(h_out, d_out, N * sizeof(f32));

    bool32 ok = TRUE;
    for (u32 i = 0; i < N; i++) {
        if (!f32_close(h_out[i], h_a[i] + h_b[i], 1e-4f)) { ok = FALSE; break; }
    }
    CHECK(ok, "cuda: k_add sonucu yanlis");

    cuda_free(d_a); cuda_free(d_b); cuda_free(d_out);
}

static void test_matmul(CudaCtx* ctx, Allocator* alloc) {
    /* A: 4x3, B: 3x5 -> C: 4x5, CPU'da naif hesapla ve karsilastir. */
    const i32 M = 4, K = 3, N = 5;
    f32* h_A = (f32*)allocator_alloc(alloc, M * K * sizeof(f32));
    f32* h_B = (f32*)allocator_alloc(alloc, K * N * sizeof(f32));
    f32* h_C = (f32*)allocator_alloc(alloc, M * N * sizeof(f32));
    f32* h_C_expected = (f32*)allocator_alloc(alloc, M * N * sizeof(f32));

    for (i32 i = 0; i < M * K; i++) h_A[i] = (f32)(i % 7) - 3.0f;
    for (i32 i = 0; i < K * N; i++) h_B[i] = (f32)(i % 5) * 0.5f;

    for (i32 r = 0; r < M; r++) {
        for (i32 c = 0; c < N; c++) {
            f32 sum = 0.0f;
            for (i32 k = 0; k < K; k++) sum += h_A[r * K + k] * h_B[k * N + c];
            h_C_expected[r * N + c] = sum;
        }
    }

    CUdeviceptr d_A = cuda_alloc(M * K * sizeof(f32));
    CUdeviceptr d_B = cuda_alloc(K * N * sizeof(f32));
    CUdeviceptr d_C = cuda_alloc(M * N * sizeof(f32));
    cuda_h2d(d_A, h_A, M * K * sizeof(f32));
    cuda_h2d(d_B, h_B, K * N * sizeof(f32));

    CUfunction k = cuda_get_kernel(ctx, "k_matmul");
    void* params[] = { &d_A, &d_B, &d_C, (void*)&M, (void*)&K, (void*)&N };
    cuda_launch_2d(k, (u32)M, (u32)N, params);
    cuda_sync();

    cuda_d2h(h_C, d_C, M * N * sizeof(f32));

    bool32 ok = TRUE;
    for (i32 i = 0; i < M * N; i++) {
        if (!f32_close(h_C[i], h_C_expected[i], 1e-3f)) { ok = FALSE; break; }
    }
    CHECK(ok, "cuda: k_matmul sonucu yanlis");

    cuda_free(d_A); cuda_free(d_B); cuda_free(d_C);
}

static void test_reduce_sum(CudaCtx* ctx, Allocator* alloc) {
    const u32 N = 10000;
    const u32 BLOCK = 256;
    u32 num_blocks = (N + BLOCK - 1) / BLOCK;

    f32* h_x = (f32*)allocator_alloc(alloc, N * sizeof(f32));
    f32* h_block_sums = (f32*)allocator_alloc(alloc, num_blocks * sizeof(f32));

    f32 expected = 0.0f;
    for (u32 i = 0; i < N; i++) { h_x[i] = 1.0f; expected += 1.0f; } /* toplam = N */

    CUdeviceptr d_x = cuda_alloc(N * sizeof(f32));
    CUdeviceptr d_block_sums = cuda_alloc(num_blocks * sizeof(f32));
    cuda_h2d(d_x, h_x, N * sizeof(f32));

    CUfunction k = cuda_get_kernel(ctx, "k_reduce_sum");
    i32 n_i = (i32)N;
    void* params[] = { &d_x, &d_block_sums, &n_i };
    cuda_launch_1d_shared(k, N, BLOCK, BLOCK * (u32)sizeof(f32), params);
    cuda_sync();

    cuda_d2h(h_block_sums, d_block_sums, num_blocks * sizeof(f32));

    f32 total = 0.0f;
    for (u32 i = 0; i < num_blocks; i++) total += h_block_sums[i];

    CHECK(f32_close(total, expected, 1e-2f), "cuda: k_reduce_sum toplami yanlis");

    cuda_free(d_x); cuda_free(d_block_sums);
}

int main(void) {
    console_write_line("=== Katman 5 (CUDA Cekirdekleri) Testleri ===");

    Allocator alloc = allocator_create(16ull * 1024 * 1024);
    CudaCtx ctx = cuda_init(&alloc, "cuda/kernels.ptx");

    char devname[256];
    cuDeviceGetName(devname, 256, ctx.device);
    console_write("GPU: "); console_write_line(devname);

    test_add(&ctx, &alloc);
    test_matmul(&ctx, &alloc);
    test_reduce_sum(&ctx, &alloc);

    cuda_shutdown(&ctx);
    allocator_destroy(&alloc);

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
