#include "gpu_ops.h"
#include "../cuda/cuda_backend.h"
#include "../runtime/timer.h"
#include "../runtime/thread.h"

#define MAX_GPU_CONTEXTS 32

/* Slot 0 = gpu_ops_init'in kurdugu VARSAYILAN (mutex ile PAYLASILAN)
 * baglam -- geriye-donuk uyumluluk icin (generate.c/chat.c/tekli-thread
 * testler gibi cagiranlar hicbir sey degistirmeden calisir devam eder).
 * Slot [1, 1+num_worker_ctxs) = gpu_ops_init_workers ile onceden kurulmus
 * BAGIMSIZ baglamlar (bkz. gpu_ops.h). */
static CudaCtx g_ctxs[MAX_GPU_CONTEXTS];
static CUfunction g_kernel_matmuls[MAX_GPU_CONTEXTS];
static u32 g_num_worker_ctxs = 0;
static bool32 g_initialized = FALSE;

static MutexHandle g_gpu_mutex; /* sadece slot 0 (varsayilan/paylasilan) icin */
static MutexHandle g_debug_mutex; /* tanilama sayaclari icin (hangi slot kullanilirsa kullanilsin) */

static _Thread_local i32 g_tls_worker_id = -1;

/* Slot basina KALICI cihaz tamponlari (A, B, C). Her gpu_matmul'da
 * cuMemAlloc/cuMemFree cagirmak yerine, sadece gereken boyut mevcut
 * kapasiteyi asinca (pratikte ilk adimlarda) buyutulur. Neden: CUDA
 * surucusu cuMemAlloc/cuMemFree'yi AYNI CIHAZDAKI TUM baglamlar arasinda
 * dahili bir kilitle siralar -- 28 worker'li L4 kosusunda worker'larin
 * cogu bu kilitte bekliyordu (gdb ile olculdu, bkz. PROJE_PLANI.md
 * Bolum 20). Bir slot'a sadece sahibi olan thread dokunur (slot 0 zaten
 * g_gpu_mutex altinda), dolayisiyla ek kilit gerekmez. */
typedef struct GpuBuf {
    CUdeviceptr ptr;
    u64 capacity;
} GpuBuf;
static GpuBuf g_bufs[MAX_GPU_CONTEXTS][3];

/* Cagirmadan once slot'un baglami current yapilmis olmali. */
static CUdeviceptr gpu_buf_ensure(u32 slot, u32 which, u64 bytes) {
    GpuBuf* b = &g_bufs[slot][which];
    if (b->capacity < bytes) {
        if (b->capacity > 0) cuda_free(b->ptr);
        b->ptr = cuda_alloc(bytes);
        b->capacity = bytes;
    }
    return b->ptr;
}

static void gpu_bufs_release(u32 slot) {
    cuda_set_current(&g_ctxs[slot]);
    for (u32 w = 0; w < 3; w++) {
        if (g_bufs[slot][w].capacity > 0) cuda_free(g_bufs[slot][w].ptr);
        g_bufs[slot][w].ptr = 0;
        g_bufs[slot][w].capacity = 0;
    }
}

static f64 g_debug_total_seconds = 0.0;
static u32 g_debug_call_count = 0;

void gpu_ops_debug_stats(f64* out_total_seconds, u32* out_call_count) {
    *out_total_seconds = g_debug_total_seconds;
    *out_call_count = g_debug_call_count;
}

void gpu_ops_debug_reset(void) {
    g_debug_total_seconds = 0.0;
    g_debug_call_count = 0;
}

void gpu_ops_init(Allocator* scratch, const char* ptx_path) {
    g_ctxs[0] = cuda_init(scratch, ptx_path);
    g_kernel_matmuls[0] = cuda_get_kernel(&g_ctxs[0], "k_matmul");
    g_gpu_mutex = mutex_create();
    g_debug_mutex = mutex_create();
    g_initialized = TRUE;
}

void gpu_ops_init_workers(Allocator* scratch, const char* ptx_path, u32 num_workers) {
    if (num_workers > MAX_GPU_CONTEXTS - 1) num_workers = MAX_GPU_CONTEXTS - 1;

    /* KATMAN 19 - Coklu-GPU: worker'lar GORUNEN TUM cihazlara sirayla
     * (round-robin) dagitilir. Tek-GPU'lu bir makinede num_devices=1,
     * yani HER worker device_index=0 alir -- Bolum 16-17'deki davranisla
     * BIREBIR AYNI (geriye-donuk uyumluluk, yerelde tam dogrulanabilir).
     * Birden fazla GPU'lu bir makinede (bkz. PROJE_PLANI.md Bolum 19),
     * ornegin 4 GPU + 32 worker ile her GPU 8 worker'in bagimsiz
     * baglamina ev sahipligi yapar -- bu GPU-basina cekismeyi (contention)
     * de worker sayisi/GPU sayisi kat azaltir. */
    i32 num_devices = cuda_device_count();
    if (num_devices < 1) num_devices = 1;

    for (u32 i = 0; i < num_workers; i++) {
        i32 device_index = (i32)(i % (u32)num_devices);
        g_ctxs[1 + i] = cuda_init_on_device(scratch, ptx_path, device_index);
        g_kernel_matmuls[1 + i] = cuda_get_kernel(&g_ctxs[1 + i], "k_matmul");
    }
    g_num_worker_ctxs = num_workers;
}

void gpu_ops_set_worker_id(i32 worker_id) {
    g_tls_worker_id = worker_id;
}

void gpu_ops_shutdown(void) {
    if (g_initialized) {
        gpu_bufs_release(0);
        cuda_shutdown(&g_ctxs[0]);
        for (u32 i = 0; i < g_num_worker_ctxs; i++) {
            gpu_bufs_release(1 + i);
            cuda_shutdown(&g_ctxs[1 + i]);
        }
        g_initialized = FALSE;
    }
}

/* a,b bitisik (contiguous) olmalidir -- degilse cagiran taraf once
 * tensor_contiguous ile somutlastirmalidir (bkz. backward_matmul2d_gpu). */
static Tensor gpu_matmul(Allocator* alloc, const Tensor* a, const Tensor* b) {
    f64 t0 = timer_now_seconds();
    Tensor a_c = tensor_is_contiguous(a) ? *a : tensor_contiguous(alloc, a);
    Tensor b_c = tensor_is_contiguous(b) ? *b : tensor_contiguous(alloc, b);

    u64 M = a_c.shape[0], K = a_c.shape[1], N = b_c.shape[1];
    u64 out_shape[2] = { M, N };
    Tensor out = tensor_create(alloc, out_shape, 2);

    i32 wid = g_tls_worker_id;
    bool32 has_own_ctx = (wid >= 0 && (u32)wid < g_num_worker_ctxs);
    u32 slot = has_own_ctx ? (u32)(1 + wid) : 0;

    /* Kendi bagimsiz baglami olan bir worker icin KILIT GEREKMEZ --
     * baska hicbir thread bu SLOT'a dokunmaz. Sadece varsayilan
     * (paylasilan) slot 0 icin mutex korumasi gerekir. */
    if (!has_own_ctx) mutex_lock(&g_gpu_mutex);

    cuda_set_current(&g_ctxs[slot]);

    CUdeviceptr d_a = gpu_buf_ensure(slot, 0, M * K * sizeof(f32));
    CUdeviceptr d_b = gpu_buf_ensure(slot, 1, K * N * sizeof(f32));
    CUdeviceptr d_c = gpu_buf_ensure(slot, 2, M * N * sizeof(f32));

    cuda_h2d(d_a, a_c.data, M * K * sizeof(f32));
    cuda_h2d(d_b, b_c.data, K * N * sizeof(f32));

    i32 iM = (i32)M, iK = (i32)K, iN = (i32)N;
    void* params[] = { &d_a, &d_b, &d_c, &iM, &iK, &iN };
    cuda_launch_2d(g_kernel_matmuls[slot], (u32)M, (u32)N, params);
    cuda_sync();

    cuda_d2h(out.data, d_c, M * N * sizeof(f32));

    if (!has_own_ctx) mutex_unlock(&g_gpu_mutex);

    mutex_lock(&g_debug_mutex);
    g_debug_total_seconds += (timer_now_seconds() - t0);
    g_debug_call_count++;
    mutex_unlock(&g_debug_mutex);

    return out;
}

static void backward_matmul2d_gpu(Node* self, Allocator* alloc) {
    Node* a = self->parents[0];
    Node* b = self->parents[1];

    Tensor bt = tensor_transpose(&b->value, 0, 1);
    Tensor da = gpu_matmul(alloc, &self->grad, &bt);
    tensor_add_inplace(&a->grad, &da);

    Tensor at = tensor_transpose(&a->value, 0, 1);
    Tensor db = gpu_matmul(alloc, &at, &self->grad);
    tensor_add_inplace(&b->grad, &db);
}

Node* node_matmul2d_gpu(Allocator* alloc, Node* a, Node* b) {
    Tensor v = gpu_matmul(alloc, &a->value, &b->value);
    Node* n = node_make_custom(alloc, v, TRUE);
    n->parents[0] = a; n->parents[1] = b; n->num_parents = 2;
    n->backward_fn = backward_matmul2d_gpu;
    return n;
}
