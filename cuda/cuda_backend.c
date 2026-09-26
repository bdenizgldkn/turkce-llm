#include "cuda_backend.h"
#include "../runtime/file_io.h"
#include "../runtime/console.h"

/* KATMAN 19 - Coklu platform: Windows'ta ExitProcess, Linux'ta (Debian
 * VM) POSIX _exit (bkz. PROJE_PLANI.md Bolum 19) -- ikisi de sureci
 * ANINDA sonlandiran ham syscall sarmalayicisidir (libc'nin exit()'i
 * gibi atexit-handler/stdio-flush davranisi YOKTUR). */
#ifdef _WIN32
#include "../runtime/win32_syscalls.h"
#else
#include <unistd.h>
#endif

static void fatal_exit(void) {
#ifdef _WIN32
    ExitProcess(1);
#else
    _exit(1);
#endif
}

static void check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* name = "?";
        const char* desc = "?";
        cuGetErrorName(r, &name);
        cuGetErrorString(r, &desc);
        console_write("[CUDA HATA] "); console_write(what);
        console_write(" -> "); console_write(name);
        console_write(": "); console_write_line(desc);
        fatal_exit();
    }
}

static void* read_ptx_null_terminated(Allocator* scratch, const char* path, u64* out_size) {
    FileHandle f = file_open_read(path);
    if (!f.valid) {
        console_write("[CUDA HATA] PTX dosyasi acilamadi: ");
        console_write_line(path);
        fatal_exit();
    }
    u64 size = file_size(&f);
    u8* buf = (u8*)allocator_alloc(scratch, size + 1);
    u64 got = file_read(&f, buf, size);
    file_close(&f);
    if (got != size) {
        console_write_line("[CUDA HATA] PTX dosyasi tam okunamadi");
        fatal_exit();
    }
    buf[size] = 0; /* cuModuleLoadData null-sonlandirilmis metin bekler */
    *out_size = size;
    return buf;
}

CudaCtx cuda_init_on_device(Allocator* scratch, const char* ptx_path, i32 device_index) {
    CudaCtx c;
    check(cuInit(0), "cuInit"); /* idempotent -- tekrar cagirmak guvenlidir */

    i32 count = 0;
    check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (count < 1) {
        console_write_line("[CUDA HATA] Hicbir CUDA cihazi bulunamadi");
        fatal_exit();
    }
    if (device_index >= count) {
        console_write("[CUDA HATA] Istenen cihaz indeksi ("); console_write_u64((u64)device_index);
        console_write(") gorunen cihaz sayisini ("); console_write_u64((u64)count);
        console_write_line(") asiyor.");
        fatal_exit();
    }

    check(cuDeviceGet(&c.device, device_index), "cuDeviceGet");
    check(cuCtxCreate_v2(&c.ctx, 0, c.device), "cuCtxCreate_v2");

    u64 ptx_size;
    void* ptx = read_ptx_null_terminated(scratch, ptx_path, &ptx_size);
    check(cuModuleLoadData(&c.module, ptx), "cuModuleLoadData");

    return c;
}

CudaCtx cuda_init(Allocator* scratch, const char* ptx_path) {
    return cuda_init_on_device(scratch, ptx_path, 0);
}

i32 cuda_device_count(void) {
    check(cuInit(0), "cuInit");
    i32 count = 0;
    check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    return count;
}

void cuda_shutdown(CudaCtx* c) {
    cuModuleUnload(c->module);
    cuCtxDestroy_v2(c->ctx);
}

void cuda_set_current(CudaCtx* c) {
    check(cuCtxSetCurrent(c->ctx), "cuCtxSetCurrent");
}

CUfunction cuda_get_kernel(CudaCtx* c, const char* name) {
    CUfunction f;
    check(cuModuleGetFunction(&f, c->module, name), "cuModuleGetFunction");
    return f;
}

CUdeviceptr cuda_alloc(u64 bytes) {
    CUdeviceptr p;
    check(cuMemAlloc_v2(&p, bytes), "cuMemAlloc_v2");
    return p;
}

void cuda_free(CUdeviceptr p) {
    check(cuMemFree_v2(p), "cuMemFree_v2");
}

void cuda_h2d(CUdeviceptr dst, const void* src, u64 bytes) {
    check(cuMemcpyHtoD_v2(dst, src, bytes), "cuMemcpyHtoD_v2");
}

void cuda_d2h(void* dst, CUdeviceptr src, u64 bytes) {
    check(cuMemcpyDtoH_v2(dst, src, bytes), "cuMemcpyDtoH_v2");
}

#define BLOCK_1D 256

void cuda_launch_1d(CUfunction f, u32 n, void** params) {
    u32 grid = (n + BLOCK_1D - 1) / BLOCK_1D;
    check(cuLaunchKernel(f, grid, 1, 1, BLOCK_1D, 1, 1, 0, NULL_PTR, params, NULL_PTR),
          "cuLaunchKernel(1d)");
}

void cuda_launch_1d_shared(CUfunction f, u32 n, u32 block_size, u32 shared_bytes, void** params) {
    u32 grid = (n + block_size - 1) / block_size;
    check(cuLaunchKernel(f, grid, 1, 1, block_size, 1, 1, shared_bytes, NULL_PTR, params, NULL_PTR),
          "cuLaunchKernel(1d_shared)");
}

#define BLOCK_2D 16

void cuda_launch_2d(CUfunction f, u32 rows, u32 cols, void** params) {
    u32 grid_x = (cols + BLOCK_2D - 1) / BLOCK_2D;
    u32 grid_y = (rows + BLOCK_2D - 1) / BLOCK_2D;
    check(cuLaunchKernel(f, grid_x, grid_y, 1, BLOCK_2D, BLOCK_2D, 1, 0, NULL_PTR, params, NULL_PTR),
          "cuLaunchKernel(2d)");
}

void cuda_launch(CUfunction f, u32 gx, u32 gy, u32 gz, u32 bx, u32 by, u32 bz, u32 shared_bytes, void** params) {
    check(cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared_bytes, NULL_PTR, params, NULL_PTR),
          "cuLaunchKernel");
}

void cuda_memset_zero(CUdeviceptr p, u64 bytes) {
    check(cuMemsetD8_v2(p, 0, bytes), "cuMemsetD8_v2");
}

void cuda_sync(void) {
    check(cuCtxSynchronize(), "cuCtxSynchronize");
}
