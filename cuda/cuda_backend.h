/* Katman 5 - CUDA Driver API uzerine ince, kullanisli bir sarmalayici. */
#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "driver_api.h"

typedef struct CudaCtx {
    CUcontext ctx;
    CUdevice device;
    CUmodule module;
} CudaCtx;

/* ptx_path: build_kernels ile uretilen .ptx dosyasinin yolu. Basarisiz
 * olursa (surucu yok, dosya yok, vb.) tanilama mesaji basip programi
 * sonlandirir -- exception mekanizmamiz olmadigi icin bu bilincli bir
 * tercihtir (bkz. cuda_backend.c icindeki check()). */
CudaCtx cuda_init(Allocator* scratch, const char* ptx_path);

/* KATMAN 19 - Coklu-GPU destegi: cuda_init ile AYNI ama HANGI fiziksel
 * cihazda (0-indeksli, cuDeviceGet'e verilen ordinal) baglam kurulacagini
 * secmeye izin verir. cuda_init(scratch,path) == cuda_init_on_device(
 * scratch,path,0) -- yani TEK-GPU'lu makinelerde (bkz. Bolum 16-17)
 * davranis birebir aynidir, geriye-donuk uyumluluk garantidir. Birden
 * fazla GPU'lu bir makinede (bkz. PROJE_PLANI.md Bolum 19), her worker
 * kendi (baska bir) cihazda BAGIMSIZ bir baglam kurarak, worker'lari
 * GPU'lar arasinda dagitmak icin kullanilir (bkz. model/gpu_ops.h). */
CudaCtx cuda_init_on_device(Allocator* scratch, const char* ptx_path, i32 device_index);

/* Sistemde gorunen CUDA cihazi sayisi (cuInit + cuDeviceGetCount). */
i32 cuda_device_count(void);

void cuda_shutdown(CudaCtx* c);

/* CUDA baglamlari thread-lokaldir (bkz. driver_api.h) -- baska bir
 * thread'de olusturulmus bir baglami kullanmak isteyen HER thread,
 * herhangi bir cuMem/cuLaunch cagrisindan once bunu KENDI uzerinde
 * cagirmalidir. */
void cuda_set_current(CudaCtx* c);

CUfunction cuda_get_kernel(CudaCtx* c, const char* name);

CUdeviceptr cuda_alloc(u64 bytes);
void cuda_free(CUdeviceptr p);
void cuda_h2d(CUdeviceptr dst, const void* src, u64 bytes);
void cuda_d2h(void* dst, CUdeviceptr src, u64 bytes);

/* n elemanli 1B islemler icin (add/mul/relu vb.) grid/block hesaplayip cekirdegi baslatir. */
void cuda_launch_1d(CUfunction f, u32 n, void** params);

/* rows x cols cikti uzerinde 2B islemler icin (matmul) grid/block hesaplar. */
void cuda_launch_2d(CUfunction f, u32 rows, u32 cols, void** params);

/* Blok basina shared_bytes paylasimli bellek ile 1B baslat (reduction icin). */
void cuda_launch_1d_shared(CUfunction f, u32 n, u32 block_size, u32 shared_bytes, void** params);

/* Genel baslatma: grid (gx,gy,gz), blok (bx,by,bz), blok basina
 * shared_bytes dinamik paylasimli bellek (bkz. Katman 21, gpu_train.c). */
void cuda_launch(CUfunction f, u32 gx, u32 gy, u32 gz, u32 bx, u32 by, u32 bz, u32 shared_bytes, void** params);

/* [p, p+bytes) araligini sifirlar (cuMemsetD8_v2). */
void cuda_memset_zero(CUdeviceptr p, u64 bytes);

void cuda_sync(void);

#endif /* CUDA_BACKEND_H */
