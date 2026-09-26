/* Katman 5 - CUDA Driver API arayuzu.
 *
 * <cuda.h> DAHIL EDILMEZ (yuzlerce ilgisiz bildirim icerir). Bunun
 * yerine, tipki win32_syscalls.h'de yaptigimiz gibi, sadece ihtiyacimiz
 * olan fonksiyon imzalarini kendimiz bildiriyoruz. Bu fonksiyonlar
 * nvcuda.dll (GPU surucusuyle birlikte gelir) icinde tanimlidir ve
 * PROJE_PLANI.md'deki karara gore syscall'a esdeger kabul edilir --
 * cuBLAS/cuDNN gibi hazir algoritma kutuphaneleri DEGILDIR, sadece
 * donanima erisim kapisidir.
 *
 * Baglama: build_kernels.* ile uretilen cuda.lib (nvcuda.dll'in ice
 * aktarma kutuphanesi) linkere verilir.
 */
#ifndef CUDA_DRIVER_API_H
#define CUDA_DRIVER_API_H

#include "../runtime/types.h"

typedef i32 CUresult;
typedef i32 CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream; /* NULL = varsayilan stream */
typedef u64  CUdeviceptr;

#define CUDA_SUCCESS 0

CUresult cuInit(u32 Flags);
CUresult cuDeviceGetCount(i32* count);
CUresult cuDeviceGet(CUdevice* device, i32 ordinal);
CUresult cuDeviceGetName(char* name, i32 len, CUdevice dev);

CUresult cuCtxCreate_v2(CUcontext* pctx, u32 flags, CUdevice dev);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxSynchronize(void);
/* CUDA baglamlari THREAD-LOKALDIR -- bir baglam, olusturuldugu thread
 * disinda baska bir thread'de "gecerli" (current) DEGILDIR. Ayni
 * baglami birden fazla thread'den kullanmak icin, HER thread kendi
 * uzerinde bu fonksiyonu cagirmalidir (bkz. model/gpu_ops.c, cok
 * thread'li veri-paralel egitim -- PROJE_PLANI.md Bolum 16). */
CUresult cuCtxSetCurrent(CUcontext ctx);

CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult cuModuleUnload(CUmodule hmod);

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, u64 bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, u64 byteCount);
CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, u64 byteCount);
CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, u8 uc, u64 N);

CUresult cuLaunchKernel(
    CUfunction f,
    u32 gridDimX, u32 gridDimY, u32 gridDimZ,
    u32 blockDimX, u32 blockDimY, u32 blockDimZ,
    u32 sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra);

CUresult cuGetErrorName(CUresult error, const char** pStr);
CUresult cuGetErrorString(CUresult error, const char** pStr);

#endif /* CUDA_DRIVER_API_H */
