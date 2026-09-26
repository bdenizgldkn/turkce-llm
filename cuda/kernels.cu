/* Katman 5 - CUDA Cekirdekleri.
 * Bu dosya nvcc ile PTX'e derlenir (bkz. build_kernels.sh) ve CUDA
 * Driver API (cuModuleLoadData) ile calisma zamaninda yuklenir.
 * cuBLAS/cuDNN/Thrust GIBI HAZIR ALGORITMA KUTUPHANELERI KULLANILMAZ --
 * her cekirdek satir satir kendimiz yazdik. */

extern "C" __global__ void k_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void k_mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

extern "C" __global__ void k_relu(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

/* Karolu (tiled) matris carpimi, paylasimli bellek (shared memory)
 * kullanarak: her blok, A'nin bir [TILE x TILE] satir-parcasini ve
 * B'nin bir [TILE x TILE] sutun-parcasini paylasimli bellege BIR KEZ
 * yukler, sonra o karo icindeki TUM threadler bu veriyi TEKRAR TEKRAR
 * global bellege gitmeden kullanir. Naif versiyonun (her thread K kez
 * global bellekten okur) aksine, global bellek erisimi TILE kati
 * azalir -- bu, cuBLAS gibi kutuphanelerin de temelini olusturan
 * standart bir optimizasyondur, biz de sifirdan uyguluyoruz.
 * A: MxK, B: KxN, C: MxN (hepsi satir-majör / row-major).
 * TILE=16, cuda_backend.c'deki cuda_launch_2d'nin BLOCK_2D=16 blok
 * boyutuyla TAM ORTUSUR (blockDim.x=blockDim.y=16). */
#define TILE 16

extern "C" __global__ void k_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/* Blok-ici paralel indirgeme (reduction) toplami: paylasimli bellek
 * (shared memory) kullanarak her blok kendi kismi toplamini uretir.
 * Host tarafi, blok sayisi kadar kucuk partial-sum dizisini son adimda
 * kendi toplar (buyuk N icin bile blok sayisi kucuk kaldigindan bu
 * pragmatik ve dogrudur). */
extern "C" __global__ void k_reduce_sum(const float* x, float* block_sums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}
