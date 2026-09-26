/* Katman 21 - GPU'da Tutulan (GPU-resident) Egitim Cekirdekleri.
 *
 * Bu dosyadaki cekirdekler, modelin TAMAMINI (ileri + geri yayilim +
 * Adam) GPU bellegi uzerinde calistirir; tensorler adimlar arasinda
 * CPU'ya gidip gelmez (bkz. model/gpu_train.h, PROJE_PLANI.md Bolum 21).
 * kernels.cu gibi nvcc ile PTX'e derlenir:
 *     nvcc -ptx -arch=sm_89 cuda/train_kernels.cu -o cuda/train_kernels.ptx
 * cuBLAS/cuDNN/Thrust KULLANILMAZ; exp/log/sqrt de CUDA'nin matematik
 * kutuphanesinden degil, runtime/mathlib.c'deki algoritmalarin GPU
 * surumlerinden gelir (asagida d_*).
 *
 * Duzen sozlesmesi (hepsi satir-major / row-major):
 *   N = B*T satir (B dizi arka arkaya, her biri T token)
 *   residual/normed: [N, D]    qkv: [N, 3D]    gate_up: [N, 2F]
 *   Q/K/V/O (baslik-major): [B, H, T, hd]      P (softmax): [B, H, T, T]
 *
 * DETERMINIZM: hicbir cekirdek atomik islem kullanmaz; tum indirgemeler
 * sabit sirali agac/dongu ile yapilir. Ayni girdiyle ayni cikti BIT BIT
 * ayni uretilir (checkpoint'ten devam etmenin kesintisiz kosuyla ayni
 * sonucu vermesi icin gerekli, bkz. tests/test_gpu_train.c). */

/* ================= Matematik (runtime/mathlib.c'nin GPU surumleri) ================= */

typedef union { double f; unsigned long long u; } D64Bits;
typedef union { float f; unsigned int u; } D32Bits;

/* m_sqrt'un birebir aynisi (decompose + 6 Newton-Raphson + ldexp2),
 * sadece x > 0 sonlu girdiler icin (satir basina bir kez cagrilir:
 * RMSNorm). CPU ile ayni sonucu verir. */
__device__ double d_sqrt64(double x) {
    D64Bits c; c.f = x;
    int e = (int)((c.u & 0x7FF0000000000000ull) >> 52) - 1023;
    c.u = (c.u & 0x000FFFFFFFFFFFFFull) | (1023ull << 52);
    double m = c.f;
    if (e & 1) { m *= 2.0; e -= 1; }
    double y = m * 0.5 + 0.5;
    for (int i = 0; i < 6; i++) y = 0.5 * (y + m / y);
    D64Bits r; r.f = y;
    long long ne = (long long)((r.u & 0x7FF0000000000000ull) >> 52) + e / 2;
    r.u = (r.u & 0x800FFFFFFFFFFFFFull) | ((unsigned long long)ne << 52);
    return r.f;
}

/* m_log'un birebir aynisi (x >= 1 sonlu; capraz-entropide sum_exp). */
__device__ double d_log64(double x) {
    D64Bits c; c.f = x;
    int e = (int)((c.u & 0x7FF0000000000000ull) >> 52) - 1023;
    c.u = (c.u & 0x000FFFFFFFFFFFFFull) | (1023ull << 52);
    double m = c.f;
    const double SQRT2 = 1.4142135623730951;
    if (m > SQRT2) { m *= 0.5; e += 1; }
    double z = (m - 1.0) / (m + 1.0);
    double z2 = z * z;
    double term = z, sum = z;
    for (int n = 3; n < 60; n += 2) {
        term *= z2;
        double add = term / (double)n;
        sum += add;
        if (add < 1e-18 && add > -1e-18) break;
    }
    return 2.0 * sum + (double)e * 0.6931471805599453;
}

/* f32 exp: m_exp ile ayni fikir (x = k*ln2 + r, Taylor(r), 2^k ile
 * olcekle) ama f32'de ve sabit dereceli (7) Horner polinomuyla --
 * softmax/capraz-entropide adim basina ~150M kez cagrildigi icin f64
 * Taylor dongusu (bolmeli) GPU'da cok yavas olurdu. |r| <= ln2/2 icin
 * kesme hatasi r^8/8! < 3e-9 (f32 epsilon'un altinda). ln2 iki parcaya
 * bolunerek (Cody-Waite) r hassas hesaplanir. */
__device__ float d_expf(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    float kf = x * 1.4426950408889634f;
    int k = (int)(kf + (kf >= 0.0f ? 0.5f : -0.5f));
    float r = x - (float)k * 0.693145751953125f;   /* ln2 yuksek parca (tam temsil edilir) */
    r = r - (float)k * 1.428606765330187e-06f;     /* ln2 dusuk parca */
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.6666666666666666e-01f + r * (4.1666666666666664e-02f
            + r * (8.3333333333333332e-03f + r * (1.3888888888888889e-03f + r * 1.9841269841269841e-04f))))));
    D32Bits s; s.u = (unsigned int)(k + 127) << 23;   /* 2^k, k in [-126, 127] */
    return p * s.f;
}

/* f32 sqrt (Adam: parametre basina bir kez): bit hilesiyle baslangic
 * tahmini + 3 Newton-Raphson iterasyonu (karesel yakinsama). */
__device__ float d_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    D32Bits c; c.f = x;
    c.u = 0x1fbd1df5u + (c.u >> 1);
    float y = c.f;
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    return y;
}

/* ================= Matris carpimi (GEMM) ================= */
/* C[M,N] = alpha * op(A)[M,K] @ op(B)[K,N] + beta * C
 *   TA=0: A saklanis [M,K] (lda)      TA=1: A saklanis [K,M] -> op(A)=A^T
 *   TB=0: B saklanis [K,N] (ldb)      TB=1: B saklanis [N,K] -> op(B)=B^T
 * blockIdx.z = batch indeksi (strideA/B/C eleman cinsinden).
 * 64x64 cikti karosu, BK=16, 16x16 thread; her thread 4x4 cikti
 * (satirlar ty+16i, sutunlar tx+16j -- paylasimli bellek okumalarinda
 * bank catismasi olmaz). Paylasimli bellek satirlari +1 dolgulu (yazma
 * sirasinda bank catismasini onler). Global okumalar her varyantta
 * ardisik thread'ler ardisik adreslere gidecek sekilde eslenir. */
#define GBM 64
#define GBN 64
#define GBK 16

template <int TA, int TB>
__device__ void gemm_body(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                          int M, int N, int K, int lda, int ldb, int ldc,
                          long long strideA, long long strideB, long long strideC,
                          float alpha, float beta) {
    __shared__ float As[GBK][GBM + 1];
    __shared__ float Bs[GBK][GBN + 1];

    A += (long long)blockIdx.z * strideA;
    B += (long long)blockIdx.z * strideB;
    C += (long long)blockIdx.z * strideC;

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * 16 + tx;
    int m0 = blockIdx.y * GBM;
    int n0 = blockIdx.x * GBN;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += GBK) {
        /* A karosu -> As[k][m] (op(A)[m0+m][k0+k]) */
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int m, k;
            if (TA == 0) { k = tid % 16; m = tid / 16 + 16 * i; }
            else         { m = tid % 64; k = tid / 64 + 4 * i; }
            int gm = m0 + m, gk = k0 + k;
            float v = 0.0f;
            if (gm < M && gk < K) v = (TA == 0) ? A[(long long)gm * lda + gk] : A[(long long)gk * lda + gm];
            As[k][m] = v;
        }
        /* B karosu -> Bs[k][n] (op(B)[k0+k][n0+n]) */
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int n, k;
            if (TB == 0) { n = tid % 64; k = tid / 64 + 4 * i; }
            else         { k = tid % 16; n = tid / 16 + 16 * i; }
            int gn = n0 + n, gk = k0 + k;
            float v = 0.0f;
            if (gn < N && gk < K) v = (TB == 0) ? B[(long long)gk * ldb + gn] : B[(long long)gn * ldb + gk];
            Bs[k][n] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GBK; k++) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[k][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[k][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int gm = m0 + ty + 16 * i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int gn = n0 + tx + 16 * j;
            if (gn >= N) continue;
            float* c = &C[(long long)gm * ldc + gn];
            *c = (beta == 0.0f) ? alpha * acc[i][j] : alpha * acc[i][j] + beta * (*c);
        }
    }
}

#define GEMM_ARGS const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, \
                  long long strideA, long long strideB, long long strideC, float alpha, float beta
#define GEMM_PASS A, B, C, M, N, K, lda, ldb, ldc, strideA, strideB, strideC, alpha, beta

extern "C" __global__ void k_gemm_nn(GEMM_ARGS) { gemm_body<0, 0>(GEMM_PASS); }
extern "C" __global__ void k_gemm_nt(GEMM_ARGS) { gemm_body<0, 1>(GEMM_PASS); }
extern "C" __global__ void k_gemm_tn(GEMM_ARGS) { gemm_body<1, 0>(GEMM_PASS); }

/* ================= Blok ici indirgeme yardimcisi ================= */
/* Sabit sirali agac indirgemesi (deterministik). blockDim.x 2'nin kuvveti
 * olmalidir; buf en az blockDim.x eleman. Sonuc TUM thread'lere doner. */
__device__ float block_sum(float v, float* buf) {
    int t = threadIdx.x;
    buf[t] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) buf[t] += buf[t + s];
        __syncthreads();
    }
    float r = buf[0];
    __syncthreads();
    return r;
}

__device__ float block_max(float v, float* buf) {
    int t = threadIdx.x;
    buf[t] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s && buf[t + s] > buf[t]) buf[t] = buf[t + s];
        __syncthreads();
    }
    float r = buf[0];
    __syncthreads();
    return r;
}

#define RED_THREADS 256

/* ================= Eleman bazli / satir bazli cekirdekler ================= */

/* x[n, c] += b[c] */
extern "C" __global__ void k_add_bias_rows(float* x, const float* b, int rows, int cols) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)rows * cols) x[i] += b[i % cols];
}

/* out = a + b */
extern "C" __global__ void k_add(const float* a, const float* b, float* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

/* Sutun toplami: out[c] += sum_r in[r*cols + c] (sabit sira).
 * Blok (32, 8): tx bir sutun, ty satirlari 8'er adimla gezer; sonra
 * ty uzerinden sabit sirali indirgeme. Bias ve norm agirligi
 * gradyanlari icin (atomik islem yok -> deterministik). */
extern "C" __global__ void k_colsum_acc(const float* in, float* out, int rows, int cols) {
    __shared__ float part[8][33];
    int tx = threadIdx.x, ty = threadIdx.y;
    int c = blockIdx.x * 32 + tx;
    float s = 0.0f;
    if (c < cols) for (int r = ty; r < rows; r += 8) s += in[(long long)r * cols + c];
    part[ty][tx] = s;
    __syncthreads();
    if (ty == 0 && c < cols) {
        float t = part[0][tx];
        for (int k = 1; k < 8; k++) t += part[k][tx];
        out[c] += t;
    }
}

/* RMSNorm ileri: out = x * r * w, r = 1/sqrt(mean(x^2)+eps). Satir basina
 * bir blok; r geri yayilim icin saklanir. */
extern "C" __global__ void k_rmsnorm_fwd(const float* x, const float* w, float* out, float* rinv, int cols, float eps) {
    __shared__ float buf[RED_THREADS];
    long long row = blockIdx.x;
    const float* xr = x + row * cols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, buf);
    float ms = ss / (float)cols;
    float r = 1.0f / (float)d_sqrt64((double)(ms + eps));
    if (threadIdx.x == 0) rinv[row] = r;
    float* orow = out + row * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = xr[i] * r * w[i];
}

/* RMSNorm geri (model/model_ops.c backward_rmsnorm ile ayni formul):
 *   dx[j] += w[j]*r*dy[j] - (r^3/cols) * x[j] * S,  S = sum dy*w*x
 *   wtmp[n, j] = dy[j]*x[j]*r   (sonra k_colsum_acc ile dw'ya toplanir) */
extern "C" __global__ void k_rmsnorm_bwd(const float* x, const float* w, const float* rinv, const float* dy,
                                         float* dx, float* wtmp, int cols) {
    __shared__ float buf[RED_THREADS];
    long long row = blockIdx.x;
    const float* xr = x + row * cols;
    const float* dyr = dy + row * cols;
    float r = rinv[row];
    float r3 = r * r * r;
    float S = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) S += dyr[i] * w[i] * xr[i];
    S = block_sum(S, buf);
    float* dxr = dx + row * cols;
    float* wt = wtmp + row * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        dxr[j] += w[j] * r * dyr[j] - (r3 / (float)cols) * xr[j] * S;
        wt[j] = dyr[j] * xr[j] * r;
    }
}

/* qkv [N,3D] -> Q,K (RoPE uygulanmis), V : [B,H,T,hd]. Thread basina
 * bir (n, h, k) cifti (k < hd/2). cos/sin: [T, hd/2]. */
extern "C" __global__ void k_qkv_rope_split(const float* qkv, const float* cos_t, const float* sin_t,
                                            float* Q, float* K, float* V, int B, int T, int H, int hd) {
    int half = hd / 2;
    int D = H * hd;
    long long total = (long long)B * T * H * half;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int k = (int)(i % half);
    long long rest = i / half;
    int h = (int)(rest % H);
    long long n = rest / H;          /* satir: b*T + t */
    int t = (int)(n % T);
    long long b = n / T;

    const float* row = qkv + n * (3LL * D);
    int c0 = h * hd + 2 * k;
    float c = cos_t[t * half + k], s = sin_t[t * half + k];
    long long o = ((b * H + h) * T + t) * hd + 2 * k;

    float q0 = row[c0], q1 = row[c0 + 1];
    Q[o] = q0 * c - q1 * s;  Q[o + 1] = q0 * s + q1 * c;
    float k0 = row[D + c0], k1 = row[D + c0 + 1];
    K[o] = k0 * c - k1 * s;  K[o + 1] = k0 * s + k1 * c;
    V[o] = row[2 * D + c0];  V[o + 1] = row[2 * D + c0 + 1];
}

/* k_qkv_rope_split'in geri yayilimi: dQ,dK (RoPE ters donusumu, bkz.
 * backward_rope), dV -> dqkv [N,3D] (uzerine YAZAR). */
extern "C" __global__ void k_qkv_rope_merge_bwd(const float* dQ, const float* dK, const float* dV,
                                                const float* cos_t, const float* sin_t,
                                                float* dqkv, int B, int T, int H, int hd) {
    int half = hd / 2;
    int D = H * hd;
    long long total = (long long)B * T * H * half;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int k = (int)(i % half);
    long long rest = i / half;
    int h = (int)(rest % H);
    long long n = rest / H;
    int t = (int)(n % T);
    long long b = n / T;

    float* row = dqkv + n * (3LL * D);
    int c0 = h * hd + 2 * k;
    float c = cos_t[t * half + k], s = sin_t[t * half + k];
    long long o = ((b * H + h) * T + t) * hd + 2 * k;

    float a0 = dQ[o], a1 = dQ[o + 1];
    row[c0] = a0 * c + a1 * s;          row[c0 + 1] = -a0 * s + a1 * c;
    float b0 = dK[o], b1 = dK[o + 1];
    row[D + c0] = b0 * c + b1 * s;      row[D + c0 + 1] = -b0 * s + b1 * c;
    row[2 * D + c0] = dV[o];            row[2 * D + c0 + 1] = dV[o + 1];
}

/* Dikkat softmax ileri, yerinde: S [BH*T, T] (ham Q.K^T) ->
 * P = softmax(S*scale + maske), maske: j > i ise -1e9 (build_causal_mask
 * ile ayni). Satir basina bir blok. */
extern "C" __global__ void k_attn_softmax_fwd(float* S, int T, float scale) {
    __shared__ float buf[RED_THREADS];
    long long row = blockIdx.x;
    int i = (int)(row % T);          /* sorgu pozisyonu */
    float* s = S + row * T;
    float mx = -3.0e38f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float v = s[j] * scale + ((j <= i) ? 0.0f : -1e9f);
        s[j] = v;
        if (v > mx) mx = v;
    }
    mx = block_max(mx, buf);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float e = d_expf(s[j] - mx);
        s[j] = e;
        sum += e;
    }
    sum = block_sum(sum, buf);
    for (int j = threadIdx.x; j < T; j += blockDim.x) s[j] /= sum;
}

/* Dikkat softmax geri, yerinde: dP -> dS_on = P*(dP - <dP,P>) * scale
 * (softmax geri + node_scale geri; maske sabit, gradyani gecirir). */
extern "C" __global__ void k_attn_softmax_bwd(const float* P, float* dP, int T, float scale) {
    __shared__ float buf[RED_THREADS];
    long long row = blockIdx.x;
    const float* p = P + row * T;
    float* d = dP + row * T;
    float dot = 0.0f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) dot += d[j] * p[j];
    dot = block_sum(dot, buf);
    for (int j = threadIdx.x; j < T; j += blockDim.x) d[j] = p[j] * (d[j] - dot) * scale;
}

/* O [B,H,T,hd] -> out [N, H*hd] (basliklarin sutun-birlestirmesi). */
extern "C" __global__ void k_heads_merge(const float* O, float* out, int B, int T, int H, int hd) {
    long long total = (long long)B * T * H * hd;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int d = (int)(i % hd);
    long long rest = i / hd;
    int h = (int)(rest % H);
    long long n = rest / H;
    int t = (int)(n % T);
    long long b = n / T;
    out[n * (long long)(H * hd) + h * hd + d] = O[((b * H + h) * T + t) * hd + d];
}

/* k_heads_merge'in tersi: dout [N, H*hd] -> dO [B,H,T,hd]. */
extern "C" __global__ void k_heads_split(const float* dout, float* dO, int B, int T, int H, int hd) {
    long long total = (long long)B * T * H * hd;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int d = (int)(i % hd);
    long long rest = i / hd;
    int h = (int)(rest % H);
    long long n = rest / H;
    int t = (int)(n % T);
    long long b = n / T;
    dO[((b * H + h) * T + t) * hd + d] = dout[n * (long long)(H * hd) + h * hd + d];
}

/* SwiGLU ileri: gu [N, 2F] (gate | up) -> h [N, F] = silu(g) * u. */
extern "C" __global__ void k_swiglu_fwd(const float* gu, float* h, int rows, int F) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * F) return;
    long long n = i / F;
    int j = (int)(i % F);
    float g = gu[n * 2 * F + j];
    float u = gu[n * 2 * F + F + j];
    float sg = 1.0f / (1.0f + d_expf(-g));
    h[i] = g * sg * u;
}

/* SwiGLU geri: dh [N,F] -> dgu [N,2F] (uzerine YAZAR).
 *   dg = dh*u*(sg + g*sg*(1-sg)),  du = dh*g*sg
 * (feedforward.c'deki node_mul/node_sigmoid zincirinin turevi). */
extern "C" __global__ void k_swiglu_bwd(const float* gu, const float* dh, float* dgu, int rows, int F) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * F) return;
    long long n = i / F;
    int j = (int)(i % F);
    float g = gu[n * 2 * F + j];
    float u = gu[n * 2 * F + F + j];
    float sg = 1.0f / (1.0f + d_expf(-g));
    float d = dh[i];
    float dsilu = d * u;
    dgu[n * 2 * F + j] = dsilu * sg + dsilu * g * sg * (1.0f - sg);
    dgu[n * 2 * F + F + j] = d * g * sg;
}

/* Gomme ileri: x[n,:] = E[ids[n],:] */
extern "C" __global__ void k_embed_fwd(const float* E, const unsigned int* ids, float* x, int rows, int D) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * D) return;
    long long n = i / D;
    int j = (int)(i % D);
    x[i] = E[(long long)ids[n] * D + j];
}

/* Gomme geri: dE[ids[n], j] += dx[n, j]. Ayni token birden fazla satirda
 * gecebildigi icin satirlar paralel islenemez (atomik gerekirdi ->
 * determinizm kaybolurdu); bunun yerine thread basina bir SUTUN j ve
 * satirlar sabit sirayla (n = 0..rows-1) gezilir. */
extern "C" __global__ void k_embed_bwd(const float* dx, const unsigned int* ids, float* dE, int rows, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    for (int n = 0; n < rows; n++) dE[(long long)ids[n] * D + j] += dx[(long long)n * D + j];
}

/* Capraz-entropi, satir basina bir blok, logits YERINDE gradyana doner:
 *   loss_rows[n] = log(sum exp(z - max)) + max - z[t]
 *   z[j] <- (softmax(z)[j] - [j==t]) * grad_scale
 * (model/loss.c ile ayni formul; grad_scale = 1/(B*T): dizi-ici
 * ortalama x worker ortalamasi). */
extern "C" __global__ void k_cross_entropy(float* logits, const unsigned int* targets, float* loss_rows,
                                           int V, float grad_scale) {
    __shared__ float buf[RED_THREADS];
    long long row = blockIdx.x;
    float* z = logits + row * (long long)V;
    unsigned int t = targets[row];

    float mx = -3.0e38f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) if (z[j] > mx) mx = z[j];
    mx = block_max(mx, buf);

    float sum = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) sum += d_expf(z[j] - mx);
    sum = block_sum(sum, buf);

    float zt = z[t];
    __syncthreads(); /* z[t] herkes tarafindan okunmadan yazilmasin */
    if (threadIdx.x == 0) loss_rows[row] = (float)d_log64((double)sum) + mx - zt;

    float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float p = d_expf(z[j] - mx) * inv;
        if (j == (int)t) p -= 1.0f;
        z[j] = p * grad_scale;
    }
}

/* Adam (training/adam.c adam_update_one ile ayni formul), duz dizi. */
extern "C" __global__ void k_adam(float* p, const float* g, float* m, float* v, long long n,
                                  float lr, float b1, float b2, float eps, float bc1, float bc2) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.0f - b1) * gi;
    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    p[i] -= lr * m_hat / (d_sqrtf(v_hat) + eps);
}
