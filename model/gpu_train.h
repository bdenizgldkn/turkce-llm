/* Katman 21 - GPU'da Tutulan (GPU-resident) Egitim.
 *
 * Katman 7-19'daki egitim yolu (training/data_parallel.c) tensorleri
 * CPU'da tutar, sadece matris carpimlari icin GPU'ya gidip gelir ve
 * B diziyi B ayri CPU thread'inde ayri ayri isler. Linux/L4'te
 * olculdugunde (PROJE_PLANI.md Bolum 20) GPU %23 mesgul, CPU zamani ise
 * cogunlukla bellek doldurma/kopyalamaya gidiyordu.
 *
 * Bu katman AYNI modeli (lm_model.c: gomme + L x [RMSNorm, RoPE'li
 * dikkat, RMSNorm, SwiGLU] + son RMSNorm + bagli cikis projeksiyonu) ve
 * AYNI kaybi, B diziyi TEK bir [B*T, D] tensorde birlestirerek TAMAMEN
 * GPU uzerinde hesaplar: agirliklar, gradyanlar, Adam durumu ve
 * aktivasyonlar adimlar arasinda GPU belleginde kalir. Geri yayilim
 * otograd grafi yerine elle yazilmis turevlerle yapilir; dogrulugu CPU
 * otograd sistemi referans alinarak kanitlanir (tests/test_gpu_train.c).
 *
 * CPU tarafindaki LMModel/AdamOptimizer, checkpoint formatinin ve
 * devam etme mantiginin DEGISMEMESI icin "ayna" olarak kullanilir:
 * gpu_trainer_upload ile GPU'ya yuklenir, gpu_trainer_download ile geri
 * alinir (checkpoint oncesi). */
#ifndef MODEL_GPU_TRAIN_H
#define MODEL_GPU_TRAIN_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../cuda/cuda_backend.h"
#include "lm_model.h"
#include "../training/adam.h"

#define GPU_TRAIN_MAX_PARAMS (LM_MAX_LAYERS * 10u + 2u)
#define GPU_TRAIN_NORM_BLOCKS 1024u

typedef struct GpuLayerActs {
    CUdeviceptr x_in;   /* [N,D] katman girdisi (residual) */
    CUdeviceptr n1;     /* [N,D] */
    CUdeviceptr r1;     /* [N]   1/rms */
    CUdeviceptr Q, K, V;/* [B,H,T,hd] (Q,K RoPE uygulanmis) */
    CUdeviceptr P;      /* [B,H,T,T] softmax */
    CUdeviceptr att;    /* [N,D] basliklar birlestirilmis dikkat ciktisi (wo'dan once) */
    CUdeviceptr x2;     /* [N,D] */
    CUdeviceptr n2;     /* [N,D] */
    CUdeviceptr r2;     /* [N] */
    CUdeviceptr gu;     /* [N,2F] */
    CUdeviceptr h;      /* [N,F] */
} GpuLayerActs;

typedef struct GpuTrainer {
    CudaCtx cuda;
    CUfunction k_gemm_nn, k_gemm_nt, k_gemm_tn;
    CUfunction k_add_bias_rows, k_add, k_colsum_acc;
    CUfunction k_rmsnorm_fwd, k_rmsnorm_bwd;
    CUfunction k_qkv_rope_split, k_qkv_rope_merge_bwd;
    CUfunction k_attn_softmax_fwd, k_attn_softmax_bwd;
    CUfunction k_heads_merge, k_heads_split;
    CUfunction k_swiglu_fwd, k_swiglu_bwd;
    CUfunction k_embed_fwd, k_embed_bwd;
    CUfunction k_cross_entropy, k_adam, k_sumsq_partial;

    u32 V, D, H, L, F, T, B, N, hd;
    f32 eps;

    u32 num_params;
    u64 offsets[GPU_TRAIN_MAX_PARAMS]; /* eleman cinsinden, lm_collect_params sirasi */
    u64 numels[GPU_TRAIN_MAX_PARAMS];
    u64 total_scalars;
    CUdeviceptr params, grads, adam_m, adam_v; /* her biri total_scalars f32 */

    GpuLayerActs layers[LM_MAX_LAYERS];
    CUdeviceptr x_final, nf, rf, logits; /* logits [N,V]: capraz-entropiden sonra gradyan */

    /* geri yayilim karalama tamponlari (katmanlar arasinda paylasilir) */
    CUdeviceptr dres, dn, wtmp, qkv, dqkv, proj, datt;
    CUdeviceptr dQ, dK, dV, dO, Obuf, dP, dh, dgu;

    CUdeviceptr ids, targets, loss_rows, cos_t, sin_t;
    f32* host_loss_rows; /* [N] */
    CUdeviceptr norm_partial; /* [GPU_TRAIN_NORM_BLOCKS] */
    f32* host_norm_partial;
} GpuTrainer;

/* model'in mimarisiyle (V, D, H, L, F, eps) B dizi x T token'lik adimlar
 * icin GPU egiticisini kurar; kendi CUDA baglamini (ptx_path =
 * cuda/train_kernels.ptx) ve tum GPU tamponlarini bir kez ayirir.
 * params: lm_collect_params sirasiyla (num_params = 10*L + 2). */
GpuTrainer gpu_trainer_create(Allocator* alloc, const char* ptx_path, const LMModel* model,
                              Node** params, u32 num_params, u32 batch_size, u32 seq_len);
void gpu_trainer_destroy(GpuTrainer* g);

/* CPU aynasi -> GPU: parametre degerleri, opt != NULL ise Adam m/v.
 * (opt->t CPU'da tutulur.) */
void gpu_trainer_upload(GpuTrainer* g, Node** params, const AdamOptimizer* opt);
/* GPU -> CPU aynasi (checkpoint oncesi): parametre degerleri, opt != NULL ise m/v. */
void gpu_trainer_download(GpuTrainer* g, Node** params, AdamOptimizer* opt);
/* GPU gradyanlarini params[i]->grad'a yazar (test/dogrulama icin). */
void gpu_trainer_download_grads(GpuTrainer* g, Node** params);
/* params[i]->grad -> GPU gradyanlari (test: Adam'i ayni gradyanlarla sinamak icin). */
void gpu_trainer_upload_grads(GpuTrainer* g, Node** params);

/* training/data_parallel.c'deki worker ornekleme formuluyle AYNI B
 * pencereyi secer (dizi w: pcg_seed(seed_base*1000003+w, 99+w)) --
 * boylece CPU yoluyla AYNI adim AYNI verileri gorur. ids/targets: [B*T]. */
void gpu_trainer_sample_batch(const u32* tokens, u64 num_tokens, u32 seq_len, u32 batch_size,
                              u64 seed_base, u32* ids_out, u32* targets_out);

/* Gradyanlari sifirlar, ileri + geri yayilimi calistirir; ortalama kaybi
 * dondurur (dizi-ici ortalama, sonra diziler uzerinden ortalama --
 * data_parallel_step ile ayni tanim). Gradyanlar da ayni ortalamadir. */
f32 gpu_trainer_forward_backward(GpuTrainer* g, const u32* ids, const u32* targets);

/* Sadece ileri yayilim + kayip (gradyanlara dokunmaz): dogrulama kaybi
 * icin. Ayni girdiyle gpu_trainer_forward_backward'in dondurdugu kayipla
 * BIT BIT aynidir (ayni ileri yol). */
f32 gpu_trainer_eval_loss(GpuTrainer* g, const u32* ids, const u32* targets);

/* Mevcut gradyanlarin global L2 normu (deterministik). */
f64 gpu_trainer_grad_norm(GpuTrainer* g);

/* opt->t'yi artirip GPU'da Adam adimi atar (hiperparametreler opt'tan). */
void gpu_trainer_adam_step(GpuTrainer* g, AdamOptimizer* opt);

/* Ayni, ama gradyanlar once grad_scale ile carpilir (gradyan kirpma:
 * grad_scale = min(1, max_norm / norm)). grad_scale=1 -> adam_step ile
 * birebir ayni. */
void gpu_trainer_adam_step_scaled(GpuTrainer* g, AdamOptimizer* opt, f32 grad_scale);

#endif /* MODEL_GPU_TRAIN_H */
