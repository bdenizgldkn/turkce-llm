/* Katman 7/8 (devam) - CUDA Entegrasyonu, V1.
 *
 * Kapsam: transformer hesaplamasinin buyuk cogunlugunu olusturan
 * MATRIS CARPIMINI (matmul) GPU'ya tasimak -- CPU tensor/otograd
 * sistemiyle (Katman 3-4) ayni arayuzu koruyarak (Tensor hala CPU'da
 * yasar, hesaplama GPU'ya "gidip gelir": host->device kopya, cekirdek,
 * device->host kopya). Digger islemler (softmax, RMSNorm, RoPE, vb.)
 * V1'de hala CPU'da -- bu, en yuksek etkili optimizasyonla (matmul,
 * FLOP'larin buyuk cogunlugu) baslayan, dogrulugu once kanitlayan
 * kademeli bir yaklasimdir (bkz. PROJE_PLANI.md).
 *
 * Basitlik icin tek bir global CUDA baglami/cekirdek tutamaci
 * kullanilir (ayni anda birden fazla GPU baglami yonetmek V1
 * kapsaminda degildir). */
#ifndef MODEL_GPU_OPS_H
#define MODEL_GPU_OPS_H

#include "../autograd/node.h"
#include "../runtime/memory.h"

/* CUDA baglamini baslatir ve kernels.ptx'i yukler. Program basinda BIR
 * KEZ cagrilmalidir. scratch, PTX dosyasini okumak icin kullanilir. */
void gpu_ops_init(Allocator* scratch, const char* ptx_path);
void gpu_ops_shutdown(void);

/* KATMAN 17 (Aday 1) - Thread-basina BAGIMSIZ CUDA baglami.
 *
 * gpu_ops_init tek bir "varsayilan" baglam kurar (mutex ile PAYLASILAN,
 * dolayisiyla cok-thread'li kullanimda kilit cekismesi/contention olusur
 * -- bkz. PROJE_PLANI.md Bolum 16). Bunun yerine, num_workers kadar
 * BAGIMSIZ baglam (her biri kendi cuCtxCreate_v2 + kendi PTX modul
 * kopyasi) onceden (egitim donguse baslamadan) kurulur -- boylece
 * calisirken hicbir baglam OLUSTURMA maliyeti odenmez, sadece
 * cuCtxSetCurrent (ucuz) cagrilir. Her worker thread'i, kendi
 * gpu_ops_set_worker_id(0..num_workers-1) cagrisiyla HANGI bagimsiz
 * baglami kullanacagini bildirir (thread-lokal depolama/TLS ile --
 * fonksiyon parametresi olarak node_matmul2d_gpu'ya kadar tasimaya
 * gerek kalmadan). Bu sayede o thread'in TUM GPU cagrilari icin HICBIR
 * KILIT gerekmez (her worker'in KENDI baglami vardir, cakisma yok) --
 * GPU donanim/surucu zamanlayicisi kendi ince taneli sirali erisimini
 * kendi yapar. */
void gpu_ops_init_workers(Allocator* scratch, const char* ptx_path, u32 num_workers);

/* Bu OS thread'inin (cagiran thread) hangi bagimsiz baglami (bkz.
 * gpu_ops_init_workers) kullanacagini bildirir. worker_id, [0,num_workers)
 * araliginda olmalidir. -1 (varsayilan) = paylasilan/mutex'li varsayilan
 * baglam (gpu_ops_init'in kurdugu). Egitim workerinin (bkz.
 * training/data_parallel.c) calismaya basladigi ANDA, kendi thread'i
 * uzerinde bir kez cagirmasi yeterlidir. */
void gpu_ops_set_worker_id(i32 worker_id);

/* node_matmul2d ile AYNI matematiksel sonucu, hesaplamayi GPU'da
 * yaparak uretir (ileri VE geri yayilim GPU matmul cekirdegini kullanir). */
Node* node_matmul2d_gpu(Allocator* alloc, Node* a, Node* b);

/* TANILAMA: her gpu_matmul cagrisinin (alloc+H2D+cekirdek+D2H+free+sync)
 * TOPLAM duvar-saati suresini ve cagri sayisini biriktirir. Darbogazin
 * GPU gidis-gelisinde mi yoksa CPU tarafinda mi oldugunu olcmek icin. */
void gpu_ops_debug_stats(f64* out_total_seconds, u32* out_call_count);
void gpu_ops_debug_reset(void);

#endif /* MODEL_GPU_OPS_H */
