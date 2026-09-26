# Türkçe LLM — Sıfırdan Eğitim Projesi Planı

Son güncelleme: 2026-09-24

## 0. Proje Özeti

- **Amaç:** Türkçe anlama, karar verme ve yorum yapma yeteneğine sahip, tamamen sıfırdan eğitilmiş temel (base) bir dil modeli oluşturmak.
- **Yaklaşım:** Gerçek anlamda sıfırdan — hiçbir 3. parti kütüphane (NumPy, PyTorch, BLAS, cuBLAS, cuDNN, HuggingFace vb.) ve hatta dilin standart kütüphanesindeki hazır algoritmalar (math.h'nin exp/log/sqrt'ı, rand() vb.) bile kullanılmayacak. Bellek yönetimi, matematik fonksiyonları, rastgele sayı üreteci, tensor/dizi yapısı, otograd (otomatik türev) motoru, GPU çekirdekleri, tokenizer, model mimarisi ve eğitim döngüsü — hepsi satır satır bizim tarafımızdan yazılacak.
- **Ekip:** Tek kişi (sen) + yapay zeka asistanları.
- **Zaman çizelgesi:** Esnek, sabit bir teslim tarihi yok — öğrenme ve doğru ilerleme hızdan önemli.

## 1. Temel İlkeler ve Kısıtlar (Netleşen Kurallar)

Bu kurallar tüm proje boyunca geçerli, geriye dönüp sorgulanmayacak temel çerçevedir:

| Konu | Kural |
|---|---|
| Dil | **C + ham CUDA C**. Derleyici (gcc/nvcc) bir "araç" olarak kabul edilir, kütüphane değildir. |
| 3. parti kütüphaneler | **Tamamen yasak.** NumPy, PyTorch, BLAS/OpenBLAS, cuBLAS, cuDNN, Thrust, HuggingFace, vb. hiçbiri kullanılmayacak. |
| Dilin standart kütüphanesi | **Algoritma içeren kısımları yasak.** `exp`, `log`, `sqrt`, `sin`, `cos`, `pow`, `rand()` gibi hazır matematik/rastgelelik fonksiyonları kullanılmayacak — kendimiz implemente edeceğiz. |
| Syscall / OS arayüzü | **İstisna, serbest.** `mmap`/`brk` (bellek), `open`/`read`/`write`/`close` (dosya I/O) gibi ham sistem çağrıları kullanılabilir. Bunlar birer "algoritma" değil, donanım/OS ile aramızdaki zorunlu geçiş noktalarıdır. Kendi bellek ayırıcımızı (allocator) ve dosya I/O yardımcılarımızı bunların üzerine biz inşa edeceğiz. |
| GPU / CUDA | **Ham CUDA C ile kendi çekirdeklerimiz (kernel), CUDA Driver API üzerinden.** nvcc bir derleyici/araçtır; çekirdekler PTX/cubin olarak derlenip **CUDA Driver API** (`cuMemAlloc`, `cuMemcpyHtoD`/`DtoH`, `cuModuleLoad`, `cuLaunchKernel` vb.) ile yüklenip çalıştırılacak — daha üst seviyeli, kolaylaştırılmış CUDA Runtime API (`cudaMalloc` vb.) kullanılmayacak. Driver API çağrıları syscall'a eşdeğer kabul edilir (donanım sürücüsüyle aramızdaki zorunlu geçiş noktası). `cuBLAS`, `cuDNN`, `Thrust` gibi **hazır algoritma** kütüphaneleri kesinlikle yasak. |
| Veri toplama | **Serbest.** Halka açık ham metin verisi (Wikipedia dump, haber arşivleri vb.) indirilip kullanılabilir. Yasak olan yazılım/altyapı kullanımıdır, veri değil. |

**Sonuç:** Bu proje aslında iki katmanlı: (A) Kendi ürettiğimiz minimal bir "derin öğrenme çatısı" (bellek yönetimi → matematik → tensor → otograd → CUDA çekirdekleri) ve (B) bu çatının üzerine inşa edilen LLM (tokenizer → model → eğitim). (A) katmanı, (B) katmanından önce gelmeli ve muhtemelen zaman olarak daha büyük bir kısmı kaplayacak.

## 2. Donanım ve Gerçekçi Kısıtlar

**Mevcut donanım:**
- GPU: RTX 4050 (laptop, 6 GB VRAM)
- CPU: Intel i5, 12. nesil
- RAM: 16 GB (teyit edildi)

**Bu donanımın ve "sıfırdan" kuralının birlikte anlamı:**
- Hazır BLAS/cuBLAS olmadan yazılan matris çarpımı, optimize edilmiş kütüphanelere göre önemli ölçüde daha yavaş olacaktır (CPU tarafında). Bu yüzden CUDA çekirdeklerinin doğru yazılması (paylaşımlı bellek/shared memory kullanımı, coalesced memory access gibi temel GPU optimizasyon teknikleri) performans açısından kritik olacak — bunları da literatürden öğrenip kendimiz uygulayacağız (bir kütüphane çağırmak yerine).
- 6 GB VRAM sınırı hâlâ geçerli: sayısal hassasiyet **sadece float32** olarak sabitlendiği için mixed precision kapsam dışı; bunun yerine gradient checkpointing ve gradient accumulation gibi bellek tasarrufu tekniklerini **kendi otograd/CUDA motorumuzda kendimiz** implemente etmemiz gerekecek.
- Debugging (hata ayıklama) hazır araçlar (örn. PyTorch'un otomatik hata mesajları, NaN tespiti vb.) olmadan çok daha zor olacak — kendi sayısal doğrulama/test araçlarımızı yazmamız gerekecek.

## 3. Sıfırdan İnşa Edilecek Katmanlar (Alt Yapı Yığını)

Bu proje, LLM'e başlamadan önce kendi "derin öğrenme çatımızı" inşa etmeyi gerektiriyor. Katmanlar, her biri bir öncekinin üzerine kurulacak şekilde sıralı:

**Katman 0 — Sistem Arayüzü [TAMAMLANDI ✓ — 2026-09-24]**
- Windows'ta olduğumuz için `mmap`/`brk`/`open`/`read`/`write` yerine Win32 karşılıkları kullanıldı: `VirtualAlloc`/`VirtualFree` (bellek), `CreateFileA`/`ReadFile`/`WriteFile`/`CloseHandle` (dosya I/O) — bunlar `<windows.h>` dahil edilmeden, sadece ihtiyaç duyulan fonksiyon imzaları kendimiz bildirilerek (`runtime/win32_syscalls.h`) kullanıldı.
- Arena (bump-pointer) ayırıcı: `runtime/memory.c` — reserve/commit ayrımı, hizalama, reset/destroy
- Genel amaçlı ayırıcı: free-list tabanlı, first-fit arama, blok bölme (split) ve komşu blok birleştirme (coalescing) dahil
- Dosya I/O sarmalayıcı: `runtime/file_io.c` — büyük dosyalar için parçalı (chunked) okuma/yazma, `file_read_entire` yardımcı fonksiyonu
- Minimum konsol çıktısı: `runtime/console.c` (test/tanılama mesajları için, printf yok)
- Doğrulama: `tests/test_layer0.c` — **115/115 test başarılı** (arena tahsisi, commit sınırı aşımı, reset/destroy, free-list yeniden kullanımı, coalescing, dosya round-trip)

**Katman 1 — Temel Matematik Kütüphanesi [TAMAMLANDI ✓ — 2026-09-24]**
- `exp`, `log`, `sqrt`, `pow`, `sin`, `cos`, `tanh` — `runtime/mathlib.c`
- **Yöntem: Doğruluk öncelikli.** `sqrt`: bit-manipülasyonla aralık indirgeme + Newton-Raphson. `exp`: aralık indirgeme (x=k·ln2+r) + Taylor serisi + ldexp-benzeri ölçekleme. `log`: IEEE-754 üstel/mantis ayrıştırma + atanh-tabanlı seri. `sin`/`cos`: kadran indirgeme (mod π/2) + Taylor serisi. `tanh`: `exp` üzerinden, taşma korumalı. `pow`: `exp(y·log(x))` + negatif taban/tam sayı üs özel durumları.
- Doğruluk testleri: `tests/test_mathlib.c` — **36/36 test başarılı** (bilinen matematiksel referans değerlerle, 1e-9 ile 1e-12 arası toleranslarla)

**Katman 2 — Rastgele Sayı Üreteci (PRNG) [TAMAMLANDI ✓ — 2026-09-24]**
- `runtime/prng.c`: **PCG32** (pcg_setseq_64_xsh_rr_32 varyantı) — literatürden algoritma, kod bizim tarafımızdan yazıldı
- Box-Muller dönüşümü ile standart normal (Gauss) dağılım üretimi (ikinci değer önbelleklenir), Fisher-Yates shuffle
- Doğrulama: `tests/test_prng.c` — **8/8 test başarılı** (üretilebilirlik/reproducibility, uniform ve Gauss dağılım momentleri istatistiksel olarak doğrulandı, shuffle'ın geçerli bir permütasyon ürettiği kontrol edildi)

**Katman 3 — Tensor / Çok Boyutlu Dizi Yapısı [TAMAMLANDI ✓ — 2026-09-24]**
- `tensor/tensor.c`: shape/stride tabanlı, satır-majör float32 tensor. Transpoze/reshape veri kopyalamayan "view" döndürür (PyTorch benzeri); `tensor_contiguous` ile somutlaştırma
- Elemanter işlemler (add/sub/mul/scale, V1: broadcasting yok, aynı shape gerekir), 2B matmul (bitişik olmayan/transpoze girdilerle de doğru çalışıyor)
- Doğrulama: `tests/test_tensor.c` — **15/15 test başarılı**

**Katman 4 — Otograd (Otomatik Türev) Motoru [TAMAMLANDI ✓ — 2026-09-24]**
- `autograd/node.c`: hesaplama grafiği (Node = value + grad + ebeveynler + backward_fn), DFS ile topolojik sıralama, ters sırada geri yayılım, çoklu-kullanım (DAG) durumunda gradyan **biriktirme** (+=)
- Desteklenen işlemler: add, sub, mul, scale, matmul2d, transpose, reshape, sum_all, relu, sigmoid, tanh, softmax (son eksen)
- Doğrulama: `tests/test_autograd.c` — **sayısal gradyan kontrolü** (kararlaştırılan yöntem, `(f(x+e)-f(x-e))/2e`), **34/34 test başarılı** (add/mul/matmul/transpose/sigmoid/relu/softmax gradyanları doğrulandı)
- "Kendi mini-PyTorch'umuzun" kalbi artık çalışıyor ve doğrulanmış durumda.

**Katman 5 — CUDA Çekirdekleri (GPU) [TAMAMLANDI ✓ — 2026-09-24]**
- `cuda/kernels.cu`: `k_add`, `k_mul`, `k_relu`, `k_matmul` (naif, satır-majör), `k_reduce_sum` (shared-memory blok içi paralel indirgeme) — hepsi ham CUDA C
- `cuda/driver_api.h`: CUDA Driver API'nin minimal bildirimleri (`<cuda.h>` dahil edilmedi, win32_syscalls.h ile aynı felsefe). `cuda.lib`'in klasik bir DLL import-library değil, doğrudan bağlanabilir "stub" sembolleri içerdiği keşfedildi — `__declspec(dllimport)` yerine düz `extern` bildirimle bağlandı.
- `cuda/cuda_backend.c`: Driver API üzerine sarmalayıcı (context/modül yönetimi, grid/block hesaplama, hata kontrolü)
- Doğrulama: `tests/test_cuda.c` — **gerçek RTX 4050 donanımında 3/3 test başarılı** (k_add, k_matmul, k_reduce_sum; CPU'da hesaplanan referans değerlerle karşılaştırıldı)
- **Karşılaşılan ve çözülen zincirleme araç sorunları** (bkz. İlerleme Günlüğü): nvcc Windows'ta MSVC gerektiriyor → Visual Studio'ya C++ bileşeni eklendi (VS 2026'da workload ID'si `NativeDesktop` olarak değişmiş) → CUDA 13.4'ün ürettiği PTX eski sürücüyle uyumsuzdu → GPU sürücüsü 617.14'e güncellendi. Sonuç: tüm araç seti artık uyumlu ve doğrulanmış durumda.

**Katman 6 — Türkçe Morfolojik Analizör + Tokenizer**

Saf istatistiksel BPE yerine, Türkçe'nin eklemeli (agglutinative) yapısına uygun **tam morfolojik analiz** yaklaşımı seçildi. Bu, tokenizer'ı tek başına bir katman olmaktan çıkarıp ayrı bir dilbilim alt sistemine dönüştürüyor — **proje kapsamını ciddi şekilde büyüten bir karardır**, ayrıca not edilmiştir (bkz. Riskler).

Alt bileşenler:
- **Kök sözlüğü:** Türkçe isim/fiil/sıfat köklerinin geniş bir listesi — halka açık bir sözlük/kelime dökümü (örn. Türkçe Vikisözlük) veri olarak indirilecek, kök çıkarma kendi kodumuzla yapılacak
- **Ek tablosu ve kurallar:** **tüm envanter** — çekim ekleri (hal ekleri, iyelik ekleri, çoğul, fiil çekimleri: zaman/kip/şahıs) ve yapım ekleri, nadir/arkaik ekler dahil; **büyük/küçük ünlü uyumu** ve **ünsüz yumuşaması/sertleşmesi** (kitap→kitabı gibi) kurallarıyla birlikte
- **Morfolojik ayrıştırıcı:** bir kelimeyi kök+ek dizisi olarak analiz eden algoritma (sonlu durum makinesi/finite-state transducer mantığıyla, kendi C kodumuzla — hazır bir FST kütüphanesi kullanılmadan)
- **Belirsizlik giderme (disambiguation):** bir kelime için birden fazla geçerli ayrıştırma çıktığında, korpüs üzerinden çıkarılan n-gram/HMM benzeri geçiş olasılıklarıyla bağlama göre en olası ayrışımı seçen istatistiksel bir model (kendi yazacağımız)
- **Fallback (BPE'ye düşme):** Kök sözlüğünde bulunamayan/analiz edilemeyen kelimeler (özel isimler, yabancı kelimeler, yazım hataları, argo) için BPE tabanlı alt-kelime ayrımına düşülecek — bu yüzden BPE implementasyonu da (Katman 6'nın bir parçası olarak) hâlâ gerekli
- **Vocab boyutu: 32.000** (kök + ek + BPE fallback token'larının toplamı bu hedefe göre dengelenecek)

**Katman 7 — Model Mimarisi [TAMAMLANDI ✓ (V1 — batch'siz) — 2026-09-24]**
- Katman 3-5 üzerine inşa edilen Transformer bileşenleri: self-attention, feed-forward
- **Normalizasyon: RMSNorm** (LayerNorm'a göre daha az işlem, daha basit backward türevi) — `tensor_rmsnorm` + elle türetilmiş backward (`model/model_ops.c`)
- **Positional encoding: RoPE** (Rotary Positional Embedding — `model/rope.c`, kendi `m_cos`/`m_sin` fonksiyonlarımızla; rotasyon normu koruma özelliği test edildi)
- **Yeni temel otograd işlemleri** (`model/model_ops.c`): `node_add_bias` (satır bazlı yayılma), `node_slice_cols` (view-tabanlı, multi-head ayırma için), `node_concat_cols2` (multi-head birleştirme)
- **Çoklu-başlıklı nedensel (causal) self-attention** (`model/attention.c`): RoPE'li Q/K, ölçeklenmiş nokta çarpımı, nedensel maske, softmax, V ile çarpım, başlıkları birleştirip çıkış projeksiyonu
- **SwiGLU ileri besleme** (`model/feedforward.c`, LLaMA tarzı): `W_down(silu(W_gate(x)) * W_up(x))`, `silu` mevcut sigmoid+mul'dan bileşim
- **Tam Transformer bloğu** (`model/transformer_block.c`, pre-norm): `x = x + Attention(RMSNorm(x))`; `x = x + SwiGLU(RMSNorm(x))`
- **V1 sınırlaması:** Tek dizi (batch boyutu yok) — mimari doğruluğu önce kanıtlanıp, gerçek eğitimde ihtiyaç duyulacak batch desteği (ya döngüyle ya da gerçek batched tensor operasyonlarıyla) ayrı bir geliştirme adımı olarak eklenecek. Ayrıca bu katman hâlâ **CPU üzerinde** çalışıyor — CUDA çekirdeklerinin (Katman 5) otograd sistemine bağlanması (GPU'da gerçek eğitim) ayrı bir entegrasyon adımı.
- Doğrulama: `tests/test_model_ops.c` (**30/30**, yeni op'ların sayısal gradyan kontrolü) + `tests/test_transformer_block.c` (**40/40**, küçük ölçekli ama TAM bir Transformer bloğunun uçtan uca sayısal gradyan kontrolü — d_model=8, 2 başlık, d_ff=16, seq=4). İlk denemede (bir `tensor_slice_cols` `numel` güncellemesi hatası dışında, hemen yakalanıp düzeltildi) tüm kontroller geçti.

**Katman 8 — Optimizer ve Eğitim Döngüsü**
- Adam (ya da seçilecek başka bir optimizer) algoritmasının kendi implementasyonu
- Checkpoint kaydetme/yükleme (kendi ikili dosya formatımızla, Katman 0'ın dosya I/O'sunu kullanarak)

**Not:** Bu katmanların her biri ayrı ayrı test edilmeli (örn. otograd motorunun ürettiği türevlerin sayısal olarak doğru olduğunu küçük örneklerle doğrulamak — "gradient checking"). Bu, ilerleyen adımlarda ayrı bir "Test Stratejisi" bölümünde detaylandırılacak.

## 4. Hedef Model Ölçeği

- 6 GB VRAM ve sıfırdan yazılmış (optimize BLAS/cuBLAS'sız) çekirdekler göz önüne alındığında, gerçekçi hedef **30M - 150M parametre** aralığıdır (kesin sayı, CUDA çekirdeklerimizin gerçek performansı ölçüldükten sonra netleşecek).
- **Faz 0 (Altyapı):** Katman 0-8'in inşası ve doğrulanması. Süre öngörülemez — bu proje boyunca en çok zaman alacak kısım muhtemelen burasıdır.
- **Faz 1 (Küçük Ölçek Doğrulama, ONAYLANDI):** ~1-5M parametre (4-6 katman, d_model 128-256), küçük bir veri alt kümesiyle uçtan uca pipeline'ın çalıştığını doğrulama.
- **Faz 2 (Asıl Eğitim, ONAYLANDI):** ~30-150M parametre. Faz 1 tamamlandıktan ve CUDA çekirdeklerinin gerçek performansı ölçüldükten sonra bu aralıkta kesin bir sayı (katman sayısı, d_model, head sayısı) netleştirilecek.

## 5. Veri Toplama ve Hazırlama

**Kaynaklar (karar verildi — dördü de kullanılacak, indirilmesi serbest):**
- Türkçe Wikipedia dump — temiz/yapılandırılmış, ilk pipeline testi için başlangıç kaynağı
- OSCAR / mC4 Türkçe alt kümesi — asıl veri hacmini sağlayacak ana kaynak (gürültülü, ciddi temizlik gerektirir)
- Türkçe haber arşivleri — güncel, çeşitli konu başlıkları (kaynak/lisans durumu siteye göre kontrol edilecek)
- Türkçe altyazı korpüsleri (OpenSubtitles-TR vb.) — konuşma diline yakın çeşitlilik (lisans/kullanım kısıtı kontrol edilecek)

**İşlem adımları (hepsi kendi C kodumuzla):**
1. Ham veri indirme
2. Dil tespiti ve filtreleme
3. Tekrar eden içerik temizliği (deduplication)
4. Kalite filtreleme
5. Unicode/Türkçe karakter normalizasyonu (İ/ı, ş, ğ, ç, ö, ü)
6. Train/validation ayrımı

## 6. Değerlendirme

- Validation set üzerinde loss/perplexity takibi (kendi hesaplama kodumuzla)
- Üretilen metinlerin elle incelenmesi (dilbilgisi, tutarlılık, anlam bütünlüğü)

## 7. Sonraki Adımlar (Bu Projenin Kapsamı Dışında)

- Instruction tuning / fine-tuning — ayrı bir faz olarak ileride ele alınabilir.
- RLHF veya hizalama teknikleri — şimdilik kapsam dışı.

## 8. Proje Klasör Yapısı ([Proje-Dosyalari](../Proje-Dosyalari))

```
Proje-Dosyalari/
  runtime/           # Katman 0-2: bellek yönetimi, matematik fonksiyonları, PRNG
  tensor/            # Katman 3: tensor/dizi veri yapısı
  autograd/          # Katman 4: otomatik türev motoru
  cuda/              # Katman 5: CUDA çekirdekleri
  tokenizer/         # Katman 6: kök sözlüğü, ek tablosu, morfolojik ayrıştırıcı, disambiguation, BPE fallback
  model/             # Katman 7: Transformer mimarisi
  training/          # Katman 8: optimizer, eğitim döngüsü, config
  data/              # ham ve işlenmiş veri seti
  checkpoints/       # eğitim sırasında kaydedilen model ağırlıkları
  tests/             # her katman için doğrulama testleri (örn. gradient checking)
  logs/              # eğitim logları
```

## 9. Açık Kararlar / Tartışılacak Noktalar

- [x] RAM miktarı: **16 GB** — büyük korpüsler işlenirken parçalara bölerek (streaming/chunking) ilerlenecek.
- [x] CUDA erişim sınırı: **CUDA Driver API** (`cuMemAlloc`, `cuMemcpyHtoD`/`DtoH`, `cuModuleLoad`, `cuLaunchKernel`) kullanılacak — Runtime API değil.
- [x] Temel matematik fonksiyonları için yöntem: **Doğruluk öncelikli** (Taylor serisi + Newton-Raphson), hız optimizasyonu ileride profil sonrası değerlendirilecek.
- [x] PRNG algoritması: **PCG** (+ Box-Muller ile Gauss dağılımı üretimi)
- [x] Sayısal hassasiyet: **Sadece float32**. Düşük hassasiyet (fp16 benzeri) formatı şimdilik kapsam dışı.
- [x] Test stratejisi: Otograd için **sayısal gradyan kontrolü** (numerical gradient checking — `(f(x+e)-f(x-e))/2e` ile otograd çıktısını karşılaştırma). Her katman için ayrı doğrulama testleri (`tests/` klasörü).
- [x] Positional embedding: **RoPE** (Rotary Positional Embedding)
- [x] Normalizasyon tipi: **RMSNorm**
- [x] Vocab boyutu: **32.000**
- [x] Veri kaynakları: **Wikipedia + OSCAR/mC4 + haber arşivleri + altyazı korpüsleri** (dördü birden). Haber ve altyazı kaynaklarının lisans/kullanım kısıtları indirmeden önce tek tek kontrol edilecek.
- [x] Faz 1 / Faz 2 model boyutu: **Faz 1 ~1-5M parametre** (4-6 katman, d_model 128-256 — pipeline doğrulama), **Faz 2 ~30-150M parametre** (Faz 1 sonrası CUDA çekirdeklerinin gerçek performansı ölçülüp kesin hedef netleştirilecek).
- [x] Tokenizasyon yaklaşımı: **Tam morfolojik analizör** (kök sözlüğü + ünlü uyumu/ünsüz mutasyonu kuralları + belirsizlik giderme), analiz edilemeyen kelimeler için BPE fallback.
- [x] Kök sözlüğü kaynağı: **Halka açık sözlük/kelime listesi** (örn. Türkçe Vikisözlük dökümü) indirilip veri olarak kullanılacak; kök çıkarma işlemi kendi kodumuzla yapılacak.
- [x] Belirsizlik giderme yöntemi: **İstatistiksel model** — korpüs üzerinden çıkarılan n-gram/HMM benzeri geçiş olasılıklarıyla bağlama göre en olası kök+ek ayrışımı seçilecek (kendi yazacağımız küçük bir istatistiksel model, Katman 6'nın parçası).
- [x] Ek tablosu kapsamı: **Tüm envanter** — nadir/arkaik ekler dahil, Türkçe dilbilgisinde tanımlanmış neredeyse tüm çekim ve yapım ekleri kodlanacak.

## 10. Riskler

- **Kapsam riski (en büyük risk):** Sıfırdan bir derin öğrenme çatısı inşa etmek, LLM eğitiminin kendisinden çok daha uzun sürebilir. Bu normal kabul edilmeli ve süreç buna göre planlanmalı.
- **Morfotaktik sıra riski (kısmen çözüldü):** İsim çekiminde KÖK-ÇOĞUL-İYELİK-HÂL sırası ve isim/fiil eklerinin karışmaması artık zorlanıyor (bkz. Katman 6 günlüğü — iki gerçek hata bu sayede yakalanıp düzeltildi). Fiil çekim eklerinin kendi arasındaki sıra (örn. olumsuzluk+zaman+şahıs) ve yapım eklerinin isim/fiil kategorisi arasında köprü kurma kuralları henüz tam modellenmedi — ileride genişletilebilir.
- **Düzensiz kök riski:** Türkçe'de bazı kökler (özellikle tek heceli/alıntı kelimeler) fonoloji kurallarına düzensiz uyar (örn. "ok"→"oku", "oğu" değil). V1 fonoloji motoru sadece düzenli davranışı kapsıyor; istisna listesi ileride eklenmelidir.
- **Morfolojik analizör riski:** Tam bir Türkçe morfolojik analizör (kök sözlüğü + kurallar + belirsizlik giderme) kendi başına büyük bir dilbilim/NLP alt projesidir; kapsamı model eğitiminin kendisiyle kıyaslanabilir büyüklükte olabilir. Belirsizlik giderme özellikle zordur (aynı kelimenin birden fazla geçerli kök+ek ayrımı olabilir).
- Sınırlı VRAM + optimize olmayan (BLAS'sız) çekirdekler nedeniyle eğitim süresi beklenenden çok uzun olabilir.
- Kendi yazdığımız matematik fonksiyonlarındaki küçük hassasiyet hataları, eğitim sırasında birikerek (numerical instability) modele zarar verebilir — bu yüzden her katmanın ayrı test edilmesi kritik.
- Hazır debugging/görselleştirme araçları olmadığı için hata bulmak zaman alabilir.
- Yetersiz/düşük kaliteli veri, model kalitesini sınırlayabilir.
- Laptop üzerinde uzun süreli eğitim sırasında termal/güç kesintisi gibi pratik sorunlar olabilir — checkpoint stratejisi bu yüzden kritik.

## 11. İlerleme Durumu (Progress Log)

**2026-09-24:**
- Araç kurulumu: **MinGW-w64 GCC 16.2.0** kuruldu — `C:\devtools\mingw64` (not: kullanıcı klasörü yolundaki boşluk — "Berat Deniz" — MinGW'nin `ld` bağlayıcısında bilinen bir hataya yol açtığı için araç seti boşluksuz bir yola taşındı; proje kaynak kodu hâlâ `AI\Proje-Dosyalari` altında, sadece derleyici araç seti taşındı).
- **CUDA Toolkit 13.4.2** indirildi (3.65 GB, doğrulandı) ve kullanıcı tarafından yönetici yetkisiyle sessiz modda kuruluyor (`-s -n`, sürücüyü etkilemez).
- **Katman 0 (Sistem Arayüzü) TAMAMLANDI:** bellek ayırıcı (arena + free-list + coalescing) ve dosya I/O, 115/115 test ile doğrulandı. Detay için Bölüm 3'e bakınız.
- Sıradaki adım: **Katman 1 — Temel Matematik Kütüphanesi** (exp/log/sqrt/sin/cos, Taylor serisi + Newton-Raphson, doğruluk testleriyle).

**2026-09-24 (devam):**
- **CUDA Toolkit 13.4 kuruldu ve doğrulandı** (`nvcc --version` çalışıyor, sürücü bozulmadı, RTX 4050 6141 MiB olarak görünüyor).
- **Katman 1 (Matematik Kütüphanesi) TAMAMLANDI:** 36/36 test başarılı.
- **Katman 2 (PRNG) TAMAMLANDI:** 8/8 test başarılı.
- **Katman 3 (Tensor) TAMAMLANDI:** 15/15 test başarılı.
- **Katman 4 (Otograd) TAMAMLANDI:** 34/34 sayısal gradyan kontrolü başarılı — sistemin kalbi çalışıyor.
- Toplam: **93/93 test başarılı**, 4 katman tamamlandı, araç seti (GCC + CUDA Toolkit) tamamen hazır.
- Sıradaki adım: **Katman 5 — CUDA Çekirdekleri** (Driver API ile matmul/elementwise/reduction çekirdekleri, GPU'da ilk çalıştırma).

**2026-09-24 (Katman 5 — CUDA Çekirdekleri):**
- `cuda/kernels.cu`: k_add, k_mul, k_relu, k_matmul, k_reduce_sum (ham CUDA C, cuBLAS/cuDNN yok) — `nvcc -ptx` ile PTX'e derleniyor.
- `cuda/driver_api.h`: CUDA Driver API'nin minimal bildirimleri (win32_syscalls.h ile aynı felsefe — `<cuda.h>` dahil edilmedi).
- **Beklenmedik engel 1:** nvcc, Windows'ta PTX üretimi için bile MSVC (`cl.exe`) gerektiriyor — MinGW ile çalışmıyor (NVIDIA'nın resmi kısıtı). Çözüm: Visual Studio Build Tools / mevcut VS 2026 Community kurulumuna **NativeDesktop** workload'ı (VS 2026'da eski `VCTools` ID'si `Microsoft.VisualStudio.Workload.NativeDesktop` olarak değişmiş) + `Microsoft.VisualStudio.Component.VC.Tools.x86.x64` + `Microsoft.VisualStudio.Component.Windows11SDK.26100` bileşenleri eklendi.
- **Beklenmedik engel 2:** `cuda.lib`, klasik bir DLL import-library değil, doğrudan bağlanabilir "stub" sembolleri içeriyor — `__declspec(dllimport)` kullanmadan düz `extern` bildirimle bağlanması gerekti (`driver_api.h` buna göre düzeltildi).
- **Beklenmedik engel 3:** CUDA Toolkit 13.4'ün ürettiği PTX, sürücünün desteklediğinden (581.86 → CUDA 13.0) daha yeni bir PTX ISA sürümünde (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).
  - İlk denenen çözüm (CUDA 13.0'ı yan yana kurmak) **çıkmaz sokak çıktı**: CUDA 13.0, VS 2026'nın MSVC'siyle (v14.51) temelden uyumsuz (STL şablon hataları, sürüm kontrolünü atlatsak bile derinlerde başarısız oluyor).
  - **Nihai çözüm:** CUDA 13.4 zaten bu MSVC ile sorunsuz derliyordu — asıl sorun sürücüydü. GPU sürücüsü **617.14**'e güncellendi (CUDA 13.4/PTX destekleniyor). CUDA 13.0 kurulumu kullanılmadı (gerekirse daha sonra kaldırılabilir, şu an sadece disk yer kaplıyor, zararsız).
- **Sonuç: Katman 5 TAMAMLANDI.** `tests/test_cuda.exe` gerçek RTX 4050 üzerinde 3/3 test başarılı (k_add, k_matmul, k_reduce_sum). Tüm araç seti (MinGW GCC, CUDA Toolkit 13.4, MSVC/VS2026, GPU sürücüsü 617.14) artık uyumlu ve çalışır durumda.
- Toplam durum: **5 katman tamamlandı** (Katman 0-5), **96 CPU testi + 3 GPU testi = 99 test başarılı**.
- Sıradaki adım: **Katman 6 — Türkçe Morfolojik Analizör + Tokenizer** (kök sözlüğü temini, ek tablosu, ünlü uyumu/ünsüz mutasyonu kuralları, morfolojik ayrıştırıcı, istatistiksel belirsizlik giderme, BPE fallback).

**2026-09-24 (Katman 6 başlangıç — Kök Sözlüğü):**
- `runtime/strutil.c`: temel string fonksiyonları (`str_find`, `str_eq`, vb. — `<string.h>` kullanılmadan, kendi yazdığımız).
- Türkçe Vikisözlük dökümü indirildi: `trwiktionary-latest-pages-articles.xml.bz2` (61 MB sıkıştırılmış, 1.2 GB açılmış, 40.7M satır) — `dumps.wikimedia.org`'dan, veri olarak (yazılım değil).
- `tokenizer/extract_roots.c`: XML dökümünü (hiçbir XML/regex kütüphanesi olmadan, kendi `str_find` tabanlı ayrıştırıcımızla) tarayıp ana isim alanındaki (ns=0), "Türkçe" dil başlığı içeren, tek kelimelik sayfa başlıklarını kök adayı olarak çıkarıyor.
- **Sonuç: 1.469.712 sayfa tarandı, 286.879 benzersiz kök adayı çıkarıldı** (`data/raw/kok_adaylari.txt`, 3.7 MB). Örnek doğrulama: gerçek Türkçe kökler/fiiller (empati, oğlan, didinmek, didişmek, dilekçe vb.) doğru yakalanmış.
- Sıradaki adım: **Ek tablosu ve ünlü uyumu/ünsüz mutasyonu kuralları**, ardından morfolojik ayrıştırıcı (FST mantığıyla kök+ek analizi).

**2026-09-24 (Katman 6 devam — Fonoloji Motoru):**
- `runtime/utf8.c`: UTF-8 encode/decode (Türkçe'ye özgü ş/ğ/ç/ö/ü/ı/İ dahil, hiçbir <uchar.h>/ICU kullanılmadan).
- `tokenizer/turkish_phon.c`: **Ek şablonu çözücü** — sembolik gösterim (`A`=2 yollu ünlü a/e, `I`=4 yollu ünlü ı/i/u/ü, `D`=d/t sertleşmesi, `C`=c/ç sertleşmesi), kaynaştırma ünsüzü ekleme (y/n/s/ş), ünsüz yumuşaması (p/ç/t/k→b/c/d/ğ). Kelimenin *son geçen ünlüsüne* göre uyum hesaplanır (zincirleme ek durumunda da doğru çalışır).
- Doğrulama: `tests/test_turkish_phon.c` — **16/16 test başarılı**, gerçek Türkçe çekimlerle (kitabı, ağacı, arabaya, gözcü, sütçü, kapıcı, evde/kitapta, evden/kitaptan, evler/kitaplar).
- **Bilinen sınırlama (not edildi):** Düzensiz kökler (örn. tek heceli/alıntı kelimelerdeki beklenmedik yumuşama davranışı — "ok"→"oku" gibi) için istisna listesi henüz yok; V1 sadece düzenli (regular) davranışı kapsıyor.
- Sıradaki adım: **Ek tablosu** (gerçek ek şablonlarının somut envanteri — çoğul, hâl ekleri, iyelik, temel fiil çekimleri) ve bunun üzerine **morfolojik ayrıştırıcı**.

**2026-09-24 (Katman 6 devam — Ek Tablosu):**
- `tokenizer/suffixes.c`: **20 ek** — çoğul, iyelik (1/2/3 tekil+çoğul), hâl ekleri (belirtme/yönelme/bulunma/ayrılma/tamlayan), yapım ekleri (-lık/-sız/-lı/-cı), fiil ekleri (mastar, olumsuzluk, şimdiki zaman, di'li geçmiş, yeterlilik). **Tam envanter değil, genişletilebilir bir başlangıç kümesi** (bkz. dosya başlığındaki not) — nadir/arkaik ekler ileride eklenecek.
- Doğrulama: `tests/test_suffixes.c` — **24/24 test başarılı**, gerçek kelimelerle (evler, arabam, kitabım, arabası, kitabı, kitaba, evde, kitaptan, arabanın, gözlük, tuzsuz, tuzlu, sütçü, gitmek, gelme, okuyor, gidiyor, geldi, yaptı, okuyabil, gelebil).
- **Test sürecinde gerçek bir hata yakalandı ve düzeltildi:** "-abil/-ebil" (yeterlilik) ekinin ikinci ünlüsü ("bil") ilk bakışta 4 yollu uyum sanılmıştı (`AbIl` şablonu → "okuyabul" yanlış çıktısı), ama gerçekte bu ünlü sabittir ve kökün yuvarlaklığından etkilenmez (okuyabilir, hep "i"). Şablon `Abil` olarak düzeltildi — sayısal/örnek tabanlı testlerin değerini gösteren somut bir örnek.
- **Morfotaktik sıra (hangi ek hangi sırada gelir) V1'de zorlanmıyor** — bilinen bir basitleştirme, Riskler'e eklendi.
- Sıradaki adım: **Morfolojik ayrıştırıcı** (kök sözlüğü + ek tablosunu birleştirip rastgele bir kelimeyi kök+ek dizisine ayıran algoritma).

**2026-09-24 (Katman 6 devam — Morfolojik Ayrıştırıcı):**
- `runtime/hashset.c`: sıfırdan yazılmış string hash kümesi (FNV-1a + açık adresleme), kök sözlüğü üyelik kontrolü için (294K kelime, kendi kütüphanemiz).
- `tokenizer/lexicon.c`: kök sözlüğünü yükler; **fiil köklerini de otomatik türetir** (sözlükte "gitmek" mastar hâlinde kayıtlı ama ayrıştırıcının çıplak kök "git"e ihtiyacı var — "-mek/-mak" ile bitenlerden çıplak kök ayrıca ekleniyor).
- `tokenizer/analyzer.c`: **"üreterek doğrulama" (generate-and-test)** yöntemiyle çalışan ayrıştırıcı — her olası kök uzunluğu ve her ek için, fonoloji motoruyla o kökten o ekle yüzey formun ne olması gerektiğini hesaplayıp gözlemlenen kelimeyle karşılaştırır (ters çevirmeye çalışmak yerine ileri yönde üretip doğrular). Birden fazla geçerli ayrıştırma varsa hepsini döndürür (belirsizlik giderme bir sonraki adımda).
- **Test sürecinde bulunan ve düzeltilen gerçek bir eksik:** Yumuşamış bir kök (örn. "kitab") yüzeyde göründüğünde, ayrıştırıcı onun **yumuşamamış halini de ("kitap") bir alternatif hipotez olarak** denemiyordu — bu yüzden "kitabı" gibi doğrudan yumuşamış biten kelimelerde gerçek kök bulunamıyordu (zincirleme örneklerde tesadüfen sorun çıkmamıştı). `tr_unsoften()` eklenip ayrıştırıcıya "ters yumuşama hipotezi" olarak entegre edildi.
- **Gerçek bir belirsizlik örneği yakalandı:** "gidiyorlar" kelimesi hem standart kök "git" (yumuşamayla) hem de Vikisözlük'te kayıtlı nadir/lehçesel bir kök olan "gidmek"ten (yumuşamasız) türeyebiliyor — ikisi de fonolojik olarak geçerli, aynı yüzey formu üretiyor. Bu, tam olarak plandaki **istatistiksel belirsizlik giderme** bileşeninin çözmesi gereken türden gerçek bir örnek.
- Doğrulama: `tests/test_analyzer.c` — **7/7 uçtan uca test başarılı** (gerçek kök sözlüğü + ek tablosuyla: kitap, gözlük, arabası, evlerde, kitaplardan, kitaplarımızdan, gidiyorlar).
- Sıradaki adım: **İstatistiksel belirsizlik giderme** (n-gram/HMM benzeri model, korpüs üzerinden bağlama göre en olası ayrıştırmayı seçer) ve **BPE fallback** (analiz edilemeyen kelimeler için).

**2026-09-24 (Katman 6 devam — İstatistiksel Belirsizlik Giderme):**
- `runtime/strcounter.c`: string→sayaç hash tablosu (kendi yazdığımız, hashset.c ile aynı yöntem).
- `tokenizer/build_root_freq.c`: Aynı Vikisözlük dökümünü tekrar tarayıp Türkçe bölümlerin düz metnindeki kelime frekanslarını çıkarır — **280.082 benzersiz kelime** (`data/raw/kelime_frekans.txt`).
- `tokenizer/disambiguate.c`: **V1 yöntemi — kök unigram frekansı.** Birden fazla geçerli ayrıştırma arasında korpüste en sık geçen kökü tercih eder; eşitlikte daha az ek içeren (daha sade) ayrıştırma kazanır. **Not: tam "n-gram/HMM ile bağlama göre" hedefinden daha basit bir unigram modeldir** — gerçek bağlamsal (bigram/HMM) model, etiketlenmiş bir eğitim korpüsü gerektirir (bkz. dosya başlığındaki not); ileride genişletilecek.
- **Test sürecinde ikinci gerçek hata bulundu ve KAYNAĞINDA düzeltildi:** İlk denemede "g" (kısaltma olarak metinde çok sık geçen tek harf) 4 eklik geçersiz bir zincirle (iyelik+geçmiş-zaman+şimdiki-zaman+çoğul — isim ve fiil eklerini karıştırarak) doğru kökü ("git") yeniyordu. Çözüm: frekans skoruna yama yapmak yerine **analyzer.c'ye isim/fiil kategori tutarlılık kuralı eklendi** — bir kelime aynı zincirde hem isim çekim eki hem fiil çekim eki alamaz (türetim ekleri hariç). Bu, hem sorunu kökünden çözdü hem de "-lAr" ekinin hem isim çoğulu hem fiil 3. çoğul şahıs eki olarak **iki ayrı kategori girdisiyle** tabloya eklenmesini gerektirdi.
- Doğrulama: `tests/test_disambiguate.c` — **5/5 test başarılı**, gerçek belirsiz örnekle (gidiyorlar/gidiyor → doğru şekilde "git" seçiliyor, gid=3 ve g'nin geçersiz zinciri elenmiş durumda) ve belirsiz olmayan durumlarda da (kitap, arabası→araba, evlerde→ev) doğru kök seçiliyor.
- Sıradaki adım: **BPE fallback** (kök sözlüğünde bulunamayan/analiz edilemeyen kelimeler için).

**2026-09-24 (Katman 6 TAMAMLANDI — BPE Fallback + Nihai Tokenizer):**
- **Mimari netleştirme:** 294K kök, 32.000'lik sabit vocab'a doğrudan sığamaz. Çözüm: BPE hem (a) analiz edilemeyen kelimeler için yedek, hem de (b) **ayrıştırıcının bulduğu KÖKÜN KENDİSİNİ** alt-kelime parçalarına bölerek sınırlı sözlükte temsil etmenin mekanizması oldu (sık kökler tek token, nadir kökler birden çok parça).
- `runtime/sort.c`: kendi quicksort'umuz (frekansa göre sıralama için, `qsort` kullanılmadı).
- `tokenizer/bpe.c`: Bayt-seviyeli BPE eğitimi (frekans ağırlıklı çift sayımı + en sık çifti birleştirme) ve kodlama — sıfırdan, hiçbir kütüphane yok. **31.744 sembol** (256 temel bayt + 31.488 birleştirme), 100.000 kelimelik eğitim kümesiyle **6 dk 28 sn**'de eğitildi.
- `tokenizer/tokenize.c`: Tüm sistemi birleştiren nihai fonksiyon — önce morfolojik ayrıştırma+belirsizlik giderme dener (kök→BPE, ekler→kendi özel token'ları), hiç ayrıştırma yoksa tüm kelime doğrudan BPE ile kodlanır.
- **Nihai vocab: 31.769 token** (22 ek kategorisi + 31.744 BPE sembolü + 3 özel token — PAD/BOS/EOS). ~32.000 hedefine çok yakın.
- **Test sürecinde ÜÇÜNCÜ gerçek hata bulundu ve KAYNAĞINDA düzeltildi:** "kitaplarımızdan" hem doğru (çoğul+iyelik+hâl) hem de dilbilgisel olarak geçersiz ama YÜZEYDE AYNI SONUCU ÜRETEN bir şekilde (iki iyelik eki üst üste: "iyelik-3çoğul"+"iyelik-1çoğul") ayrıştırılabiliyordu ve belirsizlik giderme yanlışını seçiyordu. Çözüm: ek tablosuna **Türkçe isim çekiminin bilinen sabit sırası (KÖK-ÇOĞUL-İYELİK-HÂL, her yuva en fazla bir kez)** kodlandı (`NounSlot` alanı + analyzer.c'de sıra/tekillik kontrolü). Bu, önceki "isim/fiil karışmasın" kuralının doğal bir genellemesi.
- Doğrulama: `tests/test_tokenize.c` — uçtan uca görsel doğrulama (kitap, kitaplarımızdan, evlerde, arabası, gidiyorlar, gözlük hepsi doğru ayrıştırılıyor; İstanbul/xzqvt123/günaydın/üniversite gibi öbekler de makul şekilde işleniyor).
- **Bilinen sınırlama (not edildi):** BPE, küçük harfe çevrilmiş kelime listesiyle eğitildi; büyük harfli özel isimlerin (İstanbul gibi) bazı karakterleri ham bayta düşebiliyor (yanlış değil, sadece daha az verimli — ileride büyük/küçük harf duyarlı eğitim veya normalizasyon ile iyileştirilebilir).
- **TOPLAM: Katman 6 (Türkçe Morfolojik Analizör + Tokenizer) TAMAMLANDI.** Kök sözlüğü → fonoloji motoru → ek tablosu (morfotaktik sıralı) → ayrıştırıcı → istatistiksel belirsizlik giderme → BPE → nihai 31.769 token'lık vocab, hepsi çalışıyor ve test edilmiş durumda.
- Sıradaki adım: **Katman 7 — Model Mimarisi** (Transformer: self-attention, feed-forward, RMSNorm, RoPE — Katman 3-5 üzerine inşa).

**2026-09-24 (Katman 7 TAMAMLANDI — V1, batch'siz, CPU):**
- RMSNorm, RoPE, çoklu-başlıklı nedensel self-attention, SwiGLU ileri besleme, tam Transformer bloğu — hepsi yazıldı ve test edildi.
- **70 yeni test** (`test_model_ops`: 30/30, `test_transformer_block`: 40/40) — toplam proje test sayısı **338/338** (335 CPU + 3 GPU).
- Sıradaki adım: **Batch desteği** + **Katman 8 — Optimizer ve Eğitim Döngüsü** (Adam, checkpoint sistemi) + CUDA entegrasyonu (GPU'da gerçek eğitim).

**2026-09-24 (Batch Desteği TAMAMLANDI):**
- **Mimari içgörü:** Tensor/otograd çekirdeğini batched N-boyutlu matmul'a genişletmeye gerek kalmadı — RMSNorm zaten herhangi bir önden boyutla çalışıyor, ve otograd sistemi paylaşılan ağırlıkların birden fazla yerde kullanılmasını (gradyan biriktirme ile) zaten doğru destekliyordu. Çözüm: aynı ağırlıkları paylaşarak dizi başına bir Transformer bloğu çalıştırmak (`model/batch.c`), kayıpları ortalamak.
- Doğrulama: `tests/test_batch.c` — **16/16 test başarılı** (2 bağımsız dizi, paylaşılan ağırlıklar, birleşik ortalama kayıp üzerinden hem girdi hem PAYLAŞILAN ağırlık gradyanları sayısal olarak doğrulandı — batch üzerinden gradyan toplama/ortalamanın doğru çalıştığının kanıtı).
- Sıradaki adım: **Katman 8 — Adam Optimizer** ve checkpoint sistemi.

**2026-09-24 (Katman 8 TAMAMLANDI — Adam + Checkpoint):**
- `training/adam.c`: standart Adam algoritması (momentum + ikinci moment + ön-yanlılık düzeltmesi), her parametre için ayrı m/v tensoru.
- `training/checkpoint.c`: kendi ikili dosya formatımız (Katman 0'ın dosya I/O'su üzerinden) — parametre değerleri VE isteğe bağlı optimizer durumu (m, v, zaman adımı t) kaydedilip yükleniyor.
- Doğrulama: `tests/test_adam.c` — **6/6 test başarılı**. (1) **Yakınsama testi:** küçük bir 2 katmanlı model 200 adımda kayıp 1.493→~0'a düşürüldü (Adam gerçekten öğreniyor). (2) **Checkpoint round-trip testi:** kaydet→değerleri boz→yükle, parametre değerleri VE optimizer momentum/zaman adımı tam olarak (bit-hassasiyetinde) geri yüklendi.
- Sıradaki adım: **CUDA entegrasyonu** — Katman 5'in CUDA çekirdeklerini otograd sistemine bağlayıp gerçek GPU eğitimi mümkün kılmak.

**2026-09-24 (CUDA Entegrasyonu TAMAMLANDI — V1, matmul):**
- **Kapsam kararı:** Transformer hesaplamasının büyük çoğunluğunu oluşturan **matris çarpımı** GPU'ya taşındı; diğer işlemler (softmax, RMSNorm, RoPE) V1'de CPU'da kalmaya devam ediyor — en yüksek etkili optimizasyondan başlayan, kademeli ve dürüst bir kapsam.
- `runtime/timer.c`: yüksek çözünürlüklü zamanlayıcı (`QueryPerformanceCounter`, performans karşılaştırması için).
- `model/gpu_ops.c`: `node_matmul2d_gpu` — Katman 5'in `k_matmul` çekirdeğini kullanan, host↔device kopyalama dahil tam otograd düğümü (ileri VE geri yayılım GPU'da).
- Doğrulama: `tests/test_gpu_ops.c` — **13/13 test başarılı**. (1) **Doğruluk:** GPU sonucu CPU sonucundan en fazla ~3×10⁻⁶ farklı (float32 hassasiyeti sınırında). (2) **Sayısal gradyan kontrolü:** GPU matmul otogradı doğru. (3) **Gerçek hızlanma:** 1024×1024 matris çarpımında gerçek RTX 4050 üzerinde **CPU 2597 ms → GPU 12 ms (veri transferi dahil) = ~202x hızlanma**.
- **Sonuç: Katman 5-8 hattı uçtan uca çalışıyor** — GPU'da gerçekten hızlanan bir matris çarpımı, otograd sistemine tam entegre ve doğrulanmış durumda.
- Sıradaki adım (ileride): softmax/RMSNorm/RoPE gibi diğer işlemleri de GPU'ya taşımak (tam GPU-resident eğitim), ve gerçek büyük ölçekli eğitim döngüsünü (veri yükleme + tokenizer + model + optimizer birleşimi) kurmak.

## 12. Gerçek Eğitim Verisi — Türkçe Vikipedi (2026-09-25)

**Veri temini ve işleme hattı, uçtan uca kuruldu ve çalıştırıldı:**

1. **İndirme:** Türkçe Vikipedi dökümü (`trwiki-latest-pages-articles.xml.bz2`, 1.05 GB sıkıştırılmış → 5.56 GB açılmış), `dumps.wikimedia.org`'dan.
2. **`tokenizer/wikitext_clean.c`:** Sıfırdan yazılmış wikitext→düz metin temizleyici — şablonları (`{{...}}`, iç içe olabilir), tabloları (`{|...|}`), referansları (`<ref>`), yorumları, matematik bloklarını **tamamen atar**; wiki bağlantılarını (`[[Bağlantı|Görünen metin]]`) görünen metne indirger; kategori/dosya bağlantılarını atar; kalın/italik işaretlemeyi temizler. Ayrıca `xml_unescape` (MediaWiki dökümlerinde wikitext bir XML `<text>` elemanı içinde geldiği için `&lt;ref&gt;` gibi kaçışları çözer — bu adım atlanırsa `<ref>` gibi etiketler hiç yakalanamaz). **12/12 test başarılı** (`tests/test_wikitext_clean.c`).
3. **`tokenizer/build_wiki_corpus.c`:** XML dökümünü **akış (streaming) tabanlı** işler (RAM kısıtlı olduğu için — bkz. Bölüm 2 — tüm 5.5GB dosya belleğe yüklenmez, 256MB'lık pencereler halinde okunur, sayfa sınırında yarım kalan veri bir sonraki pencereye taşınır). **489.592 makale, 1.07 GB temiz metin** çıkardı (`data/raw/wikipedia_corpus.txt`) — sadece 26 saniyede.
4. **`tokenizer/tokenize_corpus.c`:** Tüm tokenizer boru hattını (Katman 6: ayrıştırıcı + belirsizlik giderme + BPE) gerçek korpüs üzerinde çalıştırır, yine akış tabanlı (128MB pencere, kelime sınırında güvenli kesim).

**SONUÇ: 1.07 GB girdi → 494.603.976 token (data/raw/wikipedia_tokens.bin, 1.98 GB ikili dosya).**

### Gerçek büyük-ölçekli veride bulunan ve düzeltilen kritik hatalar

Bu, projenin "sayısal/örnek tabanlı test" felsefesinin ötesinde, **gerçek veri hacminin kendisinin bir test olduğunu** gösteren önemli bir bölüm oldu — küçük birim testlerinde asla ortaya çıkmayacak sınır durumları bulundu:

1. **Performans: Aşırı yavaş ayrıştırma.** İlk koşuda `STEP_BUDGET` (200.000) bazı içeriklerde parça başına saniyeler süren yavaşlığa yol açtı. **Çözüm:** Bütçe 5.000'e düşürüldü — gerçek Türkçe kelimeler için fazlasıyla yeterli, patolojik girdilerde en kötü durum gecikmesini sıkıca sınırlıyor.
2. **`tokenize_corpus.c`'de tampon taşması (buffer overflow).** Pencere sınırında "taşınan" (carry) veri miktarı (`carry_len`) sınırlanmamıştı — çok uzun boşluksuz bir blok (örn. büyük bir veri tablosu) bir sonraki okumanın tampon dışına taşmasına yol açabiliyordu. **Çözüm:** `MAX_CARRY` sınırı eklendi (aşan veri, nadir bir durumda güvenle atlanır).
3. **`analyzer.c`'de YIĞIN TAŞMASI (stack buffer overflow) — asıl çökme nedeni.** Vikipedi'de vandalizm/bozuk düzenlemeden kalma **84 baytlık tekrarlayan bir "ğüğüğü..." gibi çöp "kelime"**, ayrıştırıcının dahili tamponlarının (`ANALYZER_MAX_ROOT_BYTES` = 64 bayt) sınırını aştı — normal Türkçe kelimeler bu sınırı asla zorlamadığı için hiçbir testte ortaya çıkmamıştı. **Çözüm:** `analyze_word`, uzunluğu bu sınırı aşan girdileri en baştan güvenle reddedip (0 ayrıştırma, BPE'ye düşer) hiçbir dahili arama başlatmıyor. Regresyon testi eklendi (`tests/test_analyzer.c`, gerçek çökme senaryosunu birebir taklit ediyor).

Bu üç hata, izole bir 140MB'lık şüpheli bölge çıkarılıp (`tail -c`/`head -c` ile), çökme anına kadar her segmentin içeriğini yazdıran özel bir hata ayıklama aracıyla kesin olarak tespit edildi — sistematik ikili arama (binary search) yerine doğrudan gözlemle.

**Sonuç:** Katman 6'nın tüm bileşenleri artık gerçek, büyük ölçekli, "kirli" (vandalizm/bozuk içerik dahil) verilerle sağlamlığı kanıtlanmış durumda.

**Görsel doğrulama:** Çıktının ilk token'ları geri çözüldüğünde, korpüsün gerçekten ilk cümlesiyle ("Cengiz Han, (doğum adıyla Temuçin ya da Timuçin, ...)") birebir eşleştiği doğrulandı.

**Bilinen sınırlama (yeni fark edildi):** "doğum" kelimesi "doğu"(iyelik-1tekil) olarak yanlış ayrıştırıldı ("doğu"+"m" fonolojik olarak geçerli ama semantik olarak yanlış). Kök nedeni: **belirsizlik giderme frekans tablomuz Vikisözlük'ün tanım metninden çıkarıldı, tokenize ettiğimiz Vikipedi metninden değil** — bu alan uyuşmazlığı (domain mismatch), bazı kelimelerin göreli frekans sıralamasını gerçek kullanım dağılımından saptırabiliyor. Sistemin kendisi bozuk değil; ileride frekans tablosunun (aynı) Vikipedi korpüsünden de çıkarılması bu tür durumları azaltacaktır.

**Sıradaki adım:** OSCAR/mC4, haber arşivleri ve altyazı korpüslerinin temini (henüz yapılmadı — bu kaynaklar Vikipedi'den daha karmaşık erişim/format zorlukları içerebilir, ayrıca değerlendirilecek); ve/veya bu 494M token'lık veriyle Katman 7-8'in gerçek bir eğitim koşusunu başlatmak.

## 13. İlk Gerçek Eğitim Koşusu — Faz 1 Doğrulaması (2026-09-25)

**Amaç:** Tüm alt yapı yığınının (tokenizer → tensor/otograd → Transformer → Adam → checkpoint → GPU matmul) gerçek Vikipedi verisiyle uçtan uca, hatasız çalıştığını kanıtlamak (Faz 1: pipeline doğrulama, ~1-5M parametre — bkz. Bölüm 4).

### Model birleştirme (yeni kod)

1. **`autograd/node.h`:** `Node`'a genel amaçlı `const void* aux_ptr` alanı eklendi (embedding ve cross-entropy gibi "füzyonlu" (fused) op'ların geri-yayılımda ek durum saklaması için — örn. embedding token_id dizisini, cross-entropy softmax olasılıklarını önbelleğe alıyor).
2. **`model/embedding.c/h`:** `node_embedding_lookup` — ileri yayılımda satır kopyalama, geri yayılımda **saçılma-biriktirme (scatter-accumulate)** gradyan (tekrarlayan token id'lerini doğru işliyor).
3. **`model/loss.c/h`:** `node_cross_entropy_loss` — sayısal kararlı (max-çıkarmalı) softmax, ortalama negatif log-olabilirlik; geri yayılımda `(softmax - one_hot) * dy/seq_len`.
4. **`tests/test_lm_ops.c`:** **31/31 test başarılı** (embedding + cross-entropy sayısal gradyan kontrolü, tekrarlayan-id birikim testi, emin-tahmin→düşük-kayıp testi dahil).
5. **`model/lm_model.c/h`:** Tüm parçaları birleştiren `LMModel` — gömme tablosu (embed_table) → N × Transformer bloğu (RMSNorm+RoPE+çoklu-başlık dikkat+SwiGLU) → son RMSNorm → **ağırlık bağlantılı (tied)** çıkış projeksiyonu (`logits = x @ embed_table^T`, büyük vocab karşısında parametre sayısını kontrol altında tutmak için). `lm_init`, `lm_forward`, `lm_collect_params` fonksiyonları.
6. **`tests/test_lm_model.c`:** Küçük ölçekli (vocab=12, d_model=8, 2 katman) **TAM birleştirilmiş modelin** uçtan uca sayısal gradyan kontrolü — embed_table, blok0.wq, blok1.w_down, final_norm_w gibi mimarinin en uzak noktalarındaki parametrelerden örnekleme yapıldı. **12/12 test başarılı** — tüm bağlantıların (embedding→transformer katmanları→bağlı çıkış) doğru kurulduğunu kanıtlıyor.
7. **GPU entegrasyonu:** `lm_forward`'a `use_gpu_output` bayrağı eklendi — büyük vocab (31.769) yüzünden toplam FLOP'ların ezici çoğunluğunu oluşturan bağlı çıkış projeksiyonu (`node_matmul2d_gpu`) isteğe bağlı olarak GPU'ya taşınabiliyor (Bölüm 11'de ölçülen ~200x hızlanma).

### Eğitim koşusu (`training/train_lm.c`)

- **Mimari:** vocab=31.769, d_model=128, 4 katman, 4 dikkat başlığı, d_ff=256, seq_len=128 → **4.727.552 parametre** (Faz 1 hedefi ~1-5M ile birebir örtüşüyor).
- **Veri:** `data/raw/wikipedia_tokens.bin`'den (494.603.976 token, tamamı ~6GB'lık kalıcı bir arena'ya yüklendi) her adımda rastgele 128 token'lık bir pencere örnekleniyor (girdi=tokens[i..i+128), hedef=tokens[i+1..i+129), sonraki-token tahmini).
- **Optimizer:** Adam (lr=3e-4, β1=0.9, β2=0.999), her adımda `checkpoint.c` ile periyodik kayıt (`checkpoints/lm_wiki_latest.bin`, her 250 adımda; final `checkpoints/lm_wiki_final.bin`).
- **Çıkış projeksiyonu GPU'da** (`node_matmul2d_gpu`), geri kalan tüm hesaplama (embedding, RMSNorm, RoPE, dikkat, SwiGLU, Adam) CPU'da.
- **SONUÇ (3000 adım, RTX 4050, ~39.4 dakika, ~0.79 sn/adım):** Kayıp **10.361 → ~3.87 (EMA)**. Başlangıç kaybı `ln(31769)=10.366`'ya son derece yakın (rastgele başlatmanın doğruluğunu teyit ediyor); model gerçek Türkçe metin üzerinde **gerçekten öğreniyor**. Checkpoint round-trip formatı (`value+momentum+ikinci-moment`, 12 bayt/parametre) beklenen boyutu üretti (56.731.672 bayt = 4.727.552 × 12 bayt).
- **Anlam:** Bu, projenin **Faz 1 hedefinin (küçük ölçekli uçtan uca pipeline doğrulaması) başarıyla tamamlandığı** anlamına gelir. Sıradaki adım, CUDA çekirdeklerinin gerçek performansına göre Faz 2 (~30-150M parametre) ölçeğine geçiş kararının netleştirilmesidir (bkz. Bölüm 4).

## 14. Metin Üretimi (Inference) ve Terminal Sohbet Aracı (2026-09-25)

Faz 1 checkpoint'inin sonucunu **gözle görülür** şekilde test edebilmek için iki yeni araç yazıldı (projede daha önce hiç detokenizasyon/inference yolu yoktu — sadece eğitim vardı):

1. **Detokenizasyon (`decode_tokens`):** BPE sembolleri `BpeVocab.symbol_bytes[id]` ile bağlamdan bağımsız doğrudan bayt kopyası; **ek (suffix) token'ları** ise `turkish_phon.h`'nin kodlarken kullanılan AYNI fonksiyonlarının (`tr_analyze_ending`, `tr_resolve_suffix`, `tr_apply_root_change`) TERS yönde çalıştırılmasıyla — o ana kadar üretilmiş tüm metin üzerinde ileri tarama yaparak doğru uyumlu (harmonik) yüzey formu yeniden kuruluyor. Roundtrip testi (`encode("Türkiye'nin başkenti")` → `decode(...)`) birebir orijinal metni verdi — hem tokenizer'ın hem yeni decoder'ın doğruluğunu kanıtlıyor.
2. **`training/generate.c`:** Tek seferlik demo — checkpoint yükler, sabit bir tohum metinden otoregresif üretim yapar (temperature=0.85 örnekleme), sonucu decode edip yazdırır.
3. **`training/chat.c`:** Etkileşimli terminal döngüsü — aynı model bir kez yüklenir, kullanıcıdan sürekli satır okunur (`console_read_line`, yeni eklenen `runtime/console.c` fonksiyonu — Win32 `ReadFile` üzerine, libc `scanf`/`fgets` YOK), her girdi için üretim yapılıp yazdırılır. **Dürüstlük notu (araç açılışında da kullanıcıya gösteriliyor):** Bu bir sohbet/talimat (chat/instruct) modeli DEĞİL — eğitim verisi ham Vikipedi metni olduğu için, yazılan metnin cevabını değil Vikipedi tarzı bir devamını üretiyor.
4. **Gözlemlenen Faz 1 çıktı örneği:** Tohum "Türkiye'nin başkenti" → devam: "...bir tüm gözü üzere gördü... tarihine zamanı gibi..." — dilbilgisel olarak bozuk ama doğru ek uyumu ve doğru fonksiyon kelimeleri (ve, ilk, sonra, gibi) içeriyor; 4.7M parametre + 3000 adım için beklenen kalite.

## 15. Faz 2: GPU Hızlandırmasının Genişletilmesi ve Büyük Ölçekli Eğitim (2026-09-25)

**Karar:** Faz 1 başarıyla doğrulandıktan sonra kullanıcı Faz 2'yi başlatma talimatı verdi — sonuçlarına göre projenin geri kalanı (Faz 3/veri genişletme kararları) şekillendirilecek.

### Sorun: CPU-only mimari Faz 2 ölçeğinde çok yavaş kalırdı

Faz 1'de sadece bağlı (tied) çıkış projeksiyonu GPU'daydı (büyük vocab yüzünden FLOP'ların ezici çoğunluğu oradaydı). Ama d_model büyüdükçe (128→384), Transformer katmanları içindeki projeksiyon/FFN matris çarpımlarının maliyeti de **d_model² ile** büyüyor — hesaplama yapıldı: d_model=384/12 katman ölçeğinde CPU-only tahmini adım süresi ~20 saniye/adım (3000 adım ≈ 16-17 saat) çıktı, pratik değil.

**Çözüm:** `model/attention.c` (wq/wk/wv/wo projeksiyonları) ve `model/feedforward.c` (SwiGLU gate/up/down) içindeki BÜYÜK matris çarpımlarına da `use_gpu` bayrağı eklendi (`node_matmul2d_gpu`, aynı Katman 7/8'de doğrulanmış çekirdek). Sadece başına-dikkat (per-head) küçük skor/ağırlıklı-toplam çarpımları CPU'da bırakıldı (boyutları küçük, GPU gidiş-geliş maliyeti kazancı götürürdü — Faz 1'in "en büyük FLOP tüketicisinden başla" felsefesiyle tutarlı). Bayrak `transformer_block` → `lm_forward` zincirinde tek bir `use_gpu` parametresiyle uçtan uca taşınıyor.

**Doğrulama:** `tests/test_lm_model_gpu.c` — test_lm_model.c ile AYNI küçük-ölçekli tam model, ama GPU yolu (use_gpu=1) açıkken sayısal gradyan kontrolü. **18/18 test başarılı** — yeni GPU entegrasyonunun CPU ile matematiksel olarak özdeş gradyan ürettiği kanıtlandı (büyük bir eğitim koşusuna başlamadan önce bu doğrulama kritikti).

### Faz 2 mimarisi ve koşusu

- **Mimari:** vocab=31.769, **d_model=384, 12 katman, 6 dikkat başlığı (head_dim=64), d_ff=1024** → **33.490.176 parametre** (hedeflenen 30-150M aralığının alt-orta noktası; Faz 1'in ~7.1 katı).
- **Veri:** Aynı 494.603.976 token'lık Vikipedi korpüsü (henüz OSCAR/mC4/haber/altyazı eklenmedi — bkz. Bölüm 5, ayrı bir sonraki adım).
- **Kod:** `training/train_lm.c` güncellendi (hiperparametreler + checkpoint yolları `checkpoints/lm_wiki_faz2_*.bin` olarak ayrıldı, Faz 1 checkpoint'leri korunuyor).
- **SONUÇ (3000 adım, RTX 4050, ~2 saat 59 dakika, ~3.58 sn/adım):** Kayıp **10.446 → 3.937 (EMA)**. Checkpoint: `checkpoints/lm_wiki_faz2_final.bin` (401.885.144 bayt — 33.490.176 param × 12 bayt/param + başlık, beklenen boyut).
- **ÖNEMLİ VE DÜRÜST BULGU:** Faz 2'nin nihai kaybı (3.937), Faz 1'inkine (3.87) neredeyse **eşit** — 7 kat daha fazla parametreye rağmen belirgin bir iyileşme YOK. Kök neden açık: her iki koşu da **aynı 3000 adım × 128 token = 384.000 token** gördü — 494.6M token'lık korpüsün sadece **~%0,08'i**. Model kapasitesi büyüdü ama eğitim (veri) bütçesi büyümedi; bu klasik bir **yetersiz-eğitim (undertraining)** durumu. Sonuç: **daha büyük bir model tek başına yetmiyor — asıl darboğaz adım/veri sayısı.** Bir sonraki anlamlı iyileştirme, mevcut 494.6M token'ı çok daha fazla adımla (çoklu epoch) taraması veya yeni veri kaynaklarının (OSCAR/mC4/haber/altyazı, bkz. Bölüm 5) eklenmesi olacaktır.
- **Üretilen metin karşılaştırması (aynı tohum, "Türkiye'nin başkenti"):**
  - Faz 1: "...bir tüm gözü üzere gördü. Mre Şu da sonra oyunlaru Telidan yıl tarihine zamanı gibi..."
  - Faz 2: "...olan iki kızı kenarı ve tarım vardırlardan ettır. İsada ünlü bir tılmış ile uzu Bestt gibi..."
  - Faz 2'nin açılışı ("Türkiye'nin başkenti **olan**...") gramer olarak biraz daha doğal ama genel tutarlılık farkı yok — yukarıdaki bulguyla tutarlı.
- **Araç güncellemesi:** `training/generate.c` ve `training/chat.c`, Faz 2 mimarisine (`d_model=384/12 katman/6 baş/d_ff=1024`) ve checkpoint'ine (`lm_wiki_faz2_final.bin`) güncellendi. Bu sırada bir hata bulundu: `checkpoint_load`'un `scratch` arabelleği 256MB'ta sabitti, Faz 2'nin ~400MB'lık checkpoint dosyası için yetersiz kalıp sessizce (crash olmadan, `FALSE` dönerek) başarısız oluyordu — 1GB'a çıkarılarak düzeltildi.
- **Karar (kullanıcı ile, 2026-09-25):** Sonuçlar hazırlanıp kullanıcıya raporlandı; kullanıcının tercihiyle bir sonraki adım (daha fazla adım mı, yeni veri kaynağı mı) için **otomatik ilerlenmedi**, kullanıcının değerlendirmesi bekleniyor.

## 16. Performans Soruşturması ve Çok-Thread'li Veri-Paralel Eğitim (2026-09-25)

**Tetikleyici:** Kullanıcının sezgisel gözlemi — "eğitim sırasında GPU neredeyse hiç kullanılıyormuş gibi hissetmedim, bilgisayarım hiç yorulmadı." Bu gözlem, sistematik bir soruşturmayla **tamamen doğrulandı** ve projenin en büyük performans kaybının kaynağını ortaya çıkardı.

### Soruşturma adımları (sırayla denenip ölçülen hipotezler)

1. **Hipotez: Naif GPU matmul çekirdeği.** `cuda/kernels.cu`'daki `k_matmul` paylaşımlı bellek (shared memory) kullanmıyordu. Klasik "karolu" (tiled) bir versiyon yazıldı, `tests/test_gpu_ops.c` ile doğrulandı (13/13, ~3×10⁻⁶ fark) — **ama gerçek eğitim adımına etkisi ihmal edilebilir düzeydeydi (~%4)**. **Sonuç: YANLIŞ hipotez** — darboğaz çekirdek hesaplama hızı değildi.
2. **Ölçüm: `model/gpu_ops.c`'ye zamanlama enstrümantasyonu eklendi** (`gpu_ops_debug_stats`). Bulgu: adım süresinin sadece **~%21-35'i** GPU gidiş-gelişinde geçiyordu; **~%65-79'u CPU tarafında.**
3. **`training/train_lm.c`'ye evre-bazlı zamanlama eklendi** (ileri/geri/adam/alloc/destroy ayrı ayrı). Bulgu: **Adam optimizer adımı tek başına ~1 saniye** (33.5M parametrenin HER BİRİ için çağrılıyor).
4. **Kök neden bulundu: `runtime/mathlib.c`'deki `m_sqrt`, Newton-Raphson'da 12 iterasyon kullanıyordu.** Karekökte yakınsama KARESEL'dir (her iterasyon doğru basamak sayısını ikiye katlar) — 12'nin yarısından fazlası matematiksel olarak SONUCU DEĞİŞTİRMİYORDU, sadece zaman harcıyordu (yakınsadıktan sonra `y` sabit noktaya ulaşır). **Çözüm:** 6 iterasyona düşürüldü — `tests/test_mathlib.c`'nin 1e-12 toleranslı 36 testi hâlâ **36/36** geçti (doğruluktan ödün yok). **Sonuç: adım süresi ~%16-44 azaldı** (ölçüme göre değişti; adam adımı ~0.58 sn'ye düştü).
5. **Kullanıcının sezgisi takip edildi: "bilgisayar hiç yorulmuyor" → CPU çekirdek kullanımı kontrol edildi.** Makine: **8 çekirdek / 12 mantıksal işlemci** (i5-12450HX). Proje o ana kadar **tamamen tek-thread'di** (hiç `CreateThread` kullanılmamıştı) — CPU kapasitesinin en fazla 1/12'si kullanılıyordu.

### Çözüm: Veri-paralel çok-thread'li eğitim

- **`runtime/thread.h/c` (yeni):** Ham Win32 thread/mutex syscall'ları (`CreateThread`, `WaitForMultipleObjects`, `CreateMutexA` — projenin "syscall'lar araçtır, kütüphane değildir" kuralına göre kabul edilebilir, bkz. Bölüm 1) üzerine ince bir sarmalayıcı.
- **`training/data_parallel.c/h` (yeni):** Her adımda `NUM_WORKERS=8` BAĞIMSIZ dizi, 8 thread'e dağıtılıp paralel ileri+geri yayılım yapıyor. Her thread, orijinal modelin ağırlık DEĞERLERİNİ SADECE OKUYAN ama kendi TAZE (sıfır) gradyan tamponuna yazan bir "gövde/shadow" model kuruyor (`build_shadow_model`, `node_leaf`'in "value paylaşılır, grad taze tahsis edilir" davranışından yararlanılarak — node.c'de HİÇBİR değişiklik gerekmedi). Bu sayede hiçbir kilit gerekmiyor, veri yarışı (race) yok. Thread'ler bitince gradyanları toplanıp ortalaması alınıyor, TEK bir Adam adımı atılıyor.
- **`training/adam.c`:** `adam_step_parallel` eklendi — parametreler birbirinden bağımsız olduğu için Adam adımı da aynı thread havuzuna dağıtılıyor. Yük dengesi için parametre SAYISINA değil SKALER ELEMAN SAYISINA göre açgözlü (greedy) dağıtım yapılıyor (aksi halde tek başına ~%36'sını oluşturan `embed_table`, bir thread'i tek başına boğardı).
- **Bulunan gerçek hata: CUDA bağlamları THREAD-LOKALDİR.** İlk çok-thread'li denemede `CUDA_ERROR_INVALID_CONTEXT` hatası alındı — `cuCtxCreate_v2` ile kurulan bağlam sadece onu kuran thread'de "geçerli" (current). Çözüm: `cuda_backend.c`'ye `cuda_set_current()` eklendi, `model/gpu_ops.c`'nin paylaşılan-bağlam mutex'i içinde HER çağrıda `cuCtxSetCurrent` çağrılıyor (hangi thread kilidi aldıysa bağlamı kendi üzerinde "geçerli" yapıyor).
- **Doğrulama (büyük bir eğitim koşusuna güvenmeden önce kritik):**
  - `tests/test_adam_parallel.c`: `adam_step` ile `adam_step_parallel`'in (kasıtlı olarak çok dengesiz boyutlu 37 parametre üzerinde, 2 adım) BİREBİR aynı sonucu ürettiğini kanıtladı — **6817/6817 başarılı**.
  - `tests/test_data_parallel.c`: 8 thread'li `data_parallel_step`'in ürettiği ortalama gradyanın, AYNI rastgele pencerelerin tek thread'de ardı ardına biriktirilip ortalanmasıyla elde edilen referans gradyanla eşleştiğini kanıtladı — **5808/5808 başarılı**.
  - Mevcut tüm regresyon testleri (mathlib, gpu_ops, lm_model, lm_model_gpu, adam, transformer_block, batch) yeniden çalıştırılıp **hepsi hâlâ geçti**.
- **SONUÇ (gerçek ölçüm, kararlı durum, adım 1-20 ortalaması):** Tek-thread'e göre **~4.4× verim artışı** (8 dizi paralel olarak 5.57 sn'de işlendi; eskiden 1 dizi ~3.08 sn sürüyordu → 8×3.08/5.57≈4.4). Teorik 8× değil, çünkü paylaşılan CUDA bağlamı (mutex ile sıralı) artık 8 thread'in rekabet ettiği bir kaynak — bu, gelecekteki bir optimizasyon hedefi olarak not edildi.
- **Bonus — Bölüm 15'teki "yetersiz eğitim" bulgusuna doğrudan katkı:** Etkin batch boyutu artık 8 (tek dizi değil), yani AYNI adım sayısıyla artık **8 kat daha fazla token** görülüyor — hem hız hem veri kapsamı aynı anda iyileşti.
- **Durum:** Alt yapı hazır ve doğrulanmış; gerçek uzun-süreli devam eğitimi kullanıcının kararına bırakıldı (adım sayısı/süre tercihi netleşmeden başlatılmadı).
- **Gerçek hızlanma (önceki + bugünkü optimizasyonlar birlikte):** ~5,1x — `m_sqrt` düzeltmesi (~%16) × 8-thread veri-paralel eğitim (~4,4x).

## 17. Denenen Ama İşe Yaramayan Bir Optimizasyon — Dürüst Kayıt (2026-09-25)

Kullanıcı daha fazla hızlanma istedi. Bölüm 16'da not edilen kalan darboğaz (8 thread'in TEK paylaşılan CUDA bağlamı için mutex'te sıraya girmesi, ölçülen: GPU-kilit bölgesinde toplam thread-süresi aralığın **%251'i** — yani thread'ler çoğunlukla GPU'yu değil, birbirini bekliyor) üzerine bir hipotez test edildi:

**Hipotez:** Artık 8 gerçek çekirdeğimiz olduğuna göre, katman-içi (attention/FFN) matris çarpımlarını GPU'dan alıp CPU'ya (8 thread paralel) geri taşımak, GPU kilit çekişmesini tamamen ortadan kaldırıp daha hızlı olabilir.

**Uygulama:** `model/lm_model.h/c`'deki tek `use_gpu` bayrağı `use_gpu_layers` ve `use_gpu_output` olarak ikiye ayrıldı (bkz. `training/data_parallel.h/c`, `training/train_lm.c`'deki `USE_GPU_LAYERS`/`USE_GPU_OUTPUT` makroları) — böylece katman-içi ve çıkış-projeksiyonu GPU kullanımı BAĞIMSIZ açılıp kapanabiliyor.

**Sonuç: HİPOTEZ YANLIŞ ÇIKTI.** `USE_GPU_LAYERS=0` (sadece çıkış projeksiyonu GPU'da, GPU çağrı sayısı adım başına 2040'tan 24'e düştü) ile ölçülen adım süresi **13.8 sn** — `USE_GPU_LAYERS=1`'in (5.78 sn) **~2.4 kat DAHA KÖTÜSÜ**. **Ders:** GPU'nun ham hesaplama hızı, paylaşılan bağlam kilidinin getirdiği bekleme maliyetinden hâlâ ağır basıyor — 8 çekirdekli CPU bile, sıralı-GPU-erişimli halinden daha yavaş. Bayrak `USE_GPU_LAYERS=1` olarak GERİ ALINDI (mevcut en hızlı doğrulanmış yapılandırma).

**Not edilen ama henüz denenmeyen gerçek adaylar** (bu bulgudan sonra daha yüksek olasılıklı görünen yönler):
1. **Her thread'e kendi CUDA bağlamı** — şu an TEK paylaşılan bağlam bir mutex ile sıralanıyor (kaba, tüm alloc+kopya+çekirdek+kopya+free döngüsünü kilitliyor). Her thread kendi bağlamını kursa, GPU sürücüsünün kendi (muhtemelen çok daha ince taneli) zamanlayıcısı devreye girer. Orta-büyük bir değişiklik (`model/gpu_ops.c`'nin tek-global-bağlam varsayımını kaldırmak gerekir).
2. **Füzyonlu (fused) matris çarpımları** — `wq/wk/wv` tek bir `[d_model, 3×d_model]` matrise, `w_gate/w_up` tek bir `[d_model, 2×d_ff]` matrise birleştirilirse, katman başına GPU çağrı sayısı 7'den 4'e düşer (~%43 azalma) — çekişmeyi GPU'yu terk etmeden azaltır. Daha küçük, daha az riskli bir degisiklik.
- **Karar (kullanıcı ile):** "Sırayla ikisini de dene" — iki gerçek aday da uygulandı ve ölçüldü.

### Aday 2: Füzyonlu (fused) matris çarpımları — UYGULANDI, KÜÇÜK AMA GERÇEK KAZANÇ

`model/attention.h/c` ve `model/feedforward.h/c` değiştirildi: `AttentionWeights`'teki ayrı `wq/wk/wv` tek bir `w_qkv` ([d_model, 3×d_model]) matrisine, `FeedForwardWeights`'teki `w_gate/w_up` tek bir `w_gate_up` ([d_model, 2×d_ff]) matrisine birleştirildi — tek matris çarpımı, sonra `node_slice_cols` ile bölünüyor. Katman başına GPU çağrısı 7'den 4'e düştü (2040→1176 çağrı/adım). `model/lm_model.c`, `training/data_parallel.c` ve TÜM ilgili testler (`test_transformer_block.c`, `test_batch.c`, `test_lm_model.c`, `test_lm_model_gpu.c`, `test_data_parallel.c`) güncellendi — **hepsi hâlâ geçiyor**.

**ÖNEMLİ YAN ETKİ:** Bu, parametre TENSOR sayısını/şekillerini değiştirdiği için **önceki Faz 2 checkpoint'i (`lm_wiki_faz2_final.bin`) artık uyumsuz** — `checkpoint_load` parametre sayısı uyuşmazlığını güvenle yakalayıp `FALSE` döner (sessiz bozulma yok), ama bir sonraki eğitim koşusu SIFIRDAN başlamak zorunda.

**Ölçüm (kararlı durum, adım 1-20):** 5.566 sn/adım → **5.455 sn/adım** (~%2 iyileşme).

### Aday 1: Thread-başına bağımsız CUDA bağlamı — UYGULANDI, KÜÇÜK EK KAZANÇ

`model/gpu_ops.c` yeniden yazıldı: artık `gpu_ops_init_workers(scratch, ptx_path, num_workers)` ile, eğitim döngüsü BAŞLAMADAN ÖNCE, her worker için BAĞIMSIZ bir CUDA bağlamı (kendi `cuCtxCreate_v2` + kendi PTX modül kopyası) kuruluyor. Her worker thread'i, çalışmaya başlar başlamaz `gpu_ops_set_worker_id(w)` çağırıp KENDİ bağlamını kullanacağını bildiriyor (thread-lokal depolama/`_Thread_local` ile, hiçbir fonksiyon imzası değişmeden). Kendi bağlamı olan bir worker'ın GPU çağrılarında **hiç kilit gerekmiyor** — sadece varsayılan/paylaşılan bağlam (slot 0, tekli-thread araçları `generate.c`/`chat.c` için) mutex korumalı kalıyor.

**Doğrulama:** `tests/test_data_parallel.c` artık `gpu_ops_init_workers` de çağırıp bu YENİ kod yolunu da sınıyor — **5808/5808 hâlâ geçiyor** (tanılama sayaçları için ayrı küçük bir mutex eklendi, çünkü kilitsiz GPU çağrı yolunda bu sayaçlara eş-zamanlı yazma artık korumasızdı).

**Ölçüm (kararlı durum, adım 1-20):** 5.455 sn/adım → **5.234 sn/adım** (~%4 ek iyileşme). Çekişme metriği 209%→173%'e düştü ama hâlâ %100'ün üzerinde — bunun nedeni muhtemelen donanım seviyesinde (tek bir GPU'nun fiziksel hesaplama birimleri, bağımsız yazılım bağlamlarından gelen çekirdekleri de bir ölçüde sıralı işliyor); bu, yazılımla çözülebilecek bir sınırın ötesinde.

### Toplam sonuç (Bölüm 16+17 birlikte)

| Aşama | sn/dizi (karşılaştırılabilir birim) |
|---|---|
| Hiçbir optimizasyon yok | 3.580 |
| + `m_sqrt` düzeltmesi | 3.076 |
| + 8-thread veri-paralel | 0.696 |
| + Füzyonlu matmul | 0.682 |
| + Thread-başına CUDA bağlamı | **0.654** |

**Toplam hızlanma: 3.580/0.654 ≈ 5,47x.** Hem Aday 1 hem Aday 2 tutuldu (küçük ama gerçek, doğrulanmış kazançlar); büyük mimari yeniden yazımlar (gerçek batch tensörleri, tam kernel-seviyesi yeniden tasarım) gerektiren daha büyük kazançlar bu oturumun kapsamı dışında bırakıldı.
- **Durum:** Sonuçlar kullanıcıya raporlandı; gerçek uzun-süreli Faz 2 devam koşusu (mimari değiştiği için SIFIRDAN) kullanıcının kararına bırakıldı.

## 18. Gerçek Tensor-Batching Denemesi — "Projenin Çalışır Formunu Koruyarak Dene ve Ölç" (2026-09-25)

Kullanıcı, kalan en büyük potansiyel kazanım olan **gerçek tensor-batching**'i (8 bağımsız diziyi TEK büyük [8×128, 384] tensörde birleştirip büyük matris çarpımlarını TEK seferde yapmak) denememi istedi — ama **mevcut çalışan sistemi bozmadan**.

### Yeni, test edilmiş temel bloklar (KALICI — düşük risk, her şeyi korudu)

1. **`tensor_slice_rows` / `tensor_concat_rows2`** (`tensor/tensor.c/h`) ve **`node_slice_rows` / `node_concat_rows2`** (`model/model_ops.c/h`) — satır-bazlı dilimleme/birleştirme (sütun-bazlı `slice_cols`'un analogu; satır-majör düzende satırlar bitişik olduğu için sıfır-kopya view olarak DAHA BASİT). **`tests/test_model_ops.c`: 46/46** (yeni 2 test eklendi).
2. **`model/attention.c/h`, `model/transformer_block.c/h`, `model/lm_model.c/h`**: `causal_self_attention`/`transformer_block`/`lm_forward` artık bir `batch_size` parametresi alıyor. **`batch_size=1` ile davranış ORİJİNALİYLE BİREBİR AYNI** — bu, mevcut TÜM testlerin (`test_transformer_block` 30/30, `test_batch` 16/16, `test_lm_model` 12/12, `test_lm_model_gpu` 18/18, `test_data_parallel` 5808/5808) hiç değişmeden geçmesiyle kanıtlandı. **Mevcut çalışan sistem (train_lm.c/data_parallel.c) `batch_size=1` kullanıyor ve HİÇBİR ŞEKİLDE DEĞİŞMEDİ.**
3. **Doğruluk kanıtı:** `tests/test_batched_lm.c` — 3 bağımsız diziyi (a) tek tek ardı ardına ve (b) tek seferde yığınlanmış (`batch_size=3`) işleyip gradyanları karşılaştırıyor. **5808/5808 başarılı** — gerçek tensor-batching MATEMATİKSEL OLARAK DOĞRU.
4. **Yan bulgu — gerçek bir bellek güvenliği hatası bulundu ve düzeltildi:** `runtime/memory.c`'deki `arena_alloc`, rezerv sınırı aşıldığında SESSİZCE `commit`i sınıra sabitleyip yine de aşan bölgeye bir işaretçi DÖNDÜRÜYORDU — bu işaretçiye yazmak sessiz bir SEGFAULT'a yol açıyordu (gerçek ölçekli batch=8 denemesinde yakalandı, `training/probe_batched_step.c`'nin ilk çalıştırması `exit code 139` ile çöktü). **Kök neden düzeltmesi:** `arena_alloc` artık rezerv sınırı aşıldığında AÇIKÇA ve HEMEN bir tanılama mesajıyla durur (`ExitProcess`) — sessiz bellek bozulması riski ortadan kalktı. Tüm regresyon testleri bu düzeltmeyle de geçti.

### Performans ölçümü — SONUÇ: TEK-THREAD gerçek batching, mevcut 8-THREAD yaklaşımından YAVAŞ

`training/probe_batched_step.c` (kalıcı eğitim koduna dahil değil, sadece ölçüm aracı) gerçek Faz 2 ölçeğinde (d_model=384, 12 katman, batch=8) tek-thread'de gerçek batching'i çalıştırdı:

- GPU çağrısı katman başına **32'den 4'e düştü** (beklenen 8× azalma tam gerçekleşti).
- Ama duvar-saati süresi **~23.8 sn/adım** — mevcut 8-thread veri-paralel yaklaşımın (**5.234 sn/adım**) yaklaşık **4,5 KAT YAVAŞI**.

**Kök neden:** Gerçek batching, büyük matris çarpımlarını TEK GPU çağrısında birleştiriyor ama dikkat mekanizmasının skor/softmax/ağırlıklı-toplam kısmı dizi sınırlarını korumak için HÂLÂ dizi-bazında ayrı ayrı (8 dizi × 6 baş = 48 blok/katman) hesaplanmalı — ve bu artık TEK thread'de SIRALI çalışıyor. Mevcut 8-thread yaklaşımında bu 48 blokluk iş 8 GERÇEK ÇEKİRDEĞE dağılıyordu; gerçek batching bunu TEK çekirdeğe geri topluyor. **CPU paralelliğinden kaybedilen, GPU çağrı-azaltmasından kazanılandan kat kat fazla.**

Bunu kazanca çevirmenin yolu (dikkat blok döngüsünü de thread'lere dağıtmak, TEK bir grafın İÇİNDE) otogradı çok-thread'li graf inşasını destekleyecek şekilde yeniden yazmayı gerektirir — bu, bugünkü kapsamın çok üzerinde, ayrı bir katman (muhtemelen Katman 19+) olarak ele alınmalı.

**Karar:** Gerçek tensor-batching, kalıcı eğitim koduna ENTEGRE EDİLMEDİ (mevcut çalışan 8-thread yaklaşım korundu, hiç dokunulmadı). Yeni temel bloklar (satır dilimleme/birleştirme, batch_size parametresi genelleştirmesi, bellek güvenliği düzeltmesi) kalıcı olarak tutuldu çünkü hem test edilmiş/doğru hem de hiçbir mevcut davranışı bozmuyor — ileride (Katman 19+) çok-thread'li graf inşası ele alınırsa hazır bir temel oluşturuyorlar.

## 19. Bulut Ölçeklendirme — Çok-GPU Desteği (2026-09-26)

Kullanıcının Google Cloud hediye kredisiyle daha güçlü bir makinede (nihai seçim: **Windows Server 2025 Datacenter, n1-highcpu-96 — 96 vCPU/48 çekirdek, 86.4 GB RAM, 4×NVIDIA T4**) eğitimi hızlandırma isteği üzerine, mevcut kodun **4 GPU'dan sadece 1'ini kullandığı** fark edildi (`cuda_backend.c`'de cihaz secimi `cuDeviceGet(&c.device, 0)` ile sabitti).

**Çözüm — çok-GPU desteği (geriye-dönük tam uyumlu):**
- **`cuda/cuda_backend.c/h`:** Yeni `cuda_init_on_device(scratch, ptx_path, device_index)` — hangi fiziksel cihazda bağlam kurulacağını seçebiliyor. Eski `cuda_init` artık sadece `cuda_init_on_device(...,0)` çağıran ince bir sarmalayıcı — **davranışı birebir aynı**. Yeni `cuda_device_count()` (cuDeviceGetCount sarmalayıcısı).
- **`model/gpu_ops.c`:** `gpu_ops_init_workers`, artık `cuda_device_count()` ile görünen cihaz sayısını öğrenip worker'ları **sırayla (round-robin) TÜM cihazlara** dağıtıyor (`device_index = worker_id % num_devices`). Tek-GPU'lu bir makinede `num_devices=1` olduğu için HER worker device 0 alır — **Bölüm 16-17'deki davranışla birebir aynı** (yerelde tam doğrulandı: `test_gpu_ops` 13/13, `test_lm_model_gpu` 18/18, `test_data_parallel` 5808/5808, hepsi değişmeden geçti).
- 4 GPU'lu bir makinede (bulut), örneğin 32 worker ile her GPU'ya 8 worker düşer — bu, GPU-başına çekişmeyi (contention) de worker-sayısı/GPU-sayısı kadar azaltır (potansiyel ek kazanç).

**Maliyet uyarısı verildi:** n1-highcpu-96 + 4×T4 tahmini ~$5/saat — kullanıcının kredi bakiyesini kontrol etmesi gerektiği not edildi.

**Durum (güncelleme):** Kullanıcı sonunda **Debian 13, G2 makine ailesi, 32 vCPU, 1×NVIDIA L4** bir VM ile devam etti (Windows/T4 zone kotasi/stok sorunları nedeniyle). Bu, kod tarafında **çok daha büyük bir değişiklik** gerektirdi: proje o ana kadar SADECE Win32 syscall'ları kullanıyordu, Linux'ta bunlar mevcut değil.

### Katman 19 (devam) — POSIX/Linux Taşıması

Her runtime dosyası `#ifdef _WIN32` ile iki platformu da destekleyecek şekilde güncellendi (aynı kaynak dosyalar, derleme zamanında platform otomatik seçilir — MinGW `_WIN32` tanımlar, Linux GCC tanımlamaz):

- **`runtime/memory.c`:** Windows `VirtualAlloc` (reserve/commit ayrımıyla) ↔ Linux `mmap` (anonim/özel eşlem — Linux'ta ayrı bir "commit" adımı yok, sayfalar tembel/lazy olarak fiziksel RAM'e bağlanır, bu yüzden `arena_ensure_committed` Linux'ta doğal olarak no-op).
- **`runtime/file_io.c`:** `CreateFileA/ReadFile/WriteFile/CloseHandle/GetFileSizeEx` ↔ POSIX `open/read/write/close/lseek`.
- **`runtime/console.c`:** `GetStdHandle+WriteFile/ReadFile` ↔ doğrudan `write(1,...)/read(0,...)` (POSIX'te sabit fd'ler sayesinde "handle almaya" bile gerek yok).
- **`runtime/timer.c`:** `QueryPerformanceCounter` ↔ `clock_gettime(CLOCK_MONOTONIC)`.
- **`runtime/thread.c/h`:** `CreateThread/WaitForMultipleObjects/CreateMutexA` ↔ POSIX `<pthread.h>` (`pthread_create/pthread_join/pthread_mutex_t`). **Önemli API değişikliği:** `thread_create` artık deger DÖNDÜRMÜYOR, bir OUT-parametre alıyor (`void thread_create(ThreadHandle* out, ...)`) — çünkü pthread'in trampoline argümanının (fn+arg çifti) STABİL bir adrese ihtiyacı var (döndürülen-deger kopyalarında adres değişirdi). `mutex_lock/unlock` da benzer nedenle artık `MutexHandle*` alıyor (by-value geçirilen bir `pthread_mutex_t` kopyalanırsa senkronizasyon durumu bozulurdu). `ThreadFn`'den `__stdcall` kaldırıldı (x86-64'te zaten anlamsızdı, Linux'ta ise tanımsız/hataya yol açardı).
- **`cuda/cuda_backend.c`:** `ExitProcess` ↔ POSIX `_exit` (libc `exit()` DEĞİL — atexit/stdio-flush davranışı olmayan ham syscall sarmalayıcısı, felsefe olarak `ExitProcess`'in tam esdegeri). `driver_api.h`/`cuda_backend.h` zaten tamamen platform-tarafsızdı (hiç `__stdcall` kullanılmamıştı), değişiklik gerekmedi.
- **`training/adam.c`, `training/data_parallel.c`:** Worker fonksiyonlarından `__stdcall` kaldırıldı, `thread_create` çağrıları yeni OUT-parametre imzasına güncellendi.

**Şans faktörü:** L4, RTX 4050 ile AYNI mimari (Ada Lovelace, compute capability 8.9) — yani mevcut `cuda/kernels.ptx` (sm_89 için derlenmiş) **hiç yeniden derlenmeden doğrudan çalışır**, Linux'ta nvcc/MSVC kurulumuna gerek yok.

**Doğrulama:** Tüm mevcut Windows regresyon testleri (test_layer0 115/115, test_mathlib 36/36, test_prng 8/8, test_gpu_ops 13/13, test_adam_parallel 6817/6817, test_data_parallel 5808/5808) bu taşımadan SONRA yeniden çalıştırıldı ve **hepsi değişmeden geçti** — Windows tarafı hiç bozulmadı. Üretim `train_lm.exe` de yeniden derlenip bir adımlık gerçek koşuyla (kayıp=10.353, önceki tüm temiz koşularla bit-hassasiyetinde aynı) doğruluk teyit edildi.

**Not:** Linux tarafı bu oturumda TEST EDİLEMEDİ (yerelde Linux makinesi yok, bu oturumdan VM'e SSH imkanı yok) — kullanıcının Debian VM'inde derleyip ilk hatayı/sonucu bildirmesi bekleniyor.

## 20. Linux/L4 Sunucusunda İlk Koşu — Taşıma Hataları ve Performans Düzeltmeleri (2026-09-26)

Debian 13 VM (32 vCPU, 1×NVIDIA L4, sürücü 550.163 / CUDA 12.4) üzerinde, sunucudaki Claude Code oturumu tarafından yapıldı.

### Taşıma sırasında bulunan hatalar
- **`tests/test_layer0.c` Linux'ta derlenmiyordu** — test, `file_io`'yu atlayıp doğrudan Win32 `DeleteFileA` çağırıyordu. Kök çözüm: `file_io`'ya platform-bağımsız `file_delete` (Win: `DeleteFileA`, POSIX: `unlink`) eklendi.
- **`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`** — `kernels.ptx` CUDA 13.4 ile (PTX ISA 9.4) üretilmişti, sunucu sürücüsü en fazla CUDA 12.4'ü (ISA 8.4) destekliyor. Bölüm 19'daki "PTX olduğu gibi çalışır" varsayımı mimari (sm_89) için doğruydu ama araç zinciri sürümü için değil. Çözüm: Debian'ın `nvidia-cuda-toolkit` 12.4 paketi kuruldu (sürücüye dokunulmadı), `nvcc -ptx -arch=sm_89 cuda/kernels.cu` ile yeniden üretildi. Eski dosya `cuda/kernels.cuda13.4.ptx.bak`. Yeni PTX, Windows'taki 617 sürücüsünde de çalışır (yeni sürücü eski ISA'yı destekler).
- **Checkpoint sessiz başarısızlığı** — `checkpoint_save` dosya açılamazsa/yazma eksik kalırsa hiçbir şey bildirmiyordu (sunucuda `checkpoints/` klasörü yoktu → 250 adımlık kayıtlar sessizce kaybolacaktı). Artık `bool32` döndürüyor; `train_lm` eğitimden önce klasörün yazılabilirliğini sınıyor.

### Checkpoint'ten devam etme (yeni)
- `checkpoint_save` önce `<yol>.tmp`'ye yazıp `file_rename` (Win: `MoveFileExA`+REPLACE_EXISTING, POSIX: `rename`) ile atomik değiştiriyor — kayıt anında çökme önceki sağlam checkpoint'i bozmuyor.
- `checkpoint_load` önce TÜM dosyayı doğruluyor (sınır kontrolleri, şekiller, birebir toplam boyut, opt istendiyse optimizer durumu), ancak sonra kopyalıyor — başarısızlıkta params/opt'a hiç dokunulmuyor (eskiden kesik dosyada tampon dışı okuma + yarım yükleme vardı).
- `train_lm`: `lm_wiki_faz2_latest.bin` varsa `opt.t` adımından devam; dosya var ama geçersizse sıfırdan başlamayıp hata ile duruyor.
- `tests/test_resume.c`: kesintisiz 6 adım vs 3 adım+kaydet+(farklı tohumlu yeni modele) yükle+3 adım → parametreler ve Adam m/v/t **bit bit aynı** (72/72). Mutasyon kontrolü: devam adımını 1 kaydırmak / Adam durumunu yüklememek testi 67/66 hatayla düşürüyor. `tests/test_adam.c`: kesik/fazla baytlı/optimizersiz dosya reddi, yarım yükleme olmadığı, `.tmp` kalmadığı (17/17).
- **Windows yolu (MoveFileExA) derlenmedi/test edilmedi** — Windows'ta `test_adam` + `test_resume` çalıştırılmalı.

### Performans: 28 worker ilk ölçümde laptop'tan YAVAŞTI
`NUM_WORKERS=28` ile ilk ölçüm: **31,5 sn/adım** (28 dizi → 1,14 sn/dizi; laptop 0,654 sn/dizi). gdb yığın örneklemesi iki kök neden gösterdi:
1. **`cuMemAlloc`/`cuMemFree` sürücü kilidi.** Her `gpu_matmul` 3 ayırma + 3 serbest bırakma yapıyordu; sürücü bunları aynı cihazdaki tüm bağlamlar arasında dahili bir kilitle sıralıyor. Worker'ların çoğu bu kilitte bekliyordu, GPU kullanımı ~%0. (8 worker'da çekişme küçüktü, 28'de baskın hale geldi — Bölüm 17'deki "%173 çekişme" gözleminin gerçek nedeni muhtemelen buydu.) Çözüm: `model/gpu_ops.c`'de slot başına kalıcı, büyüyebilen A/B/C cihaz tamponları.
2. **Tek-thread gradyan indirgemesi + yavaş `tensor_add_inplace`.** İndirgeme (28×33,5M eleman) ana thread'de yapılıyordu ve `tensor_add_inplace` her eleman için çok-boyutlu indeks hesabı (bölme/mod) yapıyordu. Çözüm: `tensor_add_inplace`'e bitişik+aynı-şekil hızlı yolu (aynı işlemler aynı sırada — bit bit aynı); `data_parallel.c`'de indirgeme, her parametrenin eleman aralığı thread'lere bölünerek paralel (eleman başına toplama sırası korunur — bit bit aynı).

**Doğrulama:** 13 regresyon testi değişmeden geçti (tensor 15, autograd 34, model_ops 46, transformer_block 30, batch 16, lm_model 12, batched_lm 5808, adam 17, adam_parallel 6817, gpu_ops 13, lm_model_gpu 18, data_parallel 5808, resume 72). Gerçek koşuda adım 0/20/40 kayıpları (10.358 / 6.593 / 5.621) düzeltme öncesiyle birebir aynı.

| Yapılandırma | sn/adım | dizi/adım | sn/dizi |
|---|---|---|---|
| Laptop, RTX 4050, 8 worker (Bölüm 17) | 5,234 | 8 | 0,654 |
| L4, 28 worker, düzeltme öncesi | 31,5 | 28 | 1,14 |
| **L4, 28 worker, düzeltme sonrası** | **7,73** | 28 | **0,276** |

Düzeltmeler bu sunucuda **~4,1×**, laptop'a göre **~2,4×** dizi verimi getirdi. 3000 adımlık Faz 2 koşusu ≈ 6,4 saat.

## 21. GPU'da Tutulan Eğitim — "Büyük Kazanç" (2026-09-26)

**Tetikleyici:** Bölüm 20'deki düzeltmelerden sonra `perf` profili, kalan sürenin çoğunun hesaplamaya değil CPU tarafındaki genel (strided) bellek doldurma/kopyalamaya gittiğini (`tensor_fill` %36, `tensor_copy_from` %18) ve GPU'nun sadece %23 meşgul olduğunu gösterdi. Kullanıcı küçük CPU yamaları yerine "büyük kazancı" denemeyi seçti.

**Mimari:** Model TAMAMEN GPU'da, tensörler adımlar arasında GPU belleğinde kalıyor:
- 28 dizi tek bir `[B*T=3584, 384]` tensörde birleştiriliyor (gerçek batching — Bölüm 18'deki denemenin aksine dikkat de batch'li GEMM ile GPU'da, CPU'da tek-thread'e düşme yok).
- Ağırlıklar, gradyanlar, Adam m/v ve tüm aktivasyonlar GPU'da; adım başına CPU↔GPU trafiği sadece token id'leri (↑) ve satır kayıpları (↓).
- Geri yayılım otograd grafı yerine elle yazılmış türevlerle (her formül ilgili CPU `backward_*` fonksiyonuyla aynı).
- `cuda/train_kernels.cu` (20 çekirdek, ham CUDA C): batch'li karolu GEMM (64×64 karo, 4×4 register bloklama, NN/NT/TN), RMSNorm ileri/geri, RoPE'li QKV bölme/birleştirme, maskeli softmax ileri/geri, SwiGLU ileri/geri, gömme ileri/geri, çapraz-entropi, Adam, sütun toplamı. Derleme: `nvcc -ptx -arch=sm_89 cuda/train_kernels.cu -o cuda/train_kernels.ptx`.
- **Kurallar korundu:** cuBLAS/cuDNN yok; PTX'te dış fonksiyon çağrısı (libdevice) yok. `exp` (f32, Cody-Waite + Taylor), `log` ve RMSNorm `sqrt` (`m_log`/`m_sqrt`'un birebir f64 portları), Adam `sqrt` (f32 Newton) `mathlib.c` algoritmalarının GPU sürümleri.
- **Determinizm:** hiçbir çekirdek atomik işlem kullanmıyor (gömme gradyanı sütun-başına sabit sırayla, bias/norm gradyanları sabit sıralı sütun toplamıyla) → aynı girdi bit bit aynı çıktı.
- `model/gpu_train.c/h`: `GpuTrainer` — CPU'daki `LMModel`/`AdamOptimizer` bir "ayna": başta/devamda GPU'ya yüklenir, checkpoint'ten önce indirilir → **checkpoint formatı ve devam mantığı değişmedi**, `train_lm` ve `train_lm_gpu` birbirinin checkpoint'inden devam edebilir.
- `training/train_lm_gpu.c`: `train_lm.c` ile aynı sabitler, aynı veri örnekleme (`data_parallel_step`'in worker tohum formülü, adım tohumu `2026+step`), aynı checkpoint yolları. **`train_lm.c` hiç değiştirilmedi.**

**Doğrulama (`tests/test_gpu_train.c`, 154/154):** Referans = mevcut üretim yolu (`data_parallel_step`, CPU otograd).
- Vaka 1 (karo-dışı küçük boyutlar V=50 D=32 H=2 L=2 F=48 T=12 B=5 — GEMM sınır kontrolleri): kayıp birebir, en kötü göreli gradyan farkı 1e-6.
- Vaka 2 (gerçek ölçek V=31769 D=384 H=6 L=12 F=1024 T=128 B=4): kayıp birebir (10.456326), en kötü göreli gradyan farkı 2e-6.
- Adam çekirdeği (aynı gradyanlarla) CPU `adam_step` ile eşleşiyor; aynı girdiyle iki koşu bit bit aynı; upload/download kayıpsız.
- İlk yazımda gradyanların tamamı doğru çıktı (düzeltme gerekmedi); tek hata testin CPU referansı için küçük arena seçmesiydi.

**Gerçek koşu (ayrı klasörde, CPU koşusu aynı anda çalışırken):**
- **0,23 sn/adım** (CPU sürümü 7,73 → **~34×**; Bölüm 20 öncesi 31,5 → ~137×; laptop 5,23 sn/8 dizi → dizi başına ~80×). ~3,3 TFLOPS efektif (adım başına ~750 GFLOP).
- Kayıp eğrisi CPU koşusuyla 600 adım boyunca ±0,005 içinde aynı (adım 0/60/600: 10.358/5.342/3.237 vs 10.358/5.342/3.239) — farkın tek kaynağı kayan nokta toplama sırası.
- 500. adım checkpoint'inden devam: 520–720 arası tüm kayıplar kesintisiz koşuyla aynı.
- 3000 adımlık Faz 2 koşusu ≈ **11,5 dakika** (CPU: ~6,4 saat).

**Olası sonraki kazanımlar (denenmedi):** GEMM verimi (L4 f32 tepe ~30 TFLOPS; şu an ~%11 — daha büyük karo/çift tamponlama), adım başına dizi sayısını artırmak (GPU belleği bol: ~3 GB kullanılıyor), çapraz-entropide logits'i tek geçişte işlemek.
