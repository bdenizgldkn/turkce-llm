# Sunucudaki Claude Code için Görev Talimatı

Bu metni olduğu gibi Claude Code'a yapıştır. Kendi başına (bu konuşmanın bağlamı olmadan) anlaşılabilir olacak şekilde yazıldı.

---

## Bağlam

Sıfırdan (üçüncü parti kütüphane KULLANMADAN) C + ham CUDA C ile yazılmış bir Türkçe LLM projesi üzerinde çalışıyorum. Proje şu ana kadar Windows'ta (MinGW-w64 GCC + CUDA + RTX 4050) geliştirildi ve tamamen çalışır durumda. Şimdi bu Debian sunucuya (32 vCPU, 1×NVIDIA L4 GPU) taşıyorum çünkü burada çok daha fazla CPU çekirdeği ve GPU gücü var.

**Kritik proje kuralları (bunlara MUTLAKA uy, ihlal etme):**
- **Hiçbir üçüncü parti kütüphane yok.** NumPy, PyTorch, cuBLAS, cuDNN, pthread'in ÜZERİNE kurulu hiçbir soyutlama kütüphanesi (örn. OpenMP, TBB) — YASAK.
- **Standart kütüphane minimal.** `malloc/free` yok (kendi arena/free-list ayırıcımız var), `printf` yok (kendi `console_write` fonksiyonlarımız var), matematik fonksiyonları (`sqrt/exp/log/sin/cos`) `<math.h>`'den DEĞİL, kendi yazdığımız Taylor serisi/Newton-Raphson implementasyonlarından (`runtime/mathlib.c`) geliyor.
- **İzin verilen tek istisna: ham işletim sistemi syscall'ları ve CUDA Driver API.** Bunlar "araç" sayılıyor (derleyici gibi), "kütüphane" değil. Yani: POSIX `<unistd.h>`, `<fcntl.h>`, `<sys/mman.h>`, `<pthread.h>`, `<time.h>` — bunlar tamam (zaten minimal, ham syscall arayüzleri). Ama `<pthread.h>`'nin üzerine kurulu bir "thread pool" kütüphanesi gibi bir şey EKLEMEK yasak — biz kendi `runtime/thread.c` sarmalayıcımızı zaten yazdık, onu kullan.
- **CUDA:** `<cuda.h>` dahil edilmiyor (kendi minimal bildirimlerimiz `cuda/driver_api.h`'de). `cuBLAS/cuDNN` KESİNLİKLE yasak — tüm GPU çekirdekleri (`cuda/kernels.cu`) sıfırdan bizim yazdığımız ham CUDA C kodu.
- Bir hata/eksiklik bulursan **kök nedenini düzelt, geçici yama/workaround yapma.** Bu proje boyunca tutarlı şekilde izlenen yaklaşım bu oldu.
- Yeni bir şey yazarsan veya bir hata düzeltirsen, mümkünse **test ile doğrula** (proje genelinde sayısal gradyan kontrolü / bit-hassasiyetinde eşdeğerlik testi metodolojisi kullanılıyor, `tests/` klasöründeki mevcut testlere bak, aynı üslupta yaz).

## ÖNCE BU: Kod dosyaları sunucuda YOK, sen indireceksin

Bu makinede proje dosyaları henüz yok — bana (kullanıcı) `proje_kod.tar.gz` (111 KB, sadece kaynak kod, veri/checkpoint hariç) dosyasını verdim, onu bir şekilde bu sunucuya senin taşıman gerekiyor (Google Drive linki / `gsutil cp gs://...` / GCP konsolu SSH penceresinin dosya yükleme düğmesi / `scp` — hangisi mümkünse). Bana nasıl taşıyacağını sorabilir, ya da doğrudan bir yöntem önerebilirsin.

Taşındıktan sonra:
```bash
mkdir -p ~/proje && cd ~/proje
tar -xzf /yol/proje_kod.tar.gz
ls   # runtime/ tensor/ autograd/ model/ tokenizer/ cuda/ training/ tests/ gormelisin
```

**Eğitim verisi (`data/raw/wikipedia_tokens.bin`, ~1.98 GB) bu pakette YOK** (çok büyük, ayrı taşınmalı) — bunu da bir şekilde (GCS bucket önerilir, büyük dosyalar için scp'den daha güvenilir) sunucuya getirmemiz gerekecek. Şimdilik **1-4. adımları (derleme+test) bu dosya OLMADAN yapabilirsin** — sadece 5. adımdaki gerçek eğitim koşusu için gerekli. O adıma gelince bana/kullanıcıya hatırlat.

## Şu anki durum

Proje bugün Windows'tan Linux'a taşınacak şekilde güncellendi: `runtime/memory.c`, `runtime/file_io.c`, `runtime/console.c`, `runtime/timer.c`, `runtime/thread.c/h`, `cuda/cuda_backend.c` dosyalarının hepsi `#ifdef _WIN32` ile hem Windows (VirtualAlloc/CreateFileA/CreateThread/...) hem POSIX (mmap/open-read-write/pthread/clock_gettime) yolunu içerecek şekilde yazıldı. **Ama bu POSIX/Linux tarafı HİÇ TEST EDİLMEDİ** — yerelde (Windows) geliştirildi, Linux'ta ilk kez sen derleyeceksin. Yani muhtemelen ilk denemede bir-iki derleme hatası çıkacak (platform farkları, eksik include, tip uyuşmazlığı vb.) — bunları KÖK NEDENİNDEN düzelt.

Ayrıca proje son olarak **8 çekirdekli bir Windows makinede** veri-paralel çok-thread'li eğitim (`training/data_parallel.c`, `NUM_WORKERS=8`) ve çoklu-GPU desteği (`model/gpu_ops.c`'de her worker kendi CUDA bağlamını kurup mevcut TÜM GPU'lara sırayla/round-robin dağılıyor) ile ~5,47x hızlanma elde etti. Bu sunucuda **32 vCPU + 1×L4** var — `NUM_WORKERS`'ı buna göre ayarlaman gerekecek.

**Şans faktörü:** L4, geliştirmenin yapıldığı RTX 4050 ile AYNI mimari (Ada Lovelace, compute capability 8.9). Yani `cuda/kernels.ptx` dosyası (önceden derlenmiş, `sm_89` hedefli) OLDUĞU GİBİ çalışmalı — yeniden derlemek (nvcc gerekmiyor) GEREKMEMELİ. Ama emin olmak için doğrula (aşağıdaki adımlarda var).

## Görev — sırayla yap

### 1. Ortamı doğrula
```bash
nvidia-smi                    # L4 gorunuyor mu, surucu versiyonu ne
gcc --version                 # GCC var mi (yoksa: sudo apt install -y gcc make)
ls /usr/lib/x86_64-linux-gnu/libcuda* 2>/dev/null || find / -name "libcuda.so*" 2>/dev/null
```
`libcuda.so` bulunamazsa (sadece `libcuda.so.1` varsa), derlerken `-L<yol> -lcuda` yerine `-l:libcuda.so.1` kullanman gerekebilir, ya da bir symlink oluştur.

### 2. Proje yapısını incele
Proje kök dizininde (muhtemelen `Proje-Dosyalari` adlı bir klasörde, yanında ayrı bir `Proje-Planlari` klasöründe `PROJE_PLANI.md` olabilir — varsa MUTLAKA oku, projenin tüm geçmişi/kararları orada, özellikle **Bölüm 16-19** çok-thread/çok-GPU/Linux taşıması hakkında). `runtime/`, `tensor/`, `autograd/`, `model/`, `tokenizer/`, `cuda/`, `training/`, `tests/` klasörlerine göz at, her dosyanın başındaki yorum bloğu ne işe yaradığını açıklıyor.

### 3. Katman katman derle ve test et (bağımlılık sırasıyla — bir katman patlarsa bir sonrakine geçme, önce onu düzelt)

Her komut için: **derleme hatası çıkarsa kök nedenini bul ve düzelt** (platform farkı, eksik `#include`, tip uyuşmazlığı — Windows'ta çalışan ama Linux'ta çalışmayan bir şey varsa muhtemelen gözden kaçan bir `#ifdef _WIN32` bloğu ya da POSIX'te farklı davranan bir syscall'dır). Düzelttikten sonra HEM bu testin HEM daha önce geçen testlerin hâlâ geçtiğini doğrula (regresyon riski).

```bash
cd Proje-Dosyalari   # gercek yolu kontrol et

# Katman 0-2 (temel: bellek, dosya I/O, konsol, matematik, PRNG) -- GPU/thread YOK, en basit katman
gcc -Wall -Wextra -std=c11 -O2 -o tests/test_layer0 tests/test_layer0.c runtime/memory.c runtime/file_io.c runtime/console.c runtime/strutil.c runtime/utf8.c runtime/hashset.c runtime/strcounter.c runtime/sort.c -I.
./tests/test_layer0    # beklenen: 115/115

gcc -Wall -Wextra -std=c11 -O2 -o tests/test_mathlib tests/test_mathlib.c runtime/mathlib.c runtime/console.c -I.
./tests/test_mathlib   # beklenen: 36/36

gcc -Wall -Wextra -std=c11 -O2 -o tests/test_prng tests/test_prng.c runtime/prng.c runtime/mathlib.c runtime/console.c -I.
./tests/test_prng      # beklenen: 8/8

# Thread katmani (pthread) -- ilk gercek Linux-spesifik test
gcc -Wall -Wextra -std=c11 -O2 -pthread -o tests/test_adam_parallel tests/test_adam_parallel.c training/adam.c autograd/node.c tensor/tensor.c runtime/memory.c runtime/console.c runtime/mathlib.c runtime/prng.c runtime/thread.c -I.
./tests/test_adam_parallel   # beklenen: 6817/6817

# GPU katmani -- CUDA Driver API'nin gercekten calistigini dogrular
gcc -Wall -Wextra -std=c11 -O2 -pthread -o tests/test_gpu_ops tests/test_gpu_ops.c model/gpu_ops.c cuda/cuda_backend.c autograd/node.c tensor/tensor.c runtime/memory.c runtime/console.c runtime/mathlib.c runtime/prng.c runtime/timer.c runtime/file_io.c runtime/thread.c -I. -lcuda
./tests/test_gpu_ops    # beklenen: 13/13, ve bir "CPU vs GPU hizlanma" olcumu yazdirir (L4'te muhtemelen RTX4050'den DAHA HIZLI cikacak)

# Tam model + coklu-thread veri-paralel egitim dogrulugu (EN KAPSAMLI test -- bu gecerse pipeline saglam demektir)
SRC_MODEL="model/lm_model.c model/embedding.c model/loss.c model/transformer_block.c model/attention.c model/feedforward.c model/model_ops.c model/rope.c model/gpu_ops.c cuda/cuda_backend.c autograd/node.c tensor/tensor.c runtime/memory.c runtime/console.c runtime/mathlib.c runtime/prng.c runtime/file_io.c runtime/timer.c runtime/thread.c"
gcc -Wall -Wextra -std=c11 -O2 -pthread -o tests/test_lm_model_gpu tests/test_lm_model_gpu.c $SRC_MODEL -I. -lcuda
./tests/test_lm_model_gpu   # beklenen: 18/18

gcc -Wall -Wextra -std=c11 -O2 -pthread -o tests/test_data_parallel tests/test_data_parallel.c training/data_parallel.c $SRC_MODEL -I. -lcuda
./tests/test_data_parallel  # beklenen: 5808/5808 -- bu, coklu-thread + coklu-GPU-baglami mekanizmasinin MATEMATIKSEL OLARAK DOGRU oldugunu kanitliyor
```

Bunların HEPSİ geçmeden bir sonraki adıma geçme.

### 4. `NUM_WORKERS`'ı sunucuya göre ayarla

`training/train_lm.c` dosyasında `#define NUM_WORKERS 8u` satırını bul. Bu makine 32 vCPU olduğu için `28` veya `30` yap (birkaç çekirdeği sistem için bırak). Bunun dışında bu dosyada BAŞKA HİÇBİR ŞEYİ değiştirme (mimari sabitleri — `D_MODEL=384, NUM_LAYERS=12` vb. — olduğu gibi kalsın).

### 5. Gerçek eğitim programını derle ve KISA bir tanılama koşusu yap

```bash
gcc -Wall -Wextra -std=c11 -O2 -pthread -o training/train_lm \
  training/train_lm.c training/data_parallel.c \
  model/lm_model.c model/embedding.c model/loss.c model/transformer_block.c \
  model/attention.c model/feedforward.c model/model_ops.c model/rope.c model/gpu_ops.c \
  cuda/cuda_backend.c training/adam.c training/checkpoint.c \
  autograd/node.c tensor/tensor.c \
  runtime/memory.c runtime/console.c runtime/mathlib.c runtime/prng.c runtime/file_io.c \
  runtime/timer.c runtime/strutil.c runtime/thread.c \
  -I. -lcuda
```

`data/raw/wikipedia_tokens.bin` dosyasının (~1.98 GB, 494.603.976 token) var olduğunu doğrula (yoksa bana haber ver, ayrıca taşınması gerekiyor).

Çalıştır:
```bash
./training/train_lm
```

**İlk 40-60 adımın çıktısını (adım, kayıp, gecen süre satırları) olduğu gibi kaydet** — ben (başka bir Claude Code oturumunda, bu makineye SSH erişimim yok) bu sonuçları senden/kullanıcıdan alıp gerçek hızlanmayı hesaplayacağım. Adım 0 "soğuk başlangıç" olduğu için biraz yavaş olabilir, adım 20-40 civarı "kararlı durum" hızını gösterir — o kısmı özellikle işaretle.

10-20 adım civarı beklenen sonucu gördükten sonra (kayıp düşüyor, program çökmüyor) **programı durdurabilirsin** (Ctrl+C) — uzun bir eğitim koşusunu şimdi başlatmana gerek yok, önce hız/doğruluk ölçümü yeterli.

## Rapor formatı

İşin sonunda şunları özetle:
1. Hangi Linux-spesifik derleme hataları çıktı, nasıl düzelttin (kısa, dosya+satır referanslı)
2. Test sonuçları (hangi test kaç/kaç geçti)
3. `test_gpu_ops`'un CPU-vs-GPU hızlanma ölçümü (L4'ün ham gücünü gösterir)
4. `train_lm`'in adım 0 ve adım ~20-40 civarındaki "gecen" (elapsed) sürelerinden hesaplanan sn/adım
5. Karşılaşılan ama çözemediğin bir şey varsa (olası ama beklenmiyor)

Bu raporu bana (kullanıcı üzerinden) ilet, birlikte sonraki adıma (uzun eğitim koşusunu başlatmak) karar vereceğiz.
