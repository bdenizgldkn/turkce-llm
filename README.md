# turkce-llm

Sıfırdan, **üçüncü parti kütüphane kullanmadan** C + ham CUDA C ile yazılmış bir Türkçe dil modeli: kendi bellek ayırıcısı, matematik kütüphanesi, tensör/otograd sistemi, Türkçe morfolojik tokenizer'ı ve CUDA Driver API üzerinden çalışan kendi GPU çekirdekleri (cuBLAS/cuDNN yok).

Projenin tüm geçmişi, kararları ve ölçümleri: [`PROJE_PLANI.md`](PROJE_PLANI.md).

## Büyük dosyalar

GitHub'ın dosya boyutu sınırı nedeniyle veri ve checkpoint'ler depoda değil, **Releases** sayfasında:

| Dosya | Nereye | Açıklama |
|---|---|---|
| `wikipedia_tokens.bin` | `data/raw/` | 494.603.976 token (u32), eğitim verisi |
| `wikipedia_corpus.txt` | `data/raw/` | Temizlenmiş Vikipedi metni (tokenize girdisi) |
| `trwiki-latest-pages-articles.xml.bz2` | `data/raw/` | Orijinal Türkçe Vikipedi dökümü (2026-09-01) |
| `lm_wiki_faz2_latest.bin` | `checkpoints/` | Model + Adam durumu checkpoint'i |

Veri sıfırdan da üretilebilir: `tokenizer/build_wiki_corpus` → `tokenizer/tokenize_corpus` (bkz. `PROJE_PLANI.md` Bölüm 12, 20).

## Derleme (Linux, GCC + NVIDIA sürücüsü)

```bash
# GPU'da tutulan eğitim (Bölüm 21, ~0,23 sn/adım L4'te)
gcc -Wall -Wextra -std=c11 -O2 -pthread -o training/train_lm_gpu training/train_lm_gpu.c \
  model/gpu_train.c model/lm_model.c model/embedding.c model/loss.c model/transformer_block.c \
  model/attention.c model/feedforward.c model/model_ops.c model/rope.c model/gpu_ops.c \
  cuda/cuda_backend.c training/adam.c training/checkpoint.c autograd/node.c tensor/tensor.c \
  runtime/memory.c runtime/console.c runtime/mathlib.c runtime/prng.c runtime/file_io.c \
  runtime/timer.c runtime/strutil.c runtime/thread.c -I. -lcuda
./training/train_lm_gpu   # checkpoints/lm_wiki_faz2_latest.bin varsa oradan devam eder
```

`cuda/*.ptx` dosyaları depoda hazır (sm_89, PTX ISA 8.4 / CUDA 12.4+ sürücü). Çekirdekleri değiştirirseniz:

```bash
nvcc -ptx -arch=sm_89 cuda/kernels.cu -o cuda/kernels.ptx
nvcc -ptx -arch=sm_89 cuda/train_kernels.cu -o cuda/train_kernels.ptx
```

Testlerin derleme komutları için `CLAUDE_CODE_SUNUCU_PROMPT.md` ve `PROJE_PLANI.md`'ye bakın (ör. `tests/test_gpu_train.c`: GPU eğitimini CPU otograd referansına karşı doğrular).
