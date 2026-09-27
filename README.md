# turkce-llm

Sıfırdan, **üçüncü parti kütüphane kullanmadan** C + ham CUDA C ile yazılmış bir Türkçe dil modeli: kendi bellek ayırıcısı, matematik kütüphanesi, tensör/otograd sistemi, Türkçe morfolojik tokenizer'ı ve CUDA Driver API üzerinden çalışan kendi GPU çekirdekleri (cuBLAS/cuDNN yok).

- Projenin tüm geçmişi, kararları ve ölçümleri: [`PROJE_PLANI.md`](PROJE_PLANI.md)
- Son çalışma oturumunun özeti ve açık işler: [`docs/sohbet_2026-09-27.md`](docs/sohbet_2026-09-27.md)

## Durum (2026-09-27)

| Model | Parametre | Veri | Doğrulama kaybı | Dosya |
|---|---|---|---|---|
| **Faz 3** (tamamlandı) | 33.490.176 | Türkçe Vikipedi, ~1,0 milyar token | Vikipedi **1,376** | `lm_wiki_faz3_best.bin` |
| **Faz 4** (adım 1.500 / 25.000'de durduruldu) | 33.490.176 | %65 film altyazısı + %35 Vikipedi | altyazı 2,197 · Vikipedi 1,441 | `lm_faz4_best.bin` |

Mimari: d_model=384, 12 katman, 6 baş, SwiGLU d_ff=1024, RoPE, RMSNorm, bağlam 1024, vocab 31.769 (morfolojik ek token'ları + bayt BPE).

Geliştirme ortamı: Debian 13, NVIDIA L4 (sm_89), sürücü 550 (CUDA 12.4), GCC 14.2, nvcc 12.4 (yalnızca `.cu` değişirse gerekir).

## Büyük dosyalar (GitHub Releases)

Depo gizli olduğu için dosyaları `gh` (GitHub CLI) ile indirin. Hepsini doğru klasörlere koyup SHA256 ile doğrulayan betik:

```bash
gh auth login                 # bir kez
./scripts/indir.sh            # çalışmak için gerekenler (~5 GB)
./scripts/indir.sh hepsi      # + ham metinler, eski checkpoint'ler, loglar
```

| Release | Dosya | Nereye | Açıklama |
|---|---|---|---|
| `faz3-final` | `lm_wiki_faz3_best.bin` | `checkpoints/` | Faz 3 en iyi model (val 1,376, adım 69.000) |
| | `lm_wiki_faz3_final.bin` | `checkpoints/` | Faz 3 son adım (70.000) |
| | `lm_wiki_faz3_best_ctx128.bin` | `checkpoints/` | Bağlam 128 döneminin en iyisi |
| | `lm_wiki_faz2_latest.bin` | `checkpoints/` | Faz 2 (eski) |
| | `wikipedia_tokens.bin` | `data/raw/` | 494.603.976 token (u32) |
| | `wikipedia_corpus.txt` | `data/raw/` | Temizlenmiş Vikipedi metni |
| | `faz3_log.txt`, `train_log*.txt` | `training/` | Eğitim logları |
| `faz4-adim1500` | `lm_faz4_best.bin` | `checkpoints/` | Faz 4 en iyi (adım 1.500) |
| | `lm_faz4_latest.bin` | `checkpoints/` | Faz 4 devam noktası (adım 1.000, Adam durumu dahil) |
| | `subtitle_tokens.bin` | `data/raw/` | 500.591.668 altyazı token'ı (u32) |
| | `subtitle_corpus.txt` | `data/raw/` | Temizlenmiş altyazı metni (1,11 GB) |
| | `faz4_log.txt`, `subtitle_*_log.txt` | `training/`, `data/raw/` | Loglar |
| `veri-2026-09-26` | `trwiki-latest-pages-articles.xml.bz2` | `data/raw/` | Orijinal Vikipedi dökümü (2026-09-01) |

Altyazı verisi OPUS OpenSubtitles v2024'ten türetilmiştir (araştırma amaçlı; kamuya açık yeniden dağıtmayın). Ham arşiv (2,3 GB, GitHub sınırını aşar) buradan yeniden indirilebilir: `https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/tr.txt.gz`. Vikipedi içeriği CC BY-SA 4.0.

## Derleme

```bash
./scripts/build_all.sh                      # tüm programlar ve testler (GCC + libcuda)
PTX=1 ./scripts/build_all.sh                # + GPU çekirdeklerini yeniden derle (nvcc)
PTX=1 ARCH=sm_90 ./scripts/build_all.sh     # başka GPU: A100 sm_80, H100 sm_90
```

`cuda/*.ptx` depoda hazır (sm_89). L4 / RTX 40xx dışındaki bir GPU'da `PTX=1 ARCH=...` ile yeniden derleyin.

Testler (hepsi `Sonuc: N basarili, 0 basarisiz` yazmalı):

```bash
for t in tests/test_*; do [ -x "$t" ] && echo "$t: $("$t" | tail -1)"; done
```

## Çalıştırma (depo kökünden)

```bash
./training/generate     # "Türkiye'nin başkenti" tohumundan 80 token üretir
./training/chat         # yazdığınız metni devam ettirir; çıkmak için boş satır / exit
```

Her ikisi `checkpoints/lm_wiki_faz3_best.bin`'i yükler (dosyanın başındaki `CKPT_PATH`); Faz 4 modelini denemek için `checkpoints/lm_faz4_best.bin` yapıp yeniden derleyin.

## Eğitim

Uzun koşuları `tmux` içinde başlatın (SSH kopsa da sürer):

```bash
# Faz 4'e devam (lm_faz4_latest.bin'den, adım 1.000'den sürer):
tmux new -s egitim4 "./training/train_faz4 2>&1 | tee -a training/faz4_log.txt"

# Faz 3'ü sıfırdan (lm_wiki_faz3_latest.bin yoksa):
tmux new -s egitim "./training/train_lm_gpu 2>&1 | tee -a training/faz3_log.txt"
```

Veriyi sıfırdan üretmek (bkz. `PROJE_PLANI.md` Bölüm 12, 20, 24):

```bash
# Vikipedi: dökümü aç -> temizle -> tokenize
bunzip2 -k data/raw/trwiki-latest-pages-articles.xml.bz2
./tokenizer/build_wiki_corpus && ./tokenizer/tokenize_corpus
# Altyazı: ham arşivi indir+aç -> temizle/tekrar ayıkla/örnekle -> tokenize
gunzip -k data/raw/opensubtitles_tr_v2024.txt.gz
./tokenizer/build_subtitle_corpus
./tokenizer/tokenize_corpus data/raw/subtitle_corpus.txt data/raw/subtitle_tokens.bin
```
