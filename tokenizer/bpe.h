/* Katman 6 - BPE (Byte-Pair Encoding) fallback.
 *
 * Iki amaca hizmet eder:
 *  (1) Morfolojik ayristiricinin HIC analiz edemedigi kelimeler icin
 *      (ozel isimler, yabanci kelimeler, yazim hatalari) dogrudan
 *      yedek (fallback) tokenizasyon,
 *  (2) Ayristirici tarafindan bulunan KOKUN KENDISININ, 294K'lik acik
 *      uclu kok kumesini sabit (32K) bir sozlukte temsil etmenin
 *      mekanizmasi -- sik kokler tek parca (token) olarak ogrenilir,
 *      nadir kokler birden fazla alt-parcaya bolunur.
 *
 * Bayt-seviyeli (byte-level) BPE: temel birim 256 olasi bayt degeridir,
 * bu yuzden hicbir girdi "bilinmeyen" (unknown) token uretmez.
 * cuBLAS/cuDNN gibi degil, HuggingFace tokenizers gibi bir kutuphane
 * de kullanilmaz -- egitim ve kodlama algoritmasi sifirdan yazilmistir. */
#ifndef TOKENIZER_BPE_H
#define TOKENIZER_BPE_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/strcounter.h"

#define BPE_MAX_WORD_BYTES 192

typedef struct BpeMerge {
    u32 left;
    u32 right;
    u32 new_id;
} BpeMerge;

typedef struct BpeVocab {
    u8** symbol_bytes;   /* symbol_bytes[id] -> bu sembolun tam bayt karsiligi */
    u32* symbol_lens;
    u32 num_symbols;     /* 256 (bayt) + num_merges */
    BpeMerge* merges;    /* egitim sirasiyla == kodlama onceligi */
    u32 num_merges;
    StrCounter merge_rank; /* 8-baytlik (left,right) anahtar -> rank+1 (0=yok) */
} BpeVocab;

/* words[i]/word_lens[i]/freqs[i]: egitim korpüsü (kelime + bayt uzunlugu +
 * agirlik/frekans). target_vocab_size: 256'dan buyuk olmalidir (kac
 * sembole ulasilana kadar birlestirme yapilacagi). */
BpeVocab bpe_train(Allocator* alloc, const char** words, const u64* word_lens,
                    const u64* freqs, u64 num_words, u32 target_vocab_size);

/* word'u ogrenilen birlestirmelerle sembol ID dizisine kodlar.
 * out_ids en az word_len eleman alabilmelidir (en kotu durumda hic
 * birlesme olmaz, her bayt kendi sembolu kalir). Uretilen sembol
 * sayisini dondurur. */
u32 bpe_encode(const BpeVocab* vocab, const char* word, u64 word_len, u32* out_ids);

/* Birlestirme listesini "left right\n" satirlari halinde kaydeder/yukler
 * (256 temel bayttan yeniden oynatarak tum sembol tablosunu kurar). */
void bpe_save(const BpeVocab* vocab, const char* path);
BpeVocab bpe_load(Allocator* alloc, const char* path);

#endif /* TOKENIZER_BPE_H */
