/* Katman 6 - Nihai Tokenizer (butun katmanlarin birlestigi yer).
 *
 * Vocab duzeni (toplam ~32.000):
 *   [0, N_SUFFIX)                 -> ek kategorisi token'lari (g_suffix_table sirasiyla)
 *   [N_SUFFIX, N_SUFFIX+BPE_N)    -> BPE sembolleri (0-255 ham bayt + ogrenilen birlestirmeler)
 *   [N_SUFFIX+BPE_N, ...)         -> ozel token'lar (PAD, BOS, EOS)
 *
 * Bir kelime icin: once morfolojik ayristirma + istatistiksel
 * belirsizlik giderme denenir (kok+ek dizisi bulunursa kok BPE ile,
 * ekler kendi ozel token'lariyla kodlanir); hic ayristirma yoksa
 * TUM KELIME dogrudan BPE ile kodlanir (fallback). */
#ifndef TOKENIZER_TOKENIZE_H
#define TOKENIZER_TOKENIZE_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/hashset.h"
#include "../runtime/strcounter.h"
#include "bpe.h"

typedef struct TokenizerVocab {
    BpeVocab bpe;
    u32 suffix_base;  /* = 0 */
    u32 bpe_base;     /* = g_suffix_table_count */
    u32 pad_id;
    u32 bos_id;
    u32 eos_id;
    u32 vocab_size;
} TokenizerVocab;

TokenizerVocab tokenizer_init(Allocator* alloc, const char* bpe_merges_path);

#define TOKENIZE_MAX_IDS 128

/* Tek bir kelimeyi (bosluklardan onceden ayrilmis) token ID dizisine
 * cevirir. out_ids en az TOKENIZE_MAX_IDS boyutunda olmalidir.
 * Uretilen token sayisini dondurur. */
u32 tokenize_word(const TokenizerVocab* voc, const StrHashSet* lex, const StrCounter* freq,
                   const char* word, u64 word_len, u32* out_ids);

#endif /* TOKENIZER_TOKENIZE_H */
