/* Katman 6 - Morfolojik Ayristirici.
 *
 * Bir yuzey kelimeyi (orn. "kitaplarımızdan"), kok sozlugu + ek
 * tablosunu kullanarak kok+ek dizisine ayirir. Yontem: "ureterek
 * dogrulama" (generate-and-test) -- her olasi kok uzunlugu ve her ek
 * icin, o kokten o ekle SURFACE FORM'UN NE OLMASI GEREKTIGINI (fonoloji
 * motoruyla) hesaplayip, gozlemlenen kelimeyle tam eslesip eslesmedigini
 * kontrol eder. Boylece yumusama/uyum kurallarini "tersine cevirmeye"
 * calismak yerine ILERI YONDE uretip dogrular -- bu daha az hataya
 * aciktir.
 *
 * Bir kelimenin BIRDEN FAZLA gecerli ayristirmasi olabilir (belirsizlik);
 * bu katman hepsini dondurur, secim Katman 6'nin istatistiksel
 * belirsizlik giderme bileseni tarafindan yapilacaktir (bkz. PROJE_PLANI.md). */
#ifndef TOKENIZER_ANALYZER_H
#define TOKENIZER_ANALYZER_H

#include "../runtime/types.h"
#include "../runtime/hashset.h"

#define ANALYZER_MAX_SUFFIXES 6
#define ANALYZER_MAX_PARSES 16
#define ANALYZER_MAX_ROOT_BYTES 64

typedef struct Parse {
    char root[ANALYZER_MAX_ROOT_BYTES];
    u64 root_len;
    u32 suffix_indices[ANALYZER_MAX_SUFFIXES]; /* g_suffix_table icindeki indeksler, KOKTEN DISA dogru sirada */
    u32 num_suffixes;
} Parse;

typedef struct AnalysisResult {
    Parse parses[ANALYZER_MAX_PARSES];
    u32 num_parses;
    bool32 truncated_by_budget; /* guvenlik adim butcesi tukendiyse TRUE */
} AnalysisResult;

/* word: UTF-8 yuzey form, len: bayt uzunlugu. */
AnalysisResult analyze_word(const StrHashSet* lexicon, const char* word, u64 len);

#endif /* TOKENIZER_ANALYZER_H */
