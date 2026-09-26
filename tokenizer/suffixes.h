/* Katman 6 - Ek Tablosu.
 *
 * turkish_phon.h'deki sablon cozucu uzerine kurulu, somut Turkce ek
 * envanteri. V1: en yaygin/uretken cekim ve yapim ekleri (~20 adet).
 * TAM ENVANTER DEGILDIR -- PROJE_PLANI.md'de karar verildigi gibi
 * "tum envanter" hedefine giden genisletilebilir bir baslangic
 * kumesidir; nadir/arkaik ekler ileride buraya eklenecektir.
 *
 * NOT: Eklerin birbirine hangi sirada baglanabilecegi (morfotaktik
 * sira -- orn. coğul+iyelik+hal, negatif+zaman+sahis) V1'de
 * ZORLANMAZ; ayristirici herhangi bir sirada dener. Bu bilinen bir
 * basitlestirmedir (bkz. Riskler). */
#ifndef TOKENIZER_SUFFIXES_H
#define TOKENIZER_SUFFIXES_H

#include "../runtime/types.h"

typedef enum SuffixClass {
    SFX_NOUN_INFLECTION,   /* cogul, iyelik, hal ekleri */
    SFX_DERIVATIONAL,      /* yapim ekleri (-lık, -sız, -lı, -cı) */
    SFX_VERB_INFLECTION    /* fiil cekim ekleri */
} SuffixClass;

/* Turkce isim cekiminin bilinen sabit sirasi: KOK-COGUL-IYELIK-HAL,
 * her yuva (slot) en fazla bir kez kullanilabilir. analyzer.c bu
 * sirayi ve tekil kullanimi zorlar (bkz. class_is_compatible benzeri
 * kontrol) -- boylece orn. "iyelik-3cogul"+"iyelik-1cogul" gibi iki
 * iyelik ekinin ust uste gelmesi (gecersiz ama yuzeyde bazen ayni
 * sonucu ureten) bir zincir asla uretilmez. */
typedef enum NounSlot {
    NOUN_SLOT_NONE = 0,   /* bu kurala tabi degil (turetim/fiil ekleri) */
    NOUN_SLOT_PLURAL = 1,
    NOUN_SLOT_POSSESSIVE = 2,
    NOUN_SLOT_CASE = 3
} NounSlot;

typedef struct Suffix {
    const char* template_str;
    const char* gloss;
    char buffer_consonant;                 /* 0 = yok */
    bool32 triggers_softening;
    bool32 drop_initial_vowel_after_vowel;
    SuffixClass klass;
    NounSlot noun_slot;
} Suffix;

extern const Suffix g_suffix_table[];
extern const u32 g_suffix_table_count;

#endif /* TOKENIZER_SUFFIXES_H */
