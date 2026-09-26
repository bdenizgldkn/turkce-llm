/* Katman 6 - Turkce fonoloji kurallari.
 *
 * Buyuk/kucuk unlu uyumu, unsuz yumusamasi (p/ç/t/k -> b/c/d/ğ),
 * unsuz sertlesmesi/sedasizlesmesi (D/C tipi ekler) ve kaynastirma
 * unsuzleri (y/n/s/ş) icin temel kurallar.
 *
 * BILINEN SINIRLAMA: Bu kurallar DUZENLI (regular) davranisi kapsar.
 * Turkce'de bazi koklerin (ozellikle tek heceli ve/veya alinti
 * kelimeler: "ok"->"oku" degil "oğu" gibi) DUZENSIZ (istisna) bir
 * davranisi vardir ve bunlar kok bazinda ayrica isaretlenmelidir.
 * V1'de istisna listesi yoktur -- PROJE_PLANI.md Riskler bolumunde
 * not edilmistir. */
#ifndef TOKENIZER_TURKISH_PHON_H
#define TOKENIZER_TURKISH_PHON_H

#include "../runtime/types.h"

/* Turkce'ye ozgu buyuk/kucuk harf donusumu (I->ı, İ->i dahil; ASCII
 * A-Z ve Ç/Ğ/Ö/Ş/Ü icin dogru es-esleme). Harf degilse degistirmeden
 * dondurur. */
u32 tr_to_lower_cp(u32 cp);

/* Turkce alfabesindeki bir harf mi (ASCII a-z/A-Z + ş/ğ/ç/ö/ü/ı/İ ve
 * buyukleri)? Kelime sinirlarini bulmak (tokenizasyon) icin kullanilir. */
bool32 tr_is_letter(u32 cp);

bool32 tr_is_vowel(u32 cp);
bool32 tr_vowel_is_back(u32 cp);      /* a,ı,o,u -> TRUE ; e,i,ö,ü -> FALSE (yalniz unluler icin gecerlidir) */
bool32 tr_vowel_is_rounded(u32 cp);   /* o,ö,u,ü -> TRUE ; a,e,ı,i -> FALSE */

/* FISTIKÇI ŞAHAP: f,s,t,k,ç,ş,h,p */
bool32 tr_is_voiceless_consonant(u32 cp);

/* p,ç,t,k -> TRUE (yumusamaya aday unsuzler) */
bool32 tr_is_softenable(u32 cp);
/* p->b, ç->c, t->d, k->ğ; digerlerinde degismeden dondurur. */
u32 tr_soften(u32 cp);
/* TERSI: b->p, c->ç, d->t, ğ->k; digerlerinde 0 dondurur (uygulanamaz). */
u32 tr_unsoften(u32 cp);

typedef struct WordEnding {
    bool32 ends_in_vowel;
    u32 last_cp;             /* kelimenin son kod noktasi */
    bool32 has_vowel;        /* kelimede hic unlu var mi (uyum icin gerekli) */
    bool32 last_vowel_back;  /* son unlunun art/on ozelligi */
    bool32 last_vowel_rounded;
} WordEnding;

/* word (UTF-8, len bayt) icin ek cozumlemede kullanilacak fonolojik
 * bilgiyi cikarir: son harf unlu mu, ve (kelimenin sonunda unsuz olsa
 * bile) EN SON GECEN unlunun art/yuvarlak ozellikleri. */
WordEnding tr_analyze_ending(const char* word, u64 len);

/* --- Ek sablonu cozucu ---
 * Sablon sozdizimi: 'A' = 2 yollu unlu (a/e), 'I' = 4 yollu unlu
 * (ı/i/u/ü), 'D' = d/t donusumlu unsuz, 'C' = c/ç donusumlu unsuz,
 * digerleri (kucuk harf) oldugu gibi kopyalanir.
 *
 * Turkce'de unlu-baslangicli eklerin kok unluyle bittiginde iki farkli
 * davranisi vardir:
 *   (1) KAYNASTIRMA UNSUZU eklenir (orn. akuzatif -(y)I, datif -(y)A,
 *       3. tekil iyelik -(s)I, genitif -(n)In) -> buffer_consonant kullanilir.
 *   (2) EKIN BASINDAKI UNLU DUSER (orn. 1./2. sahis iyelik -(I)m/-(I)n,
 *       simdiki zaman -(I)yor) -> drop_initial_vowel_after_vowel=TRUE.
 * Bu ikisi karsilikli disiktir (bir ek ayni anda ikisini de yapmaz).
 *
 * out_buf en az 16 bayt olmalidir. Donen deger: yazilan bayt sayisi.
 * *out_softened_root_cp, kok sonu yumusamaya ugradiysa (out_buf'un
 * ONUNE degil, cagiranin kok stringinin SON KOD NOKTASINI degistirmesi
 * icin) yeni kod noktayi yazar; ugramadiysa 0 yazar. */
u64 tr_resolve_suffix(const char* template_str, WordEnding ending,
                       char buffer_consonant, bool32 triggers_softening,
                       bool32 drop_initial_vowel_after_vowel,
                       char* out_buf, u32* out_softened_root_cp);

/* root'un son kod noktasini new_last_cp ile degistirip out_buf'a yazar
 * (yumusama uygulandiktan sonra kok+ek birlestirmek icin). out_buf en
 * az root_len+3 bayt olmalidir. Yazilan yeni uzunlugu dondurur. */
u64 tr_apply_root_change(const char* root, u64 root_len, u32 new_last_cp, char* out_buf);

#endif /* TOKENIZER_TURKISH_PHON_H */
