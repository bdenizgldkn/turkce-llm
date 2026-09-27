/* Faz 4 (veri hazirligi) - Film altyazisi satiri -> duz metin temizleyici.
 *
 * OpenSubtitles (OPUS) Turkce tek-dilli dokumu satir basina bir altyazi
 * cumlesi icerir. Tek gecisli, elle yazilmis temizleme (kutuphane yok):
 *   - <i>...</i> gibi HTML etiketleri ve {\an8} gibi stil kodlari -> etiket silinir, ici kalir
 *   - [GÜLER] gibi ses betimlemeleri ({...] karisik kapanis dahil) ve {12345 kare numaralari -> ATILIR
 *   - cevirmen/site imzasi iceren satirlar ("Çeviren: ...", "www.") -> ATILIR (0 dondurur)
 *   - satir basindaki diyalog tireleri ("-Evet." -> "Evet.")
 *   - OCR hatasi: kucuk harften sonra gelen buyuk 'I' -> 'l'
 *     ("oIdu" -> "oldu"). Duzeltilmis onceki harfe bakildigi icin ardisik
 *     I'lar da duzelir ("haIIederiz" -> "hallederiz").
 *     Kelime basindaki 'I' dokunulmaz (Işık, Irmak gecerli Turkce).
 *   - ardisik bosluklar tek bosluga, bas/son bosluklar atilir
 *   - sarki satirlari (♪ / ♫ iceren) ve hic harf icermeyen satirlar -> ATILIR (0 dondurur)
 */
#ifndef TOKENIZER_SUBTITLE_CLEAN_H
#define TOKENIZER_SUBTITLE_CLEAN_H

#include "../runtime/types.h"

/* in: tek satir (satir sonu karakteri HARIC). out en az in_len bayt
 * olmalidir (temizleme sadece kisaltir). Yazilan uzunlugu dondurur;
 * satir atilmasi gerekiyorsa 0. */
u64 subtitle_clean_line(const char* in, u64 in_len, char* out);

#endif /* TOKENIZER_SUBTITLE_CLEAN_H */
