/* Katman 0 (uzantisi) - Minimal UTF-8 kodlama/cozme.
 * Turkce'ye ozgu karakterler (ş,ğ,ç,ö,ü,ı,İ) UTF-8'de 2 baytlidir;
 * morfoloji ve tokenizer katmanlarinin dogru calismasi icin bunlari
 * tek bir "kod noktasi" (codepoint) olarak okuyup yazabilmemiz gerekir.
 * Hicbir <uchar.h>/ICU/vb. kutuphane kullanilmaz. */
#ifndef RUNTIME_UTF8_H
#define RUNTIME_UTF8_H

#include "types.h"

/* s'den bir UTF-8 kod noktasi cozer, *out_cp'ye yazar, tukettigi bayt
 * sayisini (1-4) dondurur. Gecersiz/bozuk baytlarda 1 dondurup
 * *out_cp'yi o ham baytin degeriyle doldurur (guvenli geri dusum). */
u32 utf8_decode(const char* s, u32* out_cp);

/* cp kod noktasini out'a UTF-8 olarak yazar, yazilan bayt sayisini dondurur
 * (out en az 4 bayt olmalidir). */
u32 utf8_encode(u32 cp, char* out);

/* s icindeki (bayt cinsinden len uzunlugundaki) metnin kac kod noktasi
 * (karakter) icerdigini sayar. */
u64 utf8_count_codepoints(const char* s, u64 len);

/* s (len bayt) icindeki SON kod noktasinin kac bayt tuttugunu dondurur
 * (geriye dogru devam baytlarini -- 10xxxxxx -- atlayarak bulur). */
u64 utf8_last_cp_len(const char* s, u64 len);

#endif /* RUNTIME_UTF8_H */
