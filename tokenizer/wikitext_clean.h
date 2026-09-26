/* Katman 7/8 (veri hazirligi) - Vikipedi wikitext -> duz metin temizleyici.
 *
 * Hicbir wikitext/regex kutuphanesi kullanilmaz -- tek gecisli, elle
 * yazilmis bir durum makinesi. Kapsam (V1, "iyi yeterli" -- mukemmel
 * degil):
 *   - {{sablon...}}  -> TAMAMEN ATILIR (ic ice olabilir, derinlik takip edilir)
 *   - {|tablo...|}   -> TAMAMEN ATILIR
 *   - <ref>...</ref>, <ref .../>  -> ATILIR
 *   - <!-- yorum -->  -> ATILIR
 *   - <math>...</math>  -> ATILIR
 *   - diger <etiket> HTML benzeri isaretler -> sadece etiket silinir, ici kalir
 *   - [[Kategori:...]], [[Dosya:...]], [[Resim:...]]  -> ATILIR
 *   - [[Baglanti|Gorunen metin]] -> "Gorunen metin" kalir
 *   - [[Baglanti]] -> "Baglanti" kalir
 *   - [http://... gorunen metin] -> "gorunen metin" kalir (yoksa atilir)
 *   - '''kalin''', ''italik''  -> isaretleme silinir, metin kalir
 *   - == Baslik ==  -> "Baslik" kalir
 */
#ifndef TOKENIZER_WIKITEXT_CLEAN_H
#define TOKENIZER_WIKITEXT_CLEAN_H

#include "../runtime/types.h"

/* out en az in_len bayt olmalidir (temizleme sadece kisaltir).
 * Yazilan (temizlenmis) uzunlugu dondurur. */
u64 wikitext_clean(const char* in, u64 in_len, char* out);

/* MediaWiki dokumu, wikitext'i bir XML <text> elemani icine koydugu
 * icin, iceride literal olarak gecen '<','>','&' gibi karakterler
 * &lt;/&gt;/&amp;/&quot;/&apos;/&#NNN; olarak XML-kacisli gelir. Bu
 * yuzden wikitext_clean'den ONCE bu kacis dizilerini cozmemiz gerekir
 * (yoksa "<ref>" gibi etiketler "&lt;ref&gt;" olarak gorunur ve
 * wikitext_clean bunlari yakalayamaz). out en az in_len bayt olmalidir. */
u64 xml_unescape(const char* in, u64 in_len, char* out);

#endif /* TOKENIZER_WIKITEXT_CLEAN_H */
