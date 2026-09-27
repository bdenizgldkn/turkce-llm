/* Veri hazirligi dogrulama: wikitext temizleyici (xml_unescape +
 * wikitext_clean) elle hazirlanmis, gercekci ornek wikitext parcalari
 * uzerinde test edilir. */
#include "../runtime/types.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"
#include "../tokenizer/wikitext_clean.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static void check_clean(const char* input, const char* expected, const char* label) {
    u64 len = str_len(input);
    char unescaped[2048];
    u64 ulen = xml_unescape(input, len, unescaped);
    char cleaned[2048];
    u64 clen = wikitext_clean(unescaped, ulen, cleaned);
    cleaned[clen] = 0;

    if (str_eq(cleaned, expected)) { g_pass++; }
    else {
        g_fail++;
        console_write("  [FAIL] "); console_write_line(label);
        console_write("    beklenen: '"); console_write(expected); console_write_line("'");
        console_write("    bulunan:  '"); console_write(cleaned); console_write_line("'");
    }
}

int main(void) {
    console_write_line("=== Wikitext Temizleyici Testleri ===");

    check_clean("&lt;ref&gt;kaynak&lt;/ref&gt;Merhaba dunya.",
                "Merhaba dunya.", "ref etiketi atilmali");

    check_clean("{{ambox|metin=onemsiz}}Gercek metin burada.",
                "Gercek metin burada.", "sablon atilmali");

    check_clean("{{dis|{{ic}}}}Metin.",
                "Metin.", "ic ice sablon atilmali");

    check_clean("[[İstanbul|İstanbul'a]] gittim.",
                "İstanbul'a gittim.", "wiki baglantisi -> goruntu metni");

    check_clean("[[Ankara]] baskenttir.",
                "Ankara baskenttir.", "wiki baglantisi (pipesiz) -> hedef metin");

    check_clean("[[Kategori:Turkiye]]Metin kalmali.",
                "Metin kalmali.", "kategori baglantisi atilmali");

    check_clean("[[Dosya:resim.jpg|thumb|Aciklama]]Metin.",
                "Metin.", "dosya baglantisi atilmali");

    check_clean("[http://example.com Bir baglanti] burada.",
                "Bir baglanti burada.", "dis baglanti -> goruntu metni");

    check_clean("'''Kalin''' ve ''italik'' metin.",
                "Kalin ve italik metin.", "kalin/italik isaretleme silinmeli");

    check_clean("== Baslik ==\nParagraf metni.",
                " Baslik \nParagraf metni.", "baslik isaretleri silinmeli (etraf bosluklari korunur)");

    check_clean("&lt;!-- yorum satiri --&gt;Gorunen metin.",
                "Gorunen metin.", "yorum atilmali");

    check_clean("&quot;Alinti&quot; ve &amp; isareti.",
                "\"Alinti\" ve & isareti.", "xml entity cozme (quot/amp)");

    check_clean("Kelime1&nbsp;Kelime2 normal.",
                "Kelime1 Kelime2 normal.", "&nbsp; adi varligi normal bosluga cevrilmeli");

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
