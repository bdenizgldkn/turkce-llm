/* Faz 4 veri hazirligi dogrulama: altyazi satiri temizleyici
 * (subtitle_clean_line) gercek OpenSubtitles orneklerinden alinmis
 * satirlar uzerinde test edilir. */
#include "../runtime/types.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"
#include "../tokenizer/subtitle_clean.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static void check_clean(const char* input, const char* expected, const char* label) {
    u64 len = str_len(input);
    char cleaned[1024];
    u64 clen = subtitle_clean_line(input, len, cleaned);
    cleaned[clen] = 0;

    if (str_eq(cleaned, expected) && clen <= len) { g_pass++; }
    else {
        g_fail++;
        console_write("  [FAIL] "); console_write_line(label);
        console_write("    beklenen: '"); console_write(expected); console_write_line("'");
        console_write("    bulunan:  '"); console_write(cleaned); console_write_line("'");
    }
}

int main(void) {
    console_write_line("=== Altyazi Temizleyici Testleri ===");

    check_clean("Ben hırsızım.", "Ben hırsızım.", "temiz satir aynen kalmali");

    /* OCR I -> l */
    check_clean("Ne oIdu?", "Ne oldu?", "OCR: kelime ici I -> l");
    check_clean("KiIitIendin mi?", "Kilitlendin mi?", "OCR: birden fazla I");
    check_clean("Bunu haIIederiz.", "Bunu hallederiz.", "OCR: ardisik I'lar");
    check_clean("Beş yıI önceydi!", "Beş yıl önceydi!", "OCR: ı'dan sonra I");
    check_clean("Işık ve Irmak geldi.", "Işık ve Irmak geldi.", "kelime basi I dokunulmamali");
    check_clean("ISTANBUL", "ISTANBUL", "tamami buyuk harf kelime dokunulmamali");
    check_clean("Ali Irmak'a gitti.", "Ali Irmak'a gitti.", "bosluktan sonra I dokunulmamali");

    /* diyalog tireleri */
    check_clean("-Evet ama...", "Evet ama...", "bastaki tire atilmali");
    check_clean("- Evet.", "Evet.", "tire + bosluk atilmali");
    check_clean("<i>-Kimsin?</i>", "Kimsin?", "etiketten sonraki tire atilmali");
    check_clean("-Evet. -Hayır.", "Evet. -Hayır.", "ortadaki tire kalmali");

    /* etiketler ve stil kodlari */
    check_clean("<i>Sende.</i>", "Sende.", "italik etiketi");
    check_clean("{\\an8}Son dakika:", "Son dakika:", "stil kodu");
    check_clean("<font color=\"#ffff00\">Tamam.</font>", "Tamam.", "font etiketi");
    check_clean("3 < 5 değil mi?", "3 < 5 değil mi?", "kapanmayan < literal kalmali");

    check_clean("[GÜLÜŞÜRLER] Ne oldu?", "Ne oldu?", "ses betimlemesi atilmali");
    check_clean("{MARTI SESLERİ]", "", "karisik kapanisli betimleme -> bos satir");
    check_clean("{35081Dün buldum onları.", "Dün buldum onları.", "MicroDVD kare numarasi");
    check_clean("Evet (gülerek) tamam.", "Evet (gülerek) tamam.", "parantez icerigi kalmali");

    /* imzalar */
    check_clean("Çeviren: sarkis İyi seyirler dileriz.", "", "cevirmen imzasi atilmali");
    check_clean("Altyazı: www.ornek.com", "", "site imzasi atilmali");
    check_clean("Çevir direksiyonu!", "Çevir direksiyonu!", "cevir fiili imza sanilmamali");

    /* bosluklar */
    check_clean("  Ne   dedin?  ", "Ne dedin?", "bosluklar sadelesmeli");

    /* atilacak satirlar */
    check_clean("♪ Seni seviyorum ♪", "", "sarki satiri atilmali");
    check_clean("...", "", "harfsiz satir atilmali");
    check_clean("- 1995 -", "", "sadece sayi atilmali");
    check_clean("", "", "bos satir");

    /* bozuk UTF-8: cikti girdiden uzun olmamali */
    check_clean("ab\xC3", "ab\xC3", "satir sonunda yarim UTF-8");

    console_write("\nSonuc: "); console_write_u64(g_pass); console_write(" basarili, ");
    console_write_u64(g_fail); console_write_line(" basarisiz.");
    return g_fail == 0 ? 0 : 1;
}
