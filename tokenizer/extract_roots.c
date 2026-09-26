/* Katman 6 - Kok sozlugu cikarma araci.
 *
 * Turkce Vikisozluk XML dokumunu (data/raw/trwiktionary-latest-pages-articles.xml)
 * tarar; ana isim alaninda (ns=0), "Turkce" dil basligi iceren ve tek
 * kelimeden olusan (bosluk icermeyen) sayfa basliklarini kok adayi olarak
 * cikarir. Bulunabilirse ilk soz turu (===Isim===, ===Fiil=== vb.) basligini
 * da yakalar.
 *
 * Hicbir XML/regex kutuphanesi kullanilmaz -- tum ayristirma kendi
 * str_find/str_eq fonksiyonlarimizla (strutil.h) yapilir. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/file_io.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"

static bool32 contains_byte(const char* s, u64 len, u8 byte) {
    for (u64 i = 0; i < len; i++) if ((u8)s[i] == byte) return TRUE;
    return FALSE;
}

/* [start, end) araligindaki metni cikti dosyasina yazar. */
static void write_slice(FileHandle* out, const char* start, u64 len) {
    file_write(out, start, len);
}

int main(void) {
    console_write_line("=== Kok Sozlugu Cikarma (Turkce Vikisozluk) ===");

    Allocator alloc = allocator_create(2200ull * 1024 * 1024); /* ~2.2 GB: dosya + calisma alani */

    const char* in_path = "data/raw/trwiktionary-latest-pages-articles.xml";
    console_write_line("Dosya okunuyor (bu biraz surebilir)...");
    u64 size = 0;
    char* buf = (char*)file_read_entire(in_path, &alloc, &size);
    if (!buf) {
        console_write_line("HATA: girdi dosyasi okunamadi.");
        return 1;
    }
    console_write("Okunan bayt: "); console_write_u64(size); console_write_line("");

    FileHandle out = file_open_write("data/raw/kok_adaylari.txt");
    if (!out.valid) {
        console_write_line("HATA: cikti dosyasi acilamadi.");
        return 1;
    }

    u64 pos = 0;
    u64 page_count = 0;
    u64 root_count = 0;

    const char* TAG_PAGE = "<page>";
    const char* TAG_PAGE_END = "</page>";
    const char* TAG_NS_OPEN = "<ns>";
    const char* TAG_TITLE_OPEN = "<title>";
    const char* TAG_TITLE_CLOSE = "</title>";
    const char* TAG_TEXT_OPEN = "<text";
    const char* TAG_TEXT_CLOSE = "</text>";
    const char* HDR_TR_A = "==Türkçe==";
    const char* HDR_TR_B = "== Türkçe ==";
    const char* HDR_TR_C = "==Türkçe ==";
    const char* HDR_TR_D = "== Türkçe==";

    while (pos < size) {
        const char* p_page = str_find(buf + pos, size - pos, TAG_PAGE);
        if (!p_page) break;
        u64 page_start = (u64)(p_page - buf);

        const char* p_page_end = str_find(buf + page_start, size - page_start, TAG_PAGE_END);
        if (!p_page_end) break;
        u64 page_end = (u64)(p_page_end - buf) + str_len(TAG_PAGE_END);

        u64 seg_len = page_end - page_start;
        const char* seg = buf + page_start;

        page_count++;
        if ((page_count % 200000) == 0) {
            console_write("  islenen sayfa: "); console_write_u64(page_count);
            console_write(" | bulunan kok: "); console_write_u64(root_count);
            console_write_line("");
        }

        /* --- ns kontrolu: sadece ana isim alani (ns=0) --- */
        const char* p_ns = str_find(seg, seg_len, TAG_NS_OPEN);
        bool32 is_main_ns = FALSE;
        if (p_ns) {
            const char* nsval = p_ns + str_len(TAG_NS_OPEN);
            is_main_ns = (nsval[0] == '0' && nsval[1] == '<');
        }

        if (is_main_ns) {
            /* --- baslik --- */
            const char* p_title = str_find(seg, seg_len, TAG_TITLE_OPEN);
            const char* p_title_close = p_title ? str_find(p_title, seg_len - (u64)(p_title - seg), TAG_TITLE_CLOSE) : NULL_PTR;

            /* --- metin --- */
            const char* p_text = str_find(seg, seg_len, TAG_TEXT_OPEN);
            const char* p_text_gt = p_text ? str_find(p_text, seg_len - (u64)(p_text - seg), ">") : NULL_PTR;
            const char* text_start = p_text_gt ? p_text_gt + 1 : NULL_PTR;
            const char* p_text_close = text_start ? str_find(text_start, seg_len - (u64)(text_start - seg), TAG_TEXT_CLOSE) : NULL_PTR;

            if (p_title && p_title_close && text_start && p_text_close) {
                const char* title = p_title + str_len(TAG_TITLE_OPEN);
                u64 title_len = (u64)(p_title_close - title);
                u64 text_len = (u64)(p_text_close - text_start);

                /* Turkce dil basligi ariyoruz (bicim kucuk farklarla degisebilir). */
                const char* tr_hit = str_find(text_start, text_len, HDR_TR_A);
                if (!tr_hit) tr_hit = str_find(text_start, text_len, HDR_TR_B);
                if (!tr_hit) tr_hit = str_find(text_start, text_len, HDR_TR_C);
                if (!tr_hit) tr_hit = str_find(text_start, text_len, HDR_TR_D);

                if (tr_hit && title_len > 0 && title_len < 64 && !contains_byte(title, title_len, ' ')
                    && !contains_byte(title, title_len, ':')) {
                    write_slice(&out, title, title_len);
                    file_write(&out, "\n", 1);
                    root_count++;
                }
            }
        }

        pos = page_end;
    }

    file_close(&out);

    console_write("Toplam sayfa: "); console_write_u64(page_count); console_write_line("");
    console_write("Bulunan kok adayi: "); console_write_u64(root_count); console_write_line("");

    allocator_destroy(&alloc);
    return 0;
}
