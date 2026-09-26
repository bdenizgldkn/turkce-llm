/* Veri hazirligi - Turkce Vikipedi XML dokumunden temiz duz metin
 * korpüsü cikarir. Her makale (ns=0, yonlendirme olmayan) icin:
 * XML kacislarini coz -> wikitext isaretlemesini temizle -> yaz.
 *
 * AKIS (STREAMING) TABANLI: RAM kisitli oldugu icin (bkz. PROJE_PLANI.md,
 * "buyuk korpüsler islenirken parcalara bolunur") dosyanin tamami
 * belleğe yuklenmez; sabit boyutlu bir "pencere" halinde okunur,
 * pencere sinirinda kalan yarim sayfa bir sonraki okumaya tasinir.
 *
 * Cikti: data/raw/wikipedia_corpus.txt (makaleler bos satirla ayrilir). */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/file_io.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"
#include "wikitext_clean.h"

#define WINDOW_SIZE (256ull * 1024 * 1024) /* okuma penceresi */
#define MAX_CARRY   (16ull * 1024 * 1024)  /* pencere sinirinda tasinabilecek en buyuk yarim sayfa */
#define SCRATCH_CAP (8ull * 1024 * 1024)   /* tek bir makale icin temizleme tamponu */

int main(void) {
    console_write_line("=== Vikipedi Korpüsü Olusturma (akis/streaming) ===");

    Allocator alloc = allocator_create(512ull * 1024 * 1024); /* pencere + tamponlar icin yeterli */

    FileHandle in = file_open_read("data/raw/trwiki-latest-pages-articles.xml");
    if (!in.valid) { console_write_line("HATA: girdi dosyasi acilamadi."); return 1; }

    FileHandle out = file_open_write("data/raw/wikipedia_corpus.txt");
    if (!out.valid) { console_write_line("HATA: cikti dosyasi acilamadi."); return 1; }

    char* window = (char*)allocator_alloc(&alloc, WINDOW_SIZE + MAX_CARRY);
    char* unescaped = (char*)allocator_alloc(&alloc, SCRATCH_CAP);
    char* cleaned = (char*)allocator_alloc(&alloc, SCRATCH_CAP);

    const char* TAG_PAGE = "<page>";
    const char* TAG_PAGE_END = "</page>";
    const char* TAG_NS_OPEN = "<ns>";
    const char* TAG_REDIRECT = "<redirect";
    const char* TAG_TEXT_OPEN = "<text";
    const char* TAG_TEXT_CLOSE = "</text>";

    u64 carry_len = 0;
    u64 page_count = 0;
    u64 article_count = 0;
    u64 total_clean_bytes = 0;
    u64 total_read_bytes = 0;

    for (;;) {
        u64 got = file_read(&in, window + carry_len, WINDOW_SIZE);
        u64 total_len = carry_len + got;
        total_read_bytes += got;

        if (total_len == 0) break;

        u64 process_pos = 0;
        u64 last_complete_end = 0;

        while (process_pos < total_len) {
            const char* p_page = str_find(window + process_pos, total_len - process_pos, TAG_PAGE);
            if (!p_page) break;
            u64 page_start = (u64)(p_page - window);

            const char* p_page_end = str_find(window + page_start, total_len - page_start, TAG_PAGE_END);
            if (!p_page_end) break; /* pencere sinirinda yarim kalmis sayfa -- bir sonraki okumaya tasi */
            u64 page_end = (u64)(p_page_end - window) + str_len(TAG_PAGE_END);

            u64 seg_len = page_end - page_start;
            const char* seg = window + page_start;

            page_count++;
            if ((page_count % 100000) == 0) {
                console_write("  islenen sayfa: "); console_write_u64(page_count);
                console_write(" | yazilan makale: "); console_write_u64(article_count);
                console_write(" | temiz MB: "); console_write_u64(total_clean_bytes / (1024 * 1024));
                console_write(" | okunan GB: "); console_write_u64(total_read_bytes / (1024 * 1024 * 1024));
                console_write_line("");
            }

            const char* p_ns = str_find(seg, seg_len, TAG_NS_OPEN);
            bool32 is_main_ns = FALSE;
            if (p_ns) {
                const char* nsval = p_ns + str_len(TAG_NS_OPEN);
                is_main_ns = (nsval[0] == '0' && nsval[1] == '<');
            }

            bool32 is_redirect = (str_find(seg, seg_len, TAG_REDIRECT) != NULL_PTR);

            if (is_main_ns && !is_redirect) {
                const char* p_text = str_find(seg, seg_len, TAG_TEXT_OPEN);
                const char* p_text_gt = p_text ? str_find(p_text, seg_len - (u64)(p_text - seg), ">") : NULL_PTR;
                const char* text_start = p_text_gt ? p_text_gt + 1 : NULL_PTR;
                const char* p_text_close = text_start ? str_find(text_start, seg_len - (u64)(text_start - seg), TAG_TEXT_CLOSE) : NULL_PTR;

                if (text_start && p_text_close) {
                    u64 text_len = (u64)(p_text_close - text_start);

                    if (text_len > 0 && text_len < SCRATCH_CAP) {
                        u64 ulen = xml_unescape(text_start, text_len, unescaped);
                        u64 clen = wikitext_clean(unescaped, ulen, cleaned);

                        if (clen > 200) {
                            file_write(&out, cleaned, clen);
                            file_write(&out, "\n\n", 2);
                            article_count++;
                            total_clean_bytes += clen;
                        }
                    }
                }
            }

            process_pos = page_end;
            last_complete_end = page_end;
        }

        /* Islenmemis kalan (yarim sayfa olabilecek) veriyi pencerenin
         * basina tasi; bir sonraki okuma bunun ardina eklenecek. */
        carry_len = total_len - last_complete_end;
        if (carry_len > MAX_CARRY) carry_len = MAX_CARRY; /* guvenlik: asiri buyume olmamali */
        for (u64 i = 0; i < carry_len; i++) window[i] = window[last_complete_end + i];

        if (got == 0) break; /* dosya sonu, artik yeni veri gelmeyecek */
    }

    file_close(&in);
    file_close(&out);

    console_write("Toplam okunan bayt: "); console_write_u64(total_read_bytes); console_write_line("");
    console_write("Toplam sayfa: "); console_write_u64(page_count); console_write_line("");
    console_write("Yazilan makale: "); console_write_u64(article_count); console_write_line("");
    console_write("Toplam temiz bayt: "); console_write_u64(total_clean_bytes); console_write_line("");

    allocator_destroy(&alloc);
    return 0;
}
