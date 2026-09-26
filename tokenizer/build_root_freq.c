/* Katman 6 - Kelime frekans tablosu cikarma araci.
 *
 * Ayni Vikisozluk XML dokumunu tekrar tarar; bu sefer sayfa basliklari
 * degil, "Turkce" bolumlerinin DUZ METNI (tanimlar, ornek cumleler)
 * icindeki kelimelerin frekansini sayar. Bu frekans tablosu, morfolojik
 * belirsizlik giderme (disambiguation) icin "hangi kok daha yaygin"
 * sinyalini saglar (bkz. tokenizer/disambiguate.c).
 *
 * Tokenizasyon: Turkce harf dizileri (tr_is_letter) kelime, digeri
 * ayirici kabul edilir -- wiki isaretlemesi ([[ ]], {{ }} vb.) bu
 * sekilde dogal olarak parcalara ayrilir (bir miktar gurultu kabul
 * edilebilir, gercek Turkce kelimeler frekansca baskindir). */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/file_io.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"
#include "../runtime/strcounter.h"
#include "../runtime/utf8.h"
#include "turkish_phon.h"

static void count_words_in_text(StrCounter* counter, const char* text, u64 text_len) {
    u64 i = 0;
    while (i < text_len) {
        u32 cp;
        u64 n = utf8_decode(text + i, &cp);

        if (tr_is_letter(cp)) {
            u64 word_start = i;
            char lower_buf[256];
            u64 lower_len = 0;

            while (i < text_len) {
                u32 c2;
                u64 n2 = utf8_decode(text + i, &c2);
                if (!tr_is_letter(c2)) break;
                if (lower_len + 4 < 256) {
                    lower_len += utf8_encode(tr_to_lower_cp(c2), lower_buf + lower_len);
                }
                i += n2;
            }
            (void)word_start;

            if (lower_len > 0) {
                strcounter_increment(counter, lower_buf, lower_len);
            }
        } else {
            i += n;
        }
    }
}

int main(void) {
    console_write_line("=== Kelime Frekans Tablosu Cikarma ===");

    Allocator alloc = allocator_create(2500ull * 1024 * 1024);

    const char* in_path = "data/raw/trwiktionary-latest-pages-articles.xml";
    console_write_line("Dosya okunuyor...");
    u64 size = 0;
    char* buf = (char*)file_read_entire(in_path, &alloc, &size);
    if (!buf) { console_write_line("HATA: girdi dosyasi okunamadi."); return 1; }

    StrCounter counter = strcounter_create(&alloc, 1500000);

    u64 pos = 0;
    u64 page_count = 0;

    const char* TAG_PAGE = "<page>";
    const char* TAG_PAGE_END = "</page>";
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
        if ((page_count % 300000) == 0) {
            console_write("  islenen sayfa: "); console_write_u64(page_count);
            console_write(" | benzersiz kelime: "); console_write_u64(counter.num_keys);
            console_write_line("");
        }

        const char* p_text = str_find(seg, seg_len, TAG_TEXT_OPEN);
        const char* p_text_gt = p_text ? str_find(p_text, seg_len - (u64)(p_text - seg), ">") : NULL_PTR;
        const char* text_start = p_text_gt ? p_text_gt + 1 : NULL_PTR;
        const char* p_text_close = text_start ? str_find(text_start, seg_len - (u64)(text_start - seg), TAG_TEXT_CLOSE) : NULL_PTR;

        if (text_start && p_text_close) {
            u64 text_len = (u64)(p_text_close - text_start);

            const char* tr_hit = str_find(text_start, text_len, HDR_TR_A);
            if (!tr_hit) tr_hit = str_find(text_start, text_len, HDR_TR_B);
            if (!tr_hit) tr_hit = str_find(text_start, text_len, HDR_TR_C);
            if (!tr_hit) tr_hit = str_find(text_start, text_len, HDR_TR_D);

            if (tr_hit) {
                count_words_in_text(&counter, text_start, text_len);
            }
        }

        pos = page_end;
    }

    console_write("Toplam sayfa: "); console_write_u64(page_count); console_write_line("");
    console_write("Benzersiz kelime: "); console_write_u64(counter.num_keys); console_write_line("");

    FileHandle out = file_open_write("data/raw/kelime_frekans.txt");
    if (!out.valid) { console_write_line("HATA: cikti dosyasi acilamadi."); return 1; }

    char num_buf[24];
    for (u64 slot = 0; slot < counter.capacity; slot++) {
        if (counter.keys[slot] == NULL_PTR) continue;

        file_write(&out, counter.keys[slot], counter.key_lens[slot]);
        file_write(&out, "\t", 1);

        u64 v = counter.counts[slot];
        i32 ndig = 0;
        if (v == 0) { num_buf[0] = '0'; ndig = 1; }
        else { char tmp[24]; i32 t = 0; while (v > 0) { tmp[t++] = (char)('0' + v % 10); v /= 10; } while (t > 0) num_buf[ndig++] = tmp[--t]; }
        file_write(&out, num_buf, (u64)ndig);
        file_write(&out, "\n", 1);
    }
    file_close(&out);

    console_write_line("Yazildi: data/raw/kelime_frekans.txt");

    allocator_destroy(&alloc);
    return 0;
}
