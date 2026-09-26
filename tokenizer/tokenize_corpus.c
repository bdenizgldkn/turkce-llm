/* Veri hazirligi - gercek Turkce Vikipedi korpüsünü (wikipedia_corpus.txt)
 * tam tokenizer boru hattimizdan (morfolojik ayristirma + istatistiksel
 * belirsizlik giderme + BPE fallback) gecirip token ID dizisine cevirir.
 *
 * AKIS (STREAMING) TABANLI: buyuk dosyayi pencere pencere okur, kelime
 * sinirinda (bosluk karakterinde) guvenli kesim yapar.
 *
 * Cikti: data/raw/wikipedia_tokens.bin (ham u32 token ID dizisi, kucuk-endian). */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/file_io.h"
#include "../runtime/console.h"
#include "../runtime/utf8.h"
#include "lexicon.h"
#include "freqtable.h"
#include "tokenize.h"
#include "turkish_phon.h"

#define WINDOW_SIZE (128ull * 1024 * 1024)
#define MAX_CARRY   (8ull * 1024 * 1024)
#define MAX_SEGMENT 100 /* tokenize_word'e tek seferde verilecek en fazla bayt (TOKENIZE_MAX_IDS guvenligi icin) */

static bool32 is_ws(u8 b) {
    return b == ' ' || b == '\t' || b == '\n' || b == '\r';
}

int main(void) {
    console_write_line("=== Vikipedi Korpüsünü Tokenize Etme ===");

    Allocator alloc = allocator_create(3000ull * 1024 * 1024);

    console_write_line("Kok sozlugu yukleniyor...");
    u64 word_count = 0;
    StrHashSet lex = lexicon_load(&alloc, "data/raw/kok_adaylari.txt", &word_count);

    console_write_line("Kelime frekans tablosu yukleniyor...");
    StrCounter freq = freqtable_load(&alloc, "data/raw/kelime_frekans.txt");

    console_write_line("BPE vocab yukleniyor...");
    TokenizerVocab voc = tokenizer_init(&alloc, "data/raw/bpe_merges.txt");
    console_write("Toplam vocab boyutu: "); console_write_u64(voc.vocab_size); console_write_line("");

    FileHandle in = file_open_read("data/raw/wikipedia_corpus.txt");
    if (!in.valid) { console_write_line("HATA: korpüs dosyasi acilamadi."); return 1; }
    FileHandle out = file_open_write("data/raw/wikipedia_tokens.bin");
    if (!out.valid) { console_write_line("HATA: cikti dosyasi acilamadi."); return 1; }

    char* window = (char*)allocator_alloc(&alloc, WINDOW_SIZE + MAX_CARRY);

    u64 carry_len = 0;
    u64 total_tokens = 0;
    u64 total_words_analyzed = 0;
    u64 total_bytes_in = 0;
    u32 out_buf[4096]; /* toplu yazma icin tampon */
    u64 out_buf_n = 0;

    for (;;) {
        u64 got = file_read(&in, window + carry_len, WINDOW_SIZE);
        u64 total_len = carry_len + got;
        total_bytes_in += got;
        if (total_len == 0) break;

        u64 process_end = total_len;
        if (got > 0) {
            i64 cut = (i64)total_len - 1;
            while (cut > 0 && !is_ws((u8)window[cut])) cut--;
            process_end = (u64)cut + 1;
            if (process_end == 0) process_end = total_len; /* guvenlik: cok uzun tek "kelime" */
        }

        u64 i = 0;
        while (i < process_end) {
            u32 cp;
            u64 n = utf8_decode(window + i, &cp);
            bool32 is_letter = tr_is_letter(cp);
            u64 start = i;
            i += n;
            while (i < process_end) {
                u32 cp2;
                u64 n2 = utf8_decode(window + i, &cp2);
                if (tr_is_letter(cp2) != is_letter) break;
                i += n2;
            }

            u64 seg_start = start;
            while (seg_start < i) {
                u64 seg_len = i - seg_start;
                if (seg_len > MAX_SEGMENT) seg_len = MAX_SEGMENT;
                /* UTF-8 sinirinda kesmemek icin geri git -- ama SADECE
                 * window[seg_start+seg_len] gecerli/okunmus veri sinirlari
                 * icindeyse (total_len'den once) kontrol et. Bu sinirin
                 * TAM UZERINDE (seg_start+seg_len==total_len) okuma
                 * yapmak, pencere tamponunun (nadiren) tam dolu oldugu
                 * durumlarda commit edilmemis belleğe tasip SEGFAULT'A
                 * yol aciyordu -- gercek buyuk-olcekli kosuda yakalanan
                 * bir hataydi. */
                while (seg_len > 1 && seg_start + seg_len < total_len &&
                       ((u8)window[seg_start + seg_len] & 0xC0) == 0x80) seg_len--;

                u32 ids[TOKENIZE_MAX_IDS];
                u32 n_ids = tokenize_word(&voc, &lex, &freq, window + seg_start, seg_len, ids);
                total_words_analyzed++;

                for (u32 k = 0; k < n_ids; k++) {
                    out_buf[out_buf_n++] = ids[k];
                    if (out_buf_n == 4096) {
                        file_write(&out, out_buf, out_buf_n * sizeof(u32));
                        out_buf_n = 0;
                    }
                }
                total_tokens += n_ids;
                seg_start += seg_len;
            }
        }

        console_write("  islenen MB: "); console_write_u64(total_bytes_in / (1024 * 1024));
        console_write(" | uretilen token: "); console_write_u64(total_tokens);
        console_write(" | analiz edilen parca: "); console_write_u64(total_words_analyzed);
        console_write_line("");

        carry_len = total_len - process_end;
        if (carry_len > MAX_CARRY) {
            /* Guvenlik: MAX_CARRY'den buyuk bir "tasima" (bos.uk/harf
             * sinirinda cok uzun bir bosluksuz blok anlamina gelir --
             * orn. buyuk bir veri tablosu/dizisi) bir sonraki file_read
             * cagrisinin tampon disina TASMASINA (buffer overflow) ve
             * SEGFAULT'A yol aciyordu (gercek buyuk-olcekli kosuda
             * yakalandi). MAX_CARRY ile sinirlayip fazlasini atlamak,
             * cok nadir bir kucuk veri kaybi pahasina bunu onler. */
            console_write("  [UYARI] cok uzun bosluksuz blok bulundu, bir miktar veri atlaniyor (bayt konumu ~");
            console_write_u64(total_bytes_in);
            console_write_line(")");
            carry_len = MAX_CARRY;
        }
        for (u64 k = 0; k < carry_len; k++) window[k] = window[process_end + k];

        if (got == 0) break;
    }

    if (out_buf_n > 0) file_write(&out, out_buf, out_buf_n * sizeof(u32));

    file_close(&in);
    file_close(&out);

    console_write_line("");
    console_write("TOPLAM girdi bayt: "); console_write_u64(total_bytes_in); console_write_line("");
    console_write("TOPLAM uretilen token: "); console_write_u64(total_tokens); console_write_line("");
    console_write("Sikistirma orani (bayt/token): ");
    if (total_tokens > 0) console_write_u64(total_bytes_in / total_tokens);
    console_write_line("");

    allocator_destroy(&alloc);
    return 0;
}
