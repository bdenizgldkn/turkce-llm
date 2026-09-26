/* Katman 6 - BPE egitim araci.
 * kelime_frekans.txt'yi yukler, frekansa gore sirala, en sik gecen
 * top-K kelimeyi egitim korpüsü olarak kullanip BPE birlestirmelerini
 * ogrenir ve data/raw/bpe_merges.txt'ye kaydeder. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/file_io.h"
#include "../runtime/sort.h"
#include "bpe.h"

#define TOP_K 100000
#define TARGET_VOCAB 31744 /* 32000 - 256: sufiks/ozel token'lar icin kucuk bir pay birakir */

int main(void) {
    console_write_line("=== BPE Egitimi ===");

    Allocator alloc = allocator_create(2500ull * 1024 * 1024);

    u64 size = 0;
    char* buf = (char*)file_read_entire("data/raw/kelime_frekans.txt", &alloc, &size);
    if (!buf) { console_write_line("HATA: kelime_frekans.txt okunamadi"); return 1; }

    /* Satirlari ayristir: word[], word_len[], freq[] */
    u64 cap = 400000;
    const char** words = (const char**)allocator_alloc(&alloc, cap * sizeof(char*));
    u64* wlens = (u64*)allocator_alloc(&alloc, cap * sizeof(u64));
    u64* freqs = (u64*)allocator_alloc(&alloc, cap * sizeof(u64));
    u64 n = 0;

    u64 i = 0;
    while (i < size && n < cap) {
        u64 start = i;
        while (i < size && buf[i] != '\n') i++;
        u64 end = i;
        if (end > start && buf[end - 1] == '\r') end--;

        u64 tab = start;
        while (tab < end && buf[tab] != '\t') tab++;

        if (tab < end) {
            words[n] = buf + start;
            wlens[n] = tab - start;
            u64 val = 0;
            for (u64 k = tab + 1; k < end; k++) val = val * 10 + (u64)(buf[k] - '0');
            freqs[n] = val;
            n++;
        }
        i++;
    }

    console_write("Toplam kelime turu: "); console_write_u64(n); console_write_line("");

    /* Frekansa gore azalan sirala. */
    u32* idx = (u32*)allocator_alloc(&alloc, n * sizeof(u32));
    for (u64 k = 0; k < n; k++) idx[k] = (u32)k;
    sort_indices_by_u64_desc(freqs, idx, n);

    u64 topk = (n < TOP_K) ? n : TOP_K;
    const char** twords = (const char**)allocator_alloc(&alloc, topk * sizeof(char*));
    u64* twlens = (u64*)allocator_alloc(&alloc, topk * sizeof(u64));
    u64* tfreqs = (u64*)allocator_alloc(&alloc, topk * sizeof(u64));
    for (u64 k = 0; k < topk; k++) {
        twords[k] = words[idx[k]];
        twlens[k] = wlens[idx[k]];
        tfreqs[k] = freqs[idx[k]];
    }

    console_write("Egitim icin secilen (top-K) kelime: "); console_write_u64(topk); console_write_line("");
    console_write("En sik kelime frekansi: "); console_write_u64(tfreqs[0]); console_write_line("");
    console_write("En seyrek (top-K icinde) kelime frekansi: "); console_write_u64(tfreqs[topk - 1]); console_write_line("");

    console_write_line("BPE egitimi basliyor...");
    BpeVocab vocab = bpe_train(&alloc, twords, twlens, tfreqs, topk, TARGET_VOCAB);
    console_write("Ogrenilen birlestirme sayisi: "); console_write_u64(vocab.num_merges); console_write_line("");
    console_write("Toplam sembol (vocab) sayisi: "); console_write_u64(vocab.num_symbols); console_write_line("");

    bpe_save(&vocab, "data/raw/bpe_merges.txt");
    console_write_line("Kaydedildi: data/raw/bpe_merges.txt");

    /* Ornek kodlamalar (gozle kontrol icin) */
    const char* samples[] = { "kitap", "arabalarımızdan", "gidiyorlar", "xzqvt123yabancı" };
    for (i32 s = 0; s < 4; s++) {
        u64 slen = 0; while (samples[s][slen]) slen++;
        u32 ids[256];
        u32 n_ids = bpe_encode(&vocab, samples[s], slen, ids);
        console_write(samples[s]); console_write(" -> ");
        console_write_u64(n_ids); console_write(" parca: ");
        for (u32 p = 0; p < n_ids; p++) {
            console_write("[");
            for (u32 b = 0; b < vocab.symbol_lens[ids[p]]; b++) {
                char c[2] = { (char)vocab.symbol_bytes[ids[p]][b], 0 };
                console_write(c);
            }
            console_write("]");
        }
        console_write_line("");
    }

    allocator_destroy(&alloc);
    return 0;
}
