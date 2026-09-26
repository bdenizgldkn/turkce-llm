#include "bpe.h"
#include "../runtime/sort.h"
#include "../runtime/file_io.h"

typedef struct TrainWord {
    u32 syms[BPE_MAX_WORD_BYTES];
    u32 len;
    u64 freq;
} TrainWord;

static void pair_key(u32 left, u32 right, char* out8) {
    /* 8 ham bayt: buyuk-kucuk harf sirasi onemli degil, sadece tutarli olsun. */
    out8[0] = (char)(left >> 24); out8[1] = (char)(left >> 16);
    out8[2] = (char)(left >> 8);  out8[3] = (char)left;
    out8[4] = (char)(right >> 24); out8[5] = (char)(right >> 16);
    out8[6] = (char)(right >> 8);  out8[7] = (char)right;
}

static void unpack_pair_key(const char* k8, u32* left, u32* right) {
    *left = ((u32)(u8)k8[0] << 24) | ((u32)(u8)k8[1] << 16) | ((u32)(u8)k8[2] << 8) | (u32)(u8)k8[3];
    *right = ((u32)(u8)k8[4] << 24) | ((u32)(u8)k8[5] << 16) | ((u32)(u8)k8[6] << 8) | (u32)(u8)k8[7];
}

BpeVocab bpe_train(Allocator* alloc, const char** words, const u64* word_lens,
                    const u64* freqs, u64 num_words, u32 target_vocab_size) {
    BpeVocab vocab;
    vocab.num_symbols = 256;
    vocab.symbol_bytes = (u8**)allocator_alloc(alloc, (u64)target_vocab_size * sizeof(u8*));
    vocab.symbol_lens = (u32*)allocator_alloc(alloc, (u64)target_vocab_size * sizeof(u32));
    for (u32 i = 0; i < 256; i++) {
        u8* b = (u8*)allocator_alloc(alloc, 1);
        b[0] = (u8)i;
        vocab.symbol_bytes[i] = b;
        vocab.symbol_lens[i] = 1;
    }

    u32 max_merges = target_vocab_size - 256;
    vocab.merges = (BpeMerge*)allocator_alloc(alloc, (u64)max_merges * sizeof(BpeMerge));
    vocab.num_merges = 0;

    /* Egitim kelimelerini bayt-seviyeli sembollere ayir. */
    TrainWord* tw = (TrainWord*)allocator_alloc(alloc, num_words * sizeof(TrainWord));
    for (u64 i = 0; i < num_words; i++) {
        u64 L = word_lens[i];
        if (L > BPE_MAX_WORD_BYTES) L = BPE_MAX_WORD_BYTES;
        for (u64 j = 0; j < L; j++) tw[i].syms[j] = (u8)words[i][j];
        tw[i].len = (u32)L;
        tw[i].freq = freqs[i];
    }

    /* Cift-sayim icin tekrar kullanilan, her iterasyonda sifirlanan
     * (arena reset ile) yardimci ayirici. */
    Allocator scratch = allocator_create(512ull * 1024 * 1024);

    while (vocab.num_symbols < target_vocab_size) {
        arena_reset(&scratch.backing);
        StrCounter pair_counts = strcounter_create(&scratch, 300000);

        for (u64 i = 0; i < num_words; i++) {
            for (u32 j = 0; j + 1 < tw[i].len; j++) {
                char key[8];
                pair_key(tw[i].syms[j], tw[i].syms[j + 1], key);
                /* frekans agirlikli: strcounter_increment yerine dogrudan
                 * artis miktarini uygulamak icin get+set kullaniyoruz. */
                u64 cur = strcounter_get(&pair_counts, key, 8);
                strcounter_set(&pair_counts, key, 8, cur + tw[i].freq);
            }
        }

        /* En sik gecen cifti bul. */
        u64 best_count = 0;
        u64 best_slot = (u64)-1;
        for (u64 s = 0; s < pair_counts.capacity; s++) {
            if (pair_counts.keys[s] != NULL_PTR && pair_counts.counts[s] > best_count) {
                best_count = pair_counts.counts[s];
                best_slot = s;
            }
        }

        if (best_slot == (u64)-1 || best_count == 0) break; /* birlestirilecek cift kalmadi */

        u32 left, right;
        unpack_pair_key(pair_counts.keys[best_slot], &left, &right);

        u32 new_id = vocab.num_symbols++;
        u32 blen = vocab.symbol_lens[left] + vocab.symbol_lens[right];
        u8* bbytes = (u8*)allocator_alloc(alloc, blen);
        for (u32 i = 0; i < vocab.symbol_lens[left]; i++) bbytes[i] = vocab.symbol_bytes[left][i];
        for (u32 i = 0; i < vocab.symbol_lens[right]; i++) bbytes[vocab.symbol_lens[left] + i] = vocab.symbol_bytes[right][i];
        vocab.symbol_bytes[new_id] = bbytes;
        vocab.symbol_lens[new_id] = blen;

        vocab.merges[vocab.num_merges].left = left;
        vocab.merges[vocab.num_merges].right = right;
        vocab.merges[vocab.num_merges].new_id = new_id;
        vocab.num_merges++;

        /* Bu birlestirmeyi tum kelimelere uygula (soldan saga, cakismayan). */
        for (u64 i = 0; i < num_words; i++) {
            u32 rlen = 0;
            for (u32 j = 0; j < tw[i].len; ) {
                if (j + 1 < tw[i].len && tw[i].syms[j] == left && tw[i].syms[j + 1] == right) {
                    tw[i].syms[rlen++] = new_id;
                    j += 2;
                } else {
                    tw[i].syms[rlen++] = tw[i].syms[j];
                    j++;
                }
            }
            tw[i].len = rlen;
        }
    }

    allocator_destroy(&scratch);

    /* Kodlama icin hizli oncelik tablosu: (left,right) -> rank+1. */
    vocab.merge_rank = strcounter_create(alloc, (u64)vocab.num_merges + 16);
    for (u32 r = 0; r < vocab.num_merges; r++) {
        char key[8];
        pair_key(vocab.merges[r].left, vocab.merges[r].right, key);
        strcounter_set(&vocab.merge_rank, key, 8, (u64)r + 1);
    }

    return vocab;
}

static void write_u32_decimal(FileHandle* f, u32 v) {
    char buf[12];
    i32 n = 0;
    if (v == 0) { buf[n++] = '0'; }
    else { char tmp[12]; i32 t = 0; while (v > 0) { tmp[t++] = (char)('0' + v % 10); v /= 10; } while (t > 0) buf[n++] = tmp[--t]; }
    file_write(f, buf, (u64)n);
}

void bpe_save(const BpeVocab* vocab, const char* path) {
    FileHandle f = file_open_write(path);
    for (u32 i = 0; i < vocab->num_merges; i++) {
        write_u32_decimal(&f, vocab->merges[i].left);
        file_write(&f, " ", 1);
        write_u32_decimal(&f, vocab->merges[i].right);
        file_write(&f, "\n", 1);
    }
    file_close(&f);
}

BpeVocab bpe_load(Allocator* alloc, const char* path) {
    BpeVocab vocab;
    u64 size = 0;
    char* buf = (char*)file_read_entire(path, alloc, &size);

    /* Once satir sayisini (=birlestirme sayisini) belirle. */
    u64 line_count = 0;
    for (u64 i = 0; i < size; i++) if (buf[i] == '\n') line_count++;

    u32 target_vocab_size = 256 + (u32)line_count;
    vocab.num_symbols = 256;
    vocab.symbol_bytes = (u8**)allocator_alloc(alloc, (u64)target_vocab_size * sizeof(u8*));
    vocab.symbol_lens = (u32*)allocator_alloc(alloc, (u64)target_vocab_size * sizeof(u32));
    for (u32 i = 0; i < 256; i++) {
        u8* b = (u8*)allocator_alloc(alloc, 1);
        b[0] = (u8)i;
        vocab.symbol_bytes[i] = b;
        vocab.symbol_lens[i] = 1;
    }
    vocab.merges = (BpeMerge*)allocator_alloc(alloc, line_count * sizeof(BpeMerge));
    vocab.num_merges = 0;

    u64 i = 0;
    while (i < size) {
        u64 start = i;
        while (i < size && buf[i] != '\n') i++;
        u64 end = i;
        if (end > start) {
            u64 k = start;
            u32 left = 0, right = 0;
            while (k < end && buf[k] != ' ') { left = left * 10 + (u32)(buf[k] - '0'); k++; }
            k++; /* bosluk atla */
            while (k < end) { right = right * 10 + (u32)(buf[k] - '0'); k++; }

            u32 new_id = vocab.num_symbols++;
            u32 blen = vocab.symbol_lens[left] + vocab.symbol_lens[right];
            u8* bbytes = (u8*)allocator_alloc(alloc, blen);
            for (u32 b = 0; b < vocab.symbol_lens[left]; b++) bbytes[b] = vocab.symbol_bytes[left][b];
            for (u32 b = 0; b < vocab.symbol_lens[right]; b++) bbytes[vocab.symbol_lens[left] + b] = vocab.symbol_bytes[right][b];
            vocab.symbol_bytes[new_id] = bbytes;
            vocab.symbol_lens[new_id] = blen;

            vocab.merges[vocab.num_merges].left = left;
            vocab.merges[vocab.num_merges].right = right;
            vocab.merges[vocab.num_merges].new_id = new_id;
            vocab.num_merges++;
        }
        i++;
    }

    vocab.merge_rank = strcounter_create(alloc, (u64)vocab.num_merges + 16);
    for (u32 r = 0; r < vocab.num_merges; r++) {
        char key[8];
        pair_key(vocab.merges[r].left, vocab.merges[r].right, key);
        strcounter_set(&vocab.merge_rank, key, 8, (u64)r + 1);
    }

    return vocab;
}

u32 bpe_encode(const BpeVocab* vocab, const char* word, u64 word_len, u32* out_ids) {
    u32 len = (u32)word_len;
    if (len > BPE_MAX_WORD_BYTES) len = BPE_MAX_WORD_BYTES;
    for (u32 i = 0; i < len; i++) out_ids[i] = (u8)word[i];

    for (;;) {
        u64 best_rank = (u64)-1;
        u32 best_pos = (u32)-1;
        u32 best_new_id = 0;

        for (u32 i = 0; i + 1 < len; i++) {
            char key[8];
            pair_key(out_ids[i], out_ids[i + 1], key);
            u64 r1 = strcounter_get(&vocab->merge_rank, key, 8);
            if (r1 != 0 && (r1 - 1) < best_rank) {
                best_rank = r1 - 1;
                best_pos = i;
                best_new_id = vocab->merges[r1 - 1].new_id;
            }
        }

        if (best_pos == (u32)-1) break;

        out_ids[best_pos] = best_new_id;
        for (u32 k = best_pos + 1; k + 1 < len; k++) out_ids[k] = out_ids[k + 1];
        len--;
    }

    return len;
}
