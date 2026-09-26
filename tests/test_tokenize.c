/* Katman 6 - NIHAI uctan uca test: morfolojik ayristirma + istatistiksel
 * belirsizlik giderme + BPE fallback hep birlikte, gercek verilerle. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"
#include "../tokenizer/lexicon.h"
#include "../tokenizer/freqtable.h"
#include "../tokenizer/tokenize.h"
#include "../tokenizer/suffixes.h"

static const char* id_to_label(const TokenizerVocab* voc, u32 id, char* scratch) {
    if (id < voc->bpe_base) {
        return g_suffix_table[id].gloss;
    }
    if (id == voc->pad_id) return "<pad>";
    if (id == voc->bos_id) return "<bos>";
    if (id == voc->eos_id) return "<eos>";

    u32 sym = id - voc->bpe_base;
    u32 n = voc->bpe.symbol_lens[sym];
    for (u32 i = 0; i < n; i++) scratch[i] = (char)voc->bpe.symbol_bytes[sym][i];
    scratch[n] = 0;
    return scratch;
}

static void show(const TokenizerVocab* voc, const StrHashSet* lex, const StrCounter* freq, const char* word) {
    u64 len = str_len(word);
    u32 ids[TOKENIZE_MAX_IDS];
    u32 n = tokenize_word(voc, lex, freq, word, len, ids);

    console_write(word); console_write(" ("); console_write_u64(n); console_write(" token) -> ");
    for (u32 i = 0; i < n; i++) {
        char scratch[64];
        console_write("[");
        console_write(id_to_label(voc, ids[i], scratch));
        console_write("]");
    }
    console_write_line("");
}

int main(void) {
    console_write_line("=== Katman 6 NIHAI: Tam Tokenizer Testi ===");

    Allocator alloc = allocator_create(3000ull * 1024 * 1024);

    console_write_line("Yukleniyor: kok sozlugu, frekans tablosu, BPE...");
    u64 word_count = 0;
    StrHashSet lex = lexicon_load(&alloc, "data/raw/kok_adaylari.txt", &word_count);
    StrCounter freq = freqtable_load(&alloc, "data/raw/kelime_frekans.txt");
    TokenizerVocab voc = tokenizer_init(&alloc, "data/raw/bpe_merges.txt");

    console_write("Toplam vocab boyutu: "); console_write_u64(voc.vocab_size); console_write_line("");
    console_write("  - ek token'lari: "); console_write_u64(voc.bpe_base); console_write_line("");
    console_write("  - BPE sembolleri: "); console_write_u64(voc.bpe.num_symbols); console_write_line("");
    console_write_line("");

    const char* samples[] = {
        "kitap", "kitaplarımızdan", "evlerde", "arabası", "gidiyorlar",
        "gözlük", "İstanbul", "xzqvt123", "günaydın", "üniversite"
    };
    for (i32 i = 0; i < 10; i++) show(&voc, &lex, &freq, samples[i]);

    allocator_destroy(&alloc);
    console_write_line("");
    console_write_line("(Bu dosyada otomatik PASS/FAIL yok -- gorsel/manuel inceleme icin.)");
    return 0;
}
