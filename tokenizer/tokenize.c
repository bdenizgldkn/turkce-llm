#include "tokenize.h"
#include "suffixes.h"
#include "analyzer.h"
#include "disambiguate.h"

TokenizerVocab tokenizer_init(Allocator* alloc, const char* bpe_merges_path) {
    TokenizerVocab v;
    v.bpe = bpe_load(alloc, bpe_merges_path);
    v.suffix_base = 0;
    v.bpe_base = g_suffix_table_count;
    v.pad_id = v.bpe_base + v.bpe.num_symbols;
    v.bos_id = v.pad_id + 1;
    v.eos_id = v.pad_id + 2;
    v.vocab_size = v.pad_id + 3;
    return v;
}

u32 tokenize_word(const TokenizerVocab* voc, const StrHashSet* lex, const StrCounter* freq,
                   const char* word, u64 word_len, u32* out_ids) {
    AnalysisResult r = analyze_word(lex, word, word_len);

    if (r.num_parses > 0) {
        u32 best = disambiguate(freq, &r);
        const Parse* p = &r.parses[best];

        u32 bpe_ids[BPE_MAX_WORD_BYTES];
        u32 root_n = bpe_encode(&voc->bpe, p->root, p->root_len, bpe_ids);

        u32 n = 0;
        for (u32 i = 0; i < root_n; i++) out_ids[n++] = voc->bpe_base + bpe_ids[i];
        for (u32 i = 0; i < p->num_suffixes; i++) out_ids[n++] = voc->suffix_base + p->suffix_indices[i];
        return n;
    }

    /* Fallback: hic morfolojik ayristirma bulunamadi -- tum kelimeyi
     * dogrudan BPE ile kodla (ozel isim, yabanci kelime, yazim hatasi vb.). */
    u32 bpe_ids[BPE_MAX_WORD_BYTES];
    u32 n = bpe_encode(&voc->bpe, word, word_len, bpe_ids);
    for (u32 i = 0; i < n; i++) out_ids[i] = voc->bpe_base + bpe_ids[i];
    return n;
}
