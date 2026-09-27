/* Katman 9 (devam) - METIN URETIMI (inference / generation).
 *
 * Faz 1 egitim kosusunun (train_lm.c) sonucunu GOZLE GORULUR sekilde
 * test etmek icin: checkpoints/lm_wiki_final.bin'i yukler, bir Turkce
 * tohum (seed) metni tokenlestirir, modeli otoregresif olarak calistirip
 * yeni token'lar uretir ve uretilen ID dizisini GERI Turkce metne
 * cozer (decode).
 *
 * Cozme (decode) stratejisi:
 *  - BPE sembolu (id >= bpe_base): bpe.symbol_bytes[id-bpe_base] --
 *    bayt-seviyeli BPE oldugu icin baglamdan bagimsiz, dogrudan kopya.
 *  - Ek (suffix) sembolu (id < bpe_base): tr_analyze_ending, SIMDIYE
 *    KADAR URETILEN TUM metin uzerinde calistirilir (fonoloji motoru
 *    METINDEKI EN SON unluyu/kod-noktasini ileri tarayarak bulur, bu
 *    yuzden tum tamponu vermek -- sadece son kelimeyi degil -- ayni
 *    dogru sonucu verir ama daha basittir); tr_resolve_suffix ile
 *    dogru uyumlu (harmonik) yuzey formu uretilip tampona eklenir,
 *    gerekiyorsa tr_apply_root_change ile kok sonu yumusatilir. Bu,
 *    tokenize_corpus.c'nin KODLARKEN kullandigi AYNI fonolojik
 *    kurallarin TERS yonde calistirilmasidir. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/file_io.h"
#include "../runtime/prng.h"
#include "../runtime/utf8.h"
#include "../runtime/mathlib.h"
#include "../runtime/strutil.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/rope.h"
#include "../model/attention.h"
#include "../model/lm_model.h"
#include "../model/gpu_ops.h"
#include "../training/checkpoint.h"
#include "../tokenizer/turkish_phon.h"
#include "../tokenizer/suffixes.h"
#include "../tokenizer/lexicon.h"
#include "../tokenizer/freqtable.h"
#include "../tokenizer/tokenize.h"

#define VOCAB_SIZE   31769u
#define D_MODEL      384u
#define NUM_HEADS    6u
#define NUM_LAYERS   12u
#define D_FF         1024u
#define CTX_MAX      1024u /* Faz 3b: model 1024 baglamla egitildi */
#define EPS          1e-5f
#define CKPT_PATH    "checkpoints/lm_wiki_faz3_best.bin" /* Faz 3: en dusuk dogrulama kaybi */

#define NUM_GENERATE 80u
#define TEMPERATURE  0.85f
#define DECODE_BUF_CAP 16384

static f32 g_probs[VOCAB_SIZE + 8];

static u32 sample_next(PCGState* rng, const f32* logits, u32 vocab, f32 temperature) {
    f32 max_v = logits[0];
    for (u32 j = 1; j < vocab; j++) if (logits[j] > max_v) max_v = logits[j];
    f64 sum = 0.0;
    for (u32 j = 0; j < vocab; j++) {
        f32 e = m_expf((logits[j] - max_v) / temperature);
        g_probs[j] = e;
        sum += (f64)e;
    }
    f64 r = pcg_uniform(rng) * sum;
    f64 c = 0.0;
    for (u32 j = 0; j < vocab; j++) {
        c += (f64)g_probs[j];
        if (c >= r) return j;
    }
    return vocab - 1;
}

static u32 encode_text(const TokenizerVocab* voc, const StrHashSet* lex, const StrCounter* freq,
                        const char* text, u64 text_len, u32* out_ids, u32 max_ids) {
    u32 n = 0;
    u64 i = 0;
    while (i < text_len && n < max_ids) {
        u32 cp;
        u64 sz = utf8_decode(text + i, &cp);
        bool32 is_letter = tr_is_letter(cp);
        u64 start = i;
        i += sz;
        while (i < text_len) {
            u32 cp2;
            u64 sz2 = utf8_decode(text + i, &cp2);
            if (tr_is_letter(cp2) != is_letter) break;
            i += sz2;
        }
        u64 seg_start = start;
        while (seg_start < i && n < max_ids) {
            u64 chunk = i - seg_start;
            if (chunk > 100) chunk = 100;
            u32 ids[TOKENIZE_MAX_IDS];
            u32 k = tokenize_word(voc, lex, freq, text + seg_start, chunk, ids);
            for (u32 j = 0; j < k && n < max_ids; j++) out_ids[n++] = ids[j];
            seg_start += chunk;
        }
    }
    return n;
}

/* Uretilen (veya kodlanmis) bir ID dizisini Turkce metne geri cozer. */
static u64 decode_tokens(const TokenizerVocab* voc, const u32* ids, u32 n, char* out, u64 out_cap) {
    u64 len = 0;
    for (u32 i = 0; i < n; i++) {
        u32 id = ids[i];
        if (len + 32 >= out_cap) break; /* guvenlik payi */

        if (id < voc->bpe_base) {
            const Suffix* s = &g_suffix_table[id];
            WordEnding end = tr_analyze_ending(out, len);
            char sfx_buf[16];
            u32 softened = 0;
            u64 sfx_len = tr_resolve_suffix(s->template_str, end, s->buffer_consonant,
                                             s->triggers_softening, s->drop_initial_vowel_after_vowel,
                                             sfx_buf, &softened);
            if (softened != 0 && len > 0) {
                /* Yerinde: sadece son kod noktasi degisir, onceki baytlar
                 * kendi uzerine kopyalanir (sabit boyutlu gecici tampon
                 * uzun metinlerde tasardi). */
                len = tr_apply_root_change(out, len, softened, out);
            }
            for (u64 k = 0; k < sfx_len; k++) out[len++] = sfx_buf[k];
        } else if (id < voc->bpe_base + voc->bpe.num_symbols) {
            u32 sid = id - voc->bpe_base;
            const u8* bytes = voc->bpe.symbol_bytes[sid];
            u32 blen = voc->bpe.symbol_lens[sid];
            for (u32 k = 0; k < blen; k++) out[len++] = (char)bytes[k];
        }
        /* ozel token'lar (pad/bos/eos) -- gorunur bayt uretmiyor. */
    }
    out[len] = 0;
    return len;
}

int main(void) {
    console_write_line("=== Turkce LLM - Metin Uretimi (Faz 3 modeli) ===");

    Allocator persist = allocator_create(2ull * 1024 * 1024 * 1024);

    console_write_line("Tokenizer/lexicon/frekans tablosu yukleniyor...");
    u64 word_count = 0;
    StrHashSet lex = lexicon_load(&persist, "data/raw/kok_adaylari.txt", &word_count);
    StrCounter freq = freqtable_load(&persist, "data/raw/kelime_frekans.txt");
    TokenizerVocab voc = tokenizer_init(&persist, "data/raw/bpe_merges.txt");
    console_write("Vocab boyutu: "); console_write_u64(voc.vocab_size); console_write_line("");

    gpu_ops_init(&persist, "cuda/kernels.ptx");

    console_write_line("Model olusturuluyor ve checkpoint yukleniyor...");
    PCGState dummy_rng = pcg_seed(1, 1); /* lm_init sadece agirliklarin SEKLINI/tensorlarini kurar; degerler checkpoint'ten gelecek */
    LMModel model = lm_init(&persist, &dummy_rng, voc.vocab_size, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, EPS);

    Node* params[LM_MAX_LAYERS * 12 + 2];
    u32 num_params = lm_collect_params(&model, params);

    Allocator scratch = allocator_create(1024ull * 1024 * 1024);
    bool32 ok = checkpoint_load(&scratch, CKPT_PATH, params, num_params, NULL_PTR);
    allocator_destroy(&scratch);
    if (!ok) {
        console_write("[HATA] Checkpoint yuklenemedi: "); console_write_line(CKPT_PATH);
        return 1;
    }
    console_write_line("Checkpoint yuklendi.");

    /* --- Tohum (seed) metni kodla --- */
    const char* seed_text = "Türkiye'nin başkenti";
    u32 context[4096];
    u32 context_len = encode_text(&voc, &lex, &freq, seed_text, str_len(seed_text), context, 4096);

    console_write("Tohum: \""); console_write(seed_text); console_write("\" -> ");
    console_write_u64(context_len); console_write(" token"); console_write_line("");

    /* Tur roundtrip saglamasi: kodlanan tohumu geri coz, orijinalle karsilastir. */
    char roundtrip[1024];
    decode_tokens(&voc, context, context_len, roundtrip, 1024);
    console_write("Roundtrip kontrolu: \""); console_write(roundtrip); console_write_line("\"");

    /* --- Otoregresif uretim --- */
    PCGState gen_rng = pcg_seed(4242, 7);
    u64 head_dim = D_MODEL / NUM_HEADS;

    console_write_line("Uretiliyor...");
    for (u32 step = 0; step < NUM_GENERATE && context_len < 4000; step++) {
        u32 cur_len = (context_len < CTX_MAX) ? context_len : CTX_MAX;
        const u32* window_ids = context + (context_len - cur_len);

        Allocator step_alloc = allocator_create(8ull * 1024 * 1024 * 1024); /* 1024 baglamda dikkat matrisleri buyuk; mmap tembel ayirir */

        u64 cshape[2] = { cur_len, head_dim / 2 };
        Tensor cos_t = tensor_create(&step_alloc, cshape, 2);
        Tensor sin_t = tensor_create(&step_alloc, cshape, 2);
        rope_build_tables(&cos_t, &sin_t, cur_len, head_dim, 10000.0f);
        Node* cos_leaf = node_leaf(&step_alloc, cos_t, FALSE);
        Node* sin_leaf = node_leaf(&step_alloc, sin_t, FALSE);

        u64 mshape[2] = { cur_len, cur_len };
        Tensor mask_t = tensor_create(&step_alloc, mshape, 2);
        build_causal_mask(&mask_t);
        Node* mask_leaf = node_leaf(&step_alloc, mask_t, FALSE);

        Node* logits = lm_forward(&step_alloc, &model, window_ids, 1, cur_len, cos_leaf, sin_leaf, mask_leaf, 1, 1);
        const f32* last_row = logits->value.data + (u64)(cur_len - 1) * voc.vocab_size;
        u32 next_id = sample_next(&gen_rng, last_row, voc.vocab_size, TEMPERATURE);

        allocator_destroy(&step_alloc);

        context[context_len++] = next_id;
        if (next_id == voc.eos_id) break;
    }

    char out_text[DECODE_BUF_CAP];
    decode_tokens(&voc, context, context_len, out_text, DECODE_BUF_CAP);

    console_write_line("");
    console_write_line("=== URETILEN METIN ===");
    console_write_line(out_text);

    gpu_ops_shutdown();
    allocator_destroy(&persist);
    return 0;
}
