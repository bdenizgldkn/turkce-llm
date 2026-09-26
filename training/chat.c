/* Katman 9 (devam) - ETKILESIMLI TERMINAL SOHBETI.
 *
 * generate.c'nin tek seferlik demo halinin etkilesimli (interaktif)
 * versiyonu: checkpoint bir kez yuklenir, sonra kullanicidan terminalden
 * surekli metin okunup her seferinde modelin devamini uretip yazdirir.
 *
 * ONEMLI DURUSTLUK NOTU: Bu, bir "sohbet" (chat/instruct) modeli
 * DEGILDIR -- egitim verisi ham Vikipedi metniydi (soru-cevap ya da
 * diyalog formatinda degil). Yani model, yazdiginiz metni bir SORUYA
 * CEVAP olarak degil, Vikipedi tarzi bir metnin DEVAMI gibi tamamlar.
 * Bu, projenin Faz 1 modelinin (4.7M parametre, kisa egitim) dogal ve
 * beklenen davranisidir. */
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
#define CTX_MAX      128u
#define EPS          1e-5f

#define NUM_GENERATE   80u
#define TEMPERATURE    0.85f
#define DECODE_BUF_CAP 16384
#define INPUT_BUF_CAP  2048

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

static u64 decode_tokens(const TokenizerVocab* voc, const u32* ids, u32 n, char* out, u64 out_cap) {
    u64 len = 0;
    for (u32 i = 0; i < n; i++) {
        u32 id = ids[i];
        if (len + 32 >= out_cap) break;

        if (id < voc->bpe_base) {
            const Suffix* s = &g_suffix_table[id];
            WordEnding end = tr_analyze_ending(out, len);
            char sfx_buf[16];
            u32 softened = 0;
            u64 sfx_len = tr_resolve_suffix(s->template_str, end, s->buffer_consonant,
                                             s->triggers_softening, s->drop_initial_vowel_after_vowel,
                                             sfx_buf, &softened);
            if (softened != 0 && len > 0) {
                char tmp[4096];
                u64 new_len = tr_apply_root_change(out, len, softened, tmp);
                for (u64 k = 0; k < new_len; k++) out[k] = tmp[k];
                len = new_len;
            }
            for (u64 k = 0; k < sfx_len; k++) out[len++] = sfx_buf[k];
        } else if (id < voc->bpe_base + voc->bpe.num_symbols) {
            u32 sid = id - voc->bpe_base;
            const u8* bytes = voc->bpe.symbol_bytes[sid];
            u32 blen = voc->bpe.symbol_lens[sid];
            for (u32 k = 0; k < blen; k++) out[len++] = (char)bytes[k];
        }
    }
    out[len] = 0;
    return len;
}

/* Verilen tohum metinden NUM_GENERATE token uretip tam metni (tohum +
 * devam) out_text'e yazar. */
static void generate_from_seed(const TokenizerVocab* voc, LMModel* model, PCGState* gen_rng,
                                const char* seed_text, u64 seed_len,
                                const StrHashSet* lex, const StrCounter* freq,
                                char* out_text, u64 out_cap) {
    u32 context[CTX_MAX + NUM_GENERATE + 16];
    u32 context_len = encode_text(voc, lex, freq, seed_text, seed_len, context, CTX_MAX);
    u64 head_dim = D_MODEL / NUM_HEADS;

    for (u32 step = 0; step < NUM_GENERATE; step++) {
        u32 cur_len = (context_len < CTX_MAX) ? context_len : CTX_MAX;
        const u32* window_ids = context + (context_len - cur_len);

        Allocator step_alloc = allocator_create(768ull * 1024 * 1024);

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

        Node* logits = lm_forward(&step_alloc, model, window_ids, 1, cur_len, cos_leaf, sin_leaf, mask_leaf, 1, 1);
        const f32* last_row = logits->value.data + (u64)(cur_len - 1) * voc->vocab_size;
        u32 next_id = sample_next(gen_rng, last_row, voc->vocab_size, TEMPERATURE);

        allocator_destroy(&step_alloc);

        context[context_len++] = next_id;
        if (next_id == voc->eos_id) break;
    }

    decode_tokens(voc, context, context_len, out_text, out_cap);
}

int main(void) {
    console_write_line("=== Turkce LLM - Terminal Sohbeti (Faz 1 modeli) ===");
    console_write_line("NOT: Bu bir 'sohbet/talimat' modeli DEGIL -- ham Vikipedi metniyle");
    console_write_line("egitildi. Yazdiginiz metnin CEVABINI degil, Vikipedi tarzinda bir");
    console_write_line("DEVAMINI uretir. Cikmak icin bos satir veya 'exit' yazip Enter'a basin.");
    console_write_line("");

    Allocator persist = allocator_create(2ull * 1024 * 1024 * 1024);

    console_write_line("Tokenizer/lexicon/frekans tablosu yukleniyor...");
    u64 word_count = 0;
    StrHashSet lex = lexicon_load(&persist, "data/raw/kok_adaylari.txt", &word_count);
    StrCounter freq = freqtable_load(&persist, "data/raw/kelime_frekans.txt");
    TokenizerVocab voc = tokenizer_init(&persist, "data/raw/bpe_merges.txt");

    gpu_ops_init(&persist, "cuda/kernels.ptx");

    console_write_line("Model olusturuluyor ve checkpoint yukleniyor...");
    PCGState dummy_rng = pcg_seed(1, 1);
    LMModel model = lm_init(&persist, &dummy_rng, voc.vocab_size, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, EPS);

    Node* params[LM_MAX_LAYERS * 12 + 2];
    u32 num_params = lm_collect_params(&model, params);

    Allocator scratch = allocator_create(1024ull * 1024 * 1024);
    bool32 ok = checkpoint_load(&scratch, "checkpoints/lm_wiki_faz2_final.bin", params, num_params, NULL_PTR);
    allocator_destroy(&scratch);
    if (!ok) {
        console_write_line("[HATA] Checkpoint yuklenemedi (checkpoints/lm_wiki_faz2_final.bin).");
        return 1;
    }
    console_write_line("Hazir. Bir Turkce metin yazin (orn. 'Istanbul bir').");
    console_write_line("");

    PCGState gen_rng = pcg_seed(4242, 7);
    char input_buf[INPUT_BUF_CAP];
    char out_text[DECODE_BUF_CAP];

    for (;;) {
        console_write("Siz> ");
        u32 n = console_read_line(input_buf, INPUT_BUF_CAP);
        if (n == 0) break;
        if (str_eq(input_buf, "exit") || str_eq(input_buf, "quit") || str_eq(input_buf, "cik") || str_eq(input_buf, "çık")) break;

        generate_from_seed(&voc, &model, &gen_rng, input_buf, n, &lex, &freq, out_text, DECODE_BUF_CAP);

        console_write("Model> ");
        console_write_line(out_text);
        console_write_line("");
    }

    console_write_line("Gorusmek uzere.");

    gpu_ops_shutdown();
    allocator_destroy(&persist);
    return 0;
}
