/* Faz 4 - GUNLUK KONUSMA DEVAM EGITIMI (bkz. PROJE_PLANI.md Bolum 24).
 *
 * Faz 3'un en iyi modelinden (lm_wiki_faz3_best.bin, Vikipedi dogrulama
 * kaybi 1,376) baslayip film altyazilari (OpenSubtitles v2024, tr) ile
 * Vikipedi'nin KARISIMI uzerinde egitime devam eder. Model mimarisi,
 * GPU egiticisi (model/gpu_train.c) ve checkpoint formati Faz 3 ile
 * AYNIDIR; farklar:
 *   - Her adimin BATCH_SEQS dizisinin SUB_SEQS'i altyazi, geri kalani
 *     Vikipedi (unutmayi -- catastrophic forgetting -- sinirlamak icin).
 *   - Adam durumu SIFIRDAN (yeni veri dagilimi, yeni lr takvimi); Faz 3
 *     checkpoint'inden sadece agirliklar yuklenir.
 *   - Iki dogrulama seti: Vikipedi (Faz 3 ile BIREBIR ayni pencereler ->
 *     1,376 ile karsilastirilabilir) ve altyazi. "En iyi" model agirlikli
 *     ortalamaya (karisim oranlariyla) gore secilir.
 *   - Adim 0'dan ONCE bir dogrulama olcumu yapilir (baslangic cizgisi).
 * Checkpoint'ten devam (faz4_latest) Faz 3'teki gibi calisir: veri tohumu
 * sadece adim numarasina bagli, Adam durumu checkpoint'te. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/file_io.h"
#include "../runtime/timer.h"
#include "../model/lm_model.h"
#include "../model/gpu_train.h"
#include "../training/adam.h"
#include "../training/checkpoint.h"
#include "../training/lr_schedule.h"

#define VOCAB_SIZE   31769u
#define D_MODEL      384u
#define NUM_HEADS    6u
#define NUM_LAYERS   12u
#define D_FF         1024u
#define SEQ_LEN      1024u
#define EPS          1e-5f

#define PEAK_LR      3e-4f  /* Faz 3'un yarisi: egitilmis agirliklari bozmadan yeni veriye uyum */
#define MIN_LR       3e-5f
#define WARMUP_STEPS 500u
#define GRAD_CLIP    1.0
#define BETA1        0.9f
#define BETA2        0.999f
#define ADAM_EPS     1e-8f

#define NUM_STEPS    25000u  /* ~358M token (~233M altyazi), L4'te ~8 saat */
#define LOG_EVERY    100u
#define CKPT_EVERY   1000u
#define EVAL_EVERY   500u
#define VAL_PERMILLE 10u     /* her korpusun SON %1'i dogrulama */
#define VAL_BATCHES  8u

/* Adim basina dizi: 14 x 1024 = 14.336 token (Faz 3 ile ayni); 9 altyazi + 5 Vikipedi (~%65 / %35). */
#define BATCH_SEQS   14u
#define SUB_SEQS     9u
#define WIKI_SEQS    (BATCH_SEQS - SUB_SEQS)

#define SUB_TOKENS_PATH  "data/raw/subtitle_tokens.bin"
#define WIKI_TOKENS_PATH "data/raw/wikipedia_tokens.bin"
#define INIT_CKPT_PATH   "checkpoints/lm_wiki_faz3_best.bin"
#define CKPT_LATEST_PATH "checkpoints/lm_faz4_latest.bin"
#define CKPT_FINAL_PATH  "checkpoints/lm_faz4_final.bin"
#define CKPT_BEST_PATH   "checkpoints/lm_faz4_best.bin"
#define BEST_VAL_PATH    "checkpoints/lm_faz4_best_val.bin"
#define CKPT_PROBE_PATH  "checkpoints/.yazma_testi"

#define MAX_PARAMS  (LM_MAX_LAYERS * 12u + 2u)

typedef struct Corpus {
    const u32* tokens;
    u64 num_tokens;
    u64 train_tokens;  /* egitim pencereleri sadece [0, train_tokens) */
    u64 val_tokens;
    u64 val_stride;
} Corpus;

static void write_fixed3(f64 v) {
    if (v < 0) { console_write("-"); v = -v; }
    u64 scaled = (u64)(v * 1000.0 + 0.5);
    u64 ip = scaled / 1000;
    u64 fp = scaled % 1000;
    console_write_u64(ip);
    console_write(".");
    if (fp < 100) console_write("0");
    if (fp < 10) console_write("0");
    console_write_u64(fp);
}

static bool32 load_corpus(Allocator* alloc, const char* path, Corpus* c) {
    console_write("Token korpusu yukleniyor: "); console_write(path); console_write_line(" ...");
    u64 size = 0;
    void* raw = file_read_entire(path, alloc, &size);
    if (raw == NULL_PTR || size == 0) {
        console_write("[HATA] Token dosyasi okunamadi veya bos: "); console_write_line(path);
        return FALSE;
    }
    c->tokens = (const u32*)raw;
    c->num_tokens = size / sizeof(u32);
    c->val_tokens = c->num_tokens * VAL_PERMILLE / 1000u;
    c->train_tokens = c->num_tokens - c->val_tokens;
    if (c->val_tokens < (u64)VAL_BATCHES * BATCH_SEQS + SEQ_LEN + 1) {
        console_write("[HATA] Korpus dogrulama seti icin cok kucuk: "); console_write_line(path);
        return FALSE;
    }
    c->val_stride = (c->val_tokens - SEQ_LEN - 1) / ((u64)VAL_BATCHES * BATCH_SEQS);
    console_write("  "); console_write_u64(c->num_tokens);
    console_write(" token | egitim: "); console_write_u64(c->train_tokens);
    console_write(" | dogrulama: "); console_write_u64(c->val_tokens); console_write_line("");
    return TRUE;
}

/* Dogrulama blogunda esit aralikli SABIT VAL_BATCHES x BATCH_SEQS pencere
 * (train_lm_gpu.c ile ayni duzen -> Vikipedi degeri Faz 3 ile karsilastirilabilir). */
static f32 eval_corpus(GpuTrainer* gt, const Corpus* c, u32* ids, u32* tgt) {
    f32 sum = 0.0f;
    for (u32 vb = 0; vb < VAL_BATCHES; vb++) {
        for (u32 w = 0; w < BATCH_SEQS; w++) {
            u64 start = c->train_tokens + ((u64)vb * BATCH_SEQS + w) * c->val_stride;
            for (u32 t = 0; t < SEQ_LEN; t++) {
                ids[(u64)w * SEQ_LEN + t] = c->tokens[start + t];
                tgt[(u64)w * SEQ_LEN + t] = c->tokens[start + t + 1];
            }
        }
        sum += gpu_trainer_eval_loss(gt, ids, tgt);
    }
    return sum / (f32)VAL_BATCHES;
}

int main(void) {
    console_write_line("=== Turkce LLM - Faz 4: Gunluk Konusma Devam Egitimi (altyazi + Vikipedi) ===");
    console_write("seq_len="); console_write_u64(SEQ_LEN);
    console_write(" dizi/adim="); console_write_u64(BATCH_SEQS);
    console_write(" (altyazi="); console_write_u64(SUB_SEQS);
    console_write(", vikipedi="); console_write_u64(WIKI_SEQS);
    console_write(") adim="); console_write_u64(NUM_STEPS);
    console_write_line("");

    {
        FileHandle probe = file_open_write(CKPT_PROBE_PATH);
        if (!probe.valid) {
            console_write("[HATA] Checkpoint klasorune yazilamiyor: "); console_write_line(CKPT_PROBE_PATH);
            return 1;
        }
        file_close(&probe);
        file_delete(CKPT_PROBE_PATH);
    }

    Allocator persist = allocator_create(24ull * 1024 * 1024 * 1024);

    Corpus sub, wiki;
    if (!load_corpus(&persist, SUB_TOKENS_PATH, &sub)) return 1;
    if (!load_corpus(&persist, WIKI_TOKENS_PATH, &wiki)) return 1;

    PCGState model_rng = pcg_seed(1337, 1);
    LMModel model = lm_init(&persist, &model_rng, VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, EPS);
    Node* params[MAX_PARAMS];
    u32 num_params = lm_collect_params(&model, params);
    AdamOptimizer opt = adam_create(&persist, params, num_params, PEAK_LR, BETA1, BETA2, ADAM_EPS);

    /* Devam: faz4_latest varsa (agirlik + Adam). Yoksa Faz 3'un en iyi
     * modelinin SADECE agirliklari; Adam sifirdan (opt.t = 0). */
    u32 start_step = 0;
    {
        FileHandle probe = file_open_read(CKPT_LATEST_PATH);
        bool32 resume = probe.valid;
        file_close(&probe);

        Allocator ckpt_scratch = allocator_create(2ull * 1024 * 1024 * 1024);
        bool32 loaded = resume
            ? checkpoint_load(&ckpt_scratch, CKPT_LATEST_PATH, params, num_params, &opt)
            : checkpoint_load(&ckpt_scratch, INIT_CKPT_PATH, params, num_params, NULL_PTR);
        allocator_destroy(&ckpt_scratch);

        if (!loaded) {
            console_write("[HATA] Checkpoint yuklenemedi: ");
            console_write_line(resume ? CKPT_LATEST_PATH : INIT_CKPT_PATH);
            if (resume) console_write_line("       Faz 3'ten yeniden baslamak icin bu dosyayi silin ya da tasiyin.");
            return 1;
        }
        if (resume) {
            start_step = (u32)opt.t;
            console_write("Faz 4 checkpoint'inden devam: adim "); console_write_u64(start_step); console_write_line("");
        } else {
            console_write("Baslangic agirliklari: "); console_write(INIT_CKPT_PATH);
            console_write_line(" (Adam durumu sifirdan)");
        }
    }

    f32 best_val = 3.0e38f;
    {
        u64 sz = 0;
        Allocator bv_scratch = allocator_create(1024 * 1024);
        f32* bv = (f32*)file_read_entire(BEST_VAL_PATH, &bv_scratch, &sz);
        if (bv != NULL_PTR && sz == sizeof(f32)) best_val = *bv;
        allocator_destroy(&bv_scratch);
    }

    GpuTrainer gt = gpu_trainer_create(&persist, "cuda/train_kernels.ptx", &model, params, num_params,
                                       BATCH_SEQS, SEQ_LEN);
    gpu_trainer_upload(&gt, params, &opt);
    u32* batch_ids = (u32*)allocator_alloc(&persist, (u64)BATCH_SEQS * SEQ_LEN * sizeof(u32));
    u32* batch_tgt = (u32*)allocator_alloc(&persist, (u64)BATCH_SEQS * SEQ_LEN * sizeof(u32));
    console_write_line("GPU egiticisi hazir.");

    const f32 w_sub = (f32)SUB_SEQS / (f32)BATCH_SEQS;
    const f32 w_wiki = (f32)WIKI_SEQS / (f32)BATCH_SEQS;

    f64 loss_ema = -1.0;
    f64 t_train0 = timer_now_seconds();
    f64 last_log_elapsed = 0.0;

    /* step == start_step - 1 "sanal adim"i: egitimden once baslangic olcumu. */
    for (i64 step = (i64)start_step - 1; step < (i64)NUM_STEPS; step++) {
        bool32 do_eval;
        if (step < (i64)start_step) {
            do_eval = TRUE;
        } else {
            u32 s = (u32)step;
            opt.lr = lr_warmup_cosine(s, WARMUP_STEPS, NUM_STEPS, PEAK_LR, MIN_LR);
            /* Ilk SUB_SEQS satir altyazi, kalan WIKI_SEQS satir Vikipedi.
             * Tohumlar sadece adim numarasina bagli (devam = kesintisiz kosu). */
            gpu_trainer_sample_batch(sub.tokens, sub.train_tokens, SEQ_LEN, SUB_SEQS,
                                     4000000ull + s, batch_ids, batch_tgt);
            gpu_trainer_sample_batch(wiki.tokens, wiki.train_tokens, SEQ_LEN, WIKI_SEQS,
                                     8000000ull + s, batch_ids + (u64)SUB_SEQS * SEQ_LEN,
                                     batch_tgt + (u64)SUB_SEQS * SEQ_LEN);
            f32 loss_val = gpu_trainer_forward_backward(&gt, batch_ids, batch_tgt);
            f64 grad_norm = gpu_trainer_grad_norm(&gt);
            f32 clip_scale = (grad_norm > GRAD_CLIP) ? (f32)(GRAD_CLIP / grad_norm) : 1.0f;
            gpu_trainer_adam_step_scaled(&gt, &opt, clip_scale);

            loss_ema = (loss_ema < 0.0) ? (f64)loss_val : (0.98 * loss_ema + 0.02 * (f64)loss_val);

            if (s % LOG_EVERY == 0 || s == NUM_STEPS - 1) {
                f64 elapsed = timer_now_seconds() - t_train0;
                f64 interval = elapsed - last_log_elapsed;
                u32 n_in = (s == start_step) ? 1u : ((s % LOG_EVERY == 0) ? LOG_EVERY : (s % LOG_EVERY));
                console_write("adim "); console_write_u64(s);
                console_write(" | kayip="); write_fixed3((f64)loss_val);
                console_write(" | ema="); write_fixed3(loss_ema);
                console_write(" | gecen="); write_fixed3(elapsed); console_write(" sn");
                console_write(" | sn/adim="); write_fixed3(interval / (f64)n_in);
                console_write(" | lr*1e4="); write_fixed3((f64)opt.lr * 1e4);
                console_write(" | gnorm="); write_fixed3(grad_norm);
                console_write_line("");
                last_log_elapsed = elapsed;
            }
            do_eval = ((s + 1) % EVAL_EVERY == 0) || (s == NUM_STEPS - 1);
        }

        if (do_eval) {
            f64 te = timer_now_seconds();
            f32 v_sub = eval_corpus(&gt, &sub, batch_ids, batch_tgt);
            f32 v_wiki = eval_corpus(&gt, &wiki, batch_ids, batch_tgt);
            f32 v = w_sub * v_sub + w_wiki * v_wiki;
            console_write("  [dogrulama] adim "); console_write_u64((u64)(step + 1));
            console_write(" | altyazi="); write_fixed3((f64)v_sub);
            console_write(" | vikipedi="); write_fixed3((f64)v_wiki);
            console_write(" | agirlikli="); write_fixed3((f64)v);
            console_write(" | en_iyi="); write_fixed3(best_val < 1e38f ? (f64)best_val : 0.0);
            console_write(" | sure="); write_fixed3(timer_now_seconds() - te); console_write_line(" sn");
            t_train0 += timer_now_seconds() - te;

            /* Baslangic olcumu (Faz 3 modeli) "en iyi" sayilmaz. */
            if (step >= (i64)start_step && v < best_val) {
                gpu_trainer_download(&gt, params, &opt);
                if (checkpoint_save(CKPT_BEST_PATH, params, num_params, &opt)) {
                    best_val = v;
                    FileHandle bf = file_open_write(BEST_VAL_PATH);
                    if (bf.valid) { file_write(&bf, &best_val, sizeof(f32)); file_close(&bf); }
                    console_write_line("  [en iyi model kaydedildi]");
                } else {
                    console_write("  [HATA] en iyi model KAYDEDILEMEDI: "); console_write_line(CKPT_BEST_PATH);
                }
            }
        }

        if (step > (i64)start_step && ((step + 1) % CKPT_EVERY == 0)) {
            gpu_trainer_download(&gt, params, &opt);
            if (checkpoint_save(CKPT_LATEST_PATH, params, num_params, &opt)) {
                console_write("  [checkpoint kaydedildi: "); console_write_u64((u64)(step + 1));
                console_write_line(" adim tamamlandi]");
            } else {
                console_write("  [HATA] checkpoint KAYDEDILEMEDI: "); console_write_line(CKPT_LATEST_PATH);
            }
        }
    }

    gpu_trainer_download(&gt, params, &opt);
    bool32 final_saved = checkpoint_save(CKPT_FINAL_PATH, params, num_params, &opt);
    if (final_saved) {
        console_write("Egitim tamamlandi. Nihai checkpoint: "); console_write_line(CKPT_FINAL_PATH);
    } else {
        console_write("[HATA] Egitim tamamlandi ama nihai checkpoint KAYDEDILEMEDI: "); console_write_line(CKPT_FINAL_PATH);
    }

    gpu_trainer_destroy(&gt);
    allocator_destroy(&persist);
    return final_saved ? 0 : 1;
}
