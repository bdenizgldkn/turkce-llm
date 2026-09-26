/* Katman 21 - GPU'DA TUTULAN EGITIM KOSUSU.
 *
 * training/train_lm.c ile AYNI model, AYNI hiperparametreler, AYNI veri
 * ornekleme (adim s: data_parallel_step'teki worker formuluyle ayni
 * BATCH_SEQS pencere, tohum 2026+s), AYNI checkpoint dosyalari/formati ve
 * AYNI "checkpoint'ten devam" mantigi -- tek fark, adimin kendisi CPU
 * thread'lerinde otograd yerine model/gpu_train.c ile TAMAMEN GPU'da
 * hesaplanir (bkz. PROJE_PLANI.md Bolum 21). Iki program birbirinin
 * checkpoint'inden devam edebilir.
 *
 * CPU'daki LMModel/AdamOptimizer bir "ayna"dir: baslangicta (ve devamda
 * checkpoint'ten) GPU'ya yuklenir, her checkpoint'ten once GPU'dan geri
 * indirilir. */
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
/* Faz 3b (PROJE_PLANI.md Bolum 23): adim 26.000'den itibaren baglam
 * 128 -> 1024. Adim basina token sayisi AYNI tutuldu (112x128 = 14x1024
 * = 14.336) -> lr takvimi ve toplam token butcesi degismez. Agirliklar
 * baglam uzunlugundan bagimsiz (RoPE her pozisyonda calisir), bu yuzden
 * 128 baglamli checkpoint'ten dogrudan devam edilir. */
#define SEQ_LEN      1024u
#define EPS          1e-5f

/* Faz 3 (bkz. PROJE_PLANI.md Bolum 22): isinma + kosinus lr, gradyan
 * kirpma, %1 dogrulama seti, 112 dizi/adim, ~1 milyar token (~2 epoch). */
#define PEAK_LR      6e-4f
#define MIN_LR       6e-5f
#define WARMUP_STEPS 1000u
#define GRAD_CLIP    1.0
#define BETA1        0.9f
#define BETA2        0.999f
#define ADAM_EPS     1e-8f

#define NUM_STEPS       70000u
#define LOG_EVERY       100u
#define CKPT_EVERY      1000u
#define EVAL_EVERY      500u
#define VAL_PERMILLE    10u   /* token dizisinin SON %1'i dogrulama: egitimde hic gorulmez */
#define VAL_BATCHES     8u    /* dogrulama: VAL_BATCHES x BATCH_SEQS sabit pencere */
#define CKPT_LATEST_PATH "checkpoints/lm_wiki_faz3_latest.bin"
#define CKPT_FINAL_PATH  "checkpoints/lm_wiki_faz3_final.bin"
#define CKPT_BEST_PATH   "checkpoints/lm_wiki_faz3_best.bin"
#define BEST_VAL_PATH    "checkpoints/lm_wiki_faz3_best_val.bin" /* en iyi dogrulama kaybi (f32), devamda korunur */
#define CKPT_PROBE_PATH  "checkpoints/.yazma_testi"

/* Adim basina dizi sayisi (Faz 2: 28 x 128; Faz 3: 112 x 128; Faz 3b: 14 x 1024). */
#define BATCH_SEQS 14u
#define MAX_PARAMS  (LM_MAX_LAYERS * 12u + 2u)


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

int main(void) {
    console_write_line("=== Turkce LLM - GPU'da Tutulan Egitim Kosusu (Katman 21) ===");
    console_write("Vocab="); console_write_u64(VOCAB_SIZE);
    console_write(" d_model="); console_write_u64(D_MODEL);
    console_write(" katman="); console_write_u64(NUM_LAYERS);
    console_write(" head="); console_write_u64(NUM_HEADS);
    console_write(" d_ff="); console_write_u64(D_FF);
    console_write(" seq_len="); console_write_u64(SEQ_LEN);
    console_write(" dizi/adim="); console_write_u64(BATCH_SEQS);
    console_write_line("");

    /* --- 0) Checkpoint klasorune yazilabildigini EGITIMDEN ONCE dogrula.
     * checkpoint_save'in hatasi ancak ilk kayitta (CKPT_EVERY adim sonra)
     * gorulurdu; klasor yoksa/izin yoksa bunu simdi, saniyeler icinde
     * ogrenmek istiyoruz. --- */
    {
        FileHandle probe = file_open_write(CKPT_PROBE_PATH);
        if (!probe.valid) {
            console_write("[HATA] Checkpoint klasorune yazilamiyor: "); console_write_line(CKPT_PROBE_PATH);
            console_write_line("       'checkpoints/' klasoru var mi ve yazilabilir mi kontrol edin.");
            return 1;
        }
        file_close(&probe);
        file_delete(CKPT_PROBE_PATH);
    }

    /* --- 1) Token korpusunu yukle --- */
    Allocator persist = allocator_create(8ull * 1024 * 1024 * 1024);

    console_write_line("Token korpusu yukleniyor: data/raw/wikipedia_tokens.bin ...");
    f64 t_load0 = timer_now_seconds();
    u64 data_size = 0;
    void* raw = file_read_entire("data/raw/wikipedia_tokens.bin", &persist, &data_size);
    if (raw == NULL_PTR || data_size == 0) {
        console_write_line("[HATA] Token dosyasi okunamadi veya bos.");
        return 1;
    }
    const u32* tokens = (const u32*)raw;
    u64 num_tokens = data_size / sizeof(u32);
    console_write("Yuklendi: "); console_write_u64(num_tokens); console_write(" token (");
    write_fixed3(timer_now_seconds() - t_load0); console_write_line(" sn).");

    if (num_tokens < (u64)SEQ_LEN + 2) {
        console_write_line("[HATA] Korpus SEQ_LEN icin cok kucuk.");
        return 1;
    }

    /* Egitim/dogrulama ayrimi: son %VAL_PERMILLE/10'luk BITISIK blok
     * dogrulama (rastgele pencereler bitisik oldugu icin rastgele bir
     * ayrim sizinti yaratirdi). Egitim pencereleri sadece [0, train_tokens). */
    u64 val_tokens = num_tokens * VAL_PERMILLE / 1000u;
    u64 train_tokens = num_tokens - val_tokens;
    console_write("Egitim token: "); console_write_u64(train_tokens);
    console_write(" | dogrulama token: "); console_write_u64(val_tokens); console_write_line("");

    /* --- 3) Modeli olustur --- */
    PCGState model_rng = pcg_seed(1337, 1);
    LMModel model = lm_init(&persist, &model_rng, VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, EPS);

    Node* params[MAX_PARAMS];
    u32 num_params = lm_collect_params(&model, params);
    u64 total_scalars = 0;
    for (u32 i = 0; i < num_params; i++) total_scalars += params[i]->value.numel;
    console_write("Parametre tensoru: "); console_write_u64(num_params);
    console_write(" | toplam skaler parametre: "); console_write_u64(total_scalars);
    console_write_line("");

    /* --- 4) Adam optimizer --- */
    AdamOptimizer opt = adam_create(&persist, params, num_params, PEAK_LR, BETA1, BETA2, ADAM_EPS);

    /* --- 4b) Varsa CKPT_LATEST_PATH'ten devam et. Adam durumu (m, v, t)
     * da geri yuklenir ve her adimin veri tohumu (2026+step) sadece adim
     * numarasina bagli oldugu icin, devam eden kosu kesintisiz kosuyla
     * AYNI adimlari ayni verilerle atar (bkz. tests/test_resume.c).
     * checkpoint'ler adam_step'ten SONRA kaydedildigi icin opt.t, bir
     * sonraki atilacak adimin numarasidir. --- */
    u32 start_step = 0;
    {
        FileHandle probe = file_open_read(CKPT_LATEST_PATH);
        bool32 exists = probe.valid;
        file_close(&probe);

        if (exists) {
            Allocator ckpt_scratch = allocator_create(2ull * 1024 * 1024 * 1024);
            bool32 loaded = checkpoint_load(&ckpt_scratch, CKPT_LATEST_PATH, params, num_params, &opt);
            allocator_destroy(&ckpt_scratch);

            /* Dosya var ama gecersizse (mimari degismis, bozuk) SIFIRDAN
             * baslamiyoruz: ilk CKPT_EVERY'de o dosyanin uzerine yazilir
             * ve (belki saatlerce egitilmis) model sessizce kaybolurdu. */
            if (!loaded) {
                console_write("[HATA] Checkpoint var ama yuklenemedi (bozuk ya da mimari uyusmuyor): ");
                console_write_line(CKPT_LATEST_PATH);
                console_write_line("       Sifirdan baslamak icin bu dosyayi silin ya da baska bir yere tasiyin.");
                return 1;
            }
            start_step = (u32)opt.t;
            console_write("Checkpoint'ten devam ediliyor: "); console_write(CKPT_LATEST_PATH);
            console_write(" (adim "); console_write_u64(start_step); console_write_line("'den)");
        } else {
            console_write_line("Checkpoint yok, sifirdan baslaniyor.");
        }
    }

    /* En iyi dogrulama kaybi (devamda korunur; yoksa +sonsuz). */
    f32 best_val = 3.0e38f;
    {
        u64 sz = 0;
        Allocator bv_scratch = allocator_create(1024 * 1024);
        f32* bv = (f32*)file_read_entire(BEST_VAL_PATH, &bv_scratch, &sz);
        if (bv != NULL_PTR && sz == sizeof(f32)) best_val = *bv;
        allocator_destroy(&bv_scratch);
    }

    /* --- 4c) GPU egiticisi: agirliklar + Adam durumu (devamda
     * checkpoint'ten yuklenmis haliyle) GPU'ya yuklenir. --- */
    GpuTrainer gt = gpu_trainer_create(&persist, "cuda/train_kernels.ptx", &model, params, num_params,
                                       BATCH_SEQS, SEQ_LEN);
    gpu_trainer_upload(&gt, params, &opt);
    u32* batch_ids = (u32*)allocator_alloc(&persist, (u64)BATCH_SEQS * SEQ_LEN * sizeof(u32));
    u32* batch_tgt = (u32*)allocator_alloc(&persist, (u64)BATCH_SEQS * SEQ_LEN * sizeof(u32));
    console_write_line("GPU egiticisi hazir (agirliklar, gradyanlar, Adam durumu ve aktivasyonlar GPU'da).");

    /* Dogrulama pencereleri: dogrulama blogunda esit aralikli, SABIT
     * (her olcum ayni pencerelerde -> karsilastirilabilir). */
    u64 val_windows = (u64)VAL_BATCHES * BATCH_SEQS;
    u64 val_stride = (val_tokens - SEQ_LEN - 1) / val_windows;

    /* --- 5) Egitim dongusu (GPU) --- */
    f64 loss_ema = -1.0;
    f64 t_train0 = timer_now_seconds();
    f64 last_log_elapsed = 0.0;

    f64 t_fb_acc = 0.0, t_adam_acc = 0.0;

    for (u32 step = start_step; step < NUM_STEPS; step++) {
        f64 ta = timer_now_seconds();
        opt.lr = lr_warmup_cosine(step, WARMUP_STEPS, NUM_STEPS, PEAK_LR, MIN_LR);
        gpu_trainer_sample_batch(tokens, train_tokens, SEQ_LEN, BATCH_SEQS, 2026ull + step, batch_ids, batch_tgt);
        f32 loss_val = gpu_trainer_forward_backward(&gt, batch_ids, batch_tgt);
        f64 tb = timer_now_seconds(); t_fb_acc += (tb - ta);

        /* Gradyan kirpma: global L2 normu GRAD_CLIP'i asarsa olcekle. */
        f64 grad_norm = gpu_trainer_grad_norm(&gt);
        f32 clip_scale = (grad_norm > GRAD_CLIP) ? (f32)(GRAD_CLIP / grad_norm) : 1.0f;
        gpu_trainer_adam_step_scaled(&gt, &opt, clip_scale);
        f64 tc = timer_now_seconds(); t_adam_acc += (tc - tb);

        loss_ema = (loss_ema < 0.0) ? (f64)loss_val : (0.98 * loss_ema + 0.02 * (f64)loss_val);

        if (step % LOG_EVERY == 0 || step == NUM_STEPS - 1) {
            f64 elapsed = timer_now_seconds() - t_train0;
            f64 interval_elapsed = elapsed - last_log_elapsed;
            u32 steps_in_interval = (step == start_step) ? 1u : ((step % LOG_EVERY == 0) ? LOG_EVERY : (step % LOG_EVERY));
            console_write("adim "); console_write_u64(step);
            console_write(" | kayip="); write_fixed3((f64)loss_val);
            console_write(" | ema="); write_fixed3(loss_ema);
            console_write(" | gecen="); write_fixed3(elapsed); console_write(" sn");
            console_write(" | sn/adim="); write_fixed3(interval_elapsed / (f64)steps_in_interval);
            console_write(" | lr*1e4="); write_fixed3((f64)opt.lr * 1e4);
            console_write(" | gnorm="); write_fixed3(grad_norm);
            console_write_line("");
            console_write("  [dokum] ileri+geri="); write_fixed3(t_fb_acc);
            console_write("sn adam="); write_fixed3(t_adam_acc);
            console_write("sn");
            console_write_line("");
            last_log_elapsed = elapsed;
            t_fb_acc = 0.0; t_adam_acc = 0.0;
        }

        if ((step + 1) % EVAL_EVERY == 0 || step == NUM_STEPS - 1) {
            f64 te = timer_now_seconds();
            f32 val_sum = 0.0f;
            for (u32 vb = 0; vb < VAL_BATCHES; vb++) {
                for (u32 w = 0; w < BATCH_SEQS; w++) {
                    u64 start = train_tokens + ((u64)vb * BATCH_SEQS + w) * val_stride;
                    for (u32 t = 0; t < SEQ_LEN; t++) {
                        batch_ids[(u64)w * SEQ_LEN + t] = tokens[start + t];
                        batch_tgt[(u64)w * SEQ_LEN + t] = tokens[start + t + 1];
                    }
                }
                val_sum += gpu_trainer_eval_loss(&gt, batch_ids, batch_tgt);
            }
            f32 val_loss = val_sum / (f32)VAL_BATCHES;
            console_write("  [dogrulama] adim "); console_write_u64(step + 1);
            console_write(" | val_kayip="); write_fixed3((f64)val_loss);
            console_write(" | en_iyi="); write_fixed3(best_val < 1e38f ? (f64)best_val : 0.0);
            console_write(" | sure="); write_fixed3(timer_now_seconds() - te); console_write_line(" sn");
            t_train0 += timer_now_seconds() - te; /* dogrulama suresi sn/adim olcumune girmesin */

            if (val_loss < best_val) {
                gpu_trainer_download(&gt, params, &opt);
                if (checkpoint_save(CKPT_BEST_PATH, params, num_params, &opt)) {
                    best_val = val_loss;
                    FileHandle bf = file_open_write(BEST_VAL_PATH);
                    if (bf.valid) { file_write(&bf, &best_val, sizeof(f32)); file_close(&bf); }
                    console_write_line("  [en iyi model kaydedildi]");
                } else {
                    console_write("  [HATA] en iyi model KAYDEDILEMEDI: "); console_write_line(CKPT_BEST_PATH);
                }
            }
        }

        if (step > 0 && (step % CKPT_EVERY == 0)) {
            /* Basarisiz kayitta egitimi DURDURMUYORUZ (bellekteki model
             * kaybolurdu); hatayi yuksek sesle bildirip bir sonraki
             * CKPT_EVERY'de tekrar deniyoruz. */
            gpu_trainer_download(&gt, params, &opt);
            if (checkpoint_save(CKPT_LATEST_PATH, params, num_params, &opt)) {
                console_write("  [checkpoint kaydedildi: adim "); console_write_u64(step); console_write_line("]");
            } else {
                console_write("  [HATA] checkpoint KAYDEDILEMEDI (adim "); console_write_u64(step);
                console_write("): "); console_write_line(CKPT_LATEST_PATH);
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
