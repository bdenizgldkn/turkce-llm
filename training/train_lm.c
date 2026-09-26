/* Katman 9 - GERCEK EGITIM KOSUSU.
 *
 * Gercek Turkce Wikipedia korpusunden tokenlestirilmis 494.6M token'lik
 * veriyle (data/raw/wikipedia_tokens.bin) bir dil modeli egitir.
 *
 * FAZ 2 olcegi (bkz. PROJE_PLANI.md Bolum 4): d_model=384, 12 katman,
 * 6 baslik, d_ff=1024 -> ~33.5M parametre (hedeflenen 30-150M araliginda).
 *
 * KATMAN 16 - VERI-PARALEL COK-THREAD'LI EGITIM (bkz. training/data_parallel.c):
 * Kullanicinin gozlemi ("bilgisayarim hic yorulmuyor") dogruydu -- tek
 * thread'li kod, 8 cekirdek/12 mantiksal islemcili bir CPU'nun sadece
 * 1/12'sini kullaniyordu. Cozum: her adimda NUM_WORKERS bagimsiz dizi,
 * NUM_WORKERS OS thread'ine paralel dagitilip ileri+geri yayilim yapiyor,
 * gradyanlar toplanip TEK bir Adam adimi atiliyor -- veri-paralel
 * egitim (data parallelism, bkz. tests/test_data_parallel.c). Adam
 * adiminin kendisi de (parametreler birbirinden bagimsiz) ayni thread
 * havuzuna paralel dagitilir (bkz. training/adam.c: adam_step_parallel). */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/file_io.h"
#include "../runtime/timer.h"
#include "../model/lm_model.h"
#include "../model/gpu_ops.h"
#include "../training/adam.h"
#include "../training/checkpoint.h"
#include "../training/data_parallel.h"

#define VOCAB_SIZE   31769u
#define D_MODEL      384u
#define NUM_HEADS    6u
#define NUM_LAYERS   12u
#define D_FF         1024u
#define SEQ_LEN      128u
#define EPS          1e-5f

#define LR           3e-4f
#define BETA1        0.9f
#define BETA2        0.999f
#define ADAM_EPS     1e-8f

#define NUM_STEPS       3000u
#define LOG_EVERY       20u
#define CKPT_EVERY      250u
#define STEP_ARENA_SIZE (1536ull * 1024 * 1024)
#define CKPT_LATEST_PATH "checkpoints/lm_wiki_faz2_latest.bin"
#define CKPT_FINAL_PATH  "checkpoints/lm_wiki_faz2_final.bin"
#define CKPT_PROBE_PATH  "checkpoints/.yazma_testi"

#define NUM_WORKERS 28u
#define MAX_PARAMS  (LM_MAX_LAYERS * 12u + 2u)

/* DENEY SONUCU (bkz. PROJE_PLANI.md Bolum 17): katman-ici carpimlari
 * CPU'ya (8 thread paralel) tasimak, GPU baglam cekismesinden (contention)
 * KURTULMAK yerine adim suresini ~2.4x KOTULESTIRDI (13.8sn vs 5.78sn,
 * adim 0) -- GPU'nun ham hesaplama hizi, paylasilan baglam kilidinin
 * getirdigi bekleme maliyetinden HALA agir basiyor. Bu yuzden HER IKI
 * bayrak da acik birakiliyor (mevcut en hizli olculen yapilandirma). */
#define USE_GPU_LAYERS 1
#define USE_GPU_OUTPUT 1

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
    console_write_line("=== Turkce LLM - Gercek Egitim Kosusu (Katman 9/16) ===");
    console_write("Vocab="); console_write_u64(VOCAB_SIZE);
    console_write(" d_model="); console_write_u64(D_MODEL);
    console_write(" katman="); console_write_u64(NUM_LAYERS);
    console_write(" head="); console_write_u64(NUM_HEADS);
    console_write(" d_ff="); console_write_u64(D_FF);
    console_write(" seq_len="); console_write_u64(SEQ_LEN);
    console_write(" worker="); console_write_u64(NUM_WORKERS);
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

    /* --- 2) GPU baglamlarini baslat: 1 varsayilan (paylasilan) + her
     * worker icin kendi BAGIMSIZ baglami (bkz. PROJE_PLANI.md Bolum 17,
     * Aday 1 -- mutex cekismesini kaldirmak icin). --- */
    gpu_ops_init(&persist, "cuda/kernels.ptx");
    gpu_ops_init_workers(&persist, "cuda/kernels.ptx", NUM_WORKERS);
    console_write_line("GPU baglamlari hazir (1 paylasilan + worker basina bagimsiz baglam).");

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
    AdamOptimizer opt = adam_create(&persist, params, num_params, LR, BETA1, BETA2, ADAM_EPS);

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

    /* --- 5) Egitim dongusu (veri-paralel, NUM_WORKERS thread) --- */
    f64 loss_ema = -1.0;
    f64 t_train0 = timer_now_seconds();
    f64 last_log_elapsed = 0.0;

    f64 t_parallel_acc = 0.0, t_adam_acc = 0.0;

    for (u32 step = start_step; step < NUM_STEPS; step++) {
        adam_zero_grad(&opt);

        f64 ta = timer_now_seconds();
        f32 loss_val = data_parallel_step(&model, params, num_params, tokens, num_tokens, SEQ_LEN,
                                           STEP_ARENA_SIZE, NUM_WORKERS, 2026ull + step,
                                           USE_GPU_LAYERS, USE_GPU_OUTPUT);
        f64 tb = timer_now_seconds(); t_parallel_acc += (tb - ta);

        adam_step_parallel(&opt, NUM_WORKERS);
        f64 tc = timer_now_seconds(); t_adam_acc += (tc - tb);

        loss_ema = (loss_ema < 0.0) ? (f64)loss_val : (0.98 * loss_ema + 0.02 * (f64)loss_val);

        if (step % LOG_EVERY == 0 || step == NUM_STEPS - 1) {
            f64 elapsed = timer_now_seconds() - t_train0;
            f64 interval_elapsed = elapsed - last_log_elapsed;
            f64 gpu_seconds; u32 gpu_calls;
            gpu_ops_debug_stats(&gpu_seconds, &gpu_calls);
            f64 gpu_pct = (interval_elapsed > 0.0) ? (100.0 * gpu_seconds / interval_elapsed) : 0.0;
            console_write("adim "); console_write_u64(step);
            console_write(" | kayip="); write_fixed3((f64)loss_val);
            console_write(" | ema="); write_fixed3(loss_ema);
            console_write(" | gecen="); write_fixed3(elapsed); console_write(" sn");
            console_write(" | [tanilama] gpu="); write_fixed3(gpu_seconds); console_write("sn/");
            console_write_u64(gpu_calls); console_write("cagri ("); write_fixed3(gpu_pct); console_write("% araliktan)");
            console_write_line("");
            console_write("  [dokum] paralel(ileri+geri+indirgeme)="); write_fixed3(t_parallel_acc);
            console_write("sn adam="); write_fixed3(t_adam_acc);
            console_write("sn");
            console_write_line("");
            gpu_ops_debug_reset();
            last_log_elapsed = elapsed;
            t_parallel_acc = 0.0; t_adam_acc = 0.0;
        }

        if (step > 0 && (step % CKPT_EVERY == 0)) {
            /* Basarisiz kayitta egitimi DURDURMUYORUZ (bellekteki model
             * kaybolurdu); hatayi yuksek sesle bildirip bir sonraki
             * CKPT_EVERY'de tekrar deniyoruz. */
            if (checkpoint_save(CKPT_LATEST_PATH, params, num_params, &opt)) {
                console_write("  [checkpoint kaydedildi: adim "); console_write_u64(step); console_write_line("]");
            } else {
                console_write("  [HATA] checkpoint KAYDEDILEMEDI (adim "); console_write_u64(step);
                console_write("): "); console_write_line(CKPT_LATEST_PATH);
            }
        }
    }

    bool32 final_saved = checkpoint_save(CKPT_FINAL_PATH, params, num_params, &opt);
    if (final_saved) {
        console_write("Egitim tamamlandi. Nihai checkpoint: "); console_write_line(CKPT_FINAL_PATH);
    } else {
        console_write("[HATA] Egitim tamamlandi ama nihai checkpoint KAYDEDILEMEDI: "); console_write_line(CKPT_FINAL_PATH);
    }

    gpu_ops_shutdown();
    allocator_destroy(&persist);
    return final_saved ? 0 : 1;
}
