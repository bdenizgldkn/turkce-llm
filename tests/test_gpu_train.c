/* Katman 21 dogrulama: GPU'da tutulan egitim (model/gpu_train.c) ile
 * mevcut uretim yolu (training/data_parallel.c: B thread, CPU otograd)
 * AYNI veriyle AYNI kaybi, AYNI gradyanlari ve Adam sonrasi AYNI
 * agirliklari uretmeli (kayan nokta toplama sirasi farkli oldugu icin
 * toleransla).
 *
 * Vaka 1 (kucuk): boyutlar bilerek GEMM karo katlarinin (64/16) DISINDA
 *   secildi -> sinir kontrolleri de sinanir. CPU referansi saf CPU.
 * Vaka 2 (gercek olcek): train_lm'in mimarisi (V=31769, D=384, H=6,
 *   L=12, F=1024, T=128), B=4. CPU referansi GPU-matmul'lu uretim yolu.
 *
 * Ayrica: ayni girdiyle iki kez calistirmanin BIT BIT ayni gradyan
 * urettigi (determinizm) ve upload/download'in kayipsiz oldugu sinanir. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/prng.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/lm_model.h"
#include "../model/gpu_ops.h"
#include "../model/gpu_train.h"
#include "../training/adam.h"
#include "../training/data_parallel.h"
#include "../runtime/mathlib.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; console_write("  [FAIL] "); console_write_line(msg); } \
} while (0)

static f32 absf(f32 v) { return v < 0 ? -v : v; }

static void write_e6(const char* label, f64 v) {
    console_write(label);
    if (v < 0) { console_write("-"); v = -v; }
    console_write_u64((u64)(v * 1e6 + 0.5));
    console_write("e-6 ");
}

typedef struct CaseCfg {
    const char* name;
    u32 V, D, H, L, F, T, B;
    u64 num_tokens;
    i32 cpu_use_gpu;
} CaseCfg;

/* got ile want'i, parametre tensoru basina en buyuk |want|'a gore olcekli
 * toleransla karsilastirir; basarisiz eleman sayisini dondurur. */
static u64 compare_tensor(const Tensor* got, const Tensor* want, f32 rel_tol, f32* worst_rel) {
    f32 maxabs = 0.0f;
    for (u64 k = 0; k < want->numel; k++) if (absf(want->data[k]) > maxabs) maxabs = absf(want->data[k]);
    f32 floor = maxabs * rel_tol + 1e-12f;
    u64 bad = 0;
    for (u64 k = 0; k < want->numel; k++) {
        f32 diff = absf(got->data[k] - want->data[k]);
        f32 tol = floor + rel_tol * absf(want->data[k]);
        if (diff > tol) bad++;
        f32 r = diff / (maxabs + 1e-30f);
        if (r > *worst_rel) *worst_rel = r;
    }
    return bad;
}

static void run_case(const CaseCfg* c) {
    console_write("--- "); console_write_line(c->name);

    Allocator alloc = allocator_create(8ull * 1024 * 1024 * 1024);

    u32* tokens = (u32*)allocator_alloc(&alloc, c->num_tokens * sizeof(u32));
    PCGState trng = pcg_seed(4242, 7);
    for (u64 i = 0; i < c->num_tokens; i++) tokens[i] = (u32)pcg_range_i64(&trng, 0, (i64)c->V - 1);

    if (c->cpu_use_gpu) {
        gpu_ops_init(&alloc, "cuda/kernels.ptx");
        gpu_ops_init_workers(&alloc, "cuda/kernels.ptx", c->B);
    }

    PCGState rng = pcg_seed(1337, 1);
    LMModel model = lm_init(&alloc, &rng, c->V, c->D, c->H, c->L, c->F, 1e-5f);
    Node* params[LM_MAX_LAYERS * 12 + 2];
    u32 np = lm_collect_params(&model, params);
    AdamOptimizer opt = adam_create(&alloc, params, np, 3e-4f, 0.9f, 0.999f, 1e-8f);

    /* GPU egiticisi (parametreleri CPU modelinden yukler) -- CPU adimindan
     * ONCE kurulur ki ayni baslangic agirliklarini gorsun. */
    GpuTrainer g = gpu_trainer_create(&alloc, "cuda/train_kernels.ptx", &model, params, np, c->B, c->T);
    AdamOptimizer gopt = adam_create(&alloc, params, np, 3e-4f, 0.9f, 0.999f, 1e-8f);

    const u64 seed = 2026ull;

    /* --- CPU referansi (uretim yolu) --- */
    adam_zero_grad(&opt);
    f32 cpu_loss = data_parallel_step(&model, params, np, tokens, c->num_tokens, c->T,
                                      1536ull * 1024 * 1024, c->B, seed, c->cpu_use_gpu, c->cpu_use_gpu);
    Tensor ref_grad[LM_MAX_LAYERS * 12 + 2];
    for (u32 i = 0; i < np; i++) {
        ref_grad[i] = tensor_create(&alloc, params[i]->value.shape, params[i]->value.ndim);
        for (u64 k = 0; k < params[i]->grad.numel; k++) ref_grad[i].data[k] = params[i]->grad.data[k];
    }

    /* --- GPU --- */
    u32* ids = (u32*)allocator_alloc(&alloc, (u64)c->B * c->T * sizeof(u32));
    u32* tgt = (u32*)allocator_alloc(&alloc, (u64)c->B * c->T * sizeof(u32));
    gpu_trainer_sample_batch(tokens, c->num_tokens, c->T, c->B, seed, ids, tgt);
    f32 gpu_loss = gpu_trainer_forward_backward(&g, ids, tgt);

    console_write("  kayip CPU="); console_write_u64((u64)(cpu_loss * 1e6f));
    console_write(" GPU="); console_write_u64((u64)(gpu_loss * 1e6f)); console_write_line(" (x1e6)");
    CHECK(absf(cpu_loss - gpu_loss) <= 1e-4f * absf(cpu_loss), "gpu_train: kayip CPU referansiyla eslesmiyor");

    /* Gradyanlar */
    gpu_trainer_download_grads(&g, params);
    f32 worst = 0.0f;
    u64 bad_total = 0;
    for (u32 i = 0; i < np; i++) {
        u64 bad = compare_tensor(&params[i]->grad, &ref_grad[i], 2e-3f, &worst);
        if (bad) {
            console_write("  [FAIL] parametre "); console_write_u64(i);
            console_write(": "); console_write_u64(bad); console_write(" / ");
            console_write_u64(ref_grad[i].numel); console_write_line(" eleman tolerans disi");
        }
        bad_total += bad;
        CHECK(bad == 0, "gpu_train: gradyan CPU referansiyla eslesmiyor");
    }
    write_e6("  en kotu goreli gradyan farki: ", (f64)worst); console_write_line("");

    /* Determinizm: ayni girdiyle ikinci kez -> BIT BIT ayni gradyan */
    Tensor g1[LM_MAX_LAYERS * 12 + 2];
    for (u32 i = 0; i < np; i++) {
        g1[i] = tensor_create(&alloc, params[i]->value.shape, params[i]->value.ndim);
        for (u64 k = 0; k < params[i]->grad.numel; k++) g1[i].data[k] = params[i]->grad.data[k];
    }
    f32 gpu_loss2 = gpu_trainer_forward_backward(&g, ids, tgt);
    gpu_trainer_download_grads(&g, params);
    union { f32 f; u32 u; } la, lb; la.f = gpu_loss; lb.f = gpu_loss2;
    bool32 same = (la.u == lb.u);
    for (u32 i = 0; i < np && same; i++) {
        const u32* a = (const u32*)g1[i].data;
        const u32* b = (const u32*)params[i]->grad.data;
        for (u64 k = 0; k < g1[i].numel; k++) if (a[k] != b[k]) { same = FALSE; break; }
    }
    CHECK(same, "gpu_train: ayni girdiyle iki kosu BIT BIT ayni degil (determinizm)");

    /* Dogrulama kaybi (sadece ileri) == egitim adimindaki kayip, BIT BIT */
    {
        f32 ev = gpu_trainer_eval_loss(&g, ids, tgt);
        union { f32 f; u32 u; } ea, eb; ea.f = ev; eb.f = gpu_loss2;
        CHECK(ea.u == eb.u, "gpu_train: eval_loss ileri-geri kaybiyla bit bit ayni degil");
    }

    /* Gradyan normu: GPU (deterministik blok indirgemesi) vs CPU (f64, g1 = GPU gradyanlari) */
    {
        f64 ss = 0.0;
        for (u32 i = 0; i < np; i++) for (u64 k = 0; k < g1[i].numel; k++) ss += (f64)g1[i].data[k] * (f64)g1[i].data[k];
        f64 cpu_norm = m_sqrt(ss);
        f64 gpu_norm = gpu_trainer_grad_norm(&g);
        f64 rel = (gpu_norm - cpu_norm) / cpu_norm; if (rel < 0) rel = -rel;
        write_e6("  gradyan normu goreli farki: ", rel); console_write_line("");
        CHECK(rel < 1e-5, "gpu_train: gradyan normu CPU ile eslesmiyor");
        f64 gpu_norm2 = gpu_trainer_grad_norm(&g);
        CHECK(gpu_norm == gpu_norm2, "gpu_train: gradyan normu deterministik degil");
    }

    /* Adam: CPU referans gradyanlariyla CPU Adam vs GPU gradyanlariyla GPU Adam */
    for (u32 i = 0; i < np; i++)
        for (u64 k = 0; k < ref_grad[i].numel; k++) params[i]->grad.data[k] = ref_grad[i].data[k];
    adam_step(&opt);
    Tensor ref_val[LM_MAX_LAYERS * 12 + 2];
    for (u32 i = 0; i < np; i++) {
        ref_val[i] = tensor_create(&alloc, params[i]->value.shape, params[i]->value.ndim);
        for (u64 k = 0; k < params[i]->value.numel; k++) ref_val[i].data[k] = params[i]->value.data[k];
    }
    /* Adam cekirdegini AYNI gradyanlarla sina: Adam'in ilk adimi
     * ~lr*isaret(g) oldugu icin, sifira cok yakin gradyanlarda ileri/geri
     * yayilimdaki kucuk kayan nokta farki bile isareti cevirip tam lr
     * kadar fark yaratirdi -- bu, Adam'in degil gradyanin farki olurdu. */
    for (u32 i = 0; i < np; i++)
        for (u64 k = 0; k < ref_grad[i].numel; k++) params[i]->grad.data[k] = ref_grad[i].data[k];
    gpu_trainer_upload_grads(&g, params);
    gpu_trainer_adam_step(&g, &gopt);
    gpu_trainer_download(&g, params, &gopt);
    u64 adam_bad = 0;
    for (u32 i = 0; i < np; i++) {
        for (u64 k = 0; k < ref_val[i].numel; k++) {
            f32 d = absf(params[i]->value.data[k] - ref_val[i].data[k]);
            if (d > 1e-3f * 3e-4f + 1e-6f * absf(ref_val[i].data[k])) adam_bad++;
        }
    }
    console_write("  Adam sonrasi tolerans disi agirlik: "); console_write_u64(adam_bad); console_write_line("");
    CHECK(adam_bad == 0, "gpu_train: Adam sonrasi agirliklar CPU'yla eslesmiyor");
    CHECK(gopt.t == 1, "gpu_train: Adam adim sayaci artmadi");

    /* upload/download kayipsiz: indirilen durumu tekrar yukle, tekrar indir */
    gpu_trainer_upload(&g, params, &gopt);
    Tensor snap = tensor_create(&alloc, params[0]->value.shape, params[0]->value.ndim);
    for (u64 k = 0; k < snap.numel; k++) snap.data[k] = params[0]->value.data[k];
    gpu_trainer_download(&g, params, &gopt);
    bool32 rt = TRUE;
    for (u64 k = 0; k < snap.numel; k++) if (snap.data[k] != params[0]->value.data[k]) { rt = FALSE; break; }
    CHECK(rt, "gpu_train: upload/download gidis-donusu kayipli");

    /* Olcekli Adam (kirpma): adam_step_scaled(0.5) ile G == adam_step ile 0.5*G,
     * ayni baslangic durumundan BIT BIT (0.5 ile carpim tam). */
    {
        Tensor sv[LM_MAX_LAYERS * 12 + 2], sm[LM_MAX_LAYERS * 12 + 2], svv[LM_MAX_LAYERS * 12 + 2], resA[LM_MAX_LAYERS * 12 + 2];
        for (u32 i = 0; i < np; i++) {
            sv[i] = tensor_create(&alloc, params[i]->value.shape, params[i]->value.ndim);
            sm[i] = tensor_create(&alloc, params[i]->value.shape, params[i]->value.ndim);
            svv[i] = tensor_create(&alloc, params[i]->value.shape, params[i]->value.ndim);
            resA[i] = tensor_create(&alloc, params[i]->value.shape, params[i]->value.ndim);
            for (u64 k = 0; k < sv[i].numel; k++) { sv[i].data[k] = params[i]->value.data[k]; sm[i].data[k] = gopt.m[i].data[k]; svv[i].data[k] = gopt.v[i].data[k]; }
        }
        u64 t0 = gopt.t;

        /* A: gradyan = 0.5*G, olceksiz Adam */
        for (u32 i = 0; i < np; i++) for (u64 k = 0; k < ref_grad[i].numel; k++) params[i]->grad.data[k] = 0.5f * ref_grad[i].data[k];
        gpu_trainer_upload_grads(&g, params);
        gpu_trainer_adam_step(&g, &gopt);
        gpu_trainer_download(&g, params, &gopt);
        for (u32 i = 0; i < np; i++) for (u64 k = 0; k < resA[i].numel; k++) resA[i].data[k] = params[i]->value.data[k];

        /* B: ayni baslangic durumu, gradyan = G, olcek 0.5 */
        for (u32 i = 0; i < np; i++) for (u64 k = 0; k < sv[i].numel; k++) { params[i]->value.data[k] = sv[i].data[k]; gopt.m[i].data[k] = sm[i].data[k]; gopt.v[i].data[k] = svv[i].data[k]; }
        gopt.t = t0;
        gpu_trainer_upload(&g, params, &gopt);
        for (u32 i = 0; i < np; i++) for (u64 k = 0; k < ref_grad[i].numel; k++) params[i]->grad.data[k] = ref_grad[i].data[k];
        gpu_trainer_upload_grads(&g, params);
        gpu_trainer_adam_step_scaled(&g, &gopt, 0.5f);
        gpu_trainer_download(&g, params, &gopt);

        bool32 eq = TRUE;
        for (u32 i = 0; i < np && eq; i++) {
            const u32* a = (const u32*)resA[i].data;
            const u32* b = (const u32*)params[i]->value.data;
            for (u64 k = 0; k < resA[i].numel; k++) if (a[k] != b[k]) { eq = FALSE; break; }
        }
        CHECK(eq, "gpu_train: adam_step_scaled(0.5, G) != adam_step(0.5*G)");
    }

    gpu_trainer_destroy(&g);
    if (c->cpu_use_gpu) gpu_ops_shutdown();
    allocator_destroy(&alloc);
    (void)bad_total;
}

int main(void) {
    console_write_line("=== Katman 21: GPU'da tutulan egitim vs CPU uretim yolu ===");

    CaseCfg small = { "Vaka 1: kucuk, karo-disi boyutlar (V=50 D=32 H=2 L=2 F=48 T=12 B=5)",
                      50, 32, 2, 2, 48, 12, 5, 4000, 0 };
    run_case(&small);

    CaseCfg real = { "Vaka 2: gercek olcek (V=31769 D=384 H=6 L=12 F=1024 T=128 B=4)",
                     31769, 384, 6, 12, 1024, 128, 4, 200000, 1 };
    run_case(&real);

    console_write("Sonuc: "); console_write_u64(g_pass); console_write(" basarili, ");
    console_write_u64(g_fail); console_write_line(" basarisiz.");
    return g_fail == 0 ? 0 : 1;
}
