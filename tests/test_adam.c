/* Katman 8 dogrulama:
 * 1) Adam optimizer'in kucuk bir modeli GERCEKTEN OGRENDIGINI (kayip
 *    monoton azaliyor mu) dogrular -- fonksiyonel/yakinsama testi.
 * 2) Checkpoint kaydetme/yukleme sisteminin parametre VE optimizer
 *    durumunu (m, v, t) tam olarak geri yukledigini dogrular. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/prng.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../model/model_ops.h"
#include "../training/adam.h"
#include "../training/checkpoint.h"
#include "../runtime/file_io.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; console_write("  [FAIL] "); console_write_line(msg); } \
} while (0)

static f32 fabsf_(f32 v) { return (v < 0) ? -v : v; }

/* ---- Kucuk bir 2 katmanli model: y = W2*relu(W1*x+b1)+b2, MSE kaybi ---- */

typedef struct TinyModel {
    Node* w1; Node* b1;
    Node* w2; Node* b2;
} TinyModel;

static Node* forward_loss_node(Allocator* alloc, TinyModel* m, Node* x, Node* target) {
    Node* h = node_relu(alloc, node_add_bias(alloc, node_matmul2d(alloc, x, m->w1), m->b1));
    Node* y = node_add_bias(alloc, node_matmul2d(alloc, h, m->w2), m->b2);
    Node* diff = node_sub(alloc, y, target);
    Node* sq = node_mul(alloc, diff, diff);
    return node_sum_all(alloc, sq);
}

static void test_adam_convergence(void) {
    Allocator alloc = allocator_create(200ull * 1024 * 1024);
    PCGState rng = pcg_seed(99, 1);

    u64 xs[2] = { 1, 4 }, w1s[2] = { 4, 8 }, b1s[1] = { 8 }, w2s[2] = { 8, 4 }, b2s[1] = { 4 };

    Tensor xt = tensor_create(&alloc, xs, 2);
    for (u64 i = 0; i < 4; i++) xt.data[i] = (f32)pcg_gaussian(&rng, 0.0, 1.0);
    Node* x = node_leaf(&alloc, xt, FALSE);

    Tensor tt = tensor_create(&alloc, xs, 2);
    for (u64 i = 0; i < 4; i++) tt.data[i] = (f32)pcg_gaussian(&rng, 0.0, 1.0);
    Node* target = node_leaf(&alloc, tt, FALSE);

    TinyModel m;
    Tensor w1t = tensor_create(&alloc, w1s, 2); for (u64 i=0;i<32;i++) w1t.data[i]=(f32)pcg_gaussian(&rng,0.0,0.3);
    Tensor b1t = tensor_create(&alloc, b1s, 1); for (u64 i=0;i<8;i++) b1t.data[i]=0.0f;
    Tensor w2t = tensor_create(&alloc, w2s, 2); for (u64 i=0;i<32;i++) w2t.data[i]=(f32)pcg_gaussian(&rng,0.0,0.3);
    Tensor b2t = tensor_create(&alloc, b2s, 1); for (u64 i=0;i<4;i++) b2t.data[i]=0.0f;
    m.w1 = node_leaf(&alloc, w1t, TRUE);
    m.b1 = node_leaf(&alloc, b1t, TRUE);
    m.w2 = node_leaf(&alloc, w2t, TRUE);
    m.b2 = node_leaf(&alloc, b2t, TRUE);

    Node* params[4] = { m.w1, m.b1, m.w2, m.b2 };
    AdamOptimizer opt = adam_create(&alloc, params, 4, 0.05f, 0.9f, 0.999f, 1e-8f);

    f32 first_loss = 0.0f, last_loss = 0.0f;
    bool32 monotonic_ish = TRUE;
    f32 prev_loss = 1e30f;

    for (i32 step = 0; step < 200; step++) {
        adam_zero_grad(&opt);
        Node* loss = forward_loss_node(&alloc, &m, x, target);
        backward(&alloc, loss);
        f32 lval = loss->value.data[0];
        if (step == 0) first_loss = lval;
        last_loss = lval;
        /* Adam gurultulu adim atabilir ama genel egilim asagi olmali;
         * her 20 adimda bir onceki kontrol noktasindan kucuk olmali. */
        if (step % 20 == 19) {
            if (lval > prev_loss) monotonic_ish = FALSE;
            prev_loss = lval;
        }
        adam_step(&opt);
    }

    console_write("  ilk kayip="); console_write_u64((u64)(first_loss * 1000.0f));
    console_write(" son kayip="); console_write_u64((u64)(last_loss * 1000.0f));
    console_write_line(" (x1000)");

    CHECK(last_loss < first_loss * 0.1f, "adam: kayip yeterince azalmadi (yakinsama basarisiz)");
    CHECK(monotonic_ish, "adam: 20-adimlik kontrol noktalarinda kayip genel olarak azalmadi");

    allocator_destroy(&alloc);
}

/* ---- Checkpoint kaydet/yukle testi ---- */

static void test_checkpoint_roundtrip(void) {
    Allocator alloc = allocator_create(50ull * 1024 * 1024);
    PCGState rng = pcg_seed(55, 2);

    u64 w1s[2] = { 4, 8 }, b1s[1] = { 8 };
    Tensor w1t = tensor_create(&alloc, w1s, 2); for (u64 i=0;i<32;i++) w1t.data[i]=(f32)pcg_gaussian(&rng,0.0,1.0);
    Tensor b1t = tensor_create(&alloc, b1s, 1); for (u64 i=0;i<8;i++) b1t.data[i]=(f32)pcg_gaussian(&rng,0.0,1.0);
    Node* w1 = node_leaf(&alloc, w1t, TRUE);
    Node* b1 = node_leaf(&alloc, b1t, TRUE);
    Node* params[2] = { w1, b1 };

    AdamOptimizer opt = adam_create(&alloc, params, 2, 0.01f, 0.9f, 0.999f, 1e-8f);
    /* optimizer durumuna rastgele "gecmis" degerler koy (gercek egitimi taklit et) */
    for (u64 i=0;i<32;i++) opt.m[0].data[i] = (f32)pcg_gaussian(&rng,0.0,0.1);
    for (u64 i=0;i<32;i++) opt.v[0].data[i] = fabsf_((f32)pcg_gaussian(&rng,0.0,0.1));
    opt.t = 42;

    /* Orijinal degerleri kopyala (karsilastirma icin) */
    f32 orig_w1[32]; for (u64 i=0;i<32;i++) orig_w1[i]=w1->value.data[i];
    f32 orig_m0[32]; for (u64 i=0;i<32;i++) orig_m0[i]=opt.m[0].data[i];

    CHECK(checkpoint_save("test_checkpoint.bin", params, 2, &opt), "checkpoint: kaydetme basarisiz oldu");

    /* Var olmayan bir klasore kayit SESSIZCE "basarili" gorunmemeli
     * (Linux tasimasinda bulunan hata: checkpoints/ yokken egitim
     * checkpoint'leri hic yazmadan devam ediyordu). */
    CHECK(!checkpoint_save("olmayan_klasor_xyz/ckpt.bin", params, 2, &opt),
          "checkpoint: olmayan klasore kayit FALSE dondurmeli");

    /* Degerleri boz */
    for (u64 i=0;i<32;i++) w1->value.data[i] = 0.0f;
    for (u64 i=0;i<32;i++) opt.m[0].data[i] = 0.0f;
    opt.t = 0;

    Allocator scratch = allocator_create(10ull*1024*1024);
    bool32 ok = checkpoint_load(&scratch, "test_checkpoint.bin", params, 2, &opt);
    CHECK(ok, "checkpoint: yukleme basarisiz oldu");

    bool32 w1_match = TRUE;
    for (u64 i=0;i<32;i++) if (fabsf_(w1->value.data[i]-orig_w1[i]) > 1e-8f) w1_match = FALSE;
    CHECK(w1_match, "checkpoint: parametre degerleri tam geri yuklenmedi");

    bool32 m_match = TRUE;
    for (u64 i=0;i<32;i++) if (fabsf_(opt.m[0].data[i]-orig_m0[i]) > 1e-8f) m_match = FALSE;
    CHECK(m_match, "checkpoint: optimizer momentum (m) geri yuklenmedi");

    CHECK(opt.t == 42, "checkpoint: zaman adimi (t) geri yuklenmedi");

    /* Atomik kayit: basarili kayittan sonra gecici dosya kalmamali. */
    FileHandle tmpf = file_open_read("test_checkpoint.bin.tmp");
    CHECK(!tmpf.valid, "checkpoint: kayittan sonra .tmp dosyasi kaldi");
    file_close(&tmpf);

    /* --- Bozuk/yarim dosyalar reddedilmeli ve params/opt'a HIC dokunulmamali
     * (kayit sirasinda cokme -> kesik dosya senaryosu). --- */
    u64 full_size = 0;
    u8* full = (u8*)file_read_entire("test_checkpoint.bin", &scratch, &full_size);
    CHECK(full != NULL_PTR && full_size > 16, "checkpoint: kaydedilen dosya okunamadi");

    for (u64 i=0;i<32;i++) w1->value.data[i] = 7.0f;
    opt.t = 7;

    /* Son 1 bayt eksik (en ince kesik) ve yari yarida kesik */
    u64 cut_sizes[2] = { full_size - 1, full_size / 2 };
    for (u32 c = 0; c < 2; c++) {
        FileHandle tf = file_open_write("test_checkpoint_kesik.bin");
        file_write(&tf, full, cut_sizes[c]);
        file_close(&tf);
        CHECK(!checkpoint_load(&scratch, "test_checkpoint_kesik.bin", params, 2, &opt),
              "checkpoint: kesik dosya REDDEDILMEDI");
    }
    /* Fazladan bayt da reddedilmeli (boyut birebir eslesmeli) */
    {
        FileHandle tf = file_open_write("test_checkpoint_kesik.bin");
        file_write(&tf, full, full_size);
        u8 extra = 0;
        file_write(&tf, &extra, 1);
        file_close(&tf);
        CHECK(!checkpoint_load(&scratch, "test_checkpoint_kesik.bin", params, 2, &opt),
              "checkpoint: fazladan bayt iceren dosya REDDEDILMEDI");
    }
    file_delete("test_checkpoint_kesik.bin");

    bool32 untouched = (opt.t == 7);
    for (u64 i=0;i<32;i++) if (w1->value.data[i] != 7.0f) untouched = FALSE;
    CHECK(untouched, "checkpoint: basarisiz yukleme params/opt'u YARIM yukledi");

    /* Optimizer durumu olmayan dosya, opt istendiginde reddedilmeli
     * (yoksa egitime t=0 ile yanlis devam edilirdi); opt istenmezse yuklenmeli. */
    CHECK(checkpoint_save("test_checkpoint_optsuz.bin", params, 2, NULL_PTR), "checkpoint: optimizersiz kayit basarisiz");
    CHECK(!checkpoint_load(&scratch, "test_checkpoint_optsuz.bin", params, 2, &opt),
          "checkpoint: optimizer durumu olmayan dosya, opt istenince REDDEDILMEDI");
    CHECK(checkpoint_load(&scratch, "test_checkpoint_optsuz.bin", params, 2, NULL_PTR),
          "checkpoint: optimizersiz dosya, opt istenmeyince yuklenemedi");
    file_delete("test_checkpoint_optsuz.bin");

    allocator_destroy(&scratch);
    allocator_destroy(&alloc);
}

int main(void) {
    console_write_line("=== Katman 8 (Adam + Checkpoint) Testleri ===");

    test_adam_convergence();
    test_checkpoint_roundtrip();

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
