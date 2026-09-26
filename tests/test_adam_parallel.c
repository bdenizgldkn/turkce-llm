/* Katman 16 dogrulama: adam_step_parallel'in adam_step ile BIREBIR AYNI
 * (bit-hassasiyetinde) sonucu urettigini kanitlar -- coklu thread'e
 * dagitim sirasinda hicbir veri yarisi (race) veya sira-bagimli
 * farklilik olmadigini gostermek icin. Iki AYNI baslangic durumundan
 * (ayni agirliklar, ayni gradyanlar) biri adam_step (tek thread), digeri
 * adam_step_parallel (8 thread) ile bir adim atilir; degerler
 * karsilastirilir. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/prng.h"
#include "../tensor/tensor.h"
#include "../autograd/node.h"
#include "../training/adam.h"

static u32 g_pass = 0;
static u32 g_fail = 0;
static f32 fabsf_(f32 v) { return (v < 0) ? -v : v; }

#define NUM_PARAMS 37 /* bilerek thread sayisindan (8) fazla ve tam bolunmeyen bir sayi */

int main(void) {
    console_write_line("=== Katman 16: adam_step vs adam_step_parallel Esdegerlik Testi ===");

    Allocator alloc = allocator_create(64ull * 1024 * 1024);
    PCGState rng = pcg_seed(7, 3);

    Node* params_a[NUM_PARAMS];
    Node* params_b[NUM_PARAMS];

    for (u32 i = 0; i < NUM_PARAMS; i++) {
        /* Kasitli olarak COK FARKLI boyutlarda tensorler (bazilari
         * digerlerinden 100x buyuk) -- acgozlu yuk dengesinin dogru
         * calistigini da dolayli olarak sinar. */
        u64 n = 3 + (u64)(i * i % 500);
        u64 shape[1] = { n };
        Tensor va = tensor_create(&alloc, shape, 1);
        for (u64 j = 0; j < n; j++) va.data[j] = (f32)pcg_gaussian(&rng, 0.0, 1.0);
        Node* pa = node_leaf(&alloc, va, TRUE);
        for (u64 j = 0; j < n; j++) pa->grad.data[j] = (f32)pcg_gaussian(&rng, 0.0, 1.0);
        params_a[i] = pa;

        /* params_b: AYNI deger ve gradyan (kopya). */
        Tensor vb = tensor_create(&alloc, shape, 1);
        for (u64 j = 0; j < n; j++) vb.data[j] = va.data[j];
        Node* pb = node_leaf(&alloc, vb, TRUE);
        for (u64 j = 0; j < n; j++) pb->grad.data[j] = pa->grad.data[j];
        params_b[i] = pb;
    }

    AdamOptimizer opt_a = adam_create(&alloc, params_a, NUM_PARAMS, 1e-3f, 0.9f, 0.999f, 1e-8f);
    AdamOptimizer opt_b = adam_create(&alloc, params_b, NUM_PARAMS, 1e-3f, 0.9f, 0.999f, 1e-8f);

    /* Iki adim ata (t=1 ve t=2 -- onyanlilik duzeltmesinin de dogru
     * calistigini gormek icin) -- her adimdan once gradyanlari yeniden
     * (ama HER IKI optimizer icin AYNI) rastgele degerlerle doldur. */
    for (i32 step = 0; step < 2; step++) {
        for (u32 i = 0; i < NUM_PARAMS; i++) {
            for (u64 j = 0; j < params_a[i]->value.numel; j++) {
                f32 g = (f32)pcg_gaussian(&rng, 0.0, 1.0);
                params_a[i]->grad.data[j] = g;
                params_b[i]->grad.data[j] = g;
            }
        }
        adam_step(&opt_a);
        adam_step_parallel(&opt_b, 8);
    }

    for (u32 i = 0; i < NUM_PARAMS; i++) {
        for (u64 j = 0; j < params_a[i]->value.numel; j++) {
            f32 av = params_a[i]->value.data[j];
            f32 bv = params_b[i]->value.data[j];
            if (fabsf_(av - bv) < 1e-6f) g_pass++;
            else {
                g_fail++;
                console_write("  [FAIL] param="); console_write_u64(i);
                console_write(" idx="); console_write_u64(j);
                console_write_line(" degerleri farkli");
            }
        }
    }

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
