/* Katman 8 - Adam Optimizer.
 * Standart Adam algoritmasi (Kingma & Ba) -- literatürden alinan
 * algoritma, kod satir satir bizim tarafimizdan yazilmistir. Her
 * parametre (Node*) icin ayri momentum (m) ve ikinci moment (v)
 * tensoru tutar; onyanlilik (bias) duzeltmesi uygulanir. */
#ifndef TRAINING_ADAM_H
#define TRAINING_ADAM_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../autograd/node.h"

typedef struct AdamOptimizer {
    Node** params;   /* egitilecek parametrelerin (leaf node) dizisi */
    Tensor* m;       /* birinci moment (momentum), her parametre icin ayni sekilli */
    Tensor* v;       /* ikinci moment */
    u32 num_params;
    u64 t;           /* zaman adimi (bias duzeltmesi icin) */
    f32 lr, beta1, beta2, eps;
} AdamOptimizer;

AdamOptimizer adam_create(Allocator* alloc, Node** params, u32 num_params,
                          f32 lr, f32 beta1, f32 beta2, f32 eps);

/* params[i]->grad kullanilarak params[i]->value guncellenir. */
void adam_step(AdamOptimizer* opt);

/* adam_step ile AYNI matematiksel sonucu uretir, ama parametre
 * tensorlerini (birbirinden BAGIMSIZ oldugu icin -- her tensorun
 * guncellemesi digerlerinden hicbir sey okumaz/yazmaz) num_threads
 * kadar OS thread'ine dagitir (bkz. runtime/thread.h). Yuk dengesi
 * icin tensor SAYISINA gore degil, SKALER ELEMAN SAYISINA (numel)
 * gore acgozlu (greedy) dagitim yapilir -- aksi halde tek basina
 * toplam parametrenin ~%36'sini olusturan embed_table, kaba bir
 * "N tensor/thread" bolusunde bir thread'i tek basina bogardi. */
void adam_step_parallel(AdamOptimizer* opt, u32 num_threads);

/* Tum parametrelerin gradyanini sifirlar (bir sonraki forward/backward
 * turundan once cagrilmalidir). */
void adam_zero_grad(AdamOptimizer* opt);

#endif /* TRAINING_ADAM_H */
