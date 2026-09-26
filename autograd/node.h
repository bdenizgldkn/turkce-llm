/* Katman 4 - Otograd (Otomatik Turev) Motoru.
 *
 * Her Node bir tensor degeri, o degerle ayni sekilli bir gradyan
 * (grad) tensoru, kendini ureten "ebeveyn" node'lari ve bir geri
 * yayilim (backward) fonksiyonu tasir. backward() cagrisi, kayip
 * (loss) node'undan baslayarak once bir topolojik siralama cikarir,
 * sonra bu sirayi TERSTEN izleyerek her node'un backward_fn'ini
 * calistirir -- boylece bir node birden fazla yerde kullanilmis olsa
 * bile (DAG), gradyanlar dogru sekilde BIRIKTIRILIR (+=), UZERINE
 * YAZILMAZ.
 *
 * Not: Basitlik icin her Node'un grad tensoru olusturulurken hemen
 * (sifirlanmis, bitisik) ayrilir -- requires_grad=FALSE olan node'lar
 * icin bu kucuk bir bellek israfidir ama kodu ciddi sekilde sadelestirir.
 */
#ifndef AUTOGRAD_NODE_H
#define AUTOGRAD_NODE_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../tensor/tensor.h"

#define NODE_MAX_PARENTS 3

typedef struct Node Node;

struct Node {
    Tensor value;
    Tensor grad;
    bool32 requires_grad;

    Node* parents[NODE_MAX_PARENTS];
    u32 num_parents;

    void (*backward_fn)(Node* self, Allocator* alloc);

    bool32 visited; /* topolojik siralama icin gecici isaret */

    /* Op'a ozel yardimci veriler (backward_fn tarafindan kullanilir) */
    u32 aux_dim0;
    u32 aux_dim1;
    f32 aux_scalar;
    const void* aux_ptr; /* genel amacli: embedding/cross-entropy gibi
                           * op'larin backward'da ihtiyac duydugu ekstra
                           * durumu (orn. token ID dizisi) saklamak icin */
};

Node* node_leaf(Allocator* alloc, Tensor value, bool32 requires_grad);

/* Ozel (fused) op'lar icin dusuk seviyeli yapici: node + sifirlanmis
 * grad tensoru ayirir, num_parents/backward_fn/aux alanlarini varsayilan
 * degerlere sifirlar. Cagiran taraf (orn. model/model_ops.c) parents[],
 * backward_fn ve aux_* alanlarini kendisi doldurur. */
Node* node_make_custom(Allocator* alloc, Tensor value, bool32 requires_grad);

Node* node_add(Allocator* alloc, Node* a, Node* b);
Node* node_sub(Allocator* alloc, Node* a, Node* b);
Node* node_mul(Allocator* alloc, Node* a, Node* b);
Node* node_scale(Allocator* alloc, Node* a, f32 scalar);
Node* node_matmul2d(Allocator* alloc, Node* a, Node* b);
Node* node_transpose(Allocator* alloc, Node* a, u32 dim0, u32 dim1);
Node* node_reshape(Allocator* alloc, Node* a, const u64* shape, u32 ndim);
Node* node_sum_all(Allocator* alloc, Node* a); /* -> shape [1] */
Node* node_relu(Allocator* alloc, Node* a);
Node* node_sigmoid(Allocator* alloc, Node* a);
Node* node_tanh(Allocator* alloc, Node* a);
/* Son eksen (last dim) uzerinde softmax; girdi bitisik (contiguous) olmalidir. */
Node* node_softmax_lastdim(Allocator* alloc, Node* a);

/* loss'un grad'ini 1.0 ile tohumlar (loss->value.numel == 1 olmalidir) ve
 * tum grafikte geriye dogru gradyanlari biriktirir. */
void backward(Allocator* alloc, Node* loss);

void node_zero_grad(Node* n);

#endif /* AUTOGRAD_NODE_H */
