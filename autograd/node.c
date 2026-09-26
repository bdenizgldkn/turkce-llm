#include "node.h"

/* ================= Yardimcilar ================= */

static Node* alloc_node(Allocator* alloc) {
    return (Node*)allocator_alloc(alloc, sizeof(Node));
}

static Tensor zeros_like(Allocator* alloc, const Tensor* t) {
    Tensor z = tensor_create(alloc, t->shape, t->ndim);
    tensor_fill(&z, 0.0f);
    return z;
}

static Node* make_node(Allocator* alloc, Tensor value, bool32 requires_grad) {
    Node* n = alloc_node(alloc);
    n->value = value;
    n->grad = zeros_like(alloc, &value);
    n->requires_grad = requires_grad;
    n->num_parents = 0;
    n->parents[0] = NULL_PTR;
    n->parents[1] = NULL_PTR;
    n->parents[2] = NULL_PTR;
    n->backward_fn = NULL_PTR;
    n->visited = FALSE;
    n->aux_dim0 = 0;
    n->aux_dim1 = 0;
    n->aux_scalar = 0.0f;
    n->aux_ptr = NULL_PTR;
    return n;
}

/* Genel amacli: parent->grad += (kaynak shape'i parent->grad ile ayni
 * olan) src tensoru, eleman-bazli, stride'lar farkli olabilir. */
static void accumulate(Tensor* parent_grad, const Tensor* src) {
    tensor_add_inplace(parent_grad, src);
}

Node* node_leaf(Allocator* alloc, Tensor value, bool32 requires_grad) {
    return make_node(alloc, value, requires_grad);
}

Node* node_make_custom(Allocator* alloc, Tensor value, bool32 requires_grad) {
    return make_node(alloc, value, requires_grad);
}

/* ================= add ================= */

static void backward_add(Node* self, Allocator* alloc) {
    (void)alloc;
    accumulate(&self->parents[0]->grad, &self->grad);
    accumulate(&self->parents[1]->grad, &self->grad);
}

Node* node_add(Allocator* alloc, Node* a, Node* b) {
    Tensor v = tensor_add(alloc, &a->value, &b->value);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->parents[1] = b; n->num_parents = 2;
    n->backward_fn = backward_add;
    return n;
}

/* ================= sub ================= */

static void backward_sub(Node* self, Allocator* alloc) {
    accumulate(&self->parents[0]->grad, &self->grad);
    Tensor neg = tensor_scale(alloc, &self->grad, -1.0f);
    accumulate(&self->parents[1]->grad, &neg);
}

Node* node_sub(Allocator* alloc, Node* a, Node* b) {
    Tensor v = tensor_sub(alloc, &a->value, &b->value);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->parents[1] = b; n->num_parents = 2;
    n->backward_fn = backward_sub;
    return n;
}

/* ================= mul (eleman-bazli) ================= */

static void backward_mul(Node* self, Allocator* alloc) {
    Node* a = self->parents[0];
    Node* b = self->parents[1];
    Tensor da = tensor_mul(alloc, &self->grad, &b->value);
    Tensor db = tensor_mul(alloc, &self->grad, &a->value);
    accumulate(&a->grad, &da);
    accumulate(&b->grad, &db);
}

Node* node_mul(Allocator* alloc, Node* a, Node* b) {
    Tensor v = tensor_mul(alloc, &a->value, &b->value);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->parents[1] = b; n->num_parents = 2;
    n->backward_fn = backward_mul;
    return n;
}

/* ================= scale ================= */

static void backward_scale(Node* self, Allocator* alloc) {
    Tensor d = tensor_scale(alloc, &self->grad, self->aux_scalar);
    accumulate(&self->parents[0]->grad, &d);
}

Node* node_scale(Allocator* alloc, Node* a, f32 scalar) {
    Tensor v = tensor_scale(alloc, &a->value, scalar);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->num_parents = 1;
    n->aux_scalar = scalar;
    n->backward_fn = backward_scale;
    return n;
}

/* ================= matmul2d ================= */

static void backward_matmul2d(Node* self, Allocator* alloc) {
    Node* a = self->parents[0];
    Node* b = self->parents[1];

    Tensor bt = tensor_transpose(&b->value, 0, 1);
    Tensor da = tensor_matmul2d(alloc, &self->grad, &bt);
    accumulate(&a->grad, &da);

    Tensor at = tensor_transpose(&a->value, 0, 1);
    Tensor db = tensor_matmul2d(alloc, &at, &self->grad);
    accumulate(&b->grad, &db);
}

Node* node_matmul2d(Allocator* alloc, Node* a, Node* b) {
    Tensor v = tensor_matmul2d(alloc, &a->value, &b->value);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->parents[1] = b; n->num_parents = 2;
    n->backward_fn = backward_matmul2d;
    return n;
}

/* ================= transpose ================= */

static void backward_transpose(Node* self, Allocator* alloc) {
    (void)alloc;
    Tensor view = tensor_transpose(&self->grad, self->aux_dim0, self->aux_dim1);
    accumulate(&self->parents[0]->grad, &view);
}

Node* node_transpose(Allocator* alloc, Node* a, u32 dim0, u32 dim1) {
    Tensor v = tensor_transpose(&a->value, dim0, dim1);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->num_parents = 1;
    n->aux_dim0 = dim0; n->aux_dim1 = dim1;
    n->backward_fn = backward_transpose;
    return n;
}

/* ================= reshape ================= */

static void backward_reshape(Node* self, Allocator* alloc) {
    /* self->grad ve parent->grad her zaman bitisik (contiguous) ve ayni
     * numel'e sahiptir (Node grad'lari her zaman tazeden ayrilir) -- bu
     * yuzden dogrudan duz (flat) indeksle biriktirmek gecerlidir. */
    (void)alloc;
    Tensor* pg = &self->parents[0]->grad;
    for (u64 i = 0; i < pg->numel; i++) pg->data[i] += self->grad.data[i];
}

Node* node_reshape(Allocator* alloc, Node* a, const u64* shape, u32 ndim) {
    Tensor v = tensor_reshape(&a->value, shape, ndim);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->num_parents = 1;
    n->backward_fn = backward_reshape;
    return n;
}

/* ================= sum_all ================= */

static void backward_sum_all(Node* self, Allocator* alloc) {
    (void)alloc;
    Tensor* pg = &self->parents[0]->grad;
    f32 g = self->grad.data[0];
    for (u64 i = 0; i < pg->numel; i++) pg->data[i] += g;
}

Node* node_sum_all(Allocator* alloc, Node* a) {
    Tensor v = tensor_sum_all(alloc, &a->value);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->num_parents = 1;
    n->backward_fn = backward_sum_all;
    return n;
}

/* ================= relu ================= */

static void backward_relu(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* a = self->parents[0];
    u64 idx[TENSOR_MAX_DIMS];
    for (u64 flat = 0; flat < self->grad.numel; flat++) {
        tensor_flat_to_multi(flat, self->grad.shape, self->grad.ndim, idx);
        u64 aoff = tensor_compute_offset(a->value.strides, idx, a->value.ndim);
        u64 goff = tensor_compute_offset(self->grad.strides, idx, self->grad.ndim);
        u64 pgoff = tensor_compute_offset(a->grad.strides, idx, a->grad.ndim);
        if (a->value.data[aoff] > 0.0f) {
            a->grad.data[pgoff] += self->grad.data[goff];
        }
    }
}

Node* node_relu(Allocator* alloc, Node* a) {
    Tensor v = tensor_relu(alloc, &a->value);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->num_parents = 1;
    n->backward_fn = backward_relu;
    return n;
}

/* ================= sigmoid ================= */

static void backward_sigmoid(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* a = self->parents[0];
    /* self->value ve self->grad daima bitisik ve ayni sekildedir. */
    for (u64 i = 0; i < self->grad.numel; i++) {
        f32 y = self->value.data[i];
        f32 local = y * (1.0f - y);
        a->grad.data[i] += self->grad.data[i] * local;
    }
}

Node* node_sigmoid(Allocator* alloc, Node* a) {
    Tensor v = tensor_sigmoid(alloc, &a->value);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->num_parents = 1;
    n->backward_fn = backward_sigmoid;
    return n;
}

/* ================= tanh ================= */

static void backward_tanh(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* a = self->parents[0];
    for (u64 i = 0; i < self->grad.numel; i++) {
        f32 y = self->value.data[i];
        f32 local = 1.0f - y * y;
        a->grad.data[i] += self->grad.data[i] * local;
    }
}

Node* node_tanh(Allocator* alloc, Node* a) {
    Tensor v = tensor_tanh_op(alloc, &a->value);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->num_parents = 1;
    n->backward_fn = backward_tanh;
    return n;
}

/* ================= softmax (son eksen) ================= */

static void backward_softmax_lastdim(Node* self, Allocator* alloc) {
    (void)alloc;
    Node* a = self->parents[0];
    u64 last = self->value.shape[self->value.ndim - 1];
    u64 outer = self->value.numel / last;

    for (u64 o = 0; o < outer; o++) {
        const f32* y = self->value.data + o * last;   /* softmax ciktisi (bu satir) */
        const f32* dy = self->grad.data + o * last;    /* gelen gradyan (bu satir) */
        f32* dx = a->grad.data + o * last;

        f32 dot = 0.0f;
        for (u64 i = 0; i < last; i++) dot += dy[i] * y[i];

        for (u64 i = 0; i < last; i++) {
            dx[i] += y[i] * (dy[i] - dot);
        }
    }
}

Node* node_softmax_lastdim(Allocator* alloc, Node* a) {
    Tensor v = tensor_softmax_lastdim(alloc, &a->value);
    Node* n = make_node(alloc, v, TRUE);
    n->parents[0] = a; n->num_parents = 1;
    n->backward_fn = backward_softmax_lastdim;
    return n;
}

/* ================= backward (topolojik siralama + geri yayilim) ================= */

#define TOPO_MAX_NODES 65536

static void topo_visit(Node* n, Node** topo, u64* count) {
    if (n == NULL_PTR || n->visited) return;
    n->visited = TRUE;
    for (u32 i = 0; i < n->num_parents; i++) {
        topo_visit(n->parents[i], topo, count);
    }
    topo[*count] = n;
    (*count)++;
}

void backward(Allocator* alloc, Node* loss) {
    Node** topo = (Node**)allocator_alloc(alloc, sizeof(Node*) * TOPO_MAX_NODES);
    u64 count = 0;

    topo_visit(loss, topo, &count);

    /* Kok (loss) tohumu: dLoss/dLoss = 1. */
    loss->grad.data[0] = 1.0f;

    /* Post-order listesini TERSTEN isle: once loss, sonra ebeveynleri. */
    for (i64 i = (i64)count - 1; i >= 0; i--) {
        Node* n = topo[i];
        if (n->backward_fn) n->backward_fn(n, alloc);
    }

    /* visited bayraklarini temizle (ayni arena/graf tekrar kullanilmayacaksa
     * gerekli degildir, ama guvenlik icin sifirliyoruz). */
    for (u64 i = 0; i < count; i++) topo[i]->visited = FALSE;
}

void node_zero_grad(Node* n) {
    tensor_fill(&n->grad, 0.0f);
}
