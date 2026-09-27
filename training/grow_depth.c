/* Faz 5 hazirligi - DERINLESTIRME (bkz. PROJE_PLANI.md Bolum 19/25,
 * "Buyutme Planlari" Secenek A). Faz 3'un 12 katmanli, egitilmis
 * agirliklarindan 24 katmanli bir model URETIR: katman i (0..23), eski
 * katman (i mod 12)'nin BIREBIR KOPYASIYLA baslar -- yani yigin iki kez
 * ust uste konur (StackBERT/bert2BERT'te literaturde bilinen, sifirdan
 * rastgele baslatmadan cok daha hizli yakinsayan bir teknik). embed_table
 * ve final_norm_w (katman sayisindan bagimsiz, ayni sekil) dogrudan
 * kopyalanir.
 *
 * ONEMLI: Bu, kayip-koruyan (loss-preserving) bir donusum DEGILDIR --
 * transformer bloklari toplamsal (residual) oldugu icin katmanlari
 * ikiletmek modelin ciktisini degistirir, egitime devam ederken kayip
 * bir sicrama gosterip sonra toparlanmasi beklenir. Adam durumu bu
 * yuzden BILEREK tasinmiyor (checkpoint_save opt=NULL_PTR ile), yeni
 * egitim sifirdan Adam durumuyla baslar (bkz. train_faz4.c'nin
 * INIT_CKPT_PATH deseni). */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"
#include "../model/lm_model.h"
#include "../training/checkpoint.h"
#ifdef _WIN32
#include "../runtime/win32_syscalls.h"
#else
#include <unistd.h>
#endif

#define VOCAB_SIZE 31769u
#define D_MODEL    384u
#define NUM_HEADS  6u
#define D_FF       1024u
#define EPS        1e-5f

#define OLD_LAYERS 12u
#define NEW_LAYERS 24u

#define SRC_CKPT "checkpoints/lm_wiki_faz3_best.bin"
#define DST_CKPT "checkpoints/lm_faz5_init.bin"

static void fatal(const char* msg) {
    console_write_line(msg);
#ifdef _WIN32
    ExitProcess(1);
#else
    _exit(1);
#endif
}

static void copy_tensor(Node* dst, const Node* src) {
    if (dst->value.numel != src->value.numel) {
        fatal("[HATA] copy_tensor: numel uyusmuyor (mimari beklenmedik sekilde farkli)");
    }
    mem_copy(dst->value.data, src->value.data, src->value.numel * sizeof(f32));
}

static void copy_block(BlockWeights* dst, const BlockWeights* src) {
    copy_tensor(dst->attn_norm_w, src->attn_norm_w);
    copy_tensor(dst->attn.w_qkv, src->attn.w_qkv);
    copy_tensor(dst->attn.b_qkv, src->attn.b_qkv);
    copy_tensor(dst->attn.wo, src->attn.wo);
    copy_tensor(dst->attn.bo, src->attn.bo);
    copy_tensor(dst->ffn_norm_w, src->ffn_norm_w);
    copy_tensor(dst->ffn.w_gate_up, src->ffn.w_gate_up);
    copy_tensor(dst->ffn.b_gate_up, src->ffn.b_gate_up);
    copy_tensor(dst->ffn.w_down, src->ffn.w_down);
    copy_tensor(dst->ffn.b_down, src->ffn.b_down);
}

int main(void) {
    console_write_line("=== Faz 5 Derinlestirme: 12 -> 24 katman ===");

    Allocator persist = allocator_create(4ull * 1024 * 1024 * 1024);
    PCGState rng = pcg_seed(1, 1); /* sadece tahsis/sekil icin -- degerler checkpoint'ten yuklenince gereksiz kalir */

    console_write_line("Eski (12 katman) model olusturuluyor...");
    LMModel old_m = lm_init(&persist, &rng, VOCAB_SIZE, D_MODEL, NUM_HEADS, OLD_LAYERS, D_FF, EPS);
    Node* old_params[LM_MAX_LAYERS * 12 + 2];
    u32 old_n = lm_collect_params(&old_m, old_params);

    console_write("Faz 3 checkpoint yukleniyor: "); console_write_line(SRC_CKPT);
    Allocator scratch = allocator_create(1024ull * 1024 * 1024);
    bool32 ok = checkpoint_load(&scratch, SRC_CKPT, old_params, old_n, NULL_PTR);
    allocator_destroy(&scratch);
    if (!ok) fatal("[HATA] Faz 3 checkpoint yuklenemedi (yol/mimari uyusmuyor olabilir).");

    console_write_line("Yeni (24 katman) model olusturuluyor...");
    LMModel new_m = lm_init(&persist, &rng, VOCAB_SIZE, D_MODEL, NUM_HEADS, NEW_LAYERS, D_FF, EPS);

    copy_tensor(new_m.embed_table, old_m.embed_table);
    copy_tensor(new_m.final_norm_w, old_m.final_norm_w);
    for (u32 l = 0; l < NEW_LAYERS; l++) {
        u32 src = l % OLD_LAYERS;
        copy_block(&new_m.blocks[l], &old_m.blocks[src]);
        console_write("  katman "); console_write_u64(l);
        console_write(" <- eski katman "); console_write_u64(src);
        console_write_line("");
    }

    Node* new_params[LM_MAX_LAYERS * 12 + 2];
    u32 new_n = lm_collect_params(&new_m, new_params);
    console_write("Toplam yeni parametre tensoru: "); console_write_u64(new_n); console_write_line("");

    u64 total_scalars = 0;
    for (u32 i = 0; i < new_n; i++) total_scalars += new_params[i]->value.numel;
    console_write("Toplam skaler parametre: "); console_write_u64(total_scalars); console_write_line("");

    ok = checkpoint_save(DST_CKPT, new_params, new_n, NULL_PTR);
    if (!ok) fatal("[HATA] Yeni checkpoint kaydedilemedi.");

    console_write("Kaydedildi: "); console_write_line(DST_CKPT);
    console_write_line("NOT: Adam durumu tasinmadi, egitim sifirdan Adam ile baslamali (INIT_CKPT_PATH deseni).");

    allocator_destroy(&persist);
    return 0;
}
