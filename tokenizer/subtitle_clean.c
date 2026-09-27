#include "subtitle_clean.h"
#include "turkish_phon.h"
#include "../runtime/utf8.h"
#include "../runtime/strutil.h"

static bool32 is_lower_letter(u32 cp) {
    return tr_is_letter(cp) && tr_to_lower_cp(cp) == cp;
}

static bool32 is_space(u8 b) {
    return b == ' ' || b == '\t' || b == '\r';
}

/* Cevirmen/site imzasi iceren satirlar (film metni degil). */
static const char* const k_credit_markers[] = {
    "Çeviren", "Çeviri", "ÇEVİR", "Altyazı", "ALTYAZI", "www.", "http"
};

u64 subtitle_clean_line(const char* in, u64 in_len, char* out) {
    for (u32 m = 0; m < sizeof(k_credit_markers) / sizeof(k_credit_markers[0]); m++) {
        if (str_find(in, in_len, k_credit_markers[m]) != NULL_PTR) return 0;
    }

    u64 o = 0;
    u64 i = 0;
    u32 prev_cp = 0;          /* ciktiya yazilan son kod noktasi (duzeltilmis) */
    bool32 pending_space = FALSE;
    bool32 has_letter = FALSE;

    while (i < in_len) {
        u8 b = (u8)in[i];

        /* <etiket>, {stil} kodlari ve [ses betimlemesi]: kapanisa kadar
         * atla ({...] ve [...} karisik kapanislar da gorulur). Kapanis
         * yoksa literal karakter olarak birak (orn. "3 < 5"). */
        if (b == '<' || b == '{' || b == '[') {
            u64 j = i + 1;
            if (b == '<') {
                while (j < in_len && in[j] != '>' && in[j] != '<') j++;
                if (j < in_len && in[j] == '>') { i = j + 1; continue; }
            } else {
                while (j < in_len && in[j] != '}' && in[j] != ']' && in[j] != '{' && in[j] != '[') j++;
                if (j < in_len && (in[j] == '}' || in[j] == ']')) { i = j + 1; continue; }
                /* Kapanmayan {12345 : MicroDVD kare numarasi kalintisi. */
                if (b == '{' && i + 1 < in_len && in[i + 1] >= '0' && in[i + 1] <= '9') {
                    j = i + 1;
                    while (j < in_len && in[j] >= '0' && in[j] <= '9') j++;
                    i = j;
                    continue;
                }
            }
        }

        if (is_space(b)) {
            if (o > 0) pending_space = TRUE;
            i++;
            continue;
        }

        u32 cp;
        u32 n = utf8_decode(in + i, &cp);
        if (i + n > in_len) { n = 1; cp = b; } /* satir sonunda yarim kalmis cok baytli karakter */
        const char* src = in + i;
        i += n;

        if (cp == 0x266A || cp == 0x266B) return 0; /* ♪ ♫: sarki sozu */

        /* Satir basindaki diyalog tireleri ("-Evet", "<i>- Evet"). */
        if (o == 0 && cp == '-') { pending_space = FALSE; continue; }

        if (pending_space) { out[o++] = ' '; pending_space = FALSE; }

        /* OCR: kelime icinde kucuk harften sonra buyuk I aslinda l'dir. */
        if (cp == 'I' && o > 0 && out[o - 1] != ' ' && is_lower_letter(prev_cp)) {
            cp = 'l';
            out[o++] = 'l';
        } else {
            /* Ham baytlari kopyala (gecersiz UTF-8 yeniden kodlanip
             * buyumesin: cikti girdiden asla uzun olmaz). */
            for (u32 k = 0; k < n; k++) out[o++] = src[k];
        }
        if (tr_is_letter(cp)) has_letter = TRUE;
        prev_cp = cp;
    }

    return has_letter ? o : 0;
}
