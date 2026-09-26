#include "turkish_phon.h"
#include "../runtime/utf8.h"

/* Turkce harflerin Unicode kod noktalari (UTF-8'de coğu 2 bayt). */
#define CP_a 0x0061
#define CP_A 0x0041
#define CP_e 0x0065
#define CP_E 0x0045
#define CP_i_dotless_lower 0x0131 /* ı */
#define CP_i_dotless_upper 0x0049 /* I */
#define CP_i_dotted_lower  0x0069 /* i */
#define CP_i_dotted_upper  0x0130 /* İ */
#define CP_o 0x006F
#define CP_O 0x004F
#define CP_odiaer_lower 0x00F6 /* ö */
#define CP_odiaer_upper 0x00D6 /* Ö */
#define CP_u 0x0075
#define CP_U 0x0055
#define CP_udiaer_lower 0x00FC /* ü */
#define CP_udiaer_upper 0x00DC /* Ü */

#define CP_f 0x0066
#define CP_F 0x0046
#define CP_s 0x0073
#define CP_S 0x0053
#define CP_t 0x0074
#define CP_T 0x0054
#define CP_k 0x006B
#define CP_K 0x004B
#define CP_c_cedil_lower 0x00E7 /* ç */
#define CP_c_cedil_upper 0x00C7 /* Ç */
#define CP_s_cedil_lower 0x015F /* ş */
#define CP_s_cedil_upper 0x015E /* Ş */
#define CP_h 0x0068
#define CP_H 0x0048
#define CP_p 0x0070
#define CP_P 0x0050

#define CP_b 0x0062
#define CP_c 0x0063
#define CP_d 0x0064
#define CP_gbreve_lower 0x011F /* ğ */

#define CP_A_up 0x0041
#define CP_Z_up 0x005A
#define CP_a_lo 0x0061
#define CP_C_cedil_up 0x00C7
#define CP_O_diaer_up 0x00D6
#define CP_U_diaer_up 0x00DC
#define CP_G_breve_up 0x011E
#define CP_S_cedil_up 0x015E

u32 tr_to_lower_cp(u32 cp) {
    if (cp == CP_i_dotless_upper) return CP_i_dotless_lower; /* I -> ı */
    if (cp == CP_i_dotted_upper) return CP_i_dotted_lower;   /* İ -> i */
    if (cp >= CP_A_up && cp <= CP_Z_up) return cp - CP_A_up + CP_a_lo;
    if (cp == CP_C_cedil_up) return CP_c_cedil_lower;
    if (cp == CP_O_diaer_up) return CP_odiaer_lower;
    if (cp == CP_U_diaer_up) return CP_udiaer_lower;
    if (cp == CP_G_breve_up) return CP_gbreve_lower;
    if (cp == CP_S_cedil_up) return CP_s_cedil_lower;
    return cp;
}

bool32 tr_is_letter(u32 cp) {
    if ((cp >= CP_A_up && cp <= CP_Z_up) || (cp >= CP_a_lo && cp <= (CP_a_lo + 25))) return TRUE;
    switch (cp) {
        case CP_i_dotless_lower: case CP_i_dotless_upper:
        case CP_i_dotted_upper:
        case CP_c_cedil_lower: case CP_c_cedil_upper:
        case CP_odiaer_lower: case CP_odiaer_upper:
        case CP_udiaer_lower: case CP_udiaer_upper:
        case CP_gbreve_lower: case CP_G_breve_up:
        case CP_s_cedil_lower: case CP_s_cedil_upper:
            return TRUE;
        default:
            return FALSE;
    }
}

bool32 tr_is_vowel(u32 cp) {
    switch (cp) {
        case CP_a: case CP_A:
        case CP_e: case CP_E:
        case CP_i_dotless_lower: case CP_i_dotless_upper:
        case CP_i_dotted_lower: case CP_i_dotted_upper:
        case CP_o: case CP_O:
        case CP_odiaer_lower: case CP_odiaer_upper:
        case CP_u: case CP_U:
        case CP_udiaer_lower: case CP_udiaer_upper:
            return TRUE;
        default:
            return FALSE;
    }
}

bool32 tr_vowel_is_back(u32 cp) {
    switch (cp) {
        case CP_a: case CP_A:
        case CP_i_dotless_lower: case CP_i_dotless_upper:
        case CP_o: case CP_O:
        case CP_u: case CP_U:
            return TRUE;
        default:
            return FALSE; /* e,i,ö,ü -> on (front) */
    }
}

bool32 tr_vowel_is_rounded(u32 cp) {
    switch (cp) {
        case CP_o: case CP_O:
        case CP_odiaer_lower: case CP_odiaer_upper:
        case CP_u: case CP_U:
        case CP_udiaer_lower: case CP_udiaer_upper:
            return TRUE;
        default:
            return FALSE; /* a,e,ı,i -> duz (unrounded) */
    }
}

bool32 tr_is_voiceless_consonant(u32 cp) {
    /* FISTIKÇI ŞAHAP: f,s,t,k,ç,ş,h,p */
    switch (cp) {
        case CP_f: case CP_F:
        case CP_s: case CP_S:
        case CP_t: case CP_T:
        case CP_k: case CP_K:
        case CP_c_cedil_lower: case CP_c_cedil_upper:
        case CP_s_cedil_lower: case CP_s_cedil_upper:
        case CP_h: case CP_H:
        case CP_p: case CP_P:
            return TRUE;
        default:
            return FALSE;
    }
}

bool32 tr_is_softenable(u32 cp) {
    switch (cp) {
        case CP_p: case CP_P:
        case CP_c_cedil_lower: case CP_c_cedil_upper:
        case CP_t: case CP_T:
        case CP_k: case CP_K:
            return TRUE;
        default:
            return FALSE;
    }
}

u32 tr_soften(u32 cp) {
    switch (cp) {
        case CP_p: case CP_P: return CP_b;
        case CP_c_cedil_lower: case CP_c_cedil_upper: return CP_c;
        case CP_t: case CP_T: return CP_d;
        case CP_k: case CP_K: return CP_gbreve_lower;
        default: return cp;
    }
}

u32 tr_unsoften(u32 cp) {
    switch (cp) {
        case CP_b: return CP_p;
        case CP_c: return CP_c_cedil_lower;
        case CP_d: return CP_t;
        case CP_gbreve_lower: return CP_k;
        default: return 0;
    }
}

WordEnding tr_analyze_ending(const char* word, u64 len) {
    WordEnding e;
    e.has_vowel = FALSE;
    e.last_vowel_back = FALSE;
    e.last_vowel_rounded = FALSE;
    e.last_cp = 0;

    u64 i = 0;
    u32 cp = 0;
    while (i < len) {
        u32 n = utf8_decode(word + i, &cp);
        if (tr_is_vowel(cp)) {
            e.has_vowel = TRUE;
            e.last_vowel_back = tr_vowel_is_back(cp);
            e.last_vowel_rounded = tr_vowel_is_rounded(cp);
        }
        i += n;
    }

    e.last_cp = cp;
    e.ends_in_vowel = tr_is_vowel(cp);
    return e;
}

u64 tr_resolve_suffix(const char* template_str, WordEnding ending,
                       char buffer_consonant, bool32 triggers_softening,
                       bool32 drop_initial_vowel_after_vowel,
                       char* out_buf, u32* out_softened_root_cp) {
    *out_softened_root_cp = 0;
    u64 out_len = 0;

    char first = template_str[0];
    bool32 first_is_vowel_ph = (first == 'A' || first == 'I');
    bool32 skip_first = FALSE;

    if (first_is_vowel_ph) {
        if (ending.ends_in_vowel) {
            if (drop_initial_vowel_after_vowel) {
                skip_first = TRUE;
            } else if (buffer_consonant != 0) {
                out_len += utf8_encode((u32)(u8)buffer_consonant, out_buf + out_len);
            }
        } else if (triggers_softening && tr_is_softenable(ending.last_cp)) {
            *out_softened_root_cp = tr_soften(ending.last_cp);
        }
    }

    const char* start = skip_first ? (template_str + 1) : template_str;
    for (const char* t = start; *t; t++) {
        char c = *t;
        u32 v;
        if (c == 'A') {
            v = (!ending.has_vowel || ending.last_vowel_back) ? CP_a : CP_e;
            out_len += utf8_encode(v, out_buf + out_len);
        } else if (c == 'I') {
            if (!ending.has_vowel) v = CP_i_dotted_lower;
            else if (ending.last_vowel_back && !ending.last_vowel_rounded) v = CP_i_dotless_lower;
            else if (ending.last_vowel_back && ending.last_vowel_rounded) v = CP_u;
            else if (!ending.last_vowel_back && !ending.last_vowel_rounded) v = CP_i_dotted_lower;
            else v = CP_udiaer_lower;
            out_len += utf8_encode(v, out_buf + out_len);
        } else if (c == 'D') {
            bool32 voiceless_ctx = (!ending.ends_in_vowel) && tr_is_voiceless_consonant(ending.last_cp);
            out_buf[out_len++] = voiceless_ctx ? 't' : 'd';
        } else if (c == 'C') {
            bool32 voiceless_ctx = (!ending.ends_in_vowel) && tr_is_voiceless_consonant(ending.last_cp);
            if (voiceless_ctx) out_len += utf8_encode(CP_c_cedil_lower, out_buf + out_len);
            else out_buf[out_len++] = 'c';
        } else {
            out_buf[out_len++] = c;
        }
    }

    return out_len;
}

u64 tr_apply_root_change(const char* root, u64 root_len, u32 new_last_cp, char* out_buf) {
    u64 last_len = utf8_last_cp_len(root, root_len);
    u64 keep = root_len - last_len;
    for (u64 i = 0; i < keep; i++) out_buf[i] = root[i];
    u64 written = utf8_encode(new_last_cp, out_buf + keep);
    return keep + written;
}
