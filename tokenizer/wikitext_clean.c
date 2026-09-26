#include "wikitext_clean.h"
#include "../runtime/utf8.h"

static bool32 at(const char* in, u64 i, u64 len, const char* needle) {
    u64 j = i;
    while (*needle) {
        if (j >= len || in[j] != *needle) return FALSE;
        needle++; j++;
    }
    return TRUE;
}

u64 xml_unescape(const char* in, u64 in_len, char* out) {
    u64 i = 0, o = 0;
    while (i < in_len) {
        if (in[i] == '&') {
            if (at(in, i, in_len, "&lt;"))   { out[o++] = '<';  i += 4; continue; }
            if (at(in, i, in_len, "&gt;"))   { out[o++] = '>';  i += 4; continue; }
            if (at(in, i, in_len, "&amp;"))  { out[o++] = '&';  i += 5; continue; }
            if (at(in, i, in_len, "&quot;")) { out[o++] = '"';  i += 6; continue; }
            if (at(in, i, in_len, "&apos;")) { out[o++] = '\''; i += 6; continue; }

            if (i + 1 < in_len && in[i + 1] == '#') {
                u64 j = i + 2;
                bool32 hex = (j < in_len && (in[j] == 'x' || in[j] == 'X'));
                if (hex) j++;
                u64 start = j;
                u64 val = 0;
                while (j < in_len && in[j] != ';') {
                    char c = in[j];
                    i32 digit = -1;
                    if (c >= '0' && c <= '9') digit = c - '0';
                    else if (hex && c >= 'a' && c <= 'f') digit = c - 'a' + 10;
                    else if (hex && c >= 'A' && c <= 'F') digit = c - 'A' + 10;
                    else break;
                    val = val * (u64)(hex ? 16 : 10) + (u64)digit;
                    j++;
                }
                if (j < in_len && in[j] == ';' && j > start) {
                    o += utf8_encode((u32)val, out + o);
                    i = j + 1;
                    continue;
                }
            }
        }
        out[o++] = in[i++];
    }
    return o;
}

u64 wikitext_clean(const char* in, u64 in_len, char* out) {
    u64 i = 0, o = 0;

    while (i < in_len) {
        /* {{sablon}} -- ic ice, tamamen atilir */
        if (at(in, i, in_len, "{{")) {
            i += 2;
            i32 depth = 1;
            while (i < in_len && depth > 0) {
                if (at(in, i, in_len, "{{")) { depth++; i += 2; }
                else if (at(in, i, in_len, "}}")) { depth--; i += 2; }
                else i++;
            }
            continue;
        }

        /* {|tablo|} -- tamamen atilir */
        if (at(in, i, in_len, "{|")) {
            i += 2;
            while (i < in_len && !at(in, i, in_len, "|}")) i++;
            i += 2;
            continue;
        }

        /* <!-- yorum --> */
        if (at(in, i, in_len, "<!--")) {
            i += 4;
            while (i < in_len && !at(in, i, in_len, "-->")) i++;
            i += 3;
            continue;
        }

        /* <ref>...</ref> veya <ref .../> -- tamamen atilir */
        if (at(in, i, in_len, "<ref")) {
            u64 j = i + 4;
            bool32 self_closing = FALSE;
            while (j < in_len && in[j] != '>') {
                if (in[j] == '/' && j + 1 < in_len && in[j + 1] == '>') { self_closing = TRUE; break; }
                j++;
            }
            if (self_closing) { i = j + 2; continue; }
            j++; /* '>' atla */
            while (j < in_len && !at(in, j, in_len, "</ref>")) j++;
            i = j + 6;
            continue;
        }

        /* <math>...</math> -- tamamen atilir */
        if (at(in, i, in_len, "<math")) {
            u64 j = i + 5;
            while (j < in_len && in[j] != '>') j++;
            j++;
            while (j < in_len && !at(in, j, in_len, "</math>")) j++;
            i = j + 7;
            continue;
        }

        /* [[Baglanti|Metin]] veya [[Baglanti]] -- Kategori/Dosya/Resim atilir */
        if (at(in, i, in_len, "[[")) {
            u64 j = i + 2;
            u64 start = j;
            i32 depth = 1;
            while (j < in_len && depth > 0) {
                if (at(in, j, in_len, "[[")) { depth++; j += 2; }
                else if (at(in, j, in_len, "]]")) { depth--; if (depth == 0) break; j += 2; }
                else j++;
            }
            u64 end = j; /* "]]" konumu */

            bool32 discard = at(in, start, in_len, "Kategori:") || at(in, start, in_len, "Category:") ||
                              at(in, start, in_len, "Dosya:") || at(in, start, in_len, "File:") ||
                              at(in, start, in_len, "Resim:") || at(in, start, in_len, "Image:");
            if (!discard) {
                u64 pipe = (u64)-1;
                for (u64 p = start; p < end; p++) if (in[p] == '|') pipe = p;
                u64 text_start = (pipe != (u64)-1) ? pipe + 1 : start;
                for (u64 p = text_start; p < end; p++) out[o++] = in[p];
            }
            i = (end < in_len) ? end + 2 : in_len;
            continue;
        }

        /* [http://... gorunen metin] */
        if (at(in, i, in_len, "[http")) {
            u64 j = i + 1;
            while (j < in_len && in[j] != ']') j++;
            u64 end = j;
            u64 sp = (u64)-1;
            for (u64 p = i + 1; p < end; p++) if (in[p] == ' ') { sp = p; break; }
            if (sp != (u64)-1) {
                for (u64 p = sp + 1; p < end; p++) out[o++] = in[p];
            }
            i = (end < in_len) ? end + 1 : in_len;
            continue;
        }

        /* '''kalin''' / ''italik'' -- isaretleme silinir */
        if (at(in, i, in_len, "'''")) { i += 3; continue; }
        if (at(in, i, in_len, "''"))  { i += 2; continue; }

        /* == Baslik == -- '=' isaretleri silinir, metin kalir */
        if (in[i] == '=') {
            u64 j = i;
            while (j < in_len && in[j] == '=') j++;
            i = j;
            continue;
        }

        /* Diger HTML-benzeri etiketler (<br>, <gallery>, </div> vb.) --
         * sadece etiket silinir, ic metin (varsa disaridan) kalir. */
        if (in[i] == '<') {
            u64 j = i + 1;
            while (j < in_len && in[j] != '>') j++;
            i = (j < in_len) ? j + 1 : in_len;
            continue;
        }

        out[o++] = in[i++];
    }

    return o;
}
