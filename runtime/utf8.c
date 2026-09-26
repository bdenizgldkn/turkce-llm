#include "utf8.h"

u32 utf8_decode(const char* s, u32* out_cp) {
    u8 b0 = (u8)s[0];

    if (b0 < 0x80) { /* 0xxxxxxx: ASCII */
        *out_cp = b0;
        return 1;
    }
    if ((b0 & 0xE0) == 0xC0) { /* 110xxxxx 10xxxxxx */
        u8 b1 = (u8)s[1];
        *out_cp = ((u32)(b0 & 0x1F) << 6) | (u32)(b1 & 0x3F);
        return 2;
    }
    if ((b0 & 0xF0) == 0xE0) { /* 1110xxxx 10xxxxxx 10xxxxxx */
        u8 b1 = (u8)s[1];
        u8 b2 = (u8)s[2];
        *out_cp = ((u32)(b0 & 0x0F) << 12) | ((u32)(b1 & 0x3F) << 6) | (u32)(b2 & 0x3F);
        return 3;
    }
    if ((b0 & 0xF8) == 0xF0) { /* 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx */
        u8 b1 = (u8)s[1];
        u8 b2 = (u8)s[2];
        u8 b3 = (u8)s[3];
        *out_cp = ((u32)(b0 & 0x07) << 18) | ((u32)(b1 & 0x3F) << 12) | ((u32)(b2 & 0x3F) << 6) | (u32)(b3 & 0x3F);
        return 4;
    }

    *out_cp = b0; /* bozuk bayt: oldugu gibi tek bayt olarak kabul et */
    return 1;
}

u32 utf8_encode(u32 cp, char* out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

u64 utf8_last_cp_len(const char* s, u64 len) {
    if (len == 0) return 0;
    i64 i = (i64)len - 1;
    while (i > 0 && ((u8)s[i] & 0xC0) == 0x80) i--;
    return len - (u64)i;
}

u64 utf8_count_codepoints(const char* s, u64 len) {
    u64 count = 0;
    u64 i = 0;
    while (i < len) {
        u32 cp;
        i += utf8_decode(s + i, &cp);
        count++;
    }
    return count;
}
