#include "strutil.h"

u64 str_len(const char* s) {
    u64 n = 0;
    while (s[n] != 0) n++;
    return n;
}

bool32 str_eq(const char* a, const char* b) {
    while (*a && *b) {
        if (*a != *b) return FALSE;
        a++; b++;
    }
    return *a == *b;
}

bool32 str_eq_n(const char* a, const char* b, u64 n) {
    for (u64 i = 0; i < n; i++) {
        if (a[i] != b[i]) return FALSE;
        if (a[i] == 0) return TRUE; /* ikisi de ayni yerde bitti */
    }
    return TRUE;
}

bool32 str_starts_with(const char* s, const char* prefix) {
    while (*prefix) {
        if (*s != *prefix) return FALSE;
        s++; prefix++;
    }
    return TRUE;
}

const char* str_find(const char* haystack, u64 haystack_len, const char* needle) {
    u64 nlen = str_len(needle);
    if (nlen == 0 || nlen > haystack_len) return NULL_PTR;

    u64 last_start = haystack_len - nlen;
    for (u64 i = 0; i <= last_start; i++) {
        bool32 match = TRUE;
        for (u64 j = 0; j < nlen; j++) {
            if (haystack[i + j] != needle[j]) { match = FALSE; break; }
        }
        if (match) return haystack + i;
    }
    return NULL_PTR;
}

void mem_copy(void* dst, const void* src, u64 n) {
    u8* d = (u8*)dst;
    const u8* s = (const u8*)src;
    for (u64 i = 0; i < n; i++) d[i] = s[i];
}

void mem_set(void* dst, u8 value, u64 n) {
    u8* d = (u8*)dst;
    for (u64 i = 0; i < n; i++) d[i] = value;
}
