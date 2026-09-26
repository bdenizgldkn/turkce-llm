#include "freqtable.h"
#include "../runtime/file_io.h"

StrCounter freqtable_load(Allocator* alloc, const char* path) {
    u64 size = 0;
    char* buf = (char*)file_read_entire(path, alloc, &size);

    StrCounter c = strcounter_create(alloc, 400000);
    if (!buf) return c;

    u64 i = 0;
    while (i < size) {
        u64 line_start = i;
        while (i < size && buf[i] != '\n') i++;
        u64 line_end = i;
        if (line_end > line_start && buf[line_end - 1] == '\r') line_end--;

        /* satirda '\t' ayiricisini bul */
        u64 tab = line_start;
        while (tab < line_end && buf[tab] != '\t') tab++;

        if (tab < line_end) {
            u64 word_len = tab - line_start;
            u64 val = 0;
            for (u64 k = tab + 1; k < line_end; k++) {
                val = val * 10 + (u64)(buf[k] - '0');
            }
            strcounter_set(&c, buf + line_start, word_len, val);
        }

        i++; /* '\n' atla */
    }

    return c;
}
