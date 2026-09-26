#include "lexicon.h"
#include "../runtime/file_io.h"

/* Sozlukteki fiiller mastar halinde kayitlidir (orn. "gitmek"), ama
 * ayristiricinin cekim icin CIPLAK fiil kokune ("git") ihtiyaci vardir.
 * "-mek"/"-mak" ile bitecek kadar uzun girdiler icin ciplak kok de
 * ayrica sozluge eklenir. */
static bool32 ends_with_mek_or_mak(const char* s, u64 len) {
    if (len < 4) return FALSE; /* en az 1 harflik kok + mek/mak */
    const char* tail = s + (len - 3);
    bool32 is_mek = (tail[0] == 'm' && tail[1] == 'e' && tail[2] == 'k');
    bool32 is_mak = (tail[0] == 'm' && tail[1] == 'a' && tail[2] == 'k');
    return is_mek || is_mak;
}

StrHashSet lexicon_load(Allocator* alloc, const char* path, u64* out_word_count) {
    u64 size = 0;
    char* buf = (char*)file_read_entire(path, alloc, &size);

    StrHashSet set = strset_create(alloc, 600000); /* ~287K satir + turetilmis fiil kokleri */

    u64 i = 0;
    u64 count = 0;
    while (i < size) {
        u64 start = i;
        while (i < size && buf[i] != '\n') i++;
        u64 end = i;
        if (end > start && buf[end - 1] == '\r') end--; /* CRLF guvenligi */

        if (end > start) {
            u64 wlen = end - start;
            strset_insert(&set, buf + start, wlen);
            count++;

            if (ends_with_mek_or_mak(buf + start, wlen)) {
                strset_insert(&set, buf + start, wlen - 3); /* ciplak fiil kokü */
                count++;
            }
        }
        i++; /* '\n' karakterini atla */
    }

    *out_word_count = count;
    return set;
}
