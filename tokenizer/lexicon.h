/* Katman 6 - Kok sozlugunu (data/raw/kok_adaylari.txt) belleğe yukleyip
 * bir StrHashSet olarak dondurur. */
#ifndef TOKENIZER_LEXICON_H
#define TOKENIZER_LEXICON_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/hashset.h"

/* path'teki, her satirda bir kelime olan dosyayi yukler.
 * *out_word_count'a yuklenen kelime sayisini yazar. */
StrHashSet lexicon_load(Allocator* alloc, const char* path, u64* out_word_count);

#endif /* TOKENIZER_LEXICON_H */
