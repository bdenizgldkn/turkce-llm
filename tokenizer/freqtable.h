/* Katman 6 - Kelime frekans tablosunu (data/raw/kelime_frekans.txt)
 * yukler. Bicim: her satirda "kelime\tsayi". */
#ifndef TOKENIZER_FREQTABLE_H
#define TOKENIZER_FREQTABLE_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/strcounter.h"

StrCounter freqtable_load(Allocator* alloc, const char* path);

#endif /* TOKENIZER_FREQTABLE_H */
