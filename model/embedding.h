/* Katman 7 (devam) - Token Gomme (Embedding) Katmani.
 * Cikis projeksiyonu (logits) icin AYRI bir agirlik matrisi YOK --
 * ayni gomme tablosu TRANSPOZE edilerek kullanilir (weight tying,
 * GPT-2 tarzi). Bu, buyuk vocab (31.769) ile parametre sayisini
 * neredeyse yariya indirir -- kucuk modellerde onemli bir tasarruf. */
#ifndef MODEL_EMBEDDING_H
#define MODEL_EMBEDDING_H

#include "../autograd/node.h"

/* embed_table: [vocab_size, d_model] (ogrenilebilir). token_ids: seq_len
 * uzunlugunda token ID dizisi (cagiran taraf omru boyunca gecerli
 * tutmalidir -- backward'da kullanilir, kopyalanmaz). Sonuc: [seq_len, d_model]. */
Node* node_embedding_lookup(Allocator* alloc, Node* embed_table, const u32* token_ids, u64 seq_len);

#endif /* MODEL_EMBEDDING_H */
