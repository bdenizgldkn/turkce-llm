/* Katman 8 (devam) - Dil modeli egitimi icin Cross-Entropy kaybi.
 * logits:[seq_len, vocab_size] + targets (seq_len uzunlugunda dogru
 * token ID'leri) -> tek bir skaler kayip (ortalama negatif log-olabilirlik). */
#ifndef MODEL_LOSS_H
#define MODEL_LOSS_H

#include "../autograd/node.h"

/* targets, cagiran taraf tarafindan omru boyunca gecerli tutulmalidir. */
Node* node_cross_entropy_loss(Allocator* alloc, Node* logits, const u32* targets);

#endif /* MODEL_LOSS_H */
