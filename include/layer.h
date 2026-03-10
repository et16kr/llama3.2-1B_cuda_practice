#pragma once

#include "llama_config.h"
#include "tensor.h"

void EmbeddingLookup(TokenBatch *tokens, Tensor *embedding, Tensor *output);
void EmbeddingLookup_gpu(TokenBatch *tokens, Tensor *embedding, Tensor *output);

void RMSNorm(Tensor *input, Tensor *weight, Tensor *output, float eps);
void RMSNorm_gpu(Tensor *input, Tensor *weight, Tensor *output, float eps);

void Linear(Tensor *input, Tensor *weight, Tensor *output);
void Linear_gpu(Tensor *input, Tensor *weight, Tensor *output);

void SplitHeads(Tensor *input, Tensor *output, size_t num_heads,
                size_t head_dim);
void SplitHeads_gpu(Tensor *input, Tensor *output, size_t num_heads,
                    size_t head_dim);

void ApplyRoPE(Tensor *q, Tensor *k, const LlamaConfig &config);
void ApplyRoPE_gpu(Tensor *q, Tensor *k, const LlamaConfig &config);

void AttentionScoresGrouped(Tensor *q, Tensor *k, Tensor *scores,
                            size_t num_q_heads, size_t num_kv_heads);
void AttentionScoresGrouped_gpu(Tensor *q, Tensor *k, Tensor *scores,
                                size_t num_q_heads, size_t num_kv_heads);

void ScaleMaskSoftmax(Tensor *scores, Tensor *probs, size_t head_dim,
                      const TokenBatch *tokens);
void ScaleMaskSoftmax_gpu(Tensor *scores, Tensor *probs, size_t head_dim,
                          const TokenBatch *tokens);

void AttentionContextGrouped(Tensor *probs, Tensor *v, Tensor *context,
                             size_t num_q_heads, size_t num_kv_heads);
void AttentionContextGrouped_gpu(Tensor *probs, Tensor *v, Tensor *context,
                                 size_t num_q_heads, size_t num_kv_heads);

void MergeHeads(Tensor *context, Tensor *merged);
void MergeHeads_gpu(Tensor *context, Tensor *merged);

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output);
void ResidualAdd_gpu(Tensor *input, Tensor *addend, Tensor *output);

void SiLU(Tensor *inout);
void SiLU_gpu(Tensor *inout);

void ElementwiseMul(Tensor *lhs, Tensor *rhs, Tensor *output);
void ElementwiseMul_gpu(Tensor *lhs, Tensor *rhs, Tensor *output);

void LMHead(Tensor *input, Tensor *weight, Tensor *output);
void LMHead_gpu(Tensor *input, Tensor *weight, Tensor *output);
