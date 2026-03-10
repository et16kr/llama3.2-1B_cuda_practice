#include "layer.h"

#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "util.h"

namespace {

inline size_t flat_rows(Tensor *tensor) {
  CHECK_ERROR(tensor->ndim >= 2, "Tensor must have at least 2 dimensions");
  return tensor->num_elem() / tensor->shape[tensor->ndim - 1];
}

inline size_t last_dim(Tensor *tensor) { return tensor->shape[tensor->ndim - 1]; }

std::vector<float> build_inv_freq(const LlamaConfig &config, size_t dim) {
  constexpr float kPi = 3.14159265358979323846f;
  CHECK_ERROR((dim % 2) == 0, "RoPE head_dim must be even");
  const size_t half_dim = dim / 2;
  std::vector<float> inv_freq(half_dim, 0.0f);

  for (size_t idx = 0; idx < half_dim; ++idx) {
    float exponent = (2.0f * (float)idx) / (float)dim;
    float inv = 1.0f / powf(config.rope_theta, exponent);

    if (config.rope_type == "llama3") {
      const float factor = config.rope_factor;
      const float low_freq_factor = config.rope_low_freq_factor;
      const float high_freq_factor = config.rope_high_freq_factor;
      const float old_context_len =
          (float)config.rope_original_max_position_embeddings;

      CHECK_ERROR(factor > 0.0f, "rope factor must be positive");
      CHECK_ERROR(high_freq_factor != low_freq_factor,
                  "llama3 rope freq factors must differ");
      CHECK_ERROR(old_context_len > 0.0f,
                  "llama3 rope original context length must be positive");

      const float wavelen = 2.0f * kPi / inv;
      const float low_freq_wavelen = old_context_len / low_freq_factor;
      const float high_freq_wavelen = old_context_len / high_freq_factor;

      if (wavelen > low_freq_wavelen) {
        inv /= factor;
      } else if (wavelen >= high_freq_wavelen) {
        const float smooth_factor =
            (old_context_len / wavelen - low_freq_factor) /
            (high_freq_factor - low_freq_factor);
        inv = (1.0f - smooth_factor) * (inv / factor) + smooth_factor * inv;
      }
    }

    inv_freq[idx] = inv;
  }

  return inv_freq;
}

void apply_rope_tensor(Tensor *tensor, const LlamaConfig &config) {
  const size_t B = tensor->shape[0];
  const size_t H = tensor->shape[1];
  const size_t T = tensor->shape[2];
  const size_t D = tensor->shape[3];
  CHECK_ERROR((D % 2) == 0, "RoPE head_dim must be even");
  const size_t half_dim = D / 2;
  const std::vector<float> inv_freq = build_inv_freq(config, D);

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t t = 0; t < T; ++t) {
        float *ptr = tensor->buf + ((b * H + h) * T + t) * D;
        for (size_t i = 0; i < half_dim; ++i) {
          float angle = (float)t * inv_freq[i];
          float c = cosf(angle);
          float s = sinf(angle);
          float x0 = ptr[i];
          float x1 = ptr[i + half_dim];
          ptr[i] = x0 * c - x1 * s;
          ptr[i + half_dim] = x1 * c + x0 * s;
        }
      }
    }
  }
}

}  // namespace

void EmbeddingLookup(TokenBatch *tokens, Tensor *embedding, Tensor *output) {
  CHECK_ERROR(embedding->ndim == 2, "Embedding tensor must be rank 2");
  CHECK_ERROR(output->shape[0] == tokens->B && output->shape[1] == tokens->T,
              "Embedding output shape mismatch");
  CHECK_ERROR(output->shape[2] == embedding->shape[1],
              "Embedding hidden size mismatch");

  const size_t hidden = embedding->shape[1];
  const size_t vocab_size = embedding->shape[0];

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < tokens->B; ++b) {
    for (size_t t = 0; t < tokens->T; ++t) {
      int32_t token_id = tokens->buf[b * tokens->T + t];
      CHECK_ERROR(token_id >= 0 && token_id < (int32_t)vocab_size,
                  "Token id %d out of range", token_id);
      const float *src = embedding->buf + (size_t)token_id * hidden;
      float *dst = output->buf + (b * tokens->T + t) * hidden;
      memcpy(dst, src, hidden * sizeof(float));
    }
  }
}

void EmbeddingLookup_gpu(TokenBatch *tokens, Tensor *embedding, Tensor *output) {
  EmbeddingLookup(tokens, embedding, output);

  // TODO(student): Move embedding lookup to GPU and gather rows directly.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void RMSNorm(Tensor *input, Tensor *weight, Tensor *output, float eps) {
  size_t rows = flat_rows(input);
  size_t cols = last_dim(input);
  CHECK_ERROR(weight->ndim == 1 && weight->shape[0] == cols,
              "RMSNorm parameter shape mismatch");
  CHECK_ERROR(output->num_elem() == input->num_elem(),
              "RMSNorm output shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * cols;
    float *out = output->buf + row * cols;

    float mean_sq = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
      mean_sq += in[col] * in[col];
    }
    mean_sq /= (float)cols;

    float scale = rsqrtf(mean_sq + eps);
    for (size_t col = 0; col < cols; ++col) {
      out[col] = in[col] * scale * weight->buf[col];
    }
  }
}

void RMSNorm_gpu(Tensor *input, Tensor *weight, Tensor *output, float eps) {
  RMSNorm(input, weight, output, eps);

  // TODO(student): Implement row-wise RMSNorm reduction on GPU.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void Linear(Tensor *input, Tensor *weight, Tensor *output) {
  size_t rows = flat_rows(input);
  size_t in_dim = last_dim(input);
  CHECK_ERROR(weight->ndim == 2, "Linear weight must be rank 2");
  CHECK_ERROR(weight->shape[1] == in_dim, "Linear input dim mismatch");

  size_t out_dim = weight->shape[0];
  CHECK_ERROR(output->num_elem() == rows * out_dim, "Linear output shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * in_dim;
    float *out = output->buf + row * out_dim;
    for (size_t col = 0; col < out_dim; ++col) {
      const float *w = weight->buf + col * in_dim;
      float sum = 0.0f;
      for (size_t k = 0; k < in_dim; ++k) {
        sum += in[k] * w[k];
      }
      out[col] = sum;
    }
  }
}

void Linear_gpu(Tensor *input, Tensor *weight, Tensor *output) {
  Linear(input, weight, output);

  // TODO(student): Replace the CPU reference GEMM with CUDA kernel(s) or cuBLAS.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void SplitHeads(Tensor *input, Tensor *output, size_t num_heads, size_t head_dim) {
  CHECK_ERROR(input->ndim == 3, "SplitHeads input must be rank 3");
  CHECK_ERROR(output->ndim == 4, "SplitHeads output must be rank 4");
  CHECK_ERROR(input->shape[0] == output->shape[0] &&
                  input->shape[1] == output->shape[2],
              "SplitHeads batch/sequence shape mismatch");
  CHECK_ERROR(input->shape[2] == num_heads * head_dim,
              "SplitHeads hidden size mismatch");
  CHECK_ERROR(output->shape[1] == num_heads && output->shape[3] == head_dim,
              "SplitHeads output head shape mismatch");

  const size_t B = input->shape[0];
  const size_t T = input->shape[1];

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < num_heads; ++h) {
        const size_t src_base = (b * T + t) * (num_heads * head_dim) + h * head_dim;
        const size_t dst_base = ((b * num_heads + h) * T + t) * head_dim;
        memcpy(output->buf + dst_base, input->buf + src_base, head_dim * sizeof(float));
      }
    }
  }
}

void SplitHeads_gpu(Tensor *input, Tensor *output, size_t num_heads,
                    size_t head_dim) {
  SplitHeads(input, output, num_heads, head_dim);

  // TODO(student): Implement the [B, T, H*D] -> [B, H, T, D] layout transform.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ApplyRoPE(Tensor *q, Tensor *k, const LlamaConfig &config) {
  CHECK_ERROR(q->ndim == 4 && k->ndim == 4, "RoPE expects rank-4 tensors");
  CHECK_ERROR(q->shape[2] == k->shape[2] && q->shape[3] == k->shape[3],
              "RoPE sequence/head_dim mismatch");
  apply_rope_tensor(q, config);
  apply_rope_tensor(k, config);
}

void ApplyRoPE_gpu(Tensor *q, Tensor *k, const LlamaConfig &config) {
  ApplyRoPE(q, k, config);

  // TODO(student): Apply RoPE on GPU before attention score computation.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void AttentionScoresGrouped(Tensor *q, Tensor *k, Tensor *scores,
                            size_t num_q_heads, size_t num_kv_heads) {
  CHECK_ERROR(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads");

  const size_t B = q->shape[0];
  const size_t T = q->shape[2];
  const size_t D = q->shape[3];
  const size_t heads_per_group = num_q_heads / num_kv_heads;

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < num_q_heads; ++h) {
      for (size_t tq = 0; tq < T; ++tq) {
        const size_t kv_head = h / heads_per_group;
        const size_t score_base = ((b * num_q_heads + h) * T + tq) * T;
        const size_t q_base = ((b * num_q_heads + h) * T + tq) * D;
        for (size_t tk = 0; tk < T; ++tk) {
          const size_t k_base = ((b * num_kv_heads + kv_head) * T + tk) * D;
          float sum = 0.0f;
          for (size_t d = 0; d < D; ++d) {
            sum += q->buf[q_base + d] * k->buf[k_base + d];
          }
          scores->buf[score_base + tk] = sum;
        }
      }
    }
  }
}

void AttentionScoresGrouped_gpu(Tensor *q, Tensor *k, Tensor *scores,
                                size_t num_q_heads, size_t num_kv_heads) {
  AttentionScoresGrouped(q, k, scores, num_q_heads, num_kv_heads);

  // TODO(student): Implement grouped-query QK^T on GPU.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ScaleMaskSoftmax(Tensor *scores, Tensor *probs, size_t head_dim,
                      const TokenBatch *tokens) {
  const size_t B = scores->shape[0];
  const size_t H = scores->shape[1];
  const size_t T = scores->shape[2];
  const float scale = 1.0f / sqrtf((float)head_dim);

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t tq = 0; tq < T; ++tq) {
        const size_t valid_t =
            (tokens != nullptr && tokens->lengths != nullptr) ? (size_t)tokens->lengths[b] : T;
        const size_t row_base = ((b * H + h) * T + tq) * T;
        if (tq >= valid_t) {
          for (size_t tk = 0; tk < T; ++tk) {
            probs->buf[row_base + tk] = 0.0f;
          }
          continue;
        }

        float row_max = -1e30f;
        const size_t row_end = std::min(tq, valid_t - 1);
        for (size_t tk = 0; tk <= row_end; ++tk) {
          float value = scores->buf[row_base + tk] * scale;
          row_max = fmaxf(row_max, value);
        }

        float sum = 0.0f;
        for (size_t tk = 0; tk < T; ++tk) {
          if (tk > row_end || tk >= valid_t) {
            probs->buf[row_base + tk] = 0.0f;
            continue;
          }
          float value = scores->buf[row_base + tk] * scale;
          float e = expf(value - row_max);
          probs->buf[row_base + tk] = e;
          sum += e;
        }

        for (size_t tk = 0; tk <= row_end; ++tk) {
          probs->buf[row_base + tk] /= sum;
        }
      }
    }
  }
}

void ScaleMaskSoftmax_gpu(Tensor *scores, Tensor *probs, size_t head_dim,
                          const TokenBatch *tokens) {
  ScaleMaskSoftmax(scores, probs, head_dim, tokens);

  // TODO(student): Fuse scaling, causal masking, and softmax on GPU.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void AttentionContextGrouped(Tensor *probs, Tensor *v, Tensor *context,
                             size_t num_q_heads, size_t num_kv_heads) {
  CHECK_ERROR(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads");

  const size_t B = probs->shape[0];
  const size_t T = probs->shape[2];
  const size_t D = v->shape[3];
  const size_t heads_per_group = num_q_heads / num_kv_heads;

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < num_q_heads; ++h) {
      for (size_t tq = 0; tq < T; ++tq) {
        const size_t kv_head = h / heads_per_group;
        const size_t prob_base = ((b * num_q_heads + h) * T + tq) * T;
        const size_t out_base = ((b * num_q_heads + h) * T + tq) * D;
        for (size_t d = 0; d < D; ++d) {
          float sum = 0.0f;
          for (size_t tk = 0; tk < T; ++tk) {
            const size_t v_base = ((b * num_kv_heads + kv_head) * T + tk) * D;
            sum += probs->buf[prob_base + tk] * v->buf[v_base + d];
          }
          context->buf[out_base + d] = sum;
        }
      }
    }
  }
}

void AttentionContextGrouped_gpu(Tensor *probs, Tensor *v, Tensor *context,
                                 size_t num_q_heads, size_t num_kv_heads) {
  AttentionContextGrouped(probs, v, context, num_q_heads, num_kv_heads);

  // TODO(student): Implement grouped-query AV on GPU.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void MergeHeads(Tensor *context, Tensor *merged) {
  const size_t B = context->shape[0];
  const size_t H = context->shape[1];
  const size_t T = context->shape[2];
  const size_t D = context->shape[3];

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < H; ++h) {
        const size_t src_base = ((b * H + h) * T + t) * D;
        const size_t dst_base = (b * T + t) * (H * D) + h * D;
        memcpy(merged->buf + dst_base, context->buf + src_base, D * sizeof(float));
      }
    }
  }
}

void MergeHeads_gpu(Tensor *context, Tensor *merged) {
  MergeHeads(context, merged);

  // TODO(student): Implement the [B, H, T, D] -> [B, T, H*D] layout transform.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output) {
  CHECK_ERROR(input->num_elem() == addend->num_elem() &&
                  input->num_elem() == output->num_elem(),
              "ResidualAdd shape mismatch");

#pragma omp parallel for
  for (size_t i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = input->buf[i] + addend->buf[i];
  }
}

void ResidualAdd_gpu(Tensor *input, Tensor *addend, Tensor *output) {
  ResidualAdd(input, addend, output);

  // TODO(student): Replace elementwise add with a CUDA kernel.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void SiLU(Tensor *inout) {
#pragma omp parallel for
  for (size_t i = 0; i < inout->num_elem(); ++i) {
    float x = inout->buf[i];
    inout->buf[i] = x / (1.0f + expf(-x));
  }
}

void SiLU_gpu(Tensor *inout) {
  SiLU(inout);

  // TODO(student): Implement SiLU activation on GPU.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ElementwiseMul(Tensor *lhs, Tensor *rhs, Tensor *output) {
  CHECK_ERROR(lhs->num_elem() == rhs->num_elem() &&
                  lhs->num_elem() == output->num_elem(),
              "ElementwiseMul shape mismatch");

#pragma omp parallel for
  for (size_t i = 0; i < lhs->num_elem(); ++i) {
    output->buf[i] = lhs->buf[i] * rhs->buf[i];
  }
}

void ElementwiseMul_gpu(Tensor *lhs, Tensor *rhs, Tensor *output) {
  ElementwiseMul(lhs, rhs, output);

  // TODO(student): Replace the elementwise product with a CUDA kernel.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void LMHead(Tensor *input, Tensor *weight, Tensor *output) {
  size_t rows = flat_rows(input);
  size_t hidden = last_dim(input);
  CHECK_ERROR(weight->ndim == 2 && weight->shape[1] == hidden,
              "LMHead weight shape mismatch");
  CHECK_ERROR(output->num_elem() == rows * weight->shape[0],
              "LMHead output shape mismatch");

  const size_t vocab_size = weight->shape[0];

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * hidden;
    float *out = output->buf + row * vocab_size;
    for (size_t vocab = 0; vocab < vocab_size; ++vocab) {
      const float *w = weight->buf + vocab * hidden;
      float sum = 0.0f;
      for (size_t c = 0; c < hidden; ++c) {
        sum += in[c] * w[c];
      }
      out[vocab] = sum;
    }
  }
}

void LMHead_gpu(Tensor *input, Tensor *weight, Tensor *output) {
  LMHead(input, weight, output);

  // TODO(student): Replace the vocab projection with GPU code.
  CHECK_CUDA(cudaDeviceSynchronize());
}
