#include "layer.h"

#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "util.h"

#define TILE_SIZE (32)
#define BLOCK_SIZE (1024)
#define CEIL(x, y) ((x+y-1)/y)
namespace {

inline size_t flat_rows(Tensor *tensor) {
  CHECK_ERROR(tensor->ndim >= 2, "Tensor must have at least 2 dimensions");
  return tensor->num_elem() / tensor->shape[tensor->ndim - 1];
}

inline size_t last_dim(Tensor *tensor) { return tensor->shape[tensor->ndim - 1]; }

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

__global__ void embedding_lookup(int32_t* tokens, float* embedding, float* output, size_t B, size_t T, size_t hidden) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t b = idx / (T*hidden);
  size_t t = idx / hidden % T;
  size_t h = idx % hidden;
  if (b >= B) return;
  int32_t token_id = tokens[b * T + t];
  
  output[(b * T * hidden) + (t * hidden) + h] = embedding[(size_t)token_id * hidden + h];
}
// [EmbeddingLookup_gpu] B: 32, T: 56, hidden: 2048
void EmbeddingLookup_gpu(TokenBatch *tokens, Tensor *embedding, Tensor *output) {
  CHECK_ERROR(embedding->ndim == 2, "Embedding tensor must be rank 2");
  CHECK_ERROR(output->shape[0] == tokens->B && output->shape[1] == tokens->T,
              "Embedding output shape mismatch");
  CHECK_ERROR(output->shape[2] == embedding->shape[1],
              "Embedding hidden size mismatch");

  const size_t B = tokens->B;
  const size_t T = tokens->T;
  const size_t hidden = embedding->shape[1];
//  const size_t vocab_size = embedding->shape[0];
  const size_t N = B * T * hidden;

  // TODO(student): Move embedding lookup to GPU and gather rows directly.
  dim3 gridDim(CEIL(N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);
  embedding_lookup<<<gridDim, blockDim>>>(tokens->gpu_buf, embedding->gpu_buf, output->gpu_buf, B, T, hidden);
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

__global__ void rms_norm(float *input, float *weight, float *output, float eps, size_t cols) {
  size_t row = blockIdx.x;
  size_t idx = threadIdx.x;
  __shared__ float L[BLOCK_SIZE];
  input += row * cols;
  output += row * cols;

  float mean_sq = 0.0f;
  for (size_t col = idx; col < cols; col += blockDim.x) {
    mean_sq += input[col] * input[col];
  }
  L[idx] = mean_sq;

  for(int i = blockDim.x/2 ; i > 0 ; i /= 2) {
    __syncthreads();
    if (idx < i) L[idx] += L[idx + i];
  }
  __shared__ float scale ;
  if (idx == 0) {
    mean_sq = L[0] / (float)cols;
    scale = rsqrtf(mean_sq + eps);
  }
  __syncthreads();
  
  for (size_t col = idx; col < cols; col += blockDim.x) {
    output[col] = input[col] * scale * weight[col];
  }
}

// [RMSNorm] rows: 1792, cols: 2048
void RMSNorm_gpu(Tensor *input, Tensor *weight, Tensor *output, float eps) {
  size_t rows = flat_rows(input);
  size_t cols = last_dim(input);
  CHECK_ERROR(weight->ndim == 1 && weight->shape[0] == cols,
              "RMSNorm parameter shape mismatch");
  CHECK_ERROR(output->num_elem() == input->num_elem(),
              "RMSNorm output shape mismatch");

  // TODO(student): Implement row-wise RMSNorm reduction on GPU.
  dim3 gridDim(rows);
  dim3 blockDim(BLOCK_SIZE);
  rms_norm<<<gridDim, blockDim>>>(input->gpu_buf, weight->gpu_buf, output->gpu_buf, eps, cols);
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
__global__ void linear(float *input, float *weight, float *output, size_t rows, size_t out_dim, size_t in_dim) {
  size_t ty = threadIdx.y;
  size_t tx = threadIdx.x;
  size_t rowi = blockIdx.y * TILE_SIZE + ty;
  size_t weight_idx = blockIdx.x * TILE_SIZE + ty;
  __shared__ float LI[TILE_SIZE][TILE_SIZE];
  __shared__ float LW[TILE_SIZE][TILE_SIZE + 1];
  input += rowi * in_dim;
  weight += weight_idx * in_dim;
  float sum = 0.0f;
  for (size_t k = 0; k < in_dim; k += TILE_SIZE) {
    size_t in_idx = k + tx;
    LI[ty][tx] = (rowi < rows && in_idx < in_dim)           ? input[in_idx] : 0.0f;
    LW[tx][ty] = (weight_idx < out_dim && in_idx < in_dim ) ? weight[in_idx] : 0.0f;
    __syncthreads();
    for (int i = 0 ; i < TILE_SIZE ; i++) {
      sum += LI[ty][i] * LW[i][tx];
    }
    __syncthreads();
  }
  if (rowi >= rows) return;
  size_t roww = blockIdx.x * TILE_SIZE + tx;
  if (roww >= out_dim) return;
  output[rowi * out_dim + roww] = sum;
}
/*
[Linear] rows: 1792, out_dim: 512, in_dim: 2048
[Linear] rows: 1792, out_dim: 2048, in_dim: 2048
[Linear] rows: 1792, out_dim: 8192, in_dim: 2048
[Linear] rows: 1792, out_dim: 2048, in_dim: 8192
*/
void Linear_gpu(Tensor *input, Tensor *weight, Tensor *output) {
  size_t rows = flat_rows(input);
  size_t in_dim = last_dim(input);
  CHECK_ERROR(weight->ndim == 2, "Linear weight must be rank 2");
  CHECK_ERROR(weight->shape[1] == in_dim, "Linear input dim mismatch");

  size_t out_dim = weight->shape[0];
  CHECK_ERROR(output->num_elem() == rows * out_dim, "Linear output shape mismatch");

  // TODO(student): Replace the CPU reference GEMM with CUDA kernel(s) or cuBLAS.
  dim3 gridDim(CEIL(out_dim, TILE_SIZE), CEIL(rows, TILE_SIZE));
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  linear<<<gridDim, blockDim>>>(input->gpu_buf, weight->gpu_buf, output->gpu_buf, rows, out_dim, in_dim);
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

__global__ void split_heads(float* input, float* output, size_t B, size_t T, size_t num_heads, size_t head_dim) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t b = idx / (T*num_heads* head_dim);
  size_t h = idx / (T * head_dim) %  num_heads;
  size_t t = idx / head_dim % T;
  size_t d = idx % head_dim;
  if (b >= B) return;
  const size_t src_idx = (b * T * num_heads * head_dim) + (t * num_heads * head_dim) + h * head_dim + d;
  const size_t dst_idx = (b * num_heads* T * head_dim ) + (h * T * head_dim )+ (t * head_dim) + d;
  output[dst_idx] = input[src_idx];
}

// [SplitHeads_gpu] B: 32, T: 32, N: 3670016
// [SplitHeads_gpu] B: 32, T: 8, N: 917504
// [SplitHeads_gpu] B: 32, T: 8, N: 917504
void SplitHeads_gpu(Tensor *input, Tensor *output, size_t num_heads,
                    size_t head_dim) {
  CHECK_ERROR(input->ndim == 3, "SplitHeads input must be rank 3");
  CHECK_ERROR(output->ndim == 4, "SplitHeads output must be rank 4");
  CHECK_ERROR(input->shape[0] == output->shape[0] &&
              input->shape[1] == output->shape[2],
              "SplitHeads batch/sequence shape mismatch");
  CHECK_ERROR(input->shape[2] == num_heads * head_dim,
              "SplitHeads hidden size mismatch");
  CHECK_ERROR(output->shape[1] == num_heads && output->shape[3] == head_dim,
              "SplitHeads output head shape mismatch");

  const size_t B = output->shape[0];
  const size_t T = output->shape[2];
  const size_t N = output->num_elem();

  // TODO(student): Implement the [B, T, H*D] -> [B, H, T, D] layout transform.
  dim3 gridDim(CEIL(N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);
  split_heads<<<gridDim, blockDim>>>(input->gpu_buf, output->gpu_buf, B, T, num_heads, head_dim);
  CHECK_CUDA(cudaDeviceSynchronize());
}

namespace {
  /*
   "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
   "rope_theta": 500000.0,
  */

float get_inv_freq(size_t dim, size_t idx) {
  constexpr float kPi = 3.14159265358979323846f;
  CHECK_ERROR((dim % 2) == 0, "RoPE head_dim must be even");
  const float factor = 32.0;
  const float low_freq_factor = 1.0;
  const float high_freq_factor = 4.0;
  const float rope_theta = 500000.0;
  const float rope_original_max_position_embeddings = 8192;
  const float old_context_len = rope_original_max_position_embeddings;

  float exponent = (2.0f * (float)idx) / (float)dim;
  float inv = 1.0f / powf(rope_theta, exponent);


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
  return inv;
}

/*
[build_inv_freq] half_dim: 32
*/

/*
[apply_rope_tensor] B: 32, H: 32, T: 56, D: 64, half_dim: 32
[apply_rope_tensor] B: 32, H: 8, T: 56, D: 64, half_dim: 32
*/

void apply_rope_tensor(Tensor *tensor) {
  const size_t B = tensor->shape[0];
  const size_t H = tensor->shape[1];
  const size_t T = tensor->shape[2];
  const size_t D = tensor->shape[3];
  CHECK_ERROR((D % 2) == 0, "RoPE head_dim must be even");
  const size_t half_dim = D / 2;
  std::vector<float> inv_freq;
  for (size_t idx = 0 ; idx < half_dim ; idx++) {
    inv_freq.push_back(get_inv_freq(D, idx));
  }

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

void ApplyRoPE(Tensor *q, Tensor *k) {
  CHECK_ERROR(q->ndim == 4 && k->ndim == 4, "RoPE expects rank-4 tensors");
  CHECK_ERROR(q->shape[2] == k->shape[2] && q->shape[3] == k->shape[3],
              "RoPE sequence/head_dim mismatch");
  apply_rope_tensor(q);
  apply_rope_tensor(k);
}

// XXX

__device__ float inv_freq(size_t dim, size_t idx) {
  constexpr float kPi = 3.14159265358979323846f;
  const float factor = 32.0;
  const float low_freq_factor = 1.0;
  const float high_freq_factor = 4.0;
  const float rope_theta = 500000.0;
  const float rope_original_max_position_embeddings = 8192;
  const float old_context_len = rope_original_max_position_embeddings;

  float exponent = (2.0f * (float)idx) / (float)dim;
  float inv = 1.0f / powf(rope_theta, exponent);

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
  return inv;
}

__global__ void rope(float *tensor, size_t B, size_t H, size_t T, size_t D) {
  const size_t half_dim = D / 2;
  const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t b = idx / (H*T*half_dim);
  const size_t h = idx / (T*half_dim) % H;
  const size_t t = idx / half_dim % T;
  const size_t hd = idx % half_dim;
  if (b >= B) return;
  tensor += ((b * H + h) * T + t) * D + hd;
  float angle = (float)t * inv_freq(D, hd);
  float c = cosf(angle);
  float s = sinf(angle);
  float x0 = tensor[0];
  float x1 = tensor[half_dim];
  tensor[0] = x0 * c - x1 * s;
  tensor[half_dim] = x1 * c + x0 * s;
}
/*
[apply_rope_tensor] B: 32, H: 32, T: 56, D: 64, half_dim: 32
[apply_rope_tensor] B: 32, H: 8, T: 56, D: 64, half_dim: 32
*/
void ApplyRoPE_gpu(Tensor *q, Tensor *k) {
  CHECK_ERROR(q->ndim == 4 && k->ndim == 4, "RoPE expects rank-4 tensors");
  CHECK_ERROR(q->shape[2] == k->shape[2] && q->shape[3] == k->shape[3],
              "RoPE sequence/head_dim mismatch");
  const size_t B = q->shape[0];
  const size_t T = q->shape[2];
  const size_t D = q->shape[3];
  const size_t Hq = q->shape[1];
  const size_t Hk = k->shape[1];
  const size_t half_dim = D/2;
  CHECK_ERROR((D % 2) == 0, "RoPE head_dim must be even");
  dim3 gridDim(CEIL( B * Hq * T * half_dim, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);
  rope<<<gridDim, blockDim>>>(q->gpu_buf, B, Hq, T, D);
  gridDim.x = CEIL( B * Hk * T * half_dim, BLOCK_SIZE);
  rope<<<gridDim, blockDim>>>(k->gpu_buf, B, Hk, T, D);
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

__global__ void attention_scores_grouped(float *q, float *k, float *scores,
                                          size_t num_q_heads, size_t num_kv_heads, size_t B, size_t T, size_t D) {
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;
  size_t b = x / num_q_heads;
  size_t h = x % num_q_heads;
  size_t tq = y / T;
  size_t tk = y % T;
  if (b >= B) return;
  if (tq >= T) return;

  const size_t kv_head = h / (num_q_heads / num_kv_heads);
  q += (b * num_q_heads * T * D) + (h * T * D) + (tq * D);
  k += (b * num_kv_heads * T * D) + (kv_head * T * D) + (tk * D);
  float sum = 0.0f;
  for (size_t d = 0; d < D; ++d) {
    sum += q[d] * k[d];
  }
  scores[(b * num_q_heads * T * T) + ( h * T * T) + (tq * T)+ tk] = sum;
}
//[AttentionScoresGrouped] B: 32, T: 56, D: 64, num_q_heads: 32, num_kv_heads: 8, heads_per_group: 4
void AttentionScoresGrouped_gpu(Tensor *q, Tensor *k, Tensor *scores,
                                size_t num_q_heads, size_t num_kv_heads) {
  CHECK_ERROR(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads");

  const size_t B = q->shape[0];
  const size_t T = q->shape[2];
  const size_t D = q->shape[3];

  // TODO(student): Implement grouped-query QK^T on GPU.
  dim3 gridDim(CEIL(B*num_q_heads, TILE_SIZE),CEIL(T*T, TILE_SIZE));
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  attention_scores_grouped<<<gridDim, blockDim>>>(q->gpu_buf, k->gpu_buf, scores->gpu_buf, num_q_heads, num_kv_heads, B, T, D);

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
/*
[ScaleMaskSoftmax] B: 32, H: 32, T: 56
*/
// XXX
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

__global__ void attention_context_grouped(float *probs, float *v, float *context,
                                          size_t num_q_heads, size_t num_kv_heads, size_t B, size_t T, size_t D) {
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;
  size_t b = x / num_q_heads;
  size_t h = x % num_q_heads;
  size_t tq = y / D;
  size_t d = y % D;
  if (b >= B) return;
  if (tq >= T) return;

  const size_t kv_head = h / ( num_q_heads / num_kv_heads);
  probs += ((b * num_q_heads + h) * T + tq) * T;
  v += (b * num_kv_heads * T  * D ) + (kv_head * T  * D);
  float sum = 0.0f;
  for (size_t tk = 0; tk < T; ++tk) {
    sum += probs[tk] * v[(tk * D) + d];
  }
  context[(b * num_q_heads * T * D ) + ( h * T * D ) + (tq * D) + d] = sum;
}

// [AttentionContextGrouped_gpu] B: 32, T: 56, D: 64, num_q_heads: 32, num_kv_heads: 8, heads_per_group: 4
void AttentionContextGrouped_gpu(Tensor *probs, Tensor *v, Tensor *context,
                                 size_t num_q_heads, size_t num_kv_heads) {
  CHECK_ERROR(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads");

  const size_t B = probs->shape[0];
  const size_t T = probs->shape[2];
  const size_t D = v->shape[3];

  // TODO(student): Implement grouped-query AV on GPU.
  dim3 gridDim(CEIL(B*num_q_heads, TILE_SIZE),CEIL(T*D, TILE_SIZE));
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  attention_context_grouped<<<gridDim, blockDim>>>(probs->gpu_buf, v->gpu_buf, context->gpu_buf, num_q_heads, num_kv_heads, B, T, D);
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

// [MergeHeads_gpu] B: 32, H: 32, T: 56, D: 64, N: 3670016
__global__ void merge_heads(float* context, float* merged, size_t B, size_t H, size_t T, size_t D) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t b = idx / (T*H*D);
  size_t t = idx / (H*D) % T;
  size_t h = idx / D % H;
  size_t d = idx % D;
  if ( b >= B) return;
  merged[(b * T *H * D ) + (t *H * D)  + (h * D) + d] = context[(b * H * T * D) + (h * T * D) + (t * D) + d];
}

void MergeHeads_gpu(Tensor *context, Tensor *merged) {
  const size_t B = context->shape[0];
  const size_t H = context->shape[1];
  const size_t T = context->shape[2];
  const size_t D = context->shape[3];
  const size_t N = context->num_elem();

  dim3 gridDim(CEIL(N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);
  // TODO(student): Implement the [B, H, T, D] -> [B, T, H*D] layout transform.
  merge_heads<<<gridDim, blockDim>>>(context->gpu_buf, merged->gpu_buf, B, H, T, D);

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

__global__ void residual_add(float *input, float *addend, float*output, size_t N) {
  size_t n = blockDim.x * blockIdx.x + threadIdx.x;
  if ( n >= N ) return;
  output[n] = input[n] + addend[n];
}

// [ResidualAdd_gpu] N: 3670016
void ResidualAdd_gpu(Tensor *input, Tensor *addend, Tensor *output) {
  size_t N = input->num_elem();
  CHECK_ERROR(N == addend->num_elem() && N == output->num_elem(),
              "ResidualAdd shape mismatch");
  dim3 gridDim(CEIL(N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);
  residual_add<<<gridDim, blockDim>>>(input->gpu_buf, addend->gpu_buf, output->gpu_buf, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void SiLU(Tensor *inout) {
#pragma omp parallel for
  for (size_t i = 0; i < inout->num_elem(); ++i) {
    float x = inout->buf[i];
    inout->buf[i] = x / (1.0f + expf(-x));
  }
}

__global__ void silu(float* inout, size_t N) {
  size_t n = blockDim.x * blockIdx.x + threadIdx.x;
  if ( n >= N ) return;
  float x = inout[n];
  inout[n] = x / (1.0f + expf(-x));
}

//[SiLU_gpu] N: 14680064
void SiLU_gpu(Tensor *inout) {
  size_t N = inout->num_elem();
  dim3 gridDim(CEIL(N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);
  silu<<<gridDim, blockDim>>>(inout->gpu_buf, N);
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

__global__ void elementwise_mul(float *lhs, float *rhs, float*output, size_t N) {
  size_t n = blockDim.x * blockIdx.x + threadIdx.x;
  if ( n >= N ) return;
  output[n] = lhs[n] * rhs[n];
}

// [ElementwiseMul_gpu] N: 14680064
void ElementwiseMul_gpu(Tensor *lhs, Tensor *rhs, Tensor *output) {
  CHECK_ERROR(lhs->num_elem() == rhs->num_elem() &&
                  lhs->num_elem() == output->num_elem(),
              "ElementwiseMul shape mismatch");

  size_t N = lhs->num_elem();
  dim3 gridDim(CEIL(N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);
  elementwise_mul<<<gridDim, blockDim>>>(lhs->gpu_buf, rhs->gpu_buf, output->gpu_buf, N);
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

__global__ void lmhead(float* input, float* weight, float* output, size_t rows, size_t vocab_size, size_t hidden) {
  size_t ty = threadIdx.y;
  size_t tx = threadIdx.x;
  size_t row = blockIdx.y * TILE_SIZE + ty;
  size_t weight_idx = blockIdx.x * TILE_SIZE + ty;
  __shared__ float LI[TILE_SIZE][TILE_SIZE];
  __shared__ float LW[TILE_SIZE][TILE_SIZE + 1];
  input += row * hidden;
  weight += weight_idx * hidden;
  float sum = 0.0f;
  for (size_t c = 0; c < hidden; c += TILE_SIZE) {
    size_t in_idx = c + tx;
    LI[ty][tx] = (row < rows && in_idx < hidden)               ? input[in_idx] : 0.0f;
    LW[tx][ty] = (weight_idx < vocab_size && in_idx < hidden ) ? weight[in_idx] : 0.0f;
    __syncthreads();
    for (int i = 0 ; i < TILE_SIZE ; i++) {
      sum += LI[ty][i] * LW[i][tx];
    }
    __syncthreads();
  }
  if (row >= rows) return;
   size_t vocab = blockDim.x * blockIdx.x + threadIdx.x;
 if (vocab >= vocab_size) return;
  output[row * vocab_size + vocab] = sum;
}

// [LMHead_gpu] rows: 1792, vocab_size: 128256, hidden: 2048
// [LMHead_gpu] rows: 1824, vocab_size: 128256, hidden: 2048
// [LMHead_gpu] rows: 1856, vocab_size: 128256, hidden: 2048
// [LMHead_gpu] rows: 1888, vocab_size: 128256, hidden: 2048
void LMHead_gpu(Tensor *input, Tensor *weight, Tensor *output) {
  size_t rows = flat_rows(input);
  size_t hidden = last_dim(input);
  CHECK_ERROR(weight->ndim == 2 && weight->shape[1] == hidden,
              "LMHead weight shape mismatch");
  CHECK_ERROR(output->num_elem() == rows * weight->shape[0],
              "LMHead output shape mismatch");

  const size_t vocab_size = weight->shape[0];

  // TODO(student): Replace the vocab projection with GPU code.
  dim3 gridDim(CEIL(vocab_size, TILE_SIZE), CEIL(rows, TILE_SIZE));
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  lmhead<<<gridDim, blockDim>>>(input->gpu_buf, weight->gpu_buf, output->gpu_buf, rows, vocab_size, hidden);
  CHECK_CUDA(cudaDeviceSynchronize());
}
