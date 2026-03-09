#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t status_ = call;                                              \
    if (status_ != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,     \
              cudaGetErrorName(status_), cudaGetErrorString(status_));       \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

struct Tensor {
  size_t ndim = 0;
  size_t shape[5] = {1, 1, 1, 1, 1};
  float *buf = nullptr;
  float *gpu_buf = nullptr;

  Tensor() = default;
  explicit Tensor(const std::vector<size_t> &shape_);
  Tensor(const std::vector<size_t> &shape_, const float *buf_);
  ~Tensor();

  size_t num_elem() const;
  void reshape(const std::vector<size_t> &shape_);
  void to_gpu() const;
  void to_cpu() const;
  void zero_host() const;
  void zero_device() const;
};

struct TokenBatch {
  size_t B = 0;
  size_t T = 0;
  size_t n_elem = 0;
  int32_t *buf = nullptr;
  int32_t *gpu_buf = nullptr;
  int32_t *lengths = nullptr;
  int32_t *gpu_lengths = nullptr;

  TokenBatch() = default;
  TokenBatch(size_t batch_size, size_t seq_len);
  ~TokenBatch();

  TokenBatch(const TokenBatch &) = delete;
  TokenBatch &operator=(const TokenBatch &) = delete;
  TokenBatch(TokenBatch &&other) noexcept;
  TokenBatch &operator=(TokenBatch &&other) noexcept;

  void to_gpu() const;
  void to_cpu() const;
};

typedef Tensor Parameter;
typedef Tensor Activation;
