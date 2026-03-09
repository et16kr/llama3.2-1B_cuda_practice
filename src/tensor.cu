#include "tensor.h"

#include <cstdlib>
#include <cstring>

#include "util.h"

Tensor::Tensor(const std::vector<size_t> &shape_) { reshape(shape_); }

Tensor::Tensor(const std::vector<size_t> &shape_, const float *buf_) {
  reshape(shape_);
  memcpy(buf, buf_, num_elem() * sizeof(float));
  to_gpu();
}

Tensor::~Tensor() {
  if (buf != nullptr) {
    free(buf);
  }
  if (gpu_buf != nullptr) {
    CHECK_CUDA(cudaFree(gpu_buf));
  }
}

size_t Tensor::num_elem() const {
  size_t n = 1;
  for (size_t i = 0; i < ndim; ++i) {
    n *= shape[i];
  }
  return n;
}

void Tensor::reshape(const std::vector<size_t> &shape_) {
  size_t old_numel = (buf == nullptr) ? 0 : num_elem();
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = shape_[i];
  }
  for (size_t i = ndim; i < 5; ++i) {
    shape[i] = 1;
  }

  size_t new_numel = num_elem();
  if (buf == nullptr) {
    buf = (float *)calloc(new_numel, sizeof(float));
    CHECK_CUDA(cudaMalloc((void **)&gpu_buf, new_numel * sizeof(float)));
    CHECK_CUDA(cudaMemset(gpu_buf, 0, new_numel * sizeof(float)));
    return;
  }

  CHECK_ERROR(old_numel == new_numel,
              "reshape changes tensor size (%zu -> %zu), which is not allowed",
              old_numel, new_numel);
}

void Tensor::to_gpu() const {
  CHECK_ERROR(buf != nullptr && gpu_buf != nullptr,
              "Tensor buffers must be allocated before to_gpu()");
  CHECK_CUDA(
      cudaMemcpy(gpu_buf, buf, num_elem() * sizeof(float), cudaMemcpyHostToDevice));
}

void Tensor::to_cpu() const {
  CHECK_ERROR(buf != nullptr && gpu_buf != nullptr,
              "Tensor buffers must be allocated before to_cpu()");
  CHECK_CUDA(
      cudaMemcpy(buf, gpu_buf, num_elem() * sizeof(float), cudaMemcpyDeviceToHost));
}

void Tensor::zero_host() const {
  memset(buf, 0, num_elem() * sizeof(float));
}

void Tensor::zero_device() const {
  CHECK_CUDA(cudaMemset(gpu_buf, 0, num_elem() * sizeof(float)));
}

TokenBatch::TokenBatch(size_t batch_size, size_t seq_len)
    : B(batch_size), T(seq_len), n_elem(batch_size * seq_len) {
  buf = (int32_t *)malloc(n_elem * sizeof(int32_t));
  CHECK_ERROR(buf != nullptr, "Failed to allocate host token buffer");
  CHECK_CUDA(cudaMalloc((void **)&gpu_buf, n_elem * sizeof(int32_t)));
  lengths = (int32_t *)malloc(B * sizeof(int32_t));
  CHECK_ERROR(lengths != nullptr, "Failed to allocate host length buffer");
  CHECK_CUDA(cudaMalloc((void **)&gpu_lengths, B * sizeof(int32_t)));
}

TokenBatch::~TokenBatch() {
  if (buf != nullptr) {
    free(buf);
  }
  if (gpu_buf != nullptr) {
    CHECK_CUDA(cudaFree(gpu_buf));
  }
  if (lengths != nullptr) {
    free(lengths);
  }
  if (gpu_lengths != nullptr) {
    CHECK_CUDA(cudaFree(gpu_lengths));
  }
}

TokenBatch::TokenBatch(TokenBatch &&other) noexcept {
  B = other.B;
  T = other.T;
  n_elem = other.n_elem;
  buf = other.buf;
  gpu_buf = other.gpu_buf;
  lengths = other.lengths;
  gpu_lengths = other.gpu_lengths;
  other.B = 0;
  other.T = 0;
  other.n_elem = 0;
  other.buf = nullptr;
  other.gpu_buf = nullptr;
  other.lengths = nullptr;
  other.gpu_lengths = nullptr;
}

TokenBatch &TokenBatch::operator=(TokenBatch &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  if (buf != nullptr) {
    free(buf);
  }
  if (gpu_buf != nullptr) {
    CHECK_CUDA(cudaFree(gpu_buf));
  }
  if (lengths != nullptr) {
    free(lengths);
  }
  if (gpu_lengths != nullptr) {
    CHECK_CUDA(cudaFree(gpu_lengths));
  }

  B = other.B;
  T = other.T;
  n_elem = other.n_elem;
  buf = other.buf;
  gpu_buf = other.gpu_buf;
  lengths = other.lengths;
  gpu_lengths = other.gpu_lengths;

  other.B = 0;
  other.T = 0;
  other.n_elem = 0;
  other.buf = nullptr;
  other.gpu_buf = nullptr;
  other.lengths = nullptr;
  other.gpu_lengths = nullptr;
  return *this;
}

void TokenBatch::to_gpu() const {
  CHECK_CUDA(cudaMemcpy(gpu_buf, buf, n_elem * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(gpu_lengths, lengths, B * sizeof(int32_t), cudaMemcpyHostToDevice));
}

void TokenBatch::to_cpu() const {
  CHECK_CUDA(cudaMemcpy(buf, gpu_buf, n_elem * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(lengths, gpu_lengths, B * sizeof(int32_t), cudaMemcpyDeviceToHost));
}
