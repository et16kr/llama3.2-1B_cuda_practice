#pragma once

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "tensor.h"

class SafetensorsLoader {
 public:
  explicit SafetensorsLoader(const char *path);
  ~SafetensorsLoader();

  SafetensorsLoader(const SafetensorsLoader &) = delete;
  SafetensorsLoader &operator=(const SafetensorsLoader &) = delete;

  bool has_tensor(const char *name) const;
  Parameter *load_parameter(const char *name,
                            const std::vector<size_t> &expected_shape) const;

 private:
  FILE *fp_ = nullptr;
  std::string header_;
  size_t data_base_ = 0;
};
