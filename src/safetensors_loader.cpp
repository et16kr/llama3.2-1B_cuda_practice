#include "safetensors_loader.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "util.h"

namespace {

struct TensorInfo {
  std::string dtype;
  std::vector<size_t> shape;
  size_t begin = 0;
  size_t end = 0;
};

float fp16_to_float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  const uint32_t exp = (h >> 10) & 0x1f;
  const uint32_t mant = h & 0x03ff;
  uint32_t bits = 0;

  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = -14;
      uint32_t m = mant;
      while ((m & 0x0400) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x03ff;
      bits = sign | (uint32_t)(e + 127) << 23 | (m << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000 | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }

  float value = 0.0f;
  memcpy(&value, &bits, sizeof(float));
  return value;
}

void skip_ws(const std::string &text, size_t *pos) {
  while (*pos < text.size() &&
         std::isspace(static_cast<unsigned char>(text[*pos]))) {
    ++(*pos);
  }
}

std::string parse_quoted(const std::string &text, size_t *pos) {
  skip_ws(text, pos);
  CHECK_ERROR(*pos < text.size() && text[*pos] == '"',
              "Invalid JSON string at offset %zu", *pos);
  ++(*pos);
  size_t start = *pos;
  while (*pos < text.size() && text[*pos] != '"') {
    CHECK_ERROR(text[*pos] != '\\', "Escaped JSON strings are not supported");
    ++(*pos);
  }
  CHECK_ERROR(*pos < text.size(), "Unterminated JSON string");
  std::string value = text.substr(start, *pos - start);
  ++(*pos);
  return value;
}

size_t parse_uint(const std::string &text, size_t *pos) {
  skip_ws(text, pos);
  CHECK_ERROR(*pos < text.size() &&
                  std::isdigit(static_cast<unsigned char>(text[*pos])),
              "Expected unsigned integer at offset %zu", *pos);

  size_t value = 0;
  while (*pos < text.size() &&
         std::isdigit(static_cast<unsigned char>(text[*pos]))) {
    value = value * 10 + (size_t)(text[*pos] - '0');
    ++(*pos);
  }
  return value;
}

std::vector<size_t> parse_uint_array(const std::string &text, size_t *pos) {
  skip_ws(text, pos);
  CHECK_ERROR(*pos < text.size() && text[*pos] == '[',
              "Expected array at offset %zu", *pos);
  ++(*pos);

  std::vector<size_t> values;
  while (true) {
    skip_ws(text, pos);
    CHECK_ERROR(*pos < text.size(), "Unterminated array");
    if (text[*pos] == ']') {
      ++(*pos);
      break;
    }
    values.push_back(parse_uint(text, pos));
    skip_ws(text, pos);
    CHECK_ERROR(*pos < text.size(), "Unterminated array");
    if (text[*pos] == ',') {
      ++(*pos);
      continue;
    }
    CHECK_ERROR(text[*pos] == ']', "Expected ',' or ']' in array");
  }
  return values;
}

TensorInfo parse_tensor_info(const std::string &header, const char *name) {
  std::string needle = "\"" + std::string(name) + "\"";
  size_t key_pos = header.find(needle);
  CHECK_ERROR(key_pos != std::string::npos, "Missing tensor %s in safetensors",
              name);

  size_t pos = key_pos + needle.size();
  skip_ws(header, &pos);
  CHECK_ERROR(pos < header.size() && header[pos] == ':',
              "Malformed tensor entry for %s", name);
  ++pos;
  skip_ws(header, &pos);
  CHECK_ERROR(pos < header.size() && header[pos] == '{',
              "Expected object for %s", name);
  ++pos;

  TensorInfo info;
  while (true) {
    skip_ws(header, &pos);
    CHECK_ERROR(pos < header.size(), "Unterminated tensor object for %s",
                name);
    if (header[pos] == '}') {
      ++pos;
      break;
    }

    std::string field = parse_quoted(header, &pos);
    skip_ws(header, &pos);
    CHECK_ERROR(pos < header.size() && header[pos] == ':',
                "Expected ':' in tensor object for %s", name);
    ++pos;

    if (field == "dtype") {
      info.dtype = parse_quoted(header, &pos);
    } else if (field == "shape") {
      info.shape = parse_uint_array(header, &pos);
    } else if (field == "data_offsets") {
      std::vector<size_t> offsets = parse_uint_array(header, &pos);
      CHECK_ERROR(offsets.size() == 2,
                  "data_offsets must have size 2 for tensor %s", name);
      info.begin = offsets[0];
      info.end = offsets[1];
    } else {
      CHECK_ERROR(false, "Unsupported field %s while parsing %s", field.c_str(),
                  name);
    }

    skip_ws(header, &pos);
    CHECK_ERROR(pos < header.size(), "Unterminated tensor object for %s",
                name);
    if (header[pos] == ',') {
      ++pos;
    } else {
      CHECK_ERROR(header[pos] == '}', "Malformed tensor object for %s", name);
    }
  }

  CHECK_ERROR(info.dtype == "F32" || info.dtype == "F16" || info.dtype == "BF16",
              "Tensor %s has unsupported dtype %s", name, info.dtype.c_str());
  CHECK_ERROR(!info.shape.empty(), "Tensor %s has empty shape", name);
  CHECK_ERROR(info.end >= info.begin, "Tensor %s has invalid offsets", name);
  return info;
}

}  // namespace

SafetensorsLoader::SafetensorsLoader(const char *path) {
  fp_ = fopen(path, "rb");
  CHECK_ERROR(fp_ != nullptr, "Failed to open model file %s", path);

  uint64_t header_len = 0;
  CHECK_ERROR(fread(&header_len, sizeof(uint64_t), 1, fp_) == 1,
              "Failed to read safetensors header length");

  header_.assign((size_t)header_len, '\0');
  CHECK_ERROR(fread(&header_[0], 1, (size_t)header_len, fp_) == header_len,
              "Failed to read safetensors header");

  data_base_ = sizeof(uint64_t) + (size_t)header_len;
}

SafetensorsLoader::~SafetensorsLoader() {
  if (fp_ != nullptr) {
    fclose(fp_);
  }
}

Parameter *SafetensorsLoader::load_parameter(
    const char *name, const std::vector<size_t> &expected_shape) const {
  TensorInfo info = parse_tensor_info(header_, name);
  CHECK_ERROR(info.shape == expected_shape,
              "Tensor %s shape mismatch while loading parameter", name);

  size_t numel = 1;
  for (size_t dim : info.shape) {
    numel *= dim;
  }
  size_t elem_size = 0;
  if (info.dtype == "F32") {
    elem_size = sizeof(float);
  } else if (info.dtype == "F16" || info.dtype == "BF16") {
    elem_size = sizeof(uint16_t);
  } else {
    CHECK_ERROR(false, "Tensor %s has unsupported dtype %s", name, info.dtype.c_str());
  }
  CHECK_ERROR((info.end - info.begin) == numel * elem_size,
              "Tensor %s byte size mismatch", name);

  Parameter *param = new Parameter(expected_shape);
  CHECK_ERROR(fseek(fp_, (long)(data_base_ + info.begin), SEEK_SET) == 0,
              "Failed to seek to tensor %s", name);

  if (info.dtype == "F32") {
    CHECK_ERROR(fread(param->buf, sizeof(float), numel, fp_) == numel,
                "Failed to read tensor %s", name);
  } else {
    std::vector<uint16_t> tmp(numel);
    CHECK_ERROR(fread(tmp.data(), sizeof(uint16_t), numel, fp_) == numel,
                "Failed to read tensor %s", name);

    for (size_t i = 0; i < numel; ++i) {
      if (info.dtype == "BF16") {
        uint32_t bits = (uint32_t)tmp[i] << 16;
        float value = 0.0f;
        memcpy(&value, &bits, sizeof(float));
        param->buf[i] = value;
      } else {
        param->buf[i] = fp16_to_float(tmp[i]);
      }
    }
  }
  param->to_gpu();
  return param;
}

bool SafetensorsLoader::has_tensor(const char *name) const {
  std::string needle = "\"" + std::string(name) + "\"";
  return header_.find(needle) != std::string::npos;
}
