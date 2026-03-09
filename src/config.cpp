#include "llama_config.h"

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {

template <typename T>
T read_required(const nlohmann::json &obj, const char *key) {
  if (!obj.contains(key)) {
    throw std::runtime_error(std::string("missing config key: ") + key);
  }
  return obj.at(key).get<T>();
}

template <typename T>
T read_optional(const nlohmann::json &obj, const char *key, const T &fallback) {
  if (!obj.contains(key) || obj.at(key).is_null()) {
    return fallback;
  }
  return obj.at(key).get<T>();
}

std::vector<int> read_eos_ids(const nlohmann::json &obj) {
  if (!obj.contains("eos_token_id")) {
    return {};
  }
  const nlohmann::json &value = obj.at("eos_token_id");
  if (value.is_array()) {
    return value.get<std::vector<int>>();
  }
  return {value.get<int>()};
}

}  // namespace

bool LlamaConfig::is_eos(int token_id) const {
  for (int eos_id : eos_token_ids) {
    if (token_id == eos_id) {
      return true;
    }
  }
  return false;
}

int LlamaConfig::primary_eos_token_id() const {
  return eos_token_ids.empty() ? -1 : eos_token_ids.front();
}

LlamaConfig load_llama_config(const char *path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error(std::string("failed to open config: ") + path);
  }

  nlohmann::json json;
  input >> json;

  LlamaConfig cfg;
  cfg.vocab_size = read_required<size_t>(json, "vocab_size");
  cfg.hidden_size = read_required<size_t>(json, "hidden_size");
  cfg.intermediate_size = read_required<size_t>(json, "intermediate_size");
  cfg.num_hidden_layers = read_required<size_t>(json, "num_hidden_layers");
  cfg.num_attention_heads = read_required<size_t>(json, "num_attention_heads");
  cfg.num_key_value_heads =
      read_optional<size_t>(json, "num_key_value_heads", cfg.num_attention_heads);
  cfg.max_position_embeddings =
      read_required<size_t>(json, "max_position_embeddings");
  cfg.rms_norm_eps = read_optional<float>(json, "rms_norm_eps", 1.0e-5f);
  cfg.rope_theta = read_optional<float>(json, "rope_theta", 10000.0f);
  cfg.bos_token_id = read_optional<int>(json, "bos_token_id", 0);
  cfg.pad_token_id = read_optional<int>(json, "pad_token_id", -1);
  cfg.eos_token_ids = read_eos_ids(json);

  if (cfg.vocab_size == 0 || cfg.hidden_size == 0 || cfg.intermediate_size == 0 ||
      cfg.num_hidden_layers == 0 || cfg.num_attention_heads == 0 ||
      cfg.num_key_value_heads == 0 || cfg.max_position_embeddings == 0) {
    throw std::runtime_error("config contains zero-sized dimensions");
  }
  if (cfg.hidden_size % cfg.num_attention_heads != 0) {
    throw std::runtime_error("hidden_size must be divisible by num_attention_heads");
  }
  if ((cfg.hidden_size / cfg.num_attention_heads) % 2 != 0) {
    throw std::runtime_error("head_dim must be even for RoPE");
  }
  if (cfg.num_attention_heads % cfg.num_key_value_heads != 0) {
    throw std::runtime_error(
        "num_attention_heads must be divisible by num_key_value_heads");
  }
  return cfg;
}
