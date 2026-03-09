#pragma once

#include <cstddef>
#include <vector>

struct LlamaConfig {
  size_t vocab_size = 0;
  size_t hidden_size = 0;
  size_t intermediate_size = 0;
  size_t num_hidden_layers = 0;
  size_t num_attention_heads = 0;
  size_t num_key_value_heads = 0;
  size_t max_position_embeddings = 0;
  float rms_norm_eps = 1.0e-5f;
  float rope_theta = 500000.0f;
  int bos_token_id = 0;
  int pad_token_id = -1;
  std::vector<int> eos_token_ids;

  size_t head_dim() const { return hidden_size / num_attention_heads; }

  bool is_eos(int token_id) const;
  int primary_eos_token_id() const;
};

LlamaConfig load_llama_config(const char *path);
