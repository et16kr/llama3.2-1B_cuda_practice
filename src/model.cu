#include "model.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "layer.h"
#include "safetensors_loader.h"
#include "util.h"

namespace {

LlamaConfig config_;

Parameter *tok_embeddings = nullptr;
std::vector<Parameter *> input_norm_weight;
std::vector<Parameter *> q_proj_weight;
std::vector<Parameter *> k_proj_weight;
std::vector<Parameter *> v_proj_weight;
std::vector<Parameter *> o_proj_weight;
std::vector<Parameter *> post_attn_norm_weight;
std::vector<Parameter *> gate_proj_weight;
std::vector<Parameter *> up_proj_weight;
std::vector<Parameter *> down_proj_weight;
Parameter *final_norm_weight = nullptr;
Parameter *lm_head_weight = nullptr;

Activation *x = nullptr;
Activation *residual = nullptr;
Activation *norm_buf = nullptr;
Activation *q_proj = nullptr;
Activation *k_proj = nullptr;
Activation *v_proj = nullptr;
Activation *q = nullptr;
Activation *k = nullptr;
Activation *v = nullptr;
Activation *att_scores = nullptr;
Activation *att_probs = nullptr;
Activation *context = nullptr;
Activation *merged = nullptr;
Activation *attn_out = nullptr;
Activation *gate_buf = nullptr;
Activation *up_buf = nullptr;
Activation *gated_buf = nullptr;
Activation *mlp_out = nullptr;
Activation *final_norm = nullptr;

size_t current_batch = 0;
size_t current_seq = 0;
TokenBatch *current_tokens = nullptr;

void delete_tensor(Tensor *&tensor) {
  if (tensor != nullptr) {
    delete tensor;
    tensor = nullptr;
  }
}

void delete_tensor_vector(std::vector<Parameter *> &tensors) {
  for (Tensor *tensor : tensors) {
    delete tensor;
  }
  tensors.clear();
}

void load_layer_parameters(SafetensorsLoader *loader, size_t layer_idx) {
  const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
  const size_t hidden = config_.hidden_size;
  const size_t kv_hidden = config_.num_key_value_heads * config_.head_dim();
  const size_t intermediate = config_.intermediate_size;

  input_norm_weight.push_back(
      loader->load_parameter((prefix + "input_layernorm.weight").c_str(), {hidden}));
  q_proj_weight.push_back(
      loader->load_parameter((prefix + "self_attn.q_proj.weight").c_str(),
                             {hidden, hidden}));
  k_proj_weight.push_back(
      loader->load_parameter((prefix + "self_attn.k_proj.weight").c_str(),
                             {kv_hidden, hidden}));
  v_proj_weight.push_back(
      loader->load_parameter((prefix + "self_attn.v_proj.weight").c_str(),
                             {kv_hidden, hidden}));
  o_proj_weight.push_back(
      loader->load_parameter((prefix + "self_attn.o_proj.weight").c_str(),
                             {hidden, hidden}));
  post_attn_norm_weight.push_back(
      loader->load_parameter((prefix + "post_attention_layernorm.weight").c_str(),
                             {hidden}));
  gate_proj_weight.push_back(
      loader->load_parameter((prefix + "mlp.gate_proj.weight").c_str(),
                             {intermediate, hidden}));
  up_proj_weight.push_back(
      loader->load_parameter((prefix + "mlp.up_proj.weight").c_str(),
                             {intermediate, hidden}));
  down_proj_weight.push_back(
      loader->load_parameter((prefix + "mlp.down_proj.weight").c_str(),
                             {hidden, intermediate}));
}

void load_parameters(const char *model_dir) {
  char config_path[512];
  char weights_path[512];
  snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
  snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);

  config_ = load_llama_config(config_path);
  SafetensorsLoader loader(weights_path);

  tok_embeddings =
      loader.load_parameter("model.embed_tokens.weight",
                            {config_.vocab_size, config_.hidden_size});

  for (size_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    load_layer_parameters(&loader, layer);
  }

  final_norm_weight =
      loader.load_parameter("model.norm.weight", {config_.hidden_size});
  if (loader.has_tensor("lm_head.weight")) {
    lm_head_weight =
        loader.load_parameter("lm_head.weight", {config_.vocab_size, config_.hidden_size});
  } else {
    lm_head_weight = nullptr;
  }
}

void free_parameters() {
  delete_tensor(tok_embeddings);
  delete_tensor_vector(input_norm_weight);
  delete_tensor_vector(q_proj_weight);
  delete_tensor_vector(k_proj_weight);
  delete_tensor_vector(v_proj_weight);
  delete_tensor_vector(o_proj_weight);
  delete_tensor_vector(post_attn_norm_weight);
  delete_tensor_vector(gate_proj_weight);
  delete_tensor_vector(up_proj_weight);
  delete_tensor_vector(down_proj_weight);
  delete_tensor(final_norm_weight);
  delete_tensor(lm_head_weight);
}

void transformer_block(size_t layer_idx) {
  RMSNorm(x, input_norm_weight[layer_idx], norm_buf, config_.rms_norm_eps);

  Linear(norm_buf, q_proj_weight[layer_idx], q_proj);
  Linear(norm_buf, k_proj_weight[layer_idx], k_proj);
  Linear(norm_buf, v_proj_weight[layer_idx], v_proj);

  SplitHeads(q_proj, q, config_.num_attention_heads, config_.head_dim());
  SplitHeads(k_proj, k, config_.num_key_value_heads, config_.head_dim());
  SplitHeads(v_proj, v, config_.num_key_value_heads, config_.head_dim());
  ApplyRoPE(q, k, config_);

  AttentionScoresGrouped(q, k, att_scores, config_.num_attention_heads,
                         config_.num_key_value_heads);
  ScaleMaskSoftmax(att_scores, att_probs, config_.head_dim(), current_tokens);
  AttentionContextGrouped(att_probs, v, context, config_.num_attention_heads,
                          config_.num_key_value_heads);
  MergeHeads(context, merged);
  Linear(merged, o_proj_weight[layer_idx], attn_out);
  ResidualAdd(x, attn_out, residual);
  std::swap(x, residual);

  RMSNorm(x, post_attn_norm_weight[layer_idx], norm_buf, config_.rms_norm_eps);
  Linear(norm_buf, gate_proj_weight[layer_idx], gate_buf);
  Linear(norm_buf, up_proj_weight[layer_idx], up_buf);
  SiLU(gate_buf);
  ElementwiseMul(gate_buf, up_buf, gated_buf);
  Linear(gated_buf, down_proj_weight[layer_idx], mlp_out);
  ResidualAdd(x, mlp_out, residual);
  std::swap(x, residual);
}

void llama_forward_cpu(TokenBatch *tokens, Tensor *logits) {
  CHECK_ERROR(tokens->B == current_batch && tokens->T == current_seq,
              "Token batch shape differs from allocated activations");
  CHECK_ERROR(logits->shape[0] == tokens->B && logits->shape[1] == tokens->T &&
                  logits->shape[2] == config_.vocab_size,
              "Logits tensor shape mismatch");

  EmbeddingLookup(tokens, tok_embeddings, x);

  for (size_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    transformer_block(layer);
  }

  RMSNorm(x, final_norm_weight, final_norm, config_.rms_norm_eps);
  LMHead(final_norm, lm_head_weight != nullptr ? lm_head_weight : tok_embeddings, logits);
}

}  // namespace

TokenBatch load_tokens(const char *path) {
  FILE *f = fopen(path, "rb");
  CHECK_ERROR(f != nullptr, "Failed to open token file %s", path);

  int32_t B = 0;
  int32_t T = 0;
  CHECK_ERROR(fread(&B, sizeof(int32_t), 1, f) == 1,
              "Failed to read batch size from %s", path);
  CHECK_ERROR(fread(&T, sizeof(int32_t), 1, f) == 1,
              "Failed to read sequence length from %s", path);
  CHECK_ERROR(B > 0 && T > 0, "Invalid token shape in %s", path);

  CHECK_ERROR(fseek(f, 0, SEEK_END) == 0, "Failed to seek %s", path);
  long file_size = ftell(f);
  CHECK_ERROR(file_size >= 0, "Failed to stat %s", path);
  rewind(f);
  CHECK_ERROR(fread(&B, sizeof(int32_t), 1, f) == 1,
              "Failed to read batch size from %s", path);
  CHECK_ERROR(fread(&T, sizeof(int32_t), 1, f) == 1,
              "Failed to read sequence length from %s", path);

  TokenBatch batch((size_t)B, (size_t)T);
  const size_t tokens_bytes = (size_t)B * (size_t)T * sizeof(int32_t);
  const size_t lengths_bytes = (size_t)B * sizeof(int32_t);
  const size_t header_bytes = 2 * sizeof(int32_t);
  const size_t file_bytes = (size_t)file_size;

  bool has_lengths = false;
  if (file_bytes == header_bytes + tokens_bytes) {
    has_lengths = false;
  } else if (file_bytes == header_bytes + lengths_bytes + tokens_bytes) {
    has_lengths = true;
  } else {
    CHECK_ERROR(false, "Unsupported token file size for %s", path);
  }

  if (has_lengths) {
    CHECK_ERROR(fread(batch.lengths, sizeof(int32_t), (size_t)B, f) == (size_t)B,
                "Failed to read lengths from %s", path);
  } else {
    for (size_t b = 0; b < batch.B; ++b) {
      batch.lengths[b] = (int32_t)batch.T;
    }
  }

  size_t expected = (size_t)B * (size_t)T;
  CHECK_ERROR(fread(batch.buf, sizeof(int32_t), expected, f) == expected,
              "Failed to read token ids from %s", path);
  int trailing = fgetc(f);
  fclose(f);
  CHECK_ERROR(trailing == EOF, "Unexpected trailing bytes in token file %s", path);
  for (size_t b = 0; b < batch.B; ++b) {
    CHECK_ERROR(batch.lengths[b] > 0 && batch.lengths[b] <= (int32_t)batch.T,
                "Invalid sequence length %d in %s", batch.lengths[b], path);
  }
  batch.to_gpu();
  return batch;
}

void initialize_model(const char *model_dir) { load_parameters(model_dir); }

void alloc_activations(size_t batch_size, size_t seq_len) {
  CHECK_ERROR(batch_size > 0 && seq_len > 0, "Activation shape must be positive");
  CHECK_ERROR(seq_len <= config_.max_position_embeddings,
              "Sequence length %zu exceeds max_position_embeddings %zu", seq_len,
              config_.max_position_embeddings);

  free_activations();

  current_batch = batch_size;
  current_seq = seq_len;

  const size_t hidden = config_.hidden_size;
  const size_t kv_hidden = config_.num_key_value_heads * config_.head_dim();
  const size_t intermediate = config_.intermediate_size;

  x = new Activation({batch_size, seq_len, hidden});
  residual = new Activation({batch_size, seq_len, hidden});
  norm_buf = new Activation({batch_size, seq_len, hidden});
  q_proj = new Activation({batch_size, seq_len, hidden});
  k_proj = new Activation({batch_size, seq_len, kv_hidden});
  v_proj = new Activation({batch_size, seq_len, kv_hidden});
  q = new Activation({batch_size, config_.num_attention_heads, seq_len, config_.head_dim()});
  k = new Activation(
      {batch_size, config_.num_key_value_heads, seq_len, config_.head_dim()});
  v = new Activation(
      {batch_size, config_.num_key_value_heads, seq_len, config_.head_dim()});
  att_scores = new Activation({batch_size, config_.num_attention_heads, seq_len, seq_len});
  att_probs = new Activation({batch_size, config_.num_attention_heads, seq_len, seq_len});
  context = new Activation({batch_size, config_.num_attention_heads, seq_len, config_.head_dim()});
  merged = new Activation({batch_size, seq_len, hidden});
  attn_out = new Activation({batch_size, seq_len, hidden});
  gate_buf = new Activation({batch_size, seq_len, intermediate});
  up_buf = new Activation({batch_size, seq_len, intermediate});
  gated_buf = new Activation({batch_size, seq_len, intermediate});
  mlp_out = new Activation({batch_size, seq_len, hidden});
  final_norm = new Activation({batch_size, seq_len, hidden});
}

void llama_forward(TokenBatch *tokens, Tensor *logits) {
  current_tokens = tokens;
  llama_forward_cpu(tokens, logits);
  current_tokens = nullptr;

  // TODO(student): Replace the CPU path with GPU kernels layer by layer.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void validate_against_cpu(TokenBatch *tokens, Tensor *logits_gpu) {
  Tensor reference({tokens->B, tokens->T, config_.vocab_size});
  current_tokens = tokens;
  llama_forward_cpu(tokens, &reference);
  current_tokens = nullptr;

  int diff = validate_buffer(logits_gpu->buf, reference.buf, reference.num_elem(), 1e-3f,
                             1e-3f);
  if (diff < 0) {
    printf("Validation: PASSED\n");
    return;
  }

  printf("Validation: FAILED\n");
  printf("First mismatch at index %d: output=%f reference=%f\n", diff,
         logits_gpu->buf[diff], reference.buf[diff]);
  EXIT(EXIT_FAILURE);
}

void finalize_model() { free_parameters(); }

void free_activations() {
  delete_tensor(x);
  delete_tensor(residual);
  delete_tensor(norm_buf);
  delete_tensor(q_proj);
  delete_tensor(k_proj);
  delete_tensor(v_proj);
  delete_tensor(q);
  delete_tensor(k);
  delete_tensor(v);
  delete_tensor(att_scores);
  delete_tensor(att_probs);
  delete_tensor(context);
  delete_tensor(merged);
  delete_tensor(attn_out);
  delete_tensor(gate_buf);
  delete_tensor(up_buf);
  delete_tensor(gated_buf);
  delete_tensor(mlp_out);
  delete_tensor(final_norm);
  current_batch = 0;
  current_seq = 0;
}

const LlamaConfig &model_config() { return config_; }
