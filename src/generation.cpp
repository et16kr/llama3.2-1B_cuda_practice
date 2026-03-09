#include "generation.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "model.h"
#include "util.h"

namespace {

size_t effective_context_len(const CliOptions &options, const LlamaConfig &cfg) {
  size_t ctx = (options.context_len > 0) ? (size_t)options.context_len : (size_t)512;
  if (ctx > cfg.max_position_embeddings) {
    ctx = cfg.max_position_embeddings;
  }
  return ctx;
}

int pick_pad_token(const LlamaConfig &cfg) {
  if (cfg.pad_token_id >= 0) {
    return cfg.pad_token_id;
  }
  if (cfg.primary_eos_token_id() >= 0) {
    return cfg.primary_eos_token_id();
  }
  return cfg.bos_token_id;
}

std::vector<std::vector<int>> batch_to_sequences(const TokenBatch &batch,
                                                 size_t context_len) {
  std::vector<std::vector<int>> sequences(batch.B);
  for (size_t b = 0; b < batch.B; ++b) {
    size_t valid = (batch.lengths != nullptr) ? (size_t)batch.lengths[b] : batch.T;
    CHECK_ERROR(valid > 0 && valid <= batch.T, "Invalid sequence length for batch row %zu", b);
    size_t used = std::min(valid, context_len);
    sequences[b].resize(used);
    size_t start = valid - used;
    for (size_t t = 0; t < used; ++t) {
      sequences[b][t] = batch.buf[b * batch.T + start + t];
    }
  }
  return sequences;
}

TokenBatch make_padded_batch(const std::vector<std::vector<int>> &sequences,
                             size_t context_len, int pad_token_id) {
  CHECK_ERROR(!sequences.empty(), "sequences must not be empty");

  size_t batch_size = sequences.size();
  size_t max_len = 0;
  for (const std::vector<int> &sequence : sequences) {
    size_t used = std::min(sequence.size(), context_len);
    CHECK_ERROR(used > 0, "each sequence must contain at least one token");
    max_len = std::max(max_len, used);
  }

  TokenBatch batch(batch_size, max_len);
  for (size_t i = 0; i < batch.n_elem; ++i) {
    batch.buf[i] = pad_token_id;
  }

  for (size_t b = 0; b < batch_size; ++b) {
    size_t used = std::min(sequences[b].size(), context_len);
    size_t start = sequences[b].size() - used;
    batch.lengths[b] = (int32_t)used;
    for (size_t t = 0; t < used; ++t) {
      batch.buf[b * max_len + t] = sequences[b][start + t];
    }
  }

  batch.to_gpu();
  return batch;
}

int argmax_last_token(const Tensor &logits, size_t batch_idx, size_t seq_len,
                      size_t valid_len, size_t vocab_size) {
  const float *row =
      logits.buf + ((batch_idx * seq_len) + (valid_len - 1)) * vocab_size;
  size_t best = 0;
  for (size_t i = 1; i < vocab_size; ++i) {
    if (row[i] > row[best]) {
      best = i;
    }
  }
  return (int)best;
}

void maybe_warmup_batch(const CliOptions &options,
                        const std::vector<std::vector<int>> &sequences,
                        size_t context_len, int pad_token_id, bool *did_warmup) {
  if (!options.run_warmup || *did_warmup) {
    return;
  }

  TokenBatch batch = make_padded_batch(sequences, context_len, pad_token_id);
  alloc_activations(batch.B, batch.T);
  Tensor logits({batch.B, batch.T, model_config().vocab_size});
  llama_forward(&batch, &logits);
  *did_warmup = true;
}

void write_token_sequences(const char *path, const std::vector<std::vector<int>> &sequences,
                           int pad_token_id) {
  std::ofstream output(path, std::ios::binary);
  CHECK_ERROR(output.good(), "failed to open %s for writing", path);

  int32_t B = (int32_t)sequences.size();
  int32_t T = 0;
  for (const std::vector<int> &sequence : sequences) {
    T = std::max(T, (int32_t)sequence.size());
  }

  output.write(reinterpret_cast<const char *>(&B), sizeof(int32_t));
  output.write(reinterpret_cast<const char *>(&T), sizeof(int32_t));

  std::vector<int32_t> lengths(B, 0);
  std::vector<int32_t> padded((size_t)B * (size_t)std::max(T, 0), pad_token_id);
  for (int32_t b = 0; b < B; ++b) {
    lengths[b] = (int32_t)sequences[(size_t)b].size();
    for (int32_t t = 0; t < lengths[b]; ++t) {
      padded[(size_t)b * (size_t)T + (size_t)t] = sequences[(size_t)b][(size_t)t];
    }
  }

  output.write(reinterpret_cast<const char *>(lengths.data()), sizeof(int32_t) * (size_t)B);
  if (T > 0) {
    output.write(reinterpret_cast<const char *>(padded.data()),
                 sizeof(int32_t) * padded.size());
  }
  CHECK_ERROR(output.good(), "failed to write generated token file %s", path);
}

}  // namespace

void run_generation_mode(const CliOptions &options) {
  TokenBatch prompts = load_tokens(options.token_input_path.c_str());
  const size_t context_len = effective_context_len(options, model_config());
  const int pad_token_id = pick_pad_token(model_config());

  std::vector<std::vector<int>> sequences = batch_to_sequences(prompts, context_len);
  std::vector<std::vector<int>> completions(sequences.size());
  std::vector<bool> finished(sequences.size(), false);
  bool did_warmup = false;

  maybe_warmup_batch(options, sequences, context_len, pad_token_id, &did_warmup);

  double total_elapsed = 0.0;
  for (int step = 0; step < options.max_new_tokens; ++step) {
    bool all_finished = true;
    for (bool done : finished) {
      if (!done) {
        all_finished = false;
        break;
      }
    }
    if (all_finished) {
      break;
    }

    TokenBatch batch = make_padded_batch(sequences, context_len, pad_token_id);
    alloc_activations(batch.B, batch.T);
    Tensor logits({batch.B, batch.T, model_config().vocab_size});

    double st = get_time();
    llama_forward(&batch, &logits);
    double et = get_time();
    total_elapsed += et - st;

    if (options.run_validation) {
      validate_against_cpu(&batch, &logits);
    }

    for (size_t b = 0; b < batch.B; ++b) {
      if (finished[b]) {
        continue;
      }

      int next_token = argmax_last_token(logits, b, batch.T, (size_t)batch.lengths[b],
                                         model_config().vocab_size);
      if (model_config().is_eos(next_token)) {
        finished[b] = true;
        continue;
      }
      completions[b].push_back(next_token);
      sequences[b].push_back(next_token);
    }
  }

  size_t generated = 0;
  for (const std::vector<int> &tokens : completions) {
    generated += tokens.size();
  }
  if (generated > 0 && total_elapsed > 0.0) {
    printf("Generated %zu tokens across %zu prompts in %.6f sec (%.3f tokens/sec)\n",
           generated, completions.size(), total_elapsed, (double)generated / total_elapsed);
  }

  write_token_sequences(options.token_output_path.c_str(), completions, pad_token_id);
}

void run_forward_only_mode(const CliOptions &options) {
  printf("=============================================\n");
  printf(" Llama 3.2 1B Practice (Forward Mode)\n");
  printf("---------------------------------------------\n");
  printf(" Token input      : %s\n", options.token_input_path.c_str());
  printf(" Model dir        : %s\n", options.model_dir.c_str());
  printf(" Logits output    : %s\n", options.logits_output_path.c_str());
  printf(" Validation       : %s\n", options.run_validation ? "ON" : "OFF");
  printf(" Warm-up          : %s\n", options.run_warmup ? "ON" : "OFF");
  printf("=============================================\n\n");

  TokenBatch tokens = load_tokens(options.token_input_path.c_str());
  alloc_activations(tokens.B, tokens.T);
  Tensor logits({tokens.B, tokens.T, model_config().vocab_size});

  if (options.run_warmup) {
    llama_forward(&tokens, &logits);
  }

  double st = get_time();
  llama_forward(&tokens, &logits);
  double et = get_time();

  printf("Elapsed time: %.6f sec\n", et - st);
  printf("Throughput  : %.3f tokens/sec\n", (double)(tokens.B * tokens.T) / (et - st));

  print_last_token_topk(&logits, tokens.B, tokens.T, 5);
  write_binary(options.logits_output_path.c_str(), logits.buf,
               logits.num_elem() * sizeof(float));

  if (options.run_validation) {
    validate_against_cpu(&tokens, &logits);
  }
}
