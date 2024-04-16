# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batch Inference with benchmark."""

from absl import app

import datetime
import json
import os
from typing import Sequence, List

import jax

from jetstream.engine import token_utils

import numpy as np

import inference_utils
import maxengine
import max_utils
import maxtext_utils
import pyconfig


def validate_inference_config(config):
  assert config.load_full_state_path == "", "Decode doesn't operate on full states! Convert to parameter checkpoint first."\
                                            "Using generate_param_only_checkpoint."
  assert config.inference_batching_mode in ["static", "continuous"], "Inference batching mode supports only static batching and continuous batching."


def summarize_pytree_data(params, name="Params"):
  """ Generate basic metrics of a given Pytree. """
  num_params, total_param_size, avg_param_size = max_utils.summarize_size_from_pytree(params)
  num_params_in_billions = num_params / 1e9
  total_param_size_in_gb = total_param_size / 1e9
  print(f"{name} stats: \n"
        f"\tTotal number of params: {num_params_in_billions:.3f} billion \n"
        f"\tTotal memory usage: {total_param_size_in_gb:.3f} GB \n"
        f"\tAvg size: {avg_param_size:.3f} bytes\n")
  return num_params, total_param_size, avg_param_size


def load_openorca_dataset(
    dataset_path: str
) -> List[tuple[str]]:
  # Load the dataset.

  with open(dataset_path) as f:
    dataset_json = json.load(f)

  # Tokenize the prompts and completions.
  prompts = dataset_json["prompts"]
  results = dataset_json["results"]

  return prompts, results


def record_run_time_in_seconds(start, end, record_time_in_seconds):
  run_time_in_seconds = (end - start).total_seconds()
  record_time_in_seconds.append(run_time_in_seconds)


def static_batching(config, engine, tokenizer, n, all_padded_tokens, true_lengths, stop_tokens):

  max_output_length = config.max_target_length - config.max_prefill_predict_length

  params = engine.load_params()
  decode_state = engine.init_decode_state()

  _, cache_size, _ = summarize_pytree_data(decode_state['cache'], name="Cache")
  num_model_params, model_size, _ = summarize_pytree_data(params, name="Model")

  num_slots = engine.max_concurrent_decodes
  num_batch = n // num_slots + 1

  profiler_batch_indices = [int(batch) for batch in config.inference_profiler_batch_indices.split(",")]
  profiler_generate_steps = config.profiler_steps

  print(f"{num_slots} slots.")
  print(f"Total {num_batch} batches.")

  slot_generation_complete = np.zeros(num_slots)
  slot_generate_result_tokens = dict()
  slot_generate_results = dict()

  generate_results = []
  all_generate_result_tokens = []

  prefills_in_seconds = []
  inserts_in_seconds = []
  generates_in_seconds = []

  for batch in range(num_batch):
    if batch in profiler_batch_indices:
      profile_generate_done = False
      max_utils.activate_profiler(config)
    start_i = num_slots * batch
    end_i = min(num_slots * (batch + 1) - 1, n - 1)

    for i in range(start_i, end_i+1):
      padded_tokens = all_padded_tokens[i]
      true_length=true_lengths[i]
      slot = i - start_i

      start = datetime.datetime.now()
      prefill_result = engine.prefill(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
      )
      jax.block_until_ready(prefill_result)
      end = datetime.datetime.now()
      record_run_time_in_seconds(start, end, prefills_in_seconds)

      start = datetime.datetime.now()
      decode_state = engine.insert(
        prefix=prefill_result,
        decode_state=decode_state,
        slot=slot
      )
      jax.block_until_ready(decode_state)
      end = datetime.datetime.now()
      record_run_time_in_seconds(start, end, inserts_in_seconds)

      inference_utils.delete_pytree(prefill_result)
      print(f"Batch {batch} prefilled and inserted sample {i}.")

    for slot in range(num_slots):

      slot_generation_complete[slot] = 0.
      slot_generate_result_tokens[slot] = []
      slot_generate_results[slot] = None

    for step in range(max_output_length):
      start = datetime.datetime.now()
      decode_state, result_tokens = engine.generate(
        params, decode_state
      )
      jax.block_until_ready(decode_state)
      end = datetime.datetime.now()
      record_run_time_in_seconds(start, end, generates_in_seconds)
      print(f"Batch {batch} generated step {step}.")

      for i in range(start_i, end_i+1):
        slot = i - start_i
        slot_data = result_tokens.get_result_at_slot(slot)
        slot_tokens = slot_data.tokens
        slot_lengths = slot_data.lengths

        token_id = slot_tokens[slot, 0].item()
        if slot_lengths > max_output_length or token_id in stop_tokens:
          slot_generation_complete[slot] = 1

        if slot_generation_complete[slot]==0:
          slot_generate_result_tokens[slot].append(token_id)
        else:
          slot_generate_results[slot] = tokenizer.detokenize(slot_generate_result_tokens[slot])

      if batch in profiler_batch_indices and (step + 1) == profiler_generate_steps:
        profile_generate_done = True
        max_utils.deactivate_profiler(config)

      if np.sum(slot_generation_complete[:end_i+1-start_i]) == 1:
        if batch in profiler_batch_indices and profile_generate_done is False:
          profile_generate_done = True
          max_utils.deactivate_profiler(config)

        if batch % config.log_period == 0:
          print(f"All generations for batch {batch} are completed at step {step}.")

        break

    for i in range(start_i, end_i+1):
      slot = i - start_i
      all_generate_result_tokens.append(slot_generate_result_tokens[slot])
      generate_results.append(slot_generate_results[slot])

    if batch % config.log_period == 0:
      print(f"Finished batch {batch} over {num_batch} batches.")

  benchmark_results = dict()

  prefill_time_in_ms = np.average(prefills_in_seconds) * 1000
  prefill_total_tflops, _, _ = maxtext_utils.calculate_tflops_prefill(num_model_params, config.max_prefill_predict_length, config)
  prefill_tflops_per_sec_per_device = prefill_total_tflops / jax.device_count() / prefill_time_in_ms * 1000.

  benchmark_results["prefill_time_in_ms"] = prefill_time_in_ms
  benchmark_results["prefill_total_tflops"] = prefill_total_tflops
  benchmark_results["prefill_tflops_per_sec_per_device"] = prefill_tflops_per_sec_per_device

  generate_time_in_seconds = np.average(generates_in_seconds)
  generate_time_in_ms = generate_time_in_seconds * 1000
  global_batch_size = jax.device_count() * config.per_device_batch_size
  generate_throughput = global_batch_size / generate_time_in_seconds
  generate_time_in_ms_per_sequence = generate_time_in_ms / global_batch_size

  GB_per_step_per_device = (model_size + cache_size) / 1e9 / jax.device_count()
  bw_per_device = GB_per_step_per_device / generate_time_in_seconds

  benchmark_results["generate_time_in_ms"] = generate_time_in_ms
  benchmark_results["global_batch_size"] = global_batch_size
  benchmark_results["generate_throughput"] = generate_throughput
  benchmark_results["generate_time_in_ms_per_sequence"] = generate_time_in_ms_per_sequence
  benchmark_results["GB_per_step_per_device"] = GB_per_step_per_device
  benchmark_results["bw_per_device"] = bw_per_device

  return generate_results, all_generate_result_tokens, benchmark_results


def inference(config):

  prompts, results = load_openorca_dataset(config.dataset_path)
  n = len(prompts)

  engine = maxengine.MaxEngine(config)
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer
  stop_tokens = [vocab.eos_id, vocab.pad_id]
  print(f"stop_tokens: {stop_tokens}")

  all_padded_tokens = []
  true_lengths = []
  for prompt in prompts:
    padded_tokens, true_length = token_utils.tokenize_and_pad(
        prompt,
        vocab,
        is_bos=True,
        prefill_lengths=[config.max_prefill_predict_length]
    )
    all_padded_tokens.append(padded_tokens)
    true_lengths.append(true_length)

  if config.inference_batching_mode == "static":
    generate_results, all_generate_result_tokens, benchmark_results = static_batching(
      config,
      engine,
      tokenizer,
      n,
      all_padded_tokens,
      true_lengths,
      stop_tokens
    )
  elif config.inference_batching_mode == "continuous":
    raise NotImplementedError("Continuous batching is not implemented yet.")

  inference_output_json = dict()
  inference_output_json["prompts"] = prompts
  inference_output_json["original_results"] = results
  inference_output_json["generate_results"] = generate_results
  inference_output_json["all_generate_result_tokens"] = all_generate_result_tokens

  print(f"\tPrefill step average time: {benchmark_results['prefill_time_in_ms']:.3f}ms\n"
        f"\tPrefill total TFLOPs: {benchmark_results['prefill_total_tflops']:.3f}\n"
        f"\tPrefill TFLOPs/sec/device: {benchmark_results['prefill_tflops_per_sec_per_device']:.3f}\n\n\n\n")

  print(f"AutoRegressive results:\n"
        f"\tAR step average time: {benchmark_results['generate_time_in_ms']:.3f}ms\n"
        f"\tAR step average time per seq: {benchmark_results['generate_time_in_ms_per_sequence']:.3f}ms\n"
        f"\tAR global batch size: {benchmark_results['global_batch_size']}\n"
        f"\tAR throughput: {benchmark_results['generate_throughput']:.3f} tokens/second\n"
        f"\tAR gb_per_step_per_device: {benchmark_results['GB_per_step_per_device']:.3f} GB\n"
        f"\tAR memory bandwidth per device: {benchmark_results['bw_per_device']:.3f} GB/s\n\n\n")

  if config.inference_output_path:
    with open(config.inference_output_path, "w", encoding="utf-8") as f:
      json.dump(inference_output_json, f)

  return


def main(argv: Sequence[str]) -> None:
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_inference_config(config)
  inference(config)


if __name__ == "__main__":
  app.run(main)
