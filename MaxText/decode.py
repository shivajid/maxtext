"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Training loop and Decoding of the model."""
import functools
from typing import Sequence

import datetime
import flax
import orbax

import os
from absl import app
import numpy as np

import pyconfig
import max_utils
import inference_utils
from input_pipeline.input_pipeline_interface import create_data_iterator_with_tokenizer
from layers import models

import common_types

import jax
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

import max_logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
cc.initialize_cache(os.path.expanduser("~/jax_cache"))

Transformer = models.Transformer

def match_input_and_output_stream(prompt, outputs, tokenizer):
  for i in range(len(prompt)):
    prompt_mini = prompt[0:i+1]
    prompt_mini_arr = np.array(prompt_mini, dtype=np.int32)
    prompt_mini_str = decode_tokens(prompt_mini_arr, tokenizer)
    output_mini = outputs[i:i+1]
    output_mini_arr = np.array(output_mini, dtype=np.int32)
    output_mini_str = decode_tokens(output_mini_arr, tokenizer)
    print(f"{prompt_mini_str} -> {output_mini_str}")


def decode_tokens(toks, tokenizer):
  return tokenizer.detokenize(toks).numpy().decode("utf-8"), len(toks)


def encode_strings(strs, max_len, tokenizer, mesh):
  """Pack prefill prompts into Jax.Array. The prompts are `right-aligned`, i.e. padded with zeros and all ending on the same
     index."""
  tokenized_batch = np.zeros((len(strs), max_len), np.int32)
  positions = np.zeros((len(strs), max_len), np.int32)
  segment_ids = np.zeros((len(strs), max_len), np.int32)
  for i, s in enumerate(strs):
    prompt = tokenizer.tokenize(s).numpy()
    assert prompt.shape[0] <= max_len, f"We aren't able to tokenize input {i}, len:{prompt.shape[0]} > max:{max_len}"
    start_index = max_len - prompt.shape[0]
    tokenized_batch[i, start_index:] = prompt
    padded_start_index = start_index
    segment_ids[i, padded_start_index:] = common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    positions[i, padded_start_index:] = np.arange(len(prompt))
  return jax.device_put(tokenized_batch, jax.sharding.NamedSharding(mesh, P())),\
         jax.device_put(positions, jax.sharding.NamedSharding(mesh, P())),\
         jax.device_put(segment_ids, jax.sharding.NamedSharding(mesh, P()))


def prefill_predict_step(inputs, input_positions, decoder_segment_ids,
                 state,
                 rngkey,
                 model=None):
  """Prefill KV Cache and output logits"""
  flat_logits, new_vars = model.apply(
    {
        "params": state.params
    },
    inputs,
    input_positions,
    decoder_segment_ids=decoder_segment_ids,
    enable_dropout=False,
    model_mode=common_types.MODEL_MODE_PREFILL,
    rngs={'aqt': rngkey},
    mutable=["cache"]
  )
  return flat_logits, new_vars['cache']


def ar_predict_single_token(previous_logits, token_position, kv_cache, state, rngkey, model, config):
  """Predict one token, return new cache"""
  new_token = inference_utils.sample_logits(previous_logits, rngkey, config=config)
  flat_logits, new_vars = model.apply(
    {
      "params": state.params,
      "cache": kv_cache
    },
    new_token,
    token_position,
    enable_dropout=False,
    model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
    rngs={'aqt': rngkey},
    mutable=["cache"])
  new_flat_cache = new_vars["cache"]
  return token_position + 1, new_flat_cache, flat_logits, new_token


def save_prefill_cache(config, prefill_cache, prefill_last_logit, prefill_decoder_pos):
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  blob = {"cache":prefill_cache, "last_logit":prefill_last_logit, "pos":prefill_decoder_pos}
  orbax_checkpointer.save(config.prefill_cache_dir, max_utils.unbox_logicallypartioned(blob))
  max_logging.log(f"Wrote prefill cache to {config.prefill_cache_dir}")


def compute_prefill_cache(config, model, state, rng, sp_tokenizer, mesh, 
                          state_mesh_shardings, kv_cache_mesh_shardings, replicated_sharding):
  """Compute the necessary prefill state."""
  partial_prefill_predict_step = functools.partial(prefill_predict_step, model=model)
  p_prefill_predict_step = jax.jit(
      partial_prefill_predict_step,
      in_shardings=(replicated_sharding, replicated_sharding, replicated_sharding, state_mesh_shardings, None),
      out_shardings=(replicated_sharding, kv_cache_mesh_shardings)
  )
  # Encode the demo prompt -- to measure performance we encode it multiple times.
  tokenized_prompt = [config.prompt] * int(config.per_device_batch_size * jax.device_count())
  tokenized_prompts, prompt_decoder_positions, prompt_decoder_segment_ids  = encode_strings(tokenized_prompt,
      config.max_prefill_predict_length, sp_tokenizer, mesh)
  prefill_output, prefill_cache = p_prefill_predict_step(tokenized_prompts, prompt_decoder_positions,
                                                         prompt_decoder_segment_ids, state, rng)
  max_logging.log(f"Computed prefill cache {config.prefill_cache_dir}")

  prefill_last_logit = prefill_output[:, -1:]
  prefill_decoder_pos = prompt_decoder_positions[:, -1:] + 1
  if config.prefill_cache_dir != "":
    save_prefill_cache(config, prefill_cache, prefill_last_logit, prefill_decoder_pos)
  return prefill_cache, prefill_last_logit, prefill_decoder_pos


def load_prefill_cache(config, mesh, kv_cache_mesh_shardings):
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  kv_cache_restore_args = jax.tree_map(lambda sharding: orbax.checkpoint.type_handlers.ArrayRestoreArgs(sharding=sharding),
                                        kv_cache_mesh_shardings)
  next_token_restore_args = orbax.checkpoint.type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=P(None))
  pos_restore_args = orbax.checkpoint.type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=P(None))
  restore_args = {"cache": kv_cache_restore_args, "next_token": next_token_restore_args, "pos": pos_restore_args}
  blob = orbax_checkpointer.restore(config.prefill_cache_dir, restore_args=restore_args)
  blob["cache"] = jax.tree_map(lambda x : 
    flax.linen.spmd.LogicallyPartitioned(x, mesh.axis_names), blob["cache"])
  max_logging.log(f"Restored prefill cache from {config.prefill_cache_dir}")
  return blob["cache"], blob["last_logit"], blob["pos"]


def compute_or_load_prefill_cache(config, model, state, rng, sp_tokenizer, mesh, 
                                  state_mesh_shardings, kv_cache_mesh_shardings, replicated_sharding):
  """We either load the necessary prefill state or generate it.  """
  if config.load_from_prefill_dir:
    return load_prefill_cache(config, mesh, kv_cache_mesh_shardings)
  return compute_prefill_cache(config, model, state, rng, sp_tokenizer, 
                               mesh, state_mesh_shardings, kv_cache_mesh_shardings, replicated_sharding)


def create_partial_ar_predict_step(config, model, state_mesh_shardings, kv_cache_mesh_shardings, replicated_sharding):
  partial_ar_predict_step = functools.partial(ar_predict_single_token, model=model, config=config)
  partial_ar_predict_step.__name__ = "partial_ar_predict_step"
  p_ar_predict_step = jax.jit(
      partial_ar_predict_step,
      in_shardings=(replicated_sharding, replicated_sharding, kv_cache_mesh_shardings, state_mesh_shardings, None),
      out_shardings=(replicated_sharding, kv_cache_mesh_shardings, replicated_sharding, replicated_sharding),
      donate_argnums=2
  )
  return p_ar_predict_step


def create_shardings(mesh, state_mesh_annotations, kv_cache_annotations):
  state_mesh_shardings = jax.tree_map(
    lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  kv_cache_mesh_shardings = jax.tree_map(
    lambda p: jax.sharding.NamedSharding(mesh, p), kv_cache_annotations)
  replicated_sharding = jax.sharding.NamedSharding(mesh, P(None))
  return state_mesh_shardings, kv_cache_mesh_shardings, replicated_sharding
  

def compute_and_log_performance_metrics(config, steps, starttime, endtime, total_memory_GB):
  num_steps = len(steps)
  elapsed_time = (endtime - starttime).total_seconds()
  seqs = config.per_device_batch_size * jax.device_count()

  per_step_time = elapsed_time/num_steps
  memory_bandwidth_per_device_GB_per_sec = total_memory_GB/(elapsed_time/num_steps)/jax.device_count()
  max_logging.log(f"Did {num_steps} steps in {elapsed_time:.3f} seconds for {seqs} sequences"
                  f" with a total memory footprint of {total_memory_GB:.3f} GB")
  max_logging.log(f"Therefore, a per-generate time of {per_step_time:.4f} seconds, a throughput of {seqs/per_step_time:.1f} "
                  f"tok/s and {memory_bandwidth_per_device_GB_per_sec:.1f} GB/s/device")


def compute_and_log_memory_metrics(params, params_descriptor) -> float:
  num_params, total_bytes, bytes_per_param = max_utils.summarize_size_from_pytree(params)
  max_logging.log(f"Number of {params_descriptor} ={num_params/10**9:.3f} billion, " 
                  f"total bytes usage={total_bytes/2**30:.3f}GB, "
                  f"bytes per param={bytes_per_param:.3f}")
  return num_params, total_bytes, bytes_per_param


def log_generated_output(config, sp_tokenizer, outputs):
  new_text, _ = decode_tokens([int(x[0,0]) for x in outputs], sp_tokenizer)
  max_logging.log(f"Completion: `{config.prompt}` -> `{new_text}`")
  if config.autoregressive_decode_assert != "":
    assert new_text==config.autoregressive_decode_assert, \
    f"generated text mismatch {new_text=} {config.autoregressive_decode_assert=}"


def get_first_and_last_profiling_steps(config):
  first_profiling_step = config.max_prefill_predict_length + config.skip_first_n_steps_for_profiler
  last_profiling_step = np.clip(first_profiling_step + config.profiler_steps - 1,
                                first_profiling_step, config.max_target_length - 1)
  return first_profiling_step, last_profiling_step


def decode_loop(config, state=None):
  """Decoding loop for the Transformer model."""
  rng = random.PRNGKey(0)

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and tokenizer definition
  model = Transformer(config, mesh=mesh)
  _, sp_tokenizer = create_data_iterator_with_tokenizer(config, mesh, add_bos=True, add_eos=False)

  # Create shardings for the model across the given mesh
  state, state_mesh_annotations = max_utils.setup_decode_state(model, config, rng, mesh, None)
  kv_cache_annotations = max_utils.get_kv_cache_annotations(model, config, rng, mesh)
  state_mesh_shardings, kv_cache_mesh_shardings, replicated_sharding = create_shardings(mesh, state_mesh_annotations, kv_cache_annotations)

  prefill_cache, next_logit, new_position = compute_or_load_prefill_cache(config, model, state, rng, sp_tokenizer, mesh,
                                                                          state_mesh_shardings, 
                                                                          kv_cache_mesh_shardings, 
                                                                          replicated_sharding)

  # Collect memory usage info
  _, total_prefill_bytes, _ = compute_and_log_memory_metrics(prefill_cache, "prefill cache entries")
  _, total_state_bytes, _ = compute_and_log_memory_metrics(state.params, "state params")
  total_memory_GB = (total_prefill_bytes + total_state_bytes)/2 ** 30 
  max_logging.log(f"Total memory (for cache and params) {total_memory_GB:.3f} GB")

  new_cache = prefill_cache

  # Setup for loop
  outputs = []
  p_ar_predict_step = create_partial_ar_predict_step(config, model, state_mesh_shardings, 
                                                     kv_cache_mesh_shardings, replicated_sharding)
  
  first_profiling_step, last_profiling_step = get_first_and_last_profiling_steps(config)
  steps = range(config.max_prefill_predict_length, config.max_target_length)

  # Main AR loop
  starttime = datetime.datetime.now()
  jax.block_until_ready(new_cache)
  for step in steps:
    if step == first_profiling_step:
      max_utils.activate_profiler(config)

    # Main work 
    new_position, new_cache, next_logit, selected_id = p_ar_predict_step(next_logit, new_position, new_cache, state, rng)
    rng = jax.random.fold_in(rng, step)
    outputs.append(selected_id)

    if step == last_profiling_step:
      jax.block_until_ready(outputs)
      max_utils.deactivate_profiler(config)

  compute_and_log_performance_metrics(config, steps, starttime, datetime.datetime.now(), total_memory_GB)
  log_generated_output(config, sp_tokenizer, outputs)


def validate_config(config):
  assert config.load_full_state_path == "", "Decode doesn't operate on full states! Convert to parameter checkpoint first."\
                                            "Using generate_param_only_checkpoint."


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
  os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

  validate_config(pyconfig.config)
  decode_loop(pyconfig.config)

if __name__ == "__main__":
  app.run(main)
