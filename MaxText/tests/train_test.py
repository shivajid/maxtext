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

""" Tests for the common Max Utils """
import jax
import max_utils
import unittest
import pyconfig;
# import train;
import layers;
import pyconfig;
import max_utils;

import train
import jax.numpy as jnp
import optax
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning

print("OK")
jax.config.update('jax_platform_name', 'cpu')


# This function was originally written for xprofing in colab.
def train_loop(cfg):
  # MaxText/configs/base.yaml
  # pyconfig.initialize(['', '/tmp/maxtext/MaxText/configs/base.yml', 'run_name=myrun', 'base_output_directory=/tmp/maxtext/output_dir', 'dataset_path=there_is_no_dataset',] + cfg)
  pyconfig.initialize(['', 'configs/base.yml', 'run_name=myrun', 'base_output_directory=/tmp/maxtext/output_dir', 'dataset_path=there_is_no_dataset',] + cfg)

  config = pyconfig.config
  init_rng, nextrng = jax.random.split(jax.random.PRNGKey(0), 2)

  model = layers.Transformer(config)
  tx = optax.adam(
      max_utils.create_learning_rate_schedule(
          learning_rate=config.learning_rate,
          total_steps=config.steps,
      ),
      b1=config.adam_b1,
      b2=config.adam_b2,
      eps=config.adam_eps,
      eps_root=config.adam_eps_root,
  )

  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  checkpoint_manager = None
  state, state_mesh_annotations = max_utils.setup_initial_state(
      model, tx, config, init_rng, mesh, checkpoint_manager
  )
  data_pspec = P(*config.data_sharding)

  # Define compiled top-level functions.
  p_train_step = pjit(
      train.train_step,
      in_shardings=(state_mesh_annotations, data_pspec, None),
      out_shardings=(state_mesh_annotations, None, None),
      static_argnums=(0,1),
      donate_argnums=2,
  )

  bs = config.per_device_batch_size * len(jax.devices())
  mtl = config.max_target_length

  # this is dummy data that I created
  example_batch = {
      'inputs': jnp.zeros((bs, mtl), dtype=jnp.int32),
      'targets': jnp.zeros((bs, mtl), dtype=jnp.int32),
      # TODO: Is this OK for xprof of training?
      'inputs_segmentation': None,
      'inputs_position': None,
  }
  n = 5
  for i in range(n):
    print(f'step {i+1}/{n}')
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state, metrics, nextrng = p_train_step(
          model, config, state, example_batch, nextrng
      )
  return state


class TrainTest(unittest.TestCase):
  def test_trainining(self):
    cfg = [
      'per_device_batch_size=1',
      'int8_training=True',
      'use_dqdg=False',  # set True for using aqt dq
      'base_mlp_dim=8',
      'base_emb_dim=4',
      'base_num_heads=2',
      'base_num_decoder_layers=2',
      'head_dim=4',
      # 'warmup_steps=3',

      # take args from base.yaml
      # base_mlp_dim: 8192 * 4
      # base_emb_dim: 2048 * 4
      # base_num_heads: 8 *2
      # base_num_decoder_layers: 16 / 8
      # head_dim: 256 *2
      # exagerate AQT.
    ]
    # train_loop(cfg)

if __name__ == '__main__':
  unittest.main()
