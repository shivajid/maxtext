import argparse
import jax
import numpy as np
import os
import orbax.checkpoint as ocp
import sys

from etils import epath
from etils import epath
from jax import sharding
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import type_handlers
from typing import cast

pytree = {
      'a': np.arange(32).reshape((4, 8)) * 1,
      'b': np.arange(16).reshape((4, 4)) * 2,
      'c': {
          'a': np.arange(32).reshape((4, 8)) * 3,
          'e': np.arange(16).reshape((4, 4)) * 4,
      },
  }


def is_leaf(x):
  return (
      isinstance(x, np.ndarray)
      or isinstance(x, jax.sharding.Mesh)
      or isinstance(x, jax.sharding.PartitionSpec)
      or isinstance(x, pytree_checkpoint_handler.ParamInfo)
  )


def create_sharded_array(arr, mesh, mesh_axes):
  """Create sharded jax.Array."""
  if isinstance(arr, (int, float)):
    arr = np.asarray(arr)
  return jax.make_array_from_callback(
      arr.shape, sharding.NamedSharding(mesh, mesh_axes), lambda idx: arr[idx]
  )


def setup_sharded_pytree(
    pytree: pytree_checkpoint_handler.PyTree,
    reverse_devices: bool = False,
):
  """Creates a PyTree of sharded arrays for testing."""

  devices = jax.devices()
  num_devices = len(devices)
  if reverse_devices:
    devices = np.asarray(list(reversed(devices)))
  else:
    devices = np.asarray(devices)

  data_axis_name = 'x'
  mesh_2d = jax.sharding.Mesh(
      devices.reshape((2, num_devices // 2)), (data_axis_name, 'y')
  )
  mesh_axes_2d = jax.sharding.PartitionSpec('y')

  mesh_tree = {
      'a': mesh_2d,
      'b': mesh_2d,
      'c': {
          'a': mesh_2d,
          'e': mesh_2d,
      },
  }
  axes_tree = {
      'a': mesh_axes_2d,
      'b': mesh_axes_2d,
      'c': {
          'a': mesh_axes_2d,
          'e': mesh_axes_2d,
      },
  }

  pytree = jax.tree_util.tree_map(
      create_sharded_array, pytree, mesh_tree, axes_tree, is_leaf=is_leaf
  )
  return pytree, mesh_tree, axes_tree, data_axis_name


def main(args):

  train_state, mesh_tree, axes_tree, data_axis_name = setup_sharded_pytree(pytree)

  path = epath.Path(args.path)

  options = ocp.CheckpointManagerOptions(
      max_to_keep=3,
      save_interval_steps=2
  )
  mngr = ocp.CheckpointManager(
      path,
      ocp.PyTreeCheckpointer(),
      options=options
  )

  @jax.jit
  def train_fn(state):
    return jax.tree_util.tree_map(lambda x: x + 1, state)

  # num_steps = 2
  # for step in range(num_steps):
  #   train_state = train_fn(train_state)
  #   mngr.save(step, train_state)
  mngr.save(0, train_state)
  print('state after training', train_state)
  # restored = mngr.restore(mngr.latest_step())
  # print(restored)

  empty_state =  jax.tree_util.tree_map(np.zeros_like, train_state)
  empty_state = jax.tree_util.tree_map(
      create_sharded_array,
      empty_state,
      mesh_tree,
      axes_tree,
      is_leaf=is_leaf
  )

  if args.restore_method == 'singleslice':
    type_handlers.register_type_handler(jax.Array,
                                      type_handlers.SingleSliceArrayHandler(),
                                      override=True)
    print('Restoring with single slice')
    def _create_restore_args(data, mesh, pspec):
          return type_handlers.SingleSliceArrayRestoreArgs(
              sharding=jax.sharding.NamedSharding(mesh, pspec),
              single_slice_sharding=jax.sharding.NamedSharding(mesh, pspec),
              replica_axis_name=data_axis_name,
              global_shape=data.shape,
              dtype=data.dtype,
          )

    restore_args = jax.tree_util.tree_map(
      _create_restore_args,
      empty_state,
      mesh_tree,
      axes_tree
      )

    restored = mngr.restore(
      mngr.latest_step(),
      items=empty_state,
      restore_kwargs={'restore_args':
        cast(type_handlers.SingleSliceArrayRestoreArgs, restore_args)}
      )
  elif args.restore_method == 'arrayrestore':
    print('Restoring with ArrayRestoreArgs')
    def map_to_pspec(data, mesh, pspec):
      return type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=pspec)

    restore_args = jax.tree_util.tree_map(
      map_to_pspec,
      empty_state,
      mesh_tree,
      axes_tree
      )

    restored = mngr.restore(
      mngr.latest_step(),
      empty_state,
      restore_kwargs={'restore_args': restore_args}
      )
  elif args.restore_method == 'orig':
    print('Restoring in default way')
    shardings = jax.tree_map(lambda x: x.sharding, empty_state)
    restore_args = ocp.checkpoint_utils.construct_restore_args(
        empty_state, shardings)
    print('restore args', restore_args)

    restored = mngr.restore(
      mngr.latest_step(),
      items=empty_state,
      restore_kwargs={'restore_args': restore_args},
    )
  else:
    raise ValueError(f'Wrong value for restore-method is passed. Must be one '
                     'of ["singleslice", "arrayrestore", "orig"]')

  print('restored', restored)
  print('--------------')
  print(jax.tree_util.tree_map(lambda x: x.sharding, restored))


def parser(args):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--restore-method',
    type=str,
    default='restoreargs',
    help='specifies how to restore the ckpt'
    )

  parser.add_argument(
    '--path',
    type=str,
    default='/tmp/checkpoint_manager/',
    help='whether save ckpt in new folder or reuse the path'
  )

  return parser.parse_args(args)

if __name__ == '__main__':
  args = parser(sys.argv[1:])
  main(args)
