from types import EllipsisType

import jax
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import jax.experimental.mesh_utils as mesh_utils

def shard_array(arr: Array, axes: tuple | EllipsisType) -> Array:
    num_axes = 1 if isinstance(axes, EllipsisType) else len(axes)
    print('NUM AXES IS ', num_axes)
    if num_axes == 2:
        device_tuple = (2, 8)            
    elif num_axes == 3:
        device_tuple = (2, 2, 4)
    else:
        device_tuple = (16, )
    
    devices = mesh_utils.create_device_mesh((16, ))
    shape = arr.shape

    if axes is ...:
        mesh = Mesh(devices, ('a',))
        sharding = NamedSharding(mesh, P(None))
    else:
        sharding_tuple_ = [1] * len(shape)
        for axis_num, axis in enumerate(axes):
            sharding_tuple_[axis]=device_tuple[axis_num]
        sharding_tuple = tuple(sharding_tuple_)
        name_tuple = tuple('abcdefghijklmnopqrstuvwxyz'[:len(shape)])
        mesh = Mesh(devices.reshape(sharding_tuple), name_tuple)     
        sharding = NamedSharding(mesh, P(*name_tuple))

    xs = [jax.device_put(arr[i], device) for device, i in sharding.addressable_devices_indices_map(shape).items()]
    return jax.make_array_from_single_device_arrays(shape, sharding, xs)
