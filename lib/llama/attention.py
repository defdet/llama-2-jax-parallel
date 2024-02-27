from functools import partial
import math
from typing import Any, NamedTuple

import einops as op
import jax
from jax import Array
import jax.nn as nn
import jax.numpy as jnp
import jax.random as rand

from .ModelConfig import ModelConfig
from .kv_cache import KVCache
from .rotary_embedding import RotaryValues, forward_rotary_embedding
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from .jax_flash_attn_tpu import flash_attention
from jax.experimental.shard_map import shard_map

class Attention(NamedTuple):
    q_proj: Any  # Array
    k_proj: Any  # Array
    v_proj: Any  # Array
    out_proj: Any  # Array

def check_attention(params: Attention, *, model_config: ModelConfig) -> None:
    assert isinstance(params.q_proj, Array)
    assert isinstance(params.k_proj, Array)
    assert isinstance(params.v_proj, Array)
    assert isinstance(params.out_proj, Array)

    assert params.q_proj.shape == (model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k)
    assert params.k_proj.shape == (model_config.d_model, model_config.n_heads_kv, model_config.d_k)
    assert params.v_proj.shape == (model_config.d_model, model_config.n_heads_kv, model_config.d_v)
    assert params.out_proj.shape == (model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model)

def init_attention(*, key: Array, model_config: ModelConfig) -> Attention:
    upper = 1. / math.sqrt(model_config.d_model)
    key0, key1, key2, key3 = rand.split(key, num=4)
    q_proj = rand.truncated_normal(key0, -upper, upper, (model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k))
    k_proj = rand.truncated_normal(key1, -upper, upper, (model_config.d_model, model_config.n_heads_kv, model_config.d_k))
    v_proj = rand.truncated_normal(key2, -upper, upper, (model_config.d_model, model_config.n_heads_kv, model_config.d_v))
    out_proj = rand.truncated_normal(key3, -upper, upper, (model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model))
    return Attention(q_proj, k_proj, v_proj, out_proj)

@partial(jax.jit, static_argnames=('model_config',))
def forward_attention(params: Attention, src_seq: Array, dst_seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    devices = mesh_utils.create_device_mesh((16, ))
    device_tuple = (2, 8)

    q_axes = (0, 2)
    k_axes = (0, 1)
    v_axes = (0, 1)
    out_axes = (0, 2)

    sharding_tuple_q = [1] * 5
    sharding_tuple_k = [1] * 4
    sharding_tuple_v = [1] * 4
    sharding_tuple_out = [1] * 3

    for axis_num, axis in enumerate(q_axes):
        sharding_tuple_q[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(k_axes):
        sharding_tuple_k[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(v_axes):
        sharding_tuple_v[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(out_axes):
        sharding_tuple_out[axis]=device_tuple[axis_num]

    sharding_tuple_q = tuple(sharding_tuple_q)
    sharding_tuple_k = tuple(sharding_tuple_k)
    sharding_tuple_v = tuple(sharding_tuple_v)
    sharding_tuple_out = tuple(sharding_tuple_out)
    
    name_tuple_q = tuple('abcdefghijklmnopqrstuvwxyz'[:5])
    mesh_q = Mesh(devices.reshape(sharding_tuple_q), name_tuple_q)     
    sharding_q = NamedSharding(mesh_q, P(*name_tuple_q))

    name_tuple_k = tuple('abcdefghijklmnopqrstuvwxyz'[:4])
    mesh_k = Mesh(devices.reshape(sharding_tuple_k), name_tuple_k)     
    sharding_k = NamedSharding(mesh_k, P(*name_tuple_k))

    name_tuple_v = tuple('abcdefghijklmnopqrstuvwxyz'[:4])
    mesh_v = Mesh(devices.reshape(sharding_tuple_v), name_tuple_v)     
    sharding_v = NamedSharding(mesh_v, P(*name_tuple_v))

    name_tuple_out = tuple('abcdefghijklmnopqrstuvwxyz'[:3])
    mesh_out = Mesh(devices.reshape(sharding_tuple_out), name_tuple_out)     
    sharding_out = NamedSharding(mesh_out, P(*name_tuple_out))

    q = op.einsum(src_seq, params.q_proj, 'B S M, M R H K -> B R H S K')
    k = op.einsum(dst_seq, params.k_proj, 'B D M, M H K -> B H D K')
    v = op.einsum(dst_seq, params.v_proj, 'B D M, M H V -> B H D V')

    q = jax.lax.with_sharding_constraint(q, sharding_q)
    k = jax.lax.with_sharding_constraint(k, sharding_k)
    v = jax.lax.with_sharding_constraint(v, sharding_v)

    q = forward_rotary_embedding(q, rotary_values=rotary_values)
    k = forward_rotary_embedding(k, rotary_values=rotary_values)

    q, k, v = map(
        lambda s: s.astype(jnp.float32),
        (q, k, v)
    )

    if kv_cache is not None:
        assert src_seq.shape[1] == 1
        assert dst_seq.shape[1] == 1
        k_cache, v_cache = kv_cache
        k = k_cache.at[:, :, -1:].set(k)
        v = v_cache.at[:, :, -1:].set(v)
    q = q.reshape(q.shape[0], model_config.n_rep_kv * model_config.n_heads_kv, q.shape[3], model_config.d_k)
    qk_mask = qk_mask.squeeze(1)
    qk_mask = jnp.broadcast_to(qk_mask, (qk_mask.shape[0], model_config.n_rep_kv * model_config.n_heads_kv, qk_mask.shape[3], qk_mask.shape[2]))
    attention_bias = jax.lax.select(
            qk_mask == True,
            jnp.full(qk_mask.shape, 0.0).astype(jnp.float32),
            jnp.full(qk_mask.shape, jnp.finfo(
                self.dtype).min).astype(jnp.float32),
        )
    specs_tuple = (P(*name_tuple_k),
                   P(*name_tuple_k),
                   P(*name_tuple_k),
                   P(*name_tuple_k))
    
    qkv = shard_map(partial(flash_attention, sm_scale=math.sqrt(model_config.d_k), debug=False), mesh=mesh_k, in_specs=specs_tuple, out_specs=P(*name_tuple_k), check_rep=False)(q, k, v, attention_bias)
    qkv = jnp.expand_dims(qkv, 1)
    qkv = qkv.astype(jnp.bfloat16)
    print(qkv.shape, 'product shape after dims expand')
    out = op.einsum(qkv, params.out_proj, 'B R H S V, R H V M -> B S M')
    out = jax.lax.with_sharding_constraint(out, sharding_out)
    
    kv_cache = None if not model_config.return_kv_cache else KVCache(k, v)

    return out, kv_cache
