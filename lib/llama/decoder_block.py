from functools import partial
import math
from typing import Any, NamedTuple

import jax
from jax import Array
import jax.random as rand

from ..rand_utils import split_key_nullable
from .ModelConfig import ModelConfig
from .attention import Attention, check_attention, forward_attention, init_attention
from .dropout import forward_dropout
from .kv_cache import KVCache
from .rms_norm import check_rms_norm, forward_rms_norm, init_rms_norm
from .rotary_embedding import RotaryValues
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

class DecoderBlock(NamedTuple):
    input_norm: Any  # Array
    attention: Attention
    post_attn_norm: Any  # Array
    gate_proj: Any  # Array
    up_proj: Any  # Array
    down_proj: Any  # Array

def check_decoder_block(params: DecoderBlock, *, model_config: ModelConfig) -> None:
    assert isinstance(params.input_norm, Array)
    assert isinstance(params.attention, Attention)
    assert isinstance(params.post_attn_norm, Array)
    assert isinstance(params.gate_proj, Array)
    assert isinstance(params.up_proj, Array)
    assert isinstance(params.down_proj, Array)

    check_rms_norm(params.input_norm, model_config=model_config)
    check_attention(params.attention, model_config=model_config)
    check_rms_norm(params.post_attn_norm, model_config=model_config)
    assert params.gate_proj.shape == (model_config.d_model, model_config.d_ff)
    assert params.up_proj.shape == (model_config.d_model, model_config.d_ff)
    assert params.down_proj.shape == (model_config.d_ff, model_config.d_model)

def init_decoder_block(*, key: Array, model_config: ModelConfig) -> DecoderBlock:
    upper = 1. / math.sqrt(model_config.d_model)
    key0, key1, key2, key3 = rand.split(key, num=4)
    input_norm = init_rms_norm(model_config=model_config)
    attention = init_attention(key=key0, model_config=model_config)
    post_attn_norm = init_rms_norm(model_config=model_config)
    gate_proj = rand.truncated_normal(key1, -upper, upper, (model_config.d_model, model_config.d_ff))
    up_proj = rand.truncated_normal(key2, -upper, upper, (model_config.d_model, model_config.d_ff))
    down_proj = rand.truncated_normal(key3, -upper, upper, (model_config.d_ff, model_config.d_model))
    return DecoderBlock(input_norm, attention, post_attn_norm, gate_proj, up_proj, down_proj)

@partial(jax.jit, static_argnames=('model_config',))
def forward_decoder_block(params: DecoderBlock, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    key0, key1, key2 = split_key_nullable(key, num=3)
    n_devices = jax.device_count()
    devices = mesh_utils.create_device_mesh((n_devices, ))
    if n_devices == 32:
        device_tuple = (4, 8)
    else:
        device_tuple = (2, n_devices // 2)

    ff_axes = (0, 2)
    seq_axes = (0, 2)

    sharding_tuple_ff = [1] * 3
    sharding_tuple_seq = [1] * 3

    for axis_num, axis in enumerate(ff_axes):
        sharding_tuple_ff[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(seq_axes):
        sharding_tuple_seq[axis]=device_tuple[axis_num]

    sharding_tuple_ff = tuple(sharding_tuple_ff)
    sharding_tuple_seq = tuple(sharding_tuple_seq)

    name_tuple_ff = tuple('abcdefghijklmnopqrstuvwxyz'[:3])
    mesh_ff = Mesh(devices.reshape(sharding_tuple_ff), name_tuple_ff)     
    sharding_ff = NamedSharding(mesh_ff, P(*name_tuple_ff))

    name_tuple_seq = tuple('abcdefghijklmnopqrstuvwxyz'[:3])
    mesh_seq = Mesh(devices.reshape(sharding_tuple_seq), name_tuple_seq)     
    sharding_seq = NamedSharding(mesh_seq, P(*name_tuple_seq))
    
    seq_ = seq

    seq = forward_rms_norm(params.input_norm, seq, model_config=model_config)
    seq, kv_cache = forward_attention(params.attention, seq, seq, qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, model_config=model_config)
    seq = forward_dropout(seq, key=key0, model_config=model_config)
    seq += seq_

    seq_ = seq
    seq = forward_rms_norm(params.post_attn_norm, seq, model_config=model_config)
    
    ff_1 = seq @ params.gate_proj
    ff_2 = seq @ params.up_proj

    ff_1 = jax.lax.with_sharding_constraint(ff_1, sharding_ff)
    ff_2 = jax.lax.with_sharding_constraint(ff_2, sharding_ff)
    
    ff = jax.nn.silu(ff_1) * ff_2

    ff = forward_dropout(ff, key=key1, model_config=model_config)

    seq = ff @ params.down_proj
    seq = jax.lax.with_sharding_constraint(seq, sharding_seq)

    seq = forward_dropout(seq, key=key2, model_config=model_config)
    seq += seq_

    return seq, kv_cache
