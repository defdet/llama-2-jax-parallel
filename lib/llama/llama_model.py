from functools import partial
from typing import Any, NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand

from .ModelConfig import ModelConfig
from .decoder import Decoder, check_decoder, forward_decoder, init_decoder
from .embedding import check_embedding, forward_embedding, init_embedding
from .kv_cache import KVCache
from .rms_norm import check_rms_norm, forward_rms_norm, init_rms_norm
from .rotary_embedding import RotaryValues
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

class LlamaModel(NamedTuple):
    embedding: Any  # Array
    decoder: Decoder
    norm: Any  # Array

def check_llama_model(params: LlamaModel, *, model_config: ModelConfig) -> None:
    assert isinstance(params.embedding, Array)
    assert isinstance(params.decoder, Decoder)
    assert isinstance(params.norm, Array)

    check_embedding(params.embedding, model_config=model_config)
    check_decoder(params.decoder, model_config=model_config)
    check_rms_norm(params.norm, model_config=model_config)

def init_llama_model(*, key: Array, model_config: ModelConfig) -> LlamaModel:
    key0, key1 = rand.split(key)
    embedding = init_embedding(key=key0, model_config=model_config)
    decoder = init_decoder(key=key1, model_config=model_config)
    norm = init_rms_norm(model_config=model_config)
    return LlamaModel(embedding, decoder, norm)

@partial(jax.jit, static_argnames=('model_config'))
def forward_llama_model(params: LlamaModel, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    assert isinstance(seq, Array)
    assert isinstance(qk_mask, Array)
    assert seq.dtype == jnp.uint16
    assert qk_mask.dtype == jnp.bool_
    assert model_config.d_k % 2 == 0
    assert key is None or model_config.dropout_rate is not None
    devices = mesh_utils.create_device_mesh((16, ))
    device_tuple = (2, 8)

    seq_axes = (0, 2)

    sharding_tuple_seq = [1] * 3

    for axis_num, axis in enumerate(seq_axes):
        sharding_tuple_seq[axis]=device_tuple[axis_num]

    sharding_tuple_seq = tuple(sharding_tuple_seq)

    name_tuple_seq = tuple('abcdefghijklmnopqrstuvwxyz'[:3])
    mesh_seq = Mesh(devices.reshape(sharding_tuple_seq), name_tuple_seq)     
    sharding_seq = NamedSharding(mesh_seq, P(*name_tuple_seq))

    seq = forward_embedding(params.embedding, seq)

    seq, kv_cache = forward_decoder(params.decoder, seq, qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, key=key, model_config=model_config)
    seq = forward_rms_norm(params.norm, seq, model_config=model_config)
    return seq, kv_cache
