import wandb
import pickle
import fire
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, MistralForCausalLM
from lib.proc_init_utils import initialise_tpu

import einops as op
from functools import partial
import jax
from jax import Array
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
import jax.random as rand
import math
import optax
import signal
import time
from transformers import LlamaTokenizer
from tqdm import tqdm
from typing import Any, Callable
import wandb
from lib.loss import cross_entropy_loss
from datasets import load_from_disk
from functools import partial
import torch
from lib.data import TrainData
from lib.dataloader import LlamaDataLoader
from lib.gsm_data import GSMDataset, gsm_collate_fn_train
from lib.llama import Llama, RotaryValues, forward_llama, init_llama, make_rotary_values
from lib.llama import model_config_llama2_7B as model_config
from lib.loss import cross_entropy_loss
from lib.multihost_utils import shard_model_params
from lib.param_utils import load_params, save_params

def set_save_params_signal():
    signal.signal(signal.SIGINT, save_params_signal_handler)
    signal.signal(signal.SIGTERM, save_params_signal_handler)

def unset_save_params_signal():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

def save_params(params: Any, filename: str) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(params, file=f)    
    
def save_params_to_disk(save_path) -> None:
    unset_save_params_signal()
    gathered_params = process_allgather(params)
    if is_process_0:
        save_params(gathered_params, save_path)
    set_save_params_signal()

def save_params_signal_handler(signum, frame):
    save_params_to_disk()
    print(f'Signal {signum} received. Model params have been successfully saved to disk.')
    exit(-1)

@jax.value_and_grad
def train_forward(params: Llama, rotary_values: RotaryValues, data_batch: Tuple, *, key: Array):
    seq, seq_mask, labels, labels_mask = data_batch
    
    qk_mask = op.rearrange(jnp.tril(op.einsum(seq_mask, seq_mask, 'B L1, B L2 -> B L1 L2')), 'B L1 L2 -> B 1 1 L1 L2')  # causal QK mask
    logits, _ = forward_llama(params, seq, qk_mask, rotary_values=rotary_values, key=key, model_config=model_config)
    loss = cross_entropy_loss(logits, labels, mask=labels_mask)
    return loss

@jax.jit
def train_step(params: Any, opt_state: Any, rotary_values: RotaryValues, total_loss: Array, data_batch: Tuple, key: Array) -> tuple[Any, Any, Array, Array, Array]:
    key, subkey = rand.split(key)
    loss, grads = train_forward(params, rotary_values, data_batch, key=subkey)
    total_loss += loss
    updates, opt_state = optimize(grads, opt_state, params)  # type: ignore
    params = optax.apply_updates(params, updates)
    return params, opt_state, total_loss, loss, key

def collate_fn_train(eos_id, pad_id, batch):
    # Batch: list(dict(input_ids (with inserted eos, bos)), attention_mask)
    
    input_ids = jnp.asarray([sample["input_ids"] for sample in batch]).astype(jnp.uint16)
    seq_mask = jnp.asarray([sample["attention_mask"] for sample in batch]).astype(jnp.bool_)
    
    labels = jnp.roll(input_ids, -1, axis=-1)
    labels = labels.at[:, -1].set(pad_id) # EOS token should attend to nothing

    labels_mask = jnp.roll(seq_mask, -1, axis=-1)
    labels_mask = labels_mask.at[:, -1].set(pad_id) # EOS token should attend to nothing

    return input_ids, seq_mask, labels, labels_mask 

def upload_to_hf(repo_id, path):
    from huggingface_hub import HfApi, login
    api = HfApi()
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path,
        repo_id=repo_id,
        repo_type="model",
    )
def report_to_wandb(start_time, opt_state, loss):
            wandb.log({'train loss': loss.item(), 'time': time.time() - start_time})

def main(
    tokenizer_path='Qwen/Qwen1.5-1.8B',
    dataset_path='wiki_tokenized_2.hf',
    params_path='qwen-1.8B.pickle',
    save_path='qwen-1.8B-wiki.pickle',
    repo_to_upload='Defetya/qwen-1.8B-jax-wiki',
    project_name='qwen-new-finetuning-wiki',
    batch_size=4,
    initial_lr=1e-6,
    final_lr=1e-4,
    n_epochs=1,
    num_samples=-1,
    used_length=1028,
    shuffle=True,
    use_wandb=True,
    log_steps=10):

    global is_process_0, params, optimize
    seed = 3407

    jax.distributed.initialize()
    is_process_0 = jax.process_index() == 0
    collate_fn = partial(collate_fn_train, 2, 2)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    cpu_device = jax.local_devices(None, 'cpu')[0]
    with jax.default_device(cpu_device):
        params = load_params(params_path)
    params = shard_model_params(params)
    dataset = load_from_disk(dataset_path)
    
    
    if num_samples == -1:
        subset_indices = list(range(0, len(dataset['train'])))
    else:
        subset_indices = list(range(0, num_samples))

    dataset_subset = torch.utils.data.Subset(dataset['train'], subset_indices)
    training_loader = torch.utils.data.DataLoader(dataset_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    
    set_save_params_signal()
    if is_process_0:
        wandb.init(project=project_name, config=dict(learning_rate=initial_lr, batch_size=batch_size, n_epochs=n_epochs, optimiser='adamw'))

    n_steps = math.ceil(len(training_loader))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=initial_lr,
        peak_value=final_lr,
        warmup_steps=n_steps,
        decay_steps=n_steps + 1,
        end_value=final_lr,
    )
    optimizer = optax.adamw(learning_rate=schedule)
    key = rand.key(seed, impl='rbg')
    optimize = optimizer.update
    opt_state = optimizer.init(params)

    rotary_values = make_rotary_values(None, batch_size, used_length, model_config=model_config)

    for _ in range(n_epochs):
        pbar = tqdm(total=len(training_loader))
        step_loss = 0.0
        total_loss = jnp.zeros(())
        for step, data_batch in enumerate(training_loader):
            start_time = time.time()
            params, opt_state, total_loss, loss, key = train_step(params, opt_state, rotary_values, total_loss, data_batch, key)
            if is_process_0:
                pbar.update()
                if step % log_steps == 0:
                    jax.debug.callback(report_to_wandb, start_time, opt_state, loss)
        if is_process_0:
            wandb.log({'epoch loss': total_loss.item() / (step + 1)})
            print('epoch loss, ', total_loss.item() / (step + 1))
    save_params_to_disk(save_path)
    upload_to_hf(repo_to_upload, save_path)

if __name__ == '__main__':
    fire.Fire(main)
