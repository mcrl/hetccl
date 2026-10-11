import os
import time
import torch
from datasets import load_dataset
import deepspeed
import argparse
import warnings
import logging

import random
import numpy as np

# Suppress specific transformers warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
from transformers import default_data_collator, set_seed
from torch.utils.data import DataLoader, DistributedSampler

# Suppress UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.0;7.5;8.0;8.6;8.9;9.0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


import model_llama, model_mixtral, model_gpt


BSZ_PER_GPU = 1

DS_BASELINE_CFG = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.00002,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
        },
    },
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": -1, 
    },
    "steps_per_print": 1,
    "wall_clock_breakdown": False,
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": 1000,
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.00002,
            "warmup_num_steps": 100,
        },
    },
}

def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def build_wikitext_dataset(tokenizer, seq_len):
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train") # wikitext-2-raw-v1
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train") # wikitext-103-raw-v1 

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=seq_len)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len
        result = {
            k: [concatenated[k][i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k in concatenated.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    return lm_datasets

def count_parameters(model):
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_param_count(param_count):
    """Format parameter count in human readable format (M, B)"""
    if param_count >= 1e9:
        return f"{param_count/1e9:.2f}B"
    elif param_count >= 1e6:
        return f"{param_count/1e6:.2f}M"
    elif param_count >= 1e3:
        return f"{param_count/1e3:.2f}K"
    else:
        return str(param_count)


def main():
    global BSZ_PER_GPU
    
    # Fix random seed
    seed = 42
    set_seed_all(seed)

    world_size =  int(os.environ.get('OMPI_COMM_WORLD_SIZE', int(os.environ.get('WORLD_SIZE', 1))))
    rank =  int(os.environ.get('OMPI_COMM_WORLD_RANK', int(os.environ.get('RANK', 0))))
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', int(os.environ.get('LOCAL_RANK', 0))))
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lr", type=float, default=0.00002, help="Learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_iter", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 precision")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint")
    parser.add_argument("--save_loss_curve", action="store_true", default=False, help="Save loss curve")
    parser.add_argument("--loss_curve_dir", type=str, default="./loss_curves", help="Directory to save loss curve")
    parser.add_argument("--loss_curve_filename", type=str, default="loss_curve.txt", help="File name of loss curve to save")
    parser.add_argument("--save_checkpoint", action="store_true", default=False, help="Save model checkpoints during training")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    # Model & Tokenizer Load
    if args.model == "llama":
        model, tokenizer = model_llama.get_hf_llama()
        SEQ_LEN = 2048 # 8192
    elif args.model == "mixtral":
        model, tokenizer = model_mixtral.get_hf_mixtral()
        SEQ_LEN = 8192
    elif args.model == "gpt":
        model, tokenizer = model_gpt.get_hf_gpt2()
        SEQ_LEN = 1024
    else:
        print("Not supported model.")
        exit(1)

    # Tokenizer PAD token (May need for Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable use_cache before enabling gradient checkpointing
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    model.train()
    model.gradient_checkpointing_enable()

    # Print Info
    total_params, trainable_params = count_parameters(model)
    if rank == 0:
        print(f"\n=== Model Information ===")
        print(f"Model: {args.model.upper()}")
        print(f"Total parameters: {format_param_count(total_params)} ({total_params:,})")
        print(f"Trainable parameters: {format_param_count(trainable_params)} ({trainable_params:,})")
        print(f"Sequence length: {SEQ_LEN}")
        print(f"Vocab size: {model.config.vocab_size}")
        print("=" * 25)

    device=f'cuda:{local_rank}'

    ds_stage = int(os.environ.get('DS_STAGE', -1)) 
    BSZ_PER_GPU = int(os.environ.get('BSZ_PER_GPU', 1))

    ds_config = DS_BASELINE_CFG
    ds_config["train_micro_batch_size_per_gpu"] = BSZ_PER_GPU
    ds_config["gradient_accumulation_steps"] = args.grad_accum_steps
    ds_config["zero_optimization"]["stage"] = ds_stage

    ds_config["optimizer"]["params"]["lr"] = args.lr
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.lr
    ds_config["scheduler"]["params"]["total_num_steps"] = args.num_iter // args.grad_accum_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = ds_config["scheduler"]["params"]["total_num_steps"] // 10

    if args.bf16:
        ds_config["fp16"]["enabled"] = False
        ds_config["bf16"]["enabled"] = True
    elif args.fp16:
        ds_config["fp16"]["enabled"] = True
        ds_config["bf16"]["enabled"] = False
    else: # default FP16
        ds_config["fp16"]["enabled"] = True
        ds_config["bf16"]["enabled"] = False

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
    )

    # Train Dataset Setup
    if rank == 0: 
        print("Loading and processing dataset...")
    train_dataset = build_wikitext_dataset(tokenizer, SEQ_LEN)
    
    # DistributedSampler ensures different GPUs get different parts, but consistently with seed
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed, drop_last=True)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=BSZ_PER_GPU,
        collate_fn=default_data_collator,
        pin_memory=True,
        num_workers=2
    )
    
    # Iterator for DataLoader
    data_iter = iter(train_dataloader)

    # Resume from Checkpoint
    start_it = 1
    checkpoint_save_dir = args.checkpoint_dir if args.save_checkpoint else None
    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(checkpoint_save_dir)
        if load_path is not None:
            start_it = client_state['step'] + 1
            if rank == 0:
                print(f"Resuming from iteration {start_it}")
            
            # [Fast-Forward] Already trained iterations skip
            if rank == 0: print(f"Fast-forwarding dataset by {start_it - 1} steps...")
            for _ in range(start_it - 1):
                try:
                    next(data_iter)
                except StopIteration:
                    # If dataset ends, restart iterator (Infinite Loop)
                    data_iter = iter(train_dataloader)
                    next(data_iter)

    last_time = time.time()
    last_it = start_it - 1

    accumulated_loss = 0.0
    
    num_iterations = args.num_iter
    for it in range(start_it, num_iterations + 1):

        # Get Data from Iterator      
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset iterator when epoch ends
            if rank == 0: print("Epoch finished, restarting iterator...")
            train_sampler.set_epoch(it) # Shuffle seed modification 
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
            
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model_engine(input_ids=batch['input_ids'], labels=batch['labels'])
        # Accumulated loss
        accumulated_loss += output.loss.detach()
        model_engine.backward(output.loss)
        model_engine.step()

        if it % args.grad_accum_steps == 0: 
            torch.cuda.synchronize() 
            torch.distributed.barrier() 
            cur_time = time.time()
            if rank == 0:
                cosumed_tokens = world_size * BSZ_PER_GPU * SEQ_LEN * (it - last_it)
                tokens_per_sec = cosumed_tokens / (cur_time - last_time)
                peak_gpu_mem = torch.cuda.max_memory_allocated()
                time_per_it = (cur_time - last_time) / (it - last_it)
                time_per_step = cur_time - last_time
                global_step = it // args.grad_accum_steps
                avg_loss = (accumulated_loss / args.grad_accum_steps).item()
                print(f"Step: {global_step} (Iter: {it}), "
                        f"time_per_step: {time_per_step:.2f}s, "
                        f"time_per_it: {time_per_it:.2f}s, "
                        f"throughput: {tokens_per_sec:.0f} tokens/s, "
                        f"loss: {avg_loss:.3f}, "
                        f"mem: {peak_gpu_mem/1e9:.2f}GB")
            last_time = time.time()
            last_it = it
            
            # Log loss curve only when gradient accumulation step is done (i.e., model update step)
            if args.save_loss_curve and rank == 0:
                if it % args.grad_accum_steps == 0:
                    global_step = it // args.grad_accum_steps
                    os.makedirs(args.loss_curve_dir, exist_ok=True)
                    with open(os.path.join(args.loss_curve_dir, args.loss_curve_filename), 'a') as f:
                        f.write(f"{global_step}\t{avg_loss}\n")
            
            # Save checkpoint per 1000 iterations
            if args.save_checkpoint and global_step % 100 == 0:
                if rank == 0:
                    os.makedirs(checkpoint_save_dir, exist_ok=True)
                model_engine.save_checkpoint(checkpoint_save_dir, client_state={'step': it})

            accumulated_loss = 0.0

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
    print(f"Run success")