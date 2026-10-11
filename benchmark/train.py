import os
import time
import torch
from datasets import load_dataset
import functools
import deepspeed
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import warnings
import logging

# Suppress specific transformers warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
# Suppress UserWarnings (TORCH_CUDA_ARCH_LIST, etc.)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.0;7.5;8.0;8.6;8.9;9.0')

import model_llama, model_mixtral, model_gpt


BSZ_PER_GPU = 1

DS_BASELINE_CFG = {
    "fp16": {
        "enabled": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.00002,
            "betas": [0.9, 0.95],
            "eps": 1e-5,
            "weight_decay": 0.1,
        },
    },
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": -1,
    },
    "steps_per_print": 1,
    "wall_clock_breakdown": True,
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.00002,
            "warmup_num_steps": 2,
            "warmup_type": "linear",
        },
    },
}

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    # p.export_chrome_trace("./logs/trace_" + str(p.step_num) + ".json")

def get_random_data(batch_size, seq_len, vocab_size):
    seq = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len+1))
    input_ids = seq[:,:seq_len]
    labels = seq[:,1:seq_len+1]
    return {'input_ids': input_ids, 'labels': labels}

def profile_speed(model, device, seq_len, vocab_size, iters=3):
    """Forward+backward tokens/s of this GPU alone (micro-batch 1, no communication)."""
    model.half().to(device)
    batch = get_random_data(1, seq_len, vocab_size)
    input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
    for it in range(iters + 1):  # first iteration is warm-up
        if it == 1:
            torch.cuda.synchronize(device)
            start = time.time()
        model(input_ids=input_ids, labels=labels).loss.backward()
    torch.cuda.synchronize(device)
    model.zero_grad(set_to_none=True)
    return iters * seq_len / (time.time() - start)

def balanced_micro_batches(speed, base_bsz, world_size, device):
    """GPU-aware workload balancing: b_i proportional to speed s_i, with the slowest GPU
    getting base_bsz (i.e. b_i = B * s_i / sum_j s_j for B = base_bsz * sum_j s_j / min_j s_j)."""
    speeds = [torch.zeros(1, device=device) for _ in range(world_size)]
    torch.distributed.all_gather(speeds, torch.tensor([speed], device=device))
    speeds = [s.item() for s in speeds]
    return [max(1, round(base_bsz * s / min(speeds))) for s in speeds], speeds

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
    world_size =  int(os.environ.get('OMPI_COMM_WORLD_SIZE', -1))
    rank =  int(os.environ.get('OMPI_COMM_WORLD_RANK', -1))
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', -1))
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    # Parse model type and size (e.g., llama3.2-1b -> llama, 1b)
    model_name = args.model.lower()
    if model_name.startswith("llama"):
        # Extract size: llama3.2-1b -> 1b, llama3.2-3b -> 3b
        if "-" in model_name:
            model_size = model_name.split("-")[-1]  # Get last part after dash
        else:
            model_size = "1b"  # Default
        model, tokenizer = model_llama.get_hf_llama(model_size)
        SEQ_LEN = 8192
    elif model_name.startswith("gpt"):
        # Extract size: gpt2-125m -> 125m, gpt2-355m -> 355m
        if "-" in model_name:
            model_size = model_name.split("-")[-1]  # Get last part after dash
        else:
            model_size = "355m"  # Default to medium
        model, tokenizer = model_gpt.get_hf_gpt2(model_size)
        SEQ_LEN = model.config.n_positions # 1024
    elif model_name == "mixtral":
        model, tokenizer = model_mixtral.get_hf_mixtral()
        SEQ_LEN = 8192
    else:
        print(f"Not supported model: {args.model}")
        print("Supported models: llama3.2-1b, llama3.2-3b, gpt2-125m, gpt2-355m, mixtral")
        exit(1)

    # Disable use_cache before enabling gradient checkpointing
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    model.train()
    model.gradient_checkpointing_enable()

    # Print model information
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

    ds_stage = int(os.environ.get('DS_STAGE') or 3) # Default to stage 3
    BSZ_PER_GPU = int(os.environ.get('BSZ_PER_GPU') or 1) # Default to 1

    # Optional GPU-aware workload balancing (BALANCE=1): profile each GPU, then scale its
    # micro-batch by its speed relative to the slowest GPU (which keeps BSZ_PER_GPU).
    bszs = [BSZ_PER_GPU] * world_size
    if int(os.environ.get('BALANCE') or 0):
        prof_start = time.time()
        speed = profile_speed(model, device, SEQ_LEN, tokenizer.vocab_size)
        bszs, speeds = balanced_micro_batches(speed, BSZ_PER_GPU, world_size, device)
        BSZ_PER_GPU = bszs[rank]
        if rank == 0:
            print(f"GPU-aware balancing ({time.time() - prof_start:.1f}s profiling)")
            for r in range(world_size):
                print(f"  rank {r}: {speeds[r]:.0f} tokens/s -> micro-batch {bszs[r]}")
    # Each rank averages its loss over its own micro-batch; reweight so the gradient
    # all-reduce yields the mean over the global batch.
    loss_weight = BSZ_PER_GPU * world_size / sum(bszs)

    ds_config = DS_BASELINE_CFG
    ds_config["train_micro_batch_size_per_gpu"] = BSZ_PER_GPU
    ds_config["gradient_accumulation_steps"] = 1
    ds_config["zero_optimization"]["stage"] = ds_stage

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
    )

    fixed_batch = get_random_data(BSZ_PER_GPU, SEQ_LEN, tokenizer.vocab_size)

    last_time = time.time()
    last_it = 0

    input_ids_fixed = fixed_batch['input_ids'].to(device)
    labels_fixed = fixed_batch['labels'].to(device)

    # with torch.profiler.profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(
    #         wait=1, # During this phase profiler is not active.
    #         warmup=2, # During this phase profiler starts tracing, but the results are discarded.
    #         active=2), # During this phase profiler traces and records data.
    #     # with_stack=True, # Enable stack tracing, adds extra profiling overhead.
    #     with_flops=True,
    #     # on_trace_ready=trace_handler,
    # ) as profiler:
    num_iterations = 10
    for it in range(1, num_iterations + 1):
        input_ids = input_ids_fixed.clone().detach()
        labels = labels_fixed.clone().detach()

        output = model(input_ids=input_ids, labels=labels)
        model_engine.backward(output.loss * loss_weight)
        model_engine.step()
        # profiler.step()

        if it % 1 == 0:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            cur_time = time.time()
            if rank == 0:
                cosumed_tokens = sum(bszs) * SEQ_LEN * (it - last_it)
                tokens_per_sec = cosumed_tokens / (cur_time - last_time)
                peak_gpu_mem = torch.cuda.max_memory_allocated()
                time_per_it = (cur_time - last_time) / (it - last_it)
                print(f"iteration: {it}, time_per_it: {time_per_it:.2f}s, throughput: {tokens_per_sec:.0f} tokens/s, loss: {output.loss.item():.3f}, mem: {peak_gpu_mem/1e9:.2f}GB")

            last_time = time.time()
            last_it = it

    # if rank == 0:
    #     profiler.export_chrome_trace(f"./logs/1node/profile_trace_nvidia.json")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
    print(f"Run success")