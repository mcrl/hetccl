import torch
import deepspeed
from model_gpt import get_hf_gpt2
import warnings
import os
import logging
import sys

# Comprehensive warning suppression
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress torch CUDA warnings
import torch.cuda
torch.cuda._is_in_bad_fork = lambda: False

# Redirect warnings to devnull
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# =============================
# [1] Plain model loading and forward
# =============================
with suppress_stdout_stderr():
    model, tokenizer = get_hf_gpt2()
    model.eval()

print("=== [NO DeepSpeed] ===")
print("Model class:", type(model))
print("Embedding weight shape:", model.transformer.wte.weight.shape)
print("Tokenizer vocab size:", tokenizer.vocab_size)
print("Device:", model.device)
print("Total parameters:", count_params(model))

# Random input
seq_len = 10
vocab_size = tokenizer.vocab_size
input_ids = torch.randint(low=0, high=vocab_size, size=(1, seq_len))

# Forward without DeepSpeed
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits
    print("Logits shape:", logits.shape)  # Expect: [1, 10, vocab_size]


# =============================
# [2] DeepSpeed model initialization and forward
# =============================
print("\n=== [WITH DeepSpeed] ===")
with suppress_stdout_stderr():
    model, tokenizer = get_hf_gpt2()  # re-initialize (same function, fresh state)
    model.eval()

# Dummy optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Set environment variables for single process
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
# Suppress all logging and warnings
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["NCCL_SUPPRESS_WARN"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage to avoid CUDA warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set all logging to ERROR level
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ['transformers', 'deepspeed', 'torch', 'accelerate']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Initialize torch distributed manually for single GPU
import torch.distributed as dist

if not dist.is_initialized():
    with suppress_stdout_stderr():
        try:
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
        except Exception:
            pass

# DeepSpeed initialize with complete suppression
with suppress_stdout_stderr():
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config_params={
            "train_batch_size": 1,
            "fp16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 0
            }
        },
        dist_init_required=False
    )

print("Model class:", type(model_engine))
print("Embedding weight shape:", model_engine.module.transformer.wte.weight.shape)
print("Device:", model_engine.device)
print("Total parameters:", count_params(model_engine.module))

# Random input (moved to GPU)
input_ids = torch.randint(low=0, high=vocab_size, size=(1, seq_len)).to(model_engine.device)

# Forward with DeepSpeed
with torch.no_grad():
    outputs = model_engine(input_ids)
    logits = outputs.logits
    print("Logits shape:", logits.shape)  # Expect: [1, 10, vocab_size]
