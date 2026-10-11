import transformers
import copy
import torch
import torch.nn as nn
from transformers import MixtralForCausalLM, LlamaTokenizerFast, MixtralConfig


HF_MIXTRAL_8x7B_CONFIG = transformers.MixtralConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=32768,
    initializer_range=0.02,
    rms_norm_eps=1e-05,
    use_cache=False,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    tie_word_embeddings=False,
    rope_theta=1000000.0,
    rope_scaling=None,
    attention_dropout=0.0,
)

def get_hf_mixtral8x7b():
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    config = copy.deepcopy(HF_MIXTRAL_8x7B_CONFIG)
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    model = transformers.MixtralForCausalLM(config)
    return model, tokenizer
