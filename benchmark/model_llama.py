import transformers
import copy
from transformers import LlamaForCausalLM, LlamaTokenizerFast, LlamaConfig


HF_LLAMA3_TINY_CONFIG = transformers.LlamaConfig(
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=8192,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    hidden_act="silu",
    max_position_embeddings=8192,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    use_cache=False,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=50000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)

HF_LLAMA3_1B_CONFIG = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=16,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=131072,
    initializer_range=0.02,
    rms_norm_eps=1e-05,
    use_cache=False,
    pad_token_id=None,
    bos_token_id=128000,
    eos_token_id=128001,
    pretraining_tp=1,
    tie_word_embeddings=True,
    rope_theta=500000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)

HF_LLAMA3_3B_CONFIG = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=3072,
    intermediate_size=8192,
    num_hidden_layers=28,
    num_attention_heads=24,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=131072,
    initializer_range=0.02,
    rms_norm_eps=1e-05,
    use_cache=False,
    pad_token_id=None,
    bos_token_id=128000,
    eos_token_id=128001,
    pretraining_tp=1,
    tie_word_embeddings=True,
    rope_theta=500000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)

HF_LLAMA2_7B_CONFIG = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    hidden_act="silu",
    max_position_embeddings=8192,
    initializer_range=0.02,
    rms_norm_eps=1e-05,
    use_cache=False,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_scaling=None,
)

HF_LLAMA3_8B_CONFIG = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=131072,
    initializer_range=0.02,
    rms_norm_eps=1e-05,
    use_cache=False,
    pad_token_id=None,
    bos_token_id=128000,
    eos_token_id=128001,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=500000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
)

def get_hf_llama(model_size="1b"):
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer", 
        legacy=False  # Suppress the legacy behavior warning
    )

    # Select config based on model size
    model_size = model_size.lower()
    if model_size == "1b":
        config = copy.deepcopy(HF_LLAMA3_1B_CONFIG)
    elif model_size == "3b":
        config = copy.deepcopy(HF_LLAMA3_3B_CONFIG)
    elif model_size == "8b":
        config = copy.deepcopy(HF_LLAMA3_8B_CONFIG)
    elif model_size == "tiny":
        config = copy.deepcopy(HF_LLAMA3_TINY_CONFIG)
    else:
        print(f"Unsupported Llama model size: {model_size}. Using 1B as default.")
        config = copy.deepcopy(HF_LLAMA3_1B_CONFIG)

    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    model = transformers.LlamaForCausalLM(config)
    model.init_weights() # Randomly initialize weights

    return model, tokenizer
