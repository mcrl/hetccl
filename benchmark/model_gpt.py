import transformers
import copy
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config

HF_GPT2_SMALL_CONFIG = GPT2Config(
    vocab_size=50257,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_positions=1024,
    n_ctx=1024,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=50256,
    eos_token_id=50256,
)

HF_GPT2_MEDIUM_CONFIG = GPT2Config(
    vocab_size=50257,
    n_embd=1024,
    n_layer=24,
    n_head=16,
    n_positions=1024,
    n_ctx=1024,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=50256,
    eos_token_id=50256,
)

HF_GPT2_LARGE_CONFIG = transformers.GPT2Config(
    vocab_size=50257,
    n_embd=1280,
    n_layer=36,
    n_head=20,
    n_positions=1024,
    n_ctx=1024,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=50256,
    eos_token_id=50256,
)

HF_GPT2_XL_CONFIG = transformers.GPT2Config(
    vocab_size=50257,
    n_embd=1600,
    n_layer=48,
    n_head=25,
    n_positions=1024,
    n_ctx=1024,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=50256,
    eos_token_id=50256,
)

def get_hf_gpt2(model_size="355m"):
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    
    # Select config based on model size
    model_size = model_size.lower()
    if model_size in ["125m", "small"]:
        config = copy.deepcopy(HF_GPT2_SMALL_CONFIG)
    elif model_size in ["355m", "medium"]:
        config = copy.deepcopy(HF_GPT2_MEDIUM_CONFIG)
    elif model_size in ["774m", "large"]:
        config = copy.deepcopy(HF_GPT2_LARGE_CONFIG)
    elif model_size in ["1558m", "1.5b", "xl"]:
        config = copy.deepcopy(HF_GPT2_XL_CONFIG)
    else:
        print(f"Unsupported GPT2 model size: {model_size}. Using 355M (medium) as default.")
        config = copy.deepcopy(HF_GPT2_MEDIUM_CONFIG)

    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    model = transformers.GPT2LMHeadModel(config)
    model.init_weights() # Randomly initialize weights

    return model, tokenizer
