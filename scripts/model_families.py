"""Per-model-family loading and chat formatting.

The pipeline was written for Qwen3.5. Gemma 4 and Llama 3.1 differ in four places, all handled here:
the model class, the message content format, the token that closes an assistant turn, and the
generation stop tokens. For Qwen3.5 every helper returns exactly what the original code used, so
Qwen3.5 prompts, encodings, and decoding are unchanged.
"""

from __future__ import annotations

# Model name -> family. Every supported model is pinned to a revision in its config.
FAMILIES = {
    "Qwen/Qwen3.5-4B": "qwen3_5",
    "Qwen/Qwen3.5-9B": "qwen3_5",
    "google/gemma-4-E4B-it": "gemma4",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3",
}

# Special token that ends an assistant turn; None means tokenizer.eos_token (Qwen3.5: <|im_end|>).
END_OF_TURN = {"qwen3_5": None, "gemma4": "<turn|>", "llama3": "<|eot_id|>"}

# Module families that the LoRA target pattern must reach (Qwen3.5 has hybrid linear attention).
TARGET_FAMILIES = {
    "qwen3_5": ("self_attn", "linear_attn", "mlp"),
    "gemma4": ("self_attn", "mlp"),
    "llama3": ("self_attn", "mlp"),
}

# A special token that only one family's tokenizer has, used to recognise a loaded tokenizer.
_MARKER_TOKENS = (("gemma4", "<turn|>"), ("llama3", "<|eot_id|>"))


def model_family(model_name: str) -> str:
    if model_name not in FAMILIES:
        raise ValueError(f"Unsupported model {model_name!r}; expected one of {sorted(FAMILIES)}")
    return FAMILIES[model_name]


def tokenizer_of(processor):
    """Qwen3.5 and Gemma 4 load a processor that wraps the tokenizer; Llama loads the tokenizer itself."""
    return getattr(processor, "tokenizer", processor)


def tokenizer_family(processor) -> str:
    added = tokenizer_of(processor).added_tokens_encoder
    for family, token in _MARKER_TOKENS:
        if token in added:
            return family
    return "qwen3_5"


def format_messages(messages, processor):
    """Llama 3.1's chat template needs plain-string content; the others take Qwen-style parts."""
    if tokenizer_family(processor) != "llama3":
        return messages
    return [{**message, "content": "".join(part["text"] for part in message["content"])}
            if isinstance(message["content"], list) else message for message in messages]


def end_of_turn_id(processor) -> int:
    tokenizer = tokenizer_of(processor)
    token = END_OF_TURN[tokenizer_family(processor)]
    return tokenizer.eos_token_id if token is None else tokenizer.convert_tokens_to_ids(token)


def generation_eos(processor, model):
    """Qwen3.5 keeps the original tokenizer.eos_token_id; the others stop on every token their
    generation config lists (Gemma 4: <eos>, <turn|>, ...; Llama 3.1: <|eot_id|>, <|eom_id|>, ...)."""
    if tokenizer_family(processor) == "qwen3_5":
        return tokenizer_of(processor).eos_token_id
    return model.generation_config.eos_token_id


def model_class(family: str):
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM

    return AutoModelForCausalLM if family == "llama3" else AutoModelForMultimodalLM


def load_processor(family: str, **source):
    from transformers import AutoProcessor, AutoTokenizer

    return (AutoTokenizer if family == "llama3" else AutoProcessor).from_pretrained(**source)
