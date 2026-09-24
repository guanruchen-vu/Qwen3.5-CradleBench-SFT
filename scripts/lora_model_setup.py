"""Shared BF16 LoRA initialization and JSON output helpers for formal SFT."""

from __future__ import annotations

import json
from pathlib import Path
import re

from sft_training_utils import model_source


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def initialize_model(config, torch):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    # Ordinary LoRA: no BitsAndBytesConfig, no k-bit preparation, no CPU offloading.
    base = AutoModelForMultimodalLM.from_pretrained(
        **model_source(config), local_files_only=True,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    if getattr(base, "is_quantized", False) or base.get_input_embeddings().weight.dtype != torch.bfloat16:
        raise RuntimeError("Expected an unquantized BF16 base model")
    targets = [name for name, module in base.named_modules()
               if isinstance(module, torch.nn.Linear) and re.fullmatch(config["target_pattern"], name)]
    # Qwen3.5 has both full and linear attention. Check that neither was missed.
    for family in ("self_attn", "linear_attn", "mlp"):
        if not any(f".{family}." in name for name in targets):
            raise RuntimeError(f"No LoRA targets found for {family}")
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", target_modules=targets, r=config["lora_rank"],
        lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], bias="none",
    ))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA adapter parameters should be trainable")
    return model, targets, trainable
