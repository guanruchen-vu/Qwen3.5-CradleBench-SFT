"""Dataset audit and deterministic scheduling for single-GPU CRADLE SFT."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re

from cradle_common import (
    CANONICAL_LABELS, EXPECTED_ROWS, SYSTEM_PROMPT, build_messages, clean_question_text,
    local_csv_path, parse_gold_labels,
)
from cradle_sft_data import encode_example


def validate_config(config):
    if "sample_count" in config or "optimizer_steps" in config:
        raise ValueError("Use the formal SFT config: sample_count/optimizer_steps are smoke-only")
    if config["model"] != "Qwen/Qwen3.5-4B" or not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Expected Qwen3.5-4B pinned to a 40-character model revision")
    if config["split"] not in {"train_consensus", "train_unanimous"}:
        raise ValueError("Training may only use a training split")
    for key in ("seed", "max_length", "eval_max_input_tokens", "lora_rank", "lora_alpha",
                "num_train_epochs", "gradient_accumulation_steps", "save_steps", "max_new_tokens"):
        minimum = 0 if key == "seed" else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"Invalid integer for {key}")
    for key in ("learning_rate", "max_grad_norm"):
        if not 0 < config[key] < float("inf"):
            raise ValueError(f"{key} must be positive and finite")
    if not 0 <= config["weight_decay"] < float("inf"):
        raise ValueError("weight_decay must be nonnegative and finite")
    for key in ("warmup_ratio", "lora_dropout"):
        if not 0 <= config[key] < 1:
            raise ValueError(f"{key} must be in [0, 1)")
    if config["lr_scheduler"] != "cosine":
        raise ValueError("This trainer implements a warmup + cosine scheduler")
    if type(config["evaluate_before_training"]) is not bool:
        raise ValueError("evaluate_before_training must be boolean")
    if config["best_metric"] != "macro_crisis_only_14_labels.f1":
        raise ValueError("This experiment selects checkpoints by crisis-only Macro F1")
    re.compile(config["target_pattern"])


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_split(data_dir, split):
    path = local_csv_path(Path(data_dir), split)
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != EXPECTED_ROWS[split]:
        raise ValueError(f"Unexpected {split} size: {len(rows)} vs {EXPECTED_ROWS[split]}")
    ids = [row["question_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate question IDs in {split}")
    for row in rows:
        labels = parse_gold_labels(row["final_labels"])
        if not labels or not clean_question_text(row["question_text"]):
            raise ValueError(f"Missing text or labels in {split}: {row['question_id']}")
        # Normalize official development aliases before applying the strict SFT encoder.
        row["final_labels"] = ", ".join(labels)
    counts = Counter(label for row in rows for label in parse_gold_labels(row["final_labels"]))
    return rows, {"path": str(path), "rows": len(rows),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "label_support": {label: counts[label] for label in CANONICAL_LABELS}}


def prepare_full_data(processor, config):
    train_rows, train_report = read_split(config["data_dir"], config["split"])
    validation_rows, validation_report = read_split(config["data_dir"], "validation")
    train_texts = {clean_question_text(row["question_text"]) for row in train_rows}
    validation_texts = {clean_question_text(row["question_text"]) for row in validation_rows}
    overlap = train_texts & validation_texts
    if overlap:
        raise ValueError(f"Found {len(overlap)} identical posts shared by training and validation")
    examples, overlong = [], []
    for row in train_rows:
        example = encode_example(processor, row, config["max_length"])
        if example is None:
            overlong.append(row["question_id"])
        else:
            examples.append(example)
    if overlong:
        raise ValueError(f"{len(overlong)} training posts exceed max_length; increase it. "
                         f"No samples were silently discarded. First IDs: {overlong[:5]}")
    prompt_lengths = []
    for row in validation_rows:
        prompt = processor.apply_chat_template(
            build_messages(row["question_text"]), tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        prompt_lengths.append(len(processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    if max(prompt_lengths) > config["eval_max_input_tokens"]:
        raise ValueError("Validation prompts exceed eval_max_input_tokens; increase it, do not drop rows")
    train_report.update(max_tokens=max(len(row["input_ids"]) for row in examples),
                        supervised_tokens=sum(row["answer_tokens"] for row in examples),
                        duplicate_texts=len(train_rows) - len(train_texts))
    validation_report.update(max_prompt_tokens=max(prompt_lengths),
                             duplicate_texts=len(validation_rows) - len(validation_texts))
    report = {"train": train_report, "validation": validation_report,
              "train_validation_exact_text_overlap": 0,
              "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              "encoded_training_sha256": fingerprint(examples),
              "validation_prompt_lengths_sha256": fingerprint(prompt_lengths),
              "test_split_loaded": False}
    return examples, validation_rows, report


def epoch_groups(size, accumulation_steps, seed, epoch):
    """Visit each training example exactly once per epoch, including the final short group."""
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    return [indices[start:start + accumulation_steps] for start in range(0, size, accumulation_steps)]


def lr_multiplier(step, total_steps, warmup_steps):
    """LambdaLR calls this at initialization (step=0) and after each optimizer step."""
    if warmup_steps and step < warmup_steps:
        return step / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def check_resume_compatibility(checkpoint, config, data_report):
    if checkpoint["config_fingerprint"] != fingerprint(config):
        raise ValueError("Resume config differs from the original run; start a new output directory")
    if checkpoint["data_fingerprint"] != fingerprint(data_report):
        raise ValueError("Resume data/tokenization/prompt audit differs from the original run")
