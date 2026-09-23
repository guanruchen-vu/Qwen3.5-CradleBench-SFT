"""Prepare a small, reproducible CRADLE training subset with answer-only loss."""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

from cradle_common import build_messages, is_valid_label_response, local_csv_path, parse_gold_labels


def encode_example(processor, row: dict[str, str], max_length: int) -> dict | None:
    """Keep the entire post and answer; skip overlong examples instead of truncating."""
    if not is_valid_label_response(row["final_labels"]):
        raise ValueError(f"Invalid gold labels for question {row['question_id']}")
    answer = ", ".join(parse_gold_labels(row["final_labels"]))
    messages = build_messages(row["question_text"])
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    completed = processor.apply_chat_template(
        messages + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    tokenizer = processor.tokenizer
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(completed, add_special_tokens=False)["input_ids"]
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat-template prompt is not a token prefix of the completed example")
    # Train through the assistant end-of-turn token, excluding any trailing newline.
    answer_ids = full_ids[len(prompt_ids):]
    if tokenizer.eos_token_id not in answer_ids:
        raise ValueError("Assistant completion has no end-of-turn token")
    answer_ids = answer_ids[:answer_ids.index(tokenizer.eos_token_id) + 1]
    if tokenizer.decode(answer_ids[:-1], skip_special_tokens=False).strip() != answer:
        raise ValueError("Completion includes unexpected template text instead of only labels")
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        return None
    return {
        "question_id": row["question_id"],
        "answer": answer,
        "input_ids": input_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "prompt_tokens": len(prompt_ids),
        "answer_tokens": len(answer_ids),
    }


def prepare_examples(processor, config: dict) -> tuple[list[dict], dict]:
    """Never read validation/test; the two training sets are alternatives, not a union."""
    if config["split"] not in {"train_consensus", "train_unanimous"}:
        raise ValueError("Smoke training accepts only train_consensus or train_unanimous")
    path = local_csv_path(Path(config["data_dir"]), config["split"])
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    random.Random(config["seed"]).shuffle(rows)
    examples = []
    skipped = 0
    for row in rows:
        example = encode_example(processor, row, config["max_length"])
        if example is None:
            skipped += 1
            continue
        examples.append(example)
        if len(examples) == config["sample_count"]:
            break
    if len(examples) != config["sample_count"]:
        raise ValueError("Not enough complete examples fit max_length; increase the limit")
    return examples, {
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_rows": len(rows),
        "selected_rows": len(examples),
        "scanned_rows": len(examples) + skipped,
        "overlong_skipped_in_scanned_rows": skipped,
        "selected_ids": [row["question_id"] for row in examples],
        "max_selected_tokens": max(len(row["input_ids"]) for row in examples),
        "note": "Length-filtered training subset for engineering checks, not evaluation.",
    }
