#!/usr/bin/env python3
"""Run Qwen3.5-4B as a zero-shot baseline on CRADLE Bench (BF16 on CUDA by default)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

from cradle_common import (
    DATASET_FILES,
    build_messages,
    is_valid_label_response,
    local_csv_path,
    parse_gold_labels,
    parse_model_labels,
)


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single post or a local CRADLE split through Qwen3.5-4B."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--text",
        help="Analyze one manually supplied post instead of a dataset split.",
    )
    source.add_argument(
        "--index",
        type=int,
        help="Analyze only this zero-based row from the selected dataset split.",
    )
    parser.add_argument("--title", default="", help="Optional title used with --text.")
    parser.add_argument(
        "--split",
        choices=tuple(DATASET_FILES),
        default="validation",
        help="Local split to run when --text is absent (default: validation).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/cradle/raw"),
        help="Directory created by download_cradle.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL output path; defaults to outputs/baseline_<split>.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum dataset rows to run; omit to run the complete split.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2_048)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--quantization",
        choices=("4bit", "none"),
        default="none",
        help="Default: none (unquantized BF16 on CUDA); choose 4bit explicitly for NF4.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.title and args.text is None:
        parser.error("--title can only be used together with --text")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.index is not None and args.index < 0:
        parser.error("--index must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.text is not None:
        question = f"Title: {args.title}\nText: {args.text}" if args.title else args.text
        return [
            {
                "question_id": "manual",
                "question_text": question,
                "final_labels": "",
            }
        ]

    path = local_csv_path(args.data_dir, args.split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/download_cradle.py"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    if args.index is not None:
        if args.index >= len(rows):
            raise IndexError(
                f"--index {args.index} is outside {args.split} (rows: {len(rows)})"
            )
        rows = [rows[args.index]]
    elif args.limit is not None:
        rows = rows[: args.limit]
    return rows


def batched(rows: list[dict[str, str]], batch_size: int) -> Iterable[list[dict[str, str]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def load_model(model_name: str, quantization: str):
    try:
        import torch
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )
    except ImportError as error:
        raise RuntimeError(
            "Qwen3.5 requires the configured training environment with a recent "
            "Transformers release."
        ) from error

    if quantization == "4bit" and not torch.cuda.is_available():
        raise RuntimeError("4-bit baseline inference requires a CUDA GPU.")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"device_map": "auto", "dtype": dtype}
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return torch, processor, model


def generate_batch(torch, processor, model, rows, args) -> list[str]:
    prompts = [
        processor.apply_chat_template(
            build_messages(row["question_text"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def make_result(row: dict[str, str], response: str, split: str) -> dict[str, object]:
    predicted = parse_model_labels(response)
    gold = parse_gold_labels(row.get("final_labels", ""))
    has_gold = bool(row.get("final_labels", "").strip())
    return {
        "question_id": row.get("question_id", ""),
        "source_split": split,
        "raw_response": response.strip(),
        "predicted_labels": predicted,
        "gold_labels": gold if has_gold else None,
        "exact_match": predicted == gold if has_gold else None,
        "valid_format": is_valid_label_response(response),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    rows = load_rows(args)
    torch, processor, model = load_model(args.model, args.quantization)

    output = None
    destination = None
    if args.text is None:
        output = args.output or Path(f"outputs/baseline_{args.split}.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        destination = output.open("w", encoding="utf-8")

    results: list[dict[str, object]] = []
    try:
        for batch_number, batch in enumerate(batched(rows, args.batch_size), start=1):
            responses = generate_batch(torch, processor, model, batch, args)
            for row, response in zip(batch, responses, strict=True):
                result = make_result(row, response, "manual" if args.text else args.split)
                results.append(result)
                print(json.dumps(result, ensure_ascii=False))
                if destination is not None:
                    destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                    destination.flush()
            print(
                f"Completed {min(batch_number * args.batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
    finally:
        if destination is not None:
            destination.close()

    if output is not None:
        scored = [result for result in results if result["exact_match"] is not None]
        correct = sum(result["exact_match"] is True for result in scored)
        valid = sum(result["valid_format"] is True for result in results)
        summary = {
            "rows": len(results),
            "exact_match": correct / len(scored) if scored else None,
            "valid_format_rate": valid / len(results) if results else None,
            "output": str(output),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
