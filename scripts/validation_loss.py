#!/usr/bin/env python3
r"""Validation loss for every end-of-epoch checkpoint of formal SFT runs.

Follows the CRADLE Bench paper's checkpoint rule (lowest validation loss). The loss is the training
objective itself (``train_lora_sft.answer_loss``): per-post mean cross-entropy over the answer tokens,
including the end-of-turn token, averaged over all validation posts. A token-weighted mean is reported
as well. Loss weights (label IPW) apply to training only, so validation loss is unweighted for every
run. Only forward passes run: nothing is trained, generated, or written into the run directories.

Example (from ~/llm-project)::

    python ~/Qwen3.5-CradleBench-SFT/scripts/validation_loss.py \
      --run outputs/qwen35-4b-lora-sft-v1 --run outputs/qwen35-4b-lora-sft-unanimous-v1 \
      --output-dir outputs/validation-loss
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import time

from cradle_sft_data import encode_example
from lora_model_setup import initialize_model
from sft_training_utils import model_source, read_split, validate_config
from train_lora_sft import answer_loss

# Runs that agree on these share one base model and LoRA layout; only adapter weights are swapped.
SHARED_KEYS = ("model", "revision", "model_path", "data_dir", "lora_rank", "lora_alpha", "lora_dropout", "target_pattern")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, type=Path, help="Run directory; repeat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Check data and checkpoints only; no GPU")
    return parser.parse_args()


def epoch_checkpoints(run: Path, epochs: int) -> dict[int, Path]:
    """Checkpoints saved at the end of each epoch (state epoch = k, next_group = 0)."""
    found: dict[int, Path] = {}
    for checkpoint in sorted(run.glob("checkpoint-*")):
        state_path = checkpoint / "checkpoint-state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text())
        if state["next_group"] == 0 and 1 <= state["epoch"] <= epochs:
            if state["epoch"] in found:
                raise ValueError(f"{run}: two checkpoints for epoch {state['epoch']}")
            found[state["epoch"]] = checkpoint
    if sorted(found) != list(range(1, epochs + 1)):
        raise ValueError(f"{run}: missing end-of-epoch checkpoints; found epochs {sorted(found)}")
    return found


def f1_selected_epoch(run: Path) -> int:
    checkpoint = json.loads((run / "best-adapter/selection.json").read_text())["best_checkpoint"]
    return json.loads((run / checkpoint / "checkpoint-state.json").read_text())["epoch"]


def load_adapter(model, checkpoint: Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    result = set_peft_model_state_dict(model, load_file(str(checkpoint / "adapter_model.safetensors")))
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise RuntimeError(f"Adapter state mismatch for {checkpoint}: {result}")


def main() -> None:
    args = parse_args()
    runs = []
    for run in args.run:
        config = json.loads((run / "resolved-config.json").read_text())
        validate_config(config)
        runs.append((run, config, epoch_checkpoints(run, config["num_train_epochs"]), f1_selected_epoch(run)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    import torch
    from transformers import AutoProcessor

    groups: dict[tuple, list] = {}
    for item in runs:
        groups.setdefault(tuple(json.dumps(item[1].get(key)) for key in SHARED_KEYS), []).append(item)
    rows = []
    for members in groups.values():
        config = members[0][1]
        processor = AutoProcessor.from_pretrained(**model_source(config), local_files_only=True)
        validation, report = read_split(config["data_dir"], "validation")
        # No length limit: this is a forward pass over complete posts (the longest prompt is 2,027 tokens).
        examples = [encode_example(processor, row, max_length=10**9) for row in validation]
        print(json.dumps({"model": config["model"], "validation_posts": len(examples),
                          "max_tokens": max(len(e["input_ids"]) for e in examples),
                          "answer_tokens": sum(e["answer_tokens"] for e in examples),
                          "runs": [str(item[0]) for item in members]}), flush=True)
        if args.dry_run:
            for run, _, checkpoints, f1_epoch in members:
                print(f"  {run}: {[c.name for c in checkpoints.values()]}; F1-selected epoch {f1_epoch}")
            continue
        model, _, _ = initialize_model(config, torch)
        model.eval()
        for run, run_config, checkpoints, f1_epoch in members:
            for epoch, checkpoint in checkpoints.items():
                load_adapter(model, checkpoint)
                start = time.monotonic()
                with torch.inference_mode():
                    losses = [answer_loss(model, example, torch).item() for example in examples]
                tokens = [example["answer_tokens"] for example in examples]
                row = {"run": run.name, "model": run_config["model"], "split": run_config["split"],
                       "loss_weighting": run_config.get("loss_weighting", {}).get("method", "none"),
                       "epoch": epoch, "checkpoint": checkpoint.name,
                       "validation_loss": sum(losses) / len(losses),
                       "validation_loss_token_weighted": sum(l * t for l, t in zip(losses, tokens)) / sum(tokens),
                       "posts": len(losses), "answer_tokens": sum(tokens), "f1_selected_epoch": f1_epoch,
                       "validation_sha256": report["sha256"]}
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("run", "epoch", "validation_loss",
                                                             "validation_loss_token_weighted")})
                      + f"  ({time.monotonic() - start:.0f}s)", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation-loss.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Validation loss by epoch (CRADLE Bench checkpoint rule: lowest validation loss)", "",
             "Answer-token cross-entropy (nats), per-post mean over all validation posts; "
             "token-weighted mean in parentheses. ✓ = lowest validation loss.", "",
             "| Run | Split | Weighting | Epoch 1 | Epoch 2 | Epoch 3 | Selected by loss | Selected by crisis Macro F1 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for run, config, _, f1_epoch in runs:
        mine = sorted((row for row in rows if row["run"] == run.name), key=lambda row: row["epoch"])
        best = min(mine, key=lambda row: row["validation_loss"])["epoch"]
        cells = [f"{row['validation_loss']:.4f} ({row['validation_loss_token_weighted']:.4f})"
                 + (" ✓" if row["epoch"] == best else "") for row in mine]
        lines.append(f"| `{run.name}` | {config['split']} | {mine[0]['loss_weighting']} | "
                     + " | ".join(cells) + f" | {best} | {f1_epoch} |")
    (args.output_dir / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
