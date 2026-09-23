"""Single-GPU, unquantized BF16 LoRA SFT with validation and resumable checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace
import uuid

from analyze_baseline import calculate_metrics
from run_baseline import generate_batch, make_result
from sft_training_utils import (
    check_resume_compatibility, epoch_groups, fingerprint, lr_multiplier,
    prepare_full_data, validate_config,
)
from lora_model_setup import initialize_model, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qwen35_4b_lora_sft.json"))
    parser.add_argument("--mode", choices=("prepare", "train"), default="prepare")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def answer_loss(model, example, torch):
    """Mean CE for one complete answer, including EOS; batch size is intentionally one."""
    device = next(model.parameters()).device
    ids = torch.tensor([example["input_ids"]], device=device, dtype=torch.long)
    count = example["answer_tokens"]
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=count + 1).logits[:, :-1, :]
    labels = torch.tensor(example["labels"][-count:], device=device, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits[0].float(), labels)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss on {example['question_id']}")
    return loss


def optimizer_update(model, trainable, examples, indices, optimizer, scheduler, config, torch):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index in indices:
        loss = answer_loss(model, examples[index], torch)
        # The last partial group uses its ACTUAL size, not the configured accumulation count.
        (loss / len(indices)).backward()
        losses.append(loss.item())
    if any(p.grad is None for p in trainable.values()):
        raise RuntimeError("A targeted LoRA parameter has no gradient")
    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), config["max_grad_norm"],
                                         error_if_nonfinite=True)
    lr_used = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    return {"loss": sum(losses) / len(losses), "grad_norm": norm.item(),
            "learning_rate": lr_used, "examples": len(indices)}


def make_optimizer(model, config, total_steps, torch):
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    warmup_steps = math.ceil(total_steps * config["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, total_steps, warmup_steps),
    )
    return trainable, optimizer, scheduler


def capture_rng(torch):
    return {"torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, torch):
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def save_checkpoint(root, model, optimizer, scheduler, metadata, torch):
    """Publish a checkpoint only after all files are written; never overwrite an existing one."""
    target = root / f"checkpoint-{metadata['global_step']:06d}"
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    temporary = root / f".checkpoint-{uuid.uuid4().hex}.incomplete"
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                **capture_rng(torch)}, temporary / "training-state.pt")
    write_json(temporary / "checkpoint-state.json", metadata)
    temporary.rename(target)
    return target


def load_checkpoint(path, model, optimizer, scheduler, torch):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_state = load_file(str(path / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, adapter_state)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise RuntimeError(f"Adapter state mismatch: {result}")
    state = torch.load(path / "training-state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state, torch)


def evaluate_validation(model, processor, rows, config, root, tag, torch):
    """Use the unchanged baseline parser/metrics, retaining all validation examples."""
    output = root / "validation"
    output.mkdir(exist_ok=True)
    # New names per invocation preserve earlier results if an interrupted epoch is replayed.
    name = f"{tag}-{uuid.uuid4().hex[:8]}"
    predictions_path = output / f"{name}.jsonl"
    state = capture_rng(torch)
    was_training = model.training
    old_cache = model.config.use_cache
    old_padding = processor.tokenizer.padding_side
    model.eval()
    model.config.use_cache = True
    options = SimpleNamespace(max_input_tokens=config["eval_max_input_tokens"],
                              max_new_tokens=config["max_new_tokens"])
    records = []
    try:
        with predictions_path.open("x", encoding="utf-8") as destination:
            for index, row in enumerate(rows):
                response = generate_batch(torch, processor, model, [row], options)[0]
                record = make_result(row, response, "validation")
                records.append(record)
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                destination.flush()
                if (index + 1) % 50 == 0:
                    print(f"Validation {tag}: {index + 1}/{len(rows)}", flush=True)
    finally:
        model.config.use_cache = old_cache
        model.train(was_training)
        processor.tokenizer.padding_side = old_padding
        restore_rng(state, torch)
    metrics = calculate_metrics(records)
    metrics["output"] = str(predictions_path)
    metrics["source_split"] = "validation"
    write_json(output / f"{name}-metrics.json", metrics)
    print(json.dumps({"validation": tag, "crisis_macro_f1": metrics["macro_crisis_only_14_labels"]["f1"],
                      "micro_f1": metrics["micro"]["f1"],
                      "valid_format_rate": metrics["valid_format_rate"]}), flush=True)
    return metrics


def execute_training(args, config, processor, examples, validation, audit, report, torch):
    from transformers import set_seed

    root = args.output_dir
    set_seed(config["seed"])
    per_epoch = math.ceil(len(examples) / config["gradient_accumulation_steps"])
    total_steps = per_epoch * config["num_train_epochs"]
    cursor = {"epoch": 0, "next_group": 0, "global_step": 0,
              "best_score": None, "best_checkpoint": None,
              "config_fingerprint": fingerprint(config), "data_fingerprint": fingerprint(audit)}
    if args.resume_from_checkpoint:
        cursor = json.loads((args.resume_from_checkpoint / "checkpoint-state.json").read_text())
        check_resume_compatibility(cursor, config, audit)
        if cursor["global_step"] != cursor["epoch"] * per_epoch + cursor["next_group"]:
            raise ValueError("Invalid resume position")
    model, targets, _ = initialize_model(config, torch)
    model.peft_config["default"].revision = config["revision"]
    trainable, optimizer, scheduler = make_optimizer(model, config, total_steps, torch)
    write_json(root / "lora-targets.json", targets)
    report["model"] = {"base_dtype": "bfloat16", "quantized": False,
                       "target_count": len(targets), "trainable_parameters": sum(p.numel() for p in trainable.values())}
    if args.resume_from_checkpoint:
        load_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler, torch)
    elif config["evaluate_before_training"]:
        metrics = evaluate_validation(model, processor, validation, config, root, "before-training", torch)
        report["baseline_validation_crisis_macro_f1"] = metrics["macro_crisis_only_14_labels"]["f1"]
    log_path = root / f"training-history-{uuid.uuid4().hex[:8]}.jsonl"
    model.train()
    with log_path.open("x", encoding="utf-8") as log:
        for epoch in range(cursor["epoch"], config["num_train_epochs"]):
            groups = epoch_groups(len(examples), config["gradient_accumulation_steps"], config["seed"], epoch)
            first_group = cursor["next_group"] if epoch == cursor["epoch"] else 0
            for group_number in range(first_group, len(groups)):
                record = optimizer_update(model, trainable, examples, groups[group_number],
                                          optimizer, scheduler, config, torch)
                cursor["global_step"] += 1
                cursor.update(epoch=epoch, next_group=group_number + 1)
                record.update(step=cursor["global_step"], epoch=epoch + 1,
                              epoch_progress=(group_number + 1) / len(groups))
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                last_group = group_number == len(groups) - 1
                if last_group:
                    optimizer.zero_grad(set_to_none=True)
                    metrics = evaluate_validation(model, processor, validation, config, root,
                                                  f"epoch-{epoch + 1}", torch)
                    score = metrics["macro_crisis_only_14_labels"]["f1"]
                    if cursor["best_score"] is None or score > cursor["best_score"]:
                        cursor["best_score"] = score
                        cursor["best_checkpoint"] = f"checkpoint-{cursor['global_step']:06d}"
                    cursor.update(epoch=epoch + 1, next_group=0)
                if last_group or cursor["global_step"] % config["save_steps"] == 0:
                    checkpoint = save_checkpoint(root, model, optimizer, scheduler, cursor, torch)
                    write_json(root / "latest-checkpoint.json", {"path": str(checkpoint)})
    if cursor["best_checkpoint"] is None:
        raise RuntimeError("Training produced no evaluated checkpoint")
    # Export only the selected adapter, not the optimizer state or a merged 4B model.
    selected = root / cursor["best_checkpoint"]
    temporary = root / f".best-adapter-{uuid.uuid4().hex}.incomplete"
    temporary.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(selected / name, temporary / name)
    processor.save_pretrained(temporary)
    write_json(temporary / "training-config.json", config)
    write_json(temporary / "selection.json", cursor)
    temporary.rename(root / "best-adapter")
    report.update(status="completed", global_step=cursor["global_step"],
                  best_checkpoint=cursor["best_checkpoint"], best_validation_crisis_macro_f1=cursor["best_score"],
                  best_adapter=str(root / "best-adapter"), training_log=str(log_path))


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.resume_from_checkpoint:
        if args.mode != "train" or args.resume_from_checkpoint.resolve().parent != args.output_dir.resolve():
            raise ValueError("Resume requires train mode and a checkpoint directly inside output-dir")
        if (args.output_dir / "best-adapter").exists():
            raise ValueError("This run already exported its best adapter; use a new directory for a new run")
        # Prevent replay over newer completed checkpoints in the same run directory.
        completed = sorted(p for p in args.output_dir.glob("checkpoint-*") if (p / "checkpoint-state.json").is_file())
        if not completed or completed[-1].resolve() != args.resume_from_checkpoint.resolve():
            raise ValueError("Resume from the latest complete checkpoint in this output directory")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "resolved-config.json", config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = {"mode": args.mode, "status": "started", "output_dir": str(args.output_dir)}
    started = time.monotonic()
    try:
        import torch
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(config["model"], revision=config["revision"], local_files_only=True)
        examples, validation, audit = prepare_full_data(processor, config)
        report["data"] = audit
        report["planned_optimizer_steps"] = math.ceil(len(examples) / config["gradient_accumulation_steps"]) * config["num_train_epochs"]
        report["packages"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}
        write_json(args.output_dir / f"data-audit-{stamp}.json", audit)
        if args.mode == "prepare":
            report["status"] = "prepared_only"
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("Formal training requires a CUDA GPU")
            torch.cuda.reset_peak_memory_stats()
            execute_training(args, config, processor, examples, validation, audit, report, torch)
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt; resume from the latest complete checkpoint")
        raise
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output_dir / f"run-report-{stamp}.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
