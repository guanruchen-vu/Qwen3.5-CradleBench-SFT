"""Audit a completed SFT run and export loss curves and validation comparisons."""

import argparse
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_baseline import calculate_metrics, read_records
from compare_metrics import OVERALL_METRICS, nested_value


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_run(root, before_training_dir=None):
    config = load_json(root / "resolved-config.json")
    selection = load_json(root / "best-adapter/selection.json")
    histories = sorted(root.glob("training-history-*.jsonl"))
    by_step = {}
    for path in histories:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            step = row["step"]
            if step in by_step and by_step[step] != row:
                raise ValueError(f"Conflicting replay logs for step {step}; select the completed trajectory first")
            if not math.isfinite(row["loss"]) or row["loss"] < 0:
                raise ValueError(f"Invalid loss at step {step}")
            by_step[step] = row
    last_step = selection["global_step"]
    if sorted(by_step) != list(range(1, last_step + 1)):
        raise ValueError("Training history is not contiguous through the final selected run state")
    history = [by_step[step] for step in sorted(by_step)]
    paths = sorted((root / "validation").glob("*-metrics.json"))
    if before_training_dir is not None:
        # Runs with evaluate_before_training=false reuse the untrained model's result from another run.
        if any(path.name.startswith("before-training-") for path in paths):
            raise ValueError("This run already has its own before-training validation result")
        other = load_json(before_training_dir / "resolved-config.json")
        for key in ("model", "revision", "model_path", "data_dir", "eval_max_input_tokens", "max_new_tokens"):
            if other.get(key) != config.get(key):
                raise ValueError(f"--before-training-dir differs in {key}; the untrained model must be identical")
        paths += sorted((before_training_dir / "validation").glob("before-training-*-metrics.json"))
    stages = {}
    reference_gold = None
    for path in paths:
        match = re.match(r"(before-training|epoch-(\d+))-", path.name)
        if not match:
            continue
        epoch = int(match[2]) if match[2] else 0
        metrics = load_json(path)
        if metrics.get("source_split") != "validation":
            raise ValueError(f"Not a validation result: {path}")
        records = read_records(path.with_name(path.name.replace("-metrics.json", ".jsonl")))
        gold = {row["question_id"]: sorted(row["gold_labels"]) for row in records}
        if len(gold) != len(records) or len(records) != metrics["rows"]:
            raise ValueError("Duplicate or missing validation samples")
        if reference_gold is None:
            reference_gold = gold
        elif gold != reference_gold:
            raise ValueError("Validation sample IDs or gold labels differ between stages")
        recalculated = calculate_metrics(records)
        for key, value in recalculated.items():
            if metrics[key] != value:
                raise ValueError(f"Saved metrics do not match predictions: {path}, {key}")
        if epoch in stages and stages[epoch]["metrics"] != metrics:
            raise ValueError(f"Multiple validation results for epoch {epoch}; resolve replay provenance first")
        stages[epoch] = {"epoch": epoch, "source": str(path), "metrics": metrics}
    if sorted(stages) != list(range(config["num_train_epochs"] + 1)):
        raise ValueError("Expected before-training and one validation result per epoch "
                         "(use --before-training-dir for runs with evaluate_before_training=false)")
    candidates = [stages[epoch] for epoch in sorted(stages) if epoch > 0]
    selected = max(candidates, key=lambda item: item["metrics"]["macro_crisis_only_14_labels"]["f1"])
    best_epoch = selected["epoch"]
    best_step = max(row["step"] for row in history if row["epoch"] == best_epoch)
    if selection["best_checkpoint"] != f"checkpoint-{best_step:06d}":
        raise ValueError("Exported adapter selection does not match validation scores")
    if abs(selection["best_score"] - selected["metrics"]["macro_crisis_only_14_labels"]["f1"]) > 1e-12:
        raise ValueError("Exported adapter score does not match validation scores")
    return config, history, [stages[epoch] for epoch in sorted(stages)], selection, best_epoch


def save_plot(figure, output, name):
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    figure.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def model_name(config):
    return config["model"].split("/")[-1]


def loss_plot(history, config, output, window):
    steps = np.array([row["step"] for row in history])
    loss = np.array([row["loss"] for row in history])
    weights = np.array([row["examples"] for row in history])
    width = min(window, len(history))
    smoothed = np.convolve(loss * weights, np.ones(width), mode="valid") / np.convolve(weights, np.ones(width), mode="valid")
    figure, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    ax.plot(steps, loss, color="#9aabbc", linewidth=0.7, alpha=0.65, label="Per-update training loss")
    ax.plot(steps[width - 1:], smoothed, color="#246ca3", linewidth=2,
            label=f"{width}-update trailing mean (sample-weighted)")
    epochs = sorted({row["epoch"] for row in history})
    for epoch in epochs:
        positions = [row["step"] for row in history if row["epoch"] == epoch]
        ax.text((min(positions) + max(positions)) / 2, 0.95, f"Epoch {epoch}",
                transform=ax.get_xaxis_transform(), ha="center", va="top", color="#465361")
        if epoch != epochs[-1]:
            ax.axvline(max(positions) + 0.5, color="#bcc5ce", linewidth=0.9, linestyle="--")
    ax.set(xlabel="Optimizer update", ylabel="Answer-token cross-entropy (nats)",
           title=f"{model_name(config)} LoRA SFT | CRADLE {config['split']} training loss",
           xlim=(1, steps[-1]), ylim=(0, max(loss) * 1.12))
    ax.grid(axis="y", color="#e5e8ec", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88), frameon=False)
    save_plot(figure, output, "training-loss")


def metric_table(stages, best_epoch, config, output):
    before = stages[0]["metrics"]
    best = stages[best_epoch]["metrics"]
    rows = []
    for name, path in OVERALL_METRICS:
        a, b = nested_value(before, path), nested_value(best, path)
        rows.append([name, f"{a * 100:.2f}%", f"{b * 100:.2f}%", f"{(b - a) * 100:+.2f}"])
    figure, ax = plt.subplots(figsize=(10, 5.4))
    ax.axis("off")
    ax.set_title(f"{model_name(config)} before vs LoRA SFT (epoch {best_epoch})\nSame 420 validation samples | BF16", pad=18)
    table = ax.table(cellText=rows, colLabels=["Metric", "Before", f"SFT epoch {best_epoch}", "Change (pp)"],
                     colWidths=[0.43, 0.18, 0.19, 0.20], loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dce2e8")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#e8eef5")
        elif row % 2:
            cell.set_facecolor("#f6f8fa")
        if column == 0:
            cell.set_text_props(ha="left")
    figure.tight_layout()
    save_plot(figure, output, "before-after-table")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--before-training-dir", type=Path,
                        help="Run directory whose before-training validation to reuse (same untrained model)")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("window must be positive")
    config, history, stages, selection, best_epoch = read_run(args.run_dir, args.before_training_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    loss_plot(history, config, args.output_dir, args.window)
    rows = metric_table(stages, best_epoch, config, args.output_dir)
    epoch_losses = []
    for epoch in range(1, config["num_train_epochs"] + 1):
        subset = [row for row in history if row["epoch"] == epoch]
        count = sum(row["examples"] for row in subset)
        epoch_losses.append({"epoch": epoch, "updates": len(subset), "examples": count,
                             "mean_sample_loss": sum(row["loss"] * row["examples"] for row in subset) / count})
    summary = {"training_split": config["split"], "model": config["model"], "config": config,
               "selection": selection, "best_epoch": best_epoch, "epoch_training_loss": epoch_losses,
               "validation_stages": stages, "comparison_rows_percent": rows,
               "note": "Training loss only; no validation CE was logged. Model selection uses validation crisis Macro F1."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"Training split: {config['split']}. Selected epoch: {best_epoch}.", "",
             "| Metric | Before | Selected SFT | Change (pp) |", "|---|---:|---:|---:|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(["", "| Stage | Exact match | Micro F1 | Macro F1 (15) | Crisis Macro F1 | Micro recall | Crisis Macro recall |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    paths = [("exact_match",), ("micro", "f1"), ("macro_all_15_labels", "f1"),
             ("macro_crisis_only_14_labels", "f1"), ("micro", "recall"), ("macro_crisis_only_14_labels", "recall")]
    for stage in stages:
        label = "Before" if stage["epoch"] == 0 else f"Epoch {stage['epoch']}"
        values = [f"{nested_value(stage['metrics'], path) * 100:.2f}%" for path in paths]
        lines.append("| " + " | ".join([label, *values]) + " |")
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(epoch_losses, indent=2))


if __name__ == "__main__":
    main()
