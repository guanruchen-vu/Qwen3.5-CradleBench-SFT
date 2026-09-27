#!/usr/bin/env python3
r"""Plot training loss and end-of-epoch validation loss for several SFT runs (supplement Figure S1).

Training loss is the unweighted answer-token cross-entropy logged by train_lora_sft.py
(20-update trailing mean, weighted by posts per update); validation loss comes from
validation_loss.py. One panel per model size. Writes PDF (TrueType fonts) and PNG.

Example (from the directory holding the run folders)::

    python scripts/plot_loss_curves.py \
      --run "4B,Consensus,outputs/qwen35-4b-lora-sft-v1" \
      --run "4B,Unanimous,outputs/qwen35-4b-lora-sft-unanimous-v1" \
      --run "4B,Unanimous-IPW,outputs/qwen35-4b-lora-sft-unanimous-ipw-seed42" \
      --run "9B,Consensus,outputs/qwen35-9b-lora-sft-consensus-seed42" \
      --run "9B,Unanimous,outputs/qwen35-9b-lora-sft-unanimous-seed42" \
      --run "9B,Unanimous-IPW,outputs/qwen35-9b-lora-sft-unanimous-ipw-seed42" \
      --validation-loss outputs/validation-loss/validation-loss.csv \
      --output outputs/figures/loss-curves
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

INK, INK_2, AXIS, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, validated all-pairs on white
WINDOW = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="SIZE,NAME,RUN_DIR",
                        help="Repeat; runs sharing SIZE share a panel, colors follow NAME order of first appearance")
    parser.add_argument("--validation-loss", type=Path, required=True, help="validation-loss.csv from validation_loss.py")
    parser.add_argument("--output", type=Path, required=True, help="Output path without extension")
    return parser.parse_args()


def history(run_dir):
    """Training records by step, merged across resumed logs; must be contiguous from step 1."""
    by_step = {}
    for path in sorted(run_dir.glob("training-history-*.jsonl")):
        for line in path.open():
            if line.strip():
                row = json.loads(line)
                if row["step"] in by_step and by_step[row["step"]] != row:
                    raise ValueError(f"{run_dir}: conflicting records for step {row['step']}")
                by_step[row["step"]] = row
    rows = [by_step[step] for step in sorted(by_step)]
    if [row["step"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{run_dir}: training history is not contiguous")
    return rows


def main():
    args = parse_args()
    runs = []
    for spec in args.run:
        size, name, run_dir = (part.strip() for part in spec.split(",", 2))
        runs.append((size, name, Path(run_dir)))
    sizes = list(dict.fromkeys(size for size, _, _ in runs))
    names = list(dict.fromkeys(name for _, name, _ in runs))
    if len(names) > len(COLORS):
        raise ValueError("At most three training sets per figure")
    color = dict(zip(names, COLORS))
    validation = {(r["run"], int(r["epoch"])): float(r["validation_loss"]) for r in csv.DictReader(args.validation_loss.open())}

    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": AXIS,
                         "axes.linewidth": 0.6, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "legend.frameon": False})
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.25 * len(sizes), 2.5), sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        for run_size, name, run_dir in runs:
            if run_size != size:
                continue
            rows = history(run_dir)
            per_epoch = max(row["step"] for row in rows if row["epoch"] == 1)
            epochs = max(row["epoch"] for row in rows)
            x = np.array([row["step"] for row in rows]) / per_epoch
            loss = np.array([row["loss"] for row in rows])
            posts = np.array([row["examples"] for row in rows])
            smooth = np.convolve(loss * posts, np.ones(WINDOW), "valid") / np.convolve(posts, np.ones(WINDOW), "valid")
            ax.plot(x[WINDOW - 1:], smooth, color=color[name], linewidth=1.3)
            points = [(e, validation[(run_dir.name, e)]) for e in range(1, epochs + 1) if (run_dir.name, e) in validation]
            if points:
                ax.plot(*zip(*points), color=color[name], linewidth=0.8, marker="D", markersize=4,
                        markeredgecolor="white", markeredgewidth=0.5)
        ax.set_yscale("log")
        ax.set_xlim(0, epochs + 0.05)
        ax.set_xticks(range(epochs + 1))
        ax.set_xlabel("Epoch")
        ax.set_title(f"Qwen3.5-{size}", fontsize=8, color=INK, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Answer-token cross-entropy (nats, log scale)")
    handles = [Line2D([], [], color=color[name], linewidth=1.3, label=name) for name in names]
    handles += [Line2D([], [], color=INK_2, linewidth=1.3, label=f"Training ({WINDOW}-update mean)"),
                Line2D([], [], color=INK_2, linewidth=0.8, marker="D", markersize=4, markeredgecolor="white",
                       label="Validation (end of epoch)")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=6.8, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.6, columnspacing=1.1)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.83, bottom=0.17, wspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=300)
    print("wrote", args.output.with_suffix(".pdf"), args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
