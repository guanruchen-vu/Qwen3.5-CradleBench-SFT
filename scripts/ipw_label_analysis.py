#!/usr/bin/env python3
r"""Label-level view of the Unanimous-IPW experiment.

For every label: its count in train_consensus and train_unanimous, the retention rate
(unanimous / consensus), the normalized IPW training weight, and each model's test-set
predicted count, recall and F1. Over the 14 crisis labels it also reports the Spearman
correlation between retention and each pairwise recall difference, with a two-sided
permutation p-value. No model inference is performed.

Example (from a directory containing data/cradle/raw)::

    python scripts/ipw_label_analysis.py \
      --ipw-config configs/qwen35_4b_lora_sft_unanimous_ipw.json \
      --model "SFT Consensus=outputs/test/qwen35-4b-lora-sft-epoch3-test-metrics.json" \
      --model "SFT Unanimous=outputs/test/qwen35-4b-lora-sft-unanimous-epoch1-test-metrics.json" \
      --model "SFT Unanimous-IPW=outputs/test/qwen35-4b-lora-sft-unanimous-ipw-test-metrics.json" \
      --output-dir outputs/ipw-label-analysis
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from cradle_common import CANONICAL_LABELS, parse_gold_labels
from sft_training_utils import label_ipw_weights, read_split, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ipw-config", type=Path, required=True, help="The Unanimous-IPW training config")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=METRICS.json",
                        help="Test metrics from analyze_baseline.py; repeat, in comparison order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.model) < 2:
        parser.error("Provide at least two --model arguments")
    return args


def average_ranks(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = np.empty(len(values))
    ranks[np.argsort(values, kind="mergesort")] = np.arange(len(values))
    for value in np.unique(values):
        ranks[values == value] = ranks[values == value].mean()
    return ranks


def spearman(x: list[float], y: list[float], permutations: int, rng) -> tuple[float, float]:
    """Spearman rho with a two-sided permutation p-value (ties get average ranks)."""
    rx, ry = average_ranks(x), average_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    observed = float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))
    shuffled = ry[np.argsort(rng.random((permutations, len(ry))), axis=1)]
    null = shuffled @ rx / np.sqrt((rx @ rx) * (ry @ ry))
    p_value = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (permutations + 1)
    return observed, float(p_value)


def main() -> None:
    args = parse_args()
    config = json.loads(args.ipw_config.read_text(encoding="utf-8"))
    validate_config(config)
    weighting = config.get("loss_weighting")
    if not weighting:
        raise ValueError("--ipw-config has no loss_weighting")
    train_rows, _ = read_split(config["data_dir"], config["split"])
    reference_rows, _ = read_split(config["data_dir"], weighting["reference_split"])
    _, report = label_ipw_weights(train_rows, reference_rows, weighting["clip"])
    train_counts = Counter(label for row in train_rows for label in parse_gold_labels(row["final_labels"]))
    reference_counts = Counter(label for row in reference_rows for label in parse_gold_labels(row["final_labels"]))

    models = []
    for spec in args.model:
        name, path = (part.strip() for part in spec.split("=", 1))
        models.append((name, json.loads(Path(path).read_text(encoding="utf-8"))["per_label"]))
    if len({tuple(model[1][label]["support"] for label in CANONICAL_LABELS) for model in models}) != 1:
        raise ValueError("Models were not scored on the same test labels")

    rows = []
    for label in sorted(CANONICAL_LABELS, key=lambda item: report["retention_rate"][item]):
        row = {"label": label, "consensus_count": reference_counts[label], "unanimous_count": train_counts[label],
               "retention": report["retention_rate"][label], "ipw_weight": report["normalized_label_weight"][label],
               "test_support": models[0][1][label]["support"]}
        for name, per_label in models:
            for metric in ("predicted", "recall", "f1"):
                row[f"{name} {metric}"] = per_label[label][metric]
        rows.append(row)

    crisis = [row for row in rows if row["label"] != "no_crisis"]
    rng = np.random.default_rng(args.seed)
    correlations = []
    for (left, _), (right, _) in combinations(models, 2):
        difference = [row[f"{right} recall"] - row[f"{left} recall"] for row in crisis]
        rho, p_value = spearman([row["retention"] for row in crisis], difference, args.permutations, rng)
        correlations.append((f"{right} − {left}", rho, p_value))

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "label-analysis.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    names = [name for name, _ in models]
    lines = ["# Label retention, IPW weights, and test recall", "",
             f"Retention = {config['split']} count / {weighting['reference_split']} count. "
             "IPW weight = normalized training weight of that label (mean 1 over training posts). "
             "Test cells are predicted count / recall / F1 (%). Labels are sorted by retention.", "",
             "| Label | Consensus | Unanimous | Retention | IPW weight | Test support | "
             + " | ".join(names) + " |",
             "| --- | ---: | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in names) + " |"]
    for row in rows:
        cells = [f"{row[f'{name} predicted']} / {row[f'{name} recall'] * 100:.1f} / {row[f'{name} f1'] * 100:.1f}"
                 for name in names]
        lines.append(f"| `{row['label']}` | {row['consensus_count']} | {row['unanimous_count']} | "
                     f"{row['retention']:.3f} | {row['ipw_weight']:.3f} | {row['test_support']} | "
                     + " | ".join(cells) + " |")
    lines.extend(["", "## Retention vs. recall difference (14 crisis labels)", "",
                  f"Spearman correlation with a two-sided permutation p-value ({args.permutations:,} permutations, "
                  f"seed {args.seed}). Positive rho: the later model loses more recall on labels that lost more posts.", "",
                  "| Recall difference | Spearman rho | Permutation p |", "| --- | ---: | ---: |"])
    lines.extend(f"| {name} | {rho:+.3f} | {p_value:.4f} |" for name, rho, p_value in correlations)
    lines.extend(["", "Exploratory: several labels have fewer than 12 test posts, so per-label recall is noisy.", ""])
    (output / "label-analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
