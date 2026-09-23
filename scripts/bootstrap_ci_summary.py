#!/usr/bin/env python3
r"""Paired bootstrap 95% CIs and summary tables for CRADLE test predictions.

Example (from the repository root; this is NOT run automatically)::

    python scripts/bootstrap_ci_summary.py \
      --overall-csv outputs/bf16base-vs-sft-all-test/overall-metrics.csv \
      --model 'Qwen3.5-4B BF16=outputs/qwen3.5-4b-baseline-bf16-test.jsonl' \
      --model 'Qwen3.5-4B SFT Epoch3=outputs/qwen35-4b-lora-sft-epoch3-test.jsonl' \
      --model 'Qwen3.5-4B SFT Epoch1 Unanimous=outputs/qwen35-4b-lora-sft-unanimous-epoch1-test.jsonl' \
      --output-dir outputs/bf16base-vs-sft-all-test/bootstrap-ci

The aggregate CSV alone cannot be bootstrapped: each replicate resamples the
same test question IDs across *all* models. The script writes a Markdown
summary table, a machine-readable summary CSV, and a paired-differences CSV.
No model inference or training is performed.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze_baseline import calculate_metrics, read_records, validate_labels
from cradle_common import CANONICAL_LABELS


METRICS = (
    ("Exact Match", ("exact_match",)),
    ("Sample Jaccard", ("sample_jaccard",)),
    ("Micro F1", ("micro", "f1")),
    ("Macro F1 · all 15", ("macro_all_15_labels", "f1")),
    ("Macro F1 · crisis 14", ("macro_crisis_only_14_labels", "f1")),
    ("Micro Recall", ("micro", "recall")),
    ("Macro Recall · all 15", ("macro_all_15_labels", "recall")),
    ("Macro Recall · crisis 14", ("macro_crisis_only_14_labels", "recall")),
    ("Valid Format", ("valid_format_rate",)),
)
CRISIS_INDICES = [i for i, label in enumerate(CANONICAL_LABELS) if label != "no_crisis"]


@dataclass
class ModelRows:
    name: str
    path: Path
    ids: tuple[str, ...]
    gold: tuple[frozenset[str], ...]
    point: np.ndarray                     # (metrics,)
    tp: np.ndarray                        # (rows, labels)
    fp: np.ndarray
    fn: np.ndarray
    exact: np.ndarray                     # (rows,)
    jaccard: np.ndarray
    valid: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", required=True, metavar="NAME=PREDICTIONS.jsonl",
        help="Repeat for every model; NAME must match the aggregate CSV if supplied.",
    )
    parser.add_argument(
        "--overall-csv", type=Path,
        help="Optional existing overall-metrics.csv; cross-checks all point estimates.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/bootstrap-ci"),
        help="New directory for summary.md, summary.csv, and paired-differences.csv.",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Bootstrap replicates processed together to bound memory use.")
    parser.add_argument("--expected-rows", type=int, default=None,
                        help="Optional exact test-set size (e.g. 600).")
    args = parser.parse_args()
    if len(args.model) < 2:
        parser.error("Provide at least two --model arguments.")
    if args.resamples < 100 or args.batch_size < 1:
        parser.error("--resamples must be >= 100 and --batch-size must be >= 1.")
    if args.expected_rows is not None and args.expected_rows < 1:
        parser.error("--expected-rows must be positive.")
    return args


def get_metric(metrics: dict[str, object], keys: tuple[str, ...]) -> float:
    value: object = metrics
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing metric {'.'.join(keys)}")
        value = value[key]
    return float(value)


def load_model(spec: str) -> ModelRows:
    if "=" not in spec:
        raise ValueError(f"Expected --model NAME=PATH, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    name, raw_path = name.strip(), raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"Expected --model NAME=PATH, got {spec!r}")
    path = Path(raw_path).expanduser()
    records = read_records(path)
    ids: list[str] = []
    gold_sets: list[frozenset[str]] = []
    tp, fp, fn = [], [], []
    exact, jaccard, valid = [], [], []

    for row_number, record in enumerate(records, 1):
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{path}: row {row_number} has no question_id")
        if record.get("source_split") not in (None, "test"):
            raise ValueError(f"{path}: row {row_number} is not from the test split")
        predicted = validate_labels(record.get("predicted_labels"), "predicted_labels", row_number)
        gold = validate_labels(record.get("gold_labels"), "gold_labels", row_number)
        ids.append(question_id)
        gold_sets.append(frozenset(gold))
        tp.append([int(label in predicted and label in gold) for label in CANONICAL_LABELS])
        fp.append([int(label in predicted and label not in gold) for label in CANONICAL_LABELS])
        fn.append([int(label not in predicted and label in gold) for label in CANONICAL_LABELS])
        exact.append(int(predicted == gold))
        union = predicted | gold
        jaccard.append(len(predicted & gold) / len(union) if union else 1.0)
        valid.append(int(record.get("valid_format") is True))

    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate question_id; paired bootstrap needs unique IDs")
    point_metrics = calculate_metrics(records)  # Same definitions as the existing analysis.
    return ModelRows(
        name=name, path=path, ids=tuple(ids), gold=tuple(gold_sets),
        point=np.array([get_metric(point_metrics, keys) for _, keys in METRICS]),
        tp=np.asarray(tp, dtype=np.uint8), fp=np.asarray(fp, dtype=np.uint8),
        fn=np.asarray(fn, dtype=np.uint8), exact=np.asarray(exact, dtype=np.uint8),
        jaccard=np.asarray(jaccard, dtype=np.float64),
        valid=np.asarray(valid, dtype=np.uint8),
    )


def align_models(models: list[ModelRows], expected_rows: int | None) -> None:
    reference = models[0]
    if expected_rows is not None and len(reference.ids) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(reference.ids)}")
    reference_gold = dict(zip(reference.ids, reference.gold))
    for model in models[1:]:
        if set(model.ids) != set(reference.ids):
            raise ValueError(f"{model.name}: test question IDs differ from {reference.name}")
        if any(reference_gold[qid] != gold for qid, gold in zip(model.ids, model.gold)):
            raise ValueError(f"{model.name}: gold labels differ from {reference.name}")
        # Reorder arrays to match the first model; all resamples then use the same posts.
        position_by_id = {qid: i for i, qid in enumerate(model.ids)}
        positions = np.asarray([position_by_id[qid] for qid in reference.ids])
        for key in ("tp", "fp", "fn", "exact", "jaccard", "valid"):
            setattr(model, key, getattr(model, key)[positions])
        model.ids = reference.ids
        model.gold = reference.gold


def verify_overall_csv(path: Path, models: list[ModelRows]) -> None:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(row.get("model") is None for row in rows):
        raise ValueError(f"Invalid overall-metrics.csv: {path}")
    by_name = {row["model"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError(f"Duplicate model names in {path}")
    for model in models:
        if model.name not in by_name:
            raise ValueError(f"{model.name!r} not found in {path}; use the exact CSV model name")
        row = by_name[model.name]
        if int(row["rows"]) != len(model.ids):
            raise ValueError(f"{model.name}: CSV row count differs from JSONL")
        for (metric_name, _), value in zip(METRICS, model.point):
            csv_value = float(row[metric_name])
            if not math.isclose(csv_value, float(value), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"{model.name}, {metric_name}: CSV={csv_value} but JSONL={value}; "
                    "check that the files come from the same evaluation"
                )


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator != 0)


def bootstrap_batch(model: ModelRows, indices: np.ndarray) -> np.ndarray:
    tp = model.tp[indices].sum(axis=1, dtype=np.int64)
    fp = model.fp[indices].sum(axis=1, dtype=np.int64)
    fn = model.fn[indices].sum(axis=1, dtype=np.int64)
    per_label_f1 = ratio(2 * tp, 2 * tp + fp + fn)
    per_label_recall = ratio(tp, tp + fn)
    total_tp, total_fp, total_fn = tp.sum(axis=1), fp.sum(axis=1), fn.sum(axis=1)
    return np.column_stack((
        model.exact[indices].mean(axis=1),
        model.jaccard[indices].mean(axis=1),
        ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn),
        per_label_f1.mean(axis=1),
        per_label_f1[:, CRISIS_INDICES].mean(axis=1),
        ratio(total_tp, total_tp + total_fn),
        per_label_recall.mean(axis=1),
        per_label_recall[:, CRISIS_INDICES].mean(axis=1),
        model.valid[indices].mean(axis=1),
    ))


def bootstrap(models: list[ModelRows], resamples: int, seed: int,
              batch_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    draws = {model.name: np.empty((resamples, len(METRICS))) for model in models}
    n = len(models[0].ids)
    for start in range(0, resamples, batch_size):
        end = min(start + batch_size, resamples)
        indices = rng.integers(0, n, size=(end - start, n))
        for model in models:
            draws[model.name][start:end] = bootstrap_batch(model, indices)
    return draws


def limits(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return low, high


def write_tables(output_dir: Path, models: list[ModelRows],
                 draws: dict[str, np.ndarray], resamples: int, seed: int) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cis = {model.name: limits(draws[model.name]) for model in models}
    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    lines = [
        "# CRADLE test-set summary with bootstrap 95% CI", "",
        f"Paired percentile bootstrap: {resamples:,} resamples of {len(models[0].ids)} "
        f"test posts, seed {seed}. Each model uses the same resampled question IDs. "
        "CIs are percentile intervals; they are not significance tests.", "",
        "All model values below are percentages: point estimate [95% CI].", "",
        "| Metric | " + " | ".join(model.name for model in models) + " |",
        "| --- | " + " | ".join("---:" for _ in models) + " |",
    ]
    for j, (metric_name, _) in enumerate(METRICS):
        cells = []
        for model in models:
            low, high = cis[model.name]
            point = float(model.point[j])
            summary_rows.append({"model": model.name, "metric": metric_name,
                                 "estimate": point, "ci_low": low[j], "ci_high": high[j]})
            cells.append(f"{point * 100:.2f}% [{low[j] * 100:.2f}%, {high[j] * 100:.2f}%]")
        lines.append("| " + metric_name + " | " + " | ".join(cells) + " |")

    lines.extend(["", "## Paired differences", "",
                  "Positive values favor the later model. Units are percentage points (pp).", "",
                  "| Metric | " + " | ".join(
                      f"{right.name} − {left.name}" for left, right in combinations(models, 2)
                  ) + " |",
                  "| --- | " + " | ".join("---:" for _ in combinations(models, 2)) + " |"])
    pairs = list(combinations(models, 2))
    pair_intervals = {
        (left.name, right.name): limits(draws[right.name] - draws[left.name])
        for left, right in pairs
    }
    for j, (metric_name, _) in enumerate(METRICS):
        cells = []
        for left, right in pairs:
            low, high = pair_intervals[left.name, right.name]
            difference = float(right.point[j] - left.point[j])
            paired_rows.append({"reference": left.name, "candidate": right.name,
                                "metric": metric_name, "difference": difference,
                                "ci_low": low[j], "ci_high": high[j]})
            cells.append(f"{difference * 100:+.2f} pp "
                         f"[{low[j] * 100:+.2f}, {high[j] * 100:+.2f}]")
        lines.append("| " + metric_name + " | " + " | ".join(cells) + " |")
    lines.extend(["", "Note: Macro F1 averages 15 labels (or 14 crisis labels), "
                  "counting a label with no positive samples in a resample as F1 = 0, "
                  "matching the existing analysis script.", ""])

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    for filename, fieldnames, rows in (
        ("summary.csv", ("model", "metric", "estimate", "ci_low", "ci_high"), summary_rows),
        ("paired-differences.csv", ("reference", "candidate", "metric", "difference",
                                    "ci_low", "ci_high"), paired_rows),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'paired-differences.csv'}")


def main() -> None:
    args = parse_args()
    models = [load_model(spec) for spec in args.model]
    if len({model.name for model in models}) != len(models):
        raise ValueError("Model names must be unique")
    align_models(models, args.expected_rows)
    if args.overall_csv is not None:
        verify_overall_csv(args.overall_csv, models)
    draws = bootstrap(models, args.resamples, args.seed, args.batch_size)
    write_tables(args.output_dir, models, draws, args.resamples, args.seed)


if __name__ == "__main__":
    main()
