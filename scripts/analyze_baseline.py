#!/usr/bin/env python3
"""Summarize CRADLE multi-label baseline predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cradle_common import CANONICAL_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CRADLE Exact Match, Jaccard, Micro/Macro F1, recall, "
            "and per-label statistics from baseline JSONL output."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="JSONL produced by scripts/run_baseline.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the complete machine-readable metrics JSON.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        help="Fail unless the input contains exactly this many evaluated rows.",
    )
    return parser.parse_args()


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def read_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if record.get("gold_labels") is None:
                raise ValueError(
                    f"Line {line_number} has no gold labels; manual inputs cannot be scored."
                )
            records.append(record)

    if not records:
        raise ValueError(f"No prediction records found in {path}")
    return records


def validate_labels(labels: object, field: str, row_number: int) -> set[str]:
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ValueError(f"Row {row_number} field {field} must be a list of strings")
    unknown = set(labels).difference(CANONICAL_LABELS)
    if unknown:
        raise ValueError(
            f"Row {row_number} field {field} contains unknown labels: {sorted(unknown)}"
        )
    return set(labels)


def calculate_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    counts = {
        label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for label in CANONICAL_LABELS
    }
    exact_matches = 0
    jaccard_sum = 0.0
    valid_formats = 0

    for row_number, record in enumerate(records, start=1):
        predicted = validate_labels(record.get("predicted_labels"), "predicted_labels", row_number)
        gold = validate_labels(record.get("gold_labels"), "gold_labels", row_number)

        exact_matches += predicted == gold
        union = predicted | gold
        jaccard_sum += safe_divide(len(predicted & gold), len(union)) if union else 1.0
        valid_formats += record.get("valid_format") is True

        for label in CANONICAL_LABELS:
            in_prediction = label in predicted
            in_gold = label in gold
            if in_prediction and in_gold:
                counts[label]["tp"] += 1
            elif in_prediction:
                counts[label]["fp"] += 1
            elif in_gold:
                counts[label]["fn"] += 1
            else:
                counts[label]["tn"] += 1

    per_label: dict[str, dict[str, int | float]] = {}
    for label in CANONICAL_LABELS:
        tp = counts[label]["tp"]
        fp = counts[label]["fp"]
        fn = counts[label]["fn"]
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label[label] = {
            **counts[label],
            "support": tp + fn,
            "predicted": tp + fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def macro_average(labels: tuple[str, ...]) -> dict[str, float]:
        return {
            metric: sum(float(per_label[label][metric]) for label in labels) / len(labels)
            for metric in ("precision", "recall", "f1")
        }

    total_tp = sum(counts[label]["tp"] for label in CANONICAL_LABELS)
    total_fp = sum(counts[label]["fp"] for label in CANONICAL_LABELS)
    total_fn = sum(counts[label]["fn"] for label in CANONICAL_LABELS)
    micro_precision = safe_divide(total_tp, total_tp + total_fp)
    micro_recall = safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = safe_divide(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )
    crisis_labels = tuple(label for label in CANONICAL_LABELS if label != "no_crisis")

    return {
        "rows": len(records),
        "exact_match": exact_matches / len(records),
        "sample_jaccard": jaccard_sum / len(records),
        "valid_format_rate": valid_formats / len(records),
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "macro_all_15_labels": macro_average(CANONICAL_LABELS),
        "macro_crisis_only_14_labels": macro_average(crisis_labels),
        "per_label": per_label,
    }


def percent(value: float) -> str:
    return f"{value * 100:6.2f}%"


def print_report(metrics: dict[str, object]) -> None:
    micro = metrics["micro"]
    macro_all = metrics["macro_all_15_labels"]
    macro_crisis = metrics["macro_crisis_only_14_labels"]

    print("\nCRADLE baseline summary")
    print(f"Rows:              {metrics['rows']}")
    print(f"Exact Match:       {percent(metrics['exact_match'])}")
    print(f"Sample Jaccard:    {percent(metrics['sample_jaccard'])}")
    print(f"Valid format:      {percent(metrics['valid_format_rate'])}")
    print(f"Micro F1:          {percent(micro['f1'])}")
    print(f"Micro Recall:      {percent(micro['recall'])}")
    print(f"Macro F1 (15):     {percent(macro_all['f1'])}")
    print(f"Macro Recall (15): {percent(macro_all['recall'])}")
    print(f"Crisis Macro F1:   {percent(macro_crisis['f1'])}")
    print(f"Crisis Macro Rec.: {percent(macro_crisis['recall'])}")

    print("\nPer-label metrics")
    print(
        f"{'label':42} {'support':>7} {'pred':>7} "
        f"{'precision':>10} {'recall':>10} {'f1':>10}"
    )
    print("-" * 92)
    for label in CANONICAL_LABELS:
        values = metrics["per_label"][label]
        print(
            f"{label:42} {values['support']:7d} {values['predicted']:7d} "
            f"{percent(values['precision']):>10} "
            f"{percent(values['recall']):>10} "
            f"{percent(values['f1']):>10}"
        )


def main() -> None:
    args = parse_args()
    records = read_records(args.input)
    if args.expected_rows is not None and len(records) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} rows but found {len(records)} in {args.input}"
        )

    metrics = calculate_metrics(records)
    print_report(metrics)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved metrics: {args.output}")


if __name__ == "__main__":
    main()

