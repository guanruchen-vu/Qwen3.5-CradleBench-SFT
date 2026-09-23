#!/usr/bin/env python3
"""Create extensible head-to-head charts from CRADLE metrics JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cradle_common import CANONICAL_LABELS


OVERALL_METRICS = (
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

LABEL_NAMES = {
    "suicideideation_active_ongoing": "Active SI · ongoing",
    "suicideideation_active_past": "Active SI · past",
    "suicideideation_passive_ongoing": "Passive SI · ongoing",
    "suicideideation_passive_past": "Passive SI · past",
    "selfharm_ongoing": "Self-harm · ongoing",
    "selfharm_past": "Self-harm · past",
    "domesticviolence_ongoing": "Domestic violence · ongoing",
    "domesticviolence_past": "Domestic violence · past",
    "rape_ongoing": "Rape · ongoing",
    "rape_past": "Rape · past",
    "sexualharassment_ongoing": "Sexual harassment · ongoing",
    "sexualharassment_past": "Sexual harassment · past",
    "childabuse_endangerment_ongoing": "Child abuse · ongoing",
    "childabuse_endangerment_past": "Child abuse · past",
    "no_crisis": "No crisis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare any number of CRADLE metric JSON files and export overview, "
            "per-label, CSV, and JSON comparison artifacts."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=METRICS_JSON",
        help=(
            "Model name and metrics path. Repeat for every model. A bare path is "
            "also accepted and its filename becomes the model name."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/model-comparison"),
        help="Destination directory (default: outputs/model-comparison).",
    )
    parser.add_argument(
        "--title",
        default="CRADLE Bench · model comparison",
        help="Figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution (default: 180).",
    )
    return parser.parse_args()


def derive_model_name(path: Path) -> str:
    name = path.stem
    name = re.sub(r"[-_]baseline[-_]test[-_]metrics$", "", name, flags=re.I)
    name = re.sub(r"[-_]metrics$", "", name, flags=re.I)
    return name or path.stem


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
        if not name.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --model value: {spec!r}")
        return name.strip(), Path(raw_path).expanduser()
    path = Path(spec).expanduser()
    return derive_model_name(path), path


def nested_value(data: dict[str, Any], path: tuple[str, ...]) -> float:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Metrics JSON is missing {'.'.join(path)}")
        current = current[key]
    if not isinstance(current, (int, float)):
        raise ValueError(f"Metric {'.'.join(path)} is not numeric")
    return float(current)


def load_models(specs: list[str]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    expected_rows: int | None = None

    for spec in specs:
        name, path = parse_model_spec(spec)
        if name in seen_names:
            raise ValueError(f"Duplicate model name: {name}")
        if not path.exists():
            raise FileNotFoundError(f"Metrics file does not exist: {path}")

        metrics = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError(f"Metrics file must contain a JSON object: {path}")
        rows = metrics.get("rows")
        if not isinstance(rows, int) or rows < 1:
            raise ValueError(f"Invalid rows value in {path}: {rows!r}")
        if expected_rows is None:
            expected_rows = rows
        elif rows != expected_rows:
            raise ValueError(
                f"All models must use the same evaluation set: expected {expected_rows} "
                f"rows, but {name} has {rows}"
            )

        per_label = metrics.get("per_label")
        if not isinstance(per_label, dict):
            raise ValueError(f"Metrics JSON has no per_label object: {path}")
        missing = set(CANONICAL_LABELS).difference(per_label)
        if missing:
            raise ValueError(f"{path} is missing labels: {sorted(missing)}")

        models.append({"name": name, "path": str(path), "metrics": metrics})
        seen_names.add(name)

    if len(models) < 2:
        raise ValueError("At least two --model inputs are required")
    return models


def overall_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {
            "model": model["name"],
            "rows": model["metrics"]["rows"],
        }
        for display_name, path in OVERALL_METRICS:
            row[display_name] = nested_value(model["metrics"], path)
        rows.append(row)
    return rows


def color_cycle(count: int) -> list[Any]:
    cmap = plt.get_cmap("tab10" if count <= 10 else "tab20")
    return [cmap(index % cmap.N) for index in range(count)]


def add_grouped_horizontal_bars(
    ax: plt.Axes,
    categories: list[str],
    values_by_model: list[list[float]],
    model_names: list[str],
    colors: list[Any],
    *,
    title: str,
    show_y_labels: bool = True,
) -> None:
    model_count = len(model_names)
    base_y = np.arange(len(categories), dtype=float)
    group_height = 0.82
    bar_height = group_height / model_count

    for index, (name, values, color) in enumerate(
        zip(model_names, values_by_model, colors, strict=True)
    ):
        offsets = base_y - group_height / 2 + bar_height / 2 + index * bar_height
        bars = ax.barh(offsets, np.asarray(values) * 100, height=bar_height * 0.88, label=name, color=color)
        for bar, value in zip(bars, values, strict=True):
            x = value * 100
            ax.text(
                min(x + 1.0, 97.5),
                bar.get_y() + bar.get_height() / 2,
                f"{x:.1f}",
                va="center",
                ha="left" if x < 96 else "right",
                fontsize=8,
                color="#222222",
            )

    ax.set_yticks(base_y)
    if show_y_labels:
        ax.set_yticklabels(categories)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(len(categories) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)")
    ax.set_title(title, loc="left", fontweight="normal")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def create_figure(models: list[dict[str, Any]], title: str) -> plt.Figure:
    names = [model["name"] for model in models]
    colors = color_cycle(len(models))

    figure = plt.figure(figsize=(17, 14))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(0.72, 1.28),
        left=0.18,
        right=0.985,
        bottom=0.065,
        top=0.90,
        hspace=0.16,
        wspace=0.05,
    )
    overall_ax = figure.add_subplot(grid[0, :])
    recall_ax = figure.add_subplot(grid[1, 0])
    f1_ax = figure.add_subplot(grid[1, 1], sharey=recall_ax)

    overall_categories = [name for name, _ in OVERALL_METRICS]
    overall_values = [
        [nested_value(model["metrics"], path) for _, path in OVERALL_METRICS]
        for model in models
    ]
    add_grouped_horizontal_bars(
        overall_ax,
        overall_categories,
        overall_values,
        names,
        colors,
        title="Overall metrics",
    )

    supports = {
        label: int(models[0]["metrics"]["per_label"][label]["support"])
        for label in CANONICAL_LABELS
    }
    label_categories = [
        f"{LABEL_NAMES[label]}  (n={supports[label]})" for label in CANONICAL_LABELS
    ]
    recall_values = [
        [float(model["metrics"]["per_label"][label]["recall"]) for label in CANONICAL_LABELS]
        for model in models
    ]
    f1_values = [
        [float(model["metrics"]["per_label"][label]["f1"]) for label in CANONICAL_LABELS]
        for model in models
    ]
    add_grouped_horizontal_bars(
        recall_ax,
        label_categories,
        recall_values,
        names,
        colors,
        title="Per-label recall",
    )
    add_grouped_horizontal_bars(
        f1_ax,
        label_categories,
        f1_values,
        names,
        colors,
        title="Per-label F1",
        show_y_labels=False,
    )

    handles, labels = overall_ax.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=min(len(models), 4),
        frameon=False,
    )
    figure.suptitle(title, fontsize=16, fontweight="normal", y=0.985)
    figure.text(
        0.5,
        0.015,
        "All values use the same held-out split. Label support is shown in parentheses.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    return figure


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["model", "rows", *(name for name, _ in OVERALL_METRICS)]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_json(
    path: Path,
    models: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    reference = rows[0]
    payload = {
        "reference_model": reference["model"],
        "models": [
            {
                "name": model["name"],
                "source": model["path"],
                "overall": row,
                "delta_vs_reference": {
                    metric_name: float(row[metric_name]) - float(reference[metric_name])
                    for metric_name, _ in OVERALL_METRICS
                },
                "per_label": model["metrics"]["per_label"],
            }
            for model, row in zip(models, rows, strict=True)
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_table(rows: list[dict[str, Any]]) -> None:
    names = [row["model"] for row in rows]
    header = "metric".ljust(30) + "".join(name.rjust(20) for name in names)
    print(header)
    print("-" * len(header))
    for metric_name, _ in OVERALL_METRICS:
        values = "".join(f"{row[metric_name] * 100:19.2f}%" for row in rows)
        print(metric_name.ljust(30) + values)


def main() -> None:
    args = parse_args()
    models = load_models(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = overall_rows(models)
    print_table(rows)

    figure = create_figure(models, args.title)
    png_path = args.output_dir / "head-to-head.png"
    svg_path = args.output_dir / "head-to-head.svg"
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    csv_path = args.output_dir / "overall-metrics.csv"
    json_path = args.output_dir / "comparison.json"
    write_summary_csv(csv_path, rows)
    write_comparison_json(json_path, models, rows)

    print(f"\nSaved chart: {png_path}")
    print(f"Saved vector chart: {svg_path}")
    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved comparison JSON: {json_path}")


if __name__ == "__main__":
    main()
