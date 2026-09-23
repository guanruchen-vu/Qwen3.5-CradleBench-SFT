#!/usr/bin/env python3
"""Download the four official CRADLE Bench CSV files to a local directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from cradle_common import (
    DATASET_FILES,
    DATASET_REPO,
    EXPECTED_ROWS,
    REQUIRED_COLUMNS,
    parse_gold_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download CRADLE Bench without combining the consensus and "
            "unanimous training files."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cradle/raw"),
        help="Local destination (default: data/cradle/raw).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision, tag, or commit (default: main).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files already present in the destination.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_csv(path: Path, split: str) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = sum(1 for _ in reader)

    expected = EXPECTED_ROWS[split]
    if rows != expected:
        raise ValueError(
            f"{path} contains {rows} rows; expected {expected}. "
            "The upstream dataset may have changed."
        )
    return rows, columns


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def validate_training_relationship(output_dir: Path) -> dict[str, object]:
    """Verify unanimous examples occur in consensus, comparing labels as sets."""
    consensus = read_records(output_dir / DATASET_FILES["train_consensus"])
    unanimous = read_records(output_dir / DATASET_FILES["train_unanimous"])

    consensus_by_text: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in consensus:
        consensus_by_text[row["question_text"]].add(
            tuple(parse_gold_labels(row["final_labels"]))
        )

    missing_texts = 0
    mismatched_labels = 0
    for row in unanimous:
        choices = consensus_by_text.get(row["question_text"])
        if not choices:
            missing_texts += 1
            continue
        labels = tuple(parse_gold_labels(row["final_labels"]))
        if labels not in choices:
            mismatched_labels += 1

    if missing_texts or mismatched_labels:
        raise ValueError(
            "Training split relationship is invalid: "
            f"missing_texts={missing_texts}, mismatched_labels={mismatched_labels}"
        )

    return {
        "unanimous_rows_checked": len(unanimous),
        "all_unanimous_texts_in_consensus": True,
        "all_label_sets_match": True,
        "comparison": "question_text plus order-independent canonical label set",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.json"
    prior_manifest: dict[str, object] = {}
    if manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    all_files_exist = all(
        (args.output_dir / remote_name).exists()
        for remote_name in DATASET_FILES.values()
    )
    can_reuse_snapshot = (
        all_files_exist
        and not args.force
        and prior_manifest.get("dataset") == DATASET_REPO
        and args.revision
        in {
            prior_manifest.get("requested_revision"),
            prior_manifest.get("resolved_commit"),
        }
    )
    if can_reuse_snapshot:
        resolved_commit = str(prior_manifest["resolved_commit"])
        print(f"Reusing local snapshot at commit {resolved_commit}")
    else:
        info = HfApi().dataset_info(DATASET_REPO, revision=args.revision)
        resolved_commit = info.sha

    manifest_files: dict[str, dict[str, object]] = {}

    for split, remote_name in DATASET_FILES.items():
        destination = args.output_dir / remote_name
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not args.force:
            print(f"Using existing {split}: {destination}")
        else:
            cached = Path(
                hf_hub_download(
                    repo_id=DATASET_REPO,
                    repo_type="dataset",
                    filename=remote_name,
                    revision=resolved_commit,
                )
            )
            shutil.copy2(cached, destination)
            print(f"Downloaded {split}: {destination}")

        rows, columns = validate_csv(destination, split)
        manifest_files[split] = {
            "path": str(destination.relative_to(args.output_dir)),
            "rows": rows,
            "columns": columns,
            "sha256": sha256(destination),
        }

    manifest = {
        "dataset": DATASET_REPO,
        "requested_revision": args.revision,
        "resolved_commit": resolved_commit,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "train_unanimous is a high-agreement subset of train_consensus; "
            "do not concatenate them."
        ),
        "training_relationship": validate_training_relationship(args.output_dir),
        "files": manifest_files,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Validated all files. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
