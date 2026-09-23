"""Shared constants and parsing helpers for the CRADLE baseline."""

from __future__ import annotations

import re
from pathlib import Path


DATASET_REPO = "SungJoo/Cradle-Bench"

DATASET_FILES = {
    "train_consensus": "train/train_consensus.csv",
    "train_unanimous": "train/train_unanimous.csv",
    "validation": "dev/development.csv",
    "test": "test/test.csv",
}

EXPECTED_ROWS = {
    "train_consensus": 4_181,
    "train_unanimous": 3_058,
    "validation": 420,
    "test": 600,
}

REQUIRED_COLUMNS = {"question_id", "question_text", "final_labels"}

CRISIS_TYPES = (
    "suicideideation_active",
    "suicideideation_passive",
    "selfharm",
    "domesticviolence",
    "rape",
    "sexualharassment",
    "childabuse_endangerment",
)

CANONICAL_LABELS = tuple(
    f"{crisis_type}_{temporal}"
    for crisis_type in CRISIS_TYPES
    for temporal in ("ongoing", "past")
) + ("no_crisis",)

LABEL_ORDER = {label: index for index, label in enumerate(CANONICAL_LABELS)}

SYSTEM_PROMPT = """You are a strict multi-label classifier for CRADLE Bench.

Identify only crisis events that are supported by the post. The seven crisis types are:
- suicideideation_active: explicit suicidal intent, method, or preparation
- suicideideation_passive: wish to die without active intent or preparation
- selfharm: intentional non-suicidal self-injury or urges
- domesticviolence: abuse between intimate partners
- rape: non-consensual sexual acts involving penetration
- sexualharassment: unwanted sexual advances or contact not involving penetration
- childabuse_endangerment: physical/sexual abuse or neglect of a minor by an adult

Append _ongoing or _past to every detected crisis type. A post may have multiple labels.
If no listed crisis is present, return no_crisis.

Return exactly one line containing either no_crisis or a comma-separated list of canonical labels.
Do not explain your answer and do not output any other text."""

_DATASET_INSTRUCTION_RE = re.compile(
    r"\s*For any crisis present, mark whether it is Ongoing or Past\.\s*"
    r"(?:\*\*)?Leave a row blank if that crisis type is not present\.(?:\*\*)?\s*$",
    flags=re.IGNORECASE,
)

_LABEL_RE = re.compile(
    r"(?<![a-z0-9_])(" + "|".join(
        re.escape(label) for label in sorted(CANONICAL_LABELS, key=len, reverse=True)
    ) + r")(?![a-z0-9_])",
    flags=re.IGNORECASE,
)


def local_csv_path(data_dir: Path, split: str) -> Path:
    """Return the expected local CSV path for a named split."""
    return data_dir / DATASET_FILES[split]


def clean_question_text(text: str) -> str:
    """Remove the benchmark's repeated output instruction from a post."""
    return _DATASET_INSTRUCTION_RE.sub("", text).strip()


def normalize_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    """Deduplicate canonical labels and return them in a stable order."""
    normalized = {
        label.strip().lower()
        for label in labels
        if label and label.strip().lower() in LABEL_ORDER
    }
    if len(normalized) > 1:
        normalized.discard("no_crisis")
    return sorted(normalized, key=LABEL_ORDER.__getitem__)


def parse_gold_labels(raw: str) -> list[str]:
    """Parse canonical train/test labels and the official development aliases."""
    if not raw or not raw.strip():
        return []
    labels = []
    for piece in raw.split(","):
        label = piece.strip().lower()
        label = label.replace("suicideideation(active)", "suicideideation_active")
        label = label.replace("suicideideation(passive)", "suicideideation_passive")
        label = label.replace("childabuseendangerment", "childabuse_endangerment")
        if label == "no crisis":
            label = "no_crisis"
        if label not in LABEL_ORDER:
            raise ValueError(f"Unknown gold label: {piece!r}")
        labels.append(label)
    if "no_crisis" in labels and len(set(labels)) > 1:
        raise ValueError("Gold labels mix no_crisis with a crisis")
    return normalize_labels(labels)


def parse_model_labels(raw: str) -> list[str]:
    """Extract exact canonical labels from a generated model response."""
    return normalize_labels(_LABEL_RE.findall(raw))


def is_valid_label_response(raw: str) -> bool:
    """Check the requested one-line canonical-label output contract."""
    stripped = raw.strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        return False
    pieces = [piece.strip().lower() for piece in stripped.split(",")]
    if any(piece not in LABEL_ORDER for piece in pieces):
        return False
    if len(pieces) != len(set(pieces)):
        return False
    return not ("no_crisis" in pieces and len(pieces) > 1)


def build_messages(question_text: str) -> list[dict[str, object]]:
    """Build Qwen multimodal-chat-shaped messages for text-only inference."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": clean_question_text(question_text)}
            ],
        },
    ]
