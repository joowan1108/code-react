from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

NULL_STRINGS = {"", "null", "none", "nan", "nat", "<na>"}


@dataclass(frozen=True, slots=True)
class ColumnScore:
    matched_columns: int
    gold_columns: int
    predicted_columns: int
    extra_columns: int
    recall: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_columns": self.matched_columns,
            "gold_columns": self.gold_columns,
            "predicted_columns": self.predicted_columns,
            "extra_columns": self.extra_columns,
            "recall": self.recall,
            "score": self.score,
        }


def _normalize_numeric(value: str) -> str | None:
    try:
        decimal = Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    if not decimal.is_finite():
        return None
    rounded = decimal.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:.2f}"


def _normalize_datetime(value: str) -> str | None:
    candidate = value
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None

    if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0 and parsed.microsecond == 0:
        if "T" not in value and ":" not in value:
            return parsed.date().isoformat()
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat().replace("+00:00", "Z")
    return parsed.isoformat()


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\r\n", "\n").replace("\r", "\n")
    if text.lower() in NULL_STRINGS:
        return ""
    numeric = _normalize_numeric(text)
    if numeric is not None:
        return numeric
    normalized_datetime = _normalize_datetime(text)
    if normalized_datetime is not None:
        return normalized_datetime
    return text


def _read_csv_columns(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return []
    column_count = len(rows[0])
    columns = [[] for _ in range(column_count)]
    for row in rows[1:]:
        padded_row = row + [""] * max(column_count - len(row), 0)
        for index in range(column_count):
            columns[index].append(normalize_cell(padded_row[index]))
    return columns


def _column_signatures(path: Path) -> Counter[tuple[str, ...]]:
    signatures: Counter[tuple[str, ...]] = Counter()
    for column in _read_csv_columns(path):
        signatures[tuple(sorted(column))] += 1
    return signatures


def score_prediction_csv(prediction_csv: Path, gold_csv: Path, *, penalty_lambda: float = 0.1) -> ColumnScore:
    if not prediction_csv.exists():
        return ColumnScore(
            matched_columns=0,
            gold_columns=sum(_column_signatures(gold_csv).values()),
            predicted_columns=0,
            extra_columns=0,
            recall=0.0,
            score=0.0,
        )

    prediction_signatures = _column_signatures(prediction_csv)
    gold_signatures = _column_signatures(gold_csv)
    matched_columns = sum(
        min(prediction_signatures[signature], gold_signatures[signature])
        for signature in gold_signatures
    )
    gold_columns = sum(gold_signatures.values())
    predicted_columns = sum(prediction_signatures.values())
    extra_columns = max(predicted_columns - matched_columns, 0)
    recall = matched_columns / gold_columns if gold_columns else 0.0
    penalty = penalty_lambda * (extra_columns / predicted_columns) if predicted_columns else 0.0
    score = max(recall - penalty, 0.0)
    return ColumnScore(
        matched_columns=matched_columns,
        gold_columns=gold_columns,
        predicted_columns=predicted_columns,
        extra_columns=extra_columns,
        recall=recall,
        score=score,
    )


def score_run(prediction_root: Path, gold_root: Path, *, penalty_lambda: float = 0.1) -> dict[str, Any]:
    task_scores: dict[str, dict[str, Any]] = {}
    for gold_csv in sorted(gold_root.glob("task_*/gold.csv")):
        task_id = gold_csv.parent.name
        prediction_csv = prediction_root / task_id / "prediction.csv"
        task_scores[task_id] = score_prediction_csv(
            prediction_csv,
            gold_csv,
            penalty_lambda=penalty_lambda,
        ).to_dict()
    total = sum(item["score"] for item in task_scores.values())
    average = total / len(task_scores) if task_scores else 0.0
    return {
        "task_count": len(task_scores),
        "average_score": average,
        "tasks": task_scores,
    }
