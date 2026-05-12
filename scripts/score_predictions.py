from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from itertools import permutations
from pathlib import Path
from typing import Any

NULL_LITERALS = {"", "null", "none", "nan", "nat", "<na>"}
DATE_PATTERN = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
DATETIME_HINT_PATTERN = re.compile(r"[Tt ]\d{1,2}:\d{2}")
NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")


@dataclass(frozen=True, slots=True)
class ColumnRecord:
    index: int
    values: tuple[str, ...]
    signature: tuple[str, ...]
    virtual: bool = False


@dataclass(frozen=True, slots=True)
class TaskScore:
    task_id: str
    prediction_exists: bool
    gold_exists: bool
    matched_gold_columns: int
    matched_predicted_columns: int
    gold_columns: int
    predicted_columns: int
    extra_columns: int
    recall: float
    penalty: float
    score: float
    direct_matches: int
    name_pair_matches: int
    missing_prediction: bool
    notes: list[str]


def normalize_null(value: str) -> str | None:
    stripped = value.strip().replace("\r", "").replace("\n", "")
    if stripped.casefold() in NULL_LITERALS:
        return ""
    return stripped


def normalize_numeric(value: str) -> str | None:
    if not NUMERIC_PATTERN.fullmatch(value):
        return None
    try:
        normalized = Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    return format(normalized, "f")


def normalize_date(value: str) -> str | None:
    if not DATE_PATTERN.fullmatch(value):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed.isoformat()


def normalize_datetime(value: str) -> str | None:
    if not DATETIME_HINT_PATTERN.search(value):
        return None
    candidate = value
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return parsed.isoformat()


def normalize_cell(value: str) -> str:
    normalized_null = normalize_null(value)
    if normalized_null == "":
        return ""
    assert normalized_null is not None

    for normalizer in (normalize_numeric, normalize_date, normalize_datetime):
        normalized = normalizer(normalized_null)
        if normalized is not None:
            return normalized
    return normalized_null


def read_csv_matrix(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.reader(handle))


def matrix_to_columns(rows: list[list[str]]) -> list[ColumnRecord]:
    if not rows:
        return []
    body = rows[1:] if len(rows) > 1 else []
    width = max((len(row) for row in rows), default=0)
    columns: list[ColumnRecord] = []
    for column_index in range(width):
        normalized_values = tuple(
            normalize_cell(row[column_index] if column_index < len(row) else "")
            for row in body
        )
        signature = tuple(sorted(normalized_values))
        columns.append(
            ColumnRecord(
                index=column_index,
                values=normalized_values,
                signature=signature,
            )
        )
    return columns


def combine_name_values(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    combined: list[str] = []
    for first, second in zip(left, right):
        parts = [part for part in (first.strip(), second.strip()) if part]
        combined.append(" ".join(parts))
    return tuple(combined)


def build_name_pair_records(columns: list[ColumnRecord]) -> list[tuple[int, int, tuple[str, ...]]]:
    pairs: list[tuple[int, int, tuple[str, ...]]] = []
    for left_index, right_index in permutations(range(len(columns)), 2):
        left = columns[left_index]
        right = columns[right_index]
        combined_values = combine_name_values(left.values, right.values)
        pairs.append((left_index, right_index, tuple(sorted(combined_values))))
    return pairs


def match_direct_columns(
    pred_columns: list[ColumnRecord],
    gold_columns: list[ColumnRecord],
) -> tuple[set[int], set[int], int]:
    pred_by_signature: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
    gold_by_signature: dict[tuple[str, ...], deque[int]] = defaultdict(deque)

    for column in pred_columns:
        pred_by_signature[column.signature].append(column.index)
    for column in gold_columns:
        gold_by_signature[column.signature].append(column.index)

    matched_pred: set[int] = set()
    matched_gold: set[int] = set()
    direct_matches = 0

    for signature, pred_queue in pred_by_signature.items():
        gold_queue = gold_by_signature.get(signature)
        if not gold_queue:
            continue
        while pred_queue and gold_queue:
            matched_pred.add(pred_queue.popleft())
            matched_gold.add(gold_queue.popleft())
            direct_matches += 1
    return matched_pred, matched_gold, direct_matches


def match_name_pairs(
    pred_columns: list[ColumnRecord],
    gold_columns: list[ColumnRecord],
    matched_pred: set[int],
    matched_gold: set[int],
) -> tuple[int, int, int]:
    matched_gold_columns = 0
    matched_predicted_columns = 0
    pair_matches = 0

    unmatched_pred_singletons = [column for column in pred_columns if column.index not in matched_pred]
    unmatched_gold_singletons = [column for column in gold_columns if column.index not in matched_gold]

    pred_pairs = build_name_pair_records(unmatched_pred_singletons)
    gold_pairs = build_name_pair_records(unmatched_gold_singletons)

    gold_signature_to_singletons: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for column in unmatched_gold_singletons:
        gold_signature_to_singletons[column.signature].append(column.index)

    pred_signature_to_singletons: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for column in unmatched_pred_singletons:
        pred_signature_to_singletons[column.signature].append(column.index)

    # prediction split names -> gold full name
    for left_index, right_index, signature in pred_pairs:
        if left_index in matched_pred or right_index in matched_pred:
            continue
        candidate_gold = next(
            (index for index in gold_signature_to_singletons.get(signature, []) if index not in matched_gold),
            None,
        )
        if candidate_gold is None:
            continue
        matched_pred.update({left_index, right_index})
        matched_gold.add(candidate_gold)
        matched_gold_columns += 1
        matched_predicted_columns += 2
        pair_matches += 1

    # prediction full name -> gold split names
    for left_index, right_index, signature in gold_pairs:
        if left_index in matched_gold or right_index in matched_gold:
            continue
        candidate_pred = next(
            (index for index in pred_signature_to_singletons.get(signature, []) if index not in matched_pred),
            None,
        )
        if candidate_pred is None:
            continue
        matched_gold.update({left_index, right_index})
        matched_pred.add(candidate_pred)
        matched_gold_columns += 2
        matched_predicted_columns += 1
        pair_matches += 1

    return matched_gold_columns, matched_predicted_columns, pair_matches


def score_task(task_id: str, pred_path: Path, gold_path: Path, penalty_lambda: float) -> TaskScore:
    prediction_exists = pred_path.exists()
    gold_exists = gold_path.exists()

    if not gold_exists:
        return TaskScore(
            task_id=task_id,
            prediction_exists=prediction_exists,
            gold_exists=False,
            matched_gold_columns=0,
            matched_predicted_columns=0,
            gold_columns=0,
            predicted_columns=0,
            extra_columns=0,
            recall=0.0,
            penalty=0.0,
            score=0.0,
            direct_matches=0,
            name_pair_matches=0,
            missing_prediction=not prediction_exists,
            notes=["missing_gold"],
        )

    if not prediction_exists:
        gold_columns = len(matrix_to_columns(read_csv_matrix(gold_path)))
        return TaskScore(
            task_id=task_id,
            prediction_exists=False,
            gold_exists=True,
            matched_gold_columns=0,
            matched_predicted_columns=0,
            gold_columns=gold_columns,
            predicted_columns=0,
            extra_columns=0,
            recall=0.0,
            penalty=0.0,
            score=0.0,
            direct_matches=0,
            name_pair_matches=0,
            missing_prediction=True,
            notes=["missing_prediction"],
        )

    pred_columns = matrix_to_columns(read_csv_matrix(pred_path))
    gold_columns = matrix_to_columns(read_csv_matrix(gold_path))

    matched_pred, matched_gold, direct_matches = match_direct_columns(pred_columns, gold_columns)
    _, _, pair_matches = match_name_pairs(
        pred_columns,
        gold_columns,
        matched_pred,
        matched_gold,
    )

    matched_gold_columns = len(matched_gold)
    matched_predicted_columns = len(matched_pred)
    gold_column_count = len(gold_columns)
    predicted_column_count = len(pred_columns)
    extra_columns = max(predicted_column_count - matched_predicted_columns, 0)

    recall = (
        matched_gold_columns / gold_column_count
        if gold_column_count > 0
        else 0.0
    )
    penalty = (
        penalty_lambda * (extra_columns / predicted_column_count)
        if predicted_column_count > 0
        else 0.0
    )
    score = max(recall - penalty, 0.0)

    notes: list[str] = []
    if pair_matches:
        notes.append("name_pair_match_used")
    if extra_columns:
        notes.append("extra_columns_present")

    return TaskScore(
        task_id=task_id,
        prediction_exists=True,
        gold_exists=True,
        matched_gold_columns=matched_gold_columns,
        matched_predicted_columns=matched_predicted_columns,
        gold_columns=gold_column_count,
        predicted_columns=predicted_column_count,
        extra_columns=extra_columns,
        recall=recall,
        penalty=penalty,
        score=score,
        direct_matches=direct_matches,
        name_pair_matches=pair_matches,
        missing_prediction=False,
        notes=notes,
    )


def score_run(run_dir: Path, gold_root: Path, penalty_lambda: float) -> dict[str, Any]:
    prediction_tasks = {path.parent.name for path in run_dir.glob("task_*/prediction.csv")}
    gold_tasks = {path.parent.name for path in gold_root.glob("task_*/gold.csv")}
    task_ids = sorted(prediction_tasks | gold_tasks)

    task_scores = [
        score_task(
            task_id,
            run_dir / task_id / "prediction.csv",
            gold_root / task_id / "gold.csv",
            penalty_lambda,
        )
        for task_id in task_ids
    ]

    total_score = sum(task.score for task in task_scores) / len(task_scores) if task_scores else 0.0
    exact_column_coverage = sum(1 for task in task_scores if task.score == 1.0)

    return {
        "run_dir": str(run_dir),
        "gold_root": str(gold_root),
        "penalty_lambda_assumption": penalty_lambda,
        "task_count": len(task_scores),
        "prediction_count": sum(1 for task in task_scores if task.prediction_exists),
        "gold_count": sum(1 for task in task_scores if task.gold_exists),
        "total_score": total_score,
        "perfect_score_count": exact_column_coverage,
        "tasks": [asdict(task) for task in task_scores],
    }


def score_run_against_input(
    run_dir: Path,
    gold_root: Path,
    input_root: Path,
    penalty_lambda: float,
) -> dict[str, Any]:
    input_tasks = {path.name for path in input_root.glob("task_*") if path.is_dir()}
    prediction_tasks = {path.parent.name for path in run_dir.glob("task_*/prediction.csv")}
    gold_tasks = {path.parent.name for path in gold_root.glob("task_*/gold.csv")}
    task_ids = sorted(input_tasks | prediction_tasks | gold_tasks)

    task_scores = [
        score_task(
            task_id,
            run_dir / task_id / "prediction.csv",
            gold_root / task_id / "gold.csv",
            penalty_lambda,
        )
        for task_id in task_ids
    ]

    total_score = sum(task.score for task in task_scores) / len(task_scores) if task_scores else 0.0
    exact_column_coverage = sum(1 for task in task_scores if task.score == 1.0)

    return {
        "run_dir": str(run_dir),
        "gold_root": str(gold_root),
        "penalty_lambda_assumption": penalty_lambda,
        "task_count": len(task_scores),
        "prediction_count": sum(1 for task in task_scores if task.prediction_exists),
        "gold_count": sum(1 for task in task_scores if task.gold_exists),
        "total_score": total_score,
        "perfect_score_count": exact_column_coverage,
        "tasks": [asdict(task) for task in task_scores],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score prediction.csv files using the KDD Cup 2026 column-signature metric."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to an artifacts run directory containing task_*/prediction.csv files.",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=Path("data/public/output"),
        help="Path to the gold root containing task_*/gold.csv files.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/public/input"),
        help="Path to the input root containing the full task_* set to score against.",
    )
    parser.add_argument(
        "--lambda-weight",
        type=float,
        default=0.1,
        help=(
            "Penalty lambda. The official rules mention a preset constant but do not publish its value; "
            "default is 0.1 and can be overridden."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path to save the full JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    gold_root = args.gold_root.resolve()
    input_root = args.input_root.resolve()

    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    if not gold_root.is_dir():
        raise SystemExit(f"Gold root not found: {gold_root}")
    if not input_root.is_dir():
        raise SystemExit(f"Input root not found: {input_root}")

    report = score_run_against_input(
        run_dir,
        gold_root,
        input_root,
        penalty_lambda=args.lambda_weight,
    )

    summary = {
        "run_dir": report["run_dir"],
        "gold_root": report["gold_root"],
        "input_root": str(input_root),
        "penalty_lambda_assumption": report["penalty_lambda_assumption"],
        "task_count": report["task_count"],
        "prediction_count": report["prediction_count"],
        "gold_count": report["gold_count"],
        "total_score": report["total_score"],
        "perfect_score_count": report["perfect_score_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nTask scores:")
    for task in report["tasks"]:
        print(
            json.dumps(
                {
                    "task_id": task["task_id"],
                    "score": task["score"],
                    "recall": task["recall"],
                    "extra_columns": task["extra_columns"],
                    "matched_gold_columns": task["matched_gold_columns"],
                    "gold_columns": task["gold_columns"],
                    "notes": task["notes"],
                },
                ensure_ascii=False,
            )
        )

    if args.report_path is not None:
        args.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
