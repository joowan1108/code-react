from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DIFFICULTY_MAP = {
    "simple": "easy",
    "easy": "easy",
    "moderate": "medium",
    "medium": "medium",
    "challenging": "hard",
    "hard": "hard",
    "extreme": "extreme",
}

SQL_KEYS = ("SQL", "sql", "query", "gold_sql", "answer_sql")
QUESTION_KEYS = ("question", "utterance")
DB_ID_KEYS = ("db_id", "database_id", "db")
DIFFICULTY_KEYS = ("difficulty", "level", "hardness")
EVIDENCE_KEYS = ("evidence", "external_knowledge", "knowledge", "hint")
QUESTION_ID_KEYS = ("question_id", "id", "instance_id")


@dataclass(frozen=True, slots=True)
class ConvertedTask:
    task_id: str
    source_question_id: str | int | None
    db_id: str
    difficulty: str
    question: str
    sql: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _normalize_question_for_overlap(value: Any) -> str:
    text = str(value).casefold()
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _question_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize_question_for_overlap(value)))


def _question_token_overlap(left: Any, right: Any) -> float:
    left_tokens = _question_tokens(left)
    right_tokens = _question_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def _load_excluded_questions(input_roots: list[Path]) -> dict[str, list[str]]:
    excluded: dict[str, list[str]] = {}
    for input_root in input_roots:
        if input_root.is_file():
            task_json_paths = [input_root]
        else:
            shallow_task_paths = sorted(input_root.glob("task_*/task.json"))
            task_json_paths = shallow_task_paths or sorted(input_root.rglob("task.json"))
        for task_json_path in task_json_paths:
            try:
                payload = _read_json(task_json_path)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or "question" not in payload:
                continue
            normalized = _normalize_question_for_overlap(payload["question"])
            if not normalized:
                continue
            excluded.setdefault(normalized, []).append(str(task_json_path))
    return excluded


def _question_exclusion_match(
    question: Any,
    excluded_questions: dict[str, list[str]],
    *,
    near_threshold: float,
) -> dict[str, Any] | None:
    normalized = _normalize_question_for_overlap(question)
    if normalized in excluded_questions:
        return {
            "match_type": "exact_question",
            "overlap": 1.0,
            "matched_question": normalized,
            "matched_sources": excluded_questions[normalized],
        }

    if near_threshold <= 0:
        return None

    best_question: str | None = None
    best_overlap = 0.0
    for candidate in excluded_questions:
        overlap = _question_token_overlap(normalized, candidate)
        if overlap > best_overlap:
            best_overlap = overlap
            best_question = candidate

    if best_question is not None and best_overlap >= near_threshold:
        return {
            "match_type": "near_question",
            "overlap": best_overlap,
            "matched_question": best_question,
            "matched_sources": excluded_questions[best_question],
        }
    return None


def _normalize_difficulty(value: Any, default: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip().casefold()
    return DIFFICULTY_MAP.get(normalized, default)


def _parse_difficulty_quotas(values: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--difficulty-quota must be formatted as difficulty=count, "
                f"got {value!r}."
            )
        difficulty_raw, count_raw = value.split("=", 1)
        difficulty = _normalize_difficulty(difficulty_raw, default="")
        if difficulty not in {"easy", "medium", "hard", "extreme"}:
            raise ValueError(
                "--difficulty-quota difficulty must be one of easy, medium, hard, extreme; "
                f"got {difficulty_raw!r}."
            )
        try:
            count = int(count_raw)
        except ValueError as exc:
            raise ValueError(
                f"--difficulty-quota count must be an integer, got {count_raw!r}."
            ) from exc
        if count < 0:
            raise ValueError("--difficulty-quota count must be non-negative.")
        quotas[difficulty] = quotas.get(difficulty, 0) + count
    return quotas


def _find_split_json(bird_root: Path, split: str) -> Path:
    split_path = Path(split)
    if split_path.suffix == ".json":
        candidate = split_path if split_path.is_absolute() else bird_root / split_path
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Split JSON not found: {candidate}")

    candidates = [
        # BIRD minidev_0703 layout:
        #   minidev/MINIDEV/mini_dev_sqlite.json
        bird_root / "minidev" / "MINIDEV" / "mini_dev_sqlite.json",
        bird_root / "MINIDEV" / "mini_dev_sqlite.json",
        bird_root / "mini_dev_sqlite.json",
        bird_root / split / f"{split}.json",
        bird_root / f"{split}.json",
        bird_root / split / f"{split}_questions.json",
        bird_root / f"{split}_questions.json",
        bird_root / split / f"{split}_data.json",
        bird_root / f"{split}_data.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if split.casefold() in {"dev", "minidev", "mini_dev"}:
        sqlite_matches = sorted(
            path
            for path in bird_root.rglob("*dev*sqlite*.json")
            if "__MACOSX" not in path.parts and not path.name.startswith("._")
        )
        if sqlite_matches:
            return sqlite_matches[0]

    matches = sorted(
        path
        for path in bird_root.rglob(f"{split}*.json")
        if "__MACOSX" not in path.parts
        and not path.name.startswith("._")
        and path.name not in {"dev_tables.json", "tables.json"}
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find a JSON file for split '{split}' under {bird_root}."
    )


def _iter_database_roots(bird_root: Path, split: str, extra_roots: list[Path]) -> list[Path]:
    roots = [
        *(root if root.is_absolute() else bird_root / root for root in extra_roots),
        # BIRD minidev_0703 layout.
        bird_root / "minidev" / "MINIDEV" / "dev_databases",
        bird_root / "MINIDEV" / "dev_databases",
        bird_root / split / f"{split}_databases",
        bird_root / f"{split}_databases",
        bird_root / split / "databases",
        bird_root / split / "database",
        bird_root / "dev_databases",
        bird_root / "databases",
        bird_root / "database",
        bird_root,
    ]
    seen: set[Path] = set()
    unique_roots: list[Path] = []
    for root in roots:
        resolved = root.resolve() if root.exists() else root
        if resolved in seen or not root.exists():
            continue
        seen.add(resolved)
        unique_roots.append(root)
    return unique_roots


def _find_database_path(
    *,
    db_id: str,
    bird_root: Path,
    split: str,
    database_roots: list[Path],
    cache: dict[str, Path],
) -> Path:
    if db_id in cache:
        return cache[db_id]

    filenames = (f"{db_id}.sqlite", f"{db_id}.db")
    direct_candidates: list[Path] = []
    for root in _iter_database_roots(bird_root, split, database_roots):
        direct_candidates.extend(root / db_id / filename for filename in filenames)
        direct_candidates.extend(root / filename for filename in filenames)

    for candidate in direct_candidates:
        if candidate.is_file():
            cache[db_id] = candidate
            return candidate

    for root in _iter_database_roots(bird_root, split, database_roots):
        for filename in filenames:
            matches = sorted(root.rglob(filename))
            if matches:
                cache[db_id] = matches[0]
                return matches[0]

    raise FileNotFoundError(f"Could not find SQLite database for db_id={db_id!r}.")


def _find_database_description_dir(db_path: Path) -> Path | None:
    candidate = db_path.parent / "database_description"
    if candidate.is_dir():
        return candidate
    return None


def _render_database_description(description_dir: Path | None, *, max_rows_per_table: int) -> list[str]:
    if description_dir is None:
        return []

    lines = ["", "## Database Column Descriptions", ""]
    csv_paths = sorted(
        path for path in description_dir.glob("*.csv") if not path.name.startswith("._")
    )
    if not csv_paths:
        return []

    for csv_path in csv_paths:
        lines.extend([f"### {csv_path.stem}", ""])
        try:
            with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            continue

        for row in rows[:max_rows_per_table]:
            original_name = (row.get("original_column_name") or "").strip()
            semantic_name = (row.get("column_name") or "").strip()
            description = (row.get("column_description") or "").strip()
            data_format = (row.get("data_format") or "").strip()
            value_description = (row.get("value_description") or "").strip()
            rendered_parts = [part for part in (original_name, semantic_name) if part]
            if not rendered_parts:
                continue
            label = " / ".join(rendered_parts)
            details = []
            if description:
                details.append(description)
            if data_format:
                details.append(f"format={data_format}")
            if value_description:
                details.append(f"values={value_description}")
            suffix = f": {'; '.join(details)}" if details else ""
            lines.append(f"- {label}{suffix}")
        if len(rows) > max_rows_per_table:
            lines.append(f"- ... omitted {len(rows) - max_rows_per_table} more columns")
        lines.append("")

    return lines


def _ensure_records(payload: Any, split_json: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("data", "questions", "records", "examples", "instances"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            raise ValueError(
                f"{split_json} is a JSON object, but no list field such as "
                "'data' or 'questions' was found."
            )
    else:
        raise ValueError(f"{split_json} must contain a list or object with a list field.")

    bad_items = [index for index, item in enumerate(records) if not isinstance(item, dict)]
    if bad_items:
        raise ValueError(f"{split_json} contains non-object records at indexes {bad_items[:5]}.")
    return list(records)


def _execute_gold_sql(db_path: Path, sql: str) -> tuple[list[str], list[list[Any]]]:
    with sqlite3.connect(str(db_path)) as connection:
        cursor = connection.execute(sql)
        rows = cursor.fetchall()
        columns = [
            description[0] if description[0] else f"col_{index + 1}"
            for index, description in enumerate(cursor.description or [])
        ]
    if not columns:
        columns = ["result"]
    return columns, [list(row) for row in rows]


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_task_json(path: Path, *, task_id: str, difficulty: str, question: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "difficulty": difficulty,
        "question": question,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_or_link_file(source: Path, destination: Path, *, hardlink: bool) -> None:
    if hardlink:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _write_knowledge(
    path: Path,
    *,
    evidence: Any,
    db_id: str,
    original_record: dict[str, Any],
    database_description_dir: Path | None,
    max_description_rows_per_table: int,
) -> None:
    lines = [f"# BIRD Evidence for {db_id}", ""]
    if evidence:
        if isinstance(evidence, list):
            for item in evidence:
                lines.append(f"- {item}")
        else:
            lines.append(str(evidence))
    else:
        lines.append("No explicit BIRD evidence was provided for this question.")

    extra_knowledge = {
        key: value
        for key, value in original_record.items()
        if key.lower() in {"evidence", "external_knowledge", "knowledge", "hint"}
    }
    if extra_knowledge and len(extra_knowledge) > 1:
        lines.extend(["", "## Raw Knowledge Fields", ""])
        lines.append(json.dumps(extra_knowledge, ensure_ascii=False, indent=2))

    lines.extend(
        _render_database_description(
            database_description_dir,
            max_rows_per_table=max_description_rows_per_table,
        )
    )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_benchmark_config(args: argparse.Namespace) -> Path | None:
    if args.config_out is None:
        return None

    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None

    base_payload: dict[str, Any] = {}
    if args.base_config is not None and args.base_config.is_file():
        loaded = yaml.safe_load(args.base_config.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            base_payload = loaded

    def config_path(path: Path) -> str:
        try:
            return path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return path.as_posix()

    base_payload.setdefault("dataset", {})
    base_payload["dataset"]["root_path"] = config_path(args.out_root / "input")

    base_payload.setdefault("run", {})
    base_payload["run"]["output_dir"] = config_path(Path("artifacts") / "runs")
    # Avoid reusing example_run_id across BIRD test runs.
    base_payload["run"]["run_id"] = None

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        yaml.safe_dump(base_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return args.config_out


def _convert_record(
    record: dict[str, Any],
    *,
    index: int,
    args: argparse.Namespace,
    split: str,
    db_cache: dict[str, Path],
) -> ConvertedTask | None:
    question = _first_present(record, QUESTION_KEYS)
    db_id = _first_present(record, DB_ID_KEYS)
    sql = _first_present(record, SQL_KEYS)
    if question is None or db_id is None:
        raise ValueError(
            f"Record {index} is missing question or db_id. Keys: {sorted(record.keys())}"
        )
    if sql is None:
        if args.skip_missing_sql:
            return None
        raise ValueError(f"Record {index} is missing gold SQL. Keys: {sorted(record.keys())}")

    task_number = args.task_id_offset + index
    task_id = f"task_{task_number}"
    difficulty = _normalize_difficulty(
        _first_present(record, DIFFICULTY_KEYS),
        args.default_difficulty,
    )
    db_id_text = str(db_id)
    sql_text = str(sql)
    question_text = str(question)
    source_question_id = _first_present(record, QUESTION_ID_KEYS)

    db_path = _find_database_path(
        db_id=db_id_text,
        bird_root=args.bird_root,
        split=split,
        database_roots=args.database_root,
        cache=db_cache,
    )
    database_description_dir = _find_database_description_dir(db_path)

    task_input_dir = args.out_root / "input" / task_id
    context_dir = task_input_dir / "context"
    db_out_dir = context_dir / "db"
    db_out_dir.mkdir(parents=True, exist_ok=True)
    _copy_or_link_file(db_path, db_out_dir / db_path.name, hardlink=args.hardlink_databases)

    _write_task_json(
        task_input_dir / "task.json",
        task_id=task_id,
        difficulty=difficulty,
        question=question_text,
    )

    if args.write_knowledge:
        _write_knowledge(
            context_dir / "knowledge.md",
            evidence=_first_present(record, EVIDENCE_KEYS),
            db_id=db_id_text,
            original_record=record,
            database_description_dir=(
                None if args.no_database_description else database_description_dir
            ),
            max_description_rows_per_table=args.max_description_rows_per_table,
        )

    metadata = {
        "source": "BIRD",
        "split": split,
        "source_question_id": source_question_id,
        "db_id": db_id_text,
        "source_db_path": str(db_path),
        "source_database_description_dir": (
            str(database_description_dir) if database_description_dir is not None else None
        ),
        "gold_sql": sql_text,
        "original_record": record,
    }
    (task_input_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        columns, rows = _execute_gold_sql(db_path, sql_text)
    except Exception as exc:  # noqa: BLE001
        if not args.skip_sql_errors:
            raise RuntimeError(
                f"Gold SQL failed for record {index}, task_id={task_id}, db_id={db_id_text}: {exc}"
            ) from exc
        error_payload = {
            "task_id": task_id,
            "db_id": db_id_text,
            "source_question_id": source_question_id,
            "sql": sql_text,
            "error": str(exc),
        }
        args.out_root.joinpath("sql_errors.jsonl").parent.mkdir(parents=True, exist_ok=True)
        with (args.out_root / "sql_errors.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(error_payload, ensure_ascii=False) + "\n")
        shutil.rmtree(task_input_dir)
        return None

    gold_path = args.out_root / "output" / task_id / "gold.csv"
    _write_csv(gold_path, columns, rows)

    return ConvertedTask(
        task_id=task_id,
        source_question_id=source_question_id,
        db_id=db_id_text,
        difficulty=difficulty,
        question=question_text,
        sql=sql_text,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BIRD train/dev examples into the DABench public input/output "
            "directory structure used by this starter kit."
        )
    )
    parser.add_argument("--bird-root", type=Path, required=True, help="Path to the BIRD dataset root.")
    parser.add_argument(
        "--split",
        default="dev",
        help="Split name such as dev/train, or a relative/absolute JSON file path.",
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        action="append",
        default=[],
        help="Optional database root. Can be passed more than once.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/bird_eval"),
        help="Output root containing input/ and output/ directories.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum records to convert.")
    parser.add_argument("--start", type=int, default=0, help="Start index in the split JSON.")
    parser.add_argument(
        "--exclude-input-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "DABench-style input root or task.json file whose questions should be "
            "excluded from the converted BIRD set. Can be passed more than once."
        ),
    )
    parser.add_argument(
        "--exclude-near-threshold",
        type=float,
        default=0.0,
        help=(
            "Also exclude questions whose token overlap with an excluded question "
            "is at least this value. Use 0 to disable near-duplicate filtering."
        ),
    )
    parser.add_argument(
        "--task-id-offset",
        type=int,
        default=100000,
        help="Numeric offset for generated task IDs.",
    )
    parser.add_argument(
        "--default-difficulty",
        choices=("easy", "medium", "hard", "extreme"),
        default="medium",
        help="Difficulty used when the BIRD record does not provide one.",
    )
    parser.add_argument(
        "--difficulty-quota",
        action="append",
        default=[],
        metavar="DIFFICULTY=COUNT",
        help=(
            "Only convert up to COUNT records for a normalized difficulty. "
            "Can be passed multiple times, e.g. --difficulty-quota easy=10 "
            "--difficulty-quota medium=15. Difficulties not listed are skipped."
        ),
    )
    parser.add_argument(
        "--no-knowledge",
        dest="write_knowledge",
        action="store_false",
        help="Do not write BIRD evidence to context/knowledge.md.",
    )
    parser.add_argument(
        "--skip-missing-sql",
        action="store_true",
        help="Skip records without a gold SQL field instead of failing.",
    )
    parser.add_argument(
        "--skip-sql-errors",
        action="store_true",
        help="Skip records whose gold SQL cannot be executed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output root before conversion.",
    )
    parser.add_argument(
        "--hardlink-databases",
        action="store_true",
        help=(
            "Use filesystem hardlinks for copied SQLite database files when possible. "
            "This saves disk space for local evaluation sets with many tasks sharing the same DB."
        ),
    )
    parser.add_argument(
        "--no-database-description",
        action="store_true",
        help="Do not append BIRD database_description/*.csv column descriptions to knowledge.md.",
    )
    parser.add_argument(
        "--max-description-rows-per-table",
        type=int,
        default=80,
        help="Maximum database_description rows to include per table in knowledge.md.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/react_baseline.example.yaml"),
        help="Base YAML config to copy when writing a BIRD run config.",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help=(
            "Optional YAML config path to write. If omitted, defaults to "
            "<out-root>/react_baseline.bird.yaml."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.bird_root = args.bird_root.resolve()
    args.out_root = args.out_root.resolve()
    args.exclude_input_root = [
        path.resolve() if path.exists() else path for path in args.exclude_input_root
    ]
    if not 0.0 <= args.exclude_near_threshold <= 1.0:
        raise ValueError("--exclude-near-threshold must be between 0 and 1.")
    difficulty_quotas = _parse_difficulty_quotas(args.difficulty_quota)

    if args.overwrite and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.config_out is None:
        args.config_out = args.out_root / "react_baseline.bird.yaml"
    if args.base_config is not None and not args.base_config.is_absolute():
        args.base_config = Path.cwd() / args.base_config

    split_json = _find_split_json(args.bird_root, args.split)
    split_name = Path(args.split).stem if Path(args.split).suffix == ".json" else args.split
    records = _ensure_records(_read_json(split_json), split_json)
    excluded_questions = _load_excluded_questions(args.exclude_input_root)

    db_cache: dict[str, Path] = {}
    converted: list[ConvertedTask] = []
    converted_by_difficulty: dict[str, int] = {difficulty: 0 for difficulty in difficulty_quotas}
    skipped_question_overlaps: list[dict[str, Any]] = []
    skipped_difficulty_quota_count = 0
    considered_count = 0
    for local_index, record in enumerate(records[args.start :], start=args.start):
        if args.limit is not None and len(converted) >= args.limit:
            break
        if difficulty_quotas and all(
            converted_by_difficulty.get(difficulty, 0) >= count
            for difficulty, count in difficulty_quotas.items()
        ):
            break
        considered_count += 1
        question = _first_present(record, QUESTION_KEYS)
        if question is not None and excluded_questions:
            exclusion_match = _question_exclusion_match(
                question,
                excluded_questions,
                near_threshold=args.exclude_near_threshold,
            )
            if exclusion_match is not None:
                skipped_question_overlaps.append(
                    {
                        "record_index": local_index,
                        "source_question_id": _first_present(record, QUESTION_ID_KEYS),
                        "db_id": _first_present(record, DB_ID_KEYS),
                        "question": str(question),
                        **exclusion_match,
                    }
                )
                continue
        difficulty = _normalize_difficulty(
            _first_present(record, DIFFICULTY_KEYS),
            args.default_difficulty,
        )
        if difficulty_quotas:
            if difficulty not in difficulty_quotas:
                skipped_difficulty_quota_count += 1
                continue
            if converted_by_difficulty.get(difficulty, 0) >= difficulty_quotas[difficulty]:
                skipped_difficulty_quota_count += 1
                continue
        task = _convert_record(
            record,
            index=local_index,
            args=args,
            split=split_name,
            db_cache=db_cache,
        )
        if task is not None:
            converted.append(task)
            if difficulty_quotas:
                converted_by_difficulty[task.difficulty] = (
                    converted_by_difficulty.get(task.difficulty, 0) + 1
                )

    manifest = {
        "source": "BIRD",
        "split_json": str(split_json),
        "record_start": args.start,
        "requested_limit": args.limit,
        "difficulty_quotas": difficulty_quotas,
        "converted_by_difficulty": (
            converted_by_difficulty
            if difficulty_quotas
            else {
                difficulty: sum(1 for task in converted if task.difficulty == difficulty)
                for difficulty in ("easy", "medium", "hard", "extreme")
            }
        ),
        "considered_count": considered_count,
        "converted_count": len(converted),
        "exclude_input_roots": [str(path) for path in args.exclude_input_root],
        "exclude_near_threshold": args.exclude_near_threshold,
        "excluded_question_count": len(excluded_questions),
        "skipped_question_overlap_count": len(skipped_question_overlaps),
        "skipped_difficulty_quota_count": skipped_difficulty_quota_count,
        "skipped_question_overlaps": skipped_question_overlaps,
        "input_root": str(args.out_root / "input"),
        "gold_root": str(args.out_root / "output"),
        "tasks": [
            {
                "task_id": task.task_id,
                "source_question_id": task.source_question_id,
                "db_id": task.db_id,
                "difficulty": task.difficulty,
                "question": task.question,
            }
            for task in converted
        ],
    }
    (args.out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_path = _write_benchmark_config(args)

    print(f"Converted {len(converted)} BIRD records.")
    print(f"Input root: {args.out_root / 'input'}")
    print(f"Gold root:  {args.out_root / 'output'}")
    print(f"Manifest:   {args.out_root / 'manifest.json'}")
    if config_path is not None:
        print(f"Config:     {config_path}")
    else:
        print("Config:     not written because PyYAML is unavailable")
    print()
    print("Run example:")
    if config_path is not None:
        print(f"  uv run dabench run-benchmark --config {config_path}")
    else:
        print("  Copy configs/react_baseline.example.yaml and set dataset.root_path to:")
        print(f"    {args.out_root / 'input'}")
    print("Score example:")
    print(
        "  python scripts/score_predictions.py artifacts/runs/<RUN_ID> "
        f"--input-root {args.out_root / 'input'} --gold-root {args.out_root / 'output'}"
    )


if __name__ == "__main__":
    main()
