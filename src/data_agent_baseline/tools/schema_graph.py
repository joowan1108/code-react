from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

MAX_JSON_SCHEMA_BYTES = 10_000_000
MAX_VALUE_SET_SIZE = 2_000


@dataclass(slots=True)
class _ColumnProfile:
    values: set[str] = field(default_factory=set)
    non_null_count: int = 0


@dataclass(slots=True)
class _TableProfile:
    name: str
    source: str
    kind: str
    columns: list[str]
    rows_sampled: int = 0
    sqlite_types: dict[str, str] = field(default_factory=dict)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[str] = field(default_factory=list)
    inferred_pk: list[str] = field(default_factory=list)
    column_profiles: dict[str, _ColumnProfile] = field(default_factory=dict)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _is_linkish_name(value: str) -> bool:
    normalized = value.casefold()
    compact = _normalize_name(value)
    return (
        compact == "id"
        or compact.endswith("id")
        or compact.endswith("key")
        or compact.endswith("code")
        or "uuid" in compact
        or normalized.endswith("_id")
        or normalized.endswith("-id")
        or normalized.endswith(" id")
    )


def _clean_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
    else:
        cleaned = str(value).strip()
    if not cleaned:
        return None
    return cleaned.casefold()


def _add_profile_value(profile: _ColumnProfile, value: object) -> None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return
    profile.non_null_count += 1
    if len(profile.values) < MAX_VALUE_SET_SIZE:
        profile.values.add(cleaned)


def _unique_table_name(raw_name: str, used_names: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name).strip("_") or "table"
    candidate = base
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _infer_pk_candidates(table: _TableProfile, *, max_candidates: int = 5) -> list[str]:
    if table.primary_key:
        return []
    if table.rows_sampled < 2:
        return []

    candidates: list[tuple[float, str]] = []
    for column in table.columns:
        profile = table.column_profiles.get(column)
        if profile is None or profile.non_null_count == 0:
            continue
        non_null_ratio = profile.non_null_count / max(table.rows_sampled, 1)
        unique_ratio = len(profile.values) / max(profile.non_null_count, 1)
        if non_null_ratio < 0.95 or unique_ratio < 0.98:
            continue
        name_bonus = 0.2 if _is_linkish_name(column) else 0.0
        candidates.append((unique_ratio + name_bonus, column))

    candidates.sort(key=lambda item: (-item[0], table.columns.index(item[1])))
    return [column for _, column in candidates[:max_candidates]]


def _profile_records(
    *,
    name: str,
    source: str,
    kind: str,
    columns: list[str],
    records: list[dict[str, object]],
    max_sample_rows: int,
) -> _TableProfile:
    limited_columns = [str(column) for column in columns]
    table = _TableProfile(
        name=name,
        source=source,
        kind=kind,
        columns=limited_columns,
        column_profiles={column: _ColumnProfile() for column in limited_columns},
    )
    for record in records[:max_sample_rows]:
        table.rows_sampled += 1
        for column in limited_columns:
            _add_profile_value(table.column_profiles[column], record.get(column))
    table.inferred_pk = _infer_pk_candidates(table)
    return table


def _read_csv_profile(
    path: Path,
    root: Path,
    used_names: set[str],
    *,
    max_sample_rows: int,
) -> _TableProfile | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = [str(column) for column in (reader.fieldnames or [])]
            records: list[dict[str, object]] = []
            for index, row in enumerate(reader):
                if index >= max_sample_rows:
                    break
                records.append(dict(row))
    except Exception:  # noqa: BLE001
        return None

    if not columns:
        return None
    return _profile_records(
        name=_unique_table_name(path.stem, used_names),
        source=_relative_path(path, root),
        kind="csv",
        columns=columns,
        records=records,
        max_sample_rows=max_sample_rows,
    )


def _records_columns(records: list[object], *, max_columns: int) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in record:
            column = str(key)
            if column in seen:
                continue
            seen.add(column)
            columns.append(column)
            if len(columns) >= max_columns:
                return columns
    return columns


def _json_record_tables(data: object, default_name: str, *, max_columns: int) -> list[tuple[str, list[dict[str, object]], list[str]]]:
    tables: list[tuple[str, list[dict[str, object]], list[str]]] = []
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        records = [record for record in data["records"] if isinstance(record, dict)]
        table_name = str(data.get("table") or default_name)
        tables.append((table_name, records, _records_columns(records, max_columns=max_columns)))
        return tables

    if isinstance(data, list):
        records = [record for record in data if isinstance(record, dict)]
        tables.append((default_name, records, _records_columns(records, max_columns=max_columns)))
        return tables

    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(value, list):
                continue
            records = [record for record in value if isinstance(record, dict)]
            if not records:
                continue
            tables.append((str(key), records, _records_columns(records, max_columns=max_columns)))
    return tables


def _read_json_profiles(
    path: Path,
    root: Path,
    used_names: set[str],
    *,
    max_sample_rows: int,
    max_columns: int,
) -> list[_TableProfile]:
    try:
        if path.stat().st_size > MAX_JSON_SCHEMA_BYTES:
            return []
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return []

    profiles: list[_TableProfile] = []
    for raw_name, records, columns in _json_record_tables(data, path.stem, max_columns=max_columns):
        if not columns:
            continue
        profiles.append(
            _profile_records(
                name=_unique_table_name(raw_name, used_names),
                source=_relative_path(path, root),
                kind="json",
                columns=columns,
                records=records,
                max_sample_rows=max_sample_rows,
            )
        )
    return profiles


def _read_sqlite_profiles(
    path: Path,
    root: Path,
    used_names: set[str],
    *,
    max_sample_rows: int,
    max_columns: int,
) -> list[_TableProfile]:
    profiles: list[_TableProfile] = []
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001
        return profiles

    try:
        table_rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        for (raw_table_name,) in table_rows:
            sqlite_table_name = str(raw_table_name)
            display_name = _unique_table_name(sqlite_table_name, used_names)
            quoted_table = _quote_sqlite_identifier(sqlite_table_name)
            pragma_rows = connection.execute(f"PRAGMA table_info({quoted_table})").fetchall()
            columns = [str(row[1]) for row in pragma_rows[:max_columns]]
            table = _TableProfile(
                name=display_name,
                source=_relative_path(path, root),
                kind="sqlite",
                columns=columns,
                sqlite_types={str(row[1]): str(row[2]) for row in pragma_rows[:max_columns] if row[2]},
                primary_key=[str(row[1]) for row in pragma_rows if row[5]],
                column_profiles={column: _ColumnProfile() for column in columns},
            )
            for row in connection.execute(f"PRAGMA foreign_key_list({quoted_table})").fetchall():
                _, _, ref_table, from_column, to_column, *_ = row
                table.foreign_keys.append(f"{from_column}->{ref_table}.{to_column}")

            if columns:
                quoted_columns = ", ".join(_quote_sqlite_identifier(column) for column in columns)
                try:
                    sample_rows = connection.execute(
                        f"SELECT {quoted_columns} FROM {quoted_table} LIMIT ?",
                        (max_sample_rows,),
                    ).fetchall()
                except Exception:  # noqa: BLE001
                    sample_rows = []
                for sample in sample_rows:
                    table.rows_sampled += 1
                    for column, value in zip(columns, sample):
                        _add_profile_value(table.column_profiles[column], value)
            profiles.append(table)
    finally:
        connection.close()
    return profiles


def _name_similarity(left_table: _TableProfile, left_column: str, right_table: _TableProfile, right_column: str) -> float:
    left = _normalize_name(left_column)
    right = _normalize_name(right_column)
    if not left or not right:
        return 0.0
    if left == right:
        return 0.25
    right_table_name = _normalize_name(right_table.name)
    left_table_name = _normalize_name(left_table.name)
    if right_table_name and left in {f"{right_table_name}{right}", f"{right_table_name}id"}:
        return 0.2
    if left_table_name and right in {f"{left_table_name}{left}", f"{left_table_name}id"}:
        return 0.1
    if left.endswith(right) or right.endswith(left):
        return 0.1
    return 0.0


def _build_join_candidates(tables: list[_TableProfile], *, max_join_candidates: int) -> list[str]:
    scored: list[tuple[float, str]] = []
    for left in tables:
        for right in tables:
            if left is right:
                continue
            for left_column in left.columns:
                left_profile = left.column_profiles.get(left_column)
                if left_profile is None or not left_profile.values:
                    continue
                for right_column in right.columns:
                    right_profile = right.column_profiles.get(right_column)
                    if right_profile is None or not right_profile.values:
                        continue
                    overlap = left_profile.values & right_profile.values
                    if len(overlap) < 2:
                        continue
                    left_subset = len(overlap) / max(len(left_profile.values), 1)
                    right_subset = len(overlap) / max(len(right_profile.values), 1)
                    name_score = _name_similarity(left, left_column, right, right_column)
                    right_key = right_column in right.primary_key or right_column in right.inferred_pk
                    left_linkish = _is_linkish_name(left_column)
                    if left_subset < 0.5 and name_score == 0:
                        continue
                    if not (right_key or left_linkish or name_score > 0):
                        continue
                    score = left_subset + name_score + (0.25 if right_key else 0.0)
                    if score < 0.75:
                        continue
                    relation = (
                        f"{left.name}.{left_column} -> {right.name}.{right_column} "
                        f"(overlap={left_subset:.2f}, n={len(overlap)})"
                    )
                    scored.append((score + min(right_subset, 1.0) * 0.05, relation))
    scored.sort(key=lambda item: (-item[0], item[1]))

    deduped: list[str] = []
    seen: set[str] = set()
    for _, relation in scored:
        if relation in seen:
            continue
        seen.add(relation)
        deduped.append(relation)
        if len(deduped) >= max_join_candidates:
            break
    return deduped


def _public_table_payload(table: _TableProfile, *, max_columns: int) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": table.source,
        "kind": table.kind,
        "columns": table.columns[:max_columns],
        "rows_sampled": table.rows_sampled,
    }
    if table.sqlite_types:
        payload["types"] = {column: table.sqlite_types[column] for column in table.columns if column in table.sqlite_types}
    if table.primary_key:
        payload["primary_key"] = table.primary_key
    if table.inferred_pk:
        payload["pk_candidates"] = table.inferred_pk
    if table.foreign_keys:
        payload["foreign_keys"] = table.foreign_keys
    if len(table.columns) > max_columns:
        payload["truncated_columns"] = len(table.columns) - max_columns
    return payload


def inspect_context_schema(
    context_root: str | Path = ".",
    *,
    max_files: int = 80,
    max_sample_rows: int = 1000,
    max_columns: int = 40,
    max_join_candidates: int = 12,
) -> dict[str, object]:
    """Return a compact, value-free schema graph for relational reasoning.

    The helper samples values only to infer PK/FK-like relationships. It does not
    return sample values, so printing the result keeps observations small.
    """

    root = Path(context_root).resolve()
    try:
        files = sorted(path for path in root.rglob("*") if path.is_file())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"context scan failed: {exc}"}

    used_names: set[str] = set()
    tables: list[_TableProfile] = []
    for path in files[:max_files]:
        suffix = path.suffix.casefold()
        if suffix == ".csv":
            profile = _read_csv_profile(path, root, used_names, max_sample_rows=max_sample_rows)
            if profile is not None:
                tables.append(profile)
        elif suffix == ".json":
            tables.extend(
                _read_json_profiles(
                    path,
                    root,
                    used_names,
                    max_sample_rows=max_sample_rows,
                    max_columns=max_columns,
                )
            )
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            tables.extend(
                _read_sqlite_profiles(
                    path,
                    root,
                    used_names,
                    max_sample_rows=max_sample_rows,
                    max_columns=max_columns,
                )
            )

    return {
        "tables": {
            table.name: _public_table_payload(table, max_columns=max_columns)
            for table in tables
        },
        "join_candidates": _build_join_candidates(tables, max_join_candidates=max_join_candidates),
        "notes": [
            "PK/FK for CSV/JSON are inferred from sampled uniqueness and value overlap.",
            "No sample values are returned.",
        ],
    }
