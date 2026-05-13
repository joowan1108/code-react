from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

MAX_MANIFEST_CHARS = 5000
MAX_FILES = 80
MAX_COLUMNS = 28
MAX_TABLES = 20
MAX_JSON_PARSE_BYTES = 8_000_000


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _format_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "unknown"
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _trim_items(items: list[str], *, max_items: int = MAX_COLUMNS) -> str:
    if len(items) <= max_items:
        return ", ".join(items)
    kept = items[:max_items]
    return f"{', '.join(kept)}, ... (+{len(items) - len(kept)} more)"


def _csv_manifest(path: Path, root: Path) -> str:
    rel_path = _relative_path(path, root)
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader, [])
    except Exception as exc:  # noqa: BLE001
        return f"- {rel_path} [csv, {_format_size(path)}]: unreadable ({exc})"

    if columns:
        return f"- {rel_path} [csv, {_format_size(path)}]: columns={_trim_items([str(c) for c in columns])}"
    return f"- {rel_path} [csv, {_format_size(path)}]: no header detected"


def _json_columns_from_records(records: object) -> tuple[int | None, list[str]]:
    if not isinstance(records, list):
        return None, []

    columns: list[str] = []
    seen: set[str] = set()
    for record in records[:5]:
        if not isinstance(record, dict):
            continue
        for key in record:
            if key in seen:
                continue
            seen.add(str(key))
            columns.append(str(key))
    return len(records), columns


def _json_manifest_from_prefix(path: Path, root: Path) -> str:
    rel_path = _relative_path(path, root)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:200_000]
    except Exception as exc:  # noqa: BLE001
        return f"- {rel_path} [json, {_format_size(path)}]: unreadable ({exc})"

    table_match = re.search(r'"table"\s*:\s*"([^"]+)"', text)
    table = table_match.group(1) if table_match else None
    key_matches = re.findall(r'"([^"]+)"\s*:', text)
    columns: list[str] = []
    seen: set[str] = set()
    for key in key_matches:
        if key in {"table", "records"} or key in seen:
            continue
        seen.add(key)
        columns.append(key)
        if len(columns) >= MAX_COLUMNS:
            break

    parts = [f"- {rel_path} [json, {_format_size(path)}, large-prefix]"]
    if table:
        parts.append(f"table={table}")
    if columns:
        parts.append(f"observed_keys={_trim_items(columns)}")
    return ": ".join([parts[0], "; ".join(parts[1:])]) if len(parts) > 1 else parts[0]


def _json_manifest(path: Path, root: Path) -> str:
    rel_path = _relative_path(path, root)
    try:
        if path.stat().st_size > MAX_JSON_PARSE_BYTES:
            return _json_manifest_from_prefix(path, root)
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return _json_manifest_from_prefix(path, root)

    if isinstance(data, dict):
        table = data.get("table")
        record_count, columns = _json_columns_from_records(data.get("records"))
        if columns:
            pieces = [f"- {rel_path} [json, {_format_size(path)}]"]
            if table is not None:
                pieces.append(f"table={table}")
            if record_count is not None:
                pieces.append(f"records={record_count}")
            pieces.append(f"columns={_trim_items(columns)}")
            return ": ".join([pieces[0], "; ".join(pieces[1:])])
        return f"- {rel_path} [json, {_format_size(path)}]: keys={_trim_items([str(k) for k in data.keys()])}"

    if isinstance(data, list):
        _, columns = _json_columns_from_records(data)
        if columns:
            return f"- {rel_path} [json, {_format_size(path)}]: records={len(data)}; columns={_trim_items(columns)}"
        return f"- {rel_path} [json, {_format_size(path)}]: list_len={len(data)}"

    return f"- {rel_path} [json, {_format_size(path)}]: type={type(data).__name__}"


def _sqlite_manifest(path: Path, root: Path) -> list[str]:
    rel_path = _relative_path(path, root)
    lines = [f"- {rel_path} [sqlite, {_format_size(path)}]"]
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except Exception as exc:  # noqa: BLE001
        return [f"{lines[0]}: unreadable ({exc})"]

    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        for table in tables[:MAX_TABLES]:
            quoted_table = '"' + table.replace('"', '""') + '"'
            columns = [
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({quoted_table})").fetchall()
            ]
            lines.append(f"  - {table}: columns={_trim_items(columns)}")
        if len(tables) > MAX_TABLES:
            lines.append(f"  - ... (+{len(tables) - MAX_TABLES} more tables)")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  - schema unreadable ({exc})")
    finally:
        connection.close()
    return lines


def _doc_manifest(path: Path, root: Path) -> str:
    rel_path = _relative_path(path, root)
    heading = ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(40):
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped.startswith("#"):
                    heading = stripped[:120]
                    break
    except Exception:
        heading = ""

    if heading:
        return f"- {rel_path} [doc, {_format_size(path)}]: heading={heading}"
    return f"- {rel_path} [doc, {_format_size(path)}]"


def build_context_manifest(context_dir: Path) -> str:
    """Build a small schema/file manifest for the first model prompt.

    This is intentionally shallow: it exposes file paths and schema-like metadata
    without reading full dataframes or documents into the prompt.
    """

    root = context_dir.resolve()
    try:
        files = sorted(path for path in root.rglob("*") if path.is_file())
    except Exception as exc:  # noqa: BLE001
        return f"- context manifest unavailable: {exc}"

    lines = ["Concise context manifest:"]
    for path in files[:MAX_FILES]:
        suffix = path.suffix.casefold()
        if suffix == ".csv":
            lines.append(_csv_manifest(path, root))
        elif suffix == ".json":
            lines.append(_json_manifest(path, root))
        elif suffix in {".sqlite", ".sqlite3", ".db"}:
            lines.extend(_sqlite_manifest(path, root))
        elif suffix in {".md", ".txt"}:
            lines.append(_doc_manifest(path, root))
        else:
            lines.append(f"- {_relative_path(path, root)} [{suffix.lstrip('.') or 'file'}, {_format_size(path)}]")

        rendered = "\n".join(lines)
        if len(rendered) > MAX_MANIFEST_CHARS:
            lines[-1] = "[manifest truncated to stay concise]"
            break

    if len(files) > MAX_FILES and len("\n".join(lines)) < MAX_MANIFEST_CHARS:
        lines.append(f"- ... (+{len(files) - MAX_FILES} more files)")

    rendered = "\n".join(lines)
    if len(rendered) > MAX_MANIFEST_CHARS:
        return rendered[:MAX_MANIFEST_CHARS] + "\n[manifest truncated]"
    return rendered