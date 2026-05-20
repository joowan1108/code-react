from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

MAX_MANIFEST_CHARS = 8000
MAX_FILES = 80
MAX_COLUMNS = 28
MAX_TABLES = 20
MAX_FOREIGN_KEYS = 20
MAX_JSON_PARSE_BYTES = 8_000_000
MAX_SCHEMA_HINT_CHARS = 1800
MAX_HINT_COLUMNS = 8
MAX_HINT_TABLES = 5
MAX_HINT_VALUES = 4
MAX_HINT_JOINS = 4
MAX_VALUE_SCAN_ROWS = 5000
MAX_MARKDOWN_HINT_SNIPPETS = 3
MAX_MARKDOWN_LINKS = 8
MAX_MARKDOWN_ALTERNATIVES = 4
MAX_MARKDOWN_HINT_CHARS = 1400

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "many",
    "of",
    "on",
    "or",
    "than",
    "that",
    "the",
    "their",
    "them",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

TOKEN_SYNONYMS = {
    "age": {"year", "years", "old"},
    "amount": {"budget", "cost", "expense", "price", "spend"},
    "budget": {"amount", "cost", "expense", "price"},
    "comic": {"comics"},
    "comics": {"comic"},
    "cost": {"amount", "budget", "expense", "price"},
    "event": {"meeting"},
    "heroes": {"hero", "superhero"},
    "hero": {"heroes", "superhero"},
    "meeting": {"event"},
    "percent": {"percentage", "proportion", "rate", "ratio"},
    "percentage": {"percent", "proportion", "rate", "ratio"},
    "price": {"amount", "budget", "cost"},
    "published": {"publisher"},
    "publisher": {"published"},
    "ratio": {"percentage", "proportion", "rate"},
    "revenue": {"amount", "sales", "total"},
    "size": {"shirt", "tshirt", "t-shirt"},
    "superhero": {"hero", "heroes"},
    "total": {"amount", "sum"},
}

NUMERIC_INTENT_TERMS = {
    "age",
    "amount",
    "average",
    "avg",
    "budget",
    "cost",
    "count",
    "expense",
    "height",
    "many",
    "mean",
    "number",
    "percent",
    "percentage",
    "price",
    "proportion",
    "rate",
    "ratio",
    "sum",
    "times",
    "total",
    "weight",
}

NUMERIC_COLUMN_TERMS = {
    "age",
    "amount",
    "budget",
    "cost",
    "count",
    "height",
    "id",
    "number",
    "price",
    "score",
    "size",
    "total",
    "value",
    "weight",
}

SQLISH_STOPWORDS = {
    "and",
    "as",
    "asc",
    "avg",
    "by",
    "case",
    "count",
    "desc",
    "distinct",
    "else",
    "end",
    "from",
    "group",
    "having",
    "in",
    "is",
    "join",
    "like",
    "max",
    "min",
    "not",
    "null",
    "on",
    "or",
    "order",
    "select",
    "sum",
    "then",
    "when",
    "where",
}


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


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


@dataclass(frozen=True)
class _ColumnRef:
    label: str
    table_label: str
    column: str
    kind: str
    rel_path: str
    tokens: frozenset[str]
    table_tokens: frozenset[str]
    is_numeric: bool = False


@dataclass(frozen=True)
class _JoinEdge:
    label: str
    left_table: str
    right_table: str


@dataclass
class _SchemaInfo:
    columns: list[_ColumnRef]
    join_edges: list[_JoinEdge]
    lookup: dict[tuple[str, str, str, str], _ColumnRef]


def _tokenize_for_linking(text: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", spaced.replace("_", " "))
        if token and token.casefold() not in STOPWORDS
    }
    expanded: set[str] = set()
    for token in tokens:
        expanded.add(token)
        if len(token) > 3 and token.endswith("s"):
            expanded.add(token[:-1])
        expanded.update(TOKEN_SYNONYMS.get(token, set()))
    return expanded


def _compact_identifier(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _identifier_words(text: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return re.findall(r"[A-Za-z0-9]+", spaced.replace("_", " "))


def _question_terms(question: str) -> set[str]:
    terms = _tokenize_for_linking(question)
    lowered = question.casefold()
    if "how many" in lowered:
        terms.update({"count", "number"})
    if "how much" in lowered:
        terms.update({"amount", "total"})
    if re.search(r"\b\d+\b", lowered) and any(marker in lowered for marker in ("age", "old", "year", "yet")):
        terms.add("age")
    if any(phrase in lowered for phrase in ("times more", "how many times", "ratio", "percentage")):
        terms.update({"ratio", "percentage"})
    for phrase in _quoted_values(question):
        terms.update(_tokenize_for_linking(phrase))
    return terms


def _quoted_values(question: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', question):
        value = match.group(1) or match.group(2)
        if value and value not in values:
            values.append(value)
    return values


def _column_label(kind: str, rel_path: str, table_label: str, column: str) -> str:
    if kind == "sqlite":
        return f"{table_label}.{column}"
    if kind == "json" and table_label != rel_path:
        return f"{table_label}.{column} ({rel_path})"
    return f"{rel_path}:{column}"


def _is_numeric_column(column: str, column_type: str | None = None) -> bool:
    column_tokens = _tokenize_for_linking(column)
    if column_tokens & NUMERIC_COLUMN_TERMS:
        return True
    if column_type and re.search(r"int|real|numeric|decimal|double|float", column_type, flags=re.IGNORECASE):
        return True
    return False


def _make_column_ref(
    *,
    kind: str,
    rel_path: str,
    table_label: str,
    column: str,
    column_type: str | None = None,
) -> _ColumnRef:
    return _ColumnRef(
        label=_column_label(kind, rel_path, table_label, column),
        table_label=table_label,
        column=column,
        kind=kind,
        rel_path=rel_path,
        tokens=frozenset(_tokenize_for_linking(column)),
        table_tokens=frozenset(_tokenize_for_linking(table_label) | _tokenize_for_linking(rel_path)),
        is_numeric=_is_numeric_column(column, column_type),
    )


def _add_column_ref(
    info: _SchemaInfo,
    *,
    kind: str,
    rel_path: str,
    table_label: str,
    column: str,
    column_type: str | None = None,
) -> None:
    ref = _make_column_ref(
        kind=kind,
        rel_path=rel_path,
        table_label=table_label,
        column=column,
        column_type=column_type,
    )
    info.columns.append(ref)
    info.lookup[(kind, rel_path, table_label, column)] = ref


def _schema_info_for_hints(root: Path, files: list[Path]) -> _SchemaInfo:
    info = _SchemaInfo(columns=[], join_edges=[], lookup={})
    for path in files[:MAX_FILES]:
        suffix = path.suffix.casefold()
        rel_path = _relative_path(path, root)
        if suffix == ".csv":
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    columns = next(csv.reader(handle), [])
            except Exception:  # noqa: BLE001
                continue
            for column in columns:
                _add_column_ref(
                    info,
                    kind="csv",
                    rel_path=rel_path,
                    table_label=rel_path,
                    column=str(column),
                )
        elif suffix == ".json":
            try:
                if path.stat().st_size > MAX_JSON_PARSE_BYTES:
                    text = path.read_text(encoding="utf-8", errors="replace")[:200_000]
                    table_match = re.search(r'"table"\s*:\s*"([^"]+)"', text)
                    table_label = table_match.group(1) if table_match else rel_path
                    columns = [
                        key
                        for key in re.findall(r'"([^"]+)"\s*:', text)
                        if key not in {"table", "records"}
                    ]
                else:
                    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                    table_label = str(data.get("table")) if isinstance(data, dict) and data.get("table") else rel_path
                    if isinstance(data, dict):
                        _, columns = _json_columns_from_records(data.get("records"))
                        if not columns:
                            columns = [str(key) for key in data.keys()]
                    elif isinstance(data, list):
                        _, columns = _json_columns_from_records(data)
                    else:
                        columns = []
            except Exception:  # noqa: BLE001
                continue
            seen: set[str] = set()
            for column in columns:
                column = str(column)
                if column in seen:
                    continue
                seen.add(column)
                _add_column_ref(
                    info,
                    kind="json",
                    rel_path=rel_path,
                    table_label=table_label,
                    column=column,
                )
        elif suffix in {".sqlite", ".sqlite3", ".db"}:
            try:
                connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            except Exception:  # noqa: BLE001
                continue
            try:
                table_rows = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall()
                for (table_name,) in table_rows[:MAX_TABLES]:
                    table = str(table_name)
                    quoted_table = _quote_sqlite_identifier(table)
                    for row in connection.execute(f"PRAGMA table_info({quoted_table})").fetchall():
                        _, name, column_type, *_ = row
                        _add_column_ref(
                            info,
                            kind="sqlite",
                            rel_path=rel_path,
                            table_label=table,
                            column=str(name),
                            column_type=str(column_type or ""),
                        )
                    for row in connection.execute(f"PRAGMA foreign_key_list({quoted_table})").fetchall():
                        _, _, ref_table, from_column, to_column, *_ = row
                        info.join_edges.append(
                            _JoinEdge(
                                label=f"{table}.{from_column}->{ref_table}.{to_column}",
                                left_table=table,
                                right_table=str(ref_table),
                            )
                        )
            except Exception:  # noqa: BLE001
                continue
            finally:
                connection.close()
    return info


def _score_column(ref: _ColumnRef, question: str, query_terms: set[str]) -> float:
    score = 0.0
    column_overlap = query_terms & set(ref.tokens)
    table_overlap = query_terms & set(ref.table_tokens)
    score += 3.0 * len(column_overlap)
    score += 1.0 * len(table_overlap)
    column_name = ref.column.casefold()
    table_name = ref.table_label.casefold()
    if len(column_name) > 2 and re.search(rf"\b{re.escape(column_name)}\b", question.casefold()):
        score += 3.0
    if len(table_name) > 2 and re.search(rf"\b{re.escape(table_name)}\b", question.casefold()):
        score += 1.5
    if ref.is_numeric and (query_terms & NUMERIC_INTENT_TERMS):
        score += 1.0
    if "count" in query_terms and ("id" in ref.tokens or ref.column.casefold().endswith("_id")):
        score += 0.5
    return score


def _find_value_matches(root: Path, files: list[Path], info: _SchemaInfo, values: list[str]) -> list[tuple[str, _ColumnRef]]:
    if not values:
        return []
    lowered_values = [(value, value.casefold()) for value in values]
    matches: list[tuple[str, _ColumnRef]] = []
    seen: set[tuple[str, str]] = set()

    def add(value: str, ref: _ColumnRef | None) -> None:
        if ref is None:
            return
        key = (value, ref.label)
        if key in seen:
            return
        seen.add(key)
        matches.append((value, ref))

    for path in files[:MAX_FILES]:
        suffix = path.suffix.casefold()
        rel_path = _relative_path(path, root)
        if suffix == ".csv":
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.reader(handle)
                    columns = [str(column) for column in next(reader, [])]
                    for row_index, row in enumerate(reader):
                        if row_index >= MAX_VALUE_SCAN_ROWS:
                            break
                        for column, cell in zip(columns, row, strict=False):
                            cell_text = str(cell).casefold()
                            for value, lowered in lowered_values:
                                if lowered in cell_text:
                                    add(value, info.lookup.get(("csv", rel_path, rel_path, column)))
            except Exception:  # noqa: BLE001
                continue
        elif suffix == ".json":
            try:
                if path.stat().st_size > MAX_JSON_PARSE_BYTES:
                    continue
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                continue
            table_label = str(data.get("table")) if isinstance(data, dict) and data.get("table") else rel_path
            records: list[object]
            if isinstance(data, dict) and isinstance(data.get("records"), list):
                records = data["records"]
            elif isinstance(data, list):
                records = data
            else:
                records = [data]
            for record in records[:MAX_VALUE_SCAN_ROWS]:
                if not isinstance(record, dict):
                    continue
                for column, cell in record.items():
                    cell_text = str(cell).casefold()
                    for value, lowered in lowered_values:
                        if lowered in cell_text:
                            add(value, info.lookup.get(("json", rel_path, table_label, str(column))))
        elif suffix in {".sqlite", ".sqlite3", ".db"}:
            try:
                connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            except Exception:  # noqa: BLE001
                continue
            try:
                tables = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall()
                for (table_name,) in tables[:MAX_TABLES]:
                    table = str(table_name)
                    quoted_table = _quote_sqlite_identifier(table)
                    column_rows = connection.execute(f"PRAGMA table_info({quoted_table})").fetchall()
                    for row in column_rows:
                        column = str(row[1])
                        quoted_column = _quote_sqlite_identifier(column)
                        for value, lowered in lowered_values:
                            pattern = f"%{lowered.replace('%', '').replace('_', '')}%"
                            try:
                                hit = connection.execute(
                                    f"SELECT 1 FROM {quoted_table} "
                                    f"WHERE lower(CAST({quoted_column} AS TEXT)) LIKE ? LIMIT 1",
                                    (pattern,),
                                ).fetchone()
                            except Exception:  # noqa: BLE001
                                hit = None
                            if hit:
                                add(value, info.lookup.get(("sqlite", rel_path, table, column)))
            except Exception:  # noqa: BLE001
                continue
            finally:
                connection.close()
    return matches[:MAX_HINT_VALUES]


def _overlap_join_edges(info: _SchemaInfo, top_tables: set[str]) -> list[_JoinEdge]:
    grouped: dict[str, list[_ColumnRef]] = {}
    for ref in info.columns:
        if ref.table_label in top_tables:
            grouped.setdefault(ref.column.casefold(), []).append(ref)
    edges: list[_JoinEdge] = []
    seen: set[str] = set()
    for column, refs in grouped.items():
        if len(refs) < 2 or ("id" not in _tokenize_for_linking(column) and not column.endswith("_id")):
            continue
        for left in refs[:3]:
            for right in refs[:3]:
                if left.table_label >= right.table_label:
                    continue
                label = f"{left.table_label}.{left.column}~={right.table_label}.{right.column}"
                if label in seen:
                    continue
                seen.add(label)
                edges.append(_JoinEdge(label=label, left_table=left.table_label, right_table=right.table_label))
    return edges


def _split_markdown_hint_units(text: str) -> list[str]:
    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" \t-")
        if not line or line.startswith("#"):
            continue
        # Keep short rule lines intact, but split long description lines on semicolons.
        pieces = [line]
        if len(line) > 180 and ";" in line:
            pieces = [piece.strip() for piece in line.split(";") if piece.strip()]
        for piece in pieces:
            cleaned = re.sub(r"\s+", " ", piece).strip()
            if cleaned:
                units.append(cleaned[:320])
    return units


def _markdown_unit_score(unit: str, question: str, query_terms: set[str]) -> float:
    unit_terms = _tokenize_for_linking(unit)
    overlap = query_terms & unit_terms
    if not overlap:
        return 0.0

    score = float(len(overlap))
    unit_lower = unit.casefold()
    question_lower = question.casefold()
    if any(marker in unit_lower for marker in ("refers to", "means")):
        score += 2.0
    elif any(marker in unit_lower for marker in ("normal range", "coded", "rule")) and len(overlap) >= 2:
        score += 1.5
    if any(marker in question_lower for marker in ("normal", "abnormal", "range", "status", "percentage", "ratio")):
        score += 0.5
    if _compact_identifier(unit) and _compact_identifier(unit) in _compact_identifier(question):
        score += 0.5
    return score


def _extract_markdown_symbol_terms(unit: str) -> list[str]:
    symbols: list[str] = []

    def add(value: str) -> None:
        cleaned = value.strip(" `\"'.,;:()[]{}")
        if not cleaned:
            return
        lowered = cleaned.casefold()
        if lowered in SQLISH_STOPWORDS:
            return
        if len(_compact_identifier(cleaned)) < 2:
            return
        if cleaned not in symbols:
            symbols.append(cleaned)

    for match in re.finditer(r"`([^`]+)`", unit):
        add(match.group(1))
    for match in re.finditer(r"\b[A-Z][A-Z0-9_-]{1,}\b", unit):
        add(match.group(0))
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){1,3}\b",
        unit,
    ):
        add(match.group(0))
    return symbols


def _markdown_schema_links(
    units: list[tuple[str, str]],
    info: _SchemaInfo,
) -> tuple[list[str], list[_ColumnRef], list[str]]:
    links: list[str] = []
    linked_refs: list[_ColumnRef] = []
    alternatives: list[str] = []
    seen_links: set[str] = set()
    seen_refs: set[str] = set()
    seen_alternatives: set[str] = set()

    column_by_compact: dict[str, list[_ColumnRef]] = {}
    for ref in info.columns:
        compact = _compact_identifier(ref.column)
        if len(compact) >= 2:
            column_by_compact.setdefault(compact, []).append(ref)

    candidate_symbols: list[str] = []
    for _, unit in units:
        candidate_symbols.extend(_extract_markdown_symbol_terms(unit))
        compact_unit = _compact_identifier(unit)
        for compact, refs in column_by_compact.items():
            if len(compact) < 3:
                continue
            if compact in compact_unit and any(
                token in _tokenize_for_linking(unit) for token in refs[0].tokens
            ):
                candidate_symbols.append(refs[0].column)

    for symbol in candidate_symbols:
        compact_symbol = _compact_identifier(symbol)
        if not compact_symbol:
            continue
        exact_refs = column_by_compact.get(compact_symbol, [])
        if not exact_refs:
            continue
        labels = [ref.label for ref in exact_refs[:3]]
        link_text = f"{symbol} -> {', '.join(labels)}"
        if link_text not in seen_links:
            seen_links.add(link_text)
            links.append(link_text)
        for ref in exact_refs:
            if ref.label not in seen_refs:
                seen_refs.add(ref.label)
                linked_refs.append(ref)
            if len(compact_symbol) < 3 or compact_symbol in {"id"}:
                continue
            similar = [
                other
                for other in info.columns
                if other.label != ref.label
                and _compact_identifier(other.column) != compact_symbol
                and compact_symbol in _compact_identifier(other.column)
            ]
            if similar:
                rendered = f"{ref.column} also resembles {', '.join(other.label for other in similar[:2])}"
                if rendered not in seen_alternatives:
                    seen_alternatives.add(rendered)
                    alternatives.append(rendered)
        if len(links) >= MAX_MARKDOWN_LINKS:
            break

    return links[:MAX_MARKDOWN_LINKS], linked_refs[:MAX_MARKDOWN_LINKS], alternatives[:MAX_MARKDOWN_ALTERNATIVES]


def _markdown_semantic_hints(
    root: Path,
    files: list[Path],
    question: str,
    info: _SchemaInfo,
    query_terms: set[str],
) -> tuple[list[str], list[_ColumnRef]]:
    scored_units: list[tuple[float, str, str]] = []
    for path in files[:MAX_FILES]:
        if path.suffix.casefold() not in {".md", ".txt"}:
            continue
        rel_path = _relative_path(path, root)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:160_000]
        except Exception:  # noqa: BLE001
            continue
        for unit in _split_markdown_hint_units(text):
            score = _markdown_unit_score(unit, question, query_terms)
            if score < 1.5:
                continue
            scored_units.append((score, rel_path, unit))

    if not scored_units:
        return [], []

    selected: list[tuple[str, str]] = []
    seen_units: set[str] = set()
    for _, rel_path, unit in sorted(scored_units, key=lambda item: (-item[0], item[1], item[2])):
        key = unit.casefold()
        if key in seen_units:
            continue
        seen_units.add(key)
        selected.append((rel_path, unit))
        if len(selected) >= MAX_MARKDOWN_HINT_SNIPPETS:
            break

    links, linked_refs, alternatives = _markdown_schema_links(selected, info)
    lines = ["Markdown-semantic linking hints (compact; verify exact columns before computing):"]
    if links:
        lines.append(f"- markdown-mentioned schema links: {_trim_items(links, max_items=MAX_MARKDOWN_LINKS)}")
    if alternatives:
        lines.append(
            "- similar non-exact columns to distinguish: "
            + _trim_items(alternatives, max_items=MAX_MARKDOWN_ALTERNATIVES)
        )
    rendered_units = [f"{rel_path}: {unit}" for rel_path, unit in selected]
    if rendered_units:
        lines.append(f"- relevant markdown rules: {_trim_items(rendered_units, max_items=MAX_MARKDOWN_HINT_SNIPPETS)}")

    rendered = "\n".join(lines)
    if len(rendered) <= MAX_MARKDOWN_HINT_CHARS:
        return lines, linked_refs
    trimmed = rendered[:MAX_MARKDOWN_HINT_CHARS].rsplit("\n", 1)[0]
    return [*trimmed.splitlines(), "[markdown-semantic hints truncated]"], linked_refs


def _question_linked_schema_hints(root: Path, files: list[Path], question: str | None) -> list[str]:
    if not question:
        return []
    query_terms = _question_terms(question)
    if not query_terms:
        return []

    info = _schema_info_for_hints(root, files)
    if not info.columns:
        return []

    scores = {ref: _score_column(ref, question, query_terms) for ref in info.columns}
    value_matches = _find_value_matches(root, files, info, _quoted_values(question))
    for _, ref in value_matches:
        scores[ref] = scores.get(ref, 0.0) + 6.0
    markdown_hint_lines, markdown_linked_refs = _markdown_semantic_hints(
        root,
        files,
        question,
        info,
        query_terms,
    )
    for ref in markdown_linked_refs:
        scores[ref] = scores.get(ref, 0.0) + 8.0

    ranked_columns = sorted(
        [ref for ref, score in scores.items() if score > 0],
        key=lambda ref: (-scores[ref], ref.label),
    )[:MAX_HINT_COLUMNS]
    if not ranked_columns:
        return []

    table_scores: dict[str, float] = {}
    for ref, score in scores.items():
        if score <= 0:
            continue
        table_scores[ref.table_label] = table_scores.get(ref.table_label, 0.0) + score
    ranked_tables = [
        table
        for table, _ in sorted(table_scores.items(), key=lambda item: (-item[1], item[0]))[:MAX_HINT_TABLES]
    ]
    top_table_set = set(ranked_tables)
    join_edges = [
        edge
        for edge in info.join_edges
        if edge.left_table in top_table_set or edge.right_table in top_table_set
    ]
    join_edges.extend(_overlap_join_edges(info, top_table_set))

    lines = ["Question-linked schema hints (orientation only; verify with data):"]
    if ranked_tables:
        lines.append(f"- likely tables/files: {_trim_items(ranked_tables, max_items=MAX_HINT_TABLES)}")
    if ranked_columns:
        lines.append(f"- likely columns: {_trim_items([ref.label for ref in ranked_columns], max_items=MAX_HINT_COLUMNS)}")
    if value_matches:
        rendered_values = [f'"{value}" -> {ref.label}' for value, ref in value_matches]
        lines.append(f"- quoted value matches: {_trim_items(rendered_values, max_items=MAX_HINT_VALUES)}")
    if join_edges:
        lines.append(f"- possible joins: {_trim_items([edge.label for edge in join_edges], max_items=MAX_HINT_JOINS)}")
    lines.extend(markdown_hint_lines)

    rendered = "\n".join(lines)
    if len(rendered) <= MAX_SCHEMA_HINT_CHARS:
        return lines
    trimmed = rendered[:MAX_SCHEMA_HINT_CHARS].rsplit("\n", 1)[0]
    return [*trimmed.splitlines(), "[question-linked hints truncated]"]


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
        parts.append(f"columns={_trim_items(columns)}")
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
    lines = [f"- {rel_path} [sqlite, {_format_size(path)}]: schema from sqlite_master + PRAGMA table_info/foreign_key_list"]
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except Exception as exc:  # noqa: BLE001
        return [f"{lines[0]}: unreadable ({exc})"]

    try:
        table_rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        for (table,) in table_rows[:MAX_TABLES]:
            table = str(table)
            quoted_table = _quote_sqlite_identifier(table)
            column_specs: list[str] = []
            for row in connection.execute(f"PRAGMA table_info({quoted_table})").fetchall():
                _, name, column_type, notnull, default_value, pk = row
                _ = default_value
                spec = str(name)
                if column_type:
                    spec += f" {column_type}"
                flags: list[str] = []
                if pk:
                    flags.append("pk")
                if notnull:
                    flags.append("notnull")
                if flags:
                    spec += f" ({';'.join(flags)})"
                column_specs.append(spec)
            lines.append(f"  - table {table}: columns={_trim_items(column_specs)}")

            foreign_key_specs: list[str] = []
            for row in connection.execute(f"PRAGMA foreign_key_list({quoted_table})").fetchall():
                _, _, ref_table, from_column, to_column, on_update, on_delete, _ = row
                behavior = []
                if on_update and str(on_update).upper() != "NO ACTION":
                    behavior.append(f"on_update={on_update}")
                if on_delete and str(on_delete).upper() != "NO ACTION":
                    behavior.append(f"on_delete={on_delete}")
                suffix = f" ({';'.join(behavior)})" if behavior else ""
                foreign_key_specs.append(f"{from_column}->{ref_table}.{to_column}{suffix}")
            if foreign_key_specs:
                lines.append(f"    foreign_keys={_trim_items(foreign_key_specs, max_items=MAX_FOREIGN_KEYS)}")

        if len(table_rows) > MAX_TABLES:
            lines.append(f"  - ... (+{len(table_rows) - MAX_TABLES} more tables)")
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


def build_context_manifest(context_dir: Path, *, question: str | None = None) -> str:
    """Build a small schema/file manifest for the first model prompt.

    This is intentionally shallow: it exposes file paths and schema-like metadata
    without reading full dataframes or documents into the prompt.
    """

    root = context_dir.resolve()
    try:
        files = sorted(path for path in root.rglob("*") if path.is_file())
    except Exception as exc:  # noqa: BLE001
        return f"- context manifest unavailable: {exc}"

    lines = [*_question_linked_schema_hints(root, files, question), "Concise context manifest:"]
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
