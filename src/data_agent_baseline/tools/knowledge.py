from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

MAX_SCHEMA_FILES = 80
MAX_JSON_SCHEMA_BYTES = 8_000_000
MAX_SCHEMA_TERMS = 200

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
    "list",
    "many",
    "of",
    "on",
    "or",
    "provide",
    "show",
    "that",
    "the",
    "their",
    "them",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

MAPPING_PATTERN = re.compile(
    r"\b("
    r"code|coded|corresponds|denotes|indicates|label|level|maps?|means|"
    r"represents|status|type|value|where"
    r")\b",
    flags=re.IGNORECASE,
)
CODE_VALUE_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_ ]{0,40}\s*(?:=|:|->|-)\s*[A-Za-z0-9_][A-Za-z0-9_ +./-]{0,80}")


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_]+", text.casefold())
        if len(token) > 1 and token not in STOPWORDS
    }


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _sqlite_quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _json_columns_from_records(records: object) -> list[str]:
    if not isinstance(records, list):
        return []
    columns: list[str] = []
    seen: set[str] = set()
    for record in records[:5]:
        if not isinstance(record, dict):
            continue
        for key in record:
            column = str(key)
            if column in seen:
                continue
            seen.add(column)
            columns.append(column)
    return columns


def _schema_terms_from_json(path: Path) -> list[str]:
    try:
        if path.stat().st_size > MAX_JSON_SCHEMA_BYTES:
            return []
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return []

    terms: list[str] = [path.stem]
    if isinstance(data, dict):
        table = data.get("table")
        if table is not None:
            terms.append(str(table))
        columns = _json_columns_from_records(data.get("records"))
        if columns:
            terms.extend(columns)
        else:
            terms.extend(str(key) for key in data.keys())
    elif isinstance(data, list):
        terms.extend(_json_columns_from_records(data))
    return terms


def _schema_terms_from_sqlite(path: Path) -> list[str]:
    terms: list[str] = [path.stem]
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001
        return terms

    try:
        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        for (table_name,) in tables:
            table = str(table_name)
            terms.append(table)
            for row in connection.execute(f"PRAGMA table_info({_sqlite_quote(table)})").fetchall():
                terms.append(str(row[1]))
    except Exception:  # noqa: BLE001
        return terms
    finally:
        connection.close()
    return terms


def collect_schema_query_terms(context_root: str | Path) -> list[str]:
    """Collect compact table/column names for knowledge retrieval query expansion."""

    root = Path(context_root).resolve()
    try:
        files = sorted(path for path in root.rglob("*") if path.is_file())
    except Exception:  # noqa: BLE001
        return []

    terms: list[str] = []
    for path in files[:MAX_SCHEMA_FILES]:
        suffix = path.suffix.casefold()
        if suffix == ".csv":
            terms.append(path.stem)
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.reader(handle)
                    terms.extend(str(column) for column in next(reader, []))
            except Exception:  # noqa: BLE001
                continue
        elif suffix == ".json":
            terms.extend(_schema_terms_from_json(path))
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            terms.extend(_schema_terms_from_sqlite(path))

        if len(terms) >= MAX_SCHEMA_TERMS:
            break

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.strip()
        if not normalized or normalized.casefold() in seen:
            continue
        seen.add(normalized.casefold())
        deduped.append(normalized)
        if len(deduped) >= MAX_SCHEMA_TERMS:
            break
    return deduped


def extract_ambiguity_section(context_root: str | Path) -> str:
    """Return the body of the Ambiguity Resolution section from knowledge.md, or ''."""
    root = Path(context_root).resolve()
    for kpath in sorted(root.rglob("knowledge.md")):
        try:
            text = kpath.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        # Split only at ## level so ### subsections stay attached to their parent
        sections = re.split(r"\n(?=## )", text)
        for section in sections:
            if re.match(r"## [^\n]*ambiguity", section, re.IGNORECASE):
                body = re.sub(r"^## [^\n]*\n", "", section).strip()
                if body:
                    return body
    return ""


def _split_knowledge_blocks(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n|(?=^#{1,6}\s)", text, flags=re.MULTILINE)
    return [block.strip() for block in blocks if block.strip()]


def retrieve_knowledge_snippets(
    context_root: str | Path,
    question: str,
    *,
    extra_terms: list[str] | tuple[str, ...] = (),
    top_k: int = 5,
    max_chars: int = 1200,
    include_schema_terms: bool = True,
) -> list[dict[str, object]]:
    """Search knowledge.md using the question plus schema column/table names.

    The full knowledge file is scanned, but only compact relevant snippets are returned.
    """

    root = Path(context_root).resolve()
    question_tokens = _tokenize(question)
    schema_terms = collect_schema_query_terms(root) if include_schema_terms else []
    schema_tokens = set()
    for term in schema_terms:
        schema_tokens.update(_tokenize(term))
    for term in extra_terms:
        schema_tokens.update(_tokenize(str(term)))

    hits: list[tuple[float, str, str, list[str], list[str]]] = []
    for path in sorted(root.rglob("knowledge.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue

        for block in _split_knowledge_blocks(text):
            block_tokens = _tokenize(block)
            question_overlap = sorted(question_tokens & block_tokens)
            schema_overlap = sorted(schema_tokens & block_tokens)
            if not question_overlap and not schema_overlap:
                continue

            score = (3.0 * len(question_overlap)) + len(schema_overlap)
            if question_overlap and MAPPING_PATTERN.search(block):
                score += 2.0
            if question_overlap and CODE_VALUE_PATTERN.search(block):
                score += 2.0
            if not question_overlap:
                score *= 0.5

            snippet = block[:max_chars]
            if len(block) > max_chars:
                snippet = snippet.rstrip() + "\n...[snippet truncated]"
            hits.append(
                (
                    score,
                    _relative_path(path, root),
                    snippet,
                    question_overlap[:12],
                    schema_overlap[:12],
                )
            )

    hits.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {
            "score": round(score, 3),
            "path": path,
            "matched_question_terms": matched_question_terms,
            "matched_schema_terms": matched_schema_terms,
            "snippet": snippet,
        }
        for score, path, snippet, matched_question_terms, matched_schema_terms in hits[: max(top_k, 0)]
    ]
