from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import multiprocessing
import os
import re
import sqlite3
import sys
import tempfile
import traceback
import types
from pathlib import Path
from typing import Any

from data_agent_baseline.tools.knowledge import retrieve_knowledge_snippets

try:
    import pandas as pd
except Exception:  # noqa: BLE001
    pd = None


@contextlib.contextmanager
def _capture_process_streams(stdout_path: Path, stderr_path: Path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    with stdout_path.open("w+b") as stdout_file, stderr_path.open("w+b") as stderr_file:
        try:
            if original_stdout is not None:
                original_stdout.flush()
            if original_stderr is not None:
                original_stderr.flush()

            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)

            stdout_encoding = "utf-8"
            stderr_encoding = "utf-8"

            sys.stdout = io.TextIOWrapper(
                os.fdopen(os.dup(1), "wb"),
                encoding=stdout_encoding,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            sys.stderr = io.TextIOWrapper(
                os.fdopen(os.dup(2), "wb"),
                encoding=stderr_encoding,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            yield
        finally:
            if sys.stdout is not None:
                sys.stdout.flush()
            if sys.stderr is not None:
                sys.stderr.flush()

            if sys.stdout is not original_stdout:
                sys.stdout.close()
            if sys.stderr is not original_stderr:
                sys.stderr.close()

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def _read_captured_stream(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_search_terms(terms: object) -> list[str]:
    if isinstance(terms, str):
        raw_terms = [terms]
    elif isinstance(terms, (list, tuple, set)):
        raw_terms = [str(term) for term in terms]
    else:
        raw_terms = [str(terms)]
    normalized: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        value = term.strip()
        if not value or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        normalized.append(value)
    return normalized


def search_markdown_documents(
    context_root: str | Path,
    terms: object,
    *,
    path_filters: object = None,
    max_matches: int = 8,
    context_chars: int = 500,
) -> list[dict[str, object]]:
    """Return compact snippets around term matches in markdown documents."""

    root = Path(context_root).resolve()
    search_terms = _normalize_search_terms(terms)
    if not search_terms:
        return []

    matches: list[dict[str, object]] = []
    try:
        paths = sorted(root.rglob("*.md"))
    except Exception:  # noqa: BLE001
        return []
    markdown_path_filters = _normalize_search_terms(path_filters) if path_filters is not None else []
    if markdown_path_filters:
        lowered_filters = [path_filter.casefold().replace("\\", "/") for path_filter in markdown_path_filters]

        def path_matches_filter(path: Path) -> bool:
            relative = _relative_path(path, root).casefold()
            name = path.name.casefold()
            return any(
                relative == lowered_filter
                or relative.endswith("/" + lowered_filter)
                or name == lowered_filter
                or lowered_filter in relative
                for lowered_filter in lowered_filters
            )

        paths = [path for path in paths if path_matches_filter(path)]

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        lowered = text.casefold()
        for term in search_terms:
            start_at = 0
            lowered_term = term.casefold()
            while len(matches) < max_matches:
                index = lowered.find(lowered_term, start_at)
                if index < 0:
                    break
                start = max(0, index - max(0, context_chars))
                end = min(len(text), index + len(term) + max(0, context_chars))
                snippet = text[start:end].strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet += "..."
                matches.append(
                    {
                        "path": _relative_path(path, root),
                        "matched_term": term,
                        "start": index,
                        "snippet": snippet,
                    }
                )
                start_at = index + max(1, len(term))
            if len(matches) >= max_matches:
                return matches
    return matches


MARKDOWN_RECORD_ID_PATTERN = re.compile(
    r"\brec(?=[A-Za-z0-9]{6,20}\b)(?=[A-Za-z0-9]*[0-9A-Z])[A-Za-z0-9]{6,20}\b|"
    r"\b(?:[A-Za-z][A-Za-z0-9_ -]{0,30})?(?:id|ID)\s*[:=]\s*['\"]?[A-Za-z0-9_.:-]+['\"]?"
)
MARKDOWN_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[$]\s*)?[-+]?\d[\d,]*(?:\.\d+)?\s*%?"
)
RECORD_ID_PATTERN = re.compile(r"\brec(?=[A-Za-z0-9]{6,20}\b)(?=[A-Za-z0-9]*[0-9A-Z])[A-Za-z0-9]{6,20}\b")

LINKING_STOPWORDS = {
    "about",
    "above",
    "after",
    "among",
    "answer",
    "before",
    "below",
    "between",
    "calculate",
    "count",
    "does",
    "find",
    "for",
    "how",
    "from",
    "give",
    "have",
    "into",
    "many",
    "more",
    "much",
    "name",
    "names",
    "only",
    "than",
    "that",
    "their",
    "them",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}
ID_COLUMN_TERMS = {"id", "ids", "key", "code", "link", "record", "ref"}
NUMERIC_LINKING_TERMS = {
    "age",
    "amount",
    "average",
    "budget",
    "cost",
    "count",
    "height",
    "number",
    "percentage",
    "price",
    "ratio",
    "score",
    "total",
    "value",
    "weight",
}
ROW_MATCH_GENERIC_TOKENS = {
    "event",
    "game",
    "meeting",
    "month",
    "race",
    "status",
    "type",
    "year",
}
MARKDOWN_ENTITY_TABLE_TRIGGER_TERMS = {
    "age",
    "alignment",
    "alp",
    "birth",
    "birthday",
    "bilirubin",
    "born",
    "cre",
    "creatinine",
    "gender",
    "got",
    "gpt",
    "height",
    "hero",
    "heroes",
    "ldh",
    "patient",
    "patients",
    "publisher",
    "sex",
    "superhero",
    "superheroes",
    "t",
    "tbill",
    "urea",
    "uric",
    "weight",
}
MARKDOWN_ENTITY_TABLE_AVOID_TERMS = {
    "advertisement",
    "allocation",
    "amount",
    "budget",
    "budgets",
    "card",
    "cards",
    "commander",
    "content",
    "event",
    "format",
    "legal",
    "legality",
    "meeting",
    "status",
    "warning",
}
MARKDOWN_ENTITY_TABLE_FIELDS = {
    "alignment_id",
    "alp",
    "birth_year",
    "cards_id",
    "creatinine",
    "got",
    "gpt",
    "height_cm",
    "ldh",
    "patient_id",
    "publisher_id",
    "record_year",
    "sex",
    "t_bil",
    "urea_nitrogen",
    "uric_acid",
    "weight_kg",
}


def _question_markdown_terms(question: str) -> list[str]:
    quoted = [
        match.group(1) or match.group(2)
        for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', question)
    ]
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", question)
        if len(token) > 3
        and token.casefold()
        not in {
            "what",
            "which",
            "where",
            "when",
            "many",
            "much",
            "more",
            "than",
            "with",
            "from",
            "that",
            "this",
            "were",
            "have",
            "does",
            "into",
            "only",
        }
    ]
    return _normalize_search_terms([*quoted, *tokens])


def _split_markdown_blocks(text: str) -> list[tuple[str | None, str]]:
    blocks: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    pieces = re.split(r"\n\s*\n", text)
    for piece in pieces:
        block = piece.strip()
        if not block:
            continue
        heading_match = re.match(r"^(#{1,6}\s+.+)$", block)
        if heading_match is not None:
            current_heading = heading_match.group(1).strip()
        blocks.append((current_heading, block))
    return blocks


def _markdown_record_summary(
    *,
    root: Path,
    path: Path,
    block_index: int,
    heading: str | None,
    text: str,
    matched_terms: list[str],
    linked_from_ids: list[str] | None = None,
    max_chars: int,
) -> dict[str, object]:
    ids = []
    seen_ids: set[str] = set()
    for match in MARKDOWN_RECORD_ID_PATTERN.finditer(text):
        value = match.group(0).strip().strip("'\"")
        if value.casefold() in seen_ids:
            continue
        seen_ids.add(value.casefold())
        ids.append(value)

    numbers = []
    seen_numbers: set[str] = set()
    for match in MARKDOWN_NUMBER_PATTERN.finditer(text):
        value = match.group(0).strip()
        if value.casefold() in seen_numbers:
            continue
        seen_numbers.add(value.casefold())
        numbers.append(value)

    snippet = text
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "\n...[record snippet truncated]"

    return {
        "path": _relative_path(path, root),
        "block_index": block_index,
        "heading": heading,
        "matched_terms": matched_terms,
        "linked_from_ids": linked_from_ids or [],
        "ids": ids[:20],
        "numbers": numbers[:20],
        "snippet": snippet,
    }


def extract_markdown_records(
    context_root: str | Path,
    terms: object = None,
    *,
    question: str = "",
    path_filters: object = None,
    max_records: int = 12,
    max_chars: int = 900,
    include_linked_ids: bool = True,
    link_depth: int = 2,
) -> list[dict[str, object]]:
    """Return markdown blocks as lightweight record/link candidates.

    This is meant for documents that encode entities through record IDs, nearby
    numbers, and cross references. It searches term-matched blocks first, then
    optionally follows IDs found in those blocks to other blocks.
    """

    root = Path(context_root).resolve()
    search_terms = _normalize_search_terms(terms) if terms is not None else _question_markdown_terms(question)
    lowered_terms = [(term, term.casefold()) for term in search_terms]
    try:
        paths = sorted(root.rglob("*.md"))
    except Exception:  # noqa: BLE001
        return []
    markdown_path_filters = _normalize_search_terms(path_filters) if path_filters is not None else []
    if markdown_path_filters:
        lowered_filters = [path_filter.casefold().replace("\\", "/") for path_filter in markdown_path_filters]

        def path_matches_filter(path: Path) -> bool:
            relative = _relative_path(path, root).casefold()
            name = path.name.casefold()
            return any(
                relative == lowered_filter
                or relative.endswith("/" + lowered_filter)
                or name == lowered_filter
                or lowered_filter in relative
                for lowered_filter in lowered_filters
            )

        paths = [path for path in paths if path_matches_filter(path)]

    all_blocks: list[tuple[Path, int, str | None, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for block_index, (heading, block) in enumerate(_split_markdown_blocks(text)):
            all_blocks.append((path, block_index, heading, block))

    scored: list[tuple[float, Path, int, str | None, str, list[str]]] = []
    for path, block_index, heading, block in all_blocks:
        lowered_block = block.casefold()
        matched_terms = [term for term, lowered in lowered_terms if lowered and lowered in lowered_block]
        ids = MARKDOWN_RECORD_ID_PATTERN.findall(block)
        numbers = MARKDOWN_NUMBER_PATTERN.findall(block)

        if lowered_terms and not matched_terms:
            continue
        if not lowered_terms and not ids and not numbers:
            continue

        score = (4.0 * len(matched_terms)) + (0.8 * min(len(ids), 5)) + (0.4 * min(len(numbers), 5))
        if heading and any(lowered in heading.casefold() for _, lowered in lowered_terms):
            score += 2.0
        scored.append((score, path, block_index, heading, block, matched_terms))

    scored.sort(key=lambda item: (-item[0], _relative_path(item[1], root), item[2]))
    selected: list[tuple[float, Path, int, str | None, str, list[str]]] = []
    selected_keys: set[tuple[str, int]] = set()
    for term in search_terms:
        for item in scored:
            _, path, block_index, _, _, matched_terms = item
            key = (str(path), block_index)
            if key in selected_keys or term not in matched_terms:
                continue
            selected.append(item)
            selected_keys.add(key)
            break
        if len(selected) >= max_records:
            break
    for item in scored:
        if len(selected) >= max(1, max_records):
            break
        _, path, block_index, _, _, _ = item
        key = (str(path), block_index)
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
    results: list[dict[str, object]] = [
        _markdown_record_summary(
            root=root,
            path=path,
            block_index=block_index,
            heading=heading,
            text=block,
            matched_terms=matched_terms,
            max_chars=max_chars,
        )
        for _, path, block_index, heading, block, matched_terms in selected
    ]

    if include_linked_ids and len(results) < max_records:
        selected_keys = {(str(path), block_index) for _, path, block_index, _, _, _ in selected}
        seen_seed_ids: set[str] = set()
        frontier_ids: list[str] = []
        for result in results:
            for value in result.get("ids", []):
                text_value = str(value)
                if text_value.casefold() in seen_seed_ids:
                    continue
                seen_seed_ids.add(text_value.casefold())
                frontier_ids.append(text_value)

        for _ in range(max(0, link_depth)):
            if not frontier_ids or len(results) >= max_records:
                break
            linked: list[tuple[int, Path, int, str | None, str, list[str]]] = []
            for path, block_index, heading, block in all_blocks:
                key = (str(path), block_index)
                if key in selected_keys:
                    continue
                matched_ids = [record_id for record_id in frontier_ids if record_id and record_id in block]
                if not matched_ids:
                    continue
                linked.append((len(matched_ids), path, block_index, heading, block, matched_ids))

            linked.sort(key=lambda item: (-item[0], _relative_path(item[1], root), item[2]))
            next_frontier: list[str] = []
            for _, path, block_index, heading, block, matched_ids in linked:
                if len(results) >= max_records:
                    break
                selected_keys.add((str(path), block_index))
                summary = _markdown_record_summary(
                    root=root,
                    path=path,
                    block_index=block_index,
                    heading=heading,
                    text=block,
                    matched_terms=[],
                    linked_from_ids=matched_ids,
                    max_chars=max_chars,
                )
                results.append(summary)
                for value in summary.get("ids", []):
                    text_value = str(value)
                    if text_value.casefold() in seen_seed_ids:
                        continue
                    seen_seed_ids.add(text_value.casefold())
                    next_frontier.append(text_value)
            frontier_ids = next_frontier

    return results


PATIENT_ID_PATTERNS = (
    re.compile(r"\bpatient\s+(\d{2,})\b", flags=re.IGNORECASE),
    re.compile(r"\bMedical Record Number\s+(\d{2,})\b", flags=re.IGNORECASE),
    re.compile(r"\bfile(?:\s+number)?\s+(\d{2,})\b", flags=re.IGNORECASE),
    re.compile(r"\bfile\s+for\s+patient\s+(\d{2,})\b", flags=re.IGNORECASE),
)
ENTITY_ID_PATTERNS = (
    re.compile(r"\bregistered under (?:the )?(?:unique )?identifier\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\bregistered under (?:the )?(?:unique )?ID\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\bregistration number\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\bregistry number\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\bRegistry Ref:\s*(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\breference(?:\s+code|\s+ID)?\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\breference number\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\bregistered with identifier\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\btracked (?:with|under) identifier\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\btracked under reference ID\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\bcatalog(?:ed|ued) (?:with|under) (?:registry number|reference code|reference number|identifier)\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\bfiled under ID\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\bunder ID\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\bstrategic unit ID\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\blegality (?:entry|ruling) ID\s+(\d{1,})\b", flags=re.IGNORECASE),
    re.compile(r"\bID\s+(\d{1,})\b", flags=re.IGNORECASE),
)


def _numeric_value(value: object) -> float | None:
    text = str(value).replace(",", "").replace("$", "").replace("%", "").strip()
    if text.casefold() in {"nan", "none", "null", ""}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_number(value: object) -> object:
    number = _numeric_value(value)
    if number is None:
        return value
    if isinstance(number, float) and math.isnan(number):
        return None
    if float(number).is_integer():
        return int(number)
    return number


def _split_markdown_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.replace("\r", " ").replace("\n", " "))
    return [piece.strip() for piece in pieces if piece.strip()]


def _best_number_near_terms(
    text: str,
    term_patterns: tuple[str, ...],
    *,
    unit_terms: tuple[str, ...] = (),
) -> object | None:
    best: tuple[float, int, object] | None = None
    for sentence_index, sentence in enumerate(_split_markdown_sentences(text)):
        lowered = sentence.casefold()
        term_spans: list[tuple[int, int]] = []
        for pattern in term_patterns:
            term_spans.extend((match.start(), match.end()) for match in re.finditer(pattern, lowered))
        if not term_spans:
            continue

        for match in MARKDOWN_NUMBER_PATTERN.finditer(sentence):
            raw_value = match.group(0)
            value = _coerce_number(raw_value)
            if value is None:
                continue
            near_window = sentence[max(0, match.start() - 80) : match.end() + 80].casefold()
            if unit_terms and not any(unit in near_window for unit in unit_terms):
                continue
            nearest_term = min(
                term_spans,
                key=lambda span: min(abs(match.start() - span[0]), abs(match.end() - span[1])),
            )
            distance = min(abs(match.start() - nearest_term[0]), abs(match.end() - nearest_term[1]))
            score = -float(distance)
            if nearest_term[1] <= match.start():
                score += 20.0
            elif match.end() < nearest_term[0]:
                score -= 180.0
            immediate_before = sentence[max(0, match.start() - 45) : match.start()].casefold()
            immediate_after = sentence[match.end() : match.end() + 45].casefold()
            if re.search(r"\b(?:verified|confirmed|corrected|adjusted|amended|rectified|finalized|final|accurate|precise)\s+(?:at|to|as|value of|figure of)?\s*$", immediate_before):
                score += 120.0
            if re.search(r"\b(?:initial|initially|preliminary|first|originally|estimated|thought|misread|mistaken|error)\b", immediate_before):
                score -= 120.0
            if re.search(r"^\s*(?:was|were)?\s*(?:later|subsequently)?\s*(?:corrected|adjusted|confirmed|verified|amended|rectified)", immediate_after):
                score += 60.0
            if re.search(r"\b(?:corrected|confirmed|verified|final|finalized|accurate|precise|rectified|updated|amended|current)\b", near_window):
                score += 80.0
            if re.search(r"\b(?:initial|initially|preliminary|estimated|misread|mistakenly|error|incorrect)\b", near_window):
                score -= 25.0
            if re.search(r"\b(?:not available|unavailable|nan|none|null|missing)\b", near_window):
                score += 5.0
            tie_breaker = (sentence_index * 1000) + match.start()
            if best is None or (score, tie_breaker) > (best[0], best[1]):
                best = (score, tie_breaker, value)
    return None if best is None else best[2]


def _best_year_near_terms(text: str, term_patterns: tuple[str, ...]) -> int | None:
    best: tuple[float, int, int] | None = None
    for sentence_index, sentence in enumerate(_split_markdown_sentences(text)):
        lowered = sentence.casefold()
        term_spans: list[tuple[int, int]] = []
        for pattern in term_patterns:
            term_spans.extend((match.start(), match.end()) for match in re.finditer(pattern, lowered))
        if not term_spans:
            continue
        for match in re.finditer(r"\b(18\d{2}|19\d{2}|20\d{2})\b", sentence):
            year = int(match.group(1))
            distance = min(
                min(abs(match.start() - term_start), abs(match.end() - term_end))
                for term_start, term_end in term_spans
            )
            score = -float(distance)
            near_window = sentence[max(0, match.start() - 90) : match.end() + 90].casefold()
            if re.search(r"\b(?:corrected|confirmed|verified|official|correct|rectified|amended)\b", near_window):
                score += 40.0
            if re.search(r"\b(?:initial|preliminary|mistakenly|error|suggested)\b", near_window):
                score -= 20.0
            tie_breaker = (sentence_index * 1000) + match.start()
            if best is None or (score, tie_breaker) > (best[0], best[1]):
                best = (score, tie_breaker, year)
    return None if best is None else best[2]


def _first_regex_group(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            return match.group(1).strip()
    return None


def _clean_markdown_entity_name(value: str) -> str:
    cleaned = value.strip(" .,'\"")
    cleaned = re.sub(r"\s+(?:is|was|whose|who|registered|tracked|cataloged|catalogued)\b.*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .,'\"")


def _extract_markdown_names(text: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    name_patterns = (
        r"\b(?:known as|designated|called)\s+([A-Z][A-Za-z0-9'(). -]{1,60})",
        r"\b(?:codename|alias)\s+(?:is\s+|as\s+)?([A-Z][A-Za-z0-9'(). -]{1,60})",
        r"\b(?:asset|operative|unit|entity|individual|subject)\s+(?:known as|designated|called)\s+([A-Z][A-Za-z0-9'(). -]{1,60})",
    )
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match is not None:
            name = _clean_markdown_entity_name(match.group(1))
            if name and len(name) <= 80:
                fields["name"] = name
                break

    full_name_patterns = (
        r"\bfull legal name (?:of|to be|is|as)\s+([A-Z][A-Za-z'(). -]{1,80}|None|-)\b",
        r"\b(?:full name|complete legal name).{0,90}?\b(?:recorded as|documented as|confirmed as|listed as|updated.{0,30}?to(?: reflect)?|to be|is)\s+([A-Z][A-Za-z'(). -]{1,80}|None|-)\b",
        r"\b(?:full name|civilian identity).{0,90}?\b(?:marked as|logged as|listed as)\s+([A-Z][A-Za-z'(). -]{1,80}|None|-)\b",
    )
    for sentence in _split_markdown_sentences(text):
        if not re.search(r"\b(?:full name|complete legal name|civilian identity)\b", sentence, flags=re.IGNORECASE):
            continue
        for pattern in full_name_patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match is not None:
                fields["full_name"] = _clean_markdown_entity_name(match.group(1))
                return fields
    return fields


def _extract_markdown_block_fields(text: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    patient_id = _first_regex_group(PATIENT_ID_PATTERNS, text)
    cards_id = re.search(r"\bcards_id\s+(\d{1,})\b", text, flags=re.IGNORECASE)
    record_ids = []
    seen_record_ids: set[str] = set()
    for match in RECORD_ID_PATTERN.finditer(text):
        value = match.group(0)
        if value.casefold() not in seen_record_ids:
            seen_record_ids.add(value.casefold())
            record_ids.append(value)

    if patient_id is not None:
        fields["patient_id"] = int(patient_id)
    else:
        entity_id = _first_regex_group(ENTITY_ID_PATTERNS, text)
        if entity_id is not None:
            fields["id"] = int(entity_id)
    if cards_id is not None:
        fields["cards_id"] = int(cards_id.group(1))
    if record_ids:
        fields["record_ids"] = record_ids[:8]

    fields.update(_extract_markdown_names(text))

    sex_sentence = " ".join(
        sentence
        for sentence in _split_markdown_sentences(text)
        if re.search(r"\b(?:male|female|gender|sex)\b", sentence, flags=re.IGNORECASE)
    ).casefold()
    if "female" in sex_sentence:
        fields["sex"] = "F"
    elif re.search(r"\bmale\b", sex_sentence):
        fields["sex"] = "M"

    birthday_year = _best_year_near_terms(text, (r"\bborn\b", r"\bbirthdate\b", r"\bbirthday\b", r"\bdate of birth\b"))
    if birthday_year is not None:
        fields["birth_year"] = birthday_year
    record_year = _best_year_near_terms(text, (r"\bsample\b", r"\btested\b", r"\bassessed\b", r"\bdated\b", r"\brecorded\b", r"\bfrom\b", r"\bon\b"))
    if record_year is not None:
        fields["record_year"] = record_year

    for sentence in _split_markdown_sentences(text):
        lowered_sentence = sentence.casefold()
        if (
            "height" in lowered_sentence
            and "weight" in lowered_sentence
            and re.search(r"\b(?:both|placeholder|listed|recorded|fields?)\b", lowered_sentence)
        ):
            if re.search(r"\b0(?:\.0+)?\b", lowered_sentence):
                fields.setdefault("height_cm", 0)
                fields.setdefault("weight_kg", 0)
            elif re.search(r"\b(?:nan|not available|unavailable)\b", lowered_sentence):
                fields.setdefault("height_cm", None)
                fields.setdefault("weight_kg", None)

    numeric_specs = {
        "height_cm": ((r"\bheight\b", r"\bstanding height\b"), ("centimeter", "centimeters", "cm")),
        "weight_kg": ((r"\bweight\b",), ("kilogram", "kilograms", "kg")),
        "publisher_id": ((r"\bpublisher affiliation\b", r"\bpublisher code\b", r"\bwith publisher\b", r"\bunder publisher\b", r"\bpublisher\b"), ()),
        "alignment_id": ((r"\bmoral alignment\b", r"\balignment\b"), ()),
        "creatinine": ((r"\bcreatinine\b", r"\bcre\b"), ("mg/dl",)),
        "uric_acid": ((r"\buric acid\b", r"\bua\b"), ("mg/dl",)),
        "urea_nitrogen": ((r"\burea nitrogen\b", r"\bun\b"), ("mg/dl",)),
        "got": ((r"\bgot\b", r"\bglutamic oxaloacetic transaminase\b"), ("u/l",)),
        "gpt": ((r"\bgpt\b", r"\bglutamic pyruvic transaminase\b"), ("u/l",)),
        "ldh": ((r"\bldh\b", r"\blactate dehydrogenase\b"), ("u/l",)),
        "alp": ((r"\balp\b", r"\balkaline phosphatase\b"), ("u/l",)),
        "t_bil": ((r"\bt-bil\b", r"\btotal bilirubin\b"), ("mg/dl",)),
    }
    for field, (patterns, units) in numeric_specs.items():
        if field in fields:
            continue
        value = _best_number_near_terms(text, patterns, unit_terms=units)
        if value is not None:
            fields[field] = value
    return fields


def _markdown_entity_key(fields: dict[str, object]) -> str | None:
    for key in ("patient_id", "id", "cards_id"):
        value = fields.get(key)
        if value not in {None, ""}:
            return f"{key}:{value}"
    record_ids = fields.get("record_ids")
    if isinstance(record_ids, list) and record_ids:
        return f"record_id:{record_ids[0]}"
    return None


def _merge_markdown_entity_fields(
    existing: dict[str, object],
    incoming: dict[str, object],
) -> None:
    for key, value in incoming.items():
        if key.startswith("__") or value is None or value == "":
            continue
        if key not in existing or existing[key] is None or existing[key] == "" or existing[key] == []:
            existing[key] = value
            continue
        if key == "record_ids" and isinstance(existing.get(key), list) and isinstance(value, list):
            seen = {str(item).casefold() for item in existing[key]}
            for item in value:
                if str(item).casefold() not in seen:
                    existing[key].append(item)
                    seen.add(str(item).casefold())


def build_markdown_entity_table(
    context_root: str | Path,
    terms: object = None,
    *,
    question: str = "",
    path_filters: object = None,
    max_records: int | None = None,
    include_evidence: bool = True,
    evidence_chars: int = 260,
) -> dict[str, object]:
    """Parse markdown prose into merged row-like records keyed by IDs.

    The helper is intentionally deterministic and conservative. It does not try
    to answer the task. It extracts common record IDs, patient IDs, card IDs,
    numeric measurements, dates, and coded affiliations from prose blocks, then
    merges blocks that mention the same ID. This gives the model a dataframe-like
    bridge when the true table is encoded in markdown paragraphs.
    """

    root = Path(context_root).resolve()
    try:
        paths = sorted(root.rglob("*.md"))
    except Exception:  # noqa: BLE001
        return {"records": [], "field_coverage": {}, "record_count": 0, "source_paths": []}

    markdown_path_filters = _normalize_search_terms(path_filters) if path_filters is not None else []
    if markdown_path_filters:
        lowered_filters = [path_filter.casefold().replace("\\", "/") for path_filter in markdown_path_filters]

        def path_matches_filter(path: Path) -> bool:
            relative = _relative_path(path, root).casefold()
            name = path.name.casefold()
            return any(
                relative == lowered_filter
                or relative.endswith("/" + lowered_filter)
                or name == lowered_filter
                or lowered_filter in relative
                for lowered_filter in lowered_filters
            )

        paths = [path for path in paths if path_matches_filter(path)]

    records_by_key: dict[str, dict[str, object]] = {}
    source_paths: set[str] = set()
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        relative_path = _relative_path(path, root)
        for block_index, (heading, block) in enumerate(_split_markdown_blocks(text)):
            fields = _extract_markdown_block_fields(block)
            key = _markdown_entity_key(fields)
            if key is None:
                continue
            source_paths.add(relative_path)
            record = records_by_key.setdefault(
                key,
                {
                    "__key": key,
                    "__source_paths": [],
                    "__block_indexes": [],
                    "__evidence": [],
                },
            )
            _merge_markdown_entity_fields(record, fields)
            if relative_path not in record["__source_paths"]:
                record["__source_paths"].append(relative_path)
            record["__block_indexes"].append(block_index)
            if include_evidence and len(record["__evidence"]) < 4:
                snippet = block.replace("\r", " ").replace("\n", " ").strip()
                if len(snippet) > evidence_chars:
                    snippet = snippet[:evidence_chars].rstrip() + "..."
                record["__evidence"].append(
                    {
                        "path": relative_path,
                        "block_index": block_index,
                        "heading": heading,
                        "snippet": snippet,
                    }
                )

    all_records = list(records_by_key.values())
    search_terms = _normalize_search_terms(terms) if terms is not None else _question_markdown_terms(question)
    lowered_terms = [term.casefold() for term in search_terms if term]

    def rank(record: dict[str, object]) -> tuple[float, str]:
        rendered = json.dumps(record, ensure_ascii=False).casefold()
        score = sum(1 for term in lowered_terms if term in rendered)
        for term in lowered_terms:
            if term in {str(key).casefold() for key in record if not key.startswith("__")}:
                score += 1.5
        return (-float(score), str(record.get("__key", "")))

    if lowered_terms:
        all_records.sort(key=rank)
    else:
        all_records.sort(key=lambda record: str(record.get("__key", "")))

    field_coverage: dict[str, int] = {}
    for record in records_by_key.values():
        for key, value in record.items():
            if key.startswith("__") or value is None or value == "" or value == []:
                continue
            field_coverage[key] = field_coverage.get(key, 0) + 1

    if max_records is not None and max_records >= 0:
        returned_records = all_records[:max_records]
    else:
        returned_records = all_records

    return {
        "records": returned_records,
        "field_coverage": dict(sorted(field_coverage.items())),
        "record_count": len(all_records),
        "source_paths": sorted(source_paths),
        "note": (
            "Rows are parsed from markdown prose and merged by patient_id/id/cards_id/record_id. "
            "Use as grounded extraction hints and verify edge cases before finalizing."
        ),
    }


def _looks_like_markdown_paths(value: object) -> bool:
    terms = _normalize_search_terms(value)
    return bool(terms) and all(
        term.casefold().endswith(".md") or "/" in term or "\\" in term
        for term in terms
    )


def _terms_from_pattern(pattern: object) -> list[str]:
    if pattern is None:
        return []
    raw = str(pattern)
    terms: list[str] = []
    for piece in raw.split("|"):
        cleaned = re.sub(r"[^A-Za-z0-9_ -]+", " ", piece).strip()
        if cleaned:
            terms.append(cleaned)
        terms.extend(token for token in re.findall(r"[A-Za-z0-9_ -]{3,}", cleaned) if token != cleaned)
    return _normalize_search_terms(terms)


def _python_helper_modules(
    functions: dict[str, Any],
) -> dict[str, tuple[types.ModuleType | None, dict[str, Any], set[str]]]:
    previous_modules: dict[str, tuple[types.ModuleType | None, dict[str, Any], set[str]]] = {}
    markdown_helpers = {
        "retrieve_knowledge": functions["retrieve_knowledge"],
        "search_markdown": functions["search_markdown"],
        "extract_markdown_records": functions["extract_markdown_records"],
        "markdown_entity_table": functions["markdown_entity_table"],
        "link_question_to_data": functions["link_question_to_data"],
    }
    json_helpers = {
        "load_json_records": functions["load_json_records"],
        "load_json_table": functions["load_json_table"],
        "load_json_df": functions["load_json_table"],
    }
    module_specs = {
        "retrieve_knowledge": {"retrieve_knowledge": functions["retrieve_knowledge"]},
        "search_markdown": {"search_markdown": functions["search_markdown"]},
        "extract_markdown_records": {
            "extract_markdown_records": functions["extract_markdown_records"],
        },
        "markdown_entity_table": {"markdown_entity_table": functions["markdown_entity_table"]},
        "extract_markdown_table": {"extract_markdown_table": functions["markdown_entity_table"]},
        "load_json_records": {"load_json_records": functions["load_json_records"]},
        "load_json_table": {"load_json_table": functions["load_json_table"]},
        "link_question_to_data": {"link_question_to_data": functions["link_question_to_data"]},
        "json_helpers": json_helpers,
        "markdown_helpers": markdown_helpers,
        "knowledge": markdown_helpers,
        "utils": {**markdown_helpers, **json_helpers},
        "tools": {
            "retrieve_knowledge": functions["retrieve_knowledge"],
            "search_markdown": functions["search_markdown"],
            "extract_markdown_records": functions["extract_markdown_records"],
            "markdown_entity_table": functions["markdown_entity_table"],
            "extract_markdown_table": functions["markdown_entity_table"],
            "load_json_records": functions["load_json_records"],
            "load_json_table": functions["load_json_table"],
            "link_question_to_data": functions["link_question_to_data"],
        },
        "data_agent_baseline.tools": {**markdown_helpers, **json_helpers},
    }
    for module_name, attributes in module_specs.items():
        existing = sys.modules.get(module_name)
        if isinstance(existing, types.ModuleType) and module_name == "data_agent_baseline.tools":
            previous_attributes = {
                attribute_name: getattr(existing, attribute_name)
                for attribute_name in attributes
                if hasattr(existing, attribute_name)
            }
            added_attributes = set(attributes) - set(previous_attributes)
            previous_modules[module_name] = (existing, previous_attributes, added_attributes)
            for attribute_name, attribute_value in attributes.items():
                setattr(existing, attribute_name, attribute_value)
            continue

        previous_modules[module_name] = (
            existing if isinstance(existing, types.ModuleType) else None,
            {},
            set(),
        )
        module = types.ModuleType(module_name)
        for attribute_name, attribute_value in attributes.items():
            setattr(module, attribute_name, attribute_value)
        sys.modules[module_name] = module
    return previous_modules


def _restore_python_helper_modules(
    previous_modules: dict[str, tuple[types.ModuleType | None, dict[str, Any], set[str]]],
) -> None:
    for module_name, (previous_module, previous_attributes, added_attributes) in previous_modules.items():
        if previous_attributes or added_attributes:
            current_module = sys.modules.get(module_name)
            if isinstance(current_module, types.ModuleType):
                for attribute_name in added_attributes:
                    if hasattr(current_module, attribute_name):
                        delattr(current_module, attribute_name)
                for attribute_name, attribute_value in previous_attributes.items():
                    setattr(current_module, attribute_name, attribute_value)
            if previous_module is not None:
                sys.modules[module_name] = previous_module
            continue
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def _json_records_from_payload(payload: object) -> list[object]:
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return list(records)

        list_items = [(key, value) for key, value in payload.items() if isinstance(value, list)]
        dict_list_items = [
            (key, value)
            for key, value in list_items
            if value and all(isinstance(item, dict) for item in value[:5])
        ]
        if len(dict_list_items) == 1:
            return list(dict_list_items[0][1])
        if len(list_items) == 1:
            return list(list_items[0][1])
        return [payload]
    return [payload]


def _linking_tokens(text: object) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", spaced.replace("_", " "))
        if token and token.casefold() not in LINKING_STOPWORDS
    }
    expanded: set[str] = set()
    for token in tokens:
        expanded.add(token)
        if len(token) > 3 and token.endswith("s"):
            expanded.add(token[:-1])
    return expanded


def _question_linking_terms(question: str, terms: object = None) -> list[str]:
    values: list[str] = []

    def add(value: object) -> None:
        text = str(value).strip()
        if not text or text.casefold() in {item.casefold() for item in values}:
            return
        if text.casefold() in LINKING_STOPWORDS:
            return
        values.append(text)

    if terms is not None:
        for term in _normalize_search_terms(terms):
            add(term)

    for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', question):
        add(match.group(1) or match.group(2))
    for match in RECORD_ID_PATTERN.finditer(question):
        add(match.group(0))
    capitalized_pattern = r"\b(?:[A-Z][A-Za-z0-9]*|[A-Z]{2,})(?:[ -]+(?:[A-Z][A-Za-z0-9]*|[A-Z]{2,}))*\b"
    for match in re.finditer(capitalized_pattern, question):
        phrase = match.group(0).strip()
        if len(phrase) > 2 and phrase.casefold() not in LINKING_STOPWORDS:
            add(phrase)
    for token in re.findall(r"[A-Za-z0-9_+-]{3,}", question):
        if token.casefold() not in LINKING_STOPWORDS and token.isupper():
            add(token)
    return values[:24]


def _column_looks_like_id(column: str, value: object = "") -> bool:
    tokens = _linking_tokens(column)
    return bool(tokens & ID_COLUMN_TERMS) or bool(RECORD_ID_PATTERN.search(str(value)))


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _trim_linking_text(value: object, max_chars: int = 160) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _linking_condition(source_type: str, table: str, column: str, value: str, example: object) -> str:
    exact = str(example).casefold() == value.casefold()
    operator = "==" if exact else "contains"
    if source_type == "sqlite":
        return f"{table}.{column} {operator} {value!r}"
    return f"{column} {operator} {value!r}"


def _numeric_filter_hints(question: str, columns: list[dict[str, object]], max_candidates: int) -> list[dict[str, object]]:
    hints: list[dict[str, object]] = []
    patterns = [
        (r"\bbetween\s+([-+]?\d+(?:\.\d+)?)\s+(?:and|to)\s+([-+]?\d+(?:\.\d+)?)", "between"),
        (r"\b(?:less than|under|below|younger than|aren't\s+[-+]?\d+(?:\.\d+)?\s+yet)\s*([-+]?\d+(?:\.\d+)?)?", "<"),
        (r"\b(?:more than|greater than|above|over|older than)\s+([-+]?\d+(?:\.\d+)?)", ">"),
    ]
    lowered = question.casefold()
    for pattern, operator in patterns:
        for match in re.finditer(pattern, lowered):
            values = [group for group in match.groups() if group]
            if operator == "<" and not values:
                nearby_number = re.search(r"aren't\s+([-+]?\d+(?:\.\d+)?)\s+yet", lowered)
                values = [nearby_number.group(1)] if nearby_number is not None else []
            if not values:
                continue
            window_start = max(0, match.start() - 60)
            window_end = min(len(question), match.end() + 60)
            phrase = question[window_start:window_end].strip()
            local_tokens = _linking_tokens(question[window_start:window_end])
            age_context = bool(
                {"age", "old", "older", "younger", "birth", "birthday"} & local_tokens
            ) or "yet" in phrase.casefold()
            ranked_columns: list[tuple[float, dict[str, object]]] = []
            for column in columns:
                column_tokens = set(column.get("tokens", []))
                score = len(local_tokens & column_tokens)
                if column_tokens & NUMERIC_LINKING_TERMS:
                    score += 0.5
                if age_context and column_tokens & {"age", "birth", "birthday", "year"}:
                    score += 3.0
                if age_context and str(column.get("column", "")).casefold() in {"birth_year", "birthday", "birthdate"}:
                    score += 2.0
                if score > 0:
                    ranked_columns.append((score, column))
            ranked_columns.sort(key=lambda item: (-item[0], str(item[1]["label"])))
            hints.append(
                {
                    "phrase": phrase,
                    "operator": operator,
                    "values": values,
                    "candidate_columns": [
                        str(column["label"]) for _, column in ranked_columns[:max_candidates]
                    ],
                }
            )
    return hints[:max_candidates]


def _collect_row_ids(row: dict[str, object], *, max_items: int = 8) -> dict[str, str]:
    ids: dict[str, str] = {}
    for column, value in row.items():
        text = str(value)
        if not _column_looks_like_id(column, value):
            continue
        if not text or text.casefold() in {"nan", "none", "null"}:
            continue
        ids[str(column)] = _trim_linking_text(text, 80)
        if len(ids) >= max_items:
            break
    return ids


def _add_linking_value_match(
    matches: list[dict[str, object]],
    seen: set[tuple[str, str, str]],
    *,
    term: str,
    source_type: str,
    path: str,
    table: str,
    column: str,
    example: object,
    row_ids: dict[str, str] | None = None,
    max_per_term: int = 4,
) -> None:
    term_count = sum(1 for match in matches if match.get("term") == term)
    if term_count >= max_per_term:
        return
    label = f"{table}.{column}" if source_type == "sqlite" else f"{path}:{column}"
    key = (term.casefold(), label.casefold(), str(example).casefold())
    if key in seen:
        return
    seen.add(key)
    matches.append(
        {
            "term": term,
            "source_type": source_type,
            "path": path,
            "table": table,
            "column": column,
            "label": label,
            "example_value": _trim_linking_text(example),
            "row_ids": row_ids or {},
            "usable_filter": _linking_condition(source_type, table, column, term, example),
        }
    )


def _scan_csv_for_linking(
    path: Path,
    root: Path,
    search_terms: list[str],
    *,
    max_rows: int,
    max_per_term: int,
    matches: list[dict[str, object]],
    seen_matches: set[tuple[str, str, str]],
    columns: list[dict[str, object]],
    id_value_sets: dict[str, set[str]],
) -> None:
    rel_path = _relative_path(path, root)
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [str(name) for name in (reader.fieldnames or [])]
            for column in fieldnames:
                columns.append(
                    {
                        "source_type": "csv",
                        "path": rel_path,
                        "table": rel_path,
                        "column": column,
                        "label": f"{rel_path}:{column}",
                        "tokens": sorted(_linking_tokens(column) | _linking_tokens(rel_path)),
                    }
                )
            for row_index, row in enumerate(reader):
                if row_index >= max_rows:
                    break
                normalized_row = {str(key): value for key, value in row.items() if key is not None}
                row_ids = _collect_row_ids(normalized_row)
                row_tokens = _linking_tokens(" ".join(str(value) for value in normalized_row.values()))
                for term in search_terms:
                    term_tokens = _linking_tokens(term)
                    specific_term_tokens = term_tokens - ROW_MATCH_GENERIC_TOKENS
                    token_columns = {
                        column
                        for token in term_tokens
                        for column, value in normalized_row.items()
                        if token in _linking_tokens(value)
                    }
                    if (
                        len(term_tokens) > 1
                        and len(specific_term_tokens) > 1
                        and len(token_columns) > 1
                        and term_tokens <= row_tokens
                    ):
                        _add_linking_value_match(
                            matches,
                            seen_matches,
                            term=term,
                            source_type="csv",
                            path=rel_path,
                            table=rel_path,
                            column="<row>",
                            example=" | ".join(_trim_linking_text(value, 40) for value in normalized_row.values()),
                            row_ids=row_ids,
                            max_per_term=max_per_term,
                        )
                for column, value in normalized_row.items():
                    text = str(value)
                    if _column_looks_like_id(column, value):
                        label = f"{rel_path}:{column}"
                        id_value_sets.setdefault(label, set()).add(text)
                    lowered = text.casefold()
                    for term in search_terms:
                        if term.casefold() in lowered:
                            _add_linking_value_match(
                                matches,
                                seen_matches,
                                term=term,
                                source_type="csv",
                                path=rel_path,
                                table=rel_path,
                                column=column,
                                example=value,
                                row_ids=row_ids,
                                max_per_term=max_per_term,
                            )
    except Exception:  # noqa: BLE001
        return


def _scan_json_for_linking(
    path: Path,
    root: Path,
    search_terms: list[str],
    *,
    max_rows: int,
    max_per_term: int,
    matches: list[dict[str, object]],
    seen_matches: set[tuple[str, str, str]],
    columns: list[dict[str, object]],
    id_value_sets: dict[str, set[str]],
) -> None:
    rel_path = _relative_path(path, root)
    try:
        if path.stat().st_size > 8_000_000:
            return
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return
    table = str(payload.get("table")) if isinstance(payload, dict) and payload.get("table") else rel_path
    records = _json_records_from_payload(payload)
    seen_columns: set[str] = set()
    for record in records[:max_rows]:
        if not isinstance(record, dict):
            continue
        for column in record:
            column_text = str(column)
            if column_text not in seen_columns:
                seen_columns.add(column_text)
                columns.append(
                    {
                        "source_type": "json",
                        "path": rel_path,
                        "table": table,
                        "column": column_text,
                        "label": f"{table}.{column_text} ({rel_path})" if table != rel_path else f"{rel_path}:{column_text}",
                        "tokens": sorted(_linking_tokens(column_text) | _linking_tokens(table) | _linking_tokens(rel_path)),
                    }
                )
        normalized_row = {str(key): value for key, value in record.items()}
        row_ids = _collect_row_ids(normalized_row)
        row_tokens = _linking_tokens(" ".join(str(value) for value in normalized_row.values()))
        for term in search_terms:
            term_tokens = _linking_tokens(term)
            specific_term_tokens = term_tokens - ROW_MATCH_GENERIC_TOKENS
            token_columns = {
                column
                for token in term_tokens
                for column, value in normalized_row.items()
                if token in _linking_tokens(value)
            }
            if (
                len(term_tokens) > 1
                and len(specific_term_tokens) > 1
                and len(token_columns) > 1
                and term_tokens <= row_tokens
            ):
                _add_linking_value_match(
                    matches,
                    seen_matches,
                    term=term,
                    source_type="json",
                    path=rel_path,
                    table=table,
                    column="<row>",
                    example=" | ".join(_trim_linking_text(value, 40) for value in normalized_row.values()),
                    row_ids=row_ids,
                    max_per_term=max_per_term,
                )
        for column, value in normalized_row.items():
            text = str(value)
            if _column_looks_like_id(column, value):
                label = f"{rel_path}:{column}"
                id_value_sets.setdefault(label, set()).add(text)
            lowered = text.casefold()
            for term in search_terms:
                if term.casefold() in lowered:
                    _add_linking_value_match(
                        matches,
                        seen_matches,
                        term=term,
                        source_type="json",
                        path=rel_path,
                        table=table,
                        column=column,
                        example=value,
                        row_ids=row_ids,
                        max_per_term=max_per_term,
                    )


def _scan_sqlite_for_linking(
    path: Path,
    root: Path,
    search_terms: list[str],
    *,
    max_rows: int,
    max_per_term: int,
    matches: list[dict[str, object]],
    seen_matches: set[tuple[str, str, str]],
    columns: list[dict[str, object]],
    id_value_sets: dict[str, set[str]],
) -> None:
    rel_path = _relative_path(path, root)
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001
        return
    try:
        table_rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        for (table_name,) in table_rows[:20]:
            table = str(table_name)
            quoted_table = _quote_sqlite_identifier(table)
            column_rows = connection.execute(f"PRAGMA table_info({quoted_table})").fetchall()
            table_columns = [str(row[1]) for row in column_rows]
            for column in table_columns:
                columns.append(
                    {
                        "source_type": "sqlite",
                        "path": rel_path,
                        "table": table,
                        "column": column,
                        "label": f"{table}.{column}",
                        "tokens": sorted(_linking_tokens(column) | _linking_tokens(table)),
                    }
                )
            for column in table_columns:
                quoted_column = _quote_sqlite_identifier(column)
                if _column_looks_like_id(column):
                    try:
                        rows = connection.execute(
                            f"SELECT {quoted_column} FROM {quoted_table} LIMIT ?",
                            (max_rows,),
                        ).fetchall()
                    except Exception:  # noqa: BLE001
                        rows = []
                    label = f"{table}.{column}"
                    for (value,) in rows:
                        if value is not None:
                            id_value_sets.setdefault(label, set()).add(str(value))
                for term in search_terms:
                    clean = term.replace("%", "").replace("_", "")
                    if not clean:
                        continue
                    try:
                        row = connection.execute(
                            f"SELECT * FROM {quoted_table} "
                            f"WHERE lower(CAST({quoted_column} AS TEXT)) LIKE ? LIMIT 1",
                            (f"%{clean.casefold()}%",),
                        ).fetchone()
                    except Exception:  # noqa: BLE001
                        row = None
                    if row is None:
                        continue
                    normalized_row = dict(zip(table_columns, row, strict=False))
                    _add_linking_value_match(
                        matches,
                        seen_matches,
                        term=term,
                        source_type="sqlite",
                        path=rel_path,
                        table=table,
                        column=column,
                        example=normalized_row.get(column, ""),
                        row_ids=_collect_row_ids(normalized_row),
                        max_per_term=max_per_term,
                    )
    except Exception:  # noqa: BLE001
        return
    finally:
        connection.close()


def _join_candidates_from_id_sets(id_value_sets: dict[str, set[str]], *, max_candidates: int) -> list[dict[str, object]]:
    labels = sorted(id_value_sets)
    candidates: list[tuple[int, str, str, list[str]]] = []
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            left_values = {value for value in id_value_sets[left] if value}
            right_values = {value for value in id_value_sets[right] if value}
            if not left_values or not right_values:
                continue
            overlap = sorted((left_values & right_values), key=str)[:5]
            if not overlap:
                continue
            candidates.append((len(overlap), left, right, overlap))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {
            "left": left,
            "right": right,
            "overlap_count": overlap_count,
            "sample_values": overlap,
        }
        for overlap_count, left, right, overlap in candidates[:max_candidates]
    ]


def _question_prefers_markdown_entity_table(question: str) -> bool:
    tokens = _linking_tokens(question)
    positive = bool(tokens & MARKDOWN_ENTITY_TABLE_TRIGGER_TERMS)
    negative = bool(tokens & MARKDOWN_ENTITY_TABLE_AVOID_TERMS)
    if not positive:
        return False
    if negative and not (tokens & {"height", "publisher", "patient", "patients", "creatinine", "birth", "birthday", "age"}):
        return False
    return True


def _markdown_entity_table_is_relevant(
    question: str,
    parsed_table: dict[str, object],
) -> bool:
    if not _question_prefers_markdown_entity_table(question):
        return False
    field_coverage = parsed_table.get("field_coverage")
    if not isinstance(field_coverage, dict):
        return False
    record_count = int(parsed_table.get("record_count", 0) or 0)
    if record_count < 1:
        return False
    fields = {str(field) for field, count in field_coverage.items() if int(count or 0) > 0}
    if not fields & MARKDOWN_ENTITY_TABLE_FIELDS:
        return False

    question_tokens = _linking_tokens(question)
    field_tokens = {
        token
        for field in fields
        for token in _linking_tokens(field)
    }
    if question_tokens & field_tokens:
        return True
    if question_tokens & {"hero", "heroes", "superhero", "superheroes"} and fields & {"id", "height_cm", "publisher_id"}:
        return True
    if question_tokens & {"patient", "patients"} and fields & {"patient_id", "birth_year", "creatinine"}:
        return True
    return False


def build_question_data_links(
    context_root: str | Path,
    question: str,
    terms: object = None,
    *,
    max_candidates: int = 5,
    max_rows: int = 5000,
    include_markdown: bool = True,
) -> dict[str, object]:
    """Return compact, data-grounded linking hints for question terms.

    The helper intentionally returns candidates, not final answers. It grounds
    natural-language mentions in observed structured values, markdown record IDs,
    row IDs, and overlap-based join hints so the model does less guessing.
    """

    root = Path(context_root).resolve()
    question_terms = _question_linking_terms(question, terms)
    markdown_records: list[dict[str, object]] = []
    markdown_ids: list[str] = []
    if include_markdown:
        markdown_records = extract_markdown_records(
            root,
            question_terms or None,
            question=question,
            max_records=max_candidates * 2,
            max_chars=360,
            include_linked_ids=True,
            link_depth=2,
        )
        seen_ids: set[str] = set()
        for record in markdown_records:
            for value in record.get("ids", []):
                text_value = str(value)
                if RECORD_ID_PATTERN.fullmatch(text_value) and text_value.casefold() not in seen_ids:
                    seen_ids.add(text_value.casefold())
                    markdown_ids.append(text_value)

    search_terms = _normalize_search_terms([*question_terms, *markdown_ids])[:40]
    columns: list[dict[str, object]] = []
    value_matches: list[dict[str, object]] = []
    seen_matches: set[tuple[str, str, str]] = set()
    id_value_sets: dict[str, set[str]] = {}

    try:
        files = sorted(path for path in root.rglob("*") if path.is_file())
    except Exception:  # noqa: BLE001
        files = []
    for path in files[:80]:
        suffix = path.suffix.casefold()
        if suffix == ".csv":
            _scan_csv_for_linking(
                path,
                root,
                search_terms,
                max_rows=max_rows,
                max_per_term=max_candidates,
                matches=value_matches,
                seen_matches=seen_matches,
                columns=columns,
                id_value_sets=id_value_sets,
            )
        elif suffix == ".json":
            _scan_json_for_linking(
                path,
                root,
                search_terms,
                max_rows=max_rows,
                max_per_term=max_candidates,
                matches=value_matches,
                seen_matches=seen_matches,
                columns=columns,
                id_value_sets=id_value_sets,
            )
        elif suffix in {".sqlite", ".sqlite3", ".db"}:
            _scan_sqlite_for_linking(
                path,
                root,
                search_terms,
                max_rows=max_rows,
                max_per_term=max_candidates,
                matches=value_matches,
                seen_matches=seen_matches,
                columns=columns,
                id_value_sets=id_value_sets,
            )

    markdown_entity_summary: dict[str, object] | None = None
    if include_markdown and _question_prefers_markdown_entity_table(question):
        parsed_table = build_markdown_entity_table(
            root,
            question_terms or None,
            question=question,
            max_records=None,
            include_evidence=True,
            evidence_chars=220,
        )
        if not _markdown_entity_table_is_relevant(question, parsed_table):
            parsed_table = {}
        parsed_records = [
            record
            for record in parsed_table.get("records", [])
            if isinstance(record, dict)
        ]
        if parsed_records:
            markdown_entity_summary = {
                "record_count": parsed_table.get("record_count", len(parsed_records)),
                "field_coverage": parsed_table.get("field_coverage", {}),
                "source_paths": parsed_table.get("source_paths", []),
                "sample_records": [
                    {
                        key: value
                        for key, value in record.items()
                        if key != "__evidence"
                    }
                    for record in parsed_records[:max_candidates]
                ],
                "note": (
                    "These are row-like records parsed from markdown and merged by id/patient_id/cards_id. "
                    "Use only when field_coverage directly contains requested entity attributes."
                ),
            }
            parsed_field_names = sorted(
                {
                    key
                    for record in parsed_records
                    for key, value in record.items()
                    if not key.startswith("__") and value is not None and value != "" and value != []
                }
            )
            for field in parsed_field_names:
                columns.append(
                    {
                        "source_type": "markdown",
                        "path": ",".join(str(path) for path in parsed_table.get("source_paths", [])) or "<markdown>",
                        "table": "markdown_entity_table",
                        "column": field,
                        "label": f"markdown_entity_table.{field}",
                        "tokens": sorted(_linking_tokens(field) | _linking_tokens("markdown entity table")),
                    }
                )
            for record in parsed_records:
                row_ids = _collect_row_ids(
                    {
                        key: value
                        for key, value in record.items()
                        if not key.startswith("__")
                    }
                )
                for field, value in record.items():
                    if field.startswith("__") or value is None or value == "" or value == []:
                        continue
                    label = f"markdown_entity_table.{field}"
                    if _column_looks_like_id(field, value):
                        if isinstance(value, list):
                            values = [str(item) for item in value]
                        else:
                            values = [str(value)]
                        for item in values:
                            id_value_sets.setdefault(label, set()).add(item)
                    lowered_value = str(value).casefold()
                    for term in search_terms:
                        if term.casefold() in lowered_value:
                            _add_linking_value_match(
                                value_matches,
                                seen_matches,
                                term=term,
                                source_type="markdown",
                                path=",".join(str(path) for path in record.get("__source_paths", [])) or "<markdown>",
                                table="markdown_entity_table",
                                column=field,
                                example=value,
                                row_ids=row_ids,
                                max_per_term=max_candidates,
                            )

    if include_markdown:
        row_id_terms: list[str] = []
        seen_row_ids: set[str] = set()
        for match in value_matches:
            row_ids = match.get("row_ids")
            if not isinstance(row_ids, dict):
                continue
            for value in row_ids.values():
                text_value = str(value)
                if text_value.casefold() in seen_row_ids:
                    continue
                seen_row_ids.add(text_value.casefold())
                row_id_terms.append(text_value)
        if row_id_terms:
            linked_records = extract_markdown_records(
                root,
                row_id_terms[: max_candidates * 3],
                question=question,
                max_records=max_candidates * 3,
                max_chars=360,
                include_linked_ids=True,
                link_depth=2,
            )
            seen_record_keys = {
                (str(record.get("path")), str(record.get("block_index")))
                for record in markdown_records
            }
            for record in linked_records:
                key = (str(record.get("path")), str(record.get("block_index")))
                if key in seen_record_keys:
                    continue
                seen_record_keys.add(key)
                markdown_records.append(record)
                for value in record.get("ids", []):
                    text_value = str(value)
                    if RECORD_ID_PATTERN.fullmatch(text_value) and text_value.casefold() not in {item.casefold() for item in markdown_ids}:
                        markdown_ids.append(text_value)

    question_tokens = _linking_tokens(question)
    ranked_columns: list[tuple[float, dict[str, object]]] = []
    for column in columns:
        column_tokens = set(column.get("tokens", []))
        score = len(question_tokens & column_tokens)
        if column_tokens & NUMERIC_LINKING_TERMS and question_tokens & NUMERIC_LINKING_TERMS:
            score += 0.5
        if score > 0:
            ranked_columns.append((score, column))
    ranked_columns.sort(key=lambda item: (-item[0], str(item[1]["label"])))

    matches_by_term: dict[str, list[dict[str, object]]] = {}
    for match in value_matches:
        matches_by_term.setdefault(str(match["term"]), []).append(match)
    entity_links: list[dict[str, object]] = []
    for term in question_terms[:max_candidates * 2]:
        direct_matches = matches_by_term.get(term, [])[:max_candidates]
        direct_row_id_values = {
            str(value)
            for match in direct_matches
            if isinstance(match.get("row_ids"), dict)
            for value in match["row_ids"].values()
        }
        related_records = [
            record
            for record in markdown_records
            if term.casefold() in " ".join(str(value) for value in record.get("matched_terms", [])).casefold()
            or term.casefold() in str(record.get("snippet", "")).casefold()
            or bool(direct_row_id_values & {str(value) for value in record.get("linked_from_ids", [])})
            or any(value in str(record.get("snippet", "")) for value in direct_row_id_values)
        ][:max_candidates]
        related_ids = []
        seen_related_ids: set[str] = set()
        for record in related_records:
            for value in record.get("ids", []):
                text_value = str(value)
                if text_value.casefold() in seen_related_ids:
                    continue
                seen_related_ids.add(text_value.casefold())
                related_ids.append(text_value)
        id_matches: list[dict[str, object]] = []
        for related_id in related_ids:
            id_matches.extend(matches_by_term.get(related_id, []))
        if direct_matches or related_records or id_matches:
            entity_links.append(
                {
                    "mention": term,
                    "direct_structured_matches": [
                        {
                            "label": match["label"],
                            "example_value": match["example_value"],
                            "row_ids": match["row_ids"],
                            "usable_filter": match["usable_filter"],
                        }
                        for match in direct_matches
                    ],
                    "markdown_record_ids": related_ids[:max_candidates * 2],
                    "markdown_evidence": [
                        {
                            "path": record.get("path"),
                            "linked_from_ids": record.get("linked_from_ids", []),
                            "ids": record.get("ids", [])[:8],
                            "numbers": record.get("numbers", [])[:8],
                            "snippet": _trim_linking_text(record.get("snippet", ""), 260),
                        }
                        for record in related_records[:3]
                    ],
                    "id_structured_matches": [
                        {
                            "term": match["term"],
                            "label": match["label"],
                            "example_value": match["example_value"],
                            "row_ids": match["row_ids"],
                            "usable_filter": match["usable_filter"],
                        }
                        for match in id_matches[:max_candidates]
                    ],
                }
            )

    unresolved = [
        term
        for term in question_terms
        if term not in matches_by_term
        and not any(term.casefold() in str(record.get("snippet", "")).casefold() for record in markdown_records)
    ][:max_candidates]
    compact_markdown_records = [
        {
            "path": record.get("path"),
            "matched_terms": record.get("matched_terms", []),
            "linked_from_ids": record.get("linked_from_ids", []),
            "ids": record.get("ids", [])[:10],
            "numbers": record.get("numbers", [])[:8],
            "snippet": _trim_linking_text(record.get("snippet", ""), 360),
        }
        for record in markdown_records[:max_candidates]
    ]
    return {
        "question_terms": question_terms,
        "entity_links": entity_links[:max_candidates],
        "value_matches": [
            {
                "term": match["term"],
                "label": match["label"],
                "example_value": match["example_value"],
                "row_ids": match["row_ids"],
                "usable_filter": match["usable_filter"],
            }
            for match in value_matches[: max_candidates * 4]
        ],
        "column_candidates": [
            {
                "label": column["label"],
                "source_type": column["source_type"],
                "path": column["path"],
                "table": column["table"],
                "column": column["column"],
            }
            for _, column in ranked_columns[: max_candidates * 2]
        ],
        "join_candidates": _join_candidates_from_id_sets(id_value_sets, max_candidates=max_candidates),
        "numeric_filters": _numeric_filter_hints(question, columns, max_candidates),
        "markdown_records": compact_markdown_records,
        "markdown_entity_table": markdown_entity_summary,
        "unresolved_terms": unresolved,
        "note": (
            "Candidates are grounded in observed values/IDs but still require verification. "
            "Use row_ids and usable_filter fields to connect mentions to real data before computing."
        ),
    }


def _run_python_code(
    context_root: str,
    working_dir: str,
    question: str,
    code: str,
    stdout_path: str,
    stderr_path: str,
    queue: multiprocessing.Queue[Any],
) -> None:
    resolved_context_root = Path(context_root).resolve()
    resolved_working_dir = Path(working_dir).resolve()

    def retrieve_knowledge_helper(**kwargs: Any) -> list[dict[str, object]]:
        return retrieve_knowledge_snippets(
            resolved_context_root,
            question,
            **kwargs,
        )

    def search_markdown_helper(*args: object, **kwargs: Any) -> list[dict[str, object]]:
        first_arg = args[0] if args else kwargs.pop("terms", None)
        second_arg = args[1] if len(args) > 1 else None
        path_filters = kwargs.pop("path_filters", None) or kwargs.pop("paths", None) or kwargs.pop("files", None)
        terms = first_arg

        if first_arg is not None and _looks_like_markdown_paths(first_arg):
            path_filters = first_arg
            terms = second_arg
        elif second_arg is not None:
            terms = [first_arg, second_arg] if first_arg is not None else second_arg

        pattern_terms = _terms_from_pattern(kwargs.pop("pattern", None))
        fields = kwargs.pop("fields", None)
        term_list: list[str] = []
        if terms is not None:
            term_list.extend(_normalize_search_terms(terms))
        if fields is not None:
            term_list.extend(_normalize_search_terms(fields))
        term_list.extend(pattern_terms)
        terms = _normalize_search_terms(term_list) if term_list else kwargs.pop("query", None) or question
        if "top_k" in kwargs and "max_matches" not in kwargs:
            kwargs["max_matches"] = kwargs.pop("top_k")
        if "max_chars" in kwargs and "context_chars" not in kwargs:
            kwargs["context_chars"] = kwargs.pop("max_chars")
        allowed_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"max_matches", "context_chars"}
        }
        return search_markdown_documents(
            resolved_context_root,
            terms,
            path_filters=path_filters,
            **allowed_kwargs,
        )

    def extract_markdown_records_helper(*args: object, **kwargs: Any) -> list[dict[str, object]]:
        first_arg = args[0] if args else kwargs.pop("terms", None)
        second_arg = args[1] if len(args) > 1 else None
        path_filters = kwargs.pop("path_filters", None) or kwargs.pop("paths", None) or kwargs.pop("files", None)
        terms = first_arg

        if first_arg is not None and _looks_like_markdown_paths(first_arg):
            path_filters = first_arg
            terms = second_arg
        elif second_arg is not None:
            terms = second_arg

        pattern_terms = _terms_from_pattern(kwargs.pop("pattern", None))
        fields = kwargs.pop("fields", None)
        query = str(kwargs.pop("query", question) or question)
        max_records = int(kwargs.pop("max_records", kwargs.pop("max_matches", 12)) or 12)
        max_chars = int(kwargs.pop("max_chars", kwargs.pop("context_chars", 900)) or 900)
        include_linked_ids = bool(kwargs.pop("include_linked_ids", True))
        link_depth = int(kwargs.pop("link_depth", 2) or 0)
        kwargs.pop("include_context", None)

        term_list: list[str] = []
        if terms is not None:
            term_list.extend(_normalize_search_terms(terms))
        if fields is not None:
            term_list.extend(_normalize_search_terms(fields))
        term_list.extend(pattern_terms)
        normalized_terms: object = _normalize_search_terms(term_list) if term_list else None

        return extract_markdown_records(
            resolved_context_root,
            normalized_terms,
            question=query,
            path_filters=path_filters,
            max_records=max_records,
            max_chars=max_chars,
            include_linked_ids=include_linked_ids,
            link_depth=link_depth,
        )

    def markdown_entity_table_helper(*args: object, **kwargs: Any) -> object:
        first_arg = args[0] if args else kwargs.pop("terms", None)
        second_arg = args[1] if len(args) > 1 else None
        path_filters = kwargs.pop("path_filters", None) or kwargs.pop("paths", None) or kwargs.pop("files", None)
        terms = first_arg

        if first_arg is not None and _looks_like_markdown_paths(first_arg):
            path_filters = first_arg
            terms = second_arg
        elif second_arg is not None:
            terms = second_arg

        pattern_terms = _terms_from_pattern(kwargs.pop("pattern", None))
        fields = kwargs.pop("fields", None)
        query = str(kwargs.pop("query", kwargs.pop("question", question)) or question)
        include_metadata = bool(kwargs.pop("include_metadata", kwargs.pop("metadata", False)))
        include_evidence = bool(kwargs.pop("include_evidence", True))
        evidence_chars = int(kwargs.pop("evidence_chars", kwargs.pop("max_chars", 260)) or 260)
        max_records_arg = kwargs.pop("max_records", None)
        if max_records_arg is None:
            max_records: int | None = None
        else:
            max_records = int(max_records_arg)

        term_list: list[str] = []
        if terms is not None:
            term_list.extend(_normalize_search_terms(terms))
        if fields is not None:
            term_list.extend(_normalize_search_terms(fields))
        term_list.extend(pattern_terms)
        normalized_terms: object = _normalize_search_terms(term_list) if term_list else None

        result = build_markdown_entity_table(
            resolved_context_root,
            normalized_terms,
            question=query,
            path_filters=path_filters,
            max_records=max_records,
            include_evidence=include_evidence,
            evidence_chars=max(80, min(evidence_chars, 1200)),
        )
        return result if include_metadata else result["records"]

    def resolve_context_path(path: str | Path) -> Path:
        candidate = Path(path)
        resolved = candidate.resolve() if candidate.is_absolute() else (resolved_context_root / candidate).resolve()
        if not resolved.is_relative_to(resolved_context_root):
            raise ValueError(f"Path is outside task context: {path}")
        return resolved

    def load_json_records_helper(path: str | Path) -> list[object]:
        payload = json.loads(resolve_context_path(path).read_text(encoding="utf-8"))
        return _json_records_from_payload(payload)

    def load_json_table_helper(path: str | Path) -> Any:
        if pd is None:
            raise RuntimeError("pandas is not available")
        return pd.DataFrame(load_json_records_helper(path))

    def link_question_to_data_helper(*args: object, **kwargs: Any) -> dict[str, object]:
        terms = args[0] if args else kwargs.pop("terms", None)
        max_candidates = int(kwargs.pop("max_candidates", kwargs.pop("top_k", 5)) or 5)
        max_rows = int(kwargs.pop("max_rows", kwargs.pop("scan_rows", 5000)) or 5000)
        include_markdown = bool(kwargs.pop("include_markdown", True))
        query = str(kwargs.pop("question", question) or question)
        return build_question_data_links(
            resolved_context_root,
            query,
            terms,
            max_candidates=max(1, min(max_candidates, 10)),
            max_rows=max(100, min(max_rows, 20000)),
            include_markdown=include_markdown,
        )

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "csv": csv,
        "json": json,
        "math": math,
        "os": os,
        "pd": pd,
        "re": re,
        "sqlite3": sqlite3,
        "sys": sys,
        "context_root": str(resolved_context_root),
        "CONTEXT_ROOT": str(resolved_context_root),
        "task_question": question,
        "QUESTION": question,
        "work_dir": str(resolved_working_dir),
        "WORK_DIR": str(resolved_working_dir),
        "Path": Path,
        "retrieve_knowledge": retrieve_knowledge_helper,
        "search_markdown": search_markdown_helper,
        "extract_markdown_records": extract_markdown_records_helper,
        "markdown_entity_table": markdown_entity_table_helper,
        "extract_markdown_table": markdown_entity_table_helper,
        "load_json_records": load_json_records_helper,
        "load_json_table": load_json_table_helper,
        "load_json_df": load_json_table_helper,
        "link_question_to_data": link_question_to_data_helper,
    }
    resolved_stdout_path = Path(stdout_path)
    resolved_stderr_path = Path(stderr_path)
    previous_helper_modules: dict[str, tuple[types.ModuleType | None, dict[str, Any], set[str]]] | None = None

    try:
        resolved_working_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(resolved_working_dir)
        previous_helper_modules = _python_helper_modules(
            {
                "retrieve_knowledge": retrieve_knowledge_helper,
                "search_markdown": search_markdown_helper,
                "extract_markdown_records": extract_markdown_records_helper,
                "markdown_entity_table": markdown_entity_table_helper,
                "load_json_records": load_json_records_helper,
                "load_json_table": load_json_table_helper,
                "link_question_to_data": link_question_to_data_helper,
            }
        )
        with _capture_process_streams(resolved_stdout_path, resolved_stderr_path):
            exec(code, namespace, namespace)
        queue.put({"success": True})
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if previous_helper_modules is not None:
            _restore_python_helper_modules(previous_helper_modules)


def execute_python_code(
    context_root: Path,
    code: str,
    *,
    timeout_seconds: int = 30,
    working_dir: Path | None = None,
    question: str = "",
) -> dict[str, Any]:
    resolved_context_root = context_root.resolve()
    resolved_working_dir = (working_dir or context_root).resolve()
    with tempfile.TemporaryDirectory() as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.txt"
        stderr_path = Path(temp_dir) / "stderr.txt"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")

        queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_run_python_code,
            args=(
                resolved_context_root.as_posix(),
                resolved_working_dir.as_posix(),
                question,
                code,
                stdout_path.as_posix(),
                stderr_path.as_posix(),
                queue,
            ),
        )
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": f"Python execution timed out after {timeout_seconds} seconds.",
            }

        if queue.empty():
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": "Python execution exited without returning a result.",
            }

        result = queue.get()
        result["output"] = _read_captured_stream(stdout_path)
        result["stderr"] = _read_captured_stream(stderr_path)
        return result
