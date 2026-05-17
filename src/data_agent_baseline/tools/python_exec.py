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


def _python_helper_modules(functions: dict[str, Any]) -> dict[str, types.ModuleType | None]:
    previous_modules: dict[str, types.ModuleType | None] = {}
    module_specs = {
        "retrieve_knowledge": {"retrieve_knowledge": functions["retrieve_knowledge"]},
        "search_markdown": {"search_markdown": functions["search_markdown"]},
        "extract_markdown_records": {
            "extract_markdown_records": functions["extract_markdown_records"],
        },
        "tools": {
            "retrieve_knowledge": functions["retrieve_knowledge"],
            "search_markdown": functions["search_markdown"],
            "extract_markdown_records": functions["extract_markdown_records"],
        },
    }
    for module_name, attributes in module_specs.items():
        existing = sys.modules.get(module_name)
        previous_modules[module_name] = existing if isinstance(existing, types.ModuleType) else None
        module = types.ModuleType(module_name)
        for attribute_name, attribute_value in attributes.items():
            setattr(module, attribute_name, attribute_value)
        sys.modules[module_name] = module
    return previous_modules


def _restore_python_helper_modules(previous_modules: dict[str, types.ModuleType | None]) -> None:
    for module_name, previous_module in previous_modules.items():
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


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

    def search_markdown_helper(terms: object = None, **kwargs: Any) -> list[dict[str, object]]:
        if terms is None:
            terms = kwargs.pop("terms", None) or kwargs.pop("query", None) or question
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
    }
    resolved_stdout_path = Path(stdout_path)
    resolved_stderr_path = Path(stderr_path)
    previous_helper_modules: dict[str, types.ModuleType | None] | None = None

    try:
        resolved_working_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(resolved_working_dir)
        previous_helper_modules = _python_helper_modules(
            {
                "retrieve_knowledge": retrieve_knowledge_helper,
                "search_markdown": search_markdown_helper,
                "extract_markdown_records": extract_markdown_records_helper,
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
