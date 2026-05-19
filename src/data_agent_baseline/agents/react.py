from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    CODEACT_REACT_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    prompt_history_steps: int = 5
    full_observation_threshold: int = 6000
    observation_head_chars: int = 1800
    observation_tail_chars: int = 1800
    stderr_tail_chars: int = 4000
    final_marker_chars: int = 4000


@dataclass(frozen=True, slots=True)
class CandidateAnswerDecision:
    answer: AnswerTable
    auto_submit: bool
    store_as_fallback: bool
    reasons: tuple[str, ...]


FINAL_RESULT_MARKERS = (
    "FINAL_TABLE_JSON:",
    "FINAL_RESULT:",
    "ANSWER_CANDIDATE:",
)

PLANNING_DIFFICULTIES = {"medium", "hard", "extreme"}
LINKING_REQUIREMENT_DIFFICULTIES = {"hard", "extreme"}

PLANNING_NODES = (
    (
        "schema_linking",
        (),
        "Understand structured data first: inspect files/tables, schema, join "
        "keys, and available columns; use markdown retrieval only for coded "
        "meanings, rules, thresholds, or ambiguous terms. If a requested concept "
        "is absent from the schema, find the real column/table/record or markdown "
        "rule that implements it before computing.",
    ),
    (
        "target_columns",
        ("schema_linking",),
        "Decide the exact requested final columns and exclude helper, filter, "
        "sort, join, or explanation columns.",
    ),
    (
        "semantic_mapping",
        ("schema_linking", "target_columns"),
        "Map question terms, filters, coded values, units, and quoted values "
        "to real columns/values before heavy computation.",
    ),
    (
        "load_preprocess",
        ("semantic_mapping",),
        "Load only needed data and normalize dates, numbers, IDs, text, nulls, "
        "and coded values.",
    ),
    (
        "join_filter",
        ("load_preprocess",),
        "Apply verified joins and filters using the mapped columns and values.",
    ),
    (
        "aggregate_select",
        ("join_filter",),
        "Compute the requested rows, count, percentage, average, max/min, "
        "ranking, or ratio.",
    ),
    (
        "final_verify",
        ("aggregate_select",),
        "Verify ties/all rows/distinctness/units and prepare exactly the final "
        "requested columns.",
    ),
    (
        "answer_submission",
        ("final_verify",),
        "Submit the answer now with exactly the requested columns and rows. "
        "Do not run additional exploration after the final result is supported.",
    ),
)

MEDIUM_PLANNING_NODES = (
    (
        "schema_linking",
        (),
        "Confirm the actual files/tables and available columns only as much as needed; "
        "do not assume concepts that are not backed by a real data source.",
    ),
    (
        "target_columns",
        ("schema_linking",),
        "Decide the exact requested final answer columns and exclude helper, filter, "
        "sort, join, or explanation columns.",
    ),
    (
        "answer_submission",
        ("target_columns",),
        "Once the result is computed, submit Answer with only the requested columns. "
        "Do not add helper columns.",
    ),
)

MEDIUM_SEMANTIC_GUARD_LINES = (
    "BIRD medium semantic guard:",
    "- Final columns: infer the requested output columns from the question; remove helper IDs, dates, join keys, and filter columns unless explicitly requested.",
    "- Row grain: decide what one final row represents before filtering or grouping, such as event type, customer, transaction, station country, or monthly consumption.",
    "- Formula/unit: for average monthly/yearly, ratio, percentage, proportion, per unit, or total wording, verify the denominator and unit from schema, samples, or knowledge before finalizing.",
    "- Value/date scope: before returning an empty table, check every plausible date/value source across CSV, JSON wrappers, and SQLite tables.",
    "- Ambiguous columns: when words like type/category/status/date/id can map to multiple columns, inspect actual values and choose the column whose values answer the question.",
)

LINKING_TRIGGER_TERMS = {
    "abnormal",
    "classification",
    "code",
    "coded",
    "diagnosis",
    "disease",
    "legal",
    "meaning",
    "normal",
    "range",
    "rule",
    "severe",
    "status",
    "threshold",
    "warning",
}

RESULT_SIGNAL_TERMS = (
    "final:",
    "final answer",
    "final table",
    "final_table_json",
    "answer_candidate",
    "percentage",
    "ratio",
    "total ",
    "total:",
    "count ",
    "count:",
    "row_count",
    "rows:",
    "result",
    "computed",
)

RATIO_QUESTION_TERMS = (
    "how many times",
    "how much faster",
    "percentage",
    "percent",
    "ratio",
    "proportion",
)


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def _parse_json_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


def _extract_labeled_text(raw_response: str, label: str) -> str | None:
    stop_labels = "Thought|Reflexion|Action|Code|Output|Answer|Final Answer|Observation"
    pattern = rf"(?:^|\n)\s*{label}\s*:\s*(.*?)(?=\n\s*(?:{stop_labels})\s*:|\Z)"
    match = re.search(pattern, raw_response, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _extract_thought(raw_response: str) -> str:
    thought = _extract_labeled_text(raw_response, "Thought")
    if thought is not None:
        return thought
    reflexion = _extract_labeled_text(raw_response, "Reflexion")
    if reflexion is not None:
        return reflexion
    return "Using CodeAct fallback parser."


def _load_first_json_object(text: str) -> dict[str, object]:
    json_fence = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = json_fence.group(1).strip() if json_fence is not None else text.strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found.")


def _extract_answer_step(raw_response: str) -> ModelStep | None:
    answer_text = _extract_labeled_text(raw_response, "Final Answer")
    if answer_text is None:
        answer_text = _extract_labeled_text(raw_response, "Answer")
    if answer_text is None:
        return None

    payload = _load_first_json_object(answer_text)
    if payload.get("action") == "answer" and isinstance(payload.get("action_input"), dict):
        action_input = payload["action_input"]
    else:
        action_input = payload

    return ModelStep(
        thought=_extract_thought(raw_response),
        action="answer",
        action_input=action_input,
        raw_response=raw_response,
    )


def _extract_code_block(raw_response: str) -> str | None:
    code_after_label = re.search(
        r"(?:^|\n)\s*Code\s*:\s*```(?:python|py)?\s*(.*?)\s*```",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if code_after_label is not None:
        return code_after_label.group(1).strip()

    for match in re.finditer(
        r"```(?P<label>[a-zA-Z0-9_+-]*)\s*(?P<body>.*?)\s*```",
        raw_response,
        flags=re.DOTALL,
    ):
        label = match.group("label").strip().lower()
        if label in {"python", "py"}:
            return match.group("body").strip()

    code_text = _extract_labeled_text(raw_response, "Code")
    if code_text is not None:
        return code_text
    return None


def _parse_codeact_model_step(raw_response: str) -> ModelStep:
    code = _extract_code_block(raw_response)
    if code:
        return ModelStep(
            thought=_extract_thought(raw_response),
            action="execute_python",
            action_input={"code": code},
            raw_response=raw_response,
        )

    answer_step = _extract_answer_step(raw_response)
    if answer_step is not None:
        return answer_step

    raise ValueError("Model response is neither a JSON action nor a CodeAct code/answer block.")


def parse_model_step(raw_response: str) -> ModelStep:
    try:
        return _parse_json_model_step(raw_response)
    except Exception as json_exc:
        try:
            return _parse_codeact_model_step(raw_response)
        except Exception as codeact_exc:
            raise ValueError(
                "Could not parse model response. "
                f"JSON parser error: {json_exc}; CodeAct parser error: {codeact_exc}"
            ) from json_exc


def _safe_json_dumps(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _split_identifier_tokens(text: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", spaced)}
    expanded: set[str] = set()
    for token in tokens:
        if not token:
            continue
        expanded.add(token)
        if len(token) > 3 and token.endswith("s"):
            expanded.add(token[:-1])
    return expanded


def _question_requests_multiple_metrics(question: str) -> bool:
    q = question.lower()
    metric_terms = ("average", "avg", "mean", "sum", "total", "count", "percentage", "percent", "ratio")
    metric_mentions = re.findall(r"\b(?:average|avg|mean|sum|total|count|percentage|percent|ratio)\b", q)
    if len(metric_mentions) >= 2 and re.search(r"\b(?:and|,)\b", q):
        return True
    if re.search(
        r"\b(?:average|avg|mean|sum|total|count)\s+of\b.+\band\b.+\b(?:average|avg|mean|sum|total|count)\b",
        q,
    ):
        return True
    if sum(1 for term in metric_terms if term in q) >= 2 and re.search(r"\b(?:and|,)\b", q):
        return True
    return False


def _question_is_single_value(question: str) -> bool:
    q = question.lower()
    tokens = _split_identifier_tokens(question)
    if _question_requests_multiple_metrics(question):
        return False
    if any(phrase in q for phrase in ("how many", "how much", "what percentage", "calculate the percentage")):
        return True
    if any(
        phrase in q
        for phrase in (
            "list",
            "identify",
            "name the",
            "names and",
            "what are",
            "which ",
            "give their",
            "type of",
        )
    ):
        return False
    return bool(
        tokens
        & {
            "count",
            "average",
            "avg",
            "mean",
            "percentage",
            "percent",
            "ratio",
            "proportion",
            "total",
            "sum",
        }
    ) and "list" not in tokens


def _question_expects_nonempty_rows(question: str) -> bool:
    q = question.lower()
    if _question_is_single_value(question):
        return False
    return any(
        phrase in q
        for phrase in (
            "list",
            "list all",
            "which ",
            "what are",
            "what is the",
            "what's the",
            "state ",
            "identify",
            "name ",
            "give ",
        )
    )


def _normalize_answer_cell(value: object) -> object:
    if not isinstance(value, str):
        return value

    text = value.strip().replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return text

    percent_match = re.fullmatch(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*%", text)
    if percent_match is not None:
        return percent_match.group(1).replace(",", "")

    numeric_with_commas = re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text)
    if numeric_with_commas is not None:
        return text.replace(",", "")

    currency_match = re.fullmatch(r"[$]\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if currency_match is not None:
        return currency_match.group(1).replace(",", "")

    return text


def _normalize_answer_table(answer: AnswerTable) -> AnswerTable:
    return AnswerTable(
        columns=[str(column).strip() for column in answer.columns],
        rows=[[_normalize_answer_cell(cell) for cell in row] for row in answer.rows],
    )


def _project_answer_columns(answer: AnswerTable, keep_indexes: list[int]) -> AnswerTable:
    return AnswerTable(
        columns=[answer.columns[index] for index in keep_indexes],
        rows=[[row[index] for index in keep_indexes] for row in answer.rows],
    )


def _matching_column_indexes(answer: AnswerTable, token_groups: list[set[str]]) -> list[int] | None:
    if not token_groups:
        return None

    column_tokens = [_split_identifier_tokens(column) for column in answer.columns]
    keep: list[int] = []
    for group in token_groups:
        matches = [index for index, tokens in enumerate(column_tokens) if tokens & group]
        if not matches:
            return None
        keep.extend(matches)

    deduped = sorted(set(keep))
    if not deduped or len(deduped) >= len(answer.columns):
        return None
    return deduped


def _explicit_requested_column_indexes(task: PublicTask, answer: AnswerTable) -> list[int] | None:
    question = task.question.casefold()
    groups: list[set[str]] = []

    if re.search(r"\bids?\b|identifier", question):
        groups.append({"id"})
    if re.search(r"\bsex\b|\bgender\b", question):
        groups.append({"sex", "gender"})
    if re.search(r"\bdisease\b|diagnos", question):
        groups.append({"disease", "diagnosis"})
    if "type of expense" in question or "expense type" in question:
        groups.append({"type", "category"})
    if "total value" in question or "total cost" in question or "total amount" in question:
        groups.append({"total", "value", "cost", "amount", "sum"})

    if len(groups) < 2:
        return None
    return _matching_column_indexes(answer, groups)


def _single_target_column_indexes(task: PublicTask, answer: AnswerTable) -> list[int] | None:
    if len(answer.columns) <= 1 or _question_is_single_value(task.question):
        return None

    question = task.question.casefold()
    question_tokens = _split_identifier_tokens(task.question)
    column_tokens = [_split_identifier_tokens(column) for column in answer.columns]
    metric_tokens = {
        "amount",
        "budget",
        "consumption",
        "cost",
        "count",
        "date",
        "expense",
        "price",
        "score",
        "time",
        "total",
        "value",
        "view",
        "viewcount",
        "views",
    }
    multi_output_cues = (
        " and include ",
        " include ",
        " with their ",
        " with its ",
        " give their ",
        " along with ",
    )
    has_explicit_metric_request = bool(question_tokens & metric_tokens)
    has_metric_column = any(tokens & metric_tokens for tokens in column_tokens)
    has_multi_output_cue = any(cue in question for cue in multi_output_cues)

    target_groups: list[set[str]] = []
    if re.search(r"\bwhat(?:'s| is) the comment\b|\bcomment with\b", question):
        target_groups.append({"comment", "text", "body", "content"})
    elif re.search(r"\b(names?|name) of\b|\bname the\b|\bwhat are the names\b", question):
        if has_multi_output_cue or (has_explicit_metric_request and has_metric_column):
            return None
        target_groups.append({"name"})
    elif re.search(r"\bwhich event\b|\bwhat event\b", question):
        if has_multi_output_cue:
            return None
        target_groups.append({"event", "name", "title"})
    elif re.search(r"\bwhich race\b|\bwhat race\b", question):
        if has_multi_output_cue:
            return None
        target_groups.append({"race", "name", "title"})
    elif re.search(r"\bstate the date\b|\bwhat date\b|\bwhich date\b", question):
        target_groups.append({"date"})
    elif "finish time" in question:
        target_groups.append({"time"})
    elif re.search(r"\bnumber of the driver\b|\bdriver number\b", question):
        target_groups.append({"number"})
    elif re.search(r"\bcountries\b|\bcountry\b", question):
        target_groups.append({"country"})
    else:
        return None

    if target_groups[0] == {"name"}:
        name_indexes = [
            index for index, tokens in enumerate(column_tokens) if tokens & target_groups[0]
        ]
        if not name_indexes:
            return None
        subject_matches = [
            index
            for index in name_indexes
            if column_tokens[index] & (question_tokens - {"name", "names"})
        ]
        if subject_matches:
            name_indexes = subject_matches
        non_full_name = [index for index in name_indexes if "full" not in column_tokens[index]]
        if non_full_name:
            name_indexes = non_full_name
        keep = sorted(set(name_indexes))
    else:
        keep = _matching_column_indexes(answer, target_groups)
        if keep is None:
            return None

    if len(keep) >= len(answer.columns):
        return None
    return keep


def _remove_high_confidence_extra_columns(task: PublicTask, answer: AnswerTable) -> AnswerTable:
    explicit_indexes = _explicit_requested_column_indexes(task, answer)
    if explicit_indexes is not None:
        return _project_answer_columns(answer, explicit_indexes)

    single_target_indexes = _single_target_column_indexes(task, answer)
    if single_target_indexes is not None:
        return _project_answer_columns(answer, single_target_indexes)

    return answer


def _question_has_multi_column_output_cue(question: str) -> bool:
    q = f" {question.casefold()} "
    return any(
        cue in q
        for cue in (
            " and include ",
            " along with ",
            " include ",
            " with their ",
            " with its ",
            " give their ",
            " give its ",
            " please give ",
            " list their ",
        )
    )


def _task_is_easy(task: PublicTask) -> bool:
    return task.difficulty.casefold() == "easy"


def _task_is_medium(task: PublicTask) -> bool:
    return task.difficulty.casefold() == "medium"


def _easy_record_id_column_indexes(task: PublicTask, answer: AnswerTable) -> list[int] | None:
    if not _task_is_easy(task) or len(answer.columns) <= 1:
        return None
    if _question_is_single_value(task.question):
        return None
    if _question_has_multi_column_output_cue(task.question):
        return None

    question = task.question.casefold()
    column_tokens = [_split_identifier_tokens(column) for column in answer.columns]
    entity_groups: list[set[str]] = []
    if re.search(r"\btransactions?\b|\bwithdrawals?\b", question):
        entity_groups.append({"trans", "transaction"})

    if not entity_groups:
        return None

    matches: list[int] = []
    for index, tokens in enumerate(column_tokens):
        if "id" not in tokens:
            continue
        if any(tokens & group for group in entity_groups):
            matches.append(index)

    if len(matches) != 1:
        return None
    return matches


def _apply_easy_deterministic_projection(task: PublicTask, answer: AnswerTable) -> AnswerTable:
    keep_indexes = _easy_record_id_column_indexes(task, answer)
    if keep_indexes is None:
        return answer
    return _project_answer_columns(answer, keep_indexes)


def _is_missing_answer_cell(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    if isinstance(value, str):
        normalized = value.strip().casefold()
        return normalized in {"", "nan", "none", "null", "nat", "<na>"}
    return False


def _drop_incomplete_answer_rows(task: PublicTask, answer: AnswerTable) -> AnswerTable:
    if not _question_expects_nonempty_rows(task.question):
        return answer
    if not answer.rows:
        return answer

    kept_rows: list[list[object]] = []
    for row in answer.rows:
        if len(row) != len(answer.columns):
            continue
        missing_count = sum(1 for cell in row if _is_missing_answer_cell(cell))
        # A single missing requested value can be the correct answer. Multiple
        # missing cells in one row are a stronger signal of a failed join.
        if missing_count >= 2:
            continue
        kept_rows.append(row)

    if not kept_rows or len(kept_rows) == len(answer.rows):
        return answer
    return AnswerTable(columns=list(answer.columns), rows=kept_rows)


def _sanitize_answer_table(task: PublicTask, answer: AnswerTable) -> AnswerTable:
    normalized = _normalize_answer_table(answer)
    complete_rows = _drop_incomplete_answer_rows(task, normalized)
    projected = _remove_high_confidence_extra_columns(task, complete_rows)
    return _apply_easy_deterministic_projection(task, projected)


def _answer_projection_preview(task: PublicTask, answer: AnswerTable) -> dict[str, object] | None:
    normalized = _normalize_answer_table(answer)
    projected = _remove_high_confidence_extra_columns(task, normalized)
    if projected.columns == normalized.columns:
        return None
    return {
        "original_columns": list(normalized.columns),
        "projected_columns": list(projected.columns),
        "reason": "high-confidence final-column projection",
    }



def _answer_table_from_payload(payload: dict[str, object]) -> AnswerTable | None:
    if payload.get("action") == "answer" and isinstance(payload.get("action_input"), dict):
        payload = payload["action_input"]
    columns = payload.get("columns")
    rows = payload.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(column, str) for column in columns):
        return None
    if not isinstance(rows, list):
        return None
    normalized_rows: list[list[object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            return None
        normalized_rows.append(list(row))
    return AnswerTable(columns=list(columns), rows=normalized_rows)


def _answer_preview(answer: AnswerTable, *, max_rows: int = 3) -> dict[str, object]:
    return {
        "columns": list(answer.columns),
        "row_count": len(answer.rows),
        "sample_rows": [list(row) for row in answer.rows[:max_rows]],
    }


def _python_repair_hints(content: dict[str, object], code: str) -> list[str]:
    text = "\n".join(
        str(content.get(key) or "")
        for key in ("error", "traceback", "stderr", "output", "stdout")
    )
    lowered = text.casefold()
    hints: list[str] = []

    def add(hint: str) -> None:
        if hint not in hints:
            hints.append(hint)

    if "cannot operate on a closed database" in lowered:
        add(
            "SQLite connection was closed before all queries/fetches finished; reopen the connection "
            "or move every cursor/read_sql operation before conn.close()."
        )
    if (
        "no module named 'tools'" in lowered
        or "no module named 'knowledge'" in lowered
        or "no module named 'markdown_helpers'" in lowered
        or "no module named 'utils'" in lowered
    ):
        add(
            "Helper functions are injected into Python. Call retrieve_knowledge(...), search_markdown(...), "
            "extract_markdown_records(...), markdown_entity_table(...), link_question_to_data(...), load_json_records(...), "
            "or load_json_table(...) directly."
        )
    if "trying to merge on" in lowered and ("int64" in lowered or "object" in lowered or "str" in lowered):
        add(
            "Join keys have mismatched dtypes; print key samples, then cast both sides to comparable "
            "strings, preserving leading zeros for IDs/codes."
        )
    if "no such table" in lowered:
        add("SQLite table name is wrong; inspect sqlite_master or use the exact table names from the manifest.")
    if "no such column" in lowered:
        add("Column name is wrong; inspect PRAGMA table_info or dataframe columns and use exact names.")
    if "keyerror" in lowered:
        add("Pandas KeyError usually means an exact column mismatch; print df.columns.tolist() before selecting.")
    if (
        re.search(r"columns?:\s*\[?['\"]table['\"]\s*,\s*['\"]records['\"]", lowered)
        or re.search(r"\btable\s+records\b", lowered)
        or ("read_json" in code.casefold() and {"table", "records"} <= set(re.findall(r"[a-z_]+", lowered)))
    ):
        add(
            "JSON appears to be a wrapper with `table` and `records`; reload it with "
            "`load_json_table('json/file.json')` or `load_json_records('json/file.json')` instead of pd.read_json."
        )
    if "file not found" in lowered or "no such file or directory" in lowered:
        add("Use paths relative to the task context directory exactly as shown in the manifest.")
    if "timed out" in lowered:
        add("Reduce work and output: filter first, avoid printing full files/dataframes, and inspect compact samples.")
    if re.search(r"\bempty dataframe\b|\b0 rows\b|shape:\s*\(0[,)]", lowered):
        add("Result is empty; verify filters, join keys, casing, and whether the entity should be matched in another table/document.")
    if "final_table_json" in code.casefold() and not bool(content.get("success")):
        add("Do not print FINAL_TABLE_JSON until the code has computed the exact final answer without errors.")

    return hints[:4]


def _likely_question_type(question: str) -> str:
    if _question_is_single_value(question):
        return "single_value"
    if _question_expects_nonempty_rows(question):
        return "row_list"
    return "table"


def _final_answer_check(task: PublicTask, answer: AnswerTable) -> dict[str, object]:
    question_type = _likely_question_type(task.question)
    question_tokens = _split_identifier_tokens(task.question)
    warnings: list[str] = []
    notes: list[str] = []

    if question_type == "single_value":
        if len(answer.columns) != 1 or len(answer.rows) != 1:
            warnings.append("single-value question should usually submit exactly one column and one row")
        else:
            notes.append("single-value shape looks plausible")
    elif question_type == "row_list":
        if not answer.rows:
            warnings.append("question appears to expect matching rows, but candidate answer is empty")
        else:
            notes.append("row-list answer is non-empty")

    helper_terms = {
        "debug",
        "helper",
        "index",
        "rank",
        "row",
        "score",
        "avg",
        "average",
        "sum",
        "total",
        "count",
        "ratio",
        "percentage",
        "percent",
        "id",
        "ids",
        "code",
        "key",
    }
    possible_helper_columns: list[str] = []
    for column in answer.columns:
        column_tokens = _split_identifier_tokens(column)
        helper_overlap = column_tokens & helper_terms
        if helper_overlap and not helper_overlap <= question_tokens:
            possible_helper_columns.append(column)
    if possible_helper_columns:
        warnings.append(
            "possible helper/extra columns not clearly requested: "
            + ", ".join(possible_helper_columns[:5])
        )

    if any(token in question_tokens for token in {"percentage", "percent", "ratio", "average", "avg", "mean"}):
        if len(answer.rows) == 1 and len(answer.columns) == 1:
            value = answer.rows[0][0] if answer.rows and answer.rows[0] else None
            if isinstance(value, str):
                normalized = value.strip().rstrip("%").replace(",", "")
            else:
                normalized = str(value)
            try:
                float(normalized)
            except (TypeError, ValueError):
                warnings.append("numeric aggregate question produced a non-numeric-looking value")

    return {
        "question_type": question_type,
        "columns": list(answer.columns),
        "row_count": len(answer.rows),
        "warnings": warnings,
        "notes": notes,
    }


def _build_difficulty_final_answer_reminder(task: PublicTask, state: AgentRuntimeState) -> str | None:
    if not state.steps:
        return None

    difficulty = task.difficulty.casefold()
    if difficulty != "easy":
        return None
    return "\n".join(
        [
            "FINAL ANSWER REMINDER (easy):",
            f"- Question: {task.question}",
            "- Submit only the columns directly requested by the question.",
            "- If multiple rows match the entity/filter, include all final rows; do not keep only the first row.",
            "- Remove helper/debug/join-key/source columns unless the question asks for them.",
            "- For similar fields such as rank/position, number/id, or date/datetime, verify the field meaning in the data before submitting.",
            "- For easy race-driver questions, treat ranked/rank as a `rank` field before `position`; treat driver number as the driver's official `number`, not driverId.",
            "- For time-like values, compare normalized times too: 0:01:54 can correspond to 1:54.xxx values.",
        ]
    )


def _find_marker_payload(text: str, markers: tuple[str, ...]) -> dict[str, object] | None:
    lower_text = text.lower()
    marker_positions = [
        lower_text.find(marker.lower())
        for marker in markers
        if lower_text.find(marker.lower()) >= 0
    ]
    if not marker_positions:
        return None
    try:
        return _load_first_json_object(text[min(marker_positions) :])
    except ValueError:
        return None


def _candidate_decision_from_observation(
    task: PublicTask,
    observation: dict[str, object],
    answer: AnswerTable,
) -> CandidateAnswerDecision:
    reasons: list[str] = []
    hard_reasons: list[str] = []
    content = observation.get("content")
    content_dict = content if isinstance(content, dict) else {}
    stderr = str(content_dict.get("stderr") or "")
    traceback_text = str(content_dict.get("traceback") or "")

    if traceback_text or "traceback" in stderr.lower():
        reasons.append("python_traceback_present")
        hard_reasons.append("python_traceback_present")

    if _question_expects_nonempty_rows(task.question) and not answer.rows:
        reasons.append("question_expects_rows_but_candidate_is_empty")
        hard_reasons.append("question_expects_rows_but_candidate_is_empty")

    if _question_is_single_value(task.question) and (len(answer.rows) != 1 or len(answer.columns) != 1):
        reasons.append("single_value_question_requires_one_cell_answer")
        hard_reasons.append("single_value_question_requires_one_cell_answer")

    store_as_fallback = bool(answer.rows) and "python_traceback_present" not in reasons
    return CandidateAnswerDecision(
        answer=answer,
        auto_submit=not hard_reasons,
        store_as_fallback=store_as_fallback,
        reasons=tuple(reasons),
    )


def _find_final_marker_segment(text: str, max_chars: int) -> str | None:
    if max_chars <= 0:
        return None
    lower_text = text.lower()
    best_index: int | None = None
    for marker in FINAL_RESULT_MARKERS:
        index = lower_text.find(marker.lower())
        if index >= 0 and (best_index is None or index < best_index):
            best_index = index
    if best_index is None:
        return None
    segment = text[best_index : best_index + max_chars]
    omitted = len(text) - (best_index + len(segment))
    if omitted > 0:
        segment = f"{segment}\n...[final marker segment truncated {omitted} chars]"
    return segment


def _compact_text_block(
    label: str,
    text: str,
    *,
    head_chars: int,
    tail_chars: int,
    final_marker_chars: int,
) -> list[str]:
    if not text:
        return [f"{label}: <empty>"]

    if len(text) <= head_chars + tail_chars:
        return [f"{label}_chars: {len(text)}", f"{label}:\n{text}"]

    head = text[: max(0, head_chars)]
    tail = text[-max(0, tail_chars) :] if tail_chars > 0 else ""
    omitted = max(0, len(text) - len(head) - len(tail))
    lines = [f"{label}_chars: {len(text)}"]
    if head:
        lines.append(f"{label}_head:\n{head}")
    marker_segment = _find_final_marker_segment(text, final_marker_chars)
    if marker_segment is not None:
        lines.append(f"{label}_final_marker_segment:\n{marker_segment}")
    if tail:
        lines.append(f"{label}_tail:\n{tail}")
    lines.append(f"{label}_compact_note: omitted {omitted} middle chars")
    return lines


def _compact_generic_value(key: str, value: object, config: ReActAgentConfig) -> list[str]:
    if isinstance(value, str):
        return _compact_text_block(
            key,
            value,
            head_chars=config.observation_head_chars,
            tail_chars=config.observation_tail_chars,
            final_marker_chars=config.final_marker_chars,
        )

    rendered = _safe_json_dumps(value)
    return _compact_text_block(
        key,
        rendered,
        head_chars=config.observation_head_chars,
        tail_chars=config.observation_tail_chars,
        final_marker_chars=config.final_marker_chars,
    )


def _render_compact_python_observation(
    observation: dict[str, object],
    config: ReActAgentConfig,
    original_chars: int,
) -> str:
    content = observation.get("content")
    if not isinstance(content, dict):
        return _render_compact_generic_observation(observation, config, original_chars)

    lines = [
        "compact_observation: true",
        f"original_observation_chars: {original_chars}",
        f"ok: {observation.get('ok')}",
        f"tool: {observation.get('tool')}",
    ]
    for key, value in content.items():
        if key in {"output", "stdout", "stderr", "traceback"}:
            continue
        lines.append(f"{key}: {_safe_json_dumps(value) if isinstance(value, (dict, list)) else value}")

    stdout = str(content.get("output") or content.get("stdout") or "")
    stderr = str(content.get("stderr") or "")
    traceback_text = str(content.get("traceback") or "")

    if stdout:
        lines.extend(
            _compact_text_block(
                "stdout",
                stdout,
                head_chars=config.observation_head_chars,
                tail_chars=config.observation_tail_chars,
                final_marker_chars=config.final_marker_chars,
            )
        )
    if stderr:
        lines.extend(
            _compact_text_block(
                "stderr",
                stderr,
                head_chars=0,
                tail_chars=config.stderr_tail_chars,
                final_marker_chars=0,
            )
        )
    if traceback_text:
        lines.extend(
            _compact_text_block(
                "traceback",
                traceback_text,
                head_chars=0,
                tail_chars=config.stderr_tail_chars,
                final_marker_chars=0,
            )
        )
    return "\n\n".join(lines)


def _render_compact_generic_observation(
    observation: dict[str, object],
    config: ReActAgentConfig,
    original_chars: int,
) -> str:
    lines = [
        "compact_observation: true",
        f"original_observation_chars: {original_chars}",
        f"ok: {observation.get('ok')}",
        f"tool: {observation.get('tool', '<none>')}",
    ]
    for key, value in observation.items():
        if key in {"ok", "tool"}:
            continue
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for child_key, child_value in value.items():
                rendered_lines = _compact_generic_value(str(child_key), child_value, config)
                lines.append("\n".join(rendered_lines))
        else:
            lines.extend(_compact_generic_value(str(key), value, config))
    return "\n\n".join(lines)


def _render_observation_for_prompt(
    observation: dict[str, object],
    config: ReActAgentConfig,
) -> str:
    rendered = _safe_json_dumps(observation)
    if len(rendered) <= config.full_observation_threshold:
        return rendered
    if observation.get("tool") == "execute_python":
        return _render_compact_python_observation(observation, config, len(rendered))
    return _render_compact_generic_observation(observation, config, len(rendered))


def _task_uses_planning(task: PublicTask) -> bool:
    return task.difficulty.casefold() in PLANNING_DIFFICULTIES


def _task_uses_advanced_planning(task: PublicTask) -> bool:
    return task.difficulty.casefold() in {"hard", "extreme"}


def _task_has_markdown(task: PublicTask) -> bool:
    try:
        if any(task.context_dir.glob("*.md")):
            return True
        for child in task.context_dir.iterdir():
            if child.is_dir() and any(child.glob("*.md")):
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _question_has_linking_cues(question: str) -> bool:
    if re.search(r'"[^"]+"|\'[^\']+\'', question):
        return True
    if _split_identifier_tokens(question) & LINKING_TRIGGER_TERMS:
        return True
    capitalized_phrases = re.findall(
        r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)+\b",
        question,
    )
    return any(len(phrase) >= 6 for phrase in capitalized_phrases)


def _step_called_linking_helper(step: StepRecord) -> bool:
    action_input = _safe_json_dumps(step.action_input).lower()
    raw_response = step.raw_response.lower()
    return "link_question_to_data" in action_input or "link_question_to_data" in raw_response


def _has_called_linking_helper(state: AgentRuntimeState) -> bool:
    return any(_step_called_linking_helper(step) for step in state.steps)


def _should_require_link_question_to_data(task: PublicTask, state: AgentRuntimeState) -> bool:
    if task.difficulty.casefold() not in LINKING_REQUIREMENT_DIFFICULTIES:
        return False
    if _has_called_linking_helper(state):
        return False
    if not _task_has_markdown(task):
        return False
    if not _question_has_linking_cues(task.question):
        return False
    successful_python_steps = sum(1 for step in state.steps if step.action == "execute_python" and step.ok)
    return successful_python_steps < 2


def _step_text_for_planning(step: StepRecord) -> str:
    parts = [step.action, step.thought, step.raw_response[:1200]]
    content = step.observation.get("content")
    if isinstance(content, dict):
        for key in ("output", "stdout", "stderr", "traceback"):
            value = content.get(key)
            if value:
                parts.append(str(value)[:1200])
    else:
        error = step.observation.get("error")
        if error:
            parts.append(str(error)[:600])
    return "\n".join(parts).lower()


def _successful_python_steps(state: AgentRuntimeState) -> list[StepRecord]:
    return [step for step in state.steps if step.action == "execute_python" and step.ok]


def _normalize_for_fingerprint(text: str) -> str:
    normalized = re.sub(r"#.*", "", text)
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized[:4000]


def _fingerprint_text(text: str) -> str:
    return hashlib.md5(_normalize_for_fingerprint(text).encode("utf-8")).hexdigest()[:10]


def _step_code_text(step: StepRecord) -> str:
    if isinstance(step.action_input, dict):
        return str(step.action_input.get("code") or "")
    return ""


def _step_output_text(step: StepRecord) -> str:
    content = step.observation.get("content")
    if isinstance(content, dict):
        return "\n".join(
            str(content.get(key) or "")
            for key in ("output", "stdout", "stderr", "traceback")
            if content.get(key)
        )
    error = step.observation.get("error")
    return str(error or "")


def _loop_guard_summary(state: AgentRuntimeState) -> dict[str, object]:
    python_steps = _successful_python_steps(state)
    if len(python_steps) < 2:
        return {"triggered": False}

    last_step = python_steps[-1]
    last_code = _step_code_text(last_step)
    last_output = _step_output_text(last_step)
    code_fingerprint = _fingerprint_text(last_code) if last_code else ""
    output_fingerprint = _fingerprint_text(last_output) if last_output else ""

    code_repeat_count = (
        sum(1 for step in python_steps if _fingerprint_text(_step_code_text(step)) == code_fingerprint)
        if code_fingerprint
        else 0
    )
    output_repeat_count = (
        sum(1 for step in python_steps if _fingerprint_text(_step_output_text(step)) == output_fingerprint)
        if output_fingerprint
        else 0
    )

    repeated_schema_search = False
    recent_text = "\n".join(_step_text_for_planning(step) for step in python_steps[-4:])
    if len(python_steps) >= 4:
        repeated_schema_search = any(
            recent_text.count(phrase) >= 2
            for phrase in (
                "no explicit format",
                "no tables",
                "potential legality tables: []",
                "format/legality/commander columns: []",
            )
        ) or any(
            len(re.findall(pattern, recent_text)) >= 2
            for pattern in (
                r"searching for [a-z0-9_\-\s]{0,40}normal range definition",
                r"searching for [a-z0-9_\-\s]{0,40}threshold",
            )
        )

    triggered = code_repeat_count >= 2 or output_repeat_count >= 2 or repeated_schema_search
    summary: dict[str, object] = {
        "triggered": triggered,
        "last_step_index": last_step.step_index,
        "code_repeat_count": code_repeat_count,
        "output_repeat_count": output_repeat_count,
        "repeated_schema_search": repeated_schema_search,
    }
    if triggered:
        summary["guidance"] = (
            "Do not rerun the same code, schema inspection, or search. Change tactic: "
            "use a different source/parser, convert markdown prose into records, or make "
            "one final best-supported computation if the blocker has already been observed."
        )
    return summary


def _evidence_requirement_status(all_text: str, keywords: tuple[str, ...]) -> str:
    return "maybe_found" if any(keyword in all_text for keyword in keywords) else "needs_evidence"


def _task_evidence_ledger(task: PublicTask, state: AgentRuntimeState) -> list[dict[str, object]]:
    question = task.question.casefold()
    tokens = _split_identifier_tokens(task.question)
    all_text = "\n".join(_step_text_for_planning(step) for step in state.steps)
    requirements: list[tuple[str, str, tuple[str, ...]]] = [
        (
            "concrete_data_source",
            "Identify the actual file/table/markdown record source for every requested concept; do not rely on schema-only assumptions.",
            ("columns", "tables", "files in directory", "link result", "json/", "csv/", "db/", "doc/"),
        ),
        (
            "final_answer_shape",
            "Know the exact final columns and whether the answer is one value, rows, count, percentage, or ratio.",
            ("final_table_json", "answer_candidate", "final answer", "columns", "row_count"),
        ),
    ]
    if tokens & {"percentage", "percent", "ratio", "proportion"} or "how many times" in question:
        requirements.append(
            (
                "numerator_denominator",
                "Define the numerator and denominator from grounded records before computing the percentage/ratio.",
                ("numerator", "denominator", "percentage", "ratio", "total count", "count of"),
            )
        )
    if "between" in tokens or re.search(r"\b\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\b", question):
        requirements.append(
            (
                "numeric_boundaries",
                "Map numeric bounds to a real column and apply inclusive/exclusive semantics from the question.",
                ("between", ">=", "<=", "numeric_filters", "height", "age", "boundary"),
            )
        )
    if tokens & {"abnormal", "normal", "threshold", "range", "severe", "legal", "status"}:
        requirements.append(
            (
                "rule_or_threshold",
                "Find the concrete rule, coded value, threshold, or status definition in data/markdown before filtering.",
                ("threshold", "normal range", "upper limit", "abnormal", "legal status", "coded", "severity"),
            )
        )
    if tokens & {"age", "under", "yet"} or "70" in question:
        requirements.append(
            (
                "age_reference",
                "Ground age logic in patient birth/date records and the question's reference date, not a guessed current date.",
                ("birthday", "birth", "age", "70", "years old", "not yet"),
            )
        )
    if tokens & {"format", "commander", "legal", "status"}:
        requirements.append(
            (
                "format_status_source",
                "If format/status columns are absent from SQLite/CSV, treat markdown legality records as the data source and parse card IDs/status.",
                ("legalities", "format", "commander", "cards_id", "status", "legal"),
            )
        )
    if _task_has_markdown(task):
        requirements.append(
            (
                "markdown_record_strategy",
                "For facts stored only in .md, parse record-level snippets and keep IDs/values from the same record together.",
                ("extract_markdown_records", "search_markdown", "doc/", "snippet", "record", "paragraph"),
            )
        )

    return [
        {
            "id": requirement_id,
            "status": _evidence_requirement_status(all_text, keywords),
            "instruction": instruction,
        }
        for requirement_id, instruction, keywords in requirements
    ]


def _recent_result_signal(state: AgentRuntimeState) -> bool:
    recent_text = "\n".join(_step_text_for_planning(step) for step in state.steps[-2:])
    return any(term in recent_text for term in RESULT_SIGNAL_TERMS)


def _answer_has_submission_shape(task: PublicTask, answer: AnswerTable) -> bool:
    if not answer.columns:
        return False
    if _question_expects_nonempty_rows(task.question) and not answer.rows:
        return False
    if _question_is_single_value(task.question) and (len(answer.columns) != 1 or len(answer.rows) != 1):
        return False
    return True


def _answer_column_is_semantically_suspicious(task: PublicTask, answer: AnswerTable) -> bool:
    question = task.question.casefold()
    columns = {column.casefold().strip() for column in answer.columns}
    if "how many times" in question and columns & {"count", "total", "number", "num"}:
        return True
    if any(term in question for term in RATIO_QUESTION_TERMS) and len(answer.columns) == 1:
        column = next(iter(columns))
        if column in {"count", "total", "number", "num", "n"}:
            return True
    return False


def _answer_is_submission_ready(task: PublicTask, answer: AnswerTable) -> bool:
    return (
        _answer_has_submission_shape(task, answer)
        and not _answer_column_is_semantically_suspicious(task, answer)
    )


def _medium_empty_answer_recheck_seen(state: AgentRuntimeState) -> bool:
    return any(
        isinstance(step.observation, dict)
        and bool(step.observation.get("medium_empty_answer_recheck"))
        for step in state.steps
    )


def _should_block_medium_empty_answer(
    task: PublicTask,
    state: AgentRuntimeState,
    model_step: ModelStep,
) -> bool:
    if not _task_is_medium(task) or model_step.action != "answer":
        return False
    if _medium_empty_answer_recheck_seen(state):
        return False
    if not _question_expects_nonempty_rows(task.question):
        return False
    answer = _answer_table_from_payload(model_step.action_input)
    return answer is not None and not answer.rows


def _candidate_answer_from_step(task: PublicTask, step: StepRecord) -> AnswerTable | None:
    if step.action != "execute_python" or not step.ok:
        return None
    content = step.observation.get("content")
    if not isinstance(content, dict):
        return None
    stdout = str(content.get("output") or content.get("stdout") or "")
    payload = _find_marker_payload(stdout, FINAL_RESULT_MARKERS)
    if payload is None:
        return None
    answer = _answer_table_from_payload(payload)
    if answer is None:
        return None
    return _sanitize_answer_table(task, answer)


def _latest_submission_ready_candidate(
    task: PublicTask,
    state: AgentRuntimeState,
    fallback_answer: AnswerTable | None,
) -> AnswerTable | None:
    if fallback_answer is not None and _answer_is_submission_ready(task, fallback_answer):
        return fallback_answer
    for step in reversed(state.steps):
        answer = _candidate_answer_from_step(task, step)
        if answer is not None and _answer_is_submission_ready(task, answer):
            return answer
    return None


def _should_prioritize_answer_submission(
    task: PublicTask,
    state: AgentRuntimeState,
    fallback_answer: AnswerTable | None,
    *,
    step_index: int,
    max_steps: int,
) -> bool:
    _ = step_index, max_steps
    return _latest_submission_ready_candidate(task, state, fallback_answer) is not None


def _should_issue_final_focused_attempt(
    task: PublicTask,
    state: AgentRuntimeState,
    fallback_answer: AnswerTable | None,
    *,
    step_index: int,
    max_steps: int,
) -> bool:
    if _should_prioritize_answer_submission(
        task,
        state,
        fallback_answer,
        step_index=step_index,
        max_steps=max_steps,
    ):
        return False
    if max_steps - step_index + 1 > 2:
        return False
    return any(step.action == "execute_python" and step.ok for step in state.steps)


def _planning_observed_nodes(
    state: AgentRuntimeState,
    fallback_answer: AnswerTable | None,
) -> set[str]:
    steps = list(state.steps)
    successful_python_steps = [
        step for step in steps if step.action == "execute_python" and step.ok
    ]
    recent_text = "\n".join(_step_text_for_planning(step) for step in steps[-5:])
    all_text = "\n".join(_step_text_for_planning(step) for step in steps)
    has_candidate_answer = fallback_answer is not None or any(
        marker.lower() in all_text for marker in FINAL_RESULT_MARKERS
    )
    has_terminal_answer = state.answer is not None or any(
        step.action == "answer" and step.ok for step in steps
    )

    observed: set[str] = set()
    if successful_python_steps:
        observed.add("schema_linking")
    if has_candidate_answer or re.search(r"\bcolumns?\b|final_table_json|answer_candidate", recent_text):
        observed.update({"schema_linking", "target_columns"})
    if len(successful_python_steps) >= 2 or any(
        token in recent_text
        for token in (
            "unique",
            "value_counts",
            "sample",
            "head(",
            "mapping",
            "knowledge",
            "quoted",
            "matched",
            "normal range",
            "coded",
        )
    ):
        observed.update({"schema_linking", "target_columns", "semantic_mapping"})
    if len(successful_python_steps) >= 2 or any(
        token in recent_text
        for token in (
            "read_csv",
            "read_json",
            "sqlite3",
            "astype",
            "to_datetime",
            "dropna",
            "fillna",
            "normalize",
            "str.strip",
        )
    ):
        observed.update({"schema_linking", "target_columns", "semantic_mapping", "load_preprocess"})
    if any(
        token in recent_text
        for token in (
            "merge(",
            ".join(",
            "read_sql",
            " where ",
            ".query(",
            ".isin(",
            "filtered",
            "join",
        )
    ):
        observed.update(
            {
                "schema_linking",
                "target_columns",
                "semantic_mapping",
                "load_preprocess",
                "join_filter",
            }
        )
    if has_candidate_answer or any(
        token in recent_text
        for token in (
            "groupby",
            ".agg(",
            ".sum(",
            ".mean(",
            ".count(",
            ".nunique(",
            "percentage",
            "ratio",
            "idxmax",
            "idxmin",
            "sort_values",
        )
    ):
        observed.update(
            node_id
            for node_id, _, _ in PLANNING_NODES
            if node_id != "answer_submission"
        )
    if has_terminal_answer:
        observed.update(node_id for node_id, _, _ in PLANNING_NODES)
    return observed


def _planning_status(
    state: AgentRuntimeState,
    fallback_answer: AnswerTable | None,
    nodes: tuple[tuple[str, tuple[str, ...], str], ...] = PLANNING_NODES,
) -> tuple[list[dict[str, object]], str, list[str]]:
    observed = _planning_observed_nodes(state, fallback_answer)
    node_status: dict[str, str] = {}
    node_records: list[dict[str, object]] = []
    active_node: str | None = None

    for node_id, dependencies, instruction in nodes:
        if node_id in observed:
            status = "observed"
        elif all(node_status.get(dependency) == "observed" for dependency in dependencies):
            status = "focus" if active_node is None else "later"
            if status == "focus":
                active_node = node_id
        else:
            status = "later"
        node_status[node_id] = status
        node_records.append(
            {
                "id": node_id,
                "dependent_task_ids": list(dependencies),
                "status": status,
                "instruction": instruction,
            }
        )

    current_focus = active_node or "submit_answer"
    observed_nodes = [node["id"] for node in node_records if node["status"] == "observed"]
    return node_records, current_focus, observed_nodes


def _force_planning_focus(
    node_records: list[dict[str, object]],
    node_id: str,
) -> None:
    for node in node_records:
        if node.get("status") == "focus":
            node["status"] = "later"
        if node.get("id") == node_id and node.get("status") != "observed":
            node["status"] = "focus"


def _defer_answer_submission_focus(
    node_records: list[dict[str, object]],
    *,
    fallback_focus: str,
) -> str:
    for node in node_records:
        if node.get("id") == "answer_submission" and node.get("status") == "focus":
            node["status"] = "later"
    return fallback_focus


def _build_planning_context(
    task: PublicTask,
    state: AgentRuntimeState,
    fallback_answer: AnswerTable | None,
    *,
    step_index: int,
    max_steps: int,
) -> tuple[str | None, dict[str, object] | None]:
    if not _task_uses_planning(task):
        return None, None

    difficulty = task.difficulty.casefold()
    use_advanced_planning = _task_uses_advanced_planning(task)
    evidence_ledger = _task_evidence_ledger(task, state) if use_advanced_planning else []
    loop_guard = _loop_guard_summary(state) if use_advanced_planning else {"triggered": False}
    answer_urgent = _should_prioritize_answer_submission(
        task,
        state,
        fallback_answer,
        step_index=step_index,
        max_steps=max_steps,
    )
    final_focused_attempt = _should_issue_final_focused_attempt(
        task,
        state,
        fallback_answer,
        step_index=step_index,
        max_steps=max_steps,
    )
    require_linking_helper = _should_require_link_question_to_data(task, state)
    if difficulty == "medium":
        node_records, current_focus, observed_nodes = _planning_status(
            state,
            fallback_answer,
            MEDIUM_PLANNING_NODES,
        )
        if answer_urgent:
            _force_planning_focus(node_records, "answer_submission")
            current_focus = "answer_submission"
        elif current_focus == "answer_submission":
            current_focus = _defer_answer_submission_focus(
                node_records,
                fallback_focus="final_focused_attempt" if final_focused_attempt else "target_columns",
            )
        lines = [
            "PLANNING PREFIX - medium task light schema/final-column checklist.",
            "This compact checklist is inserted at the front of every model input for this task.",
            "Do not spend a step restating the checklist; inspect schema/columns only if not already clear, then solve directly.",
            f"Progress: model call {step_index}/{max_steps}; current_focus={current_focus}; "
            f"observed_signals={observed_nodes or ['none']}.",
            "Light checklist:",
        ]
        for node in node_records:
            dependencies = ", ".join(str(dep) for dep in node["dependent_task_ids"]) or "-"
            lines.append(
                f"- {node['id']} deps=[{dependencies}] status={node['status']}: "
                f"{node['instruction']}"
            )
        lines.extend(MEDIUM_SEMANTIC_GUARD_LINES)
        lines.append(
            "Priority: verify real tables/columns for requested concepts, decide the exact final "
            "answer columns, then compute and submit with no helper columns. Do not follow the "
            "full multi-stage planning graph for medium tasks."
        )
        if answer_urgent:
            lines.append(
                "FINAL-STEPS SUBMISSION RULE: a parseable final answer candidate is already "
                "available. Return `Answer:` now with exactly those requested columns and rows."
            )
        elif final_focused_attempt:
            lines.append(
                "FINAL-STEPS EXACTNESS RULE: do not submit a guess. Run at most one focused "
                "script to verify the exact requested value and print `FINAL_TABLE_JSON:` only "
                "if the table is exact."
            )
        prompt_prefix = "\n".join(lines)
        snapshot = {
            "enabled": True,
            "mode": "medium_light",
            "difficulty": task.difficulty,
            "step_index": step_index,
            "max_steps": max_steps,
            "remaining_model_calls": max_steps - step_index + 1,
            "answer_submission_urgent": answer_urgent,
            "final_focused_attempt": final_focused_attempt,
            "model_input_position": "front_after_system_before_task_prompt",
            "current_focus": current_focus,
            "observed_nodes": observed_nodes,
            "nodes": node_records,
            "semantic_guard": list(MEDIUM_SEMANTIC_GUARD_LINES),
            "prompt_prefix": prompt_prefix,
        }
        return prompt_prefix, snapshot

    node_records, current_focus, observed_nodes = _planning_status(state, fallback_answer)
    if answer_urgent:
        _force_planning_focus(node_records, "answer_submission")
        current_focus = "answer_submission"
    elif current_focus == "answer_submission":
        current_focus = _defer_answer_submission_focus(
            node_records,
            fallback_focus="final_focused_attempt" if final_focused_attempt else "final_verify",
        )
    lines = [
        "PLANNING PREFIX - hard/extreme task working checklist.",
        "This checklist is inserted at the front of every model input for this task.",
        "Do not spend a step restating the plan; use it to choose the next action.",
        "Statuses are heuristic signals, not proof that a step is fully solved; confirm evidence before relying on them.",
        f"Progress: model call {step_index}/{max_steps}; current_focus={current_focus}; "
        f"observed_signals={observed_nodes or ['none']}.",
        "Dependency checklist:",
    ]
    for node in node_records:
        dependencies = ", ".join(str(dep) for dep in node["dependent_task_ids"]) or "-"
        lines.append(
            f"- {node['id']} deps=[{dependencies}] status={node['status']}: "
            f"{node['instruction']}"
        )
    lines.append("Evidence checklist:")
    for evidence in evidence_ledger:
        lines.append(
            f"- {evidence['id']} status={evidence['status']}: "
            f"{evidence['instruction']}"
        )
    if loop_guard.get("triggered"):
        lines.append(
            "LOOP GUARD: the recent execution pattern repeated code, repeated output, "
            "or repeated failed schema/search evidence. Do not repeat the same search; "
            "change parser/source or make one final supported computation."
        )
    lines.append(
        "Priority: first link schema and decide exact final columns; then map "
        "question terms to real columns/values; submit as soon as the answer is "
        "supported, with no helper columns. Never assume schema-absent concepts; "
        "verify their concrete data source first."
    )
    if require_linking_helper:
        lines.append(
            "GROUNDED-LINK REQUIREMENT: in your next Python step, call "
            "`link_question_to_data(max_candidates=5)` before any broad markdown search. "
            "Use its row_ids, usable_filter, value_matches, and join_candidates to connect "
            "question mentions or markdown records to real data. You may continue the "
            "actual computation in the same code block after this call."
        )
    if answer_urgent:
        lines.append(
            "FINAL-STEPS SUBMISSION RULE: a parseable final answer candidate is already "
            "available. Return `Answer:` now with exactly those requested columns and rows."
        )
    elif final_focused_attempt:
        lines.append(
            "FINAL-STEPS EXACTNESS RULE: do not submit a guess. Run at most one focused "
            "script to verify the exact requested value and print `FINAL_TABLE_JSON:` only "
            "if the table is exact."
        )
    prompt_prefix = "\n".join(lines)
    snapshot = {
        "enabled": True,
        "difficulty": task.difficulty,
        "step_index": step_index,
        "max_steps": max_steps,
        "remaining_model_calls": max_steps - step_index + 1,
        "answer_submission_urgent": answer_urgent,
        "final_focused_attempt": final_focused_attempt,
        "link_question_to_data_required": require_linking_helper,
        "model_input_position": "front_after_system_before_task_prompt",
        "current_focus": current_focus,
        "observed_nodes": observed_nodes,
        "evidence_ledger": evidence_ledger,
        "loop_guard": loop_guard,
        "nodes": node_records,
        "prompt_prefix": prompt_prefix,
    }
    return prompt_prefix, snapshot


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        prompt_tool_names: tuple[str, ...] | None = None,
        checkpoint_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.prompt_tool_names = prompt_tool_names
        self.checkpoint_callback = checkpoint_callback
        self._last_planning_snapshot: dict[str, object] | None = None

    def _checkpoint(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        status: str,
        step_index: int | None = None,
        planning: dict[str, object] | None = None,
    ) -> None:
        if self.checkpoint_callback is None:
            return
        if planning is not None:
            self._last_planning_snapshot = planning
        payload = AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        ).to_dict()
        payload["checkpoint"] = {
            "status": status,
            "step_index": step_index,
        }
        if self._last_planning_snapshot is not None:
            payload["planning"] = self._last_planning_snapshot
        self.checkpoint_callback(payload)

    def _append_step(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        step: StepRecord,
    ) -> None:
        state.steps.append(step)
        self._checkpoint(task, state, status="after_step", step_index=step.step_index)

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        front_instruction: str | None = None,
        runtime_instruction: str | None = None,
    ) -> list[ModelMessage]:
        is_codeact = self.system_prompt == CODEACT_REACT_SYSTEM_PROMPT
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(self.prompt_tool_names),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        if front_instruction:
            messages.append(ModelMessage(role="user", content=front_instruction))
        messages.append(ModelMessage(role="user", content=build_task_prompt(task, codeact=is_codeact)))
        history_steps = state.steps
        if self.config.prompt_history_steps > 0:
            history_steps = history_steps[-self.config.prompt_history_steps :]
        for step in history_steps:
            assistant_content = step.raw_response
            if step.action == "__error__":
                assistant_content = (
                    "Invalid previous response omitted by runtime. "
                    "Use the observation to correct the next step."
                )
            messages.append(ModelMessage(role="assistant", content=assistant_content))
            messages.append(
                ModelMessage(
                    role="user",
                    content=build_observation_prompt(
                        _render_observation_for_prompt(step.observation, self.config)
                    ),
                )
            )
        final_answer_reminder = (
            _build_difficulty_final_answer_reminder(task, state)
            if is_codeact
            else None
        )
        if final_answer_reminder:
            messages.append(ModelMessage(role="user", content=final_answer_reminder))
        if runtime_instruction:
            messages.append(ModelMessage(role="user", content=runtime_instruction))
        return messages

    def _build_runtime_instruction(
        self,
        *,
        task: PublicTask,
        step_index: int,
        state: AgentRuntimeState,
        fallback_answer: AnswerTable | None,
        consecutive_parse_errors: int,
    ) -> str | None:
        if self.system_prompt != CODEACT_REACT_SYSTEM_PROMPT:
            return None

        instructions: list[str] = []
        if consecutive_parse_errors:
            instructions.append(
                "Runtime recovery: your previous response could not be parsed. "
                "Do not repeat Thought-only text or <think> tags. Return exactly one valid step: "
                "either `Thought:` plus a fenced `Code:` block, or `Thought:` plus `Answer:` with fenced JSON."
            )
        loop_guard = (
            _loop_guard_summary(state)
            if _task_uses_advanced_planning(task)
            else {"triggered": False}
        )
        if loop_guard.get("triggered"):
            missing_evidence = [
                str(item["id"])
                for item in _task_evidence_ledger(task, state)
                if item.get("status") == "needs_evidence"
            ][:4]
            repeated_bits = [
                f"code_repeat_count={loop_guard.get('code_repeat_count')}",
                f"output_repeat_count={loop_guard.get('output_repeat_count')}",
            ]
            if loop_guard.get("repeated_schema_search"):
                repeated_bits.append("repeated_schema_or_search=True")
            instructions.append(
                "Loop guard: recent steps repeated the same code, same output, or the same failed "
                f"schema/search conclusion ({', '.join(repeated_bits)}). "
                "Do not rerun the same schema inspection, markdown search, or parser. "
                "Change tactic now: use a different source/parser, parse markdown into record-level rows, "
                "or make one final best-supported computation if the blocker has already been observed. "
                f"Still-needed evidence: {missing_evidence or ['none obvious']}."
            )
        if _should_require_link_question_to_data(task, state):
            instructions.append(
                "Grounded-link requirement: in your next Python code block, call "
                "`link_question_to_data(max_candidates=5)` before broad markdown search or repeated manual matching. "
                "Use its `row_ids`, `usable_filter`, `value_matches`, and `join_candidates` to connect question "
                "mentions or markdown records to actual structured data. You may continue solving in the same script."
            )
        if _should_prioritize_answer_submission(
            task,
            state,
            fallback_answer,
            step_index=step_index,
            max_steps=self.config.max_steps,
        ):
            answer = _latest_submission_ready_candidate(task, state, fallback_answer)
            if answer is not None:
                preview = _safe_json_dumps(_answer_preview(answer, max_rows=2))
                instructions.append(
                    "Final submission priority: a valid final answer candidate is already available. "
                    f"Submit `Answer:` now using exactly this table shape: {preview}. "
                    "Do not run more code unless you can name a missing requested value."
                )
        elif _should_issue_final_focused_attempt(
            task,
            state,
            fallback_answer,
            step_index=step_index,
            max_steps=self.config.max_steps,
        ):
            instructions.append(
                "Final focused attempt: do not submit a guess. Run at most one compact script that verifies the "
                "exact requested operation and prints `FINAL_TABLE_JSON:` only if the table is exact. "
                "For 'how many times', compute a ratio, not a count. For percentage questions, return a percentage. "
                "For normal/abnormal wording, verify the threshold source before finalizing."
            )
        if not instructions:
            return None
        return "\n\n".join(instructions)

    def _candidate_answer_from_observation(
        self,
        task: PublicTask,
        observation: dict[str, object],
    ) -> AnswerTable | None:
        if observation.get("tool") != "execute_python":
            return None
        content = observation.get("content")
        if not isinstance(content, dict):
            return None
        stdout = str(content.get("output") or content.get("stdout") or "")
        payload = _find_marker_payload(stdout, FINAL_RESULT_MARKERS)
        if payload is None:
            return None

        answer = _answer_table_from_payload(payload)
        if answer is None:
            return None
        return _sanitize_answer_table(task, answer)

    def _sanitize_model_step(self, task: PublicTask, model_step: ModelStep) -> ModelStep:
        if model_step.action != "answer":
            return model_step
        answer = _answer_table_from_payload(model_step.action_input)
        if answer is None:
            return model_step
        sanitized = _sanitize_answer_table(task, answer)
        return ModelStep(
            thought=model_step.thought,
            action=model_step.action,
            action_input=sanitized.to_dict(),
            raw_response=model_step.raw_response,
        )

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        self._last_planning_snapshot = None
        fallback_answer: AnswerTable | None = None
        consecutive_parse_errors = 0
        for step_index in range(1, self.config.max_steps + 1):
            try:
                planning_instruction, planning_snapshot = _build_planning_context(
                    task,
                    state,
                    fallback_answer,
                    step_index=step_index,
                    max_steps=self.config.max_steps,
                )
                self._checkpoint(
                    task,
                    state,
                    status="before_model_request",
                    step_index=step_index,
                    planning=planning_snapshot,
                )
                raw_response = self.model.complete(
                    self._build_messages(
                        task,
                        state,
                        front_instruction=planning_instruction,
                        runtime_instruction=self._build_runtime_instruction(
                            task=task,
                            step_index=step_index,
                            state=state,
                            fallback_answer=fallback_answer,
                            consecutive_parse_errors=consecutive_parse_errors,
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                state.failure_reason = f"Model request failed before step {step_index}: {exc}"
                self._append_step(
                    task,
                    state,
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response="",
                        observation={
                            "ok": False,
                            "error": f"Model request failed before a response was returned: {exc}",
                        },
                        ok=False,
                    )
                )
                break
            try:
                model_step = parse_model_step(raw_response)
            except Exception as exc:
                consecutive_parse_errors += 1
                observation = {
                    "ok": False,
                    "error": str(exc),
                    "recovery_instruction": (
                        "Return exactly one valid CodeAct step next. Do not repeat invalid text. "
                        "Use `Thought:` plus fenced `Code:` or `Thought:` plus `Answer:` fenced JSON."
                    ),
                }
                self._append_step(
                    task,
                    state,
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )
                if consecutive_parse_errors >= 2 and fallback_answer is not None:
                    state.answer = fallback_answer
                    break
                continue

            consecutive_parse_errors = 0
            try:
                model_step = self._sanitize_model_step(task, model_step)
                if (
                    self.system_prompt == CODEACT_REACT_SYSTEM_PROMPT
                    and model_step.action == "answer"
                ):
                    answer = _answer_table_from_payload(model_step.action_input)
                    if answer is not None:
                        answer = _sanitize_answer_table(task, answer)
                        model_step = ModelStep(
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=answer.to_dict(),
                            raw_response=model_step.raw_response,
                        )

                if _should_block_medium_empty_answer(task, state, model_step):
                    observation = {
                        "ok": False,
                        "medium_empty_answer_recheck": True,
                        "error": (
                            "Medium empty-answer guard: this is a row-returning question, but the "
                            "answer has zero rows. Before submitting an empty table, run one compact "
                            "Python check over all plausible CSV/JSON/SQLite date and value sources, "
                            "including JSON wrappers, then submit the verified result."
                        ),
                    }
                    self._append_step(
                        task,
                        state,
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=False,
                        )
                    )
                    continue

                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                content = dict(tool_result.content)
                if model_step.action == "execute_python":
                    repair_hints = _python_repair_hints(
                        content,
                        str(model_step.action_input.get("code") or ""),
                    )
                    if repair_hints:
                        content["repair_hints"] = repair_hints
                elif model_step.action == "answer":
                    direct_answer = _answer_table_from_payload(model_step.action_input)
                    if direct_answer is not None:
                        content["final_answer_check"] = _final_answer_check(task, direct_answer)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": content,
                }
                candidate_answer = self._candidate_answer_from_observation(task, observation)
                if candidate_answer is not None:
                    content["final_answer_check"] = _final_answer_check(task, candidate_answer)
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                self._append_step(task, state, step_record)
                if candidate_answer is not None:
                    decision = _candidate_decision_from_observation(
                        task,
                        observation,
                        candidate_answer,
                    )
                    self._checkpoint(
                        task,
                        state,
                        status="after_candidate_decision",
                        step_index=step_index,
                    )
                    if decision.store_as_fallback:
                        fallback_answer = decision.answer
                    if (
                        decision.auto_submit
                        and
                        self.system_prompt == CODEACT_REACT_SYSTEM_PROMPT
                        and model_step.action == "execute_python"
                        and tool_result.ok
                    ):
                        state.answer = decision.answer
                        break
                if tool_result.is_terminal:
                    state.answer = (
                        _sanitize_answer_table(task, tool_result.answer)
                        if tool_result.answer is not None
                        else None
                    )
                    break
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                self._append_step(
                    task,
                    state,
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None and fallback_answer is not None:
            state.answer = fallback_answer

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
