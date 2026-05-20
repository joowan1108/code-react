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
    verifier_enabled: bool = True
    verifier_stdout_chars: int = 2500


@dataclass(frozen=True, slots=True)
class CandidateAnswerDecision:
    answer: AnswerTable
    auto_submit: bool
    store_as_fallback: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateVerificationDecision:
    verdict: str
    reasons: tuple[str, ...]
    repair_instruction: str | None = None
    raw_response: str = ""


FINAL_RESULT_MARKERS = (
    "FINAL_TABLE_JSON:",
    "FINAL_RESULT:",
    "ANSWER_CANDIDATE:",
)
EVIDENCE_LEDGER_MARKERS = (
    "EVIDENCE_LEDGER_JSON:",
    "EVIDENCE_LEDGER:",
)
EVIDENCE_CONSISTENCY_MARKERS = (
    "EVIDENCE_CONSISTENCY_JSON:",
    "CONSISTENCY_CHECK_JSON:",
)

PLANNING_DIFFICULTIES = {"medium", "hard", "extreme"}
LINKING_REQUIREMENT_DIFFICULTIES = {"hard", "extreme"}
SPECIAL_EVIDENCE_IDS = {
    "value_entity_linking",
    "semantic_rule_or_threshold",
    "calculation_semantics",
    "temporal_scope",
    "numeric_filter_semantics",
    "markdown_context_mapping",
}

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
    "- Row grain: decide what one final row represents before filtering or grouping, using the task wording and observed table values.",
    "- Formula/unit: for average monthly/yearly, ratio, percentage, proportion, per unit, or total wording, verify the denominator and unit from schema, samples, or knowledge before finalizing.",
    "- Value/date scope: before returning an empty table, check every plausible date/value source across CSV, JSON wrappers, and SQLite tables.",
    "- Ambiguous columns: when words like type/category/status/date/id can map to multiple columns, inspect actual values and choose the column whose values answer the question.",
)

BIRD_SEMANTIC_CONTRACT_LINES = (
    "BIRD semantic contract:",
    "- Before finalizing a calculation, write down the row grain, filters, formula, denominator/unit, and final output shape in the Thought or compact Python prints.",
    "- Date/month/year mentions must be matched to observed encoded values in the data, such as YYYYMM, YYYY-MM-DD, or separate year/month columns.",
    "- Percentage, ratio, increase/decrease rate, and how-many-times questions require an explicit numerator and denominator before submission.",
    "- Average monthly/yearly wording requires checking whether the data is already monthly/yearly or needs unit conversion.",
    "- Normal/abnormal/range/severe/status wording requires a grounded threshold or coded meaning from data or markdown before filtering.",
    "- If the final answer has more than 20 rows, do not hand-write the Answer JSON; print `FINAL_TABLE_JSON:` from Python so the runtime can submit it.",
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


def _load_first_json_value(text: str) -> object:
    json_fence = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = json_fence.group(1).strip() if json_fence is not None else text.strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, (dict, list)):
            return payload
    raise ValueError("No JSON object or array found.")


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

    row_or_entity_cues = (
        "list",
        "identify",
        "name the",
        "names of",
        "full names",
        "names and",
        "what are",
        "which ",
        "who ",
        "write ",
        "give their",
        "type of",
        "what currency",
        "currency was",
    )
    if any(
        phrase in q
        for phrase in row_or_entity_cues
    ):
        return False
    if q.count("?") >= 2 and any(
        phrase in q
        for phrase in (
            "what currency",
            "which ",
            "who ",
            "name ",
            "id ",
            "identifier",
        )
    ):
        return False
    if any(phrase in q for phrase in ("how many", "how much", "what percentage", "calculate the percentage")):
        return True
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
            "amount",
            "cost",
            "price",
            "value",
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


def _task_is_medium(task: PublicTask) -> bool:
    return task.difficulty.casefold() == "medium"


def _deduplicate_answer_rows(answer: AnswerTable) -> AnswerTable:
    seen: set[tuple[object, ...]] = set()
    rows: list[list[object]] = []
    for row in answer.rows:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    if len(rows) == len(answer.rows):
        return answer
    return AnswerTable(columns=list(answer.columns), rows=rows)


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
    return _deduplicate_answer_rows(complete_rows)


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


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    omitted = len(text) - head_chars - tail_chars
    return f"{text[:head_chars]}\n...[omitted {omitted} chars]...\n{text[-tail_chars:]}"


def _normalize_verifier_verdict(value: object) -> str:
    verdict = str(value or "uncertain").strip().casefold()
    if verdict in {"accept", "accepted", "pass", "passed", "ok", "correct", "yes"}:
        return "accept"
    if verdict in {"reject", "rejected", "fail", "failed", "incorrect", "wrong", "no"}:
        return "reject"
    return "uncertain"


def _parse_verifier_decision(raw_response: str) -> CandidateVerificationDecision:
    payload = _load_first_json_object(raw_response)
    verdict = _normalize_verifier_verdict(
        payload.get("verdict", payload.get("decision", payload.get("status")))
    )
    raw_reasons = payload.get("reasons", payload.get("reason", []))
    if isinstance(raw_reasons, list):
        reasons = tuple(str(reason) for reason in raw_reasons if str(reason).strip())
    elif raw_reasons:
        reasons = (str(raw_reasons),)
    else:
        reasons = ()
    repair_instruction = payload.get("repair_instruction", payload.get("next_step"))
    if repair_instruction is not None:
        repair_instruction = str(repair_instruction).strip() or None
    return CandidateVerificationDecision(
        verdict=verdict,
        reasons=reasons,
        repair_instruction=repair_instruction,
        raw_response=raw_response,
    )


def _verifier_decision_to_dict(decision: CandidateVerificationDecision) -> dict[str, object]:
    payload: dict[str, object] = {
        "verdict": decision.verdict,
        "reasons": list(decision.reasons),
    }
    if decision.repair_instruction:
        payload["repair_instruction"] = decision.repair_instruction
    if decision.raw_response:
        payload["raw_response"] = _truncate_middle(decision.raw_response, 2000)
    return payload


def _content_verifier_verdict(content: dict[str, object]) -> str | None:
    verifier = content.get("independent_verifier")
    if not isinstance(verifier, dict):
        return None
    return _normalize_verifier_verdict(verifier.get("verdict"))


def _task_uses_independent_verifier(task: PublicTask, config: ReActAgentConfig) -> bool:
    return config.verifier_enabled and task.difficulty.casefold() in {"hard", "extreme"}


def _build_candidate_verifier_messages(
    task: PublicTask,
    model_step: ModelStep,
    observation: dict[str, object],
    answer: AnswerTable,
    *,
    stdout_chars: int,
) -> list[ModelMessage]:
    content = observation.get("content")
    content_dict = content if isinstance(content, dict) else {}
    stdout = str(content_dict.get("output") or content_dict.get("stdout") or "")
    stderr = str(content_dict.get("stderr") or "")
    traceback_text = str(content_dict.get("traceback") or "")
    code = ""
    if isinstance(model_step.action_input, dict):
        code = str(model_step.action_input.get("code") or "")

    if len(answer.rows) <= 20:
        candidate_payload: object = answer.to_dict()
    else:
        candidate_payload = _answer_preview(answer, max_rows=8)

    verifier_input = {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "question": task.question,
        "candidate_answer": candidate_payload,
        "final_answer_check": content_dict.get("final_answer_check"),
        "evidence_ledger_report": content_dict.get("evidence_ledger_report"),
        "execution": {
            "ok": observation.get("ok"),
            "stdout": _truncate_middle(stdout, stdout_chars),
            "stderr": _truncate_middle(stderr, 1200),
            "traceback": _truncate_middle(traceback_text, 1200),
        },
        "last_python_code": _truncate_middle(code, 3500),
    }
    system_message = (
        "You are an independent verifier for a data-analysis answer candidate. "
        "You do not solve from scratch. Check whether the candidate is directly supported "
        "by the shown code, execution output, and evidence. Return exactly one JSON object."
    )
    user_message = (
        "Verify the candidate answer below.\n"
        "Accept only when the executed code/output clearly supports the exact requested rows, "
        "columns, filters, formula, denominator/unit, and semantic mappings. "
        "Reject when there is a concrete mismatch, traceback, unsupported assumption, extra/missing "
        "final column, wrong formula, or ignored verified evidence. Use uncertain if there is not enough "
        "information to decide.\n\n"
        "Return JSON with this schema:\n"
        '{"verdict":"accept|reject|uncertain","reasons":["..."],"repair_instruction":"optional next action"}'
        "\n\nVerifier input:\n"
        + _safe_json_dumps(verifier_input)
    )
    return [
        ModelMessage(role="system", content=system_message),
        ModelMessage(role="user", content=user_message),
    ]


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
            "- If the question has date/month/year wording, verify the observed encoded value format before submitting.",
            "- If the question asks for a percentage, ratio, average, or rate, verify the denominator/unit and return that computed metric, not a raw count.",
            "- If the final answer has more than 20 rows, do not hand-write the full Answer JSON; run Python and print `FINAL_TABLE_JSON:` instead.",
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


def _find_marker_json_value(text: str, markers: tuple[str, ...]) -> object | None:
    lower_text = text.lower()
    marker_matches = [
        (lower_text.find(marker.lower()), marker)
        for marker in markers
        if lower_text.find(marker.lower()) >= 0
    ]
    if not marker_matches:
        return None
    marker_index, marker = min(marker_matches, key=lambda item: item[0])
    marker_end = marker_index + len(marker)
    try:
        return _load_first_json_value(text[marker_end:])
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

    answer_warnings = _final_answer_check_warnings(task, answer)
    if answer_warnings:
        reasons.append("final_answer_check_warnings_present")
        hard_reasons.append("final_answer_check_warnings_present")
    verifier_verdict = _content_verifier_verdict(content_dict)
    if content_dict.get("verifier_rejected") or verifier_verdict == "reject":
        reasons.append("independent_verifier_rejected")
        hard_reasons.append("independent_verifier_rejected")
    if content_dict.get("candidate_review_instruction"):
        reasons.append("candidate_review_required")
        hard_reasons.append("candidate_review_required")
    if len(answer.rows) <= 20 and verifier_verdict != "accept":
        reasons.append("small_candidate_requires_model_review")
        hard_reasons.append("small_candidate_requires_model_review")

    store_as_fallback = (
        bool(answer.rows)
        and "python_traceback_present" not in reasons
        and "independent_verifier_rejected" not in reasons
    )
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


QUESTION_TERM_STOPWORDS = {
    "about",
    "after",
    "among",
    "and",
    "answer",
    "are",
    "average",
    "before",
    "between",
    "by",
    "calculate",
    "column",
    "columns",
    "count",
    "does",
    "each",
    "find",
    "for",
    "from",
    "give",
    "has",
    "have",
    "highest",
    "how",
    "include",
    "into",
    "list",
    "lowest",
    "many",
    "max",
    "mean",
    "min",
    "much",
    "name",
    "number",
    "only",
    "per",
    "please",
    "provide",
    "ratio",
    "return",
    "rows",
    "show",
    "table",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "times",
    "total",
    "value",
    "values",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def _ordered_unique_strings(values: list[str], *, max_items: int = 8) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
        if len(unique) >= max_items:
            break
    return unique


def _quoted_question_terms(question: str) -> list[str]:
    terms = [
        match.group(1) or match.group(2)
        for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', question)
    ]
    return _ordered_unique_strings(terms, max_items=8)


def _capitalized_question_terms(question: str) -> list[str]:
    terms = re.findall(
        r"\b[A-Z][A-Za-z0-9]*(?:[-\s]+[A-Z][A-Za-z0-9]*)*\b",
        question,
    )
    filtered = [
        term
        for term in terms
        if len(term) >= 3 and term.casefold() not in QUESTION_TERM_STOPWORDS
    ]
    return _ordered_unique_strings(filtered, max_items=8)


def _domain_question_terms(question: str) -> list[str]:
    terms = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", question)
        if token.casefold() not in QUESTION_TERM_STOPWORDS
    ]
    return _ordered_unique_strings(terms, max_items=10)


def _numeric_question_terms(question: str) -> list[str]:
    return _ordered_unique_strings(re.findall(r"[-+]?\d+(?:\.\d+)?%?", question), max_items=8)


def _question_terms_for_evidence(task: PublicTask) -> list[str]:
    return _ordered_unique_strings(
        _quoted_question_terms(task.question)
        + _capitalized_question_terms(task.question)
        + _domain_question_terms(task.question)
        + _numeric_question_terms(task.question),
        max_items=12,
    )


def _evidence_entries_from_value(value: object) -> list[dict[str, object]]:
    if isinstance(value, dict):
        for key in ("items", "evidence", "ledger", "entries"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        if any(key in value for key in ("id", "status", "claim", "source", "evidence")):
            return [value]
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _evidence_entries_from_state(state: AgentRuntimeState) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for step in state.steps:
        content = step.observation.get("content")
        if not isinstance(content, dict):
            continue
        entries.extend(_evidence_entries_from_value(content.get("evidence_ledger_report")))
    return entries


def _evidence_entry_is_verified(entry: dict[str, object]) -> bool:
    if entry.get("verified") is True or entry.get("ok") is True:
        return True
    status = str(entry.get("status") or "").casefold()
    return status in {"verified", "grounded", "found", "satisfied", "done", "ok", "true"}


def _evidence_entry_matches(entry: dict[str, object], requirement_id: str) -> bool:
    labels = [
        entry.get("id"),
        entry.get("evidence_id"),
        entry.get("requirement_id"),
        entry.get("requirement"),
        entry.get("type"),
        entry.get("kind"),
        entry.get("evidence_type"),
    ]
    normalized = {str(label).casefold() for label in labels if label}
    return requirement_id.casefold() in normalized


def _evidence_entry_requirement_id(entry: dict[str, object]) -> str | None:
    for key in (
        "id",
        "evidence_id",
        "requirement_id",
        "requirement",
        "type",
        "kind",
        "evidence_type",
    ):
        value = entry.get(key)
        if not value:
            continue
        normalized = str(value).casefold()
        if normalized in SPECIAL_EVIDENCE_IDS:
            return normalized
    return None


def _consistency_entry_is_satisfied(entry: dict[str, object]) -> bool:
    for key in (
        "used_in_final_computation",
        "consistent",
        "verified",
        "ok",
        "satisfied",
    ):
        value = entry.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.casefold() in {"true", "yes", "used", "verified", "consistent"}:
            return True
    status = str(entry.get("status") or "").casefold()
    return status in {
        "used",
        "verified",
        "grounded",
        "consistent",
        "satisfied",
        "done",
        "ok",
        "true",
    }


def _verified_special_evidence_ids(
    task: PublicTask,
    state: AgentRuntimeState,
    *,
    current_content: dict[str, object] | None = None,
) -> list[str]:
    if not _task_uses_advanced_planning(task):
        return []
    entries = _evidence_entries_from_state(state)
    if current_content is not None:
        entries.extend(_evidence_entries_from_value(current_content.get("evidence_ledger_report")))

    ids: list[str] = []
    for entry in entries:
        requirement_id = _evidence_entry_requirement_id(entry)
        if requirement_id is None or not _evidence_entry_is_verified(entry):
            continue
        if requirement_id not in ids:
            ids.append(requirement_id)
    return ids


def _evidence_consistency_warnings(
    task: PublicTask,
    state: AgentRuntimeState,
    *,
    current_content: dict[str, object],
) -> tuple[list[str], list[str]]:
    required_ids = _verified_special_evidence_ids(
        task,
        state,
        current_content=current_content,
    )
    if not required_ids:
        return [], []

    consistency_entries = _evidence_entries_from_value(
        current_content.get("evidence_consistency_report")
    )
    if not consistency_entries:
        return required_ids, [
            "missing EVIDENCE_CONSISTENCY_JSON showing how verified evidence was used in the final computation"
        ]

    warnings: list[str] = []
    for requirement_id in required_ids:
        matches = [
            entry
            for entry in consistency_entries
            if _evidence_entry_matches(entry, requirement_id)
        ]
        if not matches:
            warnings.append(f"{requirement_id} is not covered by EVIDENCE_CONSISTENCY_JSON")
            continue
        if not any(_consistency_entry_is_satisfied(entry) for entry in matches):
            warnings.append(f"{requirement_id} is reported but not marked used/consistent")
    return required_ids, warnings


def _latest_evidence_consistency_warnings(state: AgentRuntimeState) -> list[str]:
    for step in reversed(state.steps):
        content = step.observation.get("content")
        if not isinstance(content, dict):
            continue
        if content.get("evidence_consistency_resolved"):
            return []
        warnings = content.get("evidence_consistency_warnings")
        if isinstance(warnings, list) and warnings:
            return [str(warning) for warning in warnings]
    return []


def _evidence_requirement_status(
    all_text: str,
    requirement_id: str,
    keywords: tuple[str, ...],
    evidence_entries: list[dict[str, object]],
) -> str:
    if any(
        _evidence_entry_matches(entry, requirement_id) and _evidence_entry_is_verified(entry)
        for entry in evidence_entries
    ):
        return "verified"
    if any(keyword in all_text for keyword in keywords):
        return "maybe_found"
    return "needs_evidence"


def _task_evidence_ledger(
    task: PublicTask,
    state: AgentRuntimeState,
    *,
    extra_evidence_entries: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    question = task.question.casefold()
    tokens = _split_identifier_tokens(task.question)
    all_text = "\n".join(_step_text_for_planning(step) for step in state.steps)
    evidence_entries = _evidence_entries_from_state(state)
    if extra_evidence_entries:
        evidence_entries = [*evidence_entries, *extra_evidence_entries]
    question_terms = _question_terms_for_evidence(task)
    requirements: list[dict[str, object]] = [
        (
            {
                "id": "concrete_data_source",
                "question_terms": question_terms,
                "instruction": (
                    "Map each important question term to an observed file/table/column/value or markdown record; "
                    "do not rely on schema-only assumptions."
                ),
                "accepted_sources": ["manifest schema", "CSV/JSON/SQLite samples", "markdown records/rules"],
                "keywords": ("columns", "tables", "files in directory", "link result", "json/", "csv/", "db/", "doc/"),
            }
        ),
        {
            "id": "row_grain",
            "question_terms": question_terms,
            "instruction": "State what one final row represents before joins, filters, or grouping.",
            "accepted_sources": ["question wording", "selected tables", "grouping keys"],
            "keywords": ("row grain", "one row", "groupby", "group by", "distinct", "per "),
        },
        {
            "id": "final_answer_shape",
            "question_terms": question_terms,
            "instruction": (
                "Know the exact final columns and whether the answer is one value, rows, count, percentage, or ratio."
            ),
            "accepted_sources": ["question wording", "computed final table"],
            "keywords": ("final_table_json", "answer_candidate", "final answer", "columns", "row_count"),
        },
    ]
    if (
        tokens & {"percentage", "percent", "ratio", "proportion", "average", "avg", "mean", "rate"}
        or "how many times" in question
        or "how much" in question
    ):
        requirements.append(
            {
                "id": "calculation_semantics",
                "question_terms": sorted(tokens & {"percentage", "percent", "ratio", "proportion", "average", "avg", "mean", "rate"}),
                "instruction": (
                    "Define the formula, numerator/denominator or unit conversion from grounded records before computing."
                ),
                "accepted_sources": ["question wording", "schema/sample values", "markdown formula/rule"],
                "keywords": ("numerator", "denominator", "percentage", "ratio", "average", "unit conversion", "total count"),
            }
        )
    numeric_terms = _numeric_question_terms(task.question)
    if numeric_terms or "between" in tokens or re.search(r"\b\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\b", question):
        requirements.append(
            {
                "id": "numeric_filter_semantics",
                "question_terms": numeric_terms,
                "instruction": "Map every numeric value or bound to a real column and apply the question's inclusive/exclusive wording.",
                "accepted_sources": ["question wording", "column samples", "markdown threshold/range"],
                "keywords": ("between", ">=", "<=", "numeric_filters", "boundary", "threshold", "range"),
            }
        )
    if tokens & {"abnormal", "normal", "threshold", "range", "severe", "legal", "status", "diagnosis", "disease", "code", "coded"}:
        requirements.append(
            {
                "id": "semantic_rule_or_threshold",
                "question_terms": sorted(tokens & {"abnormal", "normal", "threshold", "range", "severe", "legal", "status", "diagnosis", "disease", "code", "coded"}),
                "instruction": "Find the concrete rule, coded value, threshold, or status definition before filtering.",
                "accepted_sources": ["structured code/value table", "markdown rule", "observed values"],
                "keywords": ("threshold", "normal range", "upper limit", "abnormal", "legal", "coded", "severity", "diagnosis"),
            }
        )
    if (
        _quoted_question_terms(task.question)
        or _capitalized_question_terms(task.question)
        or tokens & {"named", "called", "label", "category", "type", "class", "entity", "record"}
    ):
        requirements.append(
            {
                "id": "value_entity_linking",
                "question_terms": _ordered_unique_strings(
                    _quoted_question_terms(task.question) + _capitalized_question_terms(task.question),
                    max_items=10,
                )
                or question_terms,
                "instruction": "Link named entities, quoted strings, labels, or categories to exact observed values before filtering.",
                "accepted_sources": ["database value samples", "link_question_to_data output", "markdown records"],
                "keywords": ("value_matches", "usable_filter", "row_ids", "matched", "exact value", "link result"),
            }
        )
    if tokens & {"date", "month", "monthly", "year", "yearly", "day", "before", "after", "between"}:
        requirements.append(
            {
                "id": "temporal_scope",
                "question_terms": sorted(tokens & {"date", "month", "monthly", "year", "yearly", "day", "before", "after", "between"}),
                "instruction": "Verify the observed date encoding and the exact temporal scope before filtering.",
                "accepted_sources": ["date column samples", "derived year/month columns", "markdown date rule"],
                "keywords": ("yyyy", "yyyy-mm-dd", "yyyymm", "to_datetime", "date encoding", "year", "month"),
            }
        )
    if _task_has_markdown(task):
        requirements.append(
            {
                "id": "markdown_context_mapping",
                "question_terms": question_terms,
                "instruction": (
                    "If a needed fact lives in .md, extract compact record/rule evidence and map it back to structured IDs, "
                    "columns, or final rows before computing."
                ),
                "accepted_sources": ["retrieve_knowledge", "search_markdown", "extract_markdown_records", "link_question_to_data"],
                "keywords": ("retrieve_knowledge", "extract_markdown_records", "search_markdown", "doc/", "snippet", "record", "paragraph"),
            }
        )

    return [
        {
            "id": str(requirement["id"]),
            "status": _evidence_requirement_status(
                all_text,
                str(requirement["id"]),
                tuple(str(keyword) for keyword in requirement["keywords"]),
                evidence_entries,
            ),
            "question_terms": list(requirement["question_terms"]),
            "accepted_sources": list(requirement["accepted_sources"]),
            "instruction": str(requirement["instruction"]),
        }
        for requirement in requirements
    ]


def _missing_special_evidence_ids(
    task: PublicTask,
    state: AgentRuntimeState,
    *,
    current_content: dict[str, object] | None = None,
) -> list[str]:
    if not _task_uses_advanced_planning(task):
        return []
    extra_entries: list[dict[str, object]] = []
    if current_content is not None:
        extra_entries = _evidence_entries_from_value(current_content.get("evidence_ledger_report"))
    return [
        str(item["id"])
        for item in _task_evidence_ledger(
            task,
            state,
            extra_evidence_entries=extra_entries,
        )
        if item.get("id") in SPECIAL_EVIDENCE_IDS and item.get("status") == "needs_evidence"
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


def _final_answer_check_warnings(task: PublicTask, answer: AnswerTable) -> list[str]:
    check = _final_answer_check(task, answer)
    warnings = check.get("warnings", [])
    if not isinstance(warnings, list):
        return []
    return [str(warning) for warning in warnings if warning]


def _answer_is_submission_ready(task: PublicTask, answer: AnswerTable) -> bool:
    return (
        _answer_has_submission_shape(task, answer)
        and not _answer_column_is_semantically_suspicious(task, answer)
        and not _final_answer_check_warnings(task, answer)
    )


def _latest_candidate_review_instruction(state: AgentRuntimeState) -> str | None:
    for step in reversed(state.steps[-3:]):
        observation = step.observation
        if not isinstance(observation, dict):
            continue
        content = observation.get("content")
        if not isinstance(content, dict):
            continue
        instruction = content.get("candidate_review_instruction")
        if isinstance(instruction, str) and instruction:
            return instruction
    return None


def _latest_verifier_rejection_instruction(state: AgentRuntimeState) -> str | None:
    for step in reversed(state.steps[-4:]):
        observation = step.observation
        if not isinstance(observation, dict):
            continue
        content = observation.get("content")
        if not isinstance(content, dict):
            continue
        verdict = _content_verifier_verdict(content)
        if verdict == "accept":
            return None
        if not content.get("verifier_rejected") and verdict != "reject":
            continue
        verifier = content.get("independent_verifier")
        repair_instruction = None
        reasons: list[str] = []
        if isinstance(verifier, dict):
            raw_repair = verifier.get("repair_instruction")
            if isinstance(raw_repair, str) and raw_repair.strip():
                repair_instruction = raw_repair.strip()
            raw_reasons = verifier.get("reasons")
            if isinstance(raw_reasons, list):
                reasons = [str(reason) for reason in raw_reasons if str(reason).strip()]
        message = (
            "Independent verifier rejected the previous Python final-table candidate. "
            "Do not submit that same table directly. Run a compact Python repair/check, "
            "fix the concrete mismatch, and print a new `FINAL_TABLE_JSON:` only if it is supported."
        )
        if reasons:
            message += " Verifier reasons: " + "; ".join(reasons[:3]) + "."
        if repair_instruction:
            message += " Suggested repair: " + repair_instruction
        return message
    return None


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
        lines.extend(BIRD_SEMANTIC_CONTRACT_LINES)
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
            "bird_semantic_contract": list(BIRD_SEMANTIC_CONTRACT_LINES),
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
        terms = evidence.get("question_terms")
        term_text = ""
        if isinstance(terms, list) and terms:
            term_text = " terms=[" + ", ".join(str(term) for term in terms[:5]) + "]"
        accepted_sources = evidence.get("accepted_sources")
        source_text = ""
        if isinstance(accepted_sources, list) and accepted_sources:
            source_text = " sources=[" + "; ".join(str(source) for source in accepted_sources[:3]) + "]"
        lines.append(
            f"- {evidence['id']} status={evidence['status']}: "
            f"{evidence['instruction']}{term_text}{source_text}"
        )
    lines.append(
        "Evidence ledger protocol: when you resolve schema/value/markdown semantics in Python, print "
        "`EVIDENCE_LEDGER_JSON:` followed by a compact JSON list. Each item should include "
        "`id`, `status` (`verified` or `unresolved`), `source`, and the concrete columns/values/rules used. "
        "Do this before printing `FINAL_TABLE_JSON:` on hard/extreme tasks with ambiguous values, formulas, "
        "thresholds, or markdown context. You may also print `EVIDENCE_CONSISTENCY_JSON:` to explain where "
        "verified evidence was used, but final acceptance is checked by a separate runtime verifier when enabled."
    )
    lines.extend(BIRD_SEMANTIC_CONTRACT_LINES)
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
        "bird_semantic_contract": list(BIRD_SEMANTIC_CONTRACT_LINES),
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
            likely_truncated_answer = any(
                bool(step.observation.get("likely_truncated_answer"))
                for step in state.steps[-2:]
                if isinstance(step.observation, dict)
            )
            if likely_truncated_answer:
                instructions.append(
                    "Runtime recovery: your previous Answer JSON was too long or truncated. "
                    "Do not write the rows manually. Run Python, recompute the exact final table, "
                    "and print `FINAL_TABLE_JSON:` followed by JSON with `columns` and `rows`."
                )
            else:
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
        missing_special_evidence = _missing_special_evidence_ids(task, state)
        if state.steps and missing_special_evidence:
            instructions.append(
                "Evidence ledger requirement: before finalizing this hard/extreme task, ground these unresolved "
                f"semantic items with observed data or markdown and print `EVIDENCE_LEDGER_JSON:`: "
                f"{missing_special_evidence[:4]}. Keep it compact; each item needs id, status, source, "
                "and concrete columns/values/rules."
            )
        verifier_rejection_instruction = _latest_verifier_rejection_instruction(state)
        candidate_review_instruction = _latest_candidate_review_instruction(state)
        if verifier_rejection_instruction:
            instructions.append(verifier_rejection_instruction)
        elif candidate_review_instruction:
            instructions.append(candidate_review_instruction)
        elif _should_prioritize_answer_submission(
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

    def _verify_candidate_answer(
        self,
        task: PublicTask,
        model_step: ModelStep,
        observation: dict[str, object],
        answer: AnswerTable,
    ) -> CandidateVerificationDecision | None:
        if not _task_uses_independent_verifier(task, self.config):
            return None
        if self.system_prompt != CODEACT_REACT_SYSTEM_PROMPT:
            return None
        if model_step.action != "execute_python":
            return None

        try:
            raw_response = self.model.complete(
                _build_candidate_verifier_messages(
                    task,
                    model_step,
                    observation,
                    answer,
                    stdout_chars=self.config.verifier_stdout_chars,
                )
            )
            return _parse_verifier_decision(raw_response)
        except Exception as exc:  # noqa: BLE001
            return CandidateVerificationDecision(
                verdict="uncertain",
                reasons=(f"verifier request failed: {exc}",),
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
                response_lower = raw_response.casefold()
                likely_truncated_answer = (
                    "answer:" in response_lower
                    and '"rows"' in response_lower
                    and raw_response.count("[") >= 20
                )
                recovery_instruction = (
                    "Return exactly one valid CodeAct step next. Do not repeat invalid text. "
                    "Use `Thought:` plus fenced `Code:` or `Thought:` plus `Answer:` fenced JSON."
                )
                if likely_truncated_answer:
                    recovery_instruction = (
                        "Your Answer JSON appears too long or truncated. Do not hand-write the full rows. "
                        "Run one compact Python step that recomputes the final table and prints exactly "
                        "`FINAL_TABLE_JSON:` followed by JSON with `columns` and `rows`."
                    )
                observation = {
                    "ok": False,
                    "error": str(exc),
                    "recovery_instruction": recovery_instruction,
                    "likely_truncated_answer": likely_truncated_answer,
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

                verifier_rejection_instruction = (
                    _latest_verifier_rejection_instruction(state)
                    if model_step.action == "answer"
                    else None
                )
                if verifier_rejection_instruction:
                    observation = {
                        "ok": False,
                        "independent_verifier_recheck": True,
                        "error": verifier_rejection_instruction,
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
                    stdout = str(content.get("output") or content.get("stdout") or "")
                    evidence_value = _find_marker_json_value(stdout, EVIDENCE_LEDGER_MARKERS)
                    if evidence_value is not None:
                        evidence_entries = _evidence_entries_from_value(evidence_value)
                        content["evidence_ledger_report"] = (
                            evidence_entries if evidence_entries else evidence_value
                        )
                    consistency_value = _find_marker_json_value(stdout, EVIDENCE_CONSISTENCY_MARKERS)
                    if consistency_value is not None:
                        consistency_entries = _evidence_entries_from_value(consistency_value)
                        content["evidence_consistency_report"] = (
                            consistency_entries if consistency_entries else consistency_value
                        )
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
                    final_answer_check = _final_answer_check(task, candidate_answer)
                    content["final_answer_check"] = final_answer_check
                    warnings = final_answer_check.get("warnings", [])
                    review_reasons: list[str] = []
                    if isinstance(warnings, list) and warnings:
                        review_reasons.extend(str(warning) for warning in warnings[:3])
                    missing_evidence = _missing_special_evidence_ids(
                        task,
                        state,
                        current_content=content,
                    )
                    if missing_evidence:
                        review_reasons.append(
                            "semantic evidence still needs explicit grounding: "
                            + ", ".join(missing_evidence[:4])
                        )
                    required_consistency_ids, consistency_warnings = _evidence_consistency_warnings(
                        task,
                        state,
                        current_content=content,
                    )
                    if consistency_warnings:
                        content["evidence_consistency_warnings"] = consistency_warnings
                    elif required_consistency_ids:
                        content["evidence_consistency_resolved"] = True

                    verifier_decision = None
                    if (
                        tool_result.ok
                        and not review_reasons
                    ):
                        verifier_decision = self._verify_candidate_answer(
                            task,
                            model_step,
                            observation,
                            candidate_answer,
                        )
                    if verifier_decision is not None:
                        content["independent_verifier"] = _verifier_decision_to_dict(verifier_decision)
                        if verifier_decision.verdict == "reject":
                            content["verifier_rejected"] = True
                            review_reasons.append(
                                "independent verifier rejected candidate: "
                                + "; ".join(verifier_decision.reasons[:3])
                            )
                        elif verifier_decision.verdict == "uncertain":
                            review_reasons.append(
                                "independent verifier was uncertain: "
                                + "; ".join(verifier_decision.reasons[:3])
                            )
                    verifier_verdict = (
                        verifier_decision.verdict
                        if verifier_decision is not None
                        else _content_verifier_verdict(content)
                    )
                    if len(candidate_answer.rows) <= 20 and verifier_verdict != "accept":
                        review_reasons.append("candidate has <=20 rows, so the model should confirm final shape")
                    if review_reasons:
                        if verifier_decision is not None and verifier_decision.verdict == "reject":
                            content["candidate_review_instruction"] = (
                                "Runtime independent-verifier review: a separate verifier rejected the "
                                "previous final-table candidate. Do not submit the same table. Run a compact "
                                "Python repair/check that addresses the verifier reasons, then print "
                                "`FINAL_TABLE_JSON:` only if the corrected computation supports it. "
                                f"Candidate preview: {_safe_json_dumps(_answer_preview(candidate_answer, max_rows=3))}. "
                                "Review reasons: "
                                + "; ".join(review_reasons)
                            )
                        elif consistency_warnings:
                            content["candidate_review_instruction"] = (
                                "Runtime evidence note: Python printed a plausible final table, but verified "
                                "semantic evidence was not explicitly tied to the final computation. This is "
                                "not a hard rejection by itself; submit if the code clearly used the rule/value/"
                                "formula, otherwise run one compact verification and print a corrected "
                                "`FINAL_TABLE_JSON:`. "
                                f"Candidate preview: {_safe_json_dumps(_answer_preview(candidate_answer, max_rows=3))}. "
                                "Review reasons: "
                                + "; ".join(review_reasons)
                            )
                        else:
                            content["candidate_review_instruction"] = (
                                "Runtime final-answer review: Python printed a plausible final table, "
                                "but the runtime is not auto-submitting it. Do not run more exploration just because of this. "
                                "If the candidate columns and rows exactly answer the question, submit `Answer:` "
                                "with that table now. If a warning identifies a real extra/missing column or wrong "
                                "row shape, or missing semantic evidence identifies an unresolved mapping/rule, "
                                "fix that evidence first and submit the corrected table. Candidate preview: "
                                f"{_safe_json_dumps(_answer_preview(candidate_answer, max_rows=3))}. "
                                "Review reasons: "
                                + "; ".join(review_reasons)
                            )
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
