from __future__ import annotations

import json
import re
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
    prompt_history_steps: int = 4
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


@dataclass(frozen=True, slots=True)
class PendingCandidate:
    answer: AnswerTable
    reasons: tuple[str, ...]
    source_step_index: int


@dataclass(frozen=True, slots=True)
class SchemaDecision:
    decision: str
    columns: tuple[str, ...]
    reason: str = ""


SCHEMA_DECISION_ACTION = "__schema_decision__"

FINAL_RESULT_MARKERS = (
    "FINAL_TABLE_JSON:",
    "FINAL_RESULT:",
    "ANSWER_CANDIDATE:",
)

FINAL_AUDIT_MARKERS = (
    "FINAL_AUDIT_JSON:",
    "AUDIT_JSON:",
)

AUDIT_REASON_PREFIXES = (
    "calculation_audit",
    "audit_",
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
    stop_labels = "Thought|Reflexion|Action|Code|Output|Answer|Final Answer|SchemaDecision|Observation"
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


def _extract_schema_decision_step(raw_response: str) -> ModelStep | None:
    schema_text = _extract_labeled_text(raw_response, "SchemaDecision")
    if schema_text is None:
        try:
            payload = _load_single_json_object(_strip_json_fence(raw_response))
        except Exception:
            return None
        if "decision" not in payload:
            return None
    else:
        payload = _load_first_json_object(schema_text)

    decision = payload.get("decision")
    if not isinstance(decision, str) or not decision.strip():
        raise ValueError("SchemaDecision must include a non-empty decision string.")
    columns = payload.get("columns", [])
    if columns is None:
        columns = []
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        raise ValueError("SchemaDecision columns must be a list of strings.")
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    return ModelStep(
        thought=_extract_thought(raw_response),
        action=SCHEMA_DECISION_ACTION,
        action_input={
            "decision": decision.strip().lower(),
            "columns": columns,
            "reason": reason,
        },
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

    schema_decision_step = _extract_schema_decision_step(raw_response)
    if schema_decision_step is not None:
        return schema_decision_step

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


def _question_is_single_value(question: str) -> bool:
    q = question.lower()
    tokens = _split_identifier_tokens(question)
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


def _question_requires_tie_check(question: str) -> bool:
    tokens = _split_identifier_tokens(question)
    return bool(
        tokens
        & {
            "lowest",
            "highest",
            "minimum",
            "maximum",
            "min",
            "max",
            "least",
            "most",
            "best",
            "worst",
            "top",
            "smallest",
            "largest",
            "fewest",
        }
    )


def _stdout_mentions_tie_check(stdout: str) -> bool:
    lowered = stdout.lower()
    return any(
        phrase in lowered
        for phrase in (
            "tie",
            "ties",
            "all rows",
            "all matching",
            "all matches",
            "same minimum",
            "same maximum",
            "same lowest",
            "same highest",
            "including tied",
            "including ties",
        )
    )


def _contains_any(text: str, needles: set[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _question_requires_calculation_audit(question: str) -> bool:
    q = question.lower()
    tokens = _split_identifier_tokens(question)
    risky_tokens = {
        "average",
        "avg",
        "mean",
        "count",
        "distinct",
        "unique",
        "percentage",
        "percent",
        "ratio",
        "proportion",
        "total",
        "sum",
        "more",
        "less",
        "most",
        "least",
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "top",
        "last",
        "latest",
        "first",
        "list",
        "all",
        "which",
        "between",
        "before",
        "after",
        "monthly",
        "yearly",
    }
    risky_phrases = (
        "how many",
        "how much",
        "what percentage",
        "how many times",
        "more than",
        "less than",
        "at least",
        "at most",
        "greater than",
        "lower than",
        "during ",
    )
    return bool(tokens & risky_tokens) or any(phrase in q for phrase in risky_phrases)


def _flatten_audit_text(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_audit_text(child)}" for key, child in value.items())
    if isinstance(value, list):
        return " ".join(_flatten_audit_text(item) for item in value)
    return str(value)


def _audit_validation_reasons(task: PublicTask, audit: dict[str, object] | None) -> tuple[str, ...]:
    if not _question_requires_calculation_audit(task.question):
        return ()

    if audit is None:
        return ("calculation_audit_required",)

    reasons: list[str] = []
    q = task.question.lower()
    tokens = _split_identifier_tokens(task.question)
    audit_text = _flatten_audit_text(audit).lower()

    if tokens & {"percentage", "percent", "ratio", "proportion"} or "how many times" in q:
        if not _contains_any(audit_text, {"numerator", "denominator", "divide", "ratio", "percentage"}):
            reasons.append("audit_missing_numerator_denominator")

    if tokens & {"average", "avg", "mean"}:
        if not _contains_any(audit_text, {"average", "avg", "mean", "sum", "count"}):
            reasons.append("audit_missing_average_basis")

    if tokens & {"count", "many", "number"} or "how many" in q:
        if not _contains_any(audit_text, {"count", "row_count", "rows", "distinct", "unique"}):
            reasons.append("audit_missing_count_basis")

    if tokens & {"distinct", "unique", "different"}:
        if not _contains_any(audit_text, {"distinct", "unique", "dedup", "duplicate"}):
            reasons.append("audit_missing_distinct_check")

    if "monthly" in tokens or "per month" in q:
        if not _contains_any(audit_text, {"monthly", "month", "/ 12", "divide by 12", "divided by 12"}):
            reasons.append("audit_missing_monthly_unit_conversion")

    if _question_requires_tie_check(task.question):
        if not _contains_any(audit_text, {"tie", "ties", "all matching", "all rows", "rank", "sort"}):
            reasons.append("audit_missing_tie_check")

    if _question_expects_nonempty_rows(task.question):
        if not _contains_any(audit_text, {"filter", "condition", "row_count", "rows", "result_count"}):
            reasons.append("audit_missing_filter_trace")

    if tokens & {"last", "latest", "recent", "first"}:
        if not _contains_any(audit_text, {"last", "latest", "first", "sort", "order", "date", "time"}):
            reasons.append("audit_missing_ordering_check")

    return tuple(reasons)


def _has_audit_reason(reasons: tuple[str, ...]) -> bool:
    return any(reason.startswith(AUDIT_REASON_PREFIXES) for reason in reasons)


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


def _sanitize_answer_table(task: PublicTask, answer: AnswerTable) -> AnswerTable:
    _ = task
    return _normalize_answer_table(answer)


def _answer_needs_schema_validation(task: PublicTask, answer: AnswerTable) -> bool:
    _ = task
    return len(answer.columns) > 1


def _project_answer_table(answer: AnswerTable, columns: tuple[str, ...]) -> AnswerTable:
    index_by_column = {column: index for index, column in enumerate(answer.columns)}
    indices = [index_by_column[column] for column in columns]
    return AnswerTable(
        columns=list(columns),
        rows=[[row[index] for index in indices] for row in answer.rows],
    )


def _parse_schema_decision_payload(
    action_input: dict[str, object],
    pending_candidate: PendingCandidate,
) -> SchemaDecision:
    decision = action_input.get("decision")
    if not isinstance(decision, str):
        raise ValueError("SchemaDecision decision must be a string.")
    normalized_decision = decision.strip().lower()
    if normalized_decision not in {"accept", "select_columns", "reject"}:
        raise ValueError("SchemaDecision decision must be accept, select_columns, or reject.")

    reason = action_input.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    if normalized_decision == "accept":
        return SchemaDecision(
            decision=normalized_decision,
            columns=tuple(pending_candidate.answer.columns),
            reason=reason,
        )

    if normalized_decision == "reject":
        return SchemaDecision(decision=normalized_decision, columns=(), reason=reason)

    columns = action_input.get("columns")
    if not isinstance(columns, list) or not columns or not all(isinstance(column, str) for column in columns):
        raise ValueError("select_columns SchemaDecision must include a non-empty columns list.")

    available_columns = set(pending_candidate.answer.columns)
    unknown_columns = [column for column in columns if column not in available_columns]
    if unknown_columns:
        raise ValueError(f"SchemaDecision selected unknown columns: {unknown_columns}")

    return SchemaDecision(
        decision=normalized_decision,
        columns=tuple(columns),
        reason=reason,
    )


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


def _audit_payload_from_observation(observation: dict[str, object]) -> dict[str, object] | None:
    if observation.get("tool") != "execute_python":
        return None
    content = observation.get("content")
    if not isinstance(content, dict):
        return None
    stdout = str(content.get("output") or content.get("stdout") or "")
    return _find_marker_payload(stdout, FINAL_AUDIT_MARKERS)


def _latest_audit_payload_from_steps(steps: list[StepRecord]) -> dict[str, object] | None:
    for step in reversed(steps):
        audit = _audit_payload_from_observation(step.observation)
        if audit is not None:
            return audit
    return None


def _candidate_decision_from_observation(
    task: PublicTask,
    observation: dict[str, object],
    answer: AnswerTable,
) -> CandidateAnswerDecision:
    reasons: list[str] = []
    content = observation.get("content")
    content_dict = content if isinstance(content, dict) else {}
    stdout = str(content_dict.get("output") or content_dict.get("stdout") or "")
    stderr = str(content_dict.get("stderr") or "")
    traceback_text = str(content_dict.get("traceback") or "")

    if traceback_text or "traceback" in stderr.lower():
        reasons.append("python_traceback_present")

    if _question_expects_nonempty_rows(task.question) and not answer.rows:
        reasons.append("question_expects_rows_but_candidate_is_empty")

    if _question_is_single_value(task.question) and (len(answer.rows) != 1 or len(answer.columns) != 1):
        reasons.append("single_value_question_requires_one_cell_answer")

    if _answer_needs_schema_validation(task, answer):
        reasons.append("schema_validation_required")

    if (
        _question_requires_tie_check(task.question)
        and len(answer.rows) <= 1
        and not _stdout_mentions_tie_check(stdout)
    ):
        reasons.append("min_max_or_top_question_needs_explicit_tie_check")

    if len(answer.columns) > 3 and not _question_is_single_value(task.question):
        reasons.append("candidate_has_many_columns_verify_exact_schema")

    audit = _audit_payload_from_observation(observation)
    reasons.extend(_audit_validation_reasons(task, audit))

    store_as_fallback = bool(answer.rows) and "python_traceback_present" not in reasons
    return CandidateAnswerDecision(
        answer=answer,
        auto_submit=not reasons,
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
    for marker in FINAL_AUDIT_MARKERS:
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


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        prompt_tool_names: tuple[str, ...] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.prompt_tool_names = prompt_tool_names

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        runtime_instruction: str | None = None,
    ) -> list[ModelMessage]:
        is_codeact = self.system_prompt == CODEACT_REACT_SYSTEM_PROMPT
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(self.prompt_tool_names),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
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
        pending_candidate: PendingCandidate | None,
        consecutive_parse_errors: int,
    ) -> str | None:
        if self.system_prompt != CODEACT_REACT_SYSTEM_PROMPT:
            return None

        if pending_candidate is not None:
            return (
                "Semantic schema validation required. Do not run Python and do not recompute the answer.\n"
                f"Question: {task.question}\n"
                f"Candidate answer preview: {_safe_json_dumps(_answer_preview(pending_candidate.answer, max_rows=5))}\n"
                f"Runtime concerns: {', '.join(pending_candidate.reasons)}\n\n"
                "Choose columns by semantic meaning, using the question and the sample values. "
                "Keep columns that are directly needed to answer the question; remove helper columns such as ids, dates, ranks, "
                "scores, or costs only when they are not part of the requested answer. "
                "If multiple columns are semantically requested, keep all of them. "
                "If the candidate row set or calculation looks wrong, reject it.\n\n"
                "Return exactly one schema decision in this format:\n"
                "SchemaDecision:\n"
                "```json\n"
                "{\"decision\":\"accept\",\"reason\":\"columns answer the question\"}\n"
                "```\n"
                "or\n"
                "SchemaDecision:\n"
                "```json\n"
                "{\"decision\":\"select_columns\",\"columns\":[\"column_name\"],\"reason\":\"only this column is requested\"}\n"
                "```\n"
                "or\n"
                "SchemaDecision:\n"
                "```json\n"
                "{\"decision\":\"reject\",\"reason\":\"candidate does not answer the question\"}\n"
                "```"
            )

        remaining_steps = self.config.max_steps - step_index + 1
        instructions: list[str] = []
        if consecutive_parse_errors:
            instructions.append(
                "Runtime recovery: your previous response could not be parsed. "
                "Do not repeat Thought-only text or <think> tags. Return exactly one valid step: "
                "either `Thought:` plus a fenced `Code:` block, or `Thought:` plus `Answer:` with fenced JSON."
            )
        if remaining_steps <= 2:
            instructions.append(
                f"Finalization required: only {remaining_steps} step(s) remain. "
                "Stop broad exploration. Prefer submitting the best exact answer now. "
                "Use only the requested columns, include all tied/all matching rows, and check units such as monthly vs yearly."
            )
            if remaining_steps == 1:
                instructions.append(
                    "This is the last step. Do not perform another exploratory inspection. "
                    "Submit `Answer` now unless a single short Python action can print `FINAL_TABLE_JSON` directly."
                )
        if fallback_answer is not None and remaining_steps <= 2:
            instructions.append(
                "A previous candidate answer is available as fallback. If it matches the question, submit it exactly:\n"
                f"{_safe_json_dumps(_answer_preview(fallback_answer, max_rows=5))}"
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
        fallback_answer: AnswerTable | None = None
        pending_candidate: PendingCandidate | None = None
        consecutive_parse_errors = 0
        for step_index in range(1, self.config.max_steps + 1):
            try:
                raw_response = self.model.complete(
                    self._build_messages(
                        task,
                        state,
                        runtime_instruction=self._build_runtime_instruction(
                            task=task,
                            step_index=step_index,
                            state=state,
                            fallback_answer=fallback_answer,
                            pending_candidate=pending_candidate,
                            consecutive_parse_errors=consecutive_parse_errors,
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                state.steps.append(
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
                state.failure_reason = f"Model request failed before step {step_index}: {exc}"
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
                state.steps.append(
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
            if pending_candidate is not None:
                if model_step.action != SCHEMA_DECISION_ACTION:
                    observation = {
                        "ok": False,
                        "error": (
                            "A schema decision is required before continuing. "
                            "Return `SchemaDecision:` with accept, select_columns, or reject."
                        ),
                        "candidate_preview": _answer_preview(pending_candidate.answer, max_rows=5),
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action="__error__",
                            action_input={},
                            raw_response=raw_response,
                            observation=observation,
                            ok=False,
                        )
                    )
                    continue

                try:
                    schema_decision = _parse_schema_decision_payload(
                        model_step.action_input,
                        pending_candidate,
                    )
                    if schema_decision.decision == "reject":
                        observation = {
                            "ok": True,
                            "tool": SCHEMA_DECISION_ACTION,
                            "content": {
                                "decision": schema_decision.decision,
                                "reason": schema_decision.reason,
                                "next_action": (
                                    "Candidate rejected by semantic schema validation. "
                                    "Recompute or print a corrected FINAL_TABLE_JSON."
                                ),
                            },
                        }
                        state.steps.append(
                            StepRecord(
                                step_index=step_index,
                                thought=model_step.thought,
                                action=model_step.action,
                                action_input=model_step.action_input,
                                raw_response=raw_response,
                                observation=observation,
                                ok=True,
                            )
                        )
                        pending_candidate = None
                        fallback_answer = None
                        continue

                    projected_answer = _project_answer_table(
                        pending_candidate.answer,
                        schema_decision.columns,
                    )
                    if _has_audit_reason(pending_candidate.reasons):
                        fallback_answer = projected_answer
                        observation = {
                            "ok": True,
                            "tool": SCHEMA_DECISION_ACTION,
                            "content": {
                                "decision": schema_decision.decision,
                                "selected_columns": list(schema_decision.columns),
                                "reason": schema_decision.reason,
                                "candidate_preview": _answer_preview(projected_answer, max_rows=5),
                                "next_action": (
                                    "Schema accepted, but runtime still requires calculation audit. "
                                    "Run a focused verification and print FINAL_AUDIT_JSON plus FINAL_TABLE_JSON."
                                ),
                            },
                        }
                        state.steps.append(
                            StepRecord(
                                step_index=step_index,
                                thought=model_step.thought,
                                action=model_step.action,
                                action_input=model_step.action_input,
                                raw_response=raw_response,
                                observation=observation,
                                ok=True,
                            )
                        )
                        pending_candidate = None
                        continue

                    state.answer = projected_answer
                    observation = {
                        "ok": True,
                        "tool": SCHEMA_DECISION_ACTION,
                        "content": {
                            "decision": schema_decision.decision,
                            "selected_columns": list(schema_decision.columns),
                            "reason": schema_decision.reason,
                            "final_preview": _answer_preview(state.answer, max_rows=5),
                        },
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=True,
                        )
                    )
                    break
                except Exception as exc:
                    observation = {
                        "ok": False,
                        "error": str(exc),
                        "recovery_instruction": (
                            "Return a valid SchemaDecision using only candidate column names."
                        ),
                        "candidate_preview": _answer_preview(pending_candidate.answer, max_rows=5),
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action="__error__",
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=False,
                        )
                    )
                    continue

            try:
                model_step = self._sanitize_model_step(task, model_step)
                if (
                    self.system_prompt == CODEACT_REACT_SYSTEM_PROMPT
                    and model_step.action == "answer"
                ):
                    answer = _answer_table_from_payload(model_step.action_input)
                    if answer is not None:
                        answer = _normalize_answer_table(answer)
                        answer_audit = _latest_audit_payload_from_steps(state.steps)
                        answer_audit_reasons = _audit_validation_reasons(task, answer_audit)
                        if _answer_needs_schema_validation(task, answer):
                            pending_candidate = PendingCandidate(
                                answer=answer,
                                reasons=("schema_validation_required", *answer_audit_reasons),
                                source_step_index=step_index,
                            )
                            fallback_answer = answer
                            observation = {
                                "ok": True,
                                "tool": "runtime_schema_review",
                                "content": {
                                    "candidate_preview": _answer_preview(answer, max_rows=5),
                                    "audit_present": answer_audit is not None,
                                    "audit_reasons": list(answer_audit_reasons),
                                    "next_action": (
                                        "Return SchemaDecision next. Accept this schema, select only "
                                        "semantically requested columns, or reject the candidate."
                                    ),
                                },
                            }
                            state.steps.append(
                                StepRecord(
                                    step_index=step_index,
                                    thought=model_step.thought,
                                    action="__schema_review__",
                                    action_input=answer.to_dict(),
                                    raw_response=raw_response,
                                    observation=observation,
                                    ok=True,
                                )
                            )
                            continue
                        if answer_audit_reasons:
                            fallback_answer = answer
                            observation = {
                                "ok": True,
                                "tool": "runtime_audit_review",
                                "content": {
                                    "candidate_preview": _answer_preview(answer, max_rows=5),
                                    "audit_present": answer_audit is not None,
                                    "audit_reasons": list(answer_audit_reasons),
                                    "next_action": (
                                        "Runtime needs calculation audit before direct Answer. "
                                        "Run focused verification and print FINAL_AUDIT_JSON plus FINAL_TABLE_JSON."
                                    ),
                                },
                            }
                            state.steps.append(
                                StepRecord(
                                    step_index=step_index,
                                    thought=model_step.thought,
                                    action="__audit_review__",
                                    action_input=answer.to_dict(),
                                    raw_response=raw_response,
                                    observation=observation,
                                    ok=True,
                                )
                            )
                            continue

                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                content = dict(tool_result.content)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                candidate_answer = self._candidate_answer_from_observation(task, observation)
                if candidate_answer is not None:
                    decision = _candidate_decision_from_observation(task, observation, candidate_answer)
                    requires_schema_validation = (
                        "schema_validation_required" in decision.reasons
                        and "python_traceback_present" not in decision.reasons
                        and bool(decision.answer.rows)
                    )
                    requires_audit_validation = (
                        _has_audit_reason(decision.reasons)
                        and "python_traceback_present" not in decision.reasons
                        and bool(decision.answer.rows)
                    )
                    audit_payload = _audit_payload_from_observation(observation)
                    content["final_table_json_decision"] = {
                        "auto_submit": decision.auto_submit,
                        "store_as_fallback": decision.store_as_fallback,
                        "reasons": list(decision.reasons),
                        "candidate_preview": _answer_preview(decision.answer),
                        "audit_present": audit_payload is not None,
                        "audit_preview": audit_payload,
                        "next_action": (
                            "Runtime accepted this FINAL_TABLE_JSON as the final answer."
                            if decision.auto_submit
                            else (
                                "Runtime requires semantic schema validation next. Return SchemaDecision "
                                "to accept, select requested columns, or reject this candidate."
                                if requires_schema_validation
                                else (
                                    "Runtime requires calculation audit next. Run focused verification and print "
                                    "FINAL_AUDIT_JSON plus a corrected FINAL_TABLE_JSON."
                                    if requires_audit_validation
                                    else (
                                        "Runtime did not auto-submit this candidate. Fix the listed issues, "
                                        "verify all filters/ties/schema, then print a corrected FINAL_TABLE_JSON or submit Answer."
                                    )
                                )
                            )
                        ),
                    }
                    if decision.store_as_fallback:
                        fallback_answer = decision.answer
                    if requires_schema_validation:
                        pending_candidate = PendingCandidate(
                            answer=decision.answer,
                            reasons=decision.reasons,
                            source_step_index=step_index,
                        )
                        continue
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
                state.steps.append(
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
