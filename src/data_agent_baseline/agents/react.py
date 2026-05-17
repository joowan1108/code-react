from __future__ import annotations

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
        "Verify ties/all rows/distinctness/units and submit exactly the final "
        "requested columns.",
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
        observed.update(node_id for node_id, _, _ in PLANNING_NODES[:-1])
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
    if difficulty == "medium":
        node_records, current_focus, observed_nodes = _planning_status(
            state,
            fallback_answer,
            MEDIUM_PLANNING_NODES,
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
        lines.append(
            "Priority: verify real tables/columns for requested concepts, decide the exact final "
            "answer columns, then compute and submit with no helper columns. Do not follow the "
            "full multi-stage planning graph for medium tasks."
        )
        prompt_prefix = "\n".join(lines)
        snapshot = {
            "enabled": True,
            "mode": "medium_light",
            "difficulty": task.difficulty,
            "step_index": step_index,
            "max_steps": max_steps,
            "model_input_position": "front_after_system_before_task_prompt",
            "current_focus": current_focus,
            "observed_nodes": observed_nodes,
            "nodes": node_records,
            "prompt_prefix": prompt_prefix,
        }
        return prompt_prefix, snapshot

    node_records, current_focus, observed_nodes = _planning_status(state, fallback_answer)
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
    lines.append(
        "Priority: first link schema and decide exact final columns; then map "
        "question terms to real columns/values; submit as soon as the answer is "
        "supported, with no helper columns. Never assume schema-absent concepts; "
        "verify their concrete data source first."
    )
    prompt_prefix = "\n".join(lines)
    snapshot = {
        "enabled": True,
        "difficulty": task.difficulty,
        "step_index": step_index,
        "max_steps": max_steps,
        "model_input_position": "front_after_system_before_task_prompt",
        "current_focus": current_focus,
        "observed_nodes": observed_nodes,
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
                self._append_step(task, state, step_record)
                candidate_answer = self._candidate_answer_from_observation(task, observation)
                if candidate_answer is not None:
                    decision = _candidate_decision_from_observation(task, observation, candidate_answer)
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
