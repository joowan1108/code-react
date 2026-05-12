from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
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


FINAL_RESULT_MARKERS = (
    "FINAL_TABLE_JSON:",
    "FINAL_RESULT:",
    "ANSWER_CANDIDATE:",
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
    stop_labels = "Thought|Reflexion|Action|Code|Answer|Final Answer|Observation"
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
    answer_step = _extract_answer_step(raw_response)
    if answer_step is not None:
        return answer_step

    code = _extract_code_block(raw_response)
    if code:
        return ModelStep(
            thought=_extract_thought(raw_response),
            action="execute_python",
            action_input={"code": code},
            raw_response=raw_response,
        )

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


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        history_steps = state.steps
        if self.config.prompt_history_steps > 0:
            history_steps = history_steps[-self.config.prompt_history_steps :]
        for step in history_steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(
                    role="user",
                    content=build_observation_prompt(
                        _render_observation_for_prompt(step.observation, self.config)
                    ),
                )
            )
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                model_step = parse_model_step(raw_response)
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
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
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
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

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
