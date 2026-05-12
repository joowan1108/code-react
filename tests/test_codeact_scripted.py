from __future__ import annotations

import json
import multiprocessing
import tempfile
from pathlib import Path

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.react import (
    ReActAgent,
    ReActAgentConfig,
    _render_observation_for_prompt,
    parse_model_step,
)
from data_agent_baseline.agents.runtime import AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.evaluate import score_prediction_csv
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import run_single_task
from data_agent_baseline.tools.registry import ToolRegistry


def _json_response(payload: dict[str, object]) -> str:
    return "```json\n" + json.dumps(payload) + "\n```"


def _code_response(thought: str, code: str) -> str:
    return f"Thought: {thought}\nCode:\n```python\n{code.strip()}\n```"


def _answer_response(thought: str, payload: dict[str, object]) -> str:
    return f"Thought: {thought}\nAnswer:\n" + _json_response(payload)


SCRIPTED_TASK_11_CODE = """
import json
from pathlib import Path
import pandas as pd

ctx = Path(".")
patients = pd.DataFrame(json.loads((ctx / "json" / "Patient.json").read_text(encoding="utf-8"))["records"])
exams = pd.DataFrame(json.loads((ctx / "json" / "Examination.json").read_text(encoding="utf-8"))["records"])
ids = exams.loc[exams["Thrombosis"].eq(2), "ID"].dropna().astype(int).unique()
answer = patients.loc[patients["ID"].isin(ids), ["ID", "SEX", "Diagnosis"]].sort_values("ID")
print(answer.to_json(orient="records"))
"""


def test_parse_codeact_python_block() -> None:
    step = parse_model_step(
        _code_response(
            "I will inspect files using Python.",
            "from pathlib import Path\nprint(sorted(p.name for p in Path('.').iterdir()))",
        )
    )
    assert step.action == "execute_python"
    assert "Path" in step.action_input["code"]


def test_parse_codeact_answer_block() -> None:
    step = parse_model_step(
        _answer_response(
            "I have the requested result.",
            {"columns": ["value"], "rows": [["42"]]},
        )
    )
    assert step.action == "answer"
    assert step.action_input == {"columns": ["value"], "rows": [["42"]]}


def test_compact_observation_preserves_final_marker_head_and_tail() -> None:
    observation = {
        "ok": True,
        "tool": "execute_python",
        "content": {
            "success": True,
            "output": (
                "HEAD-" + ("a" * 120) + "\n"
                "middle-" + ("b" * 500) + "\n"
                "FINAL_TABLE_JSON:\n{\"columns\":[\"x\"],\"rows\":[[\"42\"]]}\n"
                "TAIL-" + ("c" * 120)
            ),
            "stderr": "",
        },
    }
    rendered = _render_observation_for_prompt(
        observation,
        ReActAgentConfig(
            full_observation_threshold=100,
            observation_head_chars=60,
            observation_tail_chars=60,
            final_marker_chars=120,
        ),
    )
    assert "compact_observation: true" in rendered
    assert "stdout_head" in rendered
    assert "stdout_tail" in rendered
    assert "FINAL_TABLE_JSON" in rendered
    assert "omitted" in rendered


def test_prompt_history_uses_recent_steps_only() -> None:
    task = PublicTask(
        record=TaskRecord(task_id="task_x", difficulty="easy", question="Return x."),
        assets=TaskAssets(task_dir=Path("."), context_dir=Path(".")),
    )
    state = AgentRuntimeState()
    for index in range(1, 7):
        state.steps.append(
            StepRecord(
                step_index=index,
                thought=f"thought {index}",
                action="execute_python",
                action_input={"code": f"print({index})"},
                raw_response=f"Thought: step {index}\nCode:\n```python\nprint({index})\n```",
                observation={
                    "ok": True,
                    "tool": "execute_python",
                    "content": {"success": True, "output": f"output {index}", "stderr": ""},
                },
                ok=True,
            )
        )

    agent = ReActAgent(
        model=ScriptedModelAdapter([]),
        tools=ToolRegistry(specs={}, handlers={}),
        config=ReActAgentConfig(prompt_history_steps=4),
    )
    messages = agent._build_messages(task, state)
    assistant_messages = [message.content for message in messages if message.role == "assistant"]
    joined_messages = "\n".join(message.content for message in messages)
    assert len(assistant_messages) == 4
    assert "step 1" not in joined_messages
    assert "output 2" not in joined_messages
    assert "step 3" in joined_messages
    assert "output 6" in joined_messages


def test_codeact_scripted_task_11() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        config = AppConfig(
            dataset=DatasetConfig(root_path=Path("data/public/input")),
            agent=AgentConfig(
                strategy="codeact",
                model="scripted",
                api_base="none",
                api_key="none",
                max_steps=2,
            ),
            run=RunConfig(
                output_dir=Path(temp_dir),
                run_id=None,
                max_workers=1,
                task_timeout_seconds=0,
            ),
        )
        artifact = run_single_task(
            task_id="task_11",
            config=config,
            run_output_dir=Path(temp_dir),
            model=ScriptedModelAdapter(
                [
                    _code_response(
                        "I will use Python to join severe thrombosis rows to patient demographics.",
                        SCRIPTED_TASK_11_CODE,
                    ),
                    _answer_response(
                        "The Python output has the requested final rows, so I will submit them.",
                        {
                            "columns": ["ID", "SEX", "Diagnosis"],
                            "rows": [
                                ["163109", "F", "SLE"],
                                ["2803470", "F", "SLE"],
                                ["4395720", "F", "SLE"],
                            ],
                        }
                    ),
                ]
            ),
        )
        assert artifact.succeeded
        assert artifact.prediction_csv_path is not None
        score = score_prediction_csv(
            artifact.prediction_csv_path,
            Path("data/public/output/task_11/gold.csv"),
        )
        assert score.score == 1.0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    test_codeact_scripted_task_11()
    print("ok")
