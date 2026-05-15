from __future__ import annotations

import json
import multiprocessing
import sqlite3
import tempfile
from pathlib import Path

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.manifest import build_context_manifest
from data_agent_baseline.agents.prompt import CODEACT_REACT_SYSTEM_PROMPT
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
from data_agent_baseline.tools.knowledge import retrieve_knowledge_snippets
from data_agent_baseline.tools.python_exec import execute_python_code
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
    temp_context = tempfile.TemporaryDirectory()
    context_path = Path(temp_context.name)
    (context_path / "data.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    task = PublicTask(
        record=TaskRecord(task_id="task_x", difficulty="easy", question="Return x."),
        assets=TaskAssets(task_dir=context_path, context_dir=context_path),
    )
    state = AgentRuntimeState()
    try:
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
            config=ReActAgentConfig(prompt_history_steps=5),
        )
        messages = agent._build_messages(task, state)
        assistant_messages = [message.content for message in messages if message.role == "assistant"]
        joined_messages = "\n".join(message.content for message in messages)
        assert len(assistant_messages) == 5
        assert "step 1" not in joined_messages
        assert "output 2" in joined_messages
        assert "step 3" in joined_messages
        assert "output 6" in joined_messages
    finally:
        temp_context.cleanup()


def test_runtime_instruction_does_not_report_current_step() -> None:
    temp_context = tempfile.TemporaryDirectory()
    context_path = Path(temp_context.name)
    task = PublicTask(
        record=TaskRecord(task_id="task_x", difficulty="easy", question="Return x."),
        assets=TaskAssets(task_dir=context_path, context_dir=context_path),
    )
    try:
        agent = ReActAgent(
            model=ScriptedModelAdapter([]),
            tools=ToolRegistry(specs={}, handlers={}),
            config=ReActAgentConfig(max_steps=16),
            system_prompt=CODEACT_REACT_SYSTEM_PROMPT,
        )
        instruction = agent._build_runtime_instruction(
            task=task,
            step_index=5,
            state=AgentRuntimeState(),
            fallback_answer=None,
            consecutive_parse_errors=0,
        )
        assert instruction is None

        recovery_instruction = agent._build_runtime_instruction(
            task=task,
            step_index=15,
            state=AgentRuntimeState(),
            fallback_answer=None,
            consecutive_parse_errors=1,
        )
        assert recovery_instruction is not None
        assert "Runtime recovery" in recovery_instruction
        assert "Progress: step" not in recovery_instruction
        assert "Finalization window" not in recovery_instruction
    finally:
        temp_context.cleanup()


def test_context_manifest_summarizes_structured_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "csv").mkdir()
        (context_path / "json").mkdir()
        (context_path / "knowledge.md").write_text("# Knowledge\nDetails", encoding="utf-8")
        (context_path / "csv" / "members.csv").write_text(
            "member_id,name,major\n1,Ada,CS\n",
            encoding="utf-8",
        )
        (context_path / "json" / "events.json").write_text(
            json.dumps(
                {
                    "table": "event",
                    "records": [
                        {"event_id": "e1", "event_name": "Talk", "cost": 10},
                        {"event_id": "e2", "event_name": "Meetup", "cost": 20},
                    ],
                }
            ),
            encoding="utf-8",
        )
        sqlite_path = context_path / "club.sqlite"
        connection = sqlite3.connect(sqlite_path)
        try:
            connection.executescript(
                """
                CREATE TABLE member (
                    member_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT 'unknown',
                    major TEXT
                );
                CREATE TABLE attendance (
                    member_id INTEGER NOT NULL,
                    event_id INTEGER NOT NULL,
                    FOREIGN KEY(member_id) REFERENCES member(member_id)
                );
                INSERT INTO member VALUES (1, 'Ada', 'CS');
                """
            )
            connection.commit()
        finally:
            connection.close()

        manifest = build_context_manifest(context_path)
        assert "Concise context manifest" in manifest
        assert "csv/members.csv" in manifest
        assert "member_id, name, major" in manifest
        assert "Ada" not in manifest
        assert "json/events.json" in manifest
        assert "table=event" in manifest
        assert "event_id, event_name, cost" in manifest
        assert "Talk" not in manifest
        assert "knowledge.md" in manifest
        assert "Details" not in manifest
        assert "club.sqlite" in manifest
        assert "table member" in manifest
        assert "member_id INTEGER (pk)" in manifest
        assert "name TEXT (notnull)" in manifest
        assert "foreign_keys=member_id->member.member_id" in manifest
        assert "create_sql=" not in manifest
        assert "default=unknown" not in manifest


def test_retrieve_knowledge_uses_full_document_and_schema_terms() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "csv").mkdir()
        (context_path / "csv" / "exams.csv").write_text(
            "ID,Thrombosis,Creatinine\n1,2,1.4\n",
            encoding="utf-8",
        )
        filler = "\n\n".join(f"Unrelated paragraph {index}." for index in range(80))
        (context_path / "knowledge.md").write_text(
            filler
            + "\n\n## Thrombosis values\n"
            + "The Thrombosis column uses coded values. Value 2 indicates severe thrombosis.",
            encoding="utf-8",
        )

        snippets = retrieve_knowledge_snippets(
            context_path,
            "List patients with severe disease.",
            top_k=2,
        )

        rendered = json.dumps(snippets, ensure_ascii=False)
        assert "Value 2 indicates severe thrombosis" in rendered
        assert "thrombosis" in snippets[0]["matched_schema_terms"]


def test_python_namespace_exposes_retrieve_knowledge_with_task_question() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "data.csv").write_text("ID,Thrombosis\n1,2\n", encoding="utf-8")
        (context_path / "knowledge.md").write_text(
            "Thrombosis value 2 means severe thrombosis.",
            encoding="utf-8",
        )
        result = execute_python_code(
            context_path,
            "print(task_question)\nprint(retrieve_knowledge(top_k=1))",
            timeout_seconds=10,
            question="Which patients have severe thrombosis?",
        )
        assert result["success"]
        assert "Which patients have severe thrombosis?" in result["output"]
        assert "value 2 means severe thrombosis" in result["output"]


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
        partial_trace_path = artifact.task_output_dir / "trace.partial.json"
        assert partial_trace_path.exists()
        partial_trace = json.loads(partial_trace_path.read_text(encoding="utf-8"))
        assert partial_trace["task_id"] == "task_11"
        assert partial_trace["steps"]
        score = score_prediction_csv(
            artifact.prediction_csv_path,
            Path("data/public/output/task_11/gold.csv"),
        )
        assert score.score == 1.0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    test_codeact_scripted_task_11()
    print("ok")
