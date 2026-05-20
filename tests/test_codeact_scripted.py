from __future__ import annotations

import json
import multiprocessing
import sqlite3
import tempfile
from pathlib import Path

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.manifest import build_context_manifest
from data_agent_baseline.agents.prompt import CODEACT_REACT_SYSTEM_PROMPT, build_task_prompt
from data_agent_baseline.agents.react import (
    ReActAgent,
    ReActAgentConfig,
    _build_planning_context,
    _candidate_decision_from_observation,
    _evidence_consistency_warnings,
    _final_answer_check,
    _python_repair_hints,
    _render_observation_for_prompt,
    _sanitize_answer_table,
    parse_model_step,
)
from data_agent_baseline.agents.runtime import AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.evaluate import score_prediction_csv
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import run_single_task
from data_agent_baseline.tools.knowledge import retrieve_knowledge_snippets
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


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
        observation_messages = [
            message.content for message in messages if message.role == "user" and "Observation:" in message.content
        ]
        joined_messages = "\n".join(message.content for message in messages)
        assert len(assistant_messages) == 5
        assert len(observation_messages) == 5
        assert "step 1" not in joined_messages
        assert "output 1" not in joined_messages
        assert "step 2" in joined_messages
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


def test_python_repair_hints_for_common_runtime_errors() -> None:
    hints = _python_repair_hints(
        {
            "success": False,
            "error": "Cannot operate on a closed database.",
            "traceback": "sqlite3.ProgrammingError: Cannot operate on a closed database.",
        },
        "conn.close()\ncursor.execute('select 1')",
    )
    assert any("closed" in hint.lower() for hint in hints)

    import_hints = _python_repair_hints(
        {
            "success": False,
            "error": "No module named 'tools'",
            "traceback": "ModuleNotFoundError: No module named 'tools'",
        },
        "from tools import retrieve_knowledge\nretrieve_knowledge()",
    )
    assert any("load_json_table" in hint for hint in import_hints)


def test_python_repair_hints_for_json_table_records_wrapper() -> None:
    hints = _python_repair_hints(
        {
            "success": True,
            "output": "Gasstations columns: ['table', 'records']\nKeyError: 'GasStationID'",
        },
        "import pandas as pd\ngas = pd.read_json('json/gasstations.json')\nprint(gas.columns.tolist())",
    )

    assert any("table" in hint and "records" in hint and "load_json_table" in hint for hint in hints)


def test_final_answer_check_warns_about_likely_extra_columns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_extra",
                difficulty="medium",
                question="List the names of schools.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        check = _final_answer_check(
            task,
            AnswerTable(
                columns=["School Name", "debug_score"],
                rows=[["A", 0.9]],
            ),
        )

        assert check["question_type"] == "row_list"
        assert check["warnings"]
        assert "debug_score" in check["warnings"][0]


def test_sanitize_answer_keeps_obvious_single_target_helper_columns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_event",
                difficulty="easy",
                question="Which event has the lowest cost?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["event_name", "cost"],
                rows=[["November Speaker", 6.0], ["October Speaker", 6.0]],
            ),
        )

        assert answer.columns == ["event_name", "cost"]
        assert answer.rows == [["November Speaker", 6.0], ["October Speaker", 6.0]]


def test_sanitize_answer_keeps_comment_helper_columns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_comment",
                difficulty="medium",
                question="Among the posts with views ranging from 100 to 150, what is the comment with the highest score?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["Id", "PostId", "Score", "Text"],
                rows=[[90813, 46764, 14, "Welcome to Cross Validated."]],
            ),
        )

        assert answer.columns == ["Id", "PostId", "Score", "Text"]
        assert answer.rows == [[90813, 46764, 14, "Welcome to Cross Validated."]]


def test_sanitize_answer_keeps_single_value_metric_context_columns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_amount",
                difficulty="easy",
                question="What is the amount of the funds that the Vice President received?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["member_id", "first_name", "last_name", "position", "total_amount_received"],
                rows=[["recD078", "Phillip", "Cullen", "Vice President", 50]],
            ),
        )

        assert answer.columns == ["member_id", "first_name", "last_name", "position", "total_amount_received"]
        assert answer.rows == [["recD078", "Phillip", "Cullen", "Vice President", 50]]


def test_sanitize_answer_keeps_entity_metric_and_currency_columns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_customer_metric",
                difficulty="medium",
                question=(
                    "Who is the top spending customer and how much is the average price per "
                    "single item purchased by this customer? What currency was being used?"
                ),
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["CustomerID", "avg_price_per_item", "Currency"],
                rows=[["C-001", 9.5, "USD"]],
            ),
        )

        assert answer.columns == ["CustomerID", "avg_price_per_item", "Currency"]
        assert answer.rows == [["C-001", 9.5, "USD"]]


def test_sanitize_answer_keeps_member_context_columns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_members",
                difficulty="medium",
                question='List all the members of the "School of Applied Sciences, Technology and Education" department.',
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["member_id", "first_name", "last_name", "email", "position"],
                rows=[["rec1", "Angela", "Sanders", "a@example.com", "Member"]],
            ),
        )

        assert answer.columns == ["member_id", "first_name", "last_name", "email", "position"]
        assert answer.rows == [["rec1", "Angela", "Sanders", "a@example.com", "Member"]]


def test_sanitize_answer_keeps_explicit_member_contact_column() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_member_phone",
                difficulty="medium",
                question="List the students in the Chess Club and their phone numbers.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["first_name", "last_name", "phone"],
                rows=[["Angela", "Sanders", "555-0101"]],
            ),
        )

        assert answer.columns == ["first_name", "last_name", "phone"]
        assert answer.rows == [["Angela", "Sanders", "555-0101"]]


def test_sanitize_answer_keeps_specific_and_full_name_columns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_names",
                difficulty="medium",
                question="What are the names of the superheroes with the power of death touch?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["superhero_name", "full_name"],
                rows=[["Black Flash", "-"], ["Blackwulf", "Lucian"]],
            ),
        )

        assert answer.columns == ["superhero_name", "full_name"]
        assert answer.rows == [["Black Flash", "-"], ["Blackwulf", "Lucian"]]


def test_sanitize_answer_keeps_explicitly_requested_columns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_multi",
                difficulty="easy",
                question="For patients with severe degree of thrombosis, list their ID, sex and disease the patient is diagnosed with.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["ID", "SEX", "Diagnosis"],
                rows=[["163109", "F", "SLE"]],
            ),
        )

        assert answer.columns == ["ID", "SEX", "Diagnosis"]
        assert answer.rows == [["163109", "F", "SLE"]]


def test_sanitize_answer_drops_incomplete_row_list_rows() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_null_rows",
                difficulty="easy",
                question="For patients with severe degree of thrombosis, list their ID, sex and disease the patient is diagnosed with.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["ID", "SEX", "Diagnosis"],
                rows=[
                    ["163109", "F", "SLE"],
                    ["1430760", "nan", "nan"],
                    ["2803470", "F", "SLE"],
                ],
            ),
        )

        assert answer.columns == ["ID", "SEX", "Diagnosis"]
        assert answer.rows == [["163109", "F", "SLE"], ["2803470", "F", "SLE"]]


def test_sanitize_answer_keeps_single_missing_requested_value() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_nullable_answer",
                difficulty="medium",
                question="List the school name and charter funding type for the qualified schools.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["School Name", "Charter Funding Type"],
                rows=[
                    ["River Springs Charter", "Directly funded"],
                    ["Arlington High", None],
                    ["Polytechnic High", ""],
                ],
            ),
        )

        assert answer.columns == ["School Name", "Charter Funding Type"]
        assert answer.rows == [
            ["River Springs Charter", "Directly funded"],
            ["Arlington High", None],
            ["Polytechnic High", ""],
        ]


def test_sanitize_answer_keeps_name_and_requested_metric_columns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_views",
                difficulty="medium",
                question="Identify the total views on the post 'Computer Game Datasets'. Name the user who posted it last time.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["total_views", "last_user_name"],
                rows=[[1708.0, "mbq"]],
            ),
        )

        assert answer.columns == ["total_views", "last_user_name"]
        assert answer.rows == [[1708.0, "mbq"]]


def test_sanitize_answer_keeps_name_when_question_says_include_cost() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_cost",
                difficulty="hard",
                question="Write the full name of the member who spent money for water, veggie tray and supplies and include the cost of it.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = _sanitize_answer_table(
            task,
            AnswerTable(
                columns=["member_name", "cost"],
                rows=[["Elijah Allen", 28.15]],
            ),
        )

        assert answer.columns == ["member_name", "cost"]
        assert answer.rows == [["Elijah Allen", 28.15]]


def test_medium_planning_prefix_is_front_of_model_input() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_medium",
                difficulty="medium",
                question="Return the average score by group.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        state = AgentRuntimeState()
        planning_instruction, planning_snapshot = _build_planning_context(
            task,
            state,
            None,
            step_index=1,
            max_steps=16,
        )
        agent = ReActAgent(
            model=ScriptedModelAdapter([]),
            tools=ToolRegistry(specs={}, handlers={}),
            system_prompt=CODEACT_REACT_SYSTEM_PROMPT,
        )

        messages = agent._build_messages(
            task,
            state,
            front_instruction=planning_instruction,
        )

        assert planning_instruction is not None
        assert planning_snapshot is not None
        assert planning_snapshot["mode"] == "medium_light"
        assert len(planning_snapshot["nodes"]) == 3
        assert "semantic_guard" in planning_snapshot
        assert messages[0].role == "system"
        assert messages[1].content.startswith("PLANNING PREFIX")
        assert "schema_linking" in messages[1].content
        assert "target_columns" in messages[1].content
        assert "answer_submission" in messages[1].content
        assert "BIRD medium semantic guard" in messages[1].content
        assert "Row grain" in messages[1].content
        assert "Formula/unit" in messages[1].content
        assert "Value/date scope" in messages[1].content
        assert "semantic_mapping" not in messages[1].content
        assert "load_preprocess" not in messages[1].content
        assert "join_filter" not in messages[1].content
        assert "aggregate_select" not in messages[1].content
        assert "Evidence checklist:" not in messages[1].content
        assert "LOOP GUARD" not in messages[1].content
        assert "evidence_ledger" not in planning_snapshot
        assert "loop_guard" not in planning_snapshot
        assert "Progress: model call 1/16" in messages[1].content
        assert "completed=" not in messages[1].content
        assert "done" not in messages[1].content
        assert "observed_signals" in messages[1].content
        assert "Return the average score by group." in messages[2].content


def test_hard_planning_prefix_is_front_of_model_input() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_hard",
                difficulty="hard",
                question="Return the average score by group.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        state = AgentRuntimeState()
        planning_instruction, planning_snapshot = _build_planning_context(
            task,
            state,
            None,
            step_index=1,
            max_steps=16,
        )
        agent = ReActAgent(
            model=ScriptedModelAdapter([]),
            tools=ToolRegistry(specs={}, handlers={}),
            system_prompt=CODEACT_REACT_SYSTEM_PROMPT,
        )

        messages = agent._build_messages(
            task,
            state,
            front_instruction=planning_instruction,
        )

        assert planning_instruction is not None
        assert planning_snapshot is not None
        assert messages[0].role == "system"
        assert messages[1].content.startswith("PLANNING PREFIX")
        assert "schema_linking" in messages[1].content
        assert "target_columns" in messages[1].content
        assert "answer_submission" in messages[1].content
        assert "Progress: model call 1/16" in messages[1].content
        assert "completed=" not in messages[1].content
        assert "done" not in messages[1].content
        assert "observed_signals" in messages[1].content
        assert "Return the average score by group." in messages[2].content


def test_planning_snapshot_is_checkpointed_for_partial_trace() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_hard",
                difficulty="hard",
                question="Return the requested count.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        state = AgentRuntimeState()
        _, planning_snapshot = _build_planning_context(
            task,
            state,
            None,
            step_index=1,
            max_steps=16,
        )
        payloads: list[dict[str, object]] = []
        agent = ReActAgent(
            model=ScriptedModelAdapter([]),
            tools=ToolRegistry(specs={}, handlers={}),
            system_prompt=CODEACT_REACT_SYSTEM_PROMPT,
            checkpoint_callback=payloads.append,
        )

        agent._checkpoint(
            task,
            state,
            status="before_model_request",
            step_index=1,
            planning=planning_snapshot,
        )

        planning = payloads[-1]["planning"]
        assert isinstance(planning, dict)
        assert planning["enabled"] is True
        assert planning["model_input_position"] == "front_after_system_before_task_prompt"
        assert "observed_nodes" in planning
        assert "PLANNING PREFIX" in str(planning["prompt_prefix"])


def test_late_planning_does_not_prioritize_answer_without_final_table() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_hard",
                difficulty="hard",
                question="What percentage of rows match the condition?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        state = AgentRuntimeState()
        state.steps.append(
            StepRecord(
                step_index=14,
                thought="Computed the requested percentage.",
                action="execute_python",
                action_input={"code": "print('Percentage: 100.00%')"},
                raw_response="Thought: compute\nCode:\n```python\nprint('Percentage: 100.00%')\n```",
                observation={
                    "ok": True,
                    "tool": "execute_python",
                    "content": {"output": "Percentage: 100.00%", "stderr": ""},
                },
                ok=True,
            )
        )

        planning_instruction, planning_snapshot = _build_planning_context(
            task,
            state,
            None,
            step_index=15,
            max_steps=16,
        )

        assert planning_instruction is not None
        assert planning_snapshot is not None
        assert planning_snapshot["current_focus"] != "answer_submission"
        assert planning_snapshot["answer_submission_urgent"] is False
        assert planning_snapshot["final_focused_attempt"] is True
        assert "FINAL-STEPS EXACTNESS RULE" in planning_instruction


def test_late_planning_prioritizes_parseable_final_table() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_hard",
                difficulty="hard",
                question="What percentage of rows match the condition?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        state = AgentRuntimeState()
        state.steps.append(
            StepRecord(
                step_index=14,
                thought="Computed the requested percentage.",
                action="execute_python",
                action_input={"code": "print final table"},
                raw_response="Thought: compute\nCode:\n```python\nprint('FINAL_TABLE_JSON')\n```",
                observation={
                    "ok": True,
                    "tool": "execute_python",
                    "content": {
                        "output": 'FINAL_TABLE_JSON:\n{"columns":["percentage"],"rows":[[100.0]]}',
                        "stderr": "",
                    },
                },
                ok=True,
            )
        )

        planning_instruction, planning_snapshot = _build_planning_context(
            task,
            state,
            None,
            step_index=15,
            max_steps=16,
        )

        assert planning_instruction is not None
        assert planning_snapshot is not None
        assert planning_snapshot["current_focus"] == "answer_submission"
        assert planning_snapshot["answer_submission_urgent"] is True
        assert planning_snapshot["final_focused_attempt"] is False
        assert "FINAL-STEPS SUBMISSION RULE" in planning_instruction


def test_how_many_times_count_candidate_is_not_submission_ready() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_hard",
                difficulty="hard",
                question='How many times was the budget for "A" more than "B"?',
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        state = AgentRuntimeState()
        state.steps.append(
            StepRecord(
                step_index=14,
                thought="Computed a candidate count.",
                action="execute_python",
                action_input={"code": "print final table"},
                raw_response="Thought: compute\nCode:\n```python\nprint('FINAL_TABLE_JSON')\n```",
                observation={
                    "ok": True,
                    "tool": "execute_python",
                    "content": {
                        "output": 'FINAL_TABLE_JSON:\n{"columns":["count"],"rows":[[1]]}',
                        "stderr": "",
                    },
                },
                ok=True,
            )
        )

        planning_instruction, planning_snapshot = _build_planning_context(
            task,
            state,
            None,
            step_index=15,
            max_steps=16,
        )

        assert planning_instruction is not None
        assert planning_snapshot is not None
        assert planning_snapshot["answer_submission_urgent"] is False
        assert planning_snapshot["final_focused_attempt"] is True
        assert "FINAL-STEPS EXACTNESS RULE" in planning_instruction


def test_hard_markdown_entity_task_requires_grounded_linking() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "knowledge.md").write_text(
            "The Yearly Kickoff record links prose IDs to budget rows.",
            encoding="utf-8",
        )
        task = PublicTask(
            record=TaskRecord(
                task_id="task_hard",
                difficulty="hard",
                question='Compare "Yearly Kickoff" with "October Meeting".',
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        state = AgentRuntimeState()

        planning_instruction, planning_snapshot = _build_planning_context(
            task,
            state,
            None,
            step_index=1,
            max_steps=16,
        )

        assert planning_instruction is not None
        assert planning_snapshot is not None
        assert planning_snapshot["link_question_to_data_required"] is True
        assert "GROUNDED-LINK REQUIREMENT" in planning_instruction


def test_hard_rule_task_gets_conservative_knowledge_reminder() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "knowledge.md").write_text(
            "Creatinine is abnormal above 1.3.",
            encoding="utf-8",
        )
        task = PublicTask(
            record=TaskRecord(
                task_id="task_rule",
                difficulty="hard",
                question="Among patients whose creatinine level is abnormal, return their IDs.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )

        prompt = build_task_prompt(task, codeact=True)

        assert "Knowledge note:" in prompt
        assert "retrieve_knowledge(top_k=2, max_chars=500)" in prompt


def test_planning_snapshot_includes_evidence_ledger() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "knowledge.md").write_text(
            "Legal status in Commander format is stored in markdown records.",
            encoding="utf-8",
        )
        task = PublicTask(
            record=TaskRecord(
                task_id="task_evidence",
                difficulty="hard",
                question="What percentage of cards with format commander and legal status do not have a content warning?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )

        planning_instruction, planning_snapshot = _build_planning_context(
            task,
            AgentRuntimeState(),
            None,
            step_index=1,
            max_steps=16,
        )

        assert planning_instruction is not None
        assert planning_snapshot is not None
        evidence_ids = {item["id"] for item in planning_snapshot["evidence_ledger"]}
        assert {
            "concrete_data_source",
            "calculation_semantics",
            "semantic_rule_or_threshold",
            "markdown_context_mapping",
        } <= evidence_ids
        assert "Evidence checklist:" in planning_instruction
        assert "EVIDENCE_LEDGER_JSON" in planning_instruction


def test_evidence_consistency_requires_final_computation_link() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_consistency",
                difficulty="hard",
                question="Which patients have abnormal PT values?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        state = AgentRuntimeState()
        content = {
            "evidence_ledger_report": [
                {
                    "id": "semantic_rule_or_threshold",
                    "status": "verified",
                    "source": "knowledge.md",
                    "rule": "abnormal PT means PT >= 14",
                }
            ]
        }

        required_ids, warnings = _evidence_consistency_warnings(
            task,
            state,
            current_content=content,
        )

        assert required_ids == ["semantic_rule_or_threshold"]
        assert warnings

        content["evidence_consistency_report"] = [
            {
                "evidence_id": "semantic_rule_or_threshold",
                "used_in_final_computation": True,
                "where_used": "filtered Examination.PT >= 14 before selecting final IDs",
            }
        ]
        _, resolved_warnings = _evidence_consistency_warnings(
            task,
            state,
            current_content=content,
        )

        assert resolved_warnings == []


def test_candidate_with_evidence_consistency_warning_is_not_fallback() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_consistency_candidate",
                difficulty="hard",
                question="Which patients have abnormal PT values?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        answer = AnswerTable(columns=["ID"], rows=[[1], [2]])
        decision = _candidate_decision_from_observation(
            task,
            {
                "ok": True,
                "tool": "execute_python",
                "content": {
                    "output": "",
                    "evidence_consistency_warnings": [
                        "missing EVIDENCE_CONSISTENCY_JSON"
                    ],
                    "candidate_review_instruction": "verify consistency first",
                },
            },
            answer,
        )

        assert decision.auto_submit is False
        assert decision.store_as_fallback is False
        assert "evidence_consistency_warnings_present" in decision.reasons


def test_loop_guard_warns_after_repeated_python_execution() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_loop",
                difficulty="hard",
                question="What percentage of rows match the condition?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        state = AgentRuntimeState()
        for index in (1, 2):
            state.steps.append(
                StepRecord(
                    step_index=index,
                    thought="Inspect schema again.",
                    action="execute_python",
                    action_input={"code": "print('Tables: []')"},
                    raw_response="Thought: inspect\nCode:\n```python\nprint('Tables: []')\n```",
                    observation={
                        "ok": True,
                        "tool": "execute_python",
                        "content": {"output": "Tables: []", "stderr": ""},
                    },
                    ok=True,
                )
            )
        agent = ReActAgent(
            model=ScriptedModelAdapter([]),
            tools=ToolRegistry(specs={}, handlers={}),
            system_prompt=CODEACT_REACT_SYSTEM_PROMPT,
        )

        planning_instruction, planning_snapshot = _build_planning_context(
            task,
            state,
            None,
            step_index=3,
            max_steps=16,
        )
        runtime_instruction = agent._build_runtime_instruction(
            task=task,
            step_index=3,
            state=state,
            fallback_answer=None,
            consecutive_parse_errors=0,
        )

        assert planning_instruction is not None
        assert planning_snapshot is not None
        assert planning_snapshot["loop_guard"]["triggered"] is True
        assert "LOOP GUARD" in planning_instruction
        assert runtime_instruction is not None
        assert "Loop guard:" in runtime_instruction
        assert "Do not rerun the same" in runtime_instruction


def test_medium_planning_does_not_use_advanced_loop_guard() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_medium_loop",
                difficulty="medium",
                question="What percentage of rows match the condition?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        state = AgentRuntimeState()
        for index in (1, 2):
            state.steps.append(
                StepRecord(
                    step_index=index,
                    thought="Inspect schema again.",
                    action="execute_python",
                    action_input={"code": "print('Tables: []')"},
                    raw_response="Thought: inspect\nCode:\n```python\nprint('Tables: []')\n```",
                    observation={
                        "ok": True,
                        "tool": "execute_python",
                        "content": {"output": "Tables: []", "stderr": ""},
                    },
                    ok=True,
                )
            )
        agent = ReActAgent(
            model=ScriptedModelAdapter([]),
            tools=ToolRegistry(specs={}, handlers={}),
            system_prompt=CODEACT_REACT_SYSTEM_PROMPT,
        )

        planning_instruction, planning_snapshot = _build_planning_context(
            task,
            state,
            None,
            step_index=3,
            max_steps=16,
        )
        runtime_instruction = agent._build_runtime_instruction(
            task=task,
            step_index=3,
            state=state,
            fallback_answer=None,
            consecutive_parse_errors=0,
        )

        assert planning_instruction is not None
        assert planning_snapshot is not None
        assert "loop_guard" not in planning_snapshot
        assert "evidence_ledger" not in planning_snapshot
        assert "LOOP GUARD" not in planning_instruction
        assert runtime_instruction is None


def test_medium_empty_answer_is_rechecked_once() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        task = PublicTask(
            record=TaskRecord(
                task_id="task_medium_empty",
                difficulty="medium",
                question="Please list the countries with transactions in June 2013.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )
        agent = ReActAgent(
            model=ScriptedModelAdapter(
                [
                    _answer_response(
                        "I think no countries match.",
                        {"columns": ["Country"], "rows": []},
                    ),
                    _answer_response(
                        "I rechecked the sources and the verified result is empty.",
                        {"columns": ["Country"], "rows": []},
                    ),
                ]
            ),
            tools=create_default_tool_registry(),
            config=ReActAgentConfig(max_steps=2),
            system_prompt=CODEACT_REACT_SYSTEM_PROMPT,
            prompt_tool_names=("execute_python", "answer"),
        )

        result = agent.run(task)

        assert len(result.steps) == 2
        assert result.steps[0].action == "answer"
        assert not result.steps[0].ok
        assert result.steps[0].observation["medium_empty_answer_recheck"] is True
        assert result.steps[1].action == "answer"
        assert result.steps[1].ok
        assert result.answer is not None
        assert result.answer.columns == ["Country"]
        assert result.answer.rows == []


def test_knowledge_reminder_skips_quoted_value_lookup() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "knowledge.md").write_text(
            "Meeting names and aliases.",
            encoding="utf-8",
        )
        task = PublicTask(
            record=TaskRecord(
                task_id="task_quote",
                difficulty="hard",
                question='How many records mention "October Meeting"?',
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )

        prompt = build_task_prompt(task, codeact=True)

        assert "Knowledge note:" not in prompt


def test_knowledge_reminder_skips_non_hard_tasks() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "knowledge.md").write_text(
            "Creatinine is abnormal above 1.3.",
            encoding="utf-8",
        )
        task = PublicTask(
            record=TaskRecord(
                task_id="task_easy",
                difficulty="easy",
                question="How many patients have abnormal creatinine?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )

        prompt = build_task_prompt(task, codeact=True)

        assert "Knowledge note:" not in prompt


def test_knowledge_reminder_skips_format_only_hard_tasks() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "knowledge.md").write_text(
            "Format names are documented here.",
            encoding="utf-8",
        )
        task = PublicTask(
            record=TaskRecord(
                task_id="task_format",
                difficulty="hard",
                question="What percentage of cards with format commander have no content?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )

        prompt = build_task_prompt(task, codeact=True)

        assert "Knowledge note:" not in prompt


def test_codeact_task_prompt_prioritizes_json_helpers_over_markdown_for_knowledge_only_tasks() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "json").mkdir()
        (context_path / "json" / "colour.json").write_text(
            json.dumps({"table": "colour", "records": [{"id": 9, "colour": "Brown"}]}),
            encoding="utf-8",
        )
        (context_path / "knowledge.md").write_text("Colour lookup guide.", encoding="utf-8")
        task = PublicTask(
            record=TaskRecord(
                task_id="task_json",
                difficulty="easy",
                question="Return the colour for id 9.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )

        prompt = build_task_prompt(task, codeact=True)

        assert "load_json_table" in prompt
        assert "load_json_records" in prompt
        assert "extract_markdown_records" not in prompt


def test_codeact_task_prompt_mentions_markdown_helpers_only_for_document_markdown() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "doc").mkdir()
        (context_path / "doc" / "budget.md").write_text(
            "Budget recA has amount 100.",
            encoding="utf-8",
        )
        task = PublicTask(
            record=TaskRecord(
                task_id="task_doc",
                difficulty="hard",
                question="Find the amount linked to budget recA.",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )

        prompt = build_task_prompt(task, codeact=True)

        assert "extract_markdown_records" in prompt
        assert "after checking structured data" in prompt
        assert "do not force `markdown_entity_table()`" in prompt


def test_codeact_task_prompt_mentions_markdown_entity_table_only_for_row_like_docs() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "doc").mkdir()
        (context_path / "doc" / "heroes.md").write_text(
            "The operative Alpha, registered under ID 7, has height 175. "
            "Her publisher affiliation is code 13.",
            encoding="utf-8",
        )
        task = PublicTask(
            record=TaskRecord(
                task_id="task_hero_doc",
                difficulty="hard",
                question="What percentage of heroes with height between 150 and 180 are published by Marvel Comics?",
            ),
            assets=TaskAssets(task_dir=context_path, context_dir=context_path),
        )

        prompt = build_task_prompt(task, codeact=True)

        assert "pd.DataFrame(markdown_entity_table())" in prompt
        assert "include_metadata=True" in prompt
        assert "not as direct DataFrame input" in prompt


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


def test_context_manifest_adds_question_linked_schema_hints() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "csv").mkdir()
        (context_path / "csv" / "patients.csv").write_text(
            "ID,Age,Creatinine\n1,68,1.4\n2,72,1.1\n",
            encoding="utf-8",
        )
        filler = "\n\n".join(f"Unrelated paragraph {index}." for index in range(80))
        (context_path / "knowledge.md").write_text(
            filler
            + "\n\n## Creatinine rule\n"
            + "Creatinine is abnormal above 1.3 and should be interpreted as an abnormal creatinine level.",
            encoding="utf-8",
        )

        manifest = build_context_manifest(
            context_path,
            question="Among patients whose creatinine level is abnormal, how many of them aren't 70 yet?",
        )

        assert "Question-linked schema hints" in manifest
        assert "csv/patients.csv:Creatinine" in manifest
        assert "csv/patients.csv:Age" in manifest
        assert "knowledge snippets" not in manifest
        assert "abnormal above 1.3" not in manifest


def test_context_manifest_links_quoted_values_to_columns() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "csv").mkdir()
        (context_path / "csv" / "meetings.csv").write_text(
            "event_name,amount\nYearly Kickoff,300\nOctober Meeting,110\n",
            encoding="utf-8",
        )

        manifest = build_context_manifest(
            context_path,
            question='How many times was the budget for "Yearly Kickoff" more than "October Meeting"?',
        )

        assert "quoted value matches" in manifest
        assert '"Yearly Kickoff" -> csv/meetings.csv:event_name' in manifest
        assert "csv/meetings.csv:amount" in manifest


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


def test_retrieve_knowledge_searches_non_knowledge_markdown() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "csv").mkdir()
        (context_path / "doc").mkdir()
        (context_path / "csv" / "cards.csv").write_text(
            "id,legalStatus\n1,legal\n",
            encoding="utf-8",
        )
        (context_path / "doc" / "legalities.md").write_text(
            "Legal status indicates whether a card is legal, banned, or restricted in a format.",
            encoding="utf-8",
        )

        snippets = retrieve_knowledge_snippets(
            context_path,
            "What does legal status mean?",
            top_k=1,
        )

        assert snippets
        assert snippets[0]["path"] == "doc/legalities.md"
        assert "legal, banned, or restricted" in snippets[0]["snippet"]


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


def test_python_namespace_exposes_search_markdown_snippets() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "doc").mkdir()
        (context_path / "doc" / "budget.md").write_text(
            "Intro text.\n\n"
            "The Advertisement allocation for Yearly Kickoff links to budget recA with amount 300.\n\n"
            "The Advertisement allocation for October Meeting links to budget recB with amount 110.\n",
            encoding="utf-8",
        )

        result = execute_python_code(
            context_path,
            "import json\nprint(json.dumps(search_markdown(['Yearly Kickoff', 'October Meeting'], max_matches=2, context_chars=60)))",
            timeout_seconds=10,
            question="How many times was the budget in Advertisement for Yearly Kickoff more than October Meeting?",
        )

        assert result["success"]
        assert "doc/budget.md" in result["output"]
        assert "Yearly Kickoff" in result["output"]
        assert "October Meeting" in result["output"]


def test_python_namespace_exposes_json_wrapper_helpers() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "json").mkdir()
        (context_path / "json" / "colour.json").write_text(
            json.dumps(
                {
                    "table": "colour",
                    "records": [
                        {"id": 7, "colour": "Blue"},
                        {"id": 9, "colour": "Brown"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (context_path / "json" / "power.json").write_text(
            json.dumps(
                {
                    "superpower": [
                        {"id": 18, "power_name": "Super Strength"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = execute_python_code(
            context_path,
            (
                "from utils import load_json_records, load_json_table\n"
                "colour = load_json_table('json/colour.json')\n"
                "powers = load_json_records('json/power.json')\n"
                "print(colour.columns.tolist())\n"
                "print(colour.loc[colour['id'].eq(9), 'colour'].iloc[0])\n"
                "print(powers[0]['power_name'])\n"
            ),
            timeout_seconds=10,
            question="Find Brown and Super Strength.",
        )

        assert result["success"]
        assert "['id', 'colour']" in result["output"]
        assert "Brown" in result["output"]
        assert "Super Strength" in result["output"]


def test_python_namespace_exposes_markdown_record_graph() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "doc").mkdir()
        (context_path / "doc" / "budget.md").write_text(
            "The Yearly Kickoff event is linked to event record recEventYear.\n\n"
            "For event record recEventYear, the Advertisement allocation uses budget recBudgetYear.\n\n"
            "Budget recBudgetYear has approved amount $300.\n\n"
            "The October Meeting event is linked to event record recEventOct.\n\n"
            "For event record recEventOct, the Advertisement allocation uses budget recBudgetOct.\n\n"
            "Budget recBudgetOct has approved amount $100.\n",
            encoding="utf-8",
        )

        result = execute_python_code(
            context_path,
            (
                "import json\n"
                "records = extract_markdown_records(['Yearly Kickoff', 'October Meeting'], max_records=8)\n"
                "print(json.dumps(records, ensure_ascii=False))"
            ),
            timeout_seconds=10,
            question='How many times was the budget in Advertisement for "Yearly Kickoff" more than "October Meeting"?',
        )

        assert result["success"]
        assert "recEventYear" in result["output"]
        assert "recBudgetYear" in result["output"]
        assert "$300" in result["output"]
        assert "recEventOct" in result["output"]
        assert "recBudgetOct" in result["output"]
        assert "$100" in result["output"]


def test_link_question_to_data_links_markdown_entities_to_structured_ids() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "csv").mkdir()
        (context_path / "doc").mkdir()
        (context_path / "csv" / "event.csv").write_text(
            "event_id,event_name\n"
            "recykdvf4LgsyA3wZ,Yearly Kickoff\n"
            "recggMW2eyCYceNcy,October Meeting\n",
            encoding="utf-8",
        )
        (context_path / "doc" / "budget.md").write_text(
            "The Yearly Kickoff event uses event record recykdvf4LgsyA3wZ.\n\n"
            "Budget recvKTAWAFKkVNnXQ is the Advertisement allocation for event recykdvf4LgsyA3wZ. "
            "It was allocated 150.\n\n"
            "The October Meeting event uses event record recggMW2eyCYceNcy.\n\n"
            "Budget recTxecmwIhCdIKvl is the Advertisement allocation for event recggMW2eyCYceNcy. "
            "It was allocated 55.\n",
            encoding="utf-8",
        )

        result = execute_python_code(
            context_path,
            (
                "import json\n"
                "links = link_question_to_data(max_candidates=5)\n"
                "print(json.dumps(links, ensure_ascii=False))\n"
            ),
            timeout_seconds=10,
            question='How many times was the budget in Advertisement for "Yearly Kickoff" more than "October Meeting"?',
        )

        assert result["success"]
        links = json.loads(result["output"])
        rendered = json.dumps(links, ensure_ascii=False)
        assert "Yearly Kickoff" in rendered
        assert "csv/event.csv:event_name" in rendered
        assert "recykdvf4LgsyA3wZ" in rendered
        assert "recvKTAWAFKkVNnXQ" in rendered
        assert "usable_filter" in rendered
        assert links["markdown_entity_table"] is None


def test_link_question_to_data_handles_structured_only_entity_and_join_hints() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "json").mkdir()
        (context_path / "csv").mkdir()
        (context_path / "json" / "drivers.json").write_text(
            json.dumps(
                {
                    "table": "drivers",
                    "records": [
                        {"driverId": 62, "forename": "Alex", "surname": "Yoong"},
                        {"driverId": 1, "forename": "Lewis", "surname": "Hamilton"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (context_path / "csv" / "results.csv").write_text(
            "raceId,driverId,number\n18,62,12\n19,1,44\n",
            encoding="utf-8",
        )
        (context_path / "csv" / "races.csv").write_text(
            "raceId,name\n18,Australian Grand Prix\n19,Malaysian Grand Prix\n",
            encoding="utf-8",
        )

        result = execute_python_code(
            context_path,
            (
                "import json\n"
                "from utils import link_question_to_data\n"
                "links = link_question_to_data(max_candidates=5, include_markdown=False)\n"
                "print(json.dumps(links, ensure_ascii=False))\n"
            ),
            timeout_seconds=10,
            question="Which race was Alex Yoong in when he was in track number less than 20?",
        )

        assert result["success"]
        links = json.loads(result["output"])
        rendered = json.dumps(links, ensure_ascii=False)
        assert "Alex Yoong" in rendered
        assert "driverId" in rendered
        assert "62" in rendered
        assert "join_candidates" in links
        assert any("driverId" in candidate["left"] + candidate["right"] for candidate in links["join_candidates"])


def test_python_markdown_helpers_support_model_style_imports() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "doc").mkdir()
        (context_path / "doc" / "races.md").write_text(
            "raceId: 18\n"
            "year: 2008\n"
            "name: Australian Grand Prix\n"
            "circuitId: 1\n\n"
            "raceId: 19\n"
            "year: 2008\n"
            "name: Malaysian Grand Prix\n",
            encoding="utf-8",
        )
        (context_path / "knowledge.md").write_text(
            "Race records in markdown use raceId and name fields.",
            encoding="utf-8",
        )

        result = execute_python_code(
            context_path,
            (
                "import json\n"
                "from extract_markdown_records import extract_markdown_records\n"
                "from markdown_helpers import search_markdown\n"
                "from knowledge import retrieve_knowledge\n"
                "from data_agent_baseline.tools import search_markdown as package_search, extract_markdown_records as package_extract\n"
                "records = extract_markdown_records(['races.md'], "
                "pattern='raceId|year|Australian.*Grand Prix', "
                "fields=['name', 'circuitId'], include_context=True)\n"
                "print(json.dumps(records, ensure_ascii=False))\n"
                "print(retrieve_knowledge(top_k=1)[0]['path'])\n"
                "print(search_markdown(['doc/races.md'], 'Australian Grand Prix', max_matches=1)[0]['path'])\n"
                "print(package_search(['races.md'], 'Malaysian Grand Prix', max_matches=1)[0]['path'])\n"
                "print(package_extract(['races.md'], pattern='Malaysian', max_records=1)[0]['path'])\n"
            ),
            timeout_seconds=10,
            question="Find the raceId for the 2008 Australian Grand Prix.",
        )

        assert result["success"]
        assert "Australian Grand Prix" in result["output"]
        assert "raceId" in result["output"]
        assert "doc/races.md" in result["output"]


def test_markdown_entity_table_merges_prose_sections_by_id() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "doc").mkdir()
        (context_path / "doc" / "heroes.md").write_text(
            "The operative Alpha, registered under ID 7, has a height that was initially "
            "estimated at 170.0 centimeters but later confirmed at 175.0 centimeters. "
            "Her weight is recorded as 60.0 kilograms.\n\n"
            "The file for the operative Alpha, registered under the unique identifier 7, "
            "contains classification data. Her publisher affiliation is logged with the code 13.\n\n"
            "The operative Beta, registered under ID 8, has a height of 190.0 centimeters. "
            "His publisher affiliation is recorded as 4.\n",
            encoding="utf-8",
        )

        result = execute_python_code(
            context_path,
            (
                "import json\n"
                "rows = markdown_entity_table(include_evidence=False)\n"
                "print(json.dumps(rows, ensure_ascii=False, sort_keys=True))\n"
            ),
            timeout_seconds=10,
            question="What percentage of heroes with height between 150 and 180 are publisher 13?",
        )

        assert result["success"]
        rows = json.loads(result["output"])
        alpha = next(row for row in rows if row.get("id") == 7)
        assert alpha["height_cm"] == 175
        assert alpha["publisher_id"] == 13
        assert alpha["weight_kg"] == 60


def test_markdown_entity_table_extracts_patient_birth_and_creatinine() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "doc").mkdir()
        (context_path / "doc" / "Patient.md").write_text(
            "The profile for patient 12345 confirms she is female, born on July 20th, 1972. "
            "Her chart was created on March 4th, 1994.\n",
            encoding="utf-8",
        )
        (context_path / "doc" / "Laboratory.md").write_text(
            "The renal panel for patient 12345 revealed that the creatinine, initially "
            "thought to be 1.1 mg/dL, was verified at 1.6 mg/dL. The urea nitrogen was 18.0 mg/dL.\n",
            encoding="utf-8",
        )

        result = execute_python_code(
            context_path,
            (
                "import json\n"
                "from markdown_entity_table import markdown_entity_table\n"
                "rows = markdown_entity_table(include_evidence=False)\n"
                "print(json.dumps(rows, ensure_ascii=False, sort_keys=True))\n"
            ),
            timeout_seconds=10,
            question="Among patients with abnormal creatinine, how many are not 70 yet?",
        )

        assert result["success"]
        rows = json.loads(result["output"])
        patient = next(row for row in rows if row.get("patient_id") == 12345)
        assert patient["birth_year"] == 1972
        assert patient["sex"] == "F"
        assert patient["creatinine"] == 1.6
        assert patient["urea_nitrogen"] == 18

        links = json.loads(
            execute_python_code(
                context_path,
                (
                    "import json\n"
                    "print(json.dumps(link_question_to_data(max_candidates=5), ensure_ascii=False))\n"
                ),
                timeout_seconds=10,
                question="Among patients with abnormal creatinine, how many aren't 70 yet?",
            )["output"]
        )
        assert "markdown_entity_table.birth_year" in json.dumps(links["numeric_filters"], ensure_ascii=False)


def test_link_question_to_data_includes_markdown_entity_table_join_hints() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        context_path = Path(temp_dir)
        (context_path / "doc").mkdir()
        (context_path / "json").mkdir()
        (context_path / "doc" / "heroes.md").write_text(
            "The operative Alpha, registered under ID 7, has a height of 175.0 centimeters. "
            "Her publisher affiliation is logged with the code 13.\n",
            encoding="utf-8",
        )
        (context_path / "json" / "publisher.json").write_text(
            json.dumps({"table": "publisher", "records": [{"id": 13, "publisher_name": "Marvel Comics"}]}),
            encoding="utf-8",
        )

        result = execute_python_code(
            context_path,
            (
                "import json\n"
                "links = link_question_to_data(max_candidates=5)\n"
                "print(json.dumps(links, ensure_ascii=False, sort_keys=True))\n"
            ),
            timeout_seconds=10,
            question="What percentage of heroes with height between 150 and 180 are published by Marvel Comics?",
        )

        assert result["success"]
        links = json.loads(result["output"])
        rendered = json.dumps(links, ensure_ascii=False)
        assert links["markdown_entity_table"]["field_coverage"]["height_cm"] == 1
        assert links["markdown_entity_table"]["field_coverage"]["publisher_id"] == 1
        assert "markdown_entity_table.publisher_id" in rendered
        assert "publisher.json:id" in rendered
        assert "Marvel Comics" in rendered


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
