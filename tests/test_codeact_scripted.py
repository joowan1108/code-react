from __future__ import annotations

import json
import multiprocessing
import tempfile
from pathlib import Path

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.react import parse_model_step
from data_agent_baseline.benchmark.evaluate import score_prediction_csv
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import run_single_task


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
