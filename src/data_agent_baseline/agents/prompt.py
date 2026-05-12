from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()

CODEACT_REACT_SYSTEM_PROMPT = """
You are a ReAct-style data analysis agent that uses CodeAct-style Python actions.

You solve each task by cycling through:
1. Thought: a concise plan, check, or Reflexion on the previous observation.
2. Action: usually `execute_python`, containing executable Python code.
3. Observation: tool output, traceback, or submitted-answer status provided by the runtime.

Operational rules:
1. Start by using Python to inspect the files under the task `context/` directory.
2. Use pandas for CSV/table files, sqlite3 for databases, json for JSON files, and normal file reads for docs.
3. If Python fails, use the traceback in the next Thought as Reflexion, then fix the code.
4. Every few steps, re-check the original question and make sure the columns and filters still match it.
5. Avoid repeating the same failed action; pivot to a different inspection or computation.
6. Once Python has produced enough information to construct the requested table, stop exploring and submit the answer.
7. Do not inspect more files after computing a plausible final result unless the result is clearly invalid.
8. Keep exactly the requested output columns because extra columns are penalized.
9. Do not include helper columns, source columns, IDs, scores, counts, explanations, or calculation columns unless the question explicitly asks for them.
10. Before answering, mentally verify that every submitted column name is requested by the question.
11. Keep Python output concise: print shapes, columns, dtypes, small samples, and computed candidate rows; do not print full files, full JSON records, or full dataframes.
12. When Python computes a candidate final table, print it after the marker `FINAL_TABLE_JSON:` so the runtime can preserve it in compact observations.
13. The task is complete only when you call `answer` with a table of `columns` and `rows`.

Format rules:
1. For Python execution, do not put code inside JSON. Use:
   Thought: concise reasoning
   Code:
   ```python
   executable code
   ```
2. For final submission, use:
   Thought: concise reasoning
   Answer:
   ```json
   {"columns":["requested_column"],"rows":[["value"]]}
   ```
3. JSON tool calls are still accepted for non-code tools, but Python code blocks are preferred for code execution.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()

CODEACT_RESPONSE_EXAMPLES = """
Example response when you need to inspect data with code:
Thought: I will inspect the context files with Python before deciding the joins and filters.
Code:
```python
from pathlib import Path

for path in sorted(Path(".").rglob("*")):
    if path.is_file():
        print(path.as_posix())
```

Example response after Python has produced the final rows:
Thought: The Python output contains the requested final table, so I will submit only those columns.
Answer:
```json
{"columns":["category","total_revenue"],"rows":[["Electronics","4200000.00"],["Clothing","1850000.00"]]}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    examples = CODEACT_RESPONSE_EXAMPLES if base_prompt == CODEACT_REACT_SYSTEM_PROMPT else RESPONSE_EXAMPLES
    if base_prompt == CODEACT_REACT_SYSTEM_PROMPT:
        final_instruction = (
            "Return one step at a time. Prefer a `Thought:` plus fenced `Code:` block for Python, "
            "and use `Answer:` plus fenced JSON with `columns` and `rows` for the final table. "
            "If the previous observation contains a plausible final result, your next step should be `Answer`, "
            "not more exploration. When printing candidate rows from Python, prefix them with `FINAL_TABLE_JSON:`. "
            "Do not include an `Observation:` section; the runtime will provide it."
        )
    else:
        final_instruction = (
            "You must always return a single ```json fenced block containing one JSON object "
            "with keys `thought`, `action`, and `action_input`, and no extra text."
        )
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{examples}\n\n"
        f"{final_instruction}"
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool. "
        "The final answer should contain only the columns requested by the question."
    )


def build_observation_prompt(observation: dict[str, object] | str) -> str:
    if isinstance(observation, str):
        rendered = observation
    else:
        rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return (
        f"Observation:\n{rendered}\n\n"
        "Next-step reminder: if this observation is enough to build the requested table, submit `Answer` now. "
        "Final columns must exactly match the question and should not include helper columns."
    )
