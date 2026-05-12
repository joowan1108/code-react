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
6. Keep only the requested output columns because extra columns are penalized.
7. The task is complete only when you call `answer` with a table of `columns` and `rows`.

Format rules:
1. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
2. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
3. Do not output any text before or after the fenced JSON block.
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
```json
{"thought":"I will inspect the context files with Python before deciding the joins and filters.","action":"execute_python","action_input":{"code":"from pathlib import Path\\nfor path in sorted(Path('.').rglob('*')):\\n    if path.is_file():\\n        print(path.as_posix())"}}
```

Example response after Python has produced the final rows:
```json
{"thought":"The Python output contains the requested final table, so I will submit only those columns.","action":"answer","action_input":{"columns":["category","total_revenue"],"rows":[["Electronics","4200000.00"],["Clothing","1850000.00"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    examples = CODEACT_RESPONSE_EXAMPLES if base_prompt == CODEACT_REACT_SYSTEM_PROMPT else RESPONSE_EXAMPLES
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{examples}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
