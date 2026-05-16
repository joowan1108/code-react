from __future__ import annotations

import json
import re

from data_agent_baseline.agents.manifest import build_context_manifest
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.knowledge import extract_ambiguity_section


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
You are a CodeAct data agent. Use Python to solve the task from the provided schema/file manifest.

Operational rules:
1. Treat the manifest in the user message as the initial context inspection.
2. Do not start by listing files or printing whole documents unless the manifest is clearly insufficient.
3. First prefer one focused Python script that loads likely relevant files, checks compact schema/samples, and computes a candidate table.
4. Use pandas for CSV/table files, sqlite3 for databases, json for JSON files, and normal file reads for docs.
5. If Python fails, use the traceback as Reflexion and fix the code without repeating the same failed action.
6. Keep exactly the requested output columns; extra columns are penalized.
7. Keep Python output concise: print shapes, columns, small samples, and candidate rows, not full files or full dataframes.
8. Print `FINAL_TABLE_JSON:` only when the JSON is the exact final answer table to submit.
9. Never use `FINAL_TABLE_JSON:` for samples, previews, intermediate candidates, or debug output.
10. If the question asks to list/all/which rows, include every matching row, not only the first one.
11. Once the exact final result exists, submit it immediately; the runtime may submit `FINAL_TABLE_JSON` directly.
12. For SQLite tasks, use exact table/column names from the manifest's sqlite_master and PRAGMA schema; do not invent table, column, or join names.

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
3. The only tools shown in CodeAct mode are `execute_python` and `answer`.
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
Thought: I will use the manifest to load the likely relevant table and inspect a compact sample.
Code:
```python
import pandas as pd

data = pd.read_csv("csv/relevant_table.csv")
print("columns:", data.columns.tolist())
print("shape:", data.shape)
print(data.head(3).to_string(index=False))
```

Example response when Python can compute the exact final table:
Thought: I can compute the requested final rows directly and print only the final table marker.
Code:
```python
import json
import pandas as pd

data = pd.read_csv("csv/relevant_table.csv")
answer = data.groupby("category", as_index=False)["revenue"].sum()
answer = answer.rename(columns={"revenue": "total_revenue"})
print("FINAL_TABLE_JSON:")
print(json.dumps({"columns": answer.columns.tolist(), "rows": answer.values.tolist()}))
```

Example response after Python has produced the final rows:
Thought: The Python output contains the requested final table, so I will submit only those columns.
Answer:
```json
{"columns":["category","total_revenue"],"rows":[["Electronics","4200000.00"],["Clothing","1850000.00"]]}
```

""".strip()

KNOWLEDGE_TRIGGER_TERMS = {
    "abnormal",
    "category",
    "classification",
    "code",
    "coded",
    "definition",
    "diagnosis",
    "disease",
    "format",
    "legal",
    "meaning",
    "monthly",
    "normal",
    "per",
    "range",
    "rule",
    "severe",
    "status",
    "threshold",
    "type",
    "unit",
    "warning",
    "yearly",
}


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    examples = CODEACT_RESPONSE_EXAMPLES if base_prompt == CODEACT_REACT_SYSTEM_PROMPT else RESPONSE_EXAMPLES
    if base_prompt == CODEACT_REACT_SYSTEM_PROMPT:
        final_instruction = (
            "Return one step at a time. Prefer a `Thought:` plus fenced `Code:` block for Python, "
            "and use `Answer:` plus fenced JSON with `columns` and `rows` for the final table. "
            "If the previous observation contains a plausible final result, your next step should be `Answer`, "
            "not more exploration. When printing final rows from Python, prefix them with `FINAL_TABLE_JSON:`. "
            "`FINAL_TABLE_JSON:` is only for the exact final answer table, never for previews or samples. "
            "For lowest/highest/min/max/top questions, verify ties and include all tied rows. "
            "For count questions, verify whether the counted entity should be distinct. "
            "For average monthly/yearly questions, verify unit conversion before finalizing. "
            "For SQLite files, rely on the manifest's sqlite_master and PRAGMA schema before writing joins. "
            "Do not include an `Observation:` or `Output:` section; the runtime will provide observations."
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


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9]+", text) if len(token) > 1}


def _has_quoted_value(text: str) -> bool:
    return bool(re.search(r'"[^"]+"|\'[^\']+\'', text))


def _knowledge_reminder(task: PublicTask) -> str:
    if task.difficulty.casefold() != "hard":
        return ""
    if _has_quoted_value(task.question):
        return ""
    if not (_tokens(task.question) & KNOWLEDGE_TRIGGER_TERMS):
        return ""
    try:
        has_knowledge = any(task.context_dir.rglob("knowledge.md"))
    except Exception:  # noqa: BLE001
        has_knowledge = False
    if not has_knowledge:
        return ""
    return (
        "Knowledge note: the question may hinge on a domain rule or coded meaning. "
        "Consider `retrieve_knowledge(top_k=2, max_chars=500)` only if structured columns or compact samples "
        "do not define the needed coded value, threshold, normal/abnormal rule, or ambiguous term. "
        "Do not use it for quoted-value lookup or broad document reading. "
    )


def _ambiguity_block(task: PublicTask) -> str:
    try:
        body = extract_ambiguity_section(task.context_dir)
    except Exception:  # noqa: BLE001
        return ""
    if not body:
        return ""
    return f"Field disambiguation from knowledge guide:\n{body}\n"


def build_task_prompt(task: PublicTask, *, codeact: bool = False) -> str:
    context_manifest = build_context_manifest(task.context_dir, question=task.question)
    ambiguity = _ambiguity_block(task)
    if codeact:
        return (
            f"Question: {task.question}\n"
            f"{context_manifest}\n"
            "The manifest above is your initial schema/file inspection. "
            "Do not spend the first step listing files again. "
            "Use it to choose a focused Python computation or a compact targeted inspection. "
            "All file paths are relative to the task context directory. "
            "For SQLite files, the manifest includes sqlite_master plus PRAGMA table_info/foreign_key_list details; "
            "use those exact table and column names for joins. "
            "If your Python code computes a plausible final table, print `FINAL_TABLE_JSON:` followed by "
            "JSON with `columns` and `rows`; use that marker only for the exact final answer. "
            f"{ambiguity}"
            f"{_knowledge_reminder(task)}"
            "The final answer should contain only the columns requested by the question. "
            "Before finalizing, check ties for min/max/top wording, include all matching rows for list/all wording, "
            "and check distinct counts or monthly/yearly unit conversions when relevant."
        )
    return (
        f"Question: {task.question}\n"
        f"{context_manifest}\n"
        f"{ambiguity}"
        "Use the manifest to choose focused Python inspections. "
        "Do not print whole files or full tables just to rediscover this schema. "
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
