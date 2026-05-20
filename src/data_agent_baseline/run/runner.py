from __future__ import annotations

import csv
import json
import multiprocessing
import queue as queue_module
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.prompt import CODEACT_REACT_SYSTEM_PROMPT
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        timeout_seconds=config.agent.model_timeout_seconds,
        max_output_tokens=config.agent.max_output_tokens,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _task_artifacts_from_dict(payload: dict[str, Any]) -> TaskRunArtifacts:
    prediction_csv = payload.get("prediction_csv_path")
    return TaskRunArtifacts(
        task_id=str(payload["task_id"]),
        task_output_dir=Path(str(payload["task_output_dir"])),
        prediction_csv_path=Path(str(prediction_csv)) if prediction_csv else None,
        trace_path=Path(str(payload["trace_path"])),
        succeeded=bool(payload.get("succeeded")),
        failure_reason=payload.get("failure_reason"),
    )


def _partial_trace_path(run_output_dir: Path, task_id: str) -> Path:
    return run_output_dir / task_id / "trace.partial.json"


def _load_partial_trace(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _load_existing_task_artifact(run_output_dir: Path, task_id: str) -> TaskRunArtifacts | None:
    task_output_dir = run_output_dir / task_id
    trace_path = task_output_dir / "trace.json"
    if not trace_path.exists():
        return None
    try:
        run_result = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(run_result, dict):
        return None
    prediction_csv_path = task_output_dir / "prediction.csv"
    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path if prediction_csv_path.exists() else None,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def _make_checkpoint_callback(
    *,
    task_id: str,
    run_output_dir: Path,
    started_at: float,
) -> Callable[[dict[str, object]], None]:
    partial_trace_path = _partial_trace_path(run_output_dir, task_id)

    def checkpoint(run_result: dict[str, object]) -> None:
        payload = dict(run_result)
        payload["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
        _write_json(partial_trace_path, payload)

    return checkpoint


def _write_failure_outputs_from_partial(
    *,
    task_id: str,
    run_output_dir: Path,
    failure_reason: str,
    elapsed_seconds: float,
) -> TaskRunArtifacts:
    run_result = _load_partial_trace(_partial_trace_path(run_output_dir, task_id))
    if run_result is None:
        run_result = _failure_run_result_payload(task_id, failure_reason)
    else:
        run_result = dict(run_result)
        run_result["task_id"] = task_id
        run_result["answer"] = None
        run_result["failure_reason"] = failure_reason
        run_result["succeeded"] = False
    run_result["e2e_elapsed_seconds"] = round(elapsed_seconds, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    checkpoint_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    effective_model = model or build_model_adapter(config)
    if config.agent.strategy not in {"react", "codeact"}:
        raise ValueError(f"Unknown agent.strategy: {config.agent.strategy}")
    system_prompt = CODEACT_REACT_SYSTEM_PROMPT if config.agent.strategy == "codeact" else None
    prompt_tool_names = ("execute_python", "answer") if config.agent.strategy == "codeact" else None
    agent = ReActAgent(
        model=effective_model,
        tools=tools
        or create_default_tool_registry(
            python_timeout_seconds=config.agent.python_timeout_seconds,
        ),
        config=ReActAgentConfig(
            max_steps=config.agent.max_steps,
            prompt_history_steps=config.agent.prompt_history_steps,
            full_observation_threshold=config.agent.full_observation_threshold,
            observation_head_chars=config.agent.observation_head_chars,
            observation_tail_chars=config.agent.observation_tail_chars,
            stderr_tail_chars=config.agent.stderr_tail_chars,
            final_marker_chars=config.agent.final_marker_chars,
            verifier_enabled=config.agent.verifier_enabled,
            verifier_stdout_chars=config.agent.verifier_stdout_chars,
        ),
        system_prompt=system_prompt,
        prompt_tool_names=prompt_tool_names,
        checkpoint_callback=checkpoint_callback,
    )
    run_result = agent.run(task)
    return run_result.to_dict()


def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    queue: multiprocessing.Queue[Any],
) -> None:
    started_at = perf_counter()
    checkpoint_callback = _make_checkpoint_callback(
        task_id=task_id,
        run_output_dir=run_output_dir,
        started_at=started_at,
    )
    try:
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            checkpoint_callback=checkpoint_callback,
        )
        run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
        artifact = _write_task_outputs(task_id, run_output_dir, run_result)
        queue.put(
            {
                "ok": True,
                "artifact": artifact.to_dict(),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        elapsed_seconds = perf_counter() - started_at
        try:
            artifact = _write_failure_outputs_from_partial(
                task_id=task_id,
                run_output_dir=run_output_dir,
                failure_reason=f"Task failed with uncaught error: {exc}",
                elapsed_seconds=elapsed_seconds,
            )
        except BaseException:  # noqa: BLE001
            artifact = None
        queue.put(
            {
                "ok": artifact is not None,
                "error": str(exc),
                "artifact": artifact.to_dict() if artifact is not None else None,
            }
        )


def _run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        checkpoint_callback = _make_checkpoint_callback(
            task_id=task_id,
            run_output_dir=run_output_dir,
            started_at=started_at,
        )
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            checkpoint_callback=checkpoint_callback,
        )
        run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
        return _write_task_outputs(task_id, run_output_dir, run_result)

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, run_output_dir, queue),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return _write_failure_outputs_from_partial(
            task_id=task_id,
            run_output_dir=run_output_dir,
            failure_reason=f"Task timed out after {timeout_seconds} seconds.",
            elapsed_seconds=perf_counter() - started_at,
        )

    try:
        result = queue.get(timeout=2.0)
    except queue_module.Empty:
        existing_artifact = _load_existing_task_artifact(run_output_dir, task_id)
        if existing_artifact is not None:
            return existing_artifact
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _write_failure_outputs_from_partial(
                task_id=task_id,
                run_output_dir=run_output_dir,
                failure_reason=f"Task exited unexpectedly with exit code {exit_code}.",
                elapsed_seconds=perf_counter() - started_at,
            )
        return _write_failure_outputs_from_partial(
            task_id=task_id,
            run_output_dir=run_output_dir,
            failure_reason="Task exited without returning a result.",
            elapsed_seconds=perf_counter() - started_at,
        )

    if result.get("ok"):
        artifact_payload = result.get("artifact")
        if isinstance(artifact_payload, dict):
            return _task_artifacts_from_dict(artifact_payload)
    artifact_payload = result.get("artifact")
    if isinstance(artifact_payload, dict):
        return _task_artifacts_from_dict(artifact_payload)
    return _write_failure_outputs_from_partial(
        task_id=task_id,
        run_output_dir=run_output_dir,
        failure_reason=f"Task failed with uncaught error: {result.get('error')}",
        elapsed_seconds=perf_counter() - started_at,
    )


def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    if model is None and tools is None:
        return _run_single_task_with_timeout(
            task_id=task_id,
            config=config,
            run_output_dir=run_output_dir,
        )
    else:
        checkpoint_callback = _make_checkpoint_callback(
            task_id=task_id,
            run_output_dir=run_output_dir,
            started_at=started_at,
        )
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            model=model,
            tools=tools,
            checkpoint_callback=checkpoint_callback,
        )
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry(
            python_timeout_seconds=config.agent.python_timeout_seconds,
        )
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
