from __future__ import annotations

import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import TaskRunArtifacts, run_single_task


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    return int(raw_value)


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    return float(raw_value)


def _normalize_model_api_url(api_base: str) -> str:
    normalized = api_base.rstrip("/")
    if not normalized:
        return normalized
    append_v1 = os.environ.get("MODEL_API_URL_APPEND_V1", "auto").strip().lower()
    if append_v1 in {"0", "false", "no"}:
        return normalized
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def build_submission_config(
    *,
    input_root: Path,
    scratch_root: Path,
) -> AppConfig:
    """Build an AppConfig that obeys the competition's injected model variables."""

    api_base = _normalize_model_api_url(os.environ.get("MODEL_API_URL", "").strip())
    model_name = os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b").strip() or "qwen3.5-35b-a3b"
    api_key = os.environ.get("MODEL_API_KEY", "EMPTY")

    return AppConfig(
        dataset=DatasetConfig(root_path=input_root),
        agent=AgentConfig(
            strategy=os.environ.get("AGENT_STRATEGY", "codeact").strip().lower() or "codeact",
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            max_steps=_env_int("AGENT_MAX_STEPS", 16),
            temperature=_env_float("AGENT_TEMPERATURE", 0.0),
            model_timeout_seconds=_env_float("AGENT_MODEL_TIMEOUT_SECONDS", 90.0),
            max_output_tokens=_env_int("AGENT_MAX_OUTPUT_TOKENS", 2048),
            python_timeout_seconds=_env_int("AGENT_PYTHON_TIMEOUT_SECONDS", 45),
            prompt_history_steps=_env_int("AGENT_PROMPT_HISTORY_STEPS", 5),
            full_observation_threshold=_env_int("AGENT_FULL_OBSERVATION_THRESHOLD", 6000),
            observation_head_chars=_env_int("AGENT_OBSERVATION_HEAD_CHARS", 1800),
            observation_tail_chars=_env_int("AGENT_OBSERVATION_TAIL_CHARS", 1800),
            stderr_tail_chars=_env_int("AGENT_STDERR_TAIL_CHARS", 4000),
            final_marker_chars=_env_int("AGENT_FINAL_MARKER_CHARS", 4000),
        ),
        run=RunConfig(
            output_dir=scratch_root,
            run_id="submission_run",
            max_workers=_env_int("AGENT_MAX_WORKERS", 4),
            task_timeout_seconds=_env_int("AGENT_TASK_TIMEOUT_SECONDS", 600),
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_failure_stub(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("empty\n", encoding="utf-8")


def _copy_prediction(artifact: TaskRunArtifacts, output_root: Path) -> Path | None:
    destination = output_root / artifact.task_id / "prediction.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if artifact.prediction_csv_path is None or not artifact.prediction_csv_path.exists():
        if os.environ.get("AGENT_WRITE_FAILURE_STUB", "1") != "0":
            _write_failure_stub(destination)
            return destination
        return None
    shutil.copyfile(artifact.prediction_csv_path, destination)
    return destination


def _run_one_task(
    *,
    task_id: str,
    config: AppConfig,
    scratch_run_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    started_at = perf_counter()
    artifact = run_single_task(task_id=task_id, config=config, run_output_dir=scratch_run_dir)
    prediction_path = _copy_prediction(artifact, output_root)
    return {
        "task_id": task_id,
        "succeeded": artifact.succeeded,
        "failure_reason": artifact.failure_reason,
        "prediction_csv_path": str(prediction_path) if prediction_path else None,
        "elapsed_seconds": round(perf_counter() - started_at, 3),
    }


def run_submission(
    *,
    input_root: Path = Path("/input"),
    output_root: Path = Path("/output"),
    logs_root: Path = Path("/logs"),
) -> dict[str, Any]:
    started_at = perf_counter()
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    logs_root = logs_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dabench_submission_") as temp_dir:
        scratch_root = Path(temp_dir)
        config = build_submission_config(input_root=input_root, scratch_root=scratch_root)
        scratch_run_dir = scratch_root / "submission_run"
        scratch_run_dir.mkdir(parents=True, exist_ok=True)

        dataset = DABenchPublicDataset(config.dataset.root_path)
        task_ids = dataset.list_task_ids()
        max_workers = max(1, config.run.max_workers)

        print(
            json.dumps(
                {
                    "event": "submission_start",
                    "task_count": len(task_ids),
                    "max_workers": max_workers,
                    "agent": {
                        "strategy": config.agent.strategy,
                        "model": config.agent.model,
                        "api_base_configured": bool(config.agent.api_base),
                        "max_steps": config.agent.max_steps,
                        "temperature": config.agent.temperature,
                        "model_timeout_seconds": config.agent.model_timeout_seconds,
                        "max_output_tokens": config.agent.max_output_tokens,
                        "python_timeout_seconds": config.agent.python_timeout_seconds,
                        "prompt_history_steps": config.agent.prompt_history_steps,
                        "full_observation_threshold": config.agent.full_observation_threshold,
                        "observation_head_chars": config.agent.observation_head_chars,
                        "observation_tail_chars": config.agent.observation_tail_chars,
                        "stderr_tail_chars": config.agent.stderr_tail_chars,
                        "final_marker_chars": config.agent.final_marker_chars,
                    },
                    "input_root": str(input_root),
                    "output_root": str(output_root),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        results: list[dict[str, Any]] = []
        if max_workers == 1:
            for task_id in task_ids:
                result = _run_one_task(
                    task_id=task_id,
                    config=config,
                    scratch_run_dir=scratch_run_dir,
                    output_root=output_root,
                )
                results.append(result)
                print(json.dumps({"event": "task_complete", **result}, ensure_ascii=False), flush=True)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task_id = {
                    executor.submit(
                        _run_one_task,
                        task_id=task_id,
                        config=config,
                        scratch_run_dir=scratch_run_dir,
                        output_root=output_root,
                    ): task_id
                    for task_id in task_ids
                }
                for future in as_completed(future_to_task_id):
                    task_id = future_to_task_id[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        prediction_path = output_root / task_id / "prediction.csv"
                        if os.environ.get("AGENT_WRITE_FAILURE_STUB", "1") != "0":
                            _write_failure_stub(prediction_path)
                        result = {
                            "task_id": task_id,
                            "succeeded": False,
                            "failure_reason": f"uncaught worker error: {exc}",
                            "prediction_csv_path": str(prediction_path),
                            "elapsed_seconds": None,
                        }
                    results.append(result)
                    print(json.dumps({"event": "task_complete", **result}, ensure_ascii=False), flush=True)

        summary = {
            "event": "submission_complete",
            "task_count": len(results),
            "succeeded_task_count": sum(1 for item in results if item.get("succeeded")),
            "elapsed_seconds": round(perf_counter() - started_at, 3),
            "tasks": sorted(results, key=lambda item: item["task_id"]),
        }
        _write_json(logs_root / "summary.json", summary)
        print(json.dumps(summary | {"tasks": []}, ensure_ascii=False), flush=True)
        return summary


def main() -> None:
    run_submission()


if __name__ == "__main__":
    main()
