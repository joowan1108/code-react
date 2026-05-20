from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    strategy: str = "codeact"
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0
    model_timeout_seconds: float = 90.0
    max_output_tokens: int | None = 2048
    python_timeout_seconds: int = 45
    prompt_history_steps: int = 5
    full_observation_threshold: int = 6000
    observation_head_chars: int = 1800
    observation_tail_chars: int = 1800
    stderr_tail_chars: int = 4000
    final_marker_chars: int = 4000
    verifier_enabled: bool = True
    verifier_stdout_chars: int = 2500


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _bool_value(raw_value: object, default_value: bool) -> bool:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return raw_value
    normalized = str(raw_value).strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default_value


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        strategy=str(agent_payload.get("strategy", agent_defaults.strategy)).strip().lower(),
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        model_timeout_seconds=float(
            agent_payload.get("model_timeout_seconds", agent_defaults.model_timeout_seconds)
        ),
        max_output_tokens=(
            None
            if agent_payload.get("max_output_tokens", agent_defaults.max_output_tokens) is None
            else int(agent_payload.get("max_output_tokens", agent_defaults.max_output_tokens))
        ),
        python_timeout_seconds=int(
            agent_payload.get("python_timeout_seconds", agent_defaults.python_timeout_seconds)
        ),
        prompt_history_steps=int(agent_payload.get("prompt_history_steps", agent_defaults.prompt_history_steps)),
        full_observation_threshold=int(
            agent_payload.get("full_observation_threshold", agent_defaults.full_observation_threshold)
        ),
        observation_head_chars=int(
            agent_payload.get("observation_head_chars", agent_defaults.observation_head_chars)
        ),
        observation_tail_chars=int(
            agent_payload.get("observation_tail_chars", agent_defaults.observation_tail_chars)
        ),
        stderr_tail_chars=int(agent_payload.get("stderr_tail_chars", agent_defaults.stderr_tail_chars)),
        final_marker_chars=int(agent_payload.get("final_marker_chars", agent_defaults.final_marker_chars)),
        verifier_enabled=_bool_value(
            agent_payload.get("verifier_enabled"), agent_defaults.verifier_enabled
        ),
        verifier_stdout_chars=int(
            agent_payload.get("verifier_stdout_chars", agent_defaults.verifier_stdout_chars)
        ),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
