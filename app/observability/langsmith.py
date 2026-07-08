from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from langsmith.run_helpers import trace as _ls_trace
from langsmith.run_helpers import tracing_context


logger = logging.getLogger("observability.langsmith")


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _split_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _dedupe_tags(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    tags: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tuple(tags)


@dataclass(frozen=True)
class LangSmithSettings:
    enabled: bool = False
    api_key: str = ""
    project: str = "deepresearch-dev"
    endpoint: str = "https://api.smith.langchain.com"
    tags: tuple[str, ...] = ()
    environment: str = ""
    sample_rate: float = 1.0
    hide_inputs: bool = False
    hide_outputs: bool = False

    @classmethod
    def from_env(cls) -> "LangSmithSettings":
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LangSmithSettings":
        explicit_enabled = _optional_bool(values.get("LANGSMITH_ENABLED"))
        langsmith_tracing = _optional_bool(values.get("LANGSMITH_TRACING"))
        legacy_tracing = _optional_bool(values.get("LANGCHAIN_TRACING_V2"))
        enabled = explicit_enabled
        if enabled is None:
            enabled = bool(langsmith_tracing or legacy_tracing)

        api_key = str(values.get("LANGSMITH_API_KEY") or values.get("LANGCHAIN_API_KEY") or "").strip()
        project = str(
            values.get("LANGSMITH_PROJECT")
            or values.get("LANGCHAIN_PROJECT")
            or "deepresearch-dev"
        ).strip()
        endpoint = str(
            values.get("LANGSMITH_ENDPOINT")
            or values.get("LANGCHAIN_ENDPOINT")
            or "https://api.smith.langchain.com"
        ).strip().rstrip("/")
        environment = str(values.get("LANGSMITH_ENVIRONMENT") or "").strip()
        tags = _split_tags(values.get("LANGSMITH_TAGS"))
        if environment:
            tags = (*tags, environment)
        tags = _dedupe_tags(tags)

        return cls(
            enabled=bool(enabled),
            api_key=api_key,
            project=project or "deepresearch-dev",
            endpoint=endpoint or "https://api.smith.langchain.com",
            tags=tags,
            environment=environment,
            sample_rate=_float(values.get("LANGSMITH_SAMPLE_RATE"), 1.0),
            hide_inputs=bool(_optional_bool(values.get("LANGSMITH_HIDE_INPUTS")) or False),
            hide_outputs=bool(_optional_bool(values.get("LANGSMITH_HIDE_OUTPUTS")) or False),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.enabled and not self.api_key:
            errors.append("LANGSMITH_API_KEY must be configured when LangSmith tracing is enabled.")
        if not 0.0 <= self.sample_rate <= 1.0:
            errors.append("LANGSMITH_SAMPLE_RATE must be between 0 and 1.")
        return errors

    def should_trace(self) -> bool:
        if not self.enabled:
            return False
        if self.sample_rate <= 0:
            return False
        if self.sample_rate >= 1:
            return True
        return random.random() < self.sample_rate

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "api_key_configured": bool(self.api_key),
            "project": self.project,
            "endpoint": self.endpoint,
            "tags": list(self.tags),
            "environment": self.environment,
            "sample_rate": self.sample_rate,
            "hide_inputs": self.hide_inputs,
            "hide_outputs": self.hide_outputs,
        }


def configure_langsmith(settings: LangSmithSettings | None = None) -> LangSmithSettings:
    raw_config = settings or LangSmithSettings.from_env()
    config = LangSmithSettings(
        enabled=raw_config.enabled,
        api_key=raw_config.api_key,
        project=raw_config.project,
        endpoint=raw_config.endpoint,
        tags=_dedupe_tags(raw_config.tags),
        environment=raw_config.environment,
        sample_rate=raw_config.sample_rate,
        hide_inputs=raw_config.hide_inputs,
        hide_outputs=raw_config.hide_outputs,
    )
    if config.enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = config.api_key
        os.environ["LANGCHAIN_API_KEY"] = config.api_key
        os.environ["LANGSMITH_PROJECT"] = config.project
        os.environ["LANGCHAIN_PROJECT"] = config.project
        os.environ["LANGSMITH_ENDPOINT"] = config.endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = config.endpoint
        if config.tags:
            os.environ["LANGSMITH_TAGS"] = ",".join(config.tags)
        logger.info(
            "LangSmith tracing enabled | project=%s endpoint=%s sample_rate=%.2f",
            config.project,
            config.endpoint,
            config.sample_rate,
        )
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled")
    return config


def get_langsmith_status(settings: LangSmithSettings | None = None) -> dict[str, Any]:
    return (settings or LangSmithSettings.from_env()).status()


class _NoopSpan:
    def end(self, outputs: Mapping[str, Any] | None = None, error: str | None = None) -> None:
        return None


class _LangSmithSpan:
    def __init__(self, run: Any, hide_outputs: bool):
        self._run = run
        self._hide_outputs = hide_outputs
        self._ended = False

    def end(self, outputs: Mapping[str, Any] | None = None, error: str | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        if error:
            self._run.end(error=error)
            return
        if outputs is not None and not self._hide_outputs:
            self._run.end(outputs=dict(outputs))
            return
        self._run.end()


@contextmanager
def trace_run(
    name: str,
    *,
    run_type: str = "chain",
    inputs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    settings: LangSmithSettings | None = None,
) -> Iterator[_LangSmithSpan | _NoopSpan]:
    config = settings or LangSmithSettings.from_env()
    if not config.should_trace():
        yield _NoopSpan()
        return

    merged_tags = list(_dedupe_tags([*config.tags, *(tags or ())]))
    safe_inputs = None if config.hide_inputs else dict(inputs or {})
    safe_metadata = dict(metadata or {})
    safe_metadata.setdefault("service", "deepresearch")
    if config.environment:
        safe_metadata.setdefault("environment", config.environment)

    with tracing_context(
        project_name=config.project,
        tags=merged_tags,
        metadata=safe_metadata,
        enabled=True,
    ):
        with _ls_trace(
            name,
            run_type=run_type,
            inputs=safe_inputs,
            project_name=config.project,
            tags=merged_tags,
            metadata=safe_metadata,
        ) as run:
            yield _LangSmithSpan(run, hide_outputs=config.hide_outputs)
