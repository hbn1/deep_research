"""Structured JSON logging with trace_id support.

Provides a JsonFormatter for Python logging that emits JSON lines
with timestamp, level, logger, message, and optional trace_id.
"""

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar

# Context variable for per-request trace_id
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def set_trace_id(trace_id: str = "") -> str:
    """Set the trace_id for the current async context. Returns the id."""
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id.get()


class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        tid = get_trace_id()
        if tid:
            payload["trace_id"] = tid
        if record.exc_info and record.exc_info[1]:
            payload["exc"] = str(record.exc_info[1])
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO, json_format: bool = False) -> None:
    """Configure root logger for structured output.

    Set JSON_FORMAT=1 env var to enable JSON lines; defaults to plain text.
    """
    root = logging.getLogger()
    handler = logging.StreamHandler()
    if json_format or os.getenv("JSON_FORMAT", "").strip() == "1":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
    root.handlers = [handler]
    root.setLevel(level)
