#!/usr/bin/env python3
"""Runtime verification for DeepResearch.

This script intentionally avoids printing secrets. It is meant to be run after
the backend and frontend dev server are started.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    data: dict[str, Any] | None = None


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_config_model() -> str:
    config_path = ROOT / "config.json"
    if not config_path.exists():
        return "qwen-plus"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return str(config.get("model") or "qwen-plus")
    except Exception:
        return "qwen-plus"


def proxies_from_env(env: dict[str, str]) -> dict[str, str]:
    proxies: dict[str, str] = {}
    if env.get("HTTP_PROXY"):
        proxies["http"] = env["HTTP_PROXY"]
    if env.get("HTTPS_PROXY"):
        proxies["https"] = env["HTTPS_PROXY"]
    return proxies


def post_json(url: str, payload: dict[str, Any], timeout: int, headers: dict[str, str] | None = None):
    return requests.post(url, json=payload, headers=headers or {}, timeout=timeout)


def check_provider(env: dict[str, str], model: str, timeout: int) -> CheckResult:
    key = env.get("DASHSCOPE_API_KEY", "")
    base = env.get("OPENAI_BASE_URL", "").rstrip("/")
    if not key:
        return CheckResult("provider", False, "DASHSCOPE_API_KEY is missing")
    if not base:
        return CheckResult("provider", False, "OPENAI_BASE_URL is missing")

    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply exactly OK."}],
        "max_tokens": 8,
    }
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            proxies=proxies_from_env(env),
            timeout=timeout,
        )
    except Exception as exc:
        return CheckResult("provider", False, f"{type(exc).__name__}: {exc}")

    body = response.text[:300]
    if not response.ok:
        return CheckResult("provider", False, f"HTTP {response.status_code}: {body}")
    return CheckResult("provider", True, f"model {model} returned HTTP {response.status_code}")


def check_backend(env: dict[str, str], backend_url: str, timeout: int) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        response = requests.get(f"{backend_url}/health", timeout=timeout)
        results.append(
            CheckResult("backend_health", response.ok, f"HTTP {response.status_code}: {response.text[:200]}")
        )
    except Exception as exc:
        return [CheckResult("backend_health", False, f"{type(exc).__name__}: {exc}")]

    headers = {}
    if env.get("API_AUTH_KEY"):
        headers["X-API-Key"] = env["API_AUTH_KEY"]
    if env.get("ADMIN_API_KEY"):
        headers["X-Admin-Key"] = env["ADMIN_API_KEY"]
    try:
        response = post_json(
            f"{backend_url}/api/v1/research/run",
            {"query": "Answer with the number only: 1+1", "max_iterations": 1, "enable_memory": False},
            timeout=timeout,
            headers=headers,
        )
        detail = f"HTTP {response.status_code}: {response.text[:300]}"
        ok = response.ok and '"final"' in response.text
        results.append(CheckResult("backend_research", ok, detail))
    except Exception as exc:
        results.append(CheckResult("backend_research", False, f"{type(exc).__name__}: {exc}"))

    try:
        response = requests.get(
            f"{backend_url}/api/v1/observability/langsmith/status",
            headers=headers,
            timeout=timeout,
        )
        detail = f"HTTP {response.status_code}: {response.text[:300]}"
        ok = response.ok and "api_key_configured" in response.text
        results.append(CheckResult("langsmith_status", ok, detail))
    except Exception as exc:
        results.append(CheckResult("langsmith_status", False, f"{type(exc).__name__}: {exc}"))
    return results


def check_frontend(frontend_url: str, timeout: int, run_eval: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        response = requests.get(f"{frontend_url}/api/v1/evals/datasets", timeout=timeout)
        ok = response.ok and "smoke" in response.text
        results.append(CheckResult("frontend_proxy_datasets", ok, f"HTTP {response.status_code}: {response.text[:300]}"))
    except Exception as exc:
        return [CheckResult("frontend_proxy_datasets", False, f"{type(exc).__name__}: {exc}")]

    try:
        response = post_json(
            f"{frontend_url}/api/v1/research/run",
            {"query": "Answer with the number only: 1+1", "max_iterations": 1, "enable_memory": False},
            timeout=timeout,
        )
        ok = response.ok and '"final"' in response.text
        results.append(CheckResult("frontend_proxy_research", ok, f"HTTP {response.status_code}: {response.text[:300]}"))
    except Exception as exc:
        results.append(CheckResult("frontend_proxy_research", False, f"{type(exc).__name__}: {exc}"))

    if run_eval:
        try:
            response = post_json(
                f"{frontend_url}/api/v1/evals/run",
                {
                    "dataset_id": "smoke",
                    "case_ids": ["smoke_direct_intro"],
                    "max_iterations": 1,
                    "enable_memory": False,
                    "score_threshold": 60,
                },
                timeout=timeout,
            )
            ok = response.ok and '"passed_cases":1' in response.text.replace(" ", "")
            results.append(CheckResult("frontend_eval_smoke", ok, f"HTTP {response.status_code}: {response.text[:300]}"))
        except Exception as exc:
            results.append(CheckResult("frontend_eval_smoke", False, f"{type(exc).__name__}: {exc}"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DeepResearch runtime connectivity.")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend-url", default="http://localhost:5173")
    parser.add_argument("--model", default=load_config_model())
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--skip-provider", action="store_true")
    parser.add_argument("--skip-backend", action="store_true")
    parser.add_argument("--skip-frontend", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    env = load_env(ROOT / ".env")
    results: list[CheckResult] = []
    if not args.skip_provider:
        results.append(check_provider(env, args.model, args.timeout))
    if not args.skip_backend:
        results.extend(check_backend(env, args.backend_url.rstrip("/"), args.timeout))
    if not args.skip_frontend:
        results.extend(check_frontend(args.frontend_url.rstrip("/"), args.timeout, not args.skip_eval))

    if args.json_output:
        print(json.dumps([result.__dict__ for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            status = "PASS" if result.ok else "FAIL"
            print(f"[{status}] {result.name}: {result.detail}")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
