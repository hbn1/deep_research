"""配置模块：统一加载 .env 与 config.json，并构建全局 AppConfig。"""

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    model: str
    thread_id: str
    user_id: str
    tenant_id: str
    max_iterations: int
    enable_memory: bool
    short_term_ttl_seconds: int
    short_term_max_messages: int
    short_term_summary_threshold: int
    short_term_backend: str
    long_term_backend: str
    long_term_scope: str
    save_conversation_task: bool
    checkpointer_backend: str
    enable_milvus: bool
    memory_top_k: int
    redis_url: str
    postgres_dsn: str
    milvus_host: str
    milvus_port: int
    milvus_collection: str
    # 搜索配置
    serper_api_key: str = ""
    tavily_api_key: str = ""
    search_backends: str = "bocha"
    search_fallback_backends: str = "serper,tavily"
    search_count: int = 4
    search_timeout: float = 15.0
    search_fetch_timeout: float = 8.0
    search_max_workers: int = 6
    search_cache_enabled: bool = True
    search_cache_ttl_seconds: int = 3600
    search_rewrite_enabled: bool = True
    search_fetch_enabled: bool = True

    def with_overrides(self, **kwargs) -> "AppConfig":
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **cleaned)

    @staticmethod
    def _default_config_path() -> Path:
        return Path(__file__).resolve().parents[2] / "config.json"

    @staticmethod
    def _resolve_str(data: dict, field: str, env_key: str, default: str = "") -> str:
        env_value = os.getenv(env_key)
        if env_value is not None and str(env_value).strip() != "":
            return str(env_value).strip()
        file_value = data.get(field)
        if file_value is not None and str(file_value).strip() != "":
            return str(file_value).strip()
        return default

    @staticmethod
    def _resolve_int(data: dict, field: str, env_key: str, default: int) -> int:
        value = AppConfig._resolve_str(data, field, env_key, str(default))
        return int(value)

    @staticmethod
    def _resolve_float(data: dict, field: str, env_key: str, default: float) -> float:
        value = AppConfig._resolve_str(data, field, env_key, str(default))
        return float(value)

    @staticmethod
    def _resolve_bool(data: dict, field: str, env_key: str, default: bool) -> bool:
        value = AppConfig._resolve_str(data, field, env_key, "true" if default else "false")
        return value.lower() == "true"

    @staticmethod
    def from_file(path: str | Path | None = None) -> "AppConfig":
        config_path = Path(path) if path else AppConfig._default_config_path()
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("配置文件格式错误")

        def _s(f, e, d=""): return AppConfig._resolve_str(data, f, e, d)
        def _i(f, e, d): return AppConfig._resolve_int(data, f, e, d)
        def _f(f, e, d): return AppConfig._resolve_float(data, f, e, d)
        def _b(f, e, d): return AppConfig._resolve_bool(data, f, e, d)

        api_key = _s("api_key", "DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(f"缺少 DASHSCOPE_API_KEY 配置，请在 {config_path} 中填写 api_key，或设置环境变量 DASHSCOPE_API_KEY")

        return AppConfig(
            api_key=api_key,
            model=_s("model", "MODEL", "qwen-plus"),
            thread_id=_s("thread_id", "THREAD_ID", "default"),
            user_id=_s("user_id", "USER_ID", "default_user"),
            tenant_id=_s("tenant_id", "TENANT_ID", "default_tenant"),
            max_iterations=_i("max_iterations", "MAX_ITERATIONS", 3),
            enable_memory=_b("enable_memory", "ENABLE_MEMORY", True),
            short_term_ttl_seconds=_i("short_term_ttl_seconds", "SHORT_TERM_TTL_SECONDS", 604800),
            short_term_max_messages=_i("short_term_max_messages", "SHORT_TERM_MAX_MESSAGES", 30),
            short_term_summary_threshold=_i("short_term_summary_threshold", "SHORT_TERM_SUMMARY_THRESHOLD", 20),
            short_term_backend=_s("short_term_backend", "SHORT_TERM_BACKEND", "postgres").lower(),
            long_term_backend=_s("long_term_backend", "LONG_TERM_BACKEND", "postgres").lower(),
            long_term_scope=_s("long_term_scope", "LONG_TERM_SCOPE", "user").lower(),
            save_conversation_task=_b("save_conversation_task", "SAVE_CONVERSATION_TASK", False),
            checkpointer_backend=_s("checkpointer_backend", "CHECKPOINTER_BACKEND", "auto").lower(),
            enable_milvus=_b("enable_milvus", "ENABLE_MILVUS", True),
            memory_top_k=_i("memory_top_k", "MEMORY_TOP_K", 6),
            redis_url=_s("redis_url", "REDIS_URL", "redis://127.0.0.1:6379"),
            postgres_dsn=_s("postgres_dsn", "POSTGRES_DSN", "postgresql://127.0.0.1:5432/postgres"),
            milvus_host=_s("milvus_host", "MILVUS_HOST", "127.0.0.1"),
            milvus_port=_i("milvus_port", "MILVUS_PORT", 19530),
            milvus_collection=_s("milvus_collection", "MILVUS_COLLECTION", "mult_agent_memory"),
            serper_api_key=_s("serper_api_key", "SERPER_API_KEY"),
            tavily_api_key=_s("tavily_api_key", "TAVILY_API_KEY"),
            search_backends=_s("search_backends", "SEARCH_BACKENDS", "bocha"),
            search_fallback_backends=_s("search_fallback_backends", "SEARCH_FALLBACK_BACKENDS", "serper,tavily"),
            search_count=_i("search_count", "SEARCH_COUNT", 4),
            search_timeout=_f("search_timeout", "SEARCH_TIMEOUT", 15.0),
            search_fetch_timeout=_f("search_fetch_timeout", "SEARCH_FETCH_TIMEOUT", 8.0),
            search_max_workers=_i("search_max_workers", "SEARCH_MAX_WORKERS", 6),
            search_cache_enabled=_b("search_cache_enabled", "SEARCH_CACHE_ENABLED", True),
            search_cache_ttl_seconds=_i("search_cache_ttl_seconds", "SEARCH_CACHE_TTL_SECONDS", 3600),
            search_rewrite_enabled=_b("search_rewrite_enabled", "SEARCH_REWRITE_ENABLED", True),
            search_fetch_enabled=_b("search_fetch_enabled", "SEARCH_FETCH_ENABLED", True),
        )

    @staticmethod
    def from_env() -> "AppConfig":
        data: dict = {}
        def _s(f, e, d=""): return AppConfig._resolve_str(data, f, e, d)
        def _i(f, e, d): return AppConfig._resolve_int(data, f, e, d)
        def _f(f, e, d): return AppConfig._resolve_float(data, f, e, d)
        def _b(f, e, d): return AppConfig._resolve_bool(data, f, e, d)

        api_key = _s("api_key", "DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY 环境变量")

        return AppConfig(
            api_key=api_key,
            model=_s("model", "MODEL", "qwen-plus"),
            thread_id=_s("thread_id", "THREAD_ID", "default"),
            user_id=_s("user_id", "USER_ID", "default_user"),
            tenant_id=_s("tenant_id", "TENANT_ID", "default_tenant"),
            max_iterations=_i("max_iterations", "MAX_ITERATIONS", 3),
            enable_memory=_b("enable_memory", "ENABLE_MEMORY", True),
            short_term_ttl_seconds=_i("short_term_ttl_seconds", "SHORT_TERM_TTL_SECONDS", 604800),
            short_term_max_messages=_i("short_term_max_messages", "SHORT_TERM_MAX_MESSAGES", 30),
            short_term_summary_threshold=_i("short_term_summary_threshold", "SHORT_TERM_SUMMARY_THRESHOLD", 20),
            short_term_backend=_s("short_term_backend", "SHORT_TERM_BACKEND", "postgres").lower(),
            long_term_backend=_s("long_term_backend", "LONG_TERM_BACKEND", "postgres").lower(),
            long_term_scope=_s("long_term_scope", "LONG_TERM_SCOPE", "user").lower(),
            save_conversation_task=_b("save_conversation_task", "SAVE_CONVERSATION_TASK", False),
            checkpointer_backend=_s("checkpointer_backend", "CHECKPOINTER_BACKEND", "auto").lower(),
            enable_milvus=_b("enable_milvus", "ENABLE_MILVUS", True),
            memory_top_k=_i("memory_top_k", "MEMORY_TOP_K", 6),
            redis_url=_s("redis_url", "REDIS_URL", "redis://127.0.0.1:6379"),
            postgres_dsn=_s("postgres_dsn", "POSTGRES_DSN", "postgresql://127.0.0.1:5432/postgres"),
            milvus_host=_s("milvus_host", "MILVUS_HOST", "127.0.0.1"),
            milvus_port=_i("milvus_port", "MILVUS_PORT", 19530),
            milvus_collection=_s("milvus_collection", "MILVUS_COLLECTION", "mult_agent_memory"),
            serper_api_key=_s("serper_api_key", "SERPER_API_KEY"),
            tavily_api_key=_s("tavily_api_key", "TAVILY_API_KEY"),
            search_backends=_s("search_backends", "SEARCH_BACKENDS", "bocha"),
            search_fallback_backends=_s("search_fallback_backends", "SEARCH_FALLBACK_BACKENDS", "serper,tavily"),
            search_count=_i("search_count", "SEARCH_COUNT", 4),
            search_timeout=_f("search_timeout", "SEARCH_TIMEOUT", 15.0),
            search_fetch_timeout=_f("search_fetch_timeout", "SEARCH_FETCH_TIMEOUT", 8.0),
            search_max_workers=_i("search_max_workers", "SEARCH_MAX_WORKERS", 6),
            search_cache_enabled=_b("search_cache_enabled", "SEARCH_CACHE_ENABLED", True),
            search_cache_ttl_seconds=_i("search_cache_ttl_seconds", "SEARCH_CACHE_TTL_SECONDS", 3600),
            search_rewrite_enabled=_b("search_rewrite_enabled", "SEARCH_REWRITE_ENABLED", True),
            search_fetch_enabled=_b("search_fetch_enabled", "SEARCH_FETCH_ENABLED", True),
        )
