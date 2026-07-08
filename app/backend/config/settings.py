from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from observability.langsmith import LangSmithSettings


class AppSettings(BaseSettings):
    app_name: str = "DeepResearch Multi-Agent Assistant"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    api_auth_key: str = ""
    api_auth_required: bool | None = None
    admin_api_key: str = ""
    admin_api_required: bool | None = None
    admin_api_key_header: str = "X-Admin-Key"
    api_docs_enabled: bool | None = None
    rate_limit_enabled: bool = True
    rate_limit_backend: str = "memory"
    rate_limit_redis_url: str = ""
    rate_limit_window_seconds: int = 60
    rate_limit_max_requests: int = 60
    trusted_proxy_headers: bool = False
    request_id_header: str = "X-Request-ID"
    langsmith_enabled: bool | None = None
    langsmith_tracing: bool | None = None
    langchain_tracing_v2: bool | None = None
    langsmith_api_key: str = ""
    langchain_api_key: str = ""
    langsmith_project: str = "deepresearch-dev"
    langchain_project: str = ""
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langchain_endpoint: str = ""
    langsmith_tags: str = ""
    langsmith_environment: str = ""
    langsmith_sample_rate: float = 1.0
    langsmith_hide_inputs: bool = False
    langsmith_hide_outputs: bool = False
    config_path: str = str(Path(__file__).resolve().parents[3] / "config.json")
    rag_upload_dir: str = str(Path(__file__).resolve().parents[3] / "rag_uploads")
    rag_max_upload_mb: int = 25
    rag_max_tenant_storage_mb: int = 512
    rag_allowed_extensions: str = ".pdf,.docx"
    rag_validate_file_signatures: bool = True

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[3] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def cors_origins(self) -> list[str]:
        values = [item.strip() for item in self.cors_allow_origins.split(",")]
        return [item for item in values if item]

    def normalized_app_env(self) -> str:
        return self.app_env.strip().lower()

    def is_production(self) -> bool:
        return self.normalized_app_env() == "production"

    def auth_required(self) -> bool:
        if self.api_auth_required is not None:
            return self.api_auth_required
        return self.is_production()

    def admin_auth_required(self) -> bool:
        if self.admin_api_required is not None:
            return self.admin_api_required
        return self.is_production()

    def docs_enabled(self) -> bool:
        if self.api_docs_enabled is not None:
            return self.api_docs_enabled
        return not self.is_production()

    def langsmith_settings(self) -> LangSmithSettings:
        values = {
            "LANGSMITH_ENABLED": self.langsmith_enabled,
            "LANGSMITH_TRACING": self.langsmith_tracing,
            "LANGCHAIN_TRACING_V2": self.langchain_tracing_v2,
            "LANGSMITH_API_KEY": self.langsmith_api_key,
            "LANGCHAIN_API_KEY": self.langchain_api_key,
            "LANGSMITH_PROJECT": self.langsmith_project,
            "LANGCHAIN_PROJECT": self.langchain_project,
            "LANGSMITH_ENDPOINT": self.langsmith_endpoint,
            "LANGCHAIN_ENDPOINT": self.langchain_endpoint,
            "LANGSMITH_TAGS": self.langsmith_tags,
            "LANGSMITH_ENVIRONMENT": self.langsmith_environment,
            "LANGSMITH_SAMPLE_RATE": self.langsmith_sample_rate,
            "LANGSMITH_HIDE_INPUTS": self.langsmith_hide_inputs,
            "LANGSMITH_HIDE_OUTPUTS": self.langsmith_hide_outputs,
        }
        cleaned = {key: value for key, value in values.items() if value is not None}
        return LangSmithSettings.from_mapping(cleaned)

    def validate_for_runtime(self) -> None:
        errors: list[str] = []
        if self.auth_required() and not self.api_auth_key.strip():
            errors.append("API_AUTH_KEY must be configured when API authentication is required.")
        if self.admin_auth_required() and not self.admin_api_key.strip():
            errors.append("ADMIN_API_KEY must be configured when admin API authentication is required.")
        if self.is_production() and "*" in self.cors_origins():
            errors.append("CORS_ALLOW_ORIGINS must not contain '*' in production.")
        if self.rate_limit_backend not in {"memory", "redis"}:
            errors.append("RATE_LIMIT_BACKEND must be either 'memory' or 'redis'.")
        if self.rate_limit_backend == "redis" and not self.rate_limit_redis_url.strip():
            errors.append("RATE_LIMIT_REDIS_URL must be configured when RATE_LIMIT_BACKEND=redis.")
        if self.rate_limit_window_seconds <= 0:
            errors.append("RATE_LIMIT_WINDOW_SECONDS must be greater than 0.")
        if self.rate_limit_max_requests <= 0:
            errors.append("RATE_LIMIT_MAX_REQUESTS must be greater than 0.")
        if self.rag_max_upload_mb <= 0:
            errors.append("RAG_MAX_UPLOAD_MB must be greater than 0.")
        if self.rag_max_tenant_storage_mb < 0:
            errors.append("RAG_MAX_TENANT_STORAGE_MB must not be negative.")
        if not self.rag_allowed_extension_set():
            errors.append("RAG_ALLOWED_EXTENSIONS must contain at least one extension.")
        errors.extend(self.langsmith_settings().validate())
        if errors:
            raise RuntimeError("Invalid runtime configuration: " + " ".join(errors))

    def rag_allowed_extension_set(self) -> set[str]:
        values = [item.strip().lower() for item in self.rag_allowed_extensions.split(",")]
        return {item if item.startswith(".") else f".{item}" for item in values if item}

    def rag_upload_path(self) -> str:
        path = Path(self.rag_upload_dir)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return str(path)
