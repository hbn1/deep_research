"""Workflow service layer: manages LangGraph app lifecycle and request execution.

Memory: tries MemoryOrchestrator (v2: PG + Redis) first;
        falls back to MemoryManager (v1: SQLite) if v2 backends unavailable.
"""

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

from mult_agents.config import AppConfig
from mult_agents.graph import build_app as build_workflow_app
from mult_agents.main import build_agents, build_checkpointer, build_memory_manager
from mult_agents.state import create_initial_state

logger = logging.getLogger("backend.service")

# v2 memory imports (optional)
try:
    import redis as _redis_lib
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


class WorkflowService:
    """Manages the LangGraph workflow app lifecycle with lazy initialization.

    Memory backends:
        v2 (preferred): MemoryOrchestrator → PG + Redis + Milvus
        v1 (fallback):  MemoryManager → SQLite

    The _memory_orchestrator attr is the primary interface (v2 or None).
    _memory_manager is the legacy fallback.
    """

    def __init__(self, config_path: str):
        self._config_path = config_path
        self._lock = asyncio.Lock()
        self._initialized = False
        self._base_config: AppConfig | None = None
        self._memory_orchestrator = None       # v2
        self._memory_manager = None            # v1 fallback
        self._app = None

    # ── Init ────────────────────────────────────────────────

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            base_config = AppConfig.from_file(self._config_path)

            # Try v2 MemoryOrchestrator first
            orch = await self._try_init_v2_memory(base_config)
            if orch:
                self._memory_orchestrator = orch
                logger.info("WorkflowService using v2 MemoryOrchestrator")
            else:
                # Fall back to v1 MemoryManager
                self._memory_manager = build_memory_manager(base_config)
                logger.info("WorkflowService using v1 MemoryManager (fallback)")

            agents = build_agents(base_config.model, base_config.api_key, base_config)
            checkpointer, _ = build_checkpointer(base_config)

            # Pass the legacy memory_manager for the graph's memory_reflect node
            legacy_mm = self._memory_orchestrator or self._memory_manager
            self._app = build_workflow_app(agents, checkpointer, memory_manager=legacy_mm)
            self._base_config = base_config
            self._initialized = True
            logger.info("WorkflowService initialized model=%s", base_config.model)

    async def _try_init_v2_memory(self, config: AppConfig):
        """Try to initialize v2 MemoryOrchestrator. Returns None if unavailable."""
        pg_dsn = config.postgres_dsn
        redis_url = config.redis_url

        if not pg_dsn or not redis_url or not config.enable_memory:
            return None

        try:
            from mult_agents.memory import (
                UnifiedMemoryStore,
                ShortTermService,
                MemoryOrchestrator,
                MemoryExtractor,
                MemoryInjector,
            )
            from mult_agents.memory.base import DashScopeEmbeddingProvider
            from mult_agents.memory.backends import MilvusBackend, NoOpVectorBackend

            # ── Redis ──
            if not _HAS_REDIS:
                logger.warning("v2: redis package not installed")
                return None
            redis_client = _redis_lib.from_url(redis_url, socket_connect_timeout=3)
            redis_client.ping()
            st = ShortTermService(redis_client, ttl_seconds=config.short_term_ttl_seconds)

            # ── PG + Milvus ──
            embedder = None
            if config.api_key:
                try:
                    embedder = DashScopeEmbeddingProvider(config.api_key)
                except Exception as exc:
                    logger.warning("v2: embedding provider init failed: %s", exc)

            mv = NoOpVectorBackend()
            if config.enable_milvus and config.milvus_host and config.api_key:
                try:
                    mv = MilvusBackend(
                        config.milvus_host, config.milvus_port,
                        config.milvus_collection, config.api_key,
                    )
                except Exception as exc:
                    logger.warning("v2: Milvus init failed, using no-op: %s", exc)

            lt = UnifiedMemoryStore(
                postgres_dsn=pg_dsn,
                embedding_provider=embedder,
                milvus_backend=mv,
            )

            # ── Extractor + Injector ──
            llm = self._build_llm_for_extractor(config)
            extractor = MemoryExtractor(llm=llm)
            injector = MemoryInjector()

            orch = MemoryOrchestrator(
                short_term=st,
                long_term=lt,
                extractor=extractor,
                injector=injector,
            )
            logger.info(
                "v2 MemoryOrchestrator ready: pg=%s milvus=%s",
                bool(pg_dsn), mv.health_check(),
            )
            return orch

        except Exception as exc:
            logger.warning("v2 MemoryOrchestrator init failed, fallback to v1: %s", exc)
            return None

    @staticmethod
    def _build_llm_for_extractor(config: AppConfig):
        """Build a lightweight LLM for memory extraction."""
        try:
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                model=config.small_model,
                temperature=0.0,
                dashscope_api_key=config.api_key,
            )
        except Exception:
            return None

    # ── Runtime config ──────────────────────────────────────

    def _build_runtime_config(
        self,
        user_id: str, thread_id: str, tenant_id: str,
        max_iterations: int | None, enable_memory: bool | None,
    ) -> AppConfig:
        if self._base_config is None:
            raise RuntimeError("service not initialized")
        overrides = {
            "user_id": user_id,
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "max_iterations": (
                max_iterations
                if max_iterations is not None
                else self._base_config.max_iterations
            ),
        }
        if enable_memory is not None:
            overrides["enable_memory"] = enable_memory
        return self._base_config.with_overrides(**overrides)

    # ── Memory: recall / persist (v2 async, v1 sync fallback) ──

    async def _recall_context(
        self, runtime_config: AppConfig, query: str,
    ) -> str:
        """Retrieve memory context. v2 async preferred, v1 sync fallback."""
        if not runtime_config.enable_memory:
            return ""

        if self._memory_orchestrator:
            try:
                return await self._memory_orchestrator.recall_context(
                    tenant_id=runtime_config.tenant_id,
                    user_id=runtime_config.user_id,
                    thread_id=runtime_config.thread_id,
                    query=query,
                    max_memories=runtime_config.memory_top_k,
                )
            except Exception:
                logger.warning("v2 recall failed, trying v1 fallback")

        if self._memory_manager:
            try:
                return self._memory_manager.build_personalized_prompt_context(
                    user_id=runtime_config.user_id,
                    thread_id=runtime_config.thread_id,
                    query=query,
                    tenant_id=runtime_config.tenant_id,
                    max_memories=runtime_config.memory_top_k,
                )
            except Exception:
                logger.warning("v1 memory context build failed")

        return ""

    async def _persist_turn(
        self, runtime_config: AppConfig, query: str, answer: str,
    ) -> None:
        """Persist after each turn. v2 async preferred, v1 sync fallback."""
        if not runtime_config.enable_memory:
            return

        if self._memory_orchestrator:
            try:
                await self._memory_orchestrator.persist_turn(
                    tenant_id=runtime_config.tenant_id,
                    user_id=runtime_config.user_id,
                    thread_id=runtime_config.thread_id,
                    query=query,
                    answer=answer,
                )
                return
            except Exception:
                logger.warning("v2 persist failed, trying v1 fallback")

        if self._memory_manager:
            try:
                self._memory_manager.persist_turn(
                    tenant_id=runtime_config.tenant_id,
                    user_id=runtime_config.user_id,
                    thread_id=runtime_config.thread_id,
                    query=query,
                    answer=answer,
                )
            except Exception:
                logger.warning("v1 memory persist failed")

    # ── State preparation ───────────────────────────────────

    async def _prepare_state(
        self, query: str, runtime_config: AppConfig,
    ) -> tuple[dict, dict]:
        memory_context = await self._recall_context(runtime_config, query)
        state = create_initial_state(
            query=query,
            max_iterations=runtime_config.max_iterations,
            user_id=runtime_config.user_id,
            tenant_id=runtime_config.tenant_id,
            memory_context=memory_context,
        )
        config = {"configurable": {"thread_id": runtime_config.thread_id}}
        return state, config

    # ── Core run ────────────────────────────────────────────

    async def _run(
        self,
        query: str, user_id: str, thread_id: str, tenant_id: str,
        max_iterations: int | None, enable_memory: bool | None,
    ) -> tuple[str, str]:
        await self._ensure_initialized()
        runtime_config = self._build_runtime_config(
            user_id=user_id, thread_id=thread_id, tenant_id=tenant_id,
            max_iterations=max_iterations, enable_memory=enable_memory,
        )
        state, config = await self._prepare_state(query, runtime_config)
        result = await self._app.ainvoke(state, config)
        final = str(result.get("final", ""))
        route = str(result.get("intent", "multiagent"))
        await self._persist_turn(runtime_config, query, final)
        return final, route

    # ── Public API ──────────────────────────────────────────

    @staticmethod
    def _node_message(node_name: str) -> str:
        mapping = {
            "intent": "Intent Router", "direct_answer": "Direct Responder",
            "plan": "Planner", "web_search": "Web Scout",
            "local_rag": "Local Scout", "deep_dive": "Evidence Judge",
            "analyze": "Analyst", "reflect": "Reflect",
            "write": "Writer", "memory_reflect": "Memory Reflect",
        }
        return mapping.get(node_name, f"{node_name} running")

    async def run(
        self, query: str, user_id: str, thread_id: str, tenant_id: str,
        max_iterations: int | None = None, enable_memory: bool | None = None,
    ) -> str:
        final, _ = await self._run(
            query, user_id, thread_id, tenant_id, max_iterations, enable_memory,
        )
        return final

    async def run_with_route(
        self, query: str, user_id: str, thread_id: str, tenant_id: str,
        max_iterations: int | None = None, enable_memory: bool | None = None,
    ) -> tuple[str, str]:
        return await self._run(
            query, user_id, thread_id, tenant_id, max_iterations, enable_memory,
        )

    async def stream_events(
        self, query: str, user_id: str, thread_id: str, tenant_id: str,
        max_iterations: int | None = None, enable_memory: bool | None = None,
    ) -> AsyncIterator[dict]:
        await self._ensure_initialized()
        runtime_config = self._build_runtime_config(
            user_id=user_id, thread_id=thread_id, tenant_id=tenant_id,
            max_iterations=max_iterations, enable_memory=enable_memory,
        )
        state, config = await self._prepare_state(query, runtime_config)
        final = ""
        route = "multiagent"

        yield {"type": "status", "message": "Research task received"}

        try:
            async for update in self._app.astream(state, config, stream_mode="updates"):
                if not isinstance(update, dict):
                    continue
                for node_name, node_output in update.items():
                    yield {
                        "type": "phase", "node": node_name,
                        "message": self._node_message(str(node_name)),
                    }
                    if isinstance(node_output, dict):
                        if node_name == "intent":
                            detected = str(node_output.get("intent", route)).strip().lower()
                            if detected in {"direct", "multiagent"}:
                                route = detected
                        value = node_output.get("final")
                        if value:
                            final = str(value)

            if not final:
                result = await self._app.ainvoke(state, config)
                final = str(result.get("final", ""))
                route = str(result.get("intent", route)).strip().lower()

            await self._persist_turn(runtime_config, query, final)

            yield {
                "type": "route", "route": route,
                "message": "Direct answer" if route == "direct" else "Deep research",
            }
            yield {
                "type": "final",
                "query": query, "user_id": user_id,
                "thread_id": thread_id, "tenant_id": tenant_id,
                "final": final,
            }
        except Exception as exc:
            logger.exception("stream_events failed")
            yield {"type": "error", "message": str(exc)}
