"""运行主入口：构建 Agent、初始化记忆与 checkpointer，并驱动工作流执行。"""

import argparse
import json
import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mult_agents"

from .config import AppConfig
from .graph import build_app as build_workflow_app
from .nodes import (
    intent_node, direct_answer_node, plan_node,
    web_search_node, local_rag_node, deep_dive_node,
    analyze_node, reflect_node, write_node,
)
from .state import ResearchState, create_initial_state
from .prompts import PROMPTS
from .utils import colorize, emit, collect_tool_calls, with_memory_context, log_inputs
from .memory import MemoryManager, ProceduralMemoryStore, MemoryExtractor
from .tools import init_rag_system, init_search_from_config
from .rag.core import RAGConfig
from observability.langsmith import configure_langsmith

logger = logging.getLogger("mult_agents")


# ── Agent 构建 ─────────────────────────────────────────────────

@dataclass
class AgentBundle:
    intent_router: any
    planner: any
    scout_web: any
    scout_local: any
    evidence_judge: any
    analyst: any
    direct_responder: any
    writer: any


def configure_dashscope_endpoint(config: AppConfig) -> None:
    """Apply a custom DashScope workspace endpoint before SDK clients are built."""
    http_base = str(getattr(config, "dashscope_http_base_url", "") or "").strip().rstrip("/")
    websocket_base = str(getattr(config, "dashscope_websocket_base_url", "") or "").strip().rstrip("/")
    if http_base:
        os.environ["DASHSCOPE_HTTP_BASE_URL"] = http_base
    if websocket_base:
        os.environ["DASHSCOPE_WEBSOCKET_BASE_URL"] = websocket_base
    if not http_base and not websocket_base:
        return
    try:
        import dashscope
        if http_base:
            dashscope.base_http_api_url = http_base
        if websocket_base:
            dashscope.base_websocket_api_url = websocket_base
        logger.info("DashScope endpoint configured | http_base=%s", http_base or "default")
    except Exception as exc:
        logger.warning("DashScope endpoint configuration failed: %s", exc)


def build_agent(model: str, api_key: str, prompt_key: str, temperature: float, tools: list):
    llm = ChatTongyi(model=model, temperature=temperature, dashscope_api_key=api_key or None)
    prompt = PROMPTS[prompt_key]
    primary = create_agent(model=llm, tools=tools, system_prompt=prompt)
    # Enterprise fallback: degrade to qwen-turbo on rate limit / timeout
    try:
        fallback_llm = ChatTongyi(model="qwen-turbo", temperature=temperature, dashscope_api_key=api_key or None)
        fallback = create_agent(model=fallback_llm, tools=tools, system_prompt=prompt)
        return primary.with_fallbacks([fallback])
    except Exception:
        return primary


def build_agents(model: str, api_key: str, config: AppConfig) -> AgentBundle:
    configure_dashscope_endpoint(config)

    configure_langsmith()

    if getattr(config, "enable_milvus", False):
        rag_config = RAGConfig(
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            collection_name=config.milvus_collection,
        )
        init_rag_system(api_key=api_key, config=rag_config, tenant_id=config.tenant_id)
    else:
        logger.info("Milvus is disabled, skipping init_rag_system.")
    init_search_from_config(
        api_key=api_key,
        serper_api_key=config.serper_api_key,
        tavily_api_key=config.tavily_api_key,
        bocha_api_key=config.bocha_api_key,
        search_backends=config.search_backends,
        search_fallback_backends=config.search_fallback_backends,
        search_count=config.search_count,
        search_timeout=config.search_timeout,
        search_fetch_timeout=config.search_fetch_timeout,
        search_max_workers=config.search_max_workers,
        search_max_fetch_pages=config.search_max_fetch_pages,
        search_cache_enabled=config.search_cache_enabled,
        search_cache_ttl_seconds=config.search_cache_ttl_seconds,
        search_cache_max_entries=config.search_cache_max_entries,
        search_rewrite_enabled=config.search_rewrite_enabled,
        search_fetch_enabled=config.search_fetch_enabled,
    )
    small_model = config.small_model
    node_overrides_raw = config.node_model_overrides
    node_overrides: dict = {}
    if isinstance(node_overrides_raw, dict):
        node_overrides = node_overrides_raw
    elif isinstance(node_overrides_raw, str) and node_overrides_raw.strip():
        try:
            node_overrides = json.loads(node_overrides_raw)
        except json.JSONDecodeError:
            logger.warning("NODE_MODEL_OVERRIDES is not valid JSON, ignoring it")
            node_overrides = {}
    default_small_nodes = {"direct_responder", "intent_router", "planner", "analyst"}

    def _agent_model(node_name: str) -> str:
        if node_name in node_overrides:
            return str(node_overrides[node_name])
        if node_name in default_small_nodes:
            return small_model
        return model

    return AgentBundle(
        intent_router=build_agent(_agent_model("intent_router"), api_key, "intent_router", 0.0, []),
        planner=build_agent(_agent_model("planner"), api_key, "plan", 0.3, []),
        scout_web=build_agent(_agent_model("scout_web"), api_key, "web_search", 0.4, []),
        scout_local=build_agent(_agent_model("scout_local"), api_key, "local_rag", 0.4, []),
        evidence_judge=build_agent(_agent_model("evidence_judge"), api_key, "deep_dive", 0.2, []),
        analyst=build_agent(_agent_model("analyst"), api_key, "analyze", 0.3, []),
        direct_responder=build_agent(_agent_model("direct_responder"), api_key, "direct_answer", 0.2, []),
        writer=build_agent(_agent_model("writer"), api_key, "write", 0.4, []),
    )


# ── 基础设施构建 ───────────────────────────────────────────────

def build_memory_manager(config: AppConfig) -> Optional[MemoryManager]:
    if not config.enable_memory:
        return None
    try:
        return MemoryManager(
            short_term_ttl=config.short_term_ttl_seconds,
            short_term_max_messages=config.short_term_max_messages,
            short_term_summary_threshold=config.short_term_summary_threshold,
            tenant_id=config.tenant_id,
            short_term_backend=config.short_term_backend,
            long_term_backend=config.long_term_backend,
            long_term_scope=config.long_term_scope,
            save_conversation_task=config.save_conversation_task,
            enable_milvus=config.enable_milvus,
            redis_url=config.redis_url,
            postgres_dsn=config.postgres_dsn,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            milvus_collection=config.milvus_collection,
            embedding_api_key=config.api_key,
            summary_model=config.summary_model,
        )
    except Exception as exc:
        logger.exception("初始化 MemoryManager 失败，已禁用外部记忆: %s", exc)
        return None


def build_checkpointer(config: AppConfig) -> tuple:
    """Build checkpointer. Returns (checkpointer, context_manager_or_None).

    The caller must call context.__exit__(None, None, None) on shutdown
    if context is not None.
    """
    backend = config.checkpointer_backend
    dsn = config.postgres_dsn
    context = None

    if backend in {"postgres", "auto"} and config.enable_memory and dsn:
        postgres_saver = None
        postgres_import_error = ""
        try:
            module = importlib.import_module("langgraph.checkpoint.postgres")
            postgres_saver = getattr(module, "PostgresSaver", None)
        except Exception as exc:
            postgres_import_error = str(exc)
        if postgres_saver is None:
            try:
                module = importlib.import_module("langgraph_checkpoint_postgres")
                postgres_saver = getattr(module, "PostgresSaver", None)
            except Exception as exc:
                postgres_import_error = postgres_import_error or str(exc)
        if postgres_saver is None:
            logger.warning(
                "PostgreSQL checkpointer 不可用: %s，降级到 memory",
                postgres_import_error,
            )
        else:
            try:
                context = postgres_saver.from_conn_string(dsn)
                return context.__enter__(), context
            except Exception as exc:
                logger.warning("PostgreSQL checkpointer 连接失败: %s", exc)

    if backend in {"redis", "auto"} and config.enable_memory and config.redis_url:
        redis_saver = None
        redis_import_error = ""
        try:
            module = importlib.import_module("langgraph.checkpoint.redis")
            redis_saver = getattr(module, "RedisSaver", None)
        except Exception as exc:
            redis_import_error = str(exc)
        if redis_saver is None:
            try:
                module = importlib.import_module("langgraph_checkpoint_redis")
                redis_saver = getattr(module, "RedisSaver", None)
            except Exception as exc:
                redis_import_error = redis_import_error or str(exc)
        if redis_saver is None:
            logger.warning(
                "Redis checkpointer 不可用: %s，降级到 memory",
                redis_import_error,
            )
        else:
            try:
                context = redis_saver.from_conn_string(config.redis_url)
                return context.__enter__(), context
            except Exception as exc:
                logger.warning("Redis checkpointer 连接失败: %s", exc)

    return InMemorySaver(), None


# ── CLI 参数解析 ────────────────────────────────────────────────

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Research CLI")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--once", type=str, dest="once_query", default=None, help="单次查询后退出")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig.from_file(args.config) if args.config else AppConfig.from_env()


# ── 查询执行 ───────────────────────────────────────────────────

def run_query(app, config: AppConfig, query: str,
              memory_manager: Optional[MemoryManager] = None) -> str:
    memory_context = ""
    if memory_manager:
        try:
            memory_context = memory_manager.build_personalized_prompt_context(
                user_id=config.user_id,
                thread_id=config.thread_id,
                query=query,
                tenant_id=config.tenant_id,
                max_memories=config.memory_top_k,
            )
        except Exception as exc:
            logger.warning("%s 读取记忆失败，忽略本轮注入: %s",
                           colorize("[memory]", "yellow"), exc)

    state = create_initial_state(
        query=query,
        max_iterations=config.max_iterations,
        user_id=config.user_id,
        tenant_id=config.tenant_id,
        memory_context=memory_context,
    )
    result = app.invoke(
        state,
        {"configurable": {"thread_id": config.thread_id}},
    )
    final = result["final"]

    if memory_manager:
        try:
            memory_manager.persist_turn(
                tenant_id=config.tenant_id,
                user_id=config.user_id,
                thread_id=config.thread_id,
                query=query,
                answer=final,
            )
        except Exception as exc:
            logger.warning("%s 持久化记忆失败，已跳过: %s",
                           colorize("[memory]", "yellow"), exc)

    return final


def read_user_input(prompt: str = "你: ") -> str:
    try:
        return input(prompt)
    except UnicodeDecodeError:
        print(prompt, end="", flush=True)
        raw = sys.stdin.buffer.readline()
        if raw == b"":
            raise EOFError
        encoding = sys.stdin.encoding or "utf-8"
        recovered = raw.decode(encoding, errors="replace").rstrip("\r\n")
        logger.warning("%s 检测到输入编码异常，已使用容错解码。",
                       colorize("[input]", "yellow"))
        return recovered


# ── CLI 入口 ───────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_cli_args()
    config = build_runtime_config(args)

    memory_manager = build_memory_manager(config)
    agents = build_agents(config.model, config.api_key, config)
    checkpointer, checkpointer_context = build_checkpointer(config)
    app = build_workflow_app(agents, checkpointer, memory_manager=memory_manager)

    try:
        if args.once_query:
            response = run_query(app, config, args.once_query,
                                 memory_manager=memory_manager)
            print(f"\nAI: {response}\n")
        else:
            while True:
                try:
                    query = read_user_input("你: ").strip()
                except EOFError:
                    break
                if not query:
                    continue
                if query.lower() in {"quit", "exit", "退出"}:
                    break
                if query.lower() in {"/memory", "memory-status"} and memory_manager:
                    print(json.dumps(memory_manager.get_memory_stats(config.user_id),
                                     ensure_ascii=False, indent=2))
                    continue
                if query.lower() == "/memory-vacuum" and memory_manager:
                    result = memory_manager.vacuum(user_id=config.user_id)
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    continue
                if query.lower() == "/memory-trace" and memory_manager:
                    print(json.dumps(memory_manager.get_last_trace(),
                                     ensure_ascii=False, indent=2))
                    continue
                response = run_query(app, config, query,
                                     memory_manager=memory_manager)
                print(f"\nAI: {response}\n")
    finally:
        if checkpointer_context:
            checkpointer_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
