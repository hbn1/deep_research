"""
Memory Manager - unified orchestrator for the memory system.
Integrates short-term, semantic, episodic, and procedural memory
with strategy-pattern backends, LLM-driven extraction, and structured injection.
"""

import json as _json
import logging as _logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .base import MemoryEntry, MemoryType, resolve_conflicts
from .short_term import ShortTermMemory
from .long_term import EpisodicMemoryStore, SemanticMemoryStore
from .utils import merge_user_profile
from .extractor import MemoryExtractor, extract_memory_from_messages
from .injector import MemoryInjector, format_memories_for_prompt
from .backends import (
    InMemoryShortTermBackend,
    RedisShortTermBackend,
    PostgresShortTermBackend,
    MilvusBackend,
    NoOpVectorBackend,
)
from .procedural import ProceduralMemoryStore

try:
    import redis as _redis
except Exception:
    _redis = None

try:
    import psycopg as _psycopg
except Exception:
    _psycopg = None

_logger = _logging.getLogger("mult_agents.memory")


class MemoryManager:
    def __init__(
        self,
        short_term_ttl=604800,
        short_term_max_messages=30,
        short_term_summary_threshold=20,
        db_path=None,
        tenant_id="default_tenant",
        short_term_backend="postgres",
        long_term_backend="postgres",
        long_term_scope="user",
        save_conversation_task=False,
        enable_milvus=True,
        redis_url=None,
        postgres_dsn=None,
        milvus_host=None,
        milvus_port=19530,
        milvus_collection="mult_agent_memory",
        embedding_api_key=None,
        embedding_model="text-embedding-v1",
        summary_model="qwen-plus",
    ):
        self.default_tenant_id = tenant_id
        self.short_term_backend_name = short_term_backend.lower()
        self.long_term_backend_name = long_term_backend.lower()
        self.long_term_scope = long_term_scope.lower()
        if self.long_term_scope not in {"user", "thread"}:
            self.long_term_scope = "user"
        self.save_conversation_task = save_conversation_task
        self.enable_long_term = self.long_term_backend_name != "disabled"
        self.enable_milvus = enable_milvus
        self.short_term_ttl = short_term_ttl
        self.short_term_max_messages = short_term_max_messages
        self.short_term_summary_threshold = short_term_summary_threshold

        # init backends
        self._short_term_backend = self._init_short_term_backend(redis_url, postgres_dsn)
        self._vector_backend = self._init_vector_backend(enable_milvus, milvus_host, milvus_port, milvus_collection, embedding_api_key, embedding_model)

        # init stores
        self.semantic = SemanticMemoryStore(db_path=db_path)
        self.episodic = EpisodicMemoryStore(db_path=db_path)
        self.procedural = ProceduralMemoryStore(db_path=db_path)

        # embedding provider
        self._init_embedding_provider(embedding_api_key, embedding_model)

        # LLM
        self._summary_llm = self._init_summary_llm(embedding_api_key, summary_model)
        self._extractor = MemoryExtractor(llm=self._summary_llm)
        self._injector = MemoryInjector()

        self._last_trace = {}
        self._last_milvus_raw_hits = []
        self._postgres_dsn = postgres_dsn
        self._redis_client = None
        if self.short_term_backend_name == "redis":
            self._init_redis_client(redis_url)

        _logger.info("MemoryManager ready | st=%s lt=%s scope=%s milvus=%s",
            self.short_term_backend_name, self.long_term_backend_name,
            self.long_term_scope, self._vector_backend.health_check())

    def _init_short_term_backend(self, redis_url, postgres_dsn):
        if self.short_term_backend_name == "redis" and redis_url:
            return RedisShortTermBackend(redis_url, self.short_term_ttl)
        if self.short_term_backend_name == "postgres" and postgres_dsn:
            return PostgresShortTermBackend(postgres_dsn, self.short_term_ttl)
        return InMemoryShortTermBackend(self.short_term_ttl)

    def _init_vector_backend(self, enable, host, port, collection, api_key, model):
        if enable and host and api_key:
            return MilvusBackend(host, port, collection, api_key, model)
        return NoOpVectorBackend()

    def _init_embedding_provider(self, api_key, model):
        from .base import DashScopeEmbeddingProvider, FallbackEmbeddingProvider
        p = None
        if api_key:
            try:
                p = DashScopeEmbeddingProvider(api_key, model)
            except Exception as exc:
                _logger.warning("DashScope embedding failed: %s", exc)
        if p is None:
            p = FallbackEmbeddingProvider(384)
        self.semantic.set_embedding_provider(p)
        self.episodic.set_embedding_provider(p)
        self.procedural.set_embedding_provider(p)

    def _init_summary_llm(self, api_key, model):
        if not api_key:
            return None
        try:
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(model=model, temperature=0.1, dashscope_api_key=api_key)
        except Exception as exc:
            _logger.warning("Summary LLM failed: %s", exc)
            return None

    def _init_redis_client(self, redis_url):
        if not redis_url or _redis is None:
            return
        try:
            c = _redis.Redis.from_url(redis_url, decode_responses=True)
            c.ping()
            self._redis_client = c
        except Exception:
            try:
                c = _redis.Redis.from_url(redis_url.replace("redis://root:", "redis://:"), decode_responses=True)
                c.ping()
                self._redis_client = c
            except Exception:
                pass

    @staticmethod
    def _serialize_message(message):
        if isinstance(message, HumanMessage):
            return {"role": "human", "content": str(message.content)}
        if isinstance(message, AIMessage):
            return {"role": "ai", "content": str(message.content)}
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": str(message.content)}
        return {"role": "human", "content": str(message.content)}

    @staticmethod
    def _deserialize_message(payload):
        role = payload.get("role", "human")
        content = payload.get("content", "")
        if role == "ai":
            return AIMessage(content=content)
        if role == "system":
            return SystemMessage(content=content)
        return HumanMessage(content=content)

    def _summarize_text(self, existing_summary, history_slice):
        lines = [f"{item.get('role', 'human')}: {item.get('content', '')}" for item in history_slice]
        ht = "\n".join(lines)
        if self._summary_llm is None:
            return (f"{existing_summary}\n{ht}".strip())[-4000:]
        prompt = ("Compress conversation preserving facts, preferences, conclusions:\n"
            f"Existing: {existing_summary or 'none'}\nNew: {ht}\nOutput 100-300 chars.")
        resp = self._summary_llm.invoke([HumanMessage(content=prompt)])
        return str(resp.content).strip()

    # -- Short-term Memory --

    def add_short_term_message(self, thread_id, message, metadata=None, user_id="default_user", tenant_id=None):
        t = tenant_id or self.default_tenant_id
        self._short_term_backend.add_message(t, user_id, thread_id, self._serialize_message(message))

    def add_short_term_messages(self, thread_id, messages, user_id="default_user", tenant_id=None):
        for m in messages:
            self.add_short_term_message(thread_id, m, user_id=user_id, tenant_id=tenant_id)

    def get_short_term_summary(self, thread_id, user_id="default_user", tenant_id=None):
        return self._short_term_backend.get_summary(tenant_id or self.default_tenant_id, user_id, thread_id)

    def get_short_term_messages(self, thread_id, include_summary=True, last_n=None, user_id="default_user", tenant_id=None):
        t = tenant_id or self.default_tenant_id
        raw = self._short_term_backend.get_messages(t, user_id, thread_id)
        if last_n:
            raw = raw[-last_n:]
        msgs = [self._deserialize_message(item) for item in raw]
        if include_summary:
            s = self.get_short_term_summary(thread_id, user_id=user_id, tenant_id=t)
            if s:
                return [SystemMessage(content=f"Conversation summary: {s}"), *msgs]
        return msgs

    def should_inject_long_term(self, user_id, thread_id, tenant_id=None):
        return len(self._short_term_backend.get_messages(tenant_id or self.default_tenant_id, user_id, thread_id)) == 0

    def mark_injection_skipped(self, thread_id, user_id="default_user", tenant_id=None):
        pass

    def update_short_term_metadata(self, thread_id, metadata):
        pass

    def get_short_term_metadata(self, thread_id):
        return {}

    def clear_short_term(self, thread_id):
        self._short_term_backend.clear(self.default_tenant_id, "default_user", thread_id)

    def list_active_threads(self):
        return []

    # -- Long-term Memory --

    def save_user_profile(self, user_id, profile, merge=True, tenant_id=None):
        if not self.enable_long_term:
            return str(uuid4())
        t = tenant_id or self.default_tenant_id
        existing = self.get_user_profile(user_id, tenant_id=t)
        merged = merge_user_profile(existing, profile) if merge and existing else profile
        mid = str(uuid4())
        if self.long_term_backend_name == "postgres" and self._postgres_dsn and _psycopg:
            self._upsert_profile_pg(t, user_id, merged)
        else:
            self.semantic.save_profile(user_id, merged, merge=False)
        self._vector_backend.index(_json.dumps(merged, ensure_ascii=False), {
            "tenant_id": t, "user_id": user_id, "memory_id": mid,
            "memory_type": MemoryType.SEMANTIC.value, "namespace": "user_profile",
            "created_at": datetime.now().isoformat(),
        })
        return mid

    def get_user_profile(self, user_id, tenant_id=None):
        if not self.enable_long_term:
            return None
        t = tenant_id or self.default_tenant_id
        if self.long_term_backend_name == "postgres" and self._postgres_dsn and _psycopg:
            with _psycopg.connect(self._postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT profile FROM user_profiles WHERE tenant_id=%s AND user_id=%s", (t, user_id))
                    row = cur.fetchone()
                    if row:
                        return row[0]
        return self.semantic.get_profile(user_id)

    def save_fact(self, user_id, fact, category=None, tenant_id=None, thread_id=None):
        if not self.enable_long_term:
            return str(uuid4())
        t = tenant_id or self.default_tenant_id
        mid = str(uuid4())
        entry = MemoryEntry(id=mid, content={"text": fact, "category": category or "general"},
            memory_type=MemoryType.SEMANTIC, user_id=user_id, thread_id=thread_id,
            namespace=f"facts/{category or 'general'}", importance=0.5,
            metadata={"tenant_id": t, "category": category or "general"})
        if self.long_term_backend_name == "postgres" and self._postgres_dsn and _psycopg:
            self._insert_memory_pg(entry, summary=fact[:500])
        else:
            self.semantic.save_fact(user_id, fact, category)
        self._vector_backend.index(fact, {
            "tenant_id": t, "user_id": user_id, "memory_id": mid,
            "memory_type": MemoryType.SEMANTIC.value,
            "namespace": f"facts/{category or 'general'}", "thread_id": thread_id,
            "created_at": datetime.now().isoformat(),
        })
        return mid

    def save_task(self, user_id, task_type, task_data, outcome=None, tenant_id=None, thread_id=None):
        if not self.enable_long_term:
            return str(uuid4())
        t = tenant_id or self.default_tenant_id
        mid = str(uuid4())
        content = {"task_type": task_type, "data": task_data, "outcome": outcome, "timestamp": datetime.now().isoformat()}
        entry = MemoryEntry(id=mid, content=content, memory_type=MemoryType.EPISODIC,
            user_id=user_id, thread_id=thread_id, namespace=f"tasks/{task_type}",
            importance=0.4, metadata={"tenant_id": t, "task_type": task_type, "has_outcome": outcome is not None})
        if self.long_term_backend_name == "postgres" and self._postgres_dsn and _psycopg:
            self._insert_memory_pg(entry, summary=str(content)[:500])
        else:
            self.episodic.save_task_record(user_id, task_type, task_data, outcome)
        self._vector_backend.index(_json.dumps(content, ensure_ascii=False), {
            "tenant_id": t, "user_id": user_id, "memory_id": mid,
            "memory_type": MemoryType.EPISODIC.value, "namespace": f"tasks/{task_type}",
            "thread_id": thread_id, "created_at": datetime.now().isoformat(),
        })
        return mid

    def get_task_history(self, user_id, task_type=None, limit=10, tenant_id=None, thread_id=None):
        return self.episodic.get_task_history(user_id, task_type, limit)

    def search_semantic(self, user_id, query, namespace=None, limit=5, tenant_id=None, thread_id=None):
        if not self.enable_long_term:
            return []
        t = tenant_id or self.default_tenant_id
        scoped = thread_id if self.long_term_scope == "thread" else None
        docs = self._vector_backend.search(query, top_k=limit)
        if docs:
            entries = []
            for doc in docs[:limit]:
                meta = doc.metadata
                entries.append(MemoryEntry(id=str(meta.get("memory_id", uuid4())),
                    content=doc.page_content,
                    memory_type=MemoryType(meta.get("memory_type", MemoryType.SEMANTIC.value)),
                    user_id=user_id, namespace=meta.get("namespace"), metadata=meta))
            for e in entries:
                e.metadata["retrieval_source"] = "milvus"
            return entries
        if self.long_term_backend_name == "postgres" and self._postgres_dsn and _psycopg:
            return self._search_postgres(t, user_id, query, MemoryType.SEMANTIC.value, namespace, scoped, limit)
        return self.semantic.search(query=query, user_id=user_id, namespace=namespace, limit=limit)

    def search_similar_tasks(self, user_id, query, limit=5, tenant_id=None, thread_id=None):
        if not self.enable_long_term:
            return []
        t = tenant_id or self.default_tenant_id
        scoped = thread_id if self.long_term_scope == "thread" else None
        docs = self._vector_backend.search(query, top_k=limit)
        if docs:
            entries = []
            for doc in docs[:limit]:
                meta = doc.metadata
                if meta.get("memory_type") != MemoryType.EPISODIC.value:
                    continue
                entries.append(MemoryEntry(id=str(meta.get("memory_id", uuid4())),
                    content=doc.page_content, memory_type=MemoryType.EPISODIC,
                    user_id=user_id, namespace=meta.get("namespace"), metadata=meta))
            for e in entries:
                e.metadata["retrieval_source"] = "milvus"
                e.record_recall()
            return entries
        if self.long_term_backend_name == "postgres" and self._postgres_dsn and _psycopg:
            return self._search_postgres(t, user_id, query, MemoryType.EPISODIC.value, None, scoped, limit)
        return self.episodic.get_similar_tasks(user_id, query, limit)

    def search_all(self, user_id, query, include_short_term=False, short_term_thread_id=None,
                   limit_per_type=5, tenant_id=None, long_term_thread_id=None):
        t = tenant_id or self.default_tenant_id
        results = {
            "semantic": self.search_semantic(query=query, user_id=user_id, limit=limit_per_type, tenant_id=t, thread_id=long_term_thread_id),
            "episodic": self.search_similar_tasks(query=query, user_id=user_id, limit=limit_per_type, tenant_id=t, thread_id=long_term_thread_id),
        }
        if include_short_term and short_term_thread_id:
            msgs = self.get_short_term_messages(thread_id=short_term_thread_id, include_summary=True, last_n=limit_per_type, user_id=user_id, tenant_id=t)
            results["short_term"] = [MemoryEntry(content=str(m.content), memory_type=MemoryType.SHORT_TERM, user_id=user_id, thread_id=short_term_thread_id, metadata={"tenant_id": t}) for m in msgs]
        return results

    # -- Context Building --

    def get_context_for_agent(self, user_id, thread_id, query=None, max_memories=10, tenant_id=None):
        t = tenant_id or self.default_tenant_id
        ctx = {
            "user_profile": self.get_user_profile(user_id, tenant_id=t) if self.long_term_scope == "user" else None,
            "recent_messages": self.get_short_term_messages(thread_id=thread_id, last_n=5, user_id=user_id, tenant_id=t),
            "recent_tasks": self.get_task_history(user_id, limit=3, tenant_id=t, thread_id=thread_id),
            "conversation_summary": self.get_short_term_summary(thread_id, user_id=user_id, tenant_id=t),
            "procedural_patterns": self.procedural.get_relevant_patterns(user_id=user_id, context=query or "", limit=3),
        }
        if query:
            all_m = self.search_all(user_id=user_id, query=query, limit_per_type=max_memories // 2, tenant_id=t, long_term_thread_id=thread_id)
            combined = []
            for mt, entries in all_m.items():
                for e in entries:
                    combined.append((e, mt))
            combined.sort(key=lambda x: x[0].created_at, reverse=True)
            ctx["relevant_memories"] = combined[:max_memories]
        _logger.info("[memory] context | t=%s u=%s th=%s r=%d m=%d tk=%d s=%d p=%d",
            t, user_id, thread_id, len(ctx.get("recent_messages", [])),
            len(ctx.get("relevant_memories", [])), len(ctx.get("recent_tasks", [])),
            len(ctx.get("conversation_summary", "")), len(ctx.get("procedural_patterns", [])))
        return ctx

    def build_personalized_prompt_context(self, user_id, thread_id, query, tenant_id=None, max_memories=8):
        self._last_milvus_raw_hits = []
        ctx = self.get_context_for_agent(user_id=user_id, thread_id=thread_id, query=query, max_memories=max_memories, tenant_id=tenant_id)
        mems = [item[0] for item in ctx.get("relevant_memories", [])]
        injected = self._injector.format_for_prompt(self._injector.build_context(
            user_profile=ctx.get("user_profile"),
            recent_messages=ctx.get("recent_messages"),
            relevant_memories=mems,
            similar_tasks=ctx.get("recent_tasks"),
            conversation_summary=ctx.get("conversation_summary", ""),
            active_procedural=[e.content if isinstance(e.content, dict) else {"trigger": "", "action": str(e.content)} for e in ctx.get("procedural_patterns", [])],
        ))
        trace_items = []
        sc = {}
        for item in mems:
            s = item.metadata.get("retrieval_source", "unknown")
            sc[s] = sc.get(s, 0) + 1
            sn = str(item.content)
            trace_items.append({"id": item.id, "type": item.memory_type.value, "source": s, "namespace": item.namespace, "thread_id": item.thread_id, "snippet": sn[:120] + ("..." if len(sn) > 120 else "")})
        self._last_trace = {"user_id": user_id, "thread_id": thread_id, "query": query, "total_memories": len(mems), "sources": sc, "items": trace_items, "injected_chars": len(injected)}
        return injected

    def get_last_trace(self):
        return self._last_trace.copy()

    # -- Persistence with LLM extraction --

    def persist_turn(self, tenant_id, user_id, thread_id, query, answer):
        self.add_short_term_messages(thread_id=thread_id, messages=[HumanMessage(content=query), AIMessage(content=answer)], user_id=user_id, tenant_id=tenant_id)
        existing_profile = self.get_user_profile(user_id, tenant_id=tenant_id)
        existing_proc = [e.content if isinstance(e.content, dict) else {} for e in self.procedural.get_relevant_patterns(user_id, query, limit=5)]
        extracted = self._extractor.extract_from_turn(query=query, answer=answer, existing_profile=existing_profile, existing_procedural=existing_proc)
        facts = extracted.get("facts", [])
        prefs = extracted.get("preferences", [])
        constraints = extracted.get("constraints", [])
        patterns = extracted.get("procedural", [])
        imp = extracted.get("importance", 0.5)
        for fact in facts:
            self.save_fact(user_id=user_id, fact=fact, category="user_fact", tenant_id=tenant_id, thread_id=thread_id)
        all_prefs = prefs + constraints
        if all_prefs:
            if self.long_term_scope == "user":
                self.save_user_profile(user_id=user_id, profile={"preferences": all_prefs}, merge=True, tenant_id=tenant_id)
            else:
                for p in all_prefs:
                    self.save_fact(user_id=user_id, fact=p, category="user_preference", tenant_id=tenant_id, thread_id=thread_id)
        for pat in patterns:
            self.procedural.learn_pattern(user_id=user_id, trigger=pat.get("trigger", query[:100]), action=pat.get("action", "auto"), context=pat.get("context", ""), importance=imp * 0.7, thread_id=thread_id)
        if self.save_conversation_task:
            self.save_task(user_id=user_id, task_type="conversation", task_data={"query": query}, outcome=answer[:1200], tenant_id=tenant_id, thread_id=thread_id)
        _logger.info("[memory] persisted | t=%s u=%s th=%s facts=%d prefs=%d const=%d pat=%d imp=%.2f", tenant_id, user_id, thread_id, len(facts), len(prefs), len(constraints), len(patterns), imp)

    # -- Maintenance --

    def clear_user_memory(self, user_id, memory_types=None, tenant_id=None):
        t = tenant_id or self.default_tenant_id
        if memory_types is None:
            memory_types = ["semantic", "episodic", "short_term", "procedural"]
        results = {}
        if "semantic" in memory_types:
            results["semantic"] = self.semantic.clear(user_id=user_id)
        if "episodic" in memory_types:
            results["episodic"] = self.episodic.clear(user_id=user_id)
        if "procedural" in memory_types:
            results["procedural"] = self.procedural.clear(user_id=user_id)
        if "short_term" in memory_types:
            self._short_term_backend.clear(t, user_id, "_all_")
            results["short_term"] = 1
        if self.long_term_backend_name == "postgres" and self._postgres_dsn and _psycopg:
            with _psycopg.connect(self._postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM memory_entries WHERE tenant_id=%s AND user_id=%s", (t, user_id))
                    cur.execute("DELETE FROM user_profiles WHERE tenant_id=%s AND user_id=%s", (t, user_id))
                    conn.commit()
        return results

    def vacuum(self, user_id=None, threshold=0.05):
        r = {
            "semantic": self.semantic.vacuum_low_score(threshold),
            "episodic": self.episodic.vacuum_low_score(threshold),
            "procedural": self.procedural.vacuum_low_score(threshold),
        }
        if user_id:
            r["procedural_decay"] = self.procedural.decay_patterns(user_id)
        _logger.info("[memory] vacuum: %s", r)
        return r

    def get_memory_stats(self, user_id=None):
        return {
            "short_term": {"backend": self.short_term_backend_name},
            "semantic": {"namespaces": self.semantic.list_namespaces(user_id)},
            "episodic": {"namespaces": self.episodic.list_namespaces(user_id)},
            "procedural": {"namespaces": self.procedural.list_namespaces(user_id)},
            "backends": {
                "postgres": bool(self._postgres_dsn and _psycopg and self.long_term_backend_name == "postgres"),
                "milvus": self._vector_backend.health_check(),
                "short_term_ok": self._short_term_backend.health_check(),
            },
            "modes": {"short_term": self.short_term_backend_name, "long_term": self.long_term_backend_name, "long_term_scope": self.long_term_scope, "save_conversation_task": self.save_conversation_task},
        }

    # -- PG helpers --

    def _upsert_profile_pg(self, tenant_id, user_id, profile):
        if not self._postgres_dsn or _psycopg is None:
            return
        with _psycopg.connect(self._postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO user_profiles (tenant_id, user_id, profile, updated_at) VALUES (%s, %s, %s::jsonb, NOW()) ON CONFLICT (tenant_id, user_id) DO UPDATE SET profile = EXCLUDED.profile, updated_at = NOW()", (tenant_id, user_id, _json.dumps(profile, ensure_ascii=False)))
                conn.commit()

    def _insert_memory_pg(self, entry, summary=""):
        if not self._postgres_dsn or _psycopg is None:
            return
        with _psycopg.connect(self._postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO memory_entries (id, tenant_id, user_id, thread_id, memory_type, namespace, content, summary, metadata, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, NOW()) ON CONFLICT (id) DO UPDATE SET content=EXCLUDED.content, summary=EXCLUDED.summary, metadata=EXCLUDED.metadata, updated_at=NOW()", (entry.id, entry.metadata.get("tenant_id", self.default_tenant_id), entry.user_id or "default_user", entry.thread_id, entry.memory_type.value, entry.namespace, _json.dumps(entry.content, ensure_ascii=False) if isinstance(entry.content, dict) else _json.dumps({"text": str(entry.content)}, ensure_ascii=False), summary, _json.dumps(entry.metadata, ensure_ascii=False), entry.created_at))
                conn.commit()

    def _search_postgres(self, tenant_id, user_id, query, memory_type, namespace, thread_id, limit):
        entries = []
        if not self._postgres_dsn or _psycopg is None:
            return entries
        try:
            with _psycopg.connect(self._postgres_dsn) as conn:
                with conn.cursor() as cur:
                    conds = ["tenant_id=%s", "user_id=%s", "memory_type=%s"]
                    params = [tenant_id, user_id, memory_type]
                    if namespace:
                        conds.append("namespace=%s")
                        params.append(namespace)
                    if thread_id:
                        conds.append("thread_id=%s")
                        params.append(thread_id)
                    w = " AND ".join(conds)
                    cur.execute(f"SELECT id, content, memory_type, user_id, namespace, metadata, created_at FROM memory_entries WHERE {w} ORDER BY created_at DESC LIMIT %s", params + [max(limit * 3, 20)])
                    rows = cur.fetchall()
            for row in rows:
                try:
                    ct = _json.loads(row[1]) if isinstance(row[1], str) else row[1]
                except _json.JSONDecodeError:
                    ct = row[1]
                entries.append(MemoryEntry(id=row[0], content=ct, memory_type=MemoryType(row[2]) if row[2] else MemoryType.SEMANTIC, user_id=row[3], namespace=row[4], metadata=row[5] if isinstance(row[5], dict) else (_json.loads(row[5]) if row[5] else {}), created_at=datetime.fromisoformat(row[6]) if row[6] else datetime.now()))
            for e in entries:
                e.metadata["retrieval_source"] = "postgres"
                e.record_recall()
            return entries[:limit]
        except Exception as exc:
            _logger.warning("PG search failed: %s", exc)
            return entries
