"""
Strategy pattern backends for memory storage.

Unified abstraction over Redis, PostgreSQL, in-memory stores.
"""

import json as _json
import logging as _logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.documents import Document

from .base import MemoryEntry, MemoryType

_logger = _logging.getLogger("mult_agents.memory")

try:
    import redis as _redis
except Exception:
    _redis = None

try:
    import psycopg as _psycopg
except Exception:
    _psycopg = None


class ShortTermBackend(ABC):
    @abstractmethod
    def add_message(self, tenant_id, user_id, thread_id, payload): pass
    @abstractmethod
    def get_messages(self, tenant_id, user_id, thread_id): pass
    @abstractmethod
    def get_summary(self, tenant_id, user_id, thread_id): pass
    @abstractmethod
    def set_summary(self, tenant_id, user_id, thread_id, summary): pass
    @abstractmethod
    def clear(self, tenant_id, user_id, thread_id): pass
    @abstractmethod
    def health_check(self): pass


class InMemoryShortTermBackend(ShortTermBackend):
    def __init__(self, ttl_seconds=604800):
        from .short_term import ShortTermMemory
        self._store = ShortTermMemory(ttl_seconds=ttl_seconds)
        self.ttl_seconds = ttl_seconds

    def add_message(self, tenant_id, user_id, thread_id, payload):
        from langchain_core.messages import HumanMessage, AIMessage
        role = payload.get("role", "human")
        if role == "ai":
            msg = AIMessage(content=payload.get("content", ""))
        else:
            msg = HumanMessage(content=payload.get("content", ""))
        self._store.add_message(thread_id, msg, {"tenant_id": tenant_id, "user_id": user_id})

    def get_messages(self, tenant_id, user_id, thread_id):
        return self._store.get_messages(thread_id, include_summary=False, last_n=None)

    def get_summary(self, tenant_id, user_id, thread_id):
        msgs = self._store.get_messages(thread_id, include_summary=True, last_n=0)
        if msgs and isinstance(msgs[0], SystemMessage):
            return str(msgs[0].content)
        return ""

    def set_summary(self, tenant_id, user_id, thread_id, summary): pass

    def clear(self, tenant_id, user_id, thread_id):
        self._store.clear_thread(thread_id)

    def health_check(self): return True


class RedisShortTermBackend(ShortTermBackend):
    def __init__(self, redis_url, ttl_seconds=604800):
        self.ttl = ttl_seconds
        self._client = None
        if _redis and redis_url:
            try:
                client = _redis.Redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self._client = client
            except Exception:
                try:
                    fallback = redis_url.replace("redis://root:", "redis://:")
                    client = _redis.Redis.from_url(fallback, decode_responses=True)
                    client.ping()
                    self._client = client
                except Exception:
                    _logger.warning("Redis init failed")

    def _msg_key(self, tenant, user, thread):
        return f"ma:short:{tenant}:{user}:{thread}"

    def _sum_key(self, tenant, user, thread):
        return f"ma:short:summary:{tenant}:{user}:{thread}"

    def add_message(self, tenant_id, user_id, thread_id, payload):
        if not self._client: return
        key = self._msg_key(tenant_id, user_id, thread_id)
        self._client.rpush(key, _json.dumps(payload, ensure_ascii=False))
        self._client.expire(key, self.ttl)

    def get_messages(self, tenant_id, user_id, thread_id):
        if not self._client: return []
        key = self._msg_key(tenant_id, user_id, thread_id)
        raw = self._client.lrange(key, 0, -1) or []
        return [_json.loads(item) for item in raw]

    def get_summary(self, tenant_id, user_id, thread_id):
        if not self._client: return ""
        key = self._sum_key(tenant_id, user_id, thread_id)
        return self._client.get(key) or ""

    def set_summary(self, tenant_id, user_id, thread_id, summary):
        if not self._client: return
        key = self._sum_key(tenant_id, user_id, thread_id)
        self._client.set(key, summary, ex=self.ttl)

    def clear(self, tenant_id, user_id, thread_id):
        if not self._client: return
        self._client.delete(self._msg_key(tenant_id, user_id, thread_id))
        self._client.delete(self._sum_key(tenant_id, user_id, thread_id))

    def health_check(self):
        try: return bool(self._client and self._client.ping())
        except: return False


class PostgresShortTermBackend(ShortTermBackend):
    def __init__(self, dsn, ttl_seconds=604800):
        self.dsn = dsn
        self.ttl = ttl_seconds

    def _connect(self):
        if not self.dsn or not _psycopg: return None
        return _psycopg.connect(self.dsn)

    def add_message(self, tenant_id, user_id, thread_id, payload):
        conn = self._connect()
        if not conn: return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO short_term_messages (id, tenant_id, user_id, thread_id, role, content, created_at) VALUES (%s, %s, %s, %s, %s, %s, NOW())',
                    (str(uuid4()), tenant_id, user_id, thread_id, payload.get("role", "human"), payload.get("content", "")),
                )
                conn.commit()
        finally:
            conn.close()

    def get_messages(self, tenant_id, user_id, thread_id):
        conn = self._connect()
        if not conn: return []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT role, content FROM short_term_messages WHERE tenant_id = %s AND user_id = %s AND thread_id = %s ORDER BY created_at ASC',
                    (tenant_id, user_id, thread_id),
                )
                rows = cur.fetchall()
            return [{"role": row[0], "content": row[1]} for row in rows]
        finally:
            conn.close()

    def get_summary(self, tenant_id, user_id, thread_id):
        conn = self._connect()
        if not conn: return ""
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT summary FROM short_term_summaries WHERE tenant_id = %s AND user_id = %s AND thread_id = %s',
                    (tenant_id, user_id, thread_id),
                )
                row = cur.fetchone()
            return row[0] if row else ""
        finally:
            conn.close()

    def set_summary(self, tenant_id, user_id, thread_id, summary):
        conn = self._connect()
        if not conn: return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO short_term_summaries (tenant_id, user_id, thread_id, summary, updated_at) VALUES (%s, %s, %s, %s, NOW()) ON CONFLICT (tenant_id, user_id, thread_id) DO UPDATE SET summary = EXCLUDED.summary, updated_at = NOW()',
                    (tenant_id, user_id, thread_id, summary),
                )
                conn.commit()
        finally:
            conn.close()

    def clear(self, tenant_id, user_id, thread_id):
        conn = self._connect()
        if not conn: return
        try:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM short_term_messages WHERE tenant_id=%s AND user_id=%s AND thread_id=%s', (tenant_id, user_id, thread_id))
                cur.execute('DELETE FROM short_term_summaries WHERE tenant_id=%s AND user_id=%s AND thread_id=%s', (tenant_id, user_id, thread_id))
                conn.commit()
        finally:
            conn.close()

    def health_check(self):
        try:
            conn = self._connect()
            if conn: conn.close(); return True
        except: pass
        return False


class VectorStoreBackend(ABC):
    @abstractmethod
    def index(self, text, metadata): pass
    @abstractmethod
    def search(self, query, top_k, filter_metadata=None): pass
    @abstractmethod
    def health_check(self): pass


class MilvusBackend(VectorStoreBackend):
    def __init__(self, host, port, collection, embedding_api_key, embedding_model):
        self._store = None
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            embeddings = DashScopeEmbeddings(model=embedding_model, dashscope_api_key=embedding_api_key)
            try:
                from langchain_milvus import Milvus as MV
            except ImportError:
                from langchain_community.vectorstores import Milvus as MV
            self._store = MV(embedding_function=embeddings, collection_name=collection, connection_args={"uri": f"http://{host}:{port}"}, auto_id=True)
        except Exception as exc:
            _logger.warning("Milvus init failed: %s", exc)

    def index(self, text, metadata):
        if not self._store or not text.strip(): return
        try:
            safe = dict(metadata or {})
            safe.setdefault("source", "memory")
            safe.setdefault("doc_id", str(safe.get("memory_id", "")))
            safe.setdefault("title", str(safe.get("namespace", "memory")))
            doc = Document(page_content=text, metadata=safe)
            self._store.add_documents([doc])
        except Exception as exc:
            _logger.warning("Milvus write failed: %s", exc)

    def search(self, query, top_k, filter_metadata=None):
        if not self._store: return []
        try:
            return self._store.similarity_search(query, k=max(top_k * 4, 20))
        except: return []

    def health_check(self): return self._store is not None


class NoOpVectorBackend(VectorStoreBackend):
    def index(self, text, metadata): pass
    def search(self, query, top_k, filter_metadata=None): return []
    def health_check(self): return True
