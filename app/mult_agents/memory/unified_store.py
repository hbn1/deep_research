"""Unified long-term memory store backed by PostgreSQL + optional Milvus.

Search flow: Milvus vector coarse-filter -> PG business-score fine-rank.

All public methods enforce tenant_id isolation.
PG operations use asyncpg (native async) with psycopg fallback.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from .base import MemoryEntry, MemoryType
from .schema import apply_schema

logger = logging.getLogger("mult_agents.memory")

_RANK_SQL = """
    (importance * 0.35 +
     (1.0 / (1 + EXTRACT(DAY FROM NOW() - COALESCE(last_recalled_at, created_at)))) * 0.30 +
     LN(recall_count + 1) * 0.15 +
     CASE memory_type
         WHEN 'semantic'   THEN 0.20
         WHEN 'procedural' THEN 0.10
         ELSE 0.0
     END
    ) AS biz_score
"""


class UnifiedMemoryStore:
    """Unified long-term memory: PG (structured) + optional Milvus (vector).

    PG backend: asyncpg (preferred) -> psycopg (fallback).
    All methods enforce tenant_id as the first filter in PG WHERE clauses.
    """

    def __init__(
        self,
        postgres_dsn: str,
        embedding_provider=None,
        milvus_backend=None,
        *,
        auto_apply_schema: bool = True,
    ):
        self._dsn = postgres_dsn
        self._embedder = embedding_provider
        self._vector = milvus_backend
        self._pool = None          # asyncpg pool
        self._use_asyncpg = False

        if auto_apply_schema and postgres_dsn:
            try:
                apply_schema(postgres_dsn)
                logger.info("UnifiedMemoryStore schema applied")
            except Exception as exc:
                logger.warning("Schema apply failed (non-fatal): %s", exc)

    # -- Lifecycle ---------------------------------------------------

    async def _ensure_pool(self) -> None:
        """Lazy-init asyncpg pool. Falls back to psycopg silently."""
        if self._pool is not None:
            return
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            # Quick sanity check
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
            self._use_asyncpg = True
            logger.info("UnifiedMemoryStore using asyncpg (pool size 2-10)")
        except Exception as exc:
            logger.warning("asyncpg unavailable, fallback to psycopg: %s", exc)
            self._pool = None
            self._use_asyncpg = False

    async def close(self) -> None:
        """Close the asyncpg pool if active."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._use_asyncpg = False

    # -- Public API --------------------------------------------------

    async def save(self, entry: MemoryEntry) -> str:
        if not entry.id:
            entry.id = str(uuid4())

        tenant_id = _tenant(entry)
        content_json = _serialize_content(entry.content)

        await self._pg_execute(
            """INSERT INTO unified_memories
               (id, tenant_id, user_id, thread_id, memory_type, namespace,
                content, importance, created_at, updated_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,NOW(),NOW())
               ON CONFLICT (id) DO UPDATE SET
                content=EXCLUDED.content, importance=EXCLUDED.importance, updated_at=NOW()""",
            entry.id, tenant_id, entry.user_id or "", entry.thread_id or "",
            entry.memory_type.value, entry.namespace or "",
            content_json, entry.importance,
        )

        self._fire_and_forget(self._index_milvus(entry))
        return entry.id

    async def save_batch(self, entries: list[MemoryEntry]) -> list[str]:
        for e in entries:
            if not e.id:
                e.id = str(uuid4())

        await self._pg_batch_insert(entries)

        for e in entries:
            self._fire_and_forget(self._index_milvus(e))

        return [e.id for e in entries]

    async def search(
        self,
        query_text: str,
        *,
        tenant_id: str,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 6,
    ) -> list[MemoryEntry]:
        mt = memory_type.value if memory_type else None

        candidate_ids = await self._milvus_coarse(query_text, tenant_id, user_id, mt, limit)

        if candidate_ids:
            entries = await self._pg_ranked_search(candidate_ids, tenant_id, limit)
        else:
            entries = await self._pg_text_search(query_text, tenant_id, user_id, mt, limit)

        if entries:
            await self._pg_execute(
                """UPDATE unified_memories
                   SET recall_count = recall_count + 1,
                       last_recalled_at = NOW(), updated_at = NOW()
                   WHERE id = ANY($1)""",
                [e.id for e in entries],
            )

        return entries

    async def get_by_id(self, memory_id: str, *, tenant_id: str) -> Optional[MemoryEntry]:
        rows = await self._pg_fetch(
            "SELECT * FROM unified_memories WHERE id = $1 AND tenant_id = $2",
            memory_id, tenant_id,
        )
        return _row_to_entry(rows[0]) if rows else None

    async def delete(self, memory_id: str, *, tenant_id: str) -> bool:
        rowcount = await self._pg_execute(
            "DELETE FROM unified_memories WHERE id = $1 AND tenant_id = $2",
            memory_id, tenant_id,
        )
        return bool(rowcount)

    async def vacuum(
        self, tenant_id: str, memory_type: Optional[MemoryType] = None,
        threshold: float = 0.05,
    ) -> int:
        mt_clause = f"AND memory_type = '{memory_type.value}'" if memory_type else ""
        rows = await self._pg_fetch(
            f"SELECT id, importance, created_at FROM unified_memories WHERE tenant_id = $1 {mt_clause}",
            tenant_id,
        )
        now = datetime.now(timezone.utc)
        to_delete = []
        for row in rows:
            created = row["created_at"]
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = max((now - created).total_seconds() / 86400.0, 0.0)
            score = row["importance"] * max(0.0, 1.0 - age_days / 30.0)
            if score < threshold:
                to_delete.append(row["id"])

        if to_delete:
            await self._pg_execute(
                "DELETE FROM unified_memories WHERE id = ANY($1) AND tenant_id = $2",
                to_delete, tenant_id,
            )
        return len(to_delete)

    async def list_namespaces(self, tenant_id: str, user_id: str) -> list[str]:
        rows = await self._pg_fetch(
            "SELECT DISTINCT namespace FROM unified_memories WHERE tenant_id = $1 AND user_id = $2 AND namespace IS NOT NULL",
            tenant_id, user_id,
        )
        return [r["namespace"] for r in rows]

    async def stats(self, tenant_id: str) -> dict[str, Any]:
        rows = await self._pg_fetch(
            "SELECT memory_type, COUNT(*) AS cnt FROM unified_memories WHERE tenant_id = $1 GROUP BY memory_type",
            tenant_id,
        )
        return {r["memory_type"]: r["cnt"] for r in rows}

    # -- Private: PG backend (asyncpg preferred, psycopg fallback) ----

    async def _pg_execute(self, sql: str, *params) -> int:
        """Execute a statement. Returns rowcount."""
        await self._ensure_pool()
        if self._use_asyncpg:
            async with self._pool.acquire() as conn:
                result = await conn.execute(sql, *params)
                # asyncpg returns e.g. "INSERT 0 1"; parse rowcount
                return _parse_asyncpg_rowcount(result)
        return await self._pg_execute_psycopg(sql, *params)

    async def _pg_fetch(self, sql: str, *params) -> list[dict]:
        """Fetch rows as list of dicts."""
        await self._ensure_pool()
        if self._use_asyncpg:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
                return [dict(r) for r in rows]
        return await self._pg_fetch_psycopg(sql, *params)

    async def _pg_ranked_search(
        self, ids: list[str], tenant_id: str, limit: int
    ) -> list[MemoryEntry]:
        await self._ensure_pool()
        if self._use_asyncpg:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT *, {_RANK_SQL} FROM unified_memories "
                    "WHERE id = ANY($1::text[]) AND tenant_id = $2 "
                    "ORDER BY biz_score DESC LIMIT $3",
                    ids, tenant_id, limit,
                )
                return [_row_to_entry(dict(r)) for r in rows]
        return await self._pg_ranked_search_psycopg(ids, tenant_id, limit)

    async def _pg_text_search(
        self, query: str, tenant_id: str, user_id: str,
        memory_type: Optional[str], limit: int,
    ) -> list[MemoryEntry]:
        await self._ensure_pool()
        conds = ["tenant_id = $1", "user_id = $2"]
        params: list = [tenant_id, user_id]
        idx = 3
        if memory_type:
            conds.append(f"memory_type = ${idx}")
            params.append(memory_type)
            idx += 1
        where = " AND ".join(conds)
        params.extend([f"%{query}%", limit])

        if self._use_asyncpg:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT * FROM unified_memories WHERE {where} "
                    f"AND content::text ILIKE ${idx} "
                    f"ORDER BY created_at DESC LIMIT ${idx + 1}",
                    *params,
                )
                return [_row_to_entry(dict(r)) for r in rows]
        return await self._pg_text_search_psycopg(query, tenant_id, user_id, memory_type, limit)

    async def _pg_batch_insert(self, entries: list[MemoryEntry]) -> None:
        await self._ensure_pool()
        if self._use_asyncpg:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for e in entries:
                        await conn.execute(
                            """INSERT INTO unified_memories
                               (id, tenant_id, user_id, thread_id, memory_type, namespace,
                                content, importance, created_at, updated_at)
                               VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,NOW(),NOW())
                               ON CONFLICT (id) DO UPDATE SET
                                content=EXCLUDED.content, importance=EXCLUDED.importance, updated_at=NOW()""",
                            e.id, _tenant(e), e.user_id or "", e.thread_id or "",
                            e.memory_type.value, e.namespace or "",
                            _serialize_content(e.content), e.importance,
                        )
            return
        await self._pg_batch_insert_psycopg(entries)

    # -- psycopg fallback (kept for compat) --------------------------

    async def _pg_execute_psycopg(self, sql: str, *params) -> int:
        import psycopg
        def _run():
            with psycopg.connect(self._dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    conn.commit()
                    return cur.rowcount or 0
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def _pg_fetch_psycopg(self, sql: str, *params) -> list[dict]:
        import psycopg
        def _run():
            with psycopg.connect(self._dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    cols = [desc[0] for desc in cur.description]
                    return [dict(zip(cols, row)) for row in cur.fetchall()]
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def _pg_ranked_search_psycopg(
        self, ids: list[str], tenant_id: str, limit: int
    ) -> list[MemoryEntry]:
        import psycopg
        def _run():
            with psycopg.connect(self._dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT *, {_RANK_SQL} FROM unified_memories "
                        "WHERE id = ANY(%s) AND tenant_id = %s "
                        "ORDER BY biz_score DESC LIMIT %s",
                        (ids, tenant_id, limit),
                    )
                    cols = [desc[0] for desc in cur.description]
                    return [_row_to_entry(dict(zip(cols, row))) for row in cur.fetchall()]
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def _pg_text_search_psycopg(
        self, query: str, tenant_id: str, user_id: str,
        memory_type: Optional[str], limit: int,
    ) -> list[MemoryEntry]:
        import psycopg
        def _run():
            conds = ["tenant_id = %s", "user_id = %s"]
            params: list = [tenant_id, user_id]
            if memory_type:
                conds.append("memory_type = %s")
                params.append(memory_type)
            where = " AND ".join(conds)
            params.extend([f"%{query}%", limit])
            with psycopg.connect(self._dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT * FROM unified_memories WHERE {where} AND content::text ILIKE %s ORDER BY created_at DESC LIMIT %s",
                        params,
                    )
                    cols = [desc[0] for desc in cur.description]
                    return [_row_to_entry(dict(zip(cols, row))) for row in cur.fetchall()]
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def _pg_batch_insert_psycopg(self, entries: list[MemoryEntry]) -> None:
        import psycopg
        def _run():
            with psycopg.connect(self._dsn) as conn:
                with conn.cursor() as cur:
                    for e in entries:
                        cur.execute(
                            """INSERT INTO unified_memories
                               (id, tenant_id, user_id, thread_id, memory_type, namespace,
                                content, importance, created_at, updated_at)
                               VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,NOW(),NOW())
                               ON CONFLICT (id) DO UPDATE SET
                                content=EXCLUDED.content, importance=EXCLUDED.importance, updated_at=NOW()""",
                            (e.id, _tenant(e), e.user_id or "", e.thread_id or "",
                             e.memory_type.value, e.namespace or "",
                             _serialize_content(e.content), e.importance),
                        )
                    conn.commit()
        await asyncio.get_event_loop().run_in_executor(None, _run)

    # -- Private: Milvus --------------------------------------------

    async def _milvus_coarse(
        self, query_text: str, tenant_id: str, user_id: str,
        memory_type: Optional[str], limit: int,
    ) -> list[str]:
        if not self._vector or not self._vector.health_check():
            return []

        filter_meta = {"tenant_id": tenant_id, "user_id": user_id}
        if memory_type:
            filter_meta["memory_type"] = memory_type

        try:
            docs = self._vector.search(query_text, top_k=limit * 10, filter_metadata=filter_meta)
            return [d.metadata.get("pg_id", "") for d in docs if d.metadata.get("pg_id")]
        except Exception as exc:
            logger.warning("Milvus search failed: %s", exc)
            return []

    async def _index_milvus(self, entry: MemoryEntry) -> None:
        if not self._vector or not self._vector.health_check():
            return
        try:
            self._vector.index(
                _text_from_content(entry.content)[:2000],
                metadata={
                    "pg_id": entry.id,
                    "tenant_id": _tenant(entry),
                    "user_id": entry.user_id or "",
                    "memory_type": entry.memory_type.value,
                },
            )
        except Exception as exc:
            logger.warning("Milvus index failed: %s", exc)

    @staticmethod
    def _fire_and_forget(coro):
        try:
            asyncio.create_task(coro)
        except RuntimeError:
            pass


# -- Helpers ---------------------------------------------------------

def _parse_asyncpg_rowcount(result: str) -> int:
    """Parse rowcount from asyncpg result string like 'INSERT 0 1'."""
    try:
        parts = result.split()
        if len(parts) >= 3 and parts[0] in ("INSERT", "UPDATE", "DELETE"):
            return int(parts[-1])
        return 0
    except (ValueError, IndexError):
        return 0


def _tenant(entry: MemoryEntry) -> str:
    return (entry.metadata or {}).get("tenant_id", "default_tenant")


def _serialize_content(content: Any) -> str:
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, str):
        return json.dumps({"text": content}, ensure_ascii=False)
    return json.dumps({"text": str(content)}, ensure_ascii=False)


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "") or content.get("trigger", "") or json.dumps(content, ensure_ascii=False)
    return str(content)


def _row_to_entry(row: dict) -> MemoryEntry:
    content = row.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass
    return MemoryEntry(
        id=row["id"],
        content=content,
        memory_type=MemoryType(row["memory_type"]),
        user_id=row.get("user_id"),
        thread_id=row.get("thread_id"),
        namespace=row.get("namespace"),
        importance=row.get("importance", 0.5),
        recall_count=row.get("recall_count", 0),
        last_recalled_at=row.get("last_recalled_at"),
        created_at=_ensure_dt(row.get("created_at")),
        updated_at=_ensure_dt(row.get("updated_at")),
        metadata={"tenant_id": row.get("tenant_id", "default_tenant")},
    )


def _ensure_dt(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return datetime.now(timezone.utc)
