"""
Long-term memory module.

Provides SemanticMemoryStore, EpisodicMemoryStore, and ProceduralMemoryStore
with SQLite / PostgreSQL backends. Now with importance scoring and recall tracking.
"""

import json as _json
import logging as _logging
import sqlite3
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseMemory, MemoryEntry, MemoryType, resolve_conflicts

_logger = _logging.getLogger("mult_agents.memory")


class BaseLongTermMemory(BaseMemory, ABC):
    """Base class for long-term memory implementations."""

    def __init__(self, memory_type: MemoryType):
        super().__init__(memory_type)
        self._embedding_provider = None

    def set_embedding_provider(self, provider):
        """Inject an EmbeddingProvider for real vector search."""
        self._embedding_provider = provider

    def _calculate_similarity(self, vec1, vec2):
        """Calculate cosine similarity."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        n1 = sum(a * a for a in vec1) ** 0.5
        n2 = sum(b * b for b in vec2) ** 0.5
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)


class SQLiteLongTermMemory(BaseLongTermMemory):
    """SQLite-based long-term memory with importance and recall tracking."""

    def __init__(self, memory_type: MemoryType, db_path: Optional[str] = None):
        super().__init__(memory_type)
        if db_path is None:
            db_path = str(Path(__file__).resolve().parents[2] / "data" / "memory.db")
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_tables()

    def _ensure_db_directory(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    user_id TEXT,
                    namespace TEXT,
                    metadata TEXT,
                    embedding TEXT,
                    importance REAL DEFAULT 0.5,
                    recall_count INTEGER DEFAULT 0,
                    last_recalled_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, memory_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)')
            conn.commit()

        # Schema migration: add new columns if missing
        self._migrate_schema()

    def _migrate_schema(self):
        """Add missing columns for backward compatibility."""
        with self._get_connection() as conn:
            existing = {row[1] for row in conn.execute('PRAGMA table_info(memories)').fetchall()}
            migrations = [
                ('importance', 'REAL DEFAULT 0.5'),
                ('recall_count', 'INTEGER DEFAULT 0'),
                ('last_recalled_at', 'TEXT'),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing:
                    conn.execute('ALTER TABLE memories ADD COLUMN {} {}'.format(col_name, col_def))
                    _logger.info('Migrated: added column %s to memories', col_name)
            conn.commit()

    def save(self, entry: MemoryEntry) -> str:
        if entry.embedding is None and isinstance(entry.content, str) and self._embedding_provider:
            try:
                entry.embedding = self._embedding_provider.embed(str(entry.content))
            except Exception:
                pass
        now = datetime.now().isoformat()
        entry.updated_at = datetime.now()

        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO memories
                (id, content, memory_type, user_id, namespace, metadata, embedding,
                 importance, recall_count, last_recalled_at, created_at, updated_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.id,
                _json.dumps(entry.content, ensure_ascii=False) if isinstance(entry.content, dict) else str(entry.content),
                entry.memory_type.value,
                entry.user_id,
                entry.namespace,
                _json.dumps(entry.metadata, ensure_ascii=False),
                _json.dumps(entry.embedding) if entry.embedding else None,
                entry.importance,
                entry.recall_count,
                entry.last_recalled_at.isoformat() if entry.last_recalled_at else None,
                entry.created_at.isoformat(),
                now,
                entry.access_count,
            ))
            conn.commit()
        return entry.id

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        with self._get_connection() as conn:
            row = conn.execute('SELECT * FROM memories WHERE id = ?', (memory_id,)).fetchone()
        if not row:
            return None
        entry = self._row_to_entry(row)
        entry.record_recall()
        self._update_recall(entry.id)
        return entry

    def search(self, query: str, user_id=None, namespace=None, limit=5, **kwargs) -> List[MemoryEntry]:
        with self._get_connection() as conn:
            conditions = ['memory_type = ?']
            params = [self.memory_type.value]
            if user_id:
                conditions.append('user_id = ?')
                params.append(user_id)
            if namespace:
                conditions.append('namespace = ?')
                params.append(namespace)
            where = ' AND '.join(conditions)
            rows = conn.execute(
                f'SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?',
                params + [max(limit * 3, 20)],
            ).fetchall()

        entries = [self._row_to_entry(row) for row in rows]

        if self._embedding_provider and entries:
            try:
                q_vec = self._embedding_provider.embed(query)
                scored = []
                for e in entries:
                    if e.embedding:
                        sim = self._calculate_similarity(q_vec, e.embedding)
                    else:
                        sim = 0.0
                    relevance = 0.6 * sim + 0.4 * e.compute_retention_score()
                    scored.append((relevance, e))
                scored.sort(key=lambda x: x[0], reverse=True)
                entries = [e for _, e in scored[:limit]]
            except Exception:
                pass

        for e in entries:
            e.record_recall()
            self._update_recall(e.id)

        return resolve_conflicts(entries)[:limit]

    def delete(self, memory_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
            conn.commit()
            return cur.rowcount > 0

    def clear(self, user_id=None, namespace=None) -> int:
        with self._get_connection() as conn:
            conditions = []
            params = []
            if user_id:
                conditions.append('user_id = ?')
                params.append(user_id)
            if namespace:
                conditions.append('namespace = ?')
                params.append(namespace)
            if conditions:
                cur = conn.execute(f'DELETE FROM memories WHERE {" AND ".join(conditions)}', params)
            else:
                cur = conn.execute('DELETE FROM memories')
            conn.commit()
            return cur.rowcount

    def list_namespaces(self, user_id=None) -> List[str]:
        with self._get_connection() as conn:
            if user_id:
                rows = conn.execute(
                    'SELECT DISTINCT namespace FROM memories WHERE user_id = ? AND namespace IS NOT NULL',
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    'SELECT DISTINCT namespace FROM memories WHERE namespace IS NOT NULL'
                ).fetchall()
        return [r[0] for r in rows if r[0]]

    def _row_to_entry(self, row) -> MemoryEntry:
        raw_content = row['content']
        try:
            content = _json.loads(raw_content)
        except (_json.JSONDecodeError, TypeError):
            content = raw_content or ''
        # sqlite3.Row uses dict-style access for existing keys
        keys = row.keys()
        return MemoryEntry(
            id=row['id'],
            content=content,
            memory_type=MemoryType(row['memory_type']),
            user_id=row['user_id'],
            namespace=row['namespace'] if 'namespace' in keys else None,
            metadata=_json.loads(row['metadata']) if row['metadata'] else {},
            embedding=_json.loads(row['embedding']) if row['embedding'] else None,
            importance=row['importance'] if 'importance' in keys and row['importance'] is not None else 0.5,
            recall_count=row['recall_count'] if 'recall_count' in keys else 0,
            last_recalled_at=datetime.fromisoformat(row['last_recalled_at']) if 'last_recalled_at' in keys and row['last_recalled_at'] else None,
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else datetime.now(),
            access_count=row['access_count'] if 'access_count' in keys else 0,
        )

    def _update_recall(self, memory_id: str):
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                'UPDATE memories SET recall_count = recall_count + 1, last_recalled_at = ?, access_count = access_count + 1, updated_at = ? WHERE id = ?',
                (now, now, memory_id),
            )
            conn.commit()

    def vacuum_low_score(self, threshold: float = 0.05) -> int:
        """Remove entries with retention score below threshold."""
        with self._get_connection() as conn:
            rows = conn.execute('SELECT * FROM memories WHERE memory_type = ?', (self.memory_type.value,)).fetchall()
        deleted = 0
        for row in rows:
            entry = self._row_to_entry(row)
            if entry.compute_retention_score() < threshold:
                self.delete(entry.id)
                deleted += 1
        if deleted:
            _logger.info("Vacuumed %d low-score entries from %s", deleted, self.memory_type.value)
        return deleted


class SemanticMemoryStore(SQLiteLongTermMemory):
    """Semantic memory: user profile, facts, knowledge."""

    def __init__(self, db_path=None):
        super().__init__(MemoryType.SEMANTIC, db_path)

    def save_profile(self, user_id, profile_data, merge=True):
        existing = self.get_profile(user_id)
        if merge and existing:
            from .utils import merge_user_profile
            profile_data = merge_user_profile(existing, profile_data)
        entry = MemoryEntry(
            content=profile_data,
            memory_type=MemoryType.SEMANTIC,
            user_id=user_id,
            namespace="user_profile",
            importance=0.85,
            metadata={"type": "profile", "version": "1.0"},
        )
        return self.save(entry)

    def get_profile(self, user_id):
        results = self.search(query="user_profile", user_id=user_id, namespace="user_profile", limit=1)
        if results:
            content = results[0].content
            if isinstance(content, dict):
                return content
            try:
                return _json.loads(content)
            except (_json.JSONDecodeError, TypeError):
                return None
        return None

    def save_fact(self, user_id, fact, category=None):
        entry = MemoryEntry(
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            user_id=user_id,
            namespace=f"facts/{category}" if category else "facts",
            importance=0.5,
            metadata={"category": category},
        )
        return self.save(entry)


class EpisodicMemoryStore(SQLiteLongTermMemory):
    """Episodic memory: historical tasks, execution traces."""

    def __init__(self, db_path=None):
        super().__init__(MemoryType.EPISODIC, db_path)

    def save_task_record(self, user_id, task_type, task_data, outcome=None):
        content = {
            "task_type": task_type,
            "data": task_data,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
        }
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.EPISODIC,
            user_id=user_id,
            namespace=f"tasks/{task_type}",
            importance=0.4,
            metadata={"task_type": task_type, "has_outcome": outcome is not None},
        )
        return self.save(entry)

    def get_similar_tasks(self, user_id, task_description, limit=5):
        return self.search(query=task_description, user_id=user_id, limit=limit)

    def get_task_history(self, user_id, task_type=None, limit=10):
        namespace = f"tasks/{task_type}" if task_type else None
        with self._get_connection() as conn:
            query_str = 'SELECT * FROM memories WHERE memory_type = ? AND user_id = ?'
            params = [self.memory_type.value, user_id]
            if namespace:
                query_str += ' AND namespace = ?'
                params.append(namespace)
            query_str += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)
            rows = conn.execute(query_str, params).fetchall()
        return [self._row_to_entry(row) for row in rows]


# Aliases
LongTermMemory = SQLiteLongTermMemory
