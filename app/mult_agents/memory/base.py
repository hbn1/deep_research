"""
????????????

?????????????????????????
?? MemoryEntry??????BaseMemory?MemoryBackend?EmbeddingProvider?
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


class MemoryType(Enum):
    """??????"""
    SHORT_TERM = "short_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry:
    """
    ????????????

    ????:
        importance: ????? (0-1)?????????
        last_recalled_at: ????????
        recall_count: ???????
    ????:
        compute_retention_score(): ??????
        record_recall(): ????????
    """
    content: Union[str, Dict[str, Any]]
    memory_type: MemoryType
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    importance: float = 0.5
    last_recalled_at: Optional[datetime] = None
    recall_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    id: str = field(default_factory=lambda: str(uuid4()))

    def compute_retention_score(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now()
        ref_time = self.last_recalled_at or self.created_at
        age_days = max((now - ref_time).total_seconds() / 86400.0, 0.0)
        decay_halflife = 7.0 * max(self.importance, 0.01)
        decay = math.exp(-age_days / decay_halflife)
        recall_bonus = min(self.recall_count * 0.05, 0.25)
        return min(self.importance * decay + recall_bonus, 1.0)

    def record_recall(self) -> None:
        self.last_recalled_at = datetime.now()
        self.recall_count += 1
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "namespace": self.namespace,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "importance": self.importance,
            "last_recalled_at": self.last_recalled_at.isoformat() if self.last_recalled_at else None,
            "recall_count": self.recall_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=data.get("id", str(uuid4())),
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            user_id=data.get("user_id"),
            thread_id=data.get("thread_id"),
            namespace=data.get("namespace"),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            importance=data.get("importance", 0.5),
            last_recalled_at=datetime.fromisoformat(data["last_recalled_at"])
                if data.get("last_recalled_at") else None,
            recall_count=data.get("recall_count", 0),
            created_at=datetime.fromisoformat(data["created_at"])
                if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data else datetime.now(),
            expires_at=datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at") else None,
            access_count=data.get("access_count", 0),
        )


class BaseMemory(ABC):
    """????????"""

    def __init__(self, memory_type: MemoryType):
        self.memory_type = memory_type

    @abstractmethod
    def save(self, entry: MemoryEntry) -> str:
        pass

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 5,
        **kwargs
    ) -> List[MemoryEntry]:
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def clear(
        self,
        user_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> int:
        pass

    @abstractmethod
    def list_namespaces(self, user_id: Optional[str] = None) -> List[str]:
        pass


class MemoryBackend(ABC):
    """?????????? - ???????????????"""

    @abstractmethod
    def health_check(self) -> bool:
        pass

    @abstractmethod
    def vacuum(self, before: datetime) -> int:
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        pass


class EmbeddingProvider(ABC):
    """????????? - ?? MD5 ???"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class FallbackEmbeddingProvider(EmbeddingProvider):
    """??????? - ?? SHA256 ??????????????"""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def embed(self, text: str) -> List[float]:
        import hashlib
        if not text:
            return [0.0] * self._dimension
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        embedding = []
        for i in range(self._dimension):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] / 255.0) * 2.0 - 1.0)
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm > 0:
            embedding = [v / norm for v in embedding]
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


class DashScopeEmbeddingProvider(EmbeddingProvider):
    """DashScope Embedding ??? - ??????"""

    def __init__(self, api_key: str, model: str = "text-embedding-v1"):
        self._api_key = api_key
        self._model = model
        self._dimension = 1536  # DashScope text-embedding-v1 ????

    def embed(self, text: str) -> List[float]:
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            embeddings = DashScopeEmbeddings(
                model=self._model,
                dashscope_api_key=self._api_key,
            )
            return embeddings.embed_query(text)
        except Exception:
            fallback = FallbackEmbeddingProvider(self._dimension)
            return fallback.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            embeddings = DashScopeEmbeddings(
                model=self._model,
                dashscope_api_key=self._api_key,
            )
            return embeddings.embed_documents(texts)
        except Exception:
            fallback = FallbackEmbeddingProvider(self._dimension)
            return fallback.embed_batch(texts)

    @property
    def dimension(self) -> int:
        return self._dimension


def resolve_conflicts(entries: List[MemoryEntry]) -> List[MemoryEntry]:
    """??????????? + ??????"""
    if len(entries) <= 1:
        return entries

    def _text_sim(a, b):
        words_a = set(str(a).lower().split())
        words_b = set(str(b).lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    kept = []
    for entry in sorted(entries, key=lambda e: e.created_at, reverse=True):
        dup = False
        for existing in kept:
            if entry.namespace == existing.namespace and _text_sim(
                str(entry.content), str(existing.content)
            ) > 0.85:
                dup = True
                break
        if not dup and entry.compute_retention_score() > 0.05:
            kept.append(entry)
    return kept
