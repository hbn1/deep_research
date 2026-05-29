"""
Agent memory system module.

Provides unified management of short-term and long-term memory:
- Short-term: LangGraph Checkpoint-based thread-level memory
- Long-term: PostgreSQL + Milvus vector-based persistent memory
- Semantic: User profile, facts, knowledge
- Episodic: Historical tasks, execution traces
- Procedural: Agent behavioral patterns, learned strategies
"""

from .base import (
    BaseMemory,
    MemoryType,
    MemoryEntry,
    MemoryBackend,
    EmbeddingProvider,
    FallbackEmbeddingProvider,
    DashScopeEmbeddingProvider,
    resolve_conflicts,
)
from .short_term import ShortTermMemory, ConversationBuffer
from .long_term import (
    LongTermMemory,
    SQLiteLongTermMemory,
    SemanticMemoryStore,
    EpisodicMemoryStore,
)
from .manager import MemoryManager
from .utils import (
    create_memory_checkpoint,
    extract_memory_from_messages,
    format_memories_for_prompt,
    merge_user_profile,
    calculate_memory_relevance,
    compress_memories,
)
from .extractor import MemoryExtractor, RuleBasedExtractor
from .injector import MemoryInjector
from .backends import (
    ShortTermBackend,
    InMemoryShortTermBackend,
    RedisShortTermBackend,
    PostgresShortTermBackend,
    VectorStoreBackend,
    MilvusBackend,
    NoOpVectorBackend,
)
from .procedural import ProceduralMemoryStore

__all__ = [
    # Base types
    "BaseMemory",
    "MemoryType",
    "MemoryEntry",
    "MemoryBackend",
    "EmbeddingProvider",
    "FallbackEmbeddingProvider",
    "DashScopeEmbeddingProvider",
    "resolve_conflicts",
    # Short-term memory
    "ShortTermMemory",
    "ConversationBuffer",
    # Long-term memory
    "LongTermMemory",
    "SQLiteLongTermMemory",
    "SemanticMemoryStore",
    "EpisodicMemoryStore",
    # Procedural memory
    "ProceduralMemoryStore",
    # Manager
    "MemoryManager",
    # Extractors & Injectors
    "MemoryExtractor",
    "RuleBasedExtractor",
    "MemoryInjector",
    # Backends
    "ShortTermBackend",
    "InMemoryShortTermBackend",
    "RedisShortTermBackend",
    "PostgresShortTermBackend",
    "VectorStoreBackend",
    "MilvusBackend",
    "NoOpVectorBackend",
    # Utility functions
    "create_memory_checkpoint",
    "extract_memory_from_messages",
    "format_memories_for_prompt",
    "merge_user_profile",
    "calculate_memory_relevance",
    "compress_memories",
]
