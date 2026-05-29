"""
Agent memory system module.

v2 architecture:
  - ShortTermService  → Redis only (conversation buffer + summary)
  - UnifiedMemoryStore → PostgreSQL + optional Milvus (long-term)
  - MemoryOrchestrator → lightweight coordinator (Phase 2)
  - MemoryExtractor    → LLM-driven extraction
  - MemoryInjector     → structured prompt context builder

Legacy classes (deprecated, kept for backward compat during migration):
  - MemoryManager, ShortTermMemory, SemanticMemoryStore,
    EpisodicMemoryStore, ProceduralMemoryStore, SQLiteLongTermMemory
"""

# ── v2: New architecture ──────────────────────────────────────

from .short_term_service import ShortTermService
from .orchestrator import MemoryOrchestrator
from .unified_store import UnifiedMemoryStore
from .schema import apply_schema, DDL_UNIFIED_MEMORIES

# ── v2-compatible: extractors and injectors (unchanged) ───────
from .extractor import MemoryExtractor, RuleBasedExtractor, extract_memory_from_messages
from .injector import MemoryInjector, format_memories_for_prompt

# ── Legacy: kept for backward compat during migration ─────────
from .base import (
    BaseMemory, MemoryType, MemoryEntry, MemoryBackend,
    EmbeddingProvider, FallbackEmbeddingProvider,
    DashScopeEmbeddingProvider, resolve_conflicts,
)
from .short_term import ShortTermMemory, ConversationBuffer
from .long_term import (
    LongTermMemory, SQLiteLongTermMemory,
    SemanticMemoryStore, EpisodicMemoryStore,
)
from .manager import MemoryManager
from .procedural import ProceduralMemoryStore
from .backends import (
    ShortTermBackend, InMemoryShortTermBackend,
    RedisShortTermBackend, PostgresShortTermBackend,
    VectorStoreBackend, MilvusBackend, NoOpVectorBackend,
)
from .utils import (
    create_memory_checkpoint, extract_memory_from_messages,
    format_memories_for_prompt, merge_user_profile,
    calculate_memory_relevance, compress_memories,
)

__all__ = [
    # v2
    "UnifiedMemoryStore",
    "ShortTermService",
    "MemoryOrchestrator",
    "apply_schema",
    "DDL_UNIFIED_MEMORIES",
    # v2-compatible
    "MemoryExtractor",
    "RuleBasedExtractor",
    "MemoryInjector",
    # legacy
    "BaseMemory", "MemoryType", "MemoryEntry", "MemoryBackend",
    "EmbeddingProvider", "FallbackEmbeddingProvider",
    "DashScopeEmbeddingProvider", "resolve_conflicts",
    "ShortTermMemory", "ConversationBuffer",
    "LongTermMemory", "SQLiteLongTermMemory",
    "SemanticMemoryStore", "EpisodicMemoryStore",
    "MemoryManager",
    "ProceduralMemoryStore",
    "ShortTermBackend", "InMemoryShortTermBackend",
    "RedisShortTermBackend", "PostgresShortTermBackend",
    "VectorStoreBackend", "MilvusBackend", "NoOpVectorBackend",
    "create_memory_checkpoint", "extract_memory_from_messages",
    "format_memories_for_prompt", "merge_user_profile",
    "calculate_memory_relevance", "compress_memories",
]

