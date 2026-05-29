"""
Agent memory system module.

=== v2 (active, recommended) ===================================
  UnifiedMemoryStore   — PG + Milvus long-term, biz_score fine-rank
  ShortTermService     — Redis only, 7-day TTL conversation buffer
  MemoryOrchestrator   — lightweight recall/persist/vacuum coordinator
  MemoryExtractor      — LLM-driven extraction
  MemoryInjector       — structured prompt context builder

=== v1 (legacy, fallback only) ==================================
  MemoryManager        — SQLite-based, deprecated, kept for fallback
  ShortTermMemory      — in-memory conversation buffer
  SemanticMemoryStore  — merged into UnifiedMemoryStore (v2)
  EpisodicMemoryStore  — merged into UnifiedMemoryStore (v2)
  ProceduralMemoryStore— merged into UnifiedMemoryStore (v2)
"""

import warnings

# ── v2: Current architecture ──────────────────────────────────
from .unified_store import UnifiedMemoryStore
from .short_term_service import ShortTermService
from .orchestrator import MemoryOrchestrator
from .schema import apply_schema, DDL_UNIFIED_MEMORIES

# ── v2-compatible: extractors and injectors ───────────────────
from .extractor import MemoryExtractor, RuleBasedExtractor, extract_memory_from_messages
from .injector import MemoryInjector, format_memories_for_prompt

# ── Base types (shared by v1 and v2) ──────────────────────────
from .base import (
    BaseMemory, MemoryType, MemoryEntry, MemoryBackend,
    EmbeddingProvider, FallbackEmbeddingProvider,
    DashScopeEmbeddingProvider, resolve_conflicts,
)

# ── Legacy (v1): kept for backward compatibility ──────────────
# These are superseded by UnifiedMemoryStore + ShortTermService.
# They remain importable during migration but will be removed
# once v2 is confirmed stable.

from .short_term import ShortTermMemory, ConversationBuffer
from .long_term import (
    LongTermMemory, SQLiteLongTermMemory,
    SemanticMemoryStore, EpisodicMemoryStore,
)
from .manager import MemoryManager
from .procedural import ProceduralMemoryStore
from .backends import (
    ShortTermBackend,
    InMemoryShortTermBackend,
    RedisShortTermBackend,
    PostgresShortTermBackend,
    VectorStoreBackend,
    MilvusBackend,
    NoOpVectorBackend,
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
    "MemoryExtractor",
    "RuleBasedExtractor",
    "MemoryInjector",
    # base
    "BaseMemory", "MemoryType", "MemoryEntry", "MemoryBackend",
    "EmbeddingProvider", "FallbackEmbeddingProvider",
    "DashScopeEmbeddingProvider", "resolve_conflicts",
    # legacy
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
