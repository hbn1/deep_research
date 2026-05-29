"""Memory orchestrator — lightweight coordinator over ShortTerm + LongTerm.

Stateless. Coordinates:
  - persist_turn:   extract memories from each turn, save to long-term
  - recall_context: assemble short-term + long-term context for prompt injection
  - vacuum:         garbage-collect low-retention long-term entries
"""

import asyncio
import logging
from typing import Any, Optional

from .base import MemoryEntry, MemoryType
from .short_term_service import ShortTermService
from .unified_store import UnifiedMemoryStore
from .extractor import MemoryExtractor
from .injector import MemoryInjector

logger = logging.getLogger("mult_agents.memory")


class MemoryOrchestrator:
    """Memory system coordinator.

    Usage:
        orch = MemoryOrchestrator(
            short_term=ShortTermService(redis_client),
            long_term=UnifiedMemoryStore(pg_dsn, embedder, milvus),
            extractor=MemoryExtractor(llm),
            injector=MemoryInjector(),
        )

        # Before each turn
        ctx = await orch.recall_context(tid, uid, thread, query)

        # After each turn
        await orch.persist_turn(tid, uid, thread, query, answer)

        # Periodic maintenance
        await orch.vacuum(tid)
    """

    def __init__(
        self,
        short_term: ShortTermService,
        long_term: UnifiedMemoryStore,
        extractor: MemoryExtractor,
        injector: Optional[MemoryInjector] = None,
    ):
        self._st = short_term
        self._lt = long_term
        self._extractor = extractor
        self._injector = injector or MemoryInjector()

    # ── Recall (before each turn) ──────────────────────────

    async def recall_context(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        query: str,
        *,
        max_memories: int = 6,
        max_short_messages: int = 10,
    ) -> str:
        """Assemble memory context for injection into the next LLM prompt.

        Returns a formatted string suitable for [Memory Context] injection,
        or empty string if no relevant memories are found.
        """
        # 1. Short-term: recent messages + summary
        recent = self._st.get_messages(
            tenant_id, user_id, thread_id, last_n=max_short_messages,
        )
        summary = self._st.get_summary(tenant_id, user_id, thread_id)

        # 2. Long-term: semantic search
        long_entries = await self._lt.search(
            query,
            tenant_id=tenant_id,
            user_id=user_id,
            limit=max_memories,
        )

        # 3. Build structured context
        context = self._injector.build_context(
            recent_messages=_langchainify(recent) if recent else None,
            conversation_summary=summary,
            relevant_memories=long_entries,
        )
        return self._injector.format_for_prompt(context)

    # ── Persist (after each turn) ───────────────────────────

    async def persist_turn(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        query: str,
        answer: str,
    ) -> None:
        """Persist a turn to short-term and long-term memory.

        1. Store messages in Redis (sync, fast)
        2. LLM extract structured memories (async)
        3. Save to UnifiedMemoryStore (async, parallel)
        4. Trigger summary if threshold reached
        """
        # 1. Short-term: store messages
        self._st.add_message(tenant_id, user_id, thread_id, "user", query)
        self._st.add_message(tenant_id, user_id, thread_id, "assistant", answer[:3000])

        # 2. Long-term: LLM extraction (fire-and-forget, non-blocking)
        try:
            extracted = await self._extractor.extract_from_turn(query, answer)
        except Exception as exc:
            logger.warning("Memory extraction failed (non-critical): %s", exc)
            extracted = {"facts": [], "preferences": [], "constraints": [], "procedural": []}

        # 3. Save extracted entries in parallel
        tasks = []
        for fact in extracted.get("facts", []):
            tasks.append(self._lt.save(MemoryEntry(
                content=fact,
                memory_type=MemoryType.SEMANTIC,
                user_id=user_id,
                namespace="facts",
                importance=extracted.get("importance", 0.6),
                metadata={"tenant_id": tenant_id, "source": "extractor"},
            )))
        for pat in extracted.get("procedural", []):
            tasks.append(self._lt.save(MemoryEntry(
                content={
                    "trigger": str(pat.get("trigger", "")),
                    "action": str(pat.get("action", "")),
                    "context": str(pat.get("context", "")),
                    "success_rate": 0.5,
                },
                memory_type=MemoryType.PROCEDURAL,
                user_id=user_id,
                namespace="patterns",
                importance=0.5,
                metadata={"tenant_id": tenant_id},
            )))

        if tasks:
            await asyncio.gather(*tasks)

        # 4. Trigger summary if threshold reached
        count = self._st.message_count(tenant_id, user_id, thread_id)
        summary_threshold = 20
        if count >= summary_threshold and not self._st.has_summary(tenant_id, user_id, thread_id):
            # Run summary generation as background task, don't block the response
            asyncio.create_task(
                self._generate_summary(tenant_id, user_id, thread_id, count)
            )

    # ── Maintenance ─────────────────────────────────────────

    async def vacuum(self, tenant_id: str, threshold: float = 0.05) -> dict[str, int]:
        """Garbage-collect low-retention entries across all memory types."""
        results = {}
        for mt in [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.PROCEDURAL]:
            n = await self._lt.vacuum(tenant_id, memory_type=mt, threshold=threshold)
            results[mt.value] = n
        logger.info("[memory_vacuum] tenant=%s results=%s", tenant_id, results)
        return results

    async def stats(self, tenant_id: str) -> dict[str, Any]:
        """Get memory statistics for a tenant."""
        return await self._lt.stats(tenant_id)

    # ── Internal ────────────────────────────────────────────

    async def _generate_summary(
        self, tenant_id: str, user_id: str, thread_id: str, msg_count: int
    ) -> None:
        """Generate conversation summary via LLM. Non-blocking, best-effort."""
        try:
            msgs = self._st.get_messages(tenant_id, user_id, thread_id, last_n=min(msg_count, 30))
            text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:200]}"
                for m in msgs
            )
            summary = await self._extractor.summarize(text)
            self._st.set_summary(tenant_id, user_id, thread_id, summary)
            # Trim old messages after summary is generated
            self._st.trim_messages(tenant_id, user_id, thread_id, keep_last=10)
            logger.info("[summary] generated for thread=%s", thread_id)
        except Exception as exc:
            logger.warning("[summary] generation failed (non-critical): %s", exc)


def _langchainify(messages: list[dict]) -> list:
    """Convert {role, content} dicts to LangChain message objects for MemoryInjector."""
    from langchain_core.messages import HumanMessage, AIMessage
    result = []
    for m in messages:
        if m["role"] == "user":
            result.append(HumanMessage(content=m["content"]))
        else:
            result.append(AIMessage(content=m["content"]))
    return result
