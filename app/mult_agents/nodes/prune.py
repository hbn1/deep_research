"""Context pruning node: prevent state bloat in multi-iteration research loops.

Trims accumulated evidence and messages before they enter the Writer node,
keeping the total token budget within a safe threshold for the target LLM.
"""

import logging
from ..state import ResearchState
from ..utils import colorize, estimate_tokens

logger = logging.getLogger("mult_agents")

# Safe token budget for qwen-plus (131K context window, leave headroom)
SAFE_TOKEN_BUDGET = 12000
MAX_EVIDENCE_ITEMS = 15
MAX_EVIDENCE_CHARS = 500  # Per evidence item
MAX_RECENT_MESSAGES = 6


def prune_context_node(state: ResearchState, agent=None, agent_name: str = "pruner") -> ResearchState:
    """Lightweight pruner: trim evidence and messages before writing.

    Only runs when iteration > 0 (multi-round research). Single-round
    research has minimal bloat and is left untouched.
    """
    iteration = state.get("iteration", 0)
    if iteration <= 0:
        logger.info(
            "%s skipping (single-round, no bloat risk)",
            colorize("[prune]", "green"),
        )
        return {}

    evidence_pool = state.get("evidence_pool", []) or []
    findings = state.get("findings", []) or []
    web_evidence = state.get("web_evidence", []) or []
    local_evidence = state.get("local_evidence", []) or []
    messages = state.get("messages", []) or []
    source_index = state.get("source_index", []) or []

    changes = {}

    # ── 1. Trim evidence_pool to top-N by relevance ──
    if len(evidence_pool) > MAX_EVIDENCE_ITEMS:
        # Sort by final_score if available, else keep first N
        sorted_pool = sorted(
            evidence_pool,
            key=lambda e: e.get("final_score", 0.5) if isinstance(e, dict) else 0.5,
            reverse=True,
        )
        changes["evidence_pool"] = sorted_pool[:MAX_EVIDENCE_ITEMS]
        logger.info(
            "%s evidence_pool trimmed: %d → %d",
            colorize("[prune]", "green"),
            len(evidence_pool),
            MAX_EVIDENCE_ITEMS,
        )

    # ── 2. Truncate large web/local evidence bodies ──
    def _trim_evidence(items: list) -> list:
        trimmed = []
        for item in items:
            if not isinstance(item, dict):
                trimmed.append(item)
                continue
            copy = dict(item)
            for field in ("full_text", "snippet", "content"):
                val = copy.get(field, "")
                if isinstance(val, str) and len(val) > MAX_EVIDENCE_CHARS:
                    copy[field] = val[:MAX_EVIDENCE_CHARS] + "..."
            trimmed.append(copy)
        return trimmed

    if web_evidence:
        changes["web_evidence"] = _trim_evidence(web_evidence)
        logger.info(
            "%s web_evidence truncated to %d chars/entry",
            colorize("[prune]", "green"),
            MAX_EVIDENCE_CHARS,
        )
    if local_evidence:
        changes["local_evidence"] = _trim_evidence(local_evidence)

    # ── 3. Keep only recent messages + summaries ──
    if len(messages) > MAX_RECENT_MESSAGES * 2:
        kept = []
        # Always keep first message (system/context)
        if messages:
            kept.append(messages[0])
        # Keep last N messages
        kept.extend(messages[-(MAX_RECENT_MESSAGES - 1):])
        changes["messages"] = kept
        logger.info(
            "%s messages trimmed: %d → %d",
            colorize("[prune]", "green"),
            len(messages),
            len(kept),
        )

    # ── 4. Trim findings to top relevance ──
    if len(findings) > MAX_EVIDENCE_ITEMS:
        sorted_findings = sorted(
            findings,
            key=lambda f: f.get("relevance", 0.5) if isinstance(f, dict) else 0.5,
            reverse=True,
        )
        changes["findings"] = sorted_findings[:MAX_EVIDENCE_ITEMS]
        logger.info(
            "%s findings trimmed: %d → %d",
            colorize("[prune]", "green"),
            len(findings),
            MAX_EVIDENCE_ITEMS,
        )

    # ── 5. Trim source_index ──
    if len(source_index) > MAX_EVIDENCE_ITEMS:
        changes["source_index"] = source_index[:MAX_EVIDENCE_ITEMS]

    # ── Estimate token savings ──
    if changes:
        total_text = str(state.get("evidence_pool", "")) + str(state.get("findings", ""))
        before = estimate_tokens(total_text)
        total_text_after = str(changes.get("evidence_pool", evidence_pool)) + str(
            changes.get("findings", findings)
        )
        after = estimate_tokens(total_text_after)
        logger.info(
            "%s token estimate: %d → %d (saved %d)",
            colorize("[prune]", "green"),
            before,
            after,
            before - after,
        )

    return changes