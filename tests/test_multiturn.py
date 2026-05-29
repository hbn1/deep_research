#!/usr/bin/env python3
"""Multi-turn conversation test with real LLM + v1 MemoryManager (SQLite).

Tests 4 scenarios across 3+ turns each, measuring memory accumulation,
recall accuracy, LLM extraction quality, and per-operation latency.
"""
import sys, os, time, json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from mult_agents.memory import MemoryManager
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

DASHSCOPE_KEY = os.getenv("DASHSCOPE_API_KEY", "")

SCENARIOS = [
    {
        "name": "personalization",
        "desc": "user preferences accumulation and recall",
        "uid": "alice",
        "tid": "pers_1",
        "turns": [
            "Hi, my name is Alice. I work as a backend engineer specializing in Go and Rust.",
            "I prefer concise answers. I strongly dislike Python for production services.",
            "What languages should I use for a new high-performance API service?",
        ],
    },
    {
        "name": "research_continuity",
        "desc": "research topic continuity",
        "uid": "bob",
        "tid": "research_1",
        "turns": [
            "I need to compare PostgreSQL and MySQL for high write throughput workloads.",
            "What indexing strategy would you recommend based on our previous discussion?",
            "Summarize everything we discussed about database selection and indexing.",
        ],
    },
    {
        "name": "cross_session",
        "desc": "cross-session memory recall",
        "uid": "carol",
        "tid": "session_1",
        "turns": [
            "I'm a data scientist working with large time-series datasets using PyTorch.",
            "I prefer visual explanations over text-heavy responses. My favorite color is blue.",
        ],
        "continuation_tid": "session_2",
        "continuation_turns": [
            "Remember me? What do you know about my work and preferences?",
        ],
    },
    {
        "name": "mixed_topics",
        "desc": "topic separation in memory",
        "uid": "dave",
        "tid": "mixed_1",
        "turns": [
            "I love hiking in mountains. My favorite trail is in Yosemite National Park.",
            "For work, I need a Kubernetes deployment strategy for 20+ microservices.",
            "What hiking gear would you recommend and also what about K8s pod security policies?",
        ],
    },
]

def generate_answer(query, llm):
    """Generate a real LLM answer for a query."""
    try:
        prompt = f"Answer concisely in 2-3 sentences. Be helpful and direct.\n\nQuery: {query}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()
    except Exception as e:
        return f"[LLM unavailable: {e}]"

def main():
    print("=" * 70)
    print("MULTI-TURN MEMORY TEST (Real LLM + SQLite MemoryManager)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API:  DashScope ({'connected' if DASHSCOPE_KEY else 'MISSING'})")
    print("=" * 70)

    # Build LLM for answer generation
    answer_llm = None
    if DASHSCOPE_KEY:
        answer_llm = ChatTongyi(model="qwen-turbo", temperature=0.3, dashscope_api_key=DASHSCOPE_KEY)

    # Create temp SQLite DB
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="memtest_")
    db_path = Path(tmpdir) / "memory.db"

    mm = MemoryManager(
        db_path=str(db_path),
        short_term_ttl=3600,
        short_term_max_messages=50,
        short_term_summary_threshold=100,  # high threshold to avoid auto-summary
        short_term_backend="memory",
        long_term_backend="sqlite",
        enable_milvus=False,
        embedding_api_key=DASHSCOPE_KEY,
        summary_model="qwen-turbo",
    )
    print(f"\nMemoryManager: SQLite at {db_path}")
    print(f"Extractor LLM: {'qwen-turbo' if DASHSCOPE_KEY else 'rule-based fallback'}")
    print()

    totals = {"turns": 0, "answer_ms": 0, "persist_ms": 0, "recall_ms": 0}
    all_memories = []

    for si, scenario in enumerate(SCENARIOS):
        print(f"{'='*70}")
        print(f"SCENARIO {si+1}: {scenario['name']} — {scenario['desc']}")
        print(f"{'='*70}")

        uid = scenario["uid"]
        tid = scenario["tid"]

        for ti, query in enumerate(scenario["turns"]):
            totals["turns"] += 1

            # Generate answer
            t0 = time.perf_counter()
            answer = generate_answer(query, answer_llm) if answer_llm else f"Response to: {query[:60]}..."
            t_ans = (time.perf_counter() - t0) * 1000
            totals["answer_ms"] += t_ans

            # Persist to memory (with LLM extraction)
            t0 = time.perf_counter()
            mm.persist_turn(
                tenant_id="default_tenant",
                user_id=uid,
                thread_id=tid,
                query=query,
                answer=answer,
            )
            t_persist = (time.perf_counter() - t0) * 1000
            totals["persist_ms"] += t_persist

            # Short-term buffer size
            st_msgs = mm.get_short_term_messages(tid)
            st_count = len(st_msgs) if st_msgs else 0

            print(f"  Turn {ti+1}: Q=\"{query[:70]}...\"")
            print(f"    A: \"{answer[:80]}...\"")
            print(f"    LLM answer: {t_ans:.0f}ms | Persist: {t_persist:.0f}ms | ST buffer: {len(st_msgs)} msgs")

        # After all turns: test memory recall
        print(f"\n  --- Memory Recall ---")
        last_query = scenario["turns"][-1]
        t0 = time.perf_counter()
        ctx = mm.build_personalized_prompt_context(
            user_id=uid,
            thread_id=tid,
            query=last_query,
            tenant_id="default_tenant",
            max_memories=6,
        )
        t_recall = (time.perf_counter() - t0) * 1000
        totals["recall_ms"] += t_recall

        stats = mm.get_memory_stats(uid)
        # Count namespace entries across all memory types
        total_mems = 0
        for key in ["semantic", "episodic", "procedural"]:
            if key in stats and isinstance(stats[key], dict):
                namespaces = stats[key].get("namespaces", [])
                total_mems += len(namespaces) if isinstance(namespaces, list) else 0
        all_memories.append(total_mems)

        print(f"    Recall latency: {t_recall:.0f}ms")
        print(f"    Long-term entries: {total_mems} (by type: {stats})")
        if ctx:
            lines = ctx.strip().split("\n")
            key_lines = [l for l in lines if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("=")]
            print(f"    Context ({len(ctx)} chars, {len(key_lines)} meaningful lines):")
            for line in key_lines[:8]:
                print(f"      {line[:100]}")
            if len(key_lines) > 8:
                print(f"      ... ({len(key_lines) - 8} more lines)")
        else:
            print(f"    Context: (empty)")

        # Cross-session test
        if "continuation_tid" in scenario:
            ctid = scenario["continuation_tid"]
            print(f"\n  --- Cross-Session Recall (new thread: {ctid}) ---")
            for qi, query in enumerate(scenario["continuation_turns"]):
                t0 = time.perf_counter()
                ctx = mm.build_personalized_prompt_context(
                    user_id=uid,
                    thread_id=ctid,
                    query=query,
                    tenant_id="default_tenant",
                    max_memories=8,
                )
                t = (time.perf_counter() - t0) * 1000
                print(f"    Q: \"{query[:80]}\"")
                print(f"    Recall: {t:.0f}ms | Context: {len(ctx)} chars")
                if ctx:
                    key_lines = [l for l in ctx.strip().split("\n") if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("=")]
                    for line in key_lines[:6]:
                        print(f"      {line[:100]}")

    # ---- Final Summary ----
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    avg_ans = totals["answer_ms"] / max(totals["turns"], 1)
    avg_persist = totals["persist_ms"] / max(totals["turns"], 1)
    avg_recall = totals["recall_ms"] / max(len(SCENARIOS), 1)
    print(f"  Total turns:           {totals['turns']}")
    print(f"  Avg LLM answer:        {avg_ans:.0f}ms")
    print(f"  Avg persist+extract:   {avg_persist:.0f}ms")
    print(f"  Avg memory recall:     {avg_recall:.0f}ms")
    print(f"  Memory entries/turn:   {sum(all_memories)/max(totals['turns'],1):.1f}")
    print(f"  Total memory entries:  {sum(all_memories)}")
    print(f"  DB path:               {db_path}")

    # Cleanup
    # mm.close()  # not available on MemoryManager in this version
    shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
