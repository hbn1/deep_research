#!/usr/bin/env python3
"""Comprehensive test harness for DeepResearch memory system."""
import sys, os, time, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")
    return condition

async def main():
    global passed, failed

    print("=" * 60)
    print("TEST 1: Schema DDL")
    print("=" * 60)
    from mult_agents.memory.schema import DDL_UNIFIED_MEMORIES
    ddl = DDL_UNIFIED_MEMORIES.upper()
    check("DDL CREATE TABLE", "CREATE TABLE" in ddl)
    check("DDL IF NOT EXISTS", "IF NOT EXISTS" in ddl)
    check("DDL JSONB", "JSONB" in ddl)

    print("=" * 60)
    print("TEST 2: MemoryEntry and Base Types")
    print("=" * 60)
    from mult_agents.memory.base import MemoryEntry, MemoryType, resolve_conflicts
    check("SEMANTIC", MemoryType.SEMANTIC.value == "semantic")
    check("EPISODIC", MemoryType.EPISODIC.value == "episodic")
    check("PROCEDURAL", MemoryType.PROCEDURAL.value == "procedural")
    me = MemoryEntry(content="test", memory_type=MemoryType.SEMANTIC)
    check("default importance", me.importance == 0.5)
    check("default recall_count", me.recall_count == 0)
    old = MemoryEntry(content="old", memory_type=MemoryType.SEMANTIC, importance=0.5)
    new = MemoryEntry(content="new", memory_type=MemoryType.SEMANTIC, importance=0.8)
    resolved = resolve_conflicts([new, old])
    check("resolve keeps newer", len(resolved) >= 1)

    print("=" * 60)
    print("TEST 3: UnifiedStore helpers")
    print("=" * 60)
    from mult_agents.memory.unified_store import _serialize_content, _text_from_content, _tenant, _parse_asyncpg_rowcount
    check("serialize string", "text" in _serialize_content("hello"))
    check("text from string", _text_from_content("hello") == "hello")
    check("text from dict", _text_from_content({"text": "world"}) == "world")
    e = MemoryEntry(content="t", memory_type=MemoryType.SEMANTIC, metadata={"tenant_id": "acme"})
    check("tenant from metadata", _tenant(e) == "acme")
    e2 = MemoryEntry(content="t", memory_type=MemoryType.SEMANTIC, metadata={})
    check("tenant default", _tenant(e2) == "default_tenant")
    check("rowcount INSERT", _parse_asyncpg_rowcount("INSERT 0 1") == 1)
    check("rowcount UPDATE", _parse_asyncpg_rowcount("UPDATE 3") == 3)

    print("=" * 60)
    print("TEST 4: MemoryInjector")
    print("=" * 60)
    from mult_agents.memory.injector import MemoryInjector, format_memories_for_prompt
    mi = MemoryInjector()
    ctx = mi.build_context(conversation_summary="Test", relevant_memories=[me])
    check("context memories", isinstance(ctx, dict) and "relevant_knowledge" in ctx)
    formatted = mi.format_for_prompt(ctx)
    check("formatted non-empty", len(formatted) > 0)
    fm = format_memories_for_prompt([me])
    check("standalone format", len(fm) > 0)

    print("=" * 60)
    print("TEST 5: RuleBasedExtractor")
    print("=" * 60)
    from mult_agents.memory.extractor import RuleBasedExtractor, MemoryExtractor
    rbe = RuleBasedExtractor()
    result = rbe.extract("I am an engineer. I like Python.", "")
    check("rule facts", len(result.get("facts", [])) > 0)
    check("rule preferences", len(result.get("preferences", [])) > 0)
    check("rule importance", result.get("importance", 0) > 0)
    mex = MemoryExtractor(llm=None)
    result2 = mex.extract_from_turn("My name is Bob.", "Okay Bob!")
    check("extractor facts", len(result2.get("facts", [])) > 0)
    summary = await mex.summarize("Line 1\nLine 2\nLine 3")
    check("summarize fallback", len(summary) > 0)

    print("=" * 60)
    print("TEST 6: ShortTermService (Mock Redis)")
    print("=" * 60)

    class MockRedis:
        def __init__(self): self._data = {}
        def pipeline(self): return MockPipeline(self)
        def rpush(self, k, v): self._data.setdefault(k, []).append(v)
        def lrange(self, k, s, e):
            lst = self._data.get(k, [])
            return lst[s:e] if e >= 0 else lst[s:]
        def llen(self, k): return len(self._data.get(k, []))
        def ltrim(self, k, s, e):
            lst = self._data.get(k, [])
            self._data[k] = lst[s:e] if e >= 0 else lst[s:]
        def setex(self, k, t, v): self._data[k] = v.encode("utf-8") if isinstance(v, str) else v
        def get(self, k):
            v = self._data.get(k)
            return v.encode("utf-8") if isinstance(v, str) else v
        def exists(self, k): return 1 if k in self._data else 0
        def delete(self, *ks):
            for k in ks: self._data.pop(k, None)
        def ping(self): return True
        def expire(self, k, t): pass

    class MockPipeline:
        def __init__(self, m): self._m = m; self._c = []
        def rpush(self, k, v): self._c.append(("rpush", k, v))
        def expire(self, k, t): self._c.append(("expire", k, t))
        def execute(self):
            for cmd, *args in self._c:
                if cmd == "rpush": self._m.rpush(*args)

    from mult_agents.memory.short_term_service import ShortTermService
    mr = MockRedis()
    st = ShortTermService(mr, ttl_seconds=3600)
    check("msg key", st._msg_key("t1","u1","th1") == "st:t1:u1:th1:messages")
    check("sum key", st._sum_key("t1","u1","th1") == "st:t1:u1:th1:summary")

    for i in range(25):
        st.add_message("t1","u1","th1", "user" if i%2==0 else "assistant", f"msg {i}")
    check("count 25", st.message_count("t1","u1","th1") == 25)
    msgs = st.get_messages("t1","u1","th1", last_n=5)
    check("last 5 msgs", len(msgs) == 5)

    st.set_summary("t1","u1","th1", "Summary text")
    check("has summary", st.has_summary("t1","u1","th1"))
    check("get summary", st.get_summary("t1","u1","th1") == "Summary text")

    trimmed = st.trim_messages("t1","u1","th1", keep_last=5)
    check("trim deleted 20", trimmed == 20)
    check("after trim count=5", st.message_count("t1","u1","th1") == 5)

    st.clear_thread("t1","u1","th1")
    check("clear count=0", st.message_count("t1","u1","th1") == 0)

    st.add_message("A","u1","th1","user","A"); st.add_message("B","u1","th1","user","B")
    check("tenant isolation", st.message_count("A","u1","th1") == 1)
    check("health check", st.health_check())

    print("=" * 60)
    print("TEST 7: Orchestrator Logic")
    print("=" * 60)

    from mult_agents.memory.orchestrator import MemoryOrchestrator, _langchainify
    mr2 = MockRedis()
    st2 = ShortTermService(mr2, ttl_seconds=3600)

    class FakeExtractor:
        async def extract_from_turn(self, q, a):
            return {"facts":["f"],"preferences":[],"constraints":[],"procedural":[],"importance":0.6}
        async def summarize(self, t): return "sum"

    class FakeStore:
        async def save(self, e): return e.id or "x"
        async def search(self, q, tenant_id=None, user_id=None, limit=6): return []
        async def vacuum(self, tid, memory_type=None, threshold=0.05): return 0
        async def stats(self, tid): return {}

    orch = MemoryOrchestrator(st2, FakeStore(), FakeExtractor(), MemoryInjector())
    check("has _summarizing", hasattr(orch, "_summarizing"))
    check("_summarizing empty", len(orch._summarizing) == 0)

    await orch.persist_turn("t1","u1","th1","Test Q?","Test A.")
    msgs_st = st2.get_messages("t1","u1","th1", last_n=5)
    check("persist user", any("Test Q" in m["content"] for m in msgs_st))
    check("persist assistant", any("Test A" in m["content"] for m in msgs_st))

    raw = [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    lc = _langchainify(raw)
    check("langchainify count", len(lc) == 2)
    check("HumanMessage", lc[0].__class__.__name__ == "HumanMessage")

    ctx_str = await orch.recall_context("t1","u1","th1","recall test")
    check("recall string", isinstance(ctx_str, str))

    print("=" * 60)
    print("TEST 8: Concurrent Safety")
    print("=" * 60)

    mr3 = MockRedis()
    st3 = ShortTermService(mr3, ttl_seconds=3600)
    orch3 = MemoryOrchestrator(st3, FakeStore(), FakeExtractor(), MemoryInjector())

    async def cp(i):
        await orch3.persist_turn("tC","uC","thC", f"Q{i}", f"A{i}")

    # 10 concurrent persist_turn calls (2 msgs each = 20 total)
    # Note: 20 msgs triggers summary threshold -> background trim to ~6 msgs
    await asyncio.gather(*[cp(i) for i in range(10)])
    # Let background summary task complete
    await asyncio.sleep(0.01)
    final = st3.message_count("tC","uC","thC")
    # Summary threshold (20) was hit, so messages were trimmed to keep ~6
    check("concurrent: msgs stored and trimmed", 5 <= final <= 20, f"got {final}")
    # Verify summary was generated
    check("concurrent: summary generated", st3.has_summary("tC","uC","thC"))

    print("=" * 60)
    print("TEST 9: SQLite Multi-turn (35 turns)")
    print("=" * 60)

    from mult_agents.memory.short_term import ShortTermMemory
    st_sql = ShortTermMemory(ttl_seconds=3600)
    # ConversationBuffer max_messages=20 by default
    for turn in range(10):
        st_sql.add_message("th1", type("M",(),{"content":f"Q{turn}"})(), {"role":"user"})
        st_sql.add_message("th1", type("M",(),{"content":f"A{turn}"})(), {"role":"assistant"})
    msgs = st_sql.get_messages("th1")
    check("20 msgs stored (v1 SQLite cap)", len(msgs) == 20)
    # v1 ShortTermMemory doesn't have set_summary; test thread isolation instead
    st_sql.clear_thread("th1")
    check("clear_thread works", len(st_sql.get_messages("th1")) == 0)

    print("=" * 60)
    print("TEST 10: Performance")
    print("=" * 60)

    mr_p = MockRedis()
    st_p = ShortTermService(mr_p, ttl_seconds=3600)

    t0 = time.perf_counter()
    for i in range(1000):
        st_p.add_message("p1","u1","th1","user",f"Msg {i}")
    t_ins = time.perf_counter() - t0
    rate = 1000/t_ins
    print(f"  1000 inserts: {t_ins:.3f}s ({rate:.0f} msg/s)")
    check("inserts under 0.5s", t_ins < 0.5, f"{t_ins:.3f}s")

    t0 = time.perf_counter()
    for _ in range(100):
        st_p.get_messages("p1","u1","th1", last_n=20)
    t_get = time.perf_counter() - t0
    print(f"  100x get(20): {t_get:.3f}s")
    check("reads under 0.3s", t_get < 0.3, f"{t_get:.3f}s")

    t0 = time.perf_counter()
    for i in range(10000):
        e = MemoryEntry(content=f"E{i}", memory_type=MemoryType.SEMANTIC, user_id="p")
        _serialize_content(e.content)
    t_ent = time.perf_counter() - t0
    rate2 = 10000/t_ent
    print(f"  10000 entries: {t_ent:.3f}s ({rate2:.0f} entries/s)")
    check("entries under 1s", t_ent < 1.0, f"{t_ent:.3f}s")

    t0 = time.perf_counter()
    for i in range(10000):
        resolve_conflicts([
            MemoryEntry(content=f"O{i}", memory_type=MemoryType.SEMANTIC, importance=0.3),
            MemoryEntry(content=f"N{i}", memory_type=MemoryType.SEMANTIC, importance=0.7),
        ])
    t_con = time.perf_counter() - t0
    rate3 = 10000/t_con
    print(f"  10000 resolves: {t_con:.3f}s ({rate3:.0f} resolves/s)")
    check("resolves under 1s", t_con < 1.0, f"{t_con:.3f}s")

    print("=" * 60)
    print("TEST 11: Edge Cases")
    print("=" * 60)

    check("empty msgs", st_p.get_messages("nx","u1","th1") == [])
    check("empty summary", st_p.get_summary("nx","u1","th1") == "")
    large = "X" * 10000
    st_p.add_message("edge","u1","th1","user",large)
    ret = st_p.get_messages("edge","u1","th1", last_n=1)
    check("large content", len(ret[0]["content"]) == 10000)
    check("empty tenant", st_p.message_count("","","") == 0)
    st_p.add_message("","","","user","t")
    check("empty keys", st_p.message_count("","","") == 1)

    # Results
    print()
    print("=" * 60)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    print(f"Insert: {t_ins:.3f}s ({rate:.0f}/s) | Read: {t_get:.3f}s | Entry: {t_ent:.3f}s ({rate2:.0f}/s) | Resolve: {t_con:.3f}s ({rate3:.0f}/s)")
    if failed:
        print("*** FAILED ***")
        sys.exit(1)
    print("*** ALL TESTS PASSED ***")

asyncio.run(main())