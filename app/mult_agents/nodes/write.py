import json, logging, re
from datetime import date
from langchain_core.messages import HumanMessage
from ..state import ResearchState
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, invoke_json_agent, _last_content, _load_json)

logger = logging.getLogger('mult_agents')

def _render_fallback_report(state: ResearchState) -> str:
    lines = ["# 调研结果", "", "## 执行摘要", state.get("analysis", "暂无分析结果"), ""]
    lines.append("## 任务规划与假设状态")
    for hypo in state.get("hypotheses", []):
        status = hypo.get("status", "unverified")
        lines.append(f"- {hypo.get('id', 'h')}: {hypo.get('content', '')} | 状态: {status}")
    lines.append("")
    lines.append("## 核心结论")
    for finding in state.get("findings", []):
        refs = "".join(f"[{source_id}]" for source_id in finding.get("source_ids", []))
        lines.append(f"- {finding.get('claim', '')} {refs}".rstrip())
    lines.append("")
    lines.append("## 风险与不确定性")
    if state.get("audit_flags"):
        for flag in state["audit_flags"]:
            lines.append(f"- {flag.get('type')}: {flag.get('reason')} ({flag.get('target')})")
    else:
        lines.append("- 当前未发现明显冲突。")
    lines.append("")
    lines.append("## 检索统计")
    web_stats = state.get("web_retrieval_stats", {})
    local_stats = state.get("local_retrieval_stats", {})
    if web_stats or local_stats:
        lines.append(f"- 网络检索：queries={web_stats.get('query_count', 0)} raw={web_stats.get('raw_count', 0)} kept={web_stats.get('kept_count', 0)} dropped={web_stats.get('dropped_count', 0)}")
        lines.append(f"- 本地检索：queries={local_stats.get('query_count', 0)} raw={local_stats.get('raw_count', 0)} kept={local_stats.get('kept_count', 0)} dropped={local_stats.get('dropped_count', 0)}")
    else:
        lines.append("- 未记录检索统计。")
    lines.append("")
    lines.append("## 引用列表")
    for source in state.get("source_index", []):
        source_type = source.get("source_type", "source")
        lines.append(f"- {source.get('source_id')} [{source_type}]: {source.get('label')} | {source.get('locator')}")
    return "\n".join(lines)




def _build_source_lookup(state: ResearchState) -> dict[str, dict]:
    lookup: dict[str, dict] = {}

    def _put(source_id: str, source_type: str, label: str, locator: str):
        if not source_id:
            return
        item = lookup.get(source_id)
        if not item:
            lookup[source_id] = {
                "source_id": source_id,
                "source_type": source_type or "source",
                "label": label or source_id,
                "locator": locator or "",
            }
            return
        if (not item.get("locator")) and locator:
            item["locator"] = locator
        if (not item.get("label")) and label:
            item["label"] = label
        if item.get("source_type") in {"source", ""} and source_type:
            item["source_type"] = source_type

    for source in state.get("source_index", []):
        _put(
            str(source.get("source_id", "")).strip(),
            str(source.get("source_type", "source")).strip(),
            str(source.get("label", "")).strip(),
            str(source.get("locator", "")).strip(),
        )
    for ev in state.get("evidence_pool", []):
        _put(
            str(ev.get("source_id", "")).strip(),
            str(ev.get("source_type", "source")).strip(),
            str(ev.get("title") or ev.get("source_label") or "").strip(),
            str(ev.get("url") or ev.get("doc_id") or "").strip(),
        )
    for ev in state.get("web_evidence", []):
        _put(
            str(ev.get("source_id", "")).strip(),
            "web",
            str(ev.get("title", "")).strip(),
            str(ev.get("url") or "").strip(),
        )
    for ev in state.get("local_evidence", []):
        _put(
            str(ev.get("source_id", "")).strip(),
            "local",
            str(ev.get("title") or ev.get("doc_id") or "").strip(),
            str(ev.get("doc_id") or "").strip(),
        )
    for key, item in lookup.items():
        if key.startswith("LOC"):
            item["source_type"] = "local"
        elif key.startswith("WEB"):
            item["source_type"] = "web"
    return lookup




def _extract_citation_ids(content: str) -> list[str]:
    """从正文中提取所有引用ID [XXX]"""
    pattern = r'\[([A-Z]+\d+_\d+-\d+)\]'
    matches = re.findall(pattern, content)
    return list(dict.fromkeys(matches))  # 去重保序




def _validate_and_fix_citations(content: str, valid_source_ids: set[str]) -> tuple[str, list[str]]:
    """校验正文中的引用ID，移除非法引用，返回修正后的内容和实际使用的合法引用列表"""
    pattern = r'\[([A-Z]+\d+_\d+-\d+)\]'
    
    def replace_citation(match):
        citation_id = match.group(1)
        if citation_id in valid_source_ids:
            return f"[{citation_id}]"
        else:
            # 非法引用，直接移除
            return ""
    
    fixed_content = re.sub(pattern, replace_citation, content)
    # 提取修正后实际使用的合法引用
    used_ids = [cid for cid in _extract_citation_ids(fixed_content) if cid in valid_source_ids]
    return fixed_content, used_ids




def _render_reference_list(state: ResearchState) -> str:
    lines = ["## 参考资料"]
    lookup = _build_source_lookup(state)
    
    # 1. 优先从正文 draft 中按出现顺序提取实际引用的 source_id
    draft_content = state.get("draft", "") or state.get("final", "")
    cited_ids: list[str] = []
    if draft_content:
        for sid in _extract_citation_ids(draft_content):
            if sid in lookup and sid not in cited_ids:
                cited_ids.append(sid)
    
    # 2. 如果正文无引用，降级到 findings
    if not cited_ids:
        for finding in state.get("findings", []):
            for sid in finding.get("source_ids", []):
                text = str(sid).strip()
                if text and text not in cited_ids and text in lookup:
                    cited_ids.append(text)
    
    # 3. 再降级：全量 lookup
    if not cited_ids:
        cited_ids = list(lookup.keys())
    
    # 4. 对 local 来源按 locator 去重展示（同一文件多个 chunk 只展示一次）
    seen_locators: set[str] = set()
    display_ids: list[str] = []
    web_ids: list[str] = []
    local_ids: list[str] = []
    
    for sid in cited_ids:
        source = lookup.get(sid)
        if not source:
            continue
        source_type = source.get("source_type", "")
        locator = source.get("locator", "").strip()
        
        if source_type == "local":
            # 同一文件路径只保留第一次出现的 source_id 做代表
            dedup_key = locator or sid
            if dedup_key in seen_locators:
                continue
            seen_locators.add(dedup_key)
            local_ids.append(sid)
        else:
            web_ids.append(sid)
    
    # 5. 排列顺序：WEB 在前（保持原始引用顺序），LOCAL 跟后
    display_ids = web_ids + local_ids
    
    if not display_ids:
        display_ids = cited_ids[:15]
    
    for sid in display_ids:
        source = lookup.get(sid)
        if not source:
            continue
        locator = source.get("locator", "").strip()
        label = source.get("label", "").strip()
        source_type = source.get("source_type", "source")
        source_id = source.get("source_id", sid)
        
        if not locator:
            locator = "链接暂不可用" if source_type == "web" else "本地知识库"
        
        lines.append(f"- [{source_id}] [{source_type}]: {label} | {locator}")
    
    if len(lines) == 1:
        lines.append("- 暂无参考资料")
    return "\n".join(lines)




def _render_execution_appendix(state: ResearchState) -> str:
    lines = ["## 规划与检索明细", "", "### 执行概览"]
    search_plan = state.get("search_plan", [])
    web_stats = state.get("web_retrieval_stats", {})
    local_stats = state.get("local_retrieval_stats", {})
    lines.append(f"- 规划生成研究问题数: {len(state.get('research_questions', []))}")
    lines.append(f"- 规划生成搜索步骤数: {len(search_plan)}")
    
    iteration = state.get("iteration", 0)
    lines.append(f"- 经过 {iteration + 1} 轮检索迭代")
    if state.get("needs_more_research"):
        lines.append(f"- 信息缺口: {state.get('missing_gaps', [])}")
        
    lines.append(
        f"- 实际执行网页检索问题数: {web_stats.get('query_count', 0)} | 原始命中: {web_stats.get('raw_count', 0)} | 保留证据: {web_stats.get('kept_count', 0)} | 丢弃: {web_stats.get('dropped_count', 0)}"
    )
    lines.append(
        f"- 实际执行本地检索问题数: {local_stats.get('query_count', 0)} | 原始命中: {local_stats.get('raw_count', 0)} | 保留证据: {local_stats.get('kept_count', 0)} | 丢弃: {local_stats.get('dropped_count', 0)}"
    )
    lines.append("")
    lines.append("### 问题拆解明细")
    for sq in state.get("sub_questions", []):
        lines.append(f"- {sq}")
    if not state.get("sub_questions"):
        lines.append("- 无")
    lines.append("")
    lines.append("### 规划输出")
    outline = state.get("outline", [])
    if outline:
        for section in outline:
            lines.append(
                f"- {section.get('id')}: {section.get('title')} | {section.get('description')} | search_queries={section.get('search_queries', [])}"
            )
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("### 研究问题")
    for index, question in enumerate(state.get("research_questions", []), 1):
        lines.append(f"- Q{index}: {question}")
    if not state.get("research_questions"):
        lines.append("- 无")
    lines.append("")
    lines.append("### 搜索计划")
    for index, item in enumerate(state.get("search_plan", []), 1):
        lines.append(
            f"- S{index}: section={item.get('section_id')} | query={item.get('query')} | source={item.get('source_preference')} | reason={item.get('reason')}"
        )
    if not state.get("search_plan"):
        lines.append("- 无")
    lines.append("")
    if state.get("supplementary_queries"):
        lines.append("### 补搜计划")
        for index, item in enumerate(state.get("supplementary_queries", []), 1):
            lines.append(f"- S{index} (补搜): query={item.get('query')} | reason={item.get('reason')}")
        lines.append("")
    lines.append("### 网页检索明细")
    for index, trace in enumerate(state.get("web_search_trace", []), 1):
        lines.append(
            f"- WQ{index}: section={trace.get('section_id')} | query={trace.get('query')} | reason={trace.get('reason')} | raw={trace.get('raw_count', 0)} | kept={trace.get('kept_count', 0)} | rejected={trace.get('rejected_count', 0)}"
        )
        lines.append(f"  - raw_ids={trace.get('raw_source_ids', [])}")
        lines.append(f"  - kept_ids={trace.get('kept_source_ids', [])}")
        lines.append(f"  - rejected_ids={trace.get('rejected_source_ids', [])}")
        if trace.get("reject_reason"):
            lines.append(f"  - reject_reason={trace.get('reject_reason')}")
        lines.append("  - raw_samples:")
        for item in trace.get("raw_records", [])[:3]:
            lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("kept_records"):
            lines.append("  - kept_samples:")
            for item in trace.get("kept_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("rejected_records"):
            lines.append("  - rejected_samples:")
            for item in trace.get("rejected_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
    if not state.get("web_search_trace"):
        lines.append("- 无")
    lines.append("")
    lines.append("### 本地检索明细")
    for index, trace in enumerate(state.get("local_rag_trace", []), 1):
        lines.append(
            f"- LQ{index}: section={trace.get('section_id')} | query={trace.get('query')} | reason={trace.get('reason')} | raw={trace.get('raw_count', 0)} | kept={trace.get('kept_count', 0)} | rejected={trace.get('rejected_count', 0)}"
        )
        lines.append(f"  - raw_ids={trace.get('raw_source_ids', [])}")
        lines.append(f"  - kept_ids={trace.get('kept_source_ids', [])}")
        lines.append(f"  - rejected_ids={trace.get('rejected_source_ids', [])}")
        if trace.get("reject_reason"):
            lines.append(f"  - reject_reason={trace.get('reject_reason')}")
        lines.append("  - raw_samples:")
        for item in trace.get("raw_records", [])[:3]:
            lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("kept_records"):
            lines.append("  - kept_samples:")
            for item in trace.get("kept_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("rejected_records"):
            lines.append("  - rejected_samples:")
            for item in trace.get("rejected_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
    if not state.get("local_rag_trace"):
        lines.append("- 无")
    return "\n".join(lines)




def _ensure_reference_section(content: str, state: ResearchState) -> str:
    base = content.rstrip()
    references = _render_reference_list(state)
    if "## 引用列表" in base or "## 来源清单" in base or "## 参考资料" in base:
        return base
    return f"{base}\n\n{references}"




def write_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[write]", "cyan"), colorize(agent_name, "magenta"))
    valid_source_ids = [str(item.get("source_id", "")).strip() for item in state.get("source_index", []) if item.get("source_id")]
    valid_source_ids = [item for item in valid_source_ids if item][:80]
    valid_source_ids_set = set(valid_source_ids)

    # Trim findings and source_index to reduce prompt size
    trimmed_findings = []
    for f in (state.get("findings") or [])[:12]:
        trimmed_findings.append({
            "claim_id": f.get("claim_id", ""),
            "claim": str(f.get("claim", ""))[:300],
            "confidence": f.get("confidence", ""),
            "source_ids": (f.get("source_ids") or [])[:4],
        })
    
    trimmed_source_index = [
        {"source_id": s.get("source_id"), "source_type": s.get("source_type"),
         "label": str(s.get("label", ""))[:120], "locator": str(s.get("locator", ""))[:200]}
        for s in (state.get("source_index") or [])
        if s.get("source_id") in valid_source_ids_set
    ][:15]

    if not trimmed_findings or not trimmed_source_index:
        final_content = _render_fallback_report(state)
        emit("write", final_content)
        return {"draft": final_content, "final": final_content, "messages": []}
    
    
    prompt = (
        "Default length: 800-1500 Chinese characters unless the user explicitly asks for a long-form report. Keep high information density and avoid filler.\n"
        f"当前日期：{date.today().isoformat()}；当前年份：{date.today().year}。如用户要求当前、最新、近期或市面上，报告标题和正文必须面向当前年份，禁止无来源地写成 2024/2025。\n"
        "请严格根据以下信息撰写最终的 Markdown 研报。请直接输出正文，绝对不要输出任何 JSON 结构，也不要复述你的指令。\n\n"
        f"核心问题：{state['query']}\n"
        f"子问题拆解：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n\n"
        "【分析结论 (Findings)】：\n"
        f"{json.dumps(trimmed_findings, ensure_ascii=False)}\n\n"
        "【可用来源索引 (source_index)】：\n"
        f"{json.dumps(trimmed_source_index, ensure_ascii=False)}\n\n"
        "【合法引用ID列表】：\n"
        f"{json.dumps(valid_source_ids, ensure_ascii=False)}\n\n"
        "【可能存在的风险/冲突 (Audit Flags)】：\n"
        f"{json.dumps(state.get('audit_flags', []), ensure_ascii=False)}\n\n"
        "要求：正文必须使用合法引用ID（例如 [WEB1_1-1]、[LOC1_1-3]）；禁止使用不存在的编号。"
        "结尾不需要你来列举引用列表，系统会自动拼接。"
    )
    human = HumanMessage(content=with_memory_context(state, prompt))
    
    # 彻底断开之前的 messages 累积，只给模型当前这一条指令，避免被前面的 JSON 带偏
    result = agent.invoke({"messages": [human]})
    content = _last_content(result)
    
    # 强制清理可能的错误 JSON 代码块
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```markdown\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"```$", "", content.strip())
    
    # 校验并修正引用ID，移除非法引用
    content, used_citation_ids = _validate_and_fix_citations(content, valid_source_ids_set)
    
    final_content = _ensure_reference_section(content, state)
    emit("write", final_content)
    return {"draft": final_content, "final": final_content, "messages": [human, result["messages"][-1]]}



