"""Multi-agent tools: Web search, RAG retrieval, utilities.

Only tools with real implementations are kept. Stub/placeholder tools
that return hardcoded strings have been removed.
"""

from datetime import datetime
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from langchain_core.tools import tool

from .rag.core import RAGManager
from .search import (
    SearchConfig,
    init_search,
)

_search_initialized = False
_search_signature: tuple | None = None
logger = logging.getLogger("mult_agents")

# -- RAG init --

_rag_tenant_id: str = "default_tenant"


def init_rag_system(api_key: str, config=None, tenant_id="default_tenant"):
    """Initialize RAG via RAGManager."""
    global _rag_tenant_id
    _rag_tenant_id = tenant_id
    try:
        RAGManager.get(api_key, config, tenant_id=tenant_id)
    except Exception as e:
        print(f"RAG init failed: {e}")


def _get_rag(tenant_id: str | None = None):
    """Get current RAG instance from RAGManager."""
    return RAGManager.get_or_none(tenant_id=tenant_id or _rag_tenant_id)


# -- Search init --

def init_search_from_config(
    api_key: str = "",
    serper_api_key: str = "",
    tavily_api_key: str = "",
    bocha_api_key: str = "",
    search_backends: str = "tavily",
    search_fallback_backends: str = "tavily",
    search_count: int = 3,
    search_timeout: float = 8.0,
    search_fetch_timeout: float = 4.0,
    search_max_workers: int = 4,
    search_max_fetch_pages: int = 3,
    search_cache_enabled: bool = True,
    search_cache_ttl_seconds: int = 3600,
    search_cache_max_entries: int = 1024,
    search_rewrite_enabled: bool = True,
    search_fetch_enabled: bool = True,
):
    """Initialize global enterprise search engine from AppConfig."""
    global _search_initialized, _search_signature
    tenant_id = os.getenv("TENANT_ID", "default_tenant").strip()
    resolved_bocha_key = bocha_api_key or os.getenv("BOCHA_API_KEY", "").strip()
    redis_url = os.getenv("REDIS_URL", "").strip()
    signature = (
        serper_api_key,
        tavily_api_key,
        resolved_bocha_key,
        search_backends,
        search_fallback_backends,
        search_count,
        search_timeout,
        search_fetch_timeout,
        search_max_workers,
        search_max_fetch_pages,
        search_cache_enabled,
        search_cache_ttl_seconds,
        search_cache_max_entries,
        search_rewrite_enabled,
        search_fetch_enabled,
        redis_url,
        tenant_id,
    )
    if _search_initialized and _search_signature == signature:
        return
    config = SearchConfig(
        serper_api_key=serper_api_key,
        tavily_api_key=tavily_api_key,
        bocha_api_key=resolved_bocha_key,
        enabled_backends=[b.strip() for b in search_backends.split(",") if b.strip()],
        fallback_backends=[b.strip() for b in search_fallback_backends.split(",") if b.strip()],
        default_count=search_count,
        request_timeout=search_timeout,
        fetch_timeout=search_fetch_timeout,
        max_workers=search_max_workers,
        max_fetch_pages=search_max_fetch_pages,
        cache_enabled=search_cache_enabled,
        cache_ttl_seconds=search_cache_ttl_seconds,
        cache_max_entries=search_cache_max_entries,
        fetch_enabled=search_fetch_enabled,
        rewrite_enabled=search_rewrite_enabled,
        rewrite_model="qwen-turbo",
    )
    init_search(config, redis_url=redis_url, tenant_id=tenant_id)
    _search_initialized = True
    _search_signature = signature


# -- Core search --

def search_knowledge_base_records(query: str, limit: int = 5, tenant_id: str | None = None) -> list[dict]:
    """Search local knowledge base with full RAG pipeline (hybrid+rerank)."""
    rag = _get_rag(tenant_id=tenant_id)
    if rag is None:
        return []
    try:
        return rag.search_records(query, k=limit)
    except Exception:
        return []


def bocha_web_search_records(query: str, count: int = 8) -> list[dict]:
    """Call Bocha Web Search API directly."""
    api_key = os.getenv("BOCHA_API_KEY", "").strip()
    if not api_key:
        logger.warning("[bocha_web_search] BOCHA_API_KEY not configured")
        return []
    payload = {
        "query": query,
        "summary": True,
        "freshness": "noLimit",
        "count": count,
    }
    request = urllib.request.Request(
        url="https://api.bocha.cn/v1/web-search",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
        result = json.loads(raw)
    except urllib.error.HTTPError as e:
        logger.error("[bocha_web_search] HTTP %s: %s", e.code, e.reason)
        return []
    except urllib.error.URLError as e:
        logger.error("[bocha_web_search] URL error: %s", e.reason)
        return []
    except (json.JSONDecodeError, Exception) as e:
        logger.error("[bocha_web_search] error: %s", e)
        return []

    data = result.get("data", {})
    pages = data.get("webPages", [])
    if isinstance(pages, dict):
        pages = pages.get("value", []) or []
    if not isinstance(pages, list):
        return []

    records = []
    for item in pages:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "") or item.get("displayUrl", "")
        records.append({
            "title": item.get("name", "") or item.get("title", ""),
            "url": url,
            "domain": url.split("://")[-1].split("/")[0] if url else "",
            "snippet": item.get("snippet", "") or item.get("summary", ""),
            "source_type": "web",
            "published_at": item.get("dateLastCrawled", ""),
        })

    return records[:count]


# -- Tools --

@tool
def get_current_time() -> str:
    """Return current datetime as ISO-8601 string."""
    return datetime.now().isoformat()


@tool
def simple_calculator(expression: str) -> str:
    """Evaluate a simple math expression safely."""
    allowed = set("0123456789+-*/().%^ ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters"
    try:
        expr = expression.replace("^", "**")
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def extract_requirements(text: str) -> str:
    """Extract structured requirements from text."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    items = []
    for line in lines:
        if any(line.startswith(p) for p in ["-", "*", "1.", "2.", "3."]):
            items.append(line.lstrip("-* 0123456789."))
    if not items:
        items = lines[:5]
    return "\n".join(f"- {item}" for item in items)


@tool
def outline_from_topics(topics: str) -> str:
    """Generate a markdown outline from topic lines."""
    topic_lines = [t.strip() for t in topics.split("\n") if t.strip()]
    return "\n".join(f"{i}. {t}" for i, t in enumerate(topic_lines, 1))


@tool
def dedupe_lines(text: str) -> str:
    """Remove duplicate lines while preserving order."""
    seen = set()
    lines = []
    for line in text.splitlines():
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines)


@tool
def web_search_stub(query: str) -> str:
    """Web search via enterprise search engine. Returns formatted results."""
    from .search import search
    records = search(query)
    if not records:
        return "No results."
    lines = ["Search results:"]
    for idx, record in enumerate(records, 1):
        lines.append(f"{idx}. {record['title']}")
        url = record.get("url", "")
        if url:
            lines.append(f"   URL: {url}")
        snippet = record.get("snippet", "")
        if snippet:
            lines.append(f"   Snippet: {snippet[:200]}")
    return "\n".join(lines)


@tool
def search_knowledge_base(query: str, limit: int = 5) -> str:
    """Search local knowledge base. Returns formatted results."""
    records = search_knowledge_base_records(query, limit=limit)
    if not records:
        return "No results in local knowledge base."
    lines = ["Local knowledge base results:"]
    for idx, record in enumerate(records, 1):
        title = record.get("title") or record.get("doc_id", f"Doc-{idx}")
        snippet = record.get("snippet", "") or record.get("content", "")
        lines.append(f"{idx}. {title}")
        if snippet:
            lines.append(f"   {snippet[:300]}")
    return "\n".join(lines)


@tool
def merge_notes(note_a: str, note_b: str) -> str:
    """Merge two note strings."""
    return f"{note_a}\n{note_b}".strip()


@tool
def summarize_points(text: str) -> str:
    """Extract key points from text."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    points = sentences[:6]
    return "\n".join(f"- {p}" for p in points)


# -- File operations (workspace sandbox) --

def _workspace_root() -> Path:
    base = os.getenv("WORKSPACE_DIR", "/workspace")
    return Path(base).resolve()


def _safe_path(path: str) -> Path:
    root = _workspace_root()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Path outside workspace")
    return target


@tool
def safe_list_dir(path: str = ".") -> str:
    """List directory contents within workspace."""
    root = _workspace_root()
    if not root.exists():
        return f"Workspace not found: {root}"
    target = _safe_path(path)
    if not target.exists() or not target.is_dir():
        return "Directory not found"
    return "\n".join(p.name for p in target.iterdir())


@tool
def safe_read_file(path: str) -> str:
    """Read file contents within workspace."""
    root = _workspace_root()
    if not root.exists():
        return f"Workspace not found: {root}"
    target = _safe_path(path)
    if not target.exists() or not target.is_file():
        return "File not found"
    return target.read_text(encoding="utf-8")


@tool
def safe_write_file(path: str, content: str) -> str:
    """Write file contents within workspace."""
    root = _workspace_root()
    if not root.exists():
        return f"Workspace not found: {root}"
    target = _safe_path(path)
    if not target.parent.exists():
        return "Parent directory not found"
    target.write_text(content, encoding="utf-8")
    return f"Written: {target}"


@tool
def safe_move_file(src: str, dst: str) -> str:
    """Move/rename file within workspace."""
    root = _workspace_root()
    if not root.exists():
        return f"Workspace not found: {root}"
    src_path = _safe_path(src)
    dst_path = _safe_path(dst)
    if not src_path.exists():
        return "Source not found"
    if not dst_path.parent.exists():
        return "Destination directory not found"
    src_path.replace(dst_path)
    return f"Moved: {dst_path}"
