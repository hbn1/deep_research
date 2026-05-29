"""??????? Web ????? RAG ????????????"""

from datetime import datetime
import ast
import json
import logging
import operator
import os
from pathlib import Path
import urllib.error
import urllib.request

from langchain_core.tools import tool
from typing import Optional
from .rag.core import RAGSystem, RAGConfig, RAGManager
from .search import (
    search as enterprise_search,
    SearchConfig,
    init_search,
    search as enterprise_search,
)

_search_initialized = False


logger = logging.getLogger("mult_agents")


# RAG: use RAGManager (thread-safe, multi-tenant)
_rag_tenant_id: str = "default_tenant"

def init_rag_system(api_key: str, config=None, tenant_id="default_tenant"):
    """Initialize RAG via RAGManager."""
    global _rag_tenant_id
    _rag_tenant_id = tenant_id
    try:
        RAGManager.get(api_key, config, tenant_id=tenant_id)
    except Exception as e:
        print(f"RAG init failed: {e}")

def _get_rag():
    """Get current RAG instance from RAGManager."""
    return RAGManager.get_or_none(tenant_id=_rag_tenant_id)



def init_search_from_config(
    api_key: str = "",
    serper_api_key: str = "",
    tavily_api_key: str = "",
    search_backends: str = "bocha",
    search_fallback_backends: str = "serper,tavily",
    search_count: int = 4,
    search_timeout: float = 15.0,
    search_fetch_timeout: float = 8.0,
    search_max_workers: int = 6,
    search_cache_enabled: bool = True,
    search_cache_ttl_seconds: int = 3600,
    search_rewrite_enabled: bool = True,
    search_fetch_enabled: bool = True,
):
    """Initialize global enterprise search engine from AppConfig."""
    global _search_initialized
    if _search_initialized:
        return
    import os as _os
    config = SearchConfig(
        bocha_api_key=_os.getenv("BOCHA_API_KEY", "").strip(),
        serper_api_key=serper_api_key,
        tavily_api_key=tavily_api_key,
        enabled_backends=[b.strip() for b in search_backends.split(",") if b.strip()],
        fallback_backends=[b.strip() for b in search_fallback_backends.split(",") if b.strip()],
        default_count=search_count,
        request_timeout=search_timeout,
        fetch_timeout=search_fetch_timeout,
        max_workers=search_max_workers,
        cache_enabled=search_cache_enabled,
        cache_ttl_seconds=search_cache_ttl_seconds,
        rewrite_enabled=search_rewrite_enabled,
        rewrite_model="qwen-turbo",
    )
    init_search(config)
    _search_initialized = True


def search_knowledge_base_records(query: str, limit: int = 5) -> list[dict]:
    """Search local knowledge base with full RAG pipeline (hybrid+rerank)."""
    rag = _get_rag()
    if rag is None:
        return []
    try:
        return rag.search_records(query, k=limit)
    except Exception:
        return []


def bocha_web_search_records(query: str, count: int = 8) -> list[dict]:
    api_key = os.getenv("BOCHA_API_KEY", "").strip()
    logger.info("[bocha_web_search] ???? | query=%s | count=%s", query, count)
    logger.info("[bocha_web_search] API Key ?? | ????=%s | Key??=%s", bool(api_key), api_key[:8] + "..." if api_key else "None")
    if not api_key:
        logger.warning("[bocha_web_search] ??? BOCHA_API_KEY?????")
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
        logger.info("[bocha_web_search] ???? | url=%s", request.full_url)
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            logger.info("[bocha_web_search] ???? | status=%s | content_length=%s", response.status, len(raw))
        result = json.loads(raw)
        logger.info("[bocha_web_search] ?????? | data????=%s", "data" in result)
    except urllib.error.HTTPError as e:
        logger.error("[bocha_web_search] HTTP ?? | code=%s | reason=%s", e.code, e.reason)
        return []
    except urllib.error.URLError as e:
        logger.error("[bocha_web_search] URL ?? | reason=%s", e.reason)
        return []
    except json.JSONDecodeError as e:
        logger.error("[bocha_web_search] JSON ???? | error=%s", e)
        return []
    except Exception as e:
        logger.error("[bocha_web_search] ???? | error=%s | type=%s", e, type(e).__name__)
        return []
    data = result.get("data", {})
    pages = data.get("webPages", [])
    logger.info("[bocha_web_search] ???? | webPages??=%s", type(pages).__name__)
    if isinstance(pages, dict):
        if isinstance(pages.get("value"), list):
            pages = pages.get("value", [])
        elif isinstance(pages.get("items"), list):
            pages = pages.get("items", [])
        else:
            pages = []
    if not isinstance(pages, list):
        logger.warning("[bocha_web_search] webPages ???? | type=%s", type(pages).__name__)
        return []
    logger.info("[bocha_web_search] ?????? | total=%s", len(pages))
    records: list[dict] = []
    for idx, page in enumerate(pages[:count], 1):
        if not isinstance(page, dict):
            logger.warning("[bocha_web_search] ? %s ??????? | type=%s", idx, type(page).__name__)
            continue
        url = str(page.get("url") or "").strip()
        domain = ""
        if "://" in url:
            domain = url.split("://", 1)[1].split("/", 1)[0]
        title = page.get("name") or f"web_result_{idx}"
        snippet = page.get("summary") or ""
        logger.info("[bocha_web_search] ???? %s | title=%s | url=%s | snippet??=%s", idx, title[:50], domain, len(snippet))
        records.append(
            {
                "source_id": f"WEB-{idx}",
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": domain,
                "source_type": "web",
                "published_at": page.get("datePublished") or page.get("dateLastCrawled") or "",
            }
        )
    logger.info("[bocha_web_search] ???? | ?????=%s", len(records))
    return records

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search local/enterprise knowledge base.
    Returns formatted retrieval results with source attribution.
    """
    rag = _get_rag()
    if rag is None:
        return "RAG system not available. Please ensure Milvus is running."
    return rag.search(query)


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _eval_node(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("Unsupported expression")


@tool
def get_current_time() -> str:
    """??????? ISO ????"""
    return datetime.now().isoformat()


@tool
def simple_calculator(expression: str) -> str:
    """???????????????"""
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree.body)
    return str(result)


@tool
def extract_requirements(text: str) -> str:
    """?????????????"""
    items = [part.strip() for part in text.replace("\n", " ").split("?") if part.strip()]
    return "\n".join(f"- {item}" for item in items[:8])


@tool
def outline_from_topics(topics: str) -> str:
    """?????????????"""
    raw = topics.replace("\n", ",")
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return "\n".join(f"{idx+1}. {item}" for idx, item in enumerate(items[:10]))


@tool
def merge_notes(note_a: str, note_b: str) -> str:
    """????????????"""
    return f"{note_a}\n{note_b}".strip()


@tool
def summarize_points(text: str) -> str:
    """???????????"""
    sentences = [s.strip() for s in text.replace("\n", " ").split("?") if s.strip()]
    points = sentences[:6]
    return "\n".join(f"- {p}" for p in points)


@tool
def dedupe_lines(text: str) -> str:
    """???????????"""
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
    """???????Bocha Web Search??"""
    records = bocha_web_search_records(query, count=5)
    if not records:
        return "??? BOCHA_API_KEY??????????"
    lines = ["Bocha ?????"]
    for idx, record in enumerate(records, 1):
        lines.append(f"{idx}. {record['title']}")
        url = record.get("url", "")
        if url:
            lines.append(f"   ??: {url}")
        snippet = record.get("snippet", "")
        if snippet:
            lines.append(f"   ??: {snippet[:200]}")
    return "\n".join(lines)


@tool
def local_docs_lookup_stub(query: str) -> str:
    """?????????"""
    return f"??????????????: {query}"


@tool
def local_vector_search_stub(query: str) -> str:
    """????????????"""
    return f"?????????????: {query}"


@tool
def optimize_query(query: str) -> str:
    """?????????????"""
    return f"????????: {query}"


@tool
def explain_term(term: str) -> str:
    """???????"""
    return f"{term} ????????????"


@tool
def python_inter(code: str) -> str:
    """?? Python ?????"""
    return f"???Python?????????: {code}"


@tool
def fig_inter(spec: str) -> str:
    """?????????"""
    return f"??????????????: {spec}"


@tool
def amap_weather(city: str) -> str:
    """?????????"""
    return f"?????API???????: {city}"


@tool
def amap_geocode(address: str) -> str:
    """?????????"""
    return f"?????API?????????: {address}"


@tool
def amap_poi_search(query: str) -> str:
    """???? POI ???"""
    return f"?????API???POI??: {query}"


@tool
def amap_route_plan(origin: str, destination: str) -> str:
    """?????????"""
    return f"?????API???????: {origin} -> {destination}"


def _workspace_root() -> Path:
    base = os.getenv("WORKSPACE_DIR", "/workspace")
    return Path(base).resolve()


def _safe_path(path: str) -> Path:
    root = _workspace_root()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("????????")
    return target


@tool
def safe_list_dir(path: str = ".") -> str:
    """?????????????????"""
    root = _workspace_root()
    if not root.exists():
        return f"???????: {root}"
    target = _safe_path(path)
    if not target.exists() or not target.is_dir():
        return "?????"
    items = [p.name for p in target.iterdir()]
    return "\n".join(items)


@tool
def safe_read_file(path: str) -> str:
    """?????????????"""
    root = _workspace_root()
    if not root.exists():
        return f"???????: {root}"
    target = _safe_path(path)
    if not target.exists() or not target.is_file():
        return "?????"
    return target.read_text(encoding="utf-8")


@tool
def safe_write_file(path: str, content: str) -> str:
    """?????????????"""
    root = _workspace_root()
    if not root.exists():
        return f"???????: {root}"
    target = _safe_path(path)
    if not target.parent.exists():
        return "?????"
    target.write_text(content, encoding="utf-8")
    return f"???: {target}"


@tool
def safe_move_file(src: str, dst: str) -> str:
    """?????????????"""
    root = _workspace_root()
    if not root.exists():
        return f"???????: {root}"
    src_path = _safe_path(src)
    dst_path = _safe_path(dst)
    if not src_path.exists():
        return "??????"
    if not dst_path.parent.exists():
        return "???????"
    src_path.replace(dst_path)
    return f"???: {dst_path}"


@tool
def sql_inter(query: str) -> str:
    """?? SQL ?????"""
    return f"?????????SQL: {query}"


@tool
def extract_data_stub(query: str) -> str:
    """?????????"""
    return f"??????????????: {query}"


@tool
def execute_terminal_command(command: str) -> str:
    """???????????"""
    return f"??????????????: {command}"


@tool
def file_operation_stub(request: str) -> str:
    """?????????"""
    return f"??????????????: {request}"


@tool
def news_search_stub(query: str) -> str:
    """?????????"""
    return f"??????????????: {query}"


@tool
def finance_search_stub(query: str) -> str:
    """?????????"""
    return f"??????????????: {query}"


@tool
def extract_url_content_stub(url: str) -> str:
    """?? URL ???????"""
    return f"???URL???????URL: {url}"

