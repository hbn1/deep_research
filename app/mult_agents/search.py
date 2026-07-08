"""

企业级搜索引擎模块 — 多后端、缓存、并行、重排序、全量抓取、时间范围、Query 改写

"""

from __future__ import annotations



import hashlib

import json

import logging

import os

import re

import threading

import time

import urllib.error

import urllib.request

from collections import OrderedDict

from concurrent.futures import ThreadPoolExecutor, as_completed

from dataclasses import dataclass, field

from datetime import datetime, timedelta

from typing import Any, Callable, Optional



logger = logging.getLogger("mult_agents.search")



# ── 可选依赖 ──────────────────────────────────────────────

try:

    import httpx

    _HAS_HTTPX = True

except ImportError:

    _HAS_HTTPX = False

    httpx = None  # type: ignore

# Global connection pool for httpx (reuse across requests)
_httpx_client: "Optional[Any]" = None
_httpx_client_lock = threading.RLock()


def _get_httpx_client() -> "Any":
    global _httpx_client
    if not _HAS_HTTPX:
        raise RuntimeError("httpx is not installed")
    with _httpx_client_lock:
        if _httpx_client is None or _httpx_client.is_closed:
            _httpx_client = httpx.Client(
                timeout=30.0,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
                headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearch/1.0; +https://deepresearch.ai)"}
            )
        return _httpx_client


def close_search_resources() -> None:
    """Close shared search resources during application shutdown."""
    global _httpx_client
    cache = globals().get("_search_cache")
    if hasattr(cache, "close"):
        try:
            cache.close()
        except Exception:
            logger.warning("[search] failed to close cache backend", exc_info=True)
    with _httpx_client_lock:
        client = _httpx_client
        _httpx_client = None
    if client is not None and not client.is_closed:
        try:
            client.close()
        except Exception:
            logger.warning("[search] failed to close httpx client", exc_info=True)



# ── 数据结构 ──────────────────────────────────────────────



@dataclass

class SearchResult:

    source_id: str

    title: str

    url: str

    snippet: str

    domain: str

    source_type: str = "web"

    published_at: str = ""

    full_text: str = ""

    authority_score: float = 0.5

    freshness_score: float = 0.5

    relevance_score: float = 0.5

    final_score: float = 0.5



    def to_dict(self) -> dict:

        return {

            "source_id": self.source_id,

            "title": self.title,

            "url": self.url,

            "snippet": self.snippet,

            "domain": self.domain,

            "source_type": self.source_type,

            "published_at": self.published_at,

            "full_text": self.full_text,

            "authority_score": self.authority_score,

            "freshness_score": self.freshness_score,

            "relevance_score": self.relevance_score,

            "final_score": self.final_score,

        }





@dataclass

class SearchConfig:

    """搜索配置"""

    serper_api_key: str = ""

    tavily_api_key: str = ""

    bocha_api_key: str = ""



    enabled_backends: list[str] = field(default_factory=lambda: ["tavily"])

    fallback_backends: list[str] = field(default_factory=lambda: ["tavily"])



    default_count: int = 3

    request_timeout: float = 8.0

    fetch_timeout: float = 4.0

    max_workers: int = 4

    max_fulltext_length: int = 3000

    max_fetch_pages: int = 3

    min_snippet_length: int = 30



    cache_enabled: bool = True

    cache_ttl_seconds: int = 3600

    cache_max_entries: int = 1024

    fetch_enabled: bool = True



    rewrite_enabled: bool = True

    rewrite_model: str = "qwen-turbo"



    blocked_domains: list[str] = field(

        default_factory=lambda: ["datasheet.com", "bdtic.com", "doc88.com", "elecfans.com"]

    )

    high_authority_suffixes: list[str] = field(

        default_factory=lambda: [".gov.cn", ".gov", ".edu.cn", ".edu"]

    )

    news_domains: list[str] = field(

        default_factory=lambda: ["news", "finance", "reuters", "bloomberg", "people", "xinhuanet"]

    )





# ── 工具函数 ──────────────────────────────────────────────



def _extract_domain(url: str) -> str:

    if "://" in url:

        return url.split("://", 1)[1].split("/", 1)[0]

    return ""





def _is_blocked_domain(domain: str, blocked: list[str]) -> bool:

    domain_lower = domain.lower()

    return any(b in domain_lower for b in blocked)





def _is_high_authority(domain: str, suffixes: list[str]) -> bool:

    domain_lower = domain.lower()

    return any(domain_lower.endswith(s) for s in suffixes)





def _is_news_domain(domain: str, news_keywords: list[str]) -> bool:

    domain_lower = domain.lower()

    return any(kw in domain_lower for kw in news_keywords)





def _compute_domain_authority(domain: str, config: SearchConfig) -> float:

    if not domain:

        return 0.3

    if _is_high_authority(domain, config.high_authority_suffixes):

        return 0.90

    if _is_news_domain(domain, config.news_domains):

        return 0.70

    if _is_blocked_domain(domain, config.blocked_domains):

        return 0.10

    return 0.50





def _compute_freshness(published_at: str) -> float:

    """根据发布时间计算新鲜度 0-1"""

    if not published_at:

        return 0.50

    try:

        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]:

            try:

                pub_date = datetime.strptime(published_at[:19] if len(published_at) >= 19 else published_at, fmt)

                break

            except ValueError:

                continue

        else:

            return 0.50

        days_ago = (datetime.now() - pub_date).days

        if days_ago <= 7:

            return 1.0

        if days_ago <= 30:

            return 0.90

        if days_ago <= 90:

            return 0.75

        if days_ago <= 365:

            return 0.50

        if days_ago <= 730:

            return 0.30

        return 0.15

    except Exception:

        return 0.50





def _compute_term_relevance(query: str, text: str) -> float:
    lowered = query.lower()
    meaningful_terms: list[str] = []
    if "agent" in lowered:
        meaningful_terms.append("agent")
    if "智能体" in query:
        meaningful_terms.append("智能体")
    if "人工智能" in query:
        meaningful_terms.append("人工智能")
    stopwords_extra = {
        "什么", "如何", "以及", "一个", "关于", "这个", "那个", "进行", "基于",
        "附带", "来源", "清单", "当前", "最新", "现在", "当下", "近期", "最近",
        "今年", "市面上", "分析", "报告", "写一份", "生成", "调研", "研究",
        "latest", "current", "recent", "today", "overview", "comparison",
        "performance", "best", "practices",
    }
    generic_fragments = [
        "写一份", "当前", "市面上", "最新", "现在", "当下", "近期", "最近",
        "今年", "分析报告", "调研报告", "研究报告", "分析", "报告", "的",
    ]
    for part in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_-]{3,}", lowered):
        if re.fullmatch(r"[A-Za-z0-9_-]{3,}", part):
            if part not in stopwords_extra and part not in meaningful_terms:
                meaningful_terms.append(part)
            continue
        cleaned = part
        for fragment in generic_fragments:
            cleaned = cleaned.replace(fragment, " ")
        sub_terms = re.findall(r"[\u4e00-\u9fff]{2,}", cleaned)
        if not sub_terms and cleaned.strip():
            sub_terms = [part]
        for sub_term in sub_terms:
            if sub_term not in stopwords_extra and sub_term not in meaningful_terms:
                meaningful_terms.append(sub_term)
    if meaningful_terms:
        haystack = text.lower()
        hits = sum(1 for t in meaningful_terms if t in haystack)
        return hits / max(len(meaningful_terms), 1)

    """基于术语命中率的相关性"""

    terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_-]{3,}", query.lower())

    stopwords = {"什么", "如何", "以及", "一个", "关于", "这个", "那个", "进行", "基于", "附带", "来源", "清单"}

    terms = [t for t in terms if t not in stopwords]

    if not terms:

        return 0.50

    haystack = text.lower()

    hits = sum(1 for t in terms if t in haystack)

    return hits / max(len(terms), 1)





# ── 缓存 ──────────────────────────────────────────────────



class RedisSearchCache:
    """Search cache backed by Redis. Same interface as SearchCache."""

    def __init__(self, redis_url, ttl_seconds=3600):
        self._ttl = ttl_seconds
        self._redis = None
        self._prefix = "search:"
        try:
            import redis as _redis
            self._redis = _redis.from_url(redis_url, socket_connect_timeout=5)
            self._redis.ping()
        except Exception:
            self._redis = None

    def get(self, query):
        if self._redis is None:
            return None
        try:
            import hashlib, json
            key = self._prefix + hashlib.md5(query.strip().lower().encode()).hexdigest()
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, query, results):
        if self._redis is None:
            return
        try:
            import hashlib, json
            key = self._prefix + hashlib.md5(query.strip().lower().encode()).hexdigest()
            self._redis.setex(key, self._ttl, json.dumps(results, ensure_ascii=False))
        except Exception:
            pass

    def clear(self):
        if self._redis is not None:
            try:
                for k in self._redis.scan_iter(self._prefix + "*"):
                    self._redis.delete(k)
            except Exception:
                pass

    def close(self):
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
            finally:
                self._redis = None


class SearchCache:

    """内存 TTL 缓存"""

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 1024):
        self._store: "OrderedDict[str, tuple[float, list[dict]]]" = OrderedDict()
        self.ttl = max(1, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._last_prune = 0.0

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def _prune_expired_locked(self, now: float) -> None:
        if now - self._last_prune < min(self.ttl, 60):
            return
        self._last_prune = now
        expired = [key for key, (timestamp, _) in self._store.items() if now - timestamp >= self.ttl]
        for key in expired:
            self._store.pop(key, None)

    def get(self, query: str) -> list[dict] | None:
        key = self._key(query)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if now - entry[0] >= self.ttl:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            logger.debug("[cache] hit | query=%s", query[:50])
            return entry[1]

    def set(self, query: str, results: list[dict]):
        key = self._key(query)
        now = time.time()
        with self._lock:
            self._prune_expired_locked(now)
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (now, results)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
        logger.debug("[cache] set | query=%s | count=%s", query[:50], len(results))

    def clear(self):
        with self._lock:
            self._store.clear()





# ── 网页内容抓取 ──────────────────────────────────────────



def _extract_text_from_html(html: str) -> str:

    """简单 HTML 正文提取，移除脚本、样式、标签"""

    # 移除 script / style

    html = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)

    html = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)

    html = re.sub(r"<head[^>]*>[\s\S]*?</head>", " ", html, flags=re.IGNORECASE)

    # 移除所有标签

    text = re.sub(r"<[^>]+>", " ", html)

    # 解码常见 HTML 实体

    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

    text = text.replace("&quot;", "\"").replace("&#39;", "'").replace("&apos;", "'")

    # 合并空白

    text = re.sub(r"\s+", " ", text).strip()

    return text





def fetch_page_content(url: str, timeout: float = 8.0, max_length: int = 3000) -> str:

    """抓取网页全文"""

    if not url or not url.startswith("http"):

        return ""

    lower_url = url.lower()
    if any(marker in lower_url for marker in [".pdf", ".doc", ".ppt", ".xls", "youtube.com/watch"]):

        return ""

    try:

        if _HAS_HTTPX:
            client = _get_httpx_client()
            resp = client.get(url, timeout=timeout)
            if resp.status_code != 200:
                return ""
            text = _extract_text_from_html(resp.text)
            return text[:max_length] if text else ""

        else:

            req = urllib.request.Request(url, headers={"User-Agent": "DeepResearch/1.0"})

            with urllib.request.urlopen(req, timeout=timeout) as resp:

                html = resp.read().decode("utf-8", errors="replace")

                text = _extract_text_from_html(html)

                return text[:max_length] if text else ""

    except Exception:

        return ""





def fetch_pages_parallel(urls: list[str], timeout: float = 8.0, max_length: int = 3000, max_workers: int = 4) -> dict[str, str]:

    """并行抓取多个 URL"""

    results: dict[str, str] = {}
    worker_count = max(1, int(max_workers or 1))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:

        futures = {executor.submit(fetch_page_content, url, timeout, max_length): url for url in urls}

        for future in as_completed(futures):

            url = futures[future]

            try:

                results[url] = future.result()

            except Exception:

                results[url] = ""

    return results





# ── 时间范围推断 ──────────────────────────────────────────



def infer_freshness(query: str) -> str:

    """根据 query 自动推断搜索时间范围"""

    current_year = datetime.now().year
    normalized_query = query.lower()
    if re.search(rf"(?<!\d){current_year}(?!\d)", normalized_query):

        return "pastYear"

    current_markers = [
        "当前", "最新", "现在", "当下", "近期", "最近", "今年", "市面上", "现状",
        "latest", "current", "recent", "today", "up-to-date",
    ]
    if any(marker in normalized_query for marker in current_markers):

        return "pastYear"



    # "2025年" / "今年" / "最新" / "近期"

    if re.search(rf"({current_year}年|今年|最新|近期|最近)", query):

        return "pastYear"

    # "本月" / "这个月"

    if re.search(r"本月|这个月", query):

        return "pastMonth"

    # "本周" / "这周"

    if re.search(r"本周|这周", query):

        return "pastWeek"

    # "今天" / "今日"

    if re.search(r"今天|今日", query):

        return "pastDay"

    # 明确年份 "2024年"

    year_match = re.search(r"(20\d{2})年", query)

    if year_match:

        y = int(year_match.group(1))

        return f"{y}-01-01..{y}-12-31"

    plain_year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", query)

    if plain_year_match:

        y = int(plain_year_match.group(1))

        return f"{y}-01-01..{y}-12-31"

    # "趋势" / "发展" / "演变"

    if any(w in query for w in ["趋势", "发展", "演变", "动态"]):

        return "past2Years"

    return "noLimit"





# ── Query 改写 ────────────────────────────────────────────



_rewrite_llm = None





def init_rewrite_llm(api_key: str, model: str = "qwen-turbo"):

    global _rewrite_llm

    if _rewrite_llm is None:

        try:

            from langchain_community.chat_models import ChatTongyi

            _rewrite_llm = ChatTongyi(model=model, temperature=0.3, api_key=api_key)

        except Exception as e:

            logger.warning("Query rewrite LLM init failed: %s", e)





def rewrite_queries(queries: list[str], original_query: str) -> list[str]:

    """对搜索词做改写扩展"""

    global _rewrite_llm

    if _rewrite_llm is None:

        return queries



    rewritten_all = []

    for q in queries[:3]:  # 只改写最多 3 个查询，控制 Token

        try:

            prompt = (

                f"原始问题：{original_query}\n"

                f"当前搜索词：{q}\n"

                "请将搜索词改写为 2 个搜索引擎友好的变体（可以添加英文关键词、调整语序），"

                "直接输出改写后的搜索词，每行一个，不要编号，不要解释。"

            )

            from langchain_core.messages import HumanMessage

            resp = _rewrite_llm.invoke([HumanMessage(content=prompt)])

            variants = [v.strip() for v in str(resp.content).split("\n") if v.strip()]

            rewritten_all.extend(variants[:2] if variants else [q])

        except Exception:

            rewritten_all.append(q)



    # 去重 + 合并未改写的查询

    seen = set()

    result = []

    for q in rewritten_all + queries:

        if q not in seen:

            seen.add(q)

            result.append(q)

    return result[:10]





# ── 重排序 ────────────────────────────────────────────────



def rerank_results(records: list[dict], query: str, config: SearchConfig) -> list[dict]:

    """多因子重排序"""

    for r in records:

        domain = r.get("domain", "")

        r["authority_score"] = _compute_domain_authority(domain, config)

        r["freshness_score"] = _compute_freshness(r.get("published_at", ""))

        r["relevance_score"] = _compute_term_relevance(

            query, f"{r.get('title', '')} {r.get('snippet', '')}"

        )

        r["final_score"] = (

            0.30 * r["relevance_score"] +

            0.25 * r["authority_score"] +

            0.25 * r["freshness_score"] +

            0.20 * min(len(r.get("snippet", "")) / 200, 1.0)

        )



    records.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    return records





# ── 多搜索引擎后端 ────────────────────────────────────────








def _serper_search(query: str, api_key: str, count: int, _freshness: str, timeout: float) -> list[dict]:

    """Serper.dev (Google Search API) 搜索"""

    if not api_key or not _HAS_HTTPX:

        return []

    try:

        client = _get_httpx_client()
        resp = client.post(
            "https://google.serper.dev/search",
            timeout=timeout,
            json={"q": query, "num": count},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        )
        if resp.status_code in {401, 403}:
            _disable_backend_temporarily("serper", f"HTTP {resp.status_code}")
            return []
        if resp.status_code != 200:
            return []
        data = resp.json()

    except Exception as e:

        logger.warning("[serper] search failed: %s", e)

        return []



    records = []

    for idx, item in enumerate(data.get("organic", [])[:count], 1):

        url = str(item.get("link") or "").strip()

        records.append({

            "source_id": f"WEB-{idx}",

            "title": item.get("title", ""),

            "url": url,

            "snippet": item.get("snippet", ""),

            "domain": _extract_domain(url),

            "source_type": "web",

            "published_at": item.get("date", ""),

            "backend": "serper",

        })

    return records





def _tavily_search(query: str, api_key: str, count: int, _freshness: str, timeout: float) -> list[dict]:

    """Tavily Search API"""

    if not api_key or not _HAS_HTTPX:

        return []

    try:

        client = _get_httpx_client()
        resp = client.post(
            "https://api.tavily.com/search",
            timeout=timeout,
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": count,
                "include_answer": False,
            },
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        if resp.status_code in {401, 403}:
            _disable_backend_temporarily("tavily", f"HTTP {resp.status_code}")
            return []
        if resp.status_code != 200:
            logger.warning("[tavily] search failed: HTTP %s", resp.status_code)
            return []
        data = resp.json()

    except Exception as e:

        logger.warning("[tavily] search failed: %s", e)

        return []



    records = []

    for idx, item in enumerate(data.get("results", [])[:count], 1):

        url = str(item.get("url") or "").strip()

        records.append({

            "source_id": f"WEB-{idx}",

            "title": item.get("title", ""),

            "url": url,

            "snippet": item.get("content", ""),

            "domain": _extract_domain(url),

            "source_type": "web",

            "backend": "tavily",

        })

    return records



# ── 统一搜索入口 ──────────────────────────────────────────



# 全局实例

_search_cache: Optional[SearchCache] = None

_search_config: Optional[SearchConfig] = None

_search_tenant_id = "default_tenant"

_backend_disabled_until: dict[str, float] = {}


def get_search_config() -> SearchConfig:
    return _search_config or SearchConfig()


def _disable_backend_temporarily(backend: str, reason: str, seconds: int = 900) -> None:

    _backend_disabled_until[backend] = time.time() + seconds

    logger.warning("[search] backend=%s disabled for %ss: %s", backend, seconds, reason)


def _backend_is_temporarily_disabled(backend: str) -> bool:

    until = _backend_disabled_until.get(backend, 0)

    if until <= time.time():

        _backend_disabled_until.pop(backend, None)

        return False

    return True


def _backend_has_api_key(backend: str, config: SearchConfig) -> bool:

    return {
        "serper": bool(config.serper_api_key),
        "tavily": bool(config.tavily_api_key),
        "bocha": bool(config.bocha_api_key),
    }.get(backend, False)


def _effective_backends(config: SearchConfig) -> list[str]:

    allow_serper_fallback = os.getenv("SEARCH_ALLOW_SERPER_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
    enabled = [b for b in config.enabled_backends if b]
    fallback = [b for b in config.fallback_backends if b and b not in enabled]
    if not allow_serper_fallback:
        fallback = [b for b in fallback if b != "serper"]
    ordered = enabled + fallback
    return [
        b for b in ordered
        if _backend_has_api_key(b, config) and not _backend_is_temporarily_disabled(b)
    ]





def _bocha_search(query: str, api_key: str, count: int, freshness: str, timeout: float) -> list[dict]:

    """Bocha Web Search API."""

    if not api_key or not _HAS_HTTPX:

        return []

    try:

        client = _get_httpx_client()
        resp = client.post(
            "https://api.bocha.cn/v1/web-search",
            timeout=timeout,
            json={
                "query": query,
                "summary": True,
                "freshness": freshness or "noLimit",
                "count": count,
            },
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        if resp.status_code in {401, 403}:
            _disable_backend_temporarily("bocha", f"HTTP {resp.status_code}")
            return []
        if resp.status_code != 200:
            logger.warning("[bocha] search failed: HTTP %s", resp.status_code)
            return []
        data = resp.json()

    except Exception as e:

        logger.warning("[bocha] search failed: %s", e)

        return []

    pages = data.get("data", {}).get("webPages", [])
    if isinstance(pages, dict):
        pages = pages.get("value", []) or []
    if not isinstance(pages, list):
        return []

    records = []

    for idx, item in enumerate(pages[:count], 1):

        if not isinstance(item, dict):

            continue

        url = str(item.get("url") or item.get("displayUrl") or "").strip()

        records.append({

            "source_id": f"WEB-{idx}",

            "title": item.get("name") or item.get("title") or "",

            "url": url,

            "snippet": item.get("snippet") or item.get("summary") or "",

            "domain": _extract_domain(url),

            "source_type": "web",

            "published_at": item.get("dateLastCrawled", ""),

            "backend": "bocha",

        })

    return records



def init_search(config: SearchConfig, redis_url: str = "", tenant_id: str = "default_tenant"):

    global _search_cache, _search_config, _search_tenant_id

    _search_config = config
    _search_tenant_id = tenant_id

    if config.cache_enabled:

        if redis_url:
            _search_cache = RedisSearchCache(redis_url=redis_url, ttl_seconds=config.cache_ttl_seconds)
        else:
            _search_cache = SearchCache(
                ttl_seconds=config.cache_ttl_seconds,
                max_entries=config.cache_max_entries,
            )

    # 初始化 rewrite LLM

    if config.rewrite_enabled:

        init_rewrite_llm(os.getenv("DASHSCOPE_API_KEY", ""), config.rewrite_model)





def _search_single_backend(

    backend: str, query: str, config: SearchConfig, freshness: str

) -> list[dict]:

    backends: dict[str, Callable[..., list[dict]]] = {


        "serper": _serper_search,

        "tavily": _tavily_search,

        "bocha": _bocha_search,

    }

    api_keys = {


        "serper": config.serper_api_key,

        "tavily": config.tavily_api_key,

        "bocha": config.bocha_api_key,

    }

    func = backends.get(backend)

    api_key = api_keys.get(backend, "")

    if _backend_is_temporarily_disabled(backend):

        logger.info("[search] backend=%s skipped (temporarily disabled)", backend)

        return []

    if func is None:

        logger.warning("[search] backend=%s not implemented, skipping", backend)

        return []

    if not api_key:

        logger.info("[search] backend=%s skipped (no API key configured)", backend)

        return []

    logger.info("[search] backend=%s | query=%s | freshness=%s", backend, query[:60], freshness)

    results = func(query, api_key, config.default_count, freshness, config.request_timeout)

    logger.info("[search] backend=%s | returned=%s", backend, len(results))

    return results





def search(

    query: str,

    config: Optional[SearchConfig] = None,

    enable_cache: bool = True,

    enable_rewrite: bool = True,

    enable_fetch: bool = True,

    enable_rerank: bool = True,

    tenant_id: str | None = None,

) -> list[dict]:

    """

    企业级统一搜索入口



    流程：cache → rewrite → multi-backend (parallel) → 

          dedup → fetch fulltext (parallel) → rerank → return

    """

    cfg = config or _search_config or SearchConfig()
    cache_tenant_id = tenant_id or _search_tenant_id
    freshness = infer_freshness(query)
    cache_key = f"{cache_tenant_id}:{freshness}:{query}"



    # ── 1. 缓存检查 ──

    if enable_cache and _search_cache is not None:

        cached = _search_cache.get(cache_key)

        if cached:

            return cached



    # ── 2. 时间范围推断 ──

    freshness = infer_freshness(query)



    # ── 3. Query 改写 ──

    if enable_rewrite and cfg.rewrite_enabled:

        search_queries = rewrite_queries([query], query)

    else:

        search_queries = [query]



    # ── 4. 并行搜索（多 query × 多后端）──

    all_backends = _effective_backends(cfg)
    if not all_backends:

        logger.warning("[search] no configured backend has an API key for query=%s", query[:60])

        return []



    all_records: list[dict] = []

    seen_urls: set[str] = set()



    with ThreadPoolExecutor(max_workers=max(1, int(cfg.max_workers or 1))) as executor:

        futures = {}

        for sq in search_queries:

            for backend in all_backends:

                futures[executor.submit(_search_single_backend, backend, sq, cfg, freshness)] = (sq, backend)



        for future in as_completed(futures):

            sq, backend = futures[future]

            try:

                results = future.result()

                for r in results:

                    url = r.get("url", "")

                    if url and url in seen_urls:

                        continue

                    seen_urls.add(url)

                    all_records.append(r)

            except Exception as e:

                logger.warning("[search] future failed | query=%s backend=%s: %s", sq[:40], backend, e)



    if not all_records:

        logger.warning("[search] all backends returned empty for query=%s", query[:60])

        return []



    # ── 5. 去重 ──

    seen = set()

    deduped = []

    for r in all_records:

        key = (r.get("url", ""), r.get("title", "")[:80])

        if key in seen:

            continue

        seen.add(key)

        deduped.append(r)

    all_records = deduped



    # ── 6. 域名过滤 ──

    all_records = [

        r for r in all_records

        if not _is_blocked_domain(r.get("domain", ""), cfg.blocked_domains)

        and len(r.get("snippet", "")) >= cfg.min_snippet_length

    ]



    # ── 7. 全文抓取（并行）──

    if enable_fetch and cfg.fetch_enabled:

        urls_to_fetch = [r["url"] for r in all_records if r.get("url", "").startswith("http")]

        urls_to_fetch = urls_to_fetch[: max(int(cfg.max_fetch_pages or 0), 0)]

        if urls_to_fetch:

            logger.info("[search] fetching fulltext for %s urls", len(urls_to_fetch))

            fulltexts = fetch_pages_parallel(urls_to_fetch, cfg.fetch_timeout, cfg.max_fulltext_length, cfg.max_workers)

            for r in all_records:

                r["full_text"] = fulltexts.get(r["url"], "") or ""



    # ── 8. 重排序 ──

    if enable_rerank:

        all_records = rerank_results(all_records, query, cfg)



    # ── 9. 写入缓存 ──

    if enable_cache and _search_cache is not None:

        _search_cache.set(cache_key, all_records)



    logger.info("[search] complete | query=%s | total_results=%s", query[:60], len(all_records))

    return all_records

