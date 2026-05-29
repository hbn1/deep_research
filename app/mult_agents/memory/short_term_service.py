"""Short-term memory service. Redis only, 7-day TTL auto-expiry.

Redis Key layout:
    st:{tenant_id}:{user_id}:{thread_id}:messages  → LIST of JSON strings
    st:{tenant_id}:{user_id}:{thread_id}:summary   → STRING
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger("mult_agents.memory")


class ShortTermService:
    """短期记忆服务。Redis 唯一后端。

    所有方法均为同步（Redis I/O 极快，async 收益有限）。
    若后续接入网络 Redis，可替换为 aioredis。
    """

    def __init__(self, redis_client, ttl_seconds: int = 604800):
        """Args:
            redis_client: A redis.Redis or redis.StrictRedis instance.
            ttl_seconds: TTL for all keys in this service (default 7 days).
        """
        self._redis = redis_client
        self._ttl = ttl_seconds

    # ── Key helpers ────────────────────────────────────────

    def _msg_key(self, tenant_id: str, user_id: str, thread_id: str) -> str:
        return f"st:{tenant_id}:{user_id}:{thread_id}:messages"

    def _sum_key(self, tenant_id: str, user_id: str, thread_id: str) -> str:
        return f"st:{tenant_id}:{user_id}:{thread_id}:summary"

    # ── Message management ──────────────────────────────────

    def add_message(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        role: str,
        content: str,
    ) -> None:
        """Append a message to the conversation buffer."""
        key = self._msg_key(tenant_id, user_id, thread_id)
        payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        pipe = self._redis.pipeline()
        pipe.rpush(key, payload)
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get_messages(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        last_n: int = 30,
    ) -> list[dict[str, str]]:
        """Get the last N messages. Returns list of {role, content}."""
        key = self._msg_key(tenant_id, user_id, thread_id)
        raw = self._redis.lrange(key, -last_n, -1)
        return [json.loads(m) for m in raw]

    def message_count(
        self, tenant_id: str, user_id: str, thread_id: str
    ) -> int:
        return self._redis.llen(self._msg_key(tenant_id, user_id, thread_id))

    def trim_messages(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        keep_last: int = 10,
    ) -> int:
        """Keep only the last N messages, delete older ones."""
        key = self._msg_key(tenant_id, user_id, thread_id)
        total = self._redis.llen(key)
        if total <= keep_last:
            return 0
        self._redis.ltrim(key, -keep_last, -1)
        self._redis.expire(key, self._ttl)
        return total - keep_last

    # ── Summary management ──────────────────────────────────

    def set_summary(
        self, tenant_id: str, user_id: str, thread_id: str, summary: str
    ) -> None:
        self._redis.setex(
            self._sum_key(tenant_id, user_id, thread_id),
            self._ttl,
            summary,
        )

    def get_summary(
        self, tenant_id: str, user_id: str, thread_id: str
    ) -> str:
        val = self._redis.get(self._sum_key(tenant_id, user_id, thread_id))
        return val.decode("utf-8") if val else ""

    def has_summary(
        self, tenant_id: str, user_id: str, thread_id: str
    ) -> bool:
        return self._redis.exists(self._sum_key(tenant_id, user_id, thread_id)) > 0

    # ── Lifecycle ───────────────────────────────────────────

    def clear_thread(
        self, tenant_id: str, user_id: str, thread_id: str
    ) -> None:
        self._redis.delete(
            self._msg_key(tenant_id, user_id, thread_id),
            self._sum_key(tenant_id, user_id, thread_id),
        )

    def health_check(self) -> bool:
        try:
            return self._redis.ping()
        except Exception:
            return False
