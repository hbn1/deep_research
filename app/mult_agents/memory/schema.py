"""PostgreSQL DDL for unified memory store.

Run once during deployment or via the migration script.
"""

DDL_UNIFIED_MEMORIES = """
CREATE TABLE IF NOT EXISTS unified_memories (
    id              TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default_tenant',
    user_id         TEXT NOT NULL DEFAULT 'default_user',
    thread_id       TEXT,
    memory_type     TEXT NOT NULL,
    namespace       TEXT,
    content         JSONB NOT NULL,
    importance      REAL NOT NULL DEFAULT 0.5,
    recall_count    INT  NOT NULL DEFAULT 0,
    last_recalled_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_um_tenant_user
    ON unified_memories (tenant_id, user_id);

CREATE INDEX IF NOT EXISTS idx_um_type
    ON unified_memories (tenant_id, memory_type);

CREATE INDEX IF NOT EXISTS idx_um_namespace
    ON unified_memories (tenant_id, namespace);

CREATE INDEX IF NOT EXISTS idx_um_created
    ON unified_memories (tenant_id, created_at DESC);
"""


def apply_schema(dsn: str) -> None:
    """Execute DDL against a PostgreSQL instance."""
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_UNIFIED_MEMORIES)
            conn.commit()
