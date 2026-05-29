"""一次性迁移：旧 SQLite memory.db → 新 PostgreSQL unified_memories 单表。

用法:
    python scripts/migrate_memory.py [--sqlite PATH] [--pg DSN]

默认:
    --sqlite  app/data/memory.db
    --pg      读取 POSTGRES_DSN 环境变量

安全特性:
    - ON CONFLICT (id) DO NOTHING — 幂等，可重复执行
    - 逐表迁移，每表完成即 commit，中断可续
    - 迁移前自动执行 DDL 建表
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone

# 确保项目根在 path 中
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


def parse_args():
    p = argparse.ArgumentParser(description="Migrate SQLite memory → PG unified_memories")
    p.add_argument("--sqlite", default=str(ROOT / "app" / "data" / "memory.db"), help="SQLite DB path")
    p.add_argument("--pg", default=os.getenv("POSTGRES_DSN", ""), help="PostgreSQL DSN")
    return p.parse_args()


def ensure_schema(pg_dsn: str) -> None:
    """Apply unified_memories DDL."""
    from mult_agents.memory.schema import apply_schema
    apply_schema(pg_dsn)
    print("[schema] DDL applied")


def migrate_table(
    sqlite_conn: sqlite3.Connection,
    pg_dsn: str,
    memory_type: str,
    tenant_id: str = "default_tenant",
) -> int:
    """Migrate one memory_type partition from SQLite to PG."""
    import psycopg

    rows = sqlite_conn.execute(
        "SELECT * FROM memories WHERE memory_type = ? ORDER BY created_at",
        (memory_type,),
    ).fetchall()

    if not rows:
        print(f"  [{memory_type}] 0 rows — skipped")
        return 0

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            for row in rows:
                row_dict = dict(row)
                # 解析 content
                content = row_dict.get("content")
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        pass
                content_json = json.dumps(
                    content if isinstance(content, dict) else {"text": str(content)},
                    ensure_ascii=False,
                )

                # 安全取字段（兼容旧表缺列）
                entry_id = row_dict.get("id") or row_dict.get("memory_id", "")
                user_id = row_dict.get("user_id") or "default_user"
                thread_id = row_dict.get("thread_id") or ""
                namespace = row_dict.get("namespace") or ""
                importance = float(row_dict.get("importance", 0.5) or 0.5)
                recall_count = int(row_dict.get("recall_count", 0) or 0)
                last_recalled = row_dict.get("last_recalled_at")
                created_at = row_dict.get("created_at", datetime.now(timezone.utc).isoformat())
                updated_at = row_dict.get("updated_at", created_at)

                cur.execute(
                    """INSERT INTO unified_memories
                       (id, tenant_id, user_id, thread_id, memory_type, namespace,
                        content, importance, recall_count, last_recalled_at,
                        created_at, updated_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s)
                       ON CONFLICT (id) DO NOTHING""",
                    (entry_id, tenant_id, user_id, thread_id, memory_type,
                     namespace, content_json, importance, recall_count,
                     last_recalled, created_at, updated_at),
                )
            conn.commit()

    print(f"  [{memory_type}] migrated {len(rows)} rows")
    return len(rows)


def main():
    args = parse_args()

    if not args.pg:
        print("ERROR: POSTGRES_DSN not set. Use --pg or set POSTGRES_DSN env var.")
        sys.exit(1)

    if not Path(args.sqlite).exists():
        print(f"ERROR: SQLite file not found: {args.sqlite}")
        sys.exit(1)

    print(f"SQLite: {args.sqlite}")
    print(f"PG:     {args.pg[:50]}...")
    print()

    # 1. Ensure schema
    ensure_schema(args.pg)

    # 2. Open SQLite
    sqlite_conn = sqlite3.connect(args.sqlite)
    sqlite_conn.row_factory = sqlite3.Row

    # 3. Migrate by type
    total = 0
    for mt in ["semantic", "episodic", "procedural"]:
        n = migrate_table(sqlite_conn, args.pg, mt)
        total += n

    sqlite_conn.close()
    print(f"\nDone. {total} total entries migrated to unified_memories.")


if __name__ == "__main__":
    main()
