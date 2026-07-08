# DeepResearch Production Deployment

## 1. Prepare Secrets

Copy the production template and fill every placeholder:

```powershell
copy .env.production.example .env.production
```

Required production secrets:

- `API_AUTH_KEY`: internal app access key between Nginx and FastAPI. Do not reuse `DASHSCOPE_API_KEY`.
- `ADMIN_API_KEY`: separate admin key for RAG import and evaluation APIs.
- `LANGSMITH_API_KEY`: optional observability key. Required only when `LANGSMITH_ENABLED=true`.
- `DASHSCOPE_API_KEY` and model endpoint URLs.
- `POSTGRES_PASSWORD`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`.
- Optional search provider keys such as `BOCHA_API_KEY`, `SERPER_API_KEY`, or `TAVILY_API_KEY`.

Rotate any key that has appeared in chat logs, screenshots, terminals, or Git history before going live.

## 2. Start Production Stack

```powershell
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
```

Only the frontend container publishes a host port. Redis, Postgres, Milvus, etcd, MinIO, and FastAPI stay on the internal Docker network.

## 3. Verify

```powershell
docker compose --env-file .env.production -f docker-compose.prod.yml ps
docker compose --env-file .env.production -f docker-compose.prod.yml logs backend --tail 100
```

Health endpoints:

- Frontend: `http://<host>/health`
- Backend through reverse proxy: `http://<host>/api/v1/health`

Admin APIs are intentionally hidden from the public UI and require `X-Admin-Key`:

```powershell
curl -H "X-Admin-Key: <ADMIN_API_KEY>" http://<host>/api/v1/evals/datasets
```

LangSmith status:

```powershell
curl -H "X-Admin-Key: <ADMIN_API_KEY>" http://<host>/api/v1/observability/langsmith/status
```

Direct backend access from the host is intentionally not exposed in production compose.

## 4. Persistent Data

The production compose defines named volumes for:

- `rag_uploads`: uploaded source documents.
- `eval_results`: evaluation run reports and case-level diagnostics.
- `pg_data`: long-term memory and checkpointer data.
- `redis_data`: short-term memory/cache and rate-limit state.
- `milvus_data`, `etcd_data`, `minio_data`: vector database metadata and object data.

Back up these volumes before upgrades. At minimum, back up Postgres and all Milvus-related volumes together so vector metadata and object data stay consistent.

## 5. Security Checklist

- Use HTTPS at the host or cloud load balancer layer.
- Keep `.env.production` out of Git.
- Do not expose Redis, Postgres, Milvus, etcd, or MinIO ports publicly.
- Set `APP_ENV=production`, `API_AUTH_REQUIRED=true`, and `API_DOCS_ENABLED=false`.
- Use long random `API_AUTH_KEY` and `ADMIN_API_KEY`, both separate from model provider keys.
- Keep `VITE_ENABLE_ADMIN_TOOLS=false` for the public frontend build.
- Keep `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true` if production prompts or answers may contain private data.
- Keep `RAG_VALIDATE_FILE_SIGNATURES=true`.
- Put host-level firewall rules around the Docker host.
- Configure external log retention and alerting before public traffic.

## 6. Upgrade Flow

```powershell
docker compose --env-file .env.production -f docker-compose.prod.yml pull
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
docker compose --env-file .env.production -f docker-compose.prod.yml ps
```

If the application behavior changes around memory, RAG, or database schema, take a backup before redeploying.
