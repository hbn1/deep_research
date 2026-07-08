# DeepResearch Multi-Agent Assistant

DeepResearch 是一个基于 FastAPI、LangGraph、Vue 3 和本地 RAG 的多智能体研究助手。它支持直接问答、深度研究、Redis/Postgres/Milvus 记忆与向量检索。RAG 文档导入和 Agent 评测属于管理能力，默认不展示给普通用户。

## 主要能力

- 多智能体研究流程：意图识别、规划、搜索、本地 RAG、分析、反思和写作。
- 本地 RAG：管理侧支持上传 PDF 和 Word 文档，写入 Milvus 向量库并可检索。
- 评测系统：管理侧可运行 smoke/intent 数据集并查看失败阶段。
- 记忆系统：支持 Redis 短期记忆、Postgres 长期记忆、Milvus 向量记忆。
- LangSmith 可观测性：可选开启请求、workflow、RAG、评测 trace。
- 生产部署：提供后端 Dockerfile、前端 Nginx 镜像、`docker-compose.prod.yml`、健康检查和持久化卷。

## 本地启动

准备环境：

```powershell
python -m pip install -r requirements.txt
cd front\agent_front
npm install
cd ..\..
copy .env.example .env
```

编辑 `.env`，至少填好模型供应商配置：

```env
DASHSCOPE_API_KEY=your-provider-key
DASHSCOPE_HTTP_BASE_URL=https://your-workspace-host/api/v1
DASHSCOPE_WEBSOCKET_BASE_URL=wss://your-workspace-host/api-ws/v1/inference
OPENAI_BASE_URL=https://your-workspace-host/compatible-mode/v1
```

启动后端：

```powershell
cd app
python -m uvicorn app_main:app --host 127.0.0.1 --port 8001
```

启动前端：

```powershell
cd front\agent_front
$env:VITE_API_TARGET="http://127.0.0.1:8001"
npm run dev -- --host 127.0.0.1 --port 5173
```

访问：

- 应用首页：http://localhost:5173
- 后端健康检查：http://127.0.0.1:8001/health

## 认证说明

后端访问认证使用独立的 `API_AUTH_KEY`，不要复用 `DASHSCOPE_API_KEY`。

本地开发默认 `API_AUTH_REQUIRED=false`，可以不带认证头。生产环境默认必须配置：

```env
APP_ENV=production
API_AUTH_REQUIRED=true
API_AUTH_KEY=replace-with-a-long-random-secret
API_DOCS_ENABLED=false
```

生产前端通过 Nginx 在服务端注入 `X-API-Key`，密钥不会进入浏览器 JavaScript。

## LangSmith 可观测性

LangSmith 默认关闭。需要观察 Agent 执行链路时，在 `.env` 中配置：

也可以用脚本自动写入配置：

```powershell
.\scripts\enable_langsmith.ps1 -ApiKey "lsv2_your_langsmith_key" -Project "deepresearch-dev"
```

```env
LANGSMITH_ENABLED=true
LANGSMITH_API_KEY=lsv2_your_langsmith_key
LANGSMITH_PROJECT=deepresearch-dev
LANGSMITH_ENVIRONMENT=development
LANGSMITH_TAGS=local,deepresearch
```

重启后端后，下面这些链路会进入 LangSmith：

- `http.request`：FastAPI 请求入口。
- `deepresearch.workflow.run` / `deepresearch.workflow.stream`：普通问答与流式问答。
- `deepresearch.workflow.run_state`：评测时的 Agent 状态执行。
- `rag.ingest_document` / `rag.search`：RAG 上传和检索。
- `evaluation.run` / `evaluation.case`：评测运行和单条 case。

管理侧状态接口：

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8001/api/v1/observability/langsmith/status" `
  -Headers @{ "X-Admin-Key" = "<ADMIN_API_KEY>" }
```

如果不希望问题和回答正文进入 LangSmith，可设置：

```env
LANGSMITH_HIDE_INPUTS=true
LANGSMITH_HIDE_OUTPUTS=true
```

## 管理工具

RAG 文档导入和 Agent 评测默认不显示在前端。需要你自己管理文档或跑评测时，启动私有管理构建：

```powershell
$env:VITE_ENABLE_ADMIN_TOOLS="true"
$env:ADMIN_API_KEY="your-admin-secret"
cd front\agent_front
npm run dev -- --host 127.0.0.1 --port 5173
```

后端管理接口需要 `X-Admin-Key`：

```powershell
curl -H "X-API-Key: <API_AUTH_KEY>" -H "X-Admin-Key: <ADMIN_API_KEY>" http://127.0.0.1:8001/api/v1/evals/datasets
```

生产环境请保持公开前端 `VITE_ENABLE_ADMIN_TOOLS=false`，只在内网、VPN 或本机临时管理会话中打开。

## RAG 文档上传

支持 `.pdf` 和 `.docx`：

- 单文件大小限制：`RAG_MAX_UPLOAD_MB`
- 租户总存储限制：`RAG_MAX_TENANT_STORAGE_MB`
- 文件头校验：`RAG_VALIDATE_FILE_SIGNATURES=true`
- 默认上传目录：`rag_uploads`，生产环境映射到 Docker volume `rag_uploads`

## 测试与验收

```powershell
python -m compileall -q app tests
python -m unittest discover -s tests -v
cd front\agent_front
npm run build
```

生产 Compose 配置校验：

```powershell
docker compose --env-file .env.production.example -f docker-compose.prod.yml config
```

## 生产部署

详见 [DEPLOYMENT.md](DEPLOYMENT.md)。
