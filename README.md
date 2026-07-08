# DeepResearch Multi-Agent Assistant

DeepResearch 是一个面向企业级研究问答场景的多智能体助手。项目由 FastAPI 后端、LangGraph 工作流、Vue 3 前端、本地 RAG、记忆系统、评测系统和生产级 Docker 部署配置组成。

它适合用来构建内部知识研究助手、资料调研助手、带引用的深度问答系统，以及需要可观测、可评测、可部署的 agent 应用原型。

## 核心能力

- 多智能体研究流程：意图识别、规划、Web 搜索、本地 RAG、证据判断、分析、反思和写作。
- 直接回答快路径：简单算术和低风险直答问题可跳过完整多智能体流程，降低延迟和成本。
- 本地 RAG：支持上传 PDF 和 DOCX，写入 Milvus 向量库，并按租户检索。
- 记忆系统：支持 Redis 短期记忆、Postgres 长期记忆、Milvus 向量记忆，并可回退到本地模式。
- 企业级搜索：支持 Bocha、Tavily、Serper，多后端 fallback、TTL/LRU 缓存、并发抓取和重排序。
- 安全与治理：API key 鉴权、独立 admin key、滑动窗口限流、请求 ID、生产环境配置校验。
- 可观测性：可选 LangSmith trace，覆盖 HTTP、workflow、RAG 和 evaluation 链路。
- 评测系统：内置 smoke 和 intent 数据集，可通过管理接口运行和查看结果。
- 生产部署：提供后端 Dockerfile、前端 Nginx 镜像、`docker-compose.prod.yml`、健康检查和持久化卷。

## 技术栈

- 后端：FastAPI、LangGraph、LangChain、Pydantic Settings、SSE
- Agent/模型：DashScope / Qwen 兼容配置，支持节点级小模型覆盖
- 数据与检索：Redis、Postgres、Milvus、pgvector、RAG 文档导入
- 前端：Vue 3、Vite、TypeScript
- 可观测与评测：LangSmith、自定义 evaluation runner
- 部署：Docker、Docker Compose、Nginx

## 目录结构

```text
app/
  app_main.py                  # FastAPI 入口
  backend/                     # API 路由、中间件、配置、服务层
  mult_agents/                 # LangGraph 多智能体工作流
  evaluation/                  # 评测数据集加载、评分、结果存储
  observability/               # LangSmith trace 封装
front/agent_front/             # Vue 3 前端
eval_datasets/                 # smoke / intent 示例评测集
scripts/                       # 本地验证和 LangSmith 配置脚本
docker-compose.yml             # 本地依赖栈
docker-compose.prod.yml        # 生产部署栈
DEPLOYMENT.md                  # 生产部署说明
```

## 本地快速开始

### 1. 安装依赖

```powershell
python -m pip install -r requirements.txt

cd front\agent_front
npm install
cd ..\..
```

### 2. 准备环境变量

```powershell
copy .env.example .env
```

至少需要配置模型供应商相关变量：

```env
DASHSCOPE_API_KEY=your-provider-key
DASHSCOPE_HTTP_BASE_URL=https://your-workspace-host/api/v1
DASHSCOPE_WEBSOCKET_BASE_URL=wss://your-workspace-host/api-ws/v1/inference
OPENAI_BASE_URL=https://your-workspace-host/compatible-mode/v1
```

如需 Web 搜索，配置至少一个搜索 provider：

```env
BOCHA_API_KEY=your-bocha-key
TAVILY_API_KEY=your-tavily-key
SERPER_API_KEY=your-serper-key
```

### 3. 启动本地基础设施

需要 Redis、Postgres、Milvus 时：

```powershell
docker compose up -d redis postgres milvus etcd minio
```

只做最小功能验证时，也可以先设置：

```env
ENABLE_MEMORY=false
ENABLE_MILVUS=false
CHECKPOINTER_BACKEND=memory
```

### 4. 启动后端

```powershell
cd app
python -m uvicorn app_main:app --host 127.0.0.1 --port 8001
```

健康检查：

```powershell
Invoke-WebRequest http://127.0.0.1:8001/health
```

### 5. 启动前端

```powershell
cd front\agent_front
$env:VITE_API_TARGET="http://127.0.0.1:8001"
npm run dev -- --host 127.0.0.1 --port 5173
```

访问：

- 前端：`http://localhost:5173`
- 后端健康检查：`http://127.0.0.1:8001/health`
- 开发环境 API 文档：`http://127.0.0.1:8001/docs`

## 常用 API

普通研究问答：

```http
POST /api/v1/research/run
POST /api/v1/research/stream
```

请求示例：

```json
{
  "query": "请调研某个行业的最新趋势，并给出来源",
  "user_id": "default_user",
  "tenant_id": "default_tenant",
  "thread_id": "default",
  "max_iterations": 3,
  "enable_memory": true
}
```

管理接口默认需要 `X-Admin-Key`：

```http
GET  /api/v1/rag/status
GET  /api/v1/rag/documents
POST /api/v1/rag/documents
POST /api/v1/rag/search
GET  /api/v1/evals/datasets
POST /api/v1/evals/run
GET  /api/v1/observability/langsmith/status
```

## 认证与安全

后端普通 API 使用独立的 `API_AUTH_KEY`，不要复用模型供应商 key。

本地开发默认：

```env
APP_ENV=development
API_AUTH_REQUIRED=false
ADMIN_API_REQUIRED=false
```

生产环境建议：

```env
APP_ENV=production
API_AUTH_REQUIRED=true
API_AUTH_KEY=replace-with-a-long-random-secret
ADMIN_API_REQUIRED=true
ADMIN_API_KEY=replace-with-a-different-long-random-admin-secret
API_DOCS_ENABLED=false
```

管理接口使用 `X-Admin-Key`。生产前端通过 Nginx 在服务端注入 `X-API-Key`，密钥不会进入浏览器 JavaScript。

## RAG 文档导入

支持格式：

- `.pdf`
- `.docx`

关键配置：

```env
RAG_UPLOAD_DIR=rag_uploads
RAG_MAX_UPLOAD_MB=25
RAG_MAX_TENANT_STORAGE_MB=512
RAG_ALLOWED_EXTENSIONS=.pdf,.docx
RAG_VALIDATE_FILE_SIGNATURES=true
```

上传示例：

```powershell
curl -X POST "http://127.0.0.1:8001/api/v1/rag/documents" `
  -H "X-Admin-Key: <ADMIN_API_KEY>" `
  -F "tenant_id=default_tenant" `
  -F "user_id=default_user" `
  -F "file=@sample.pdf"
```

## 评测系统

内置示例数据集位于 `eval_datasets/`：

- `smoke.jsonl`
- `intent.jsonl`

查看数据集：

```powershell
curl -H "X-Admin-Key: <ADMIN_API_KEY>" http://127.0.0.1:8001/api/v1/evals/datasets
```

运行评测：

```powershell
curl -X POST "http://127.0.0.1:8001/api/v1/evals/run" `
  -H "Content-Type: application/json" `
  -H "X-Admin-Key: <ADMIN_API_KEY>" `
  -d "{\"dataset\":\"smoke\",\"persist\":false}"
```

## LangSmith 可观测性

LangSmith 默认关闭。需要开启时：

```env
LANGSMITH_ENABLED=true
LANGSMITH_API_KEY=lsv2_your_langsmith_key
LANGSMITH_PROJECT=deepresearch-dev
LANGSMITH_ENVIRONMENT=development
LANGSMITH_TAGS=local,deepresearch
```

也可以使用脚本写入配置：

```powershell
.\scripts\enable_langsmith.ps1 -ApiKey "lsv2_your_langsmith_key" -Project "deepresearch-dev"
```

可观测链路包括：

- `http.request`
- `deepresearch.workflow.run`
- `deepresearch.workflow.stream`
- `deepresearch.workflow.run_state`
- `rag.ingest_document`
- `rag.search`
- `evaluation.run`
- `evaluation.case`

如果不希望问题或答案正文进入 LangSmith：

```env
LANGSMITH_HIDE_INPUTS=true
LANGSMITH_HIDE_OUTPUTS=true
```

## 前端管理工具

RAG 导入和 Agent 评测默认不显示给普通用户。需要本地临时开启管理工具时：

```powershell
$env:VITE_ENABLE_ADMIN_TOOLS="true"
$env:ADMIN_API_KEY="your-admin-secret"
cd front\agent_front
npm run dev -- --host 127.0.0.1 --port 5173
```

生产环境请保持：

```env
VITE_ENABLE_ADMIN_TOOLS=false
```

## 测试与验证

后端编译检查：

```powershell
python -m compileall -q app tests
```

后端单测：

```powershell
python -m unittest discover -s tests -v
```

前端构建：

```powershell
cd front\agent_front
npm run build
```

本地运行时验证：

```powershell
python scripts\verify_runtime.py --backend http://127.0.0.1:8001 --frontend http://127.0.0.1:5173
```

生产 Compose 配置检查：

```powershell
docker compose --env-file .env.production.example -f docker-compose.prod.yml config
```

## 生产部署

复制生产环境模板：

```powershell
copy .env.production.example .env.production
```

启动生产栈：

```powershell
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
```

生产部署细节见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 故障排查

- 后端启动失败：先检查 `APP_ENV`、`API_AUTH_KEY`、`ADMIN_API_KEY`、`RATE_LIMIT_*`、`RAG_*` 是否满足运行校验。
- 搜索无结果：确认至少配置了一个搜索 provider key，并检查 `SEARCH_BACKENDS` 与 `SEARCH_FALLBACK_BACKENDS`。
- RAG 上传失败：确认 Milvus 可用、文件签名与扩展名一致、文件有可提取文本。
- 评测接口 403：确认请求带有 `X-Admin-Key`，并且后端配置了正确的 `ADMIN_API_KEY`。
- 前端无法访问后端：本地检查 `VITE_API_TARGET`，生产检查 Nginx 代理配置和后端健康检查。

## 安全提醒

- 不要提交 `.env`、`.env.production`、本地数据库、上传文件或评测结果。
- 不要把模型供应商 key 当作应用 API key 使用。
- 任何出现在终端、截图、聊天记录或 Git 历史里的密钥都应轮换。
- 生产环境不要公开 Redis、Postgres、Milvus、etcd、MinIO 或 FastAPI 容器端口。
