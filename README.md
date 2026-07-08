# DeepResearch Multi-Agent Assistant

DeepResearch 是一个面向企业级研究问答和复杂问题调研的多智能体助手。项目使用 LangGraph 编排多个智能体节点，结合 Web 搜索、本地 RAG、会话记忆、证据审查、评测系统和可观测能力，为用户生成带来源依据的结构化研究报告。

项目同时提供 CLI、FastAPI 后端、Vue 3 前端和 Docker Compose 部署方式，既可用于课程/演示，也可继续扩展为企业知识研究工作台。

## 核心能力

- 智能意图分流：简单问答走快速回答链路，复杂调研走多智能体研究链路。
- 多智能体协作：规划、网页检索、本地知识库检索、证据判断、分析反思和报告写作分工执行。
- 双源检索增强：支持联网搜索与本地知识库 RAG 并行召回。
- 证据驱动输出：保留检索来源、证据池、引用编号和参考资料列表。
- 记忆增强：支持 Redis 短期记忆、Postgres 长期记忆、Milvus 向量记忆和跨轮次上下文注入。
- 流式交互：后端通过 Server-Sent Events 输出执行阶段，前端实时显示任务进度。
- 企业级搜索：支持 Bocha、Tavily、Serper，多后端 fallback、TTL/LRU 缓存、并发抓取和重排序。
- 安全与治理：API key 鉴权、独立 admin key、滑动窗口限流、请求 ID、生产环境配置校验。
- 可观测性：可选 LangSmith trace，覆盖 HTTP、workflow、RAG 和 evaluation 链路。
- 评测系统：内置 smoke 和 intent 数据集，可通过管理接口运行和查看结果。
- 生产部署：提供后端 Dockerfile、前端 Nginx 镜像、`docker-compose.prod.yml`、健康检查和持久化卷。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 后端 API | FastAPI, Uvicorn, SSE |
| 多智能体编排 | LangGraph, LangChain |
| 大模型 | DashScope / Qwen / ChatTongyi |
| 搜索 | Bocha, Serper, Tavily |
| 本地 RAG | Milvus, BM25, DashScope Embedding / Rerank |
| 记忆系统 | Redis, PostgreSQL, SQLite fallback, Milvus optional |
| 前端 | Vue 3, TypeScript, Vite |
| 可观测与评测 | LangSmith, custom evaluation runner |
| 部署 | Docker, Docker Compose, Nginx |

## 项目结构

```text
deep_research/
  app/
    app_main.py                     # FastAPI 应用入口
    backend/
      config/                       # API 服务配置
      dependencies/                 # 管理接口依赖
      middleware/                   # 鉴权、限流、请求 ID、trace
      router/                       # 健康检查、研究、RAG、评测、观测接口
      schemas/                      # 请求/响应模型
      service/                      # Workflow 和 RAG 服务层
    evaluation/                     # 评测数据集加载、评分、结果存储
    mult_agents/
      graph.py                      # LangGraph 工作流编排
      main.py                       # CLI 入口与 agent 构建
      config.py                     # config.json 与环境变量加载
      state.py                      # 研究状态定义
      nodes/                        # 各智能体节点实现
      search.py                     # 多搜索后端、缓存、重排
      rag/                          # 本地知识库检索与入库
      memory/                       # 短期/长期记忆系统
    observability/                  # LangSmith trace 封装
  front/agent_front/                # Vue 3 前端
  eval_datasets/                    # smoke / intent 示例评测集
  scripts/                          # 本地验证和 LangSmith 配置脚本
  docker-compose.yml                # 本地依赖栈
  docker-compose.prod.yml           # 生产部署栈
  DEPLOYMENT.md                     # 生产部署说明
```

## 本地快速开始

### 1. 安装依赖

建议使用 Python 3.10+ 和 Node.js 20.19+ 或 22.12+。

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

常用配置项：

- `MODEL`：复杂研究节点使用的大模型。
- `SMALL_MODEL`：意图识别、快速回答等轻量节点使用的小模型。
- `MAX_ITERATIONS`：反思补搜的最大轮数。
- `ENABLE_MEMORY`：是否启用记忆系统。
- `REDIS_URL`、`POSTGRES_DSN`、`MILVUS_HOST`：记忆与 RAG 基础设施配置。
- `SEARCH_BACKENDS`：默认搜索后端。
- `SEARCH_FALLBACK_BACKENDS`：默认搜索失败时的候选后端。

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

也可以在仓库根目录启动：

```powershell
python -m uvicorn app_main:app --app-dir app --host 127.0.0.1 --port 8001
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

## CLI 使用

单次提问：

```powershell
python main.py --once "请调研多智能体 DeepResearch 系统的关键技术路线"
```

交互式对话：

```powershell
python main.py
```

CLI 内置指令：

- `quit`、`exit`、`退出`：结束会话。
- `/memory` 或 `memory-status`：查看记忆统计。
- `/memory-vacuum`：清理低保留价值记忆。
- `/memory-trace`：查看最近一次记忆轨迹。

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

同步响应示例：

```json
{
  "query": "...",
  "user_id": "default_user",
  "thread_id": "default",
  "tenant_id": "default_tenant",
  "final": "Markdown 格式研究报告"
}
```

流式接口以 SSE 返回阶段性事件：

- `status`：任务状态。
- `phase`：当前执行节点。
- `route`：意图分流结果。
- `final`：最终报告。
- `error`：错误信息。

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

## 工作流说明

复杂问题的执行链路如下：

```text
intent
  ├─ direct_answer -> END
  └─ plan
       ├─ web_search
       ├─ local_rag
       └─ deep_dive -> analyze
              ├─ reflect -> web_search/local_rag
              └─ prune_context -> write -> memory_reflect -> END
```

核心流程：

1. `intent` 判断问题是简单问答还是复杂调研。
2. `plan` 拆解目标、子问题、检索计划和报告大纲。
3. `web_search` 调用搜索后端并进行去重、过滤、全文抓取和重排。
4. `local_rag` 从本地知识库召回相关材料。
5. `deep_dive` 对证据进行可靠性判断和冲突识别。
6. `analyze` 生成核心发现、信息缺口和是否需要补搜的判断。
7. `reflect` 在信息不足时生成补充检索计划。
8. `write` 基于合法来源 ID 输出 Markdown 报告。
9. `memory_reflect` 抽取本轮可复用记忆，用于后续对话。

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

也可以使用离线入库脚本：

```powershell
python app\mult_agents\rag\ingest.py
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

## 注意事项与安全提醒

- 不要提交真实 `.env`、`.env.production`、API Key、数据库密码、本地数据库、上传文件或评测结果。
- 不要把模型供应商 key 当作应用 API key 使用。
- 任何出现在终端、截图、聊天记录或 Git 历史里的密钥都应轮换。
- 搜索与模型服务依赖网络和外部服务额度，演示前建议提前验证 key 是否可用。
- 如果未启用 Redis/PostgreSQL/Milvus，系统会尽量降级到内存或 SQLite fallback，但 RAG 与长期记忆能力会受限。
- 生产环境不要公开 Redis、Postgres、Milvus、etcd、MinIO 或 FastAPI 容器端口。
