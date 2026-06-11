# DeepResearch Multi-Agent Assistant

DeepResearch 是一个面向复杂问题调研的 AI 多智能体研究助手。项目以 LangGraph 编排多个智能体节点，结合 Web 搜索、本地 RAG、会话记忆和证据审查流程，为用户生成带来源依据的结构化研究报告。同时项目提供 FastAPI 后端、Vue 前端和命令行入口，既可以作为课程演示系统，也可以继续扩展为企业知识研究工作台。

## 功能特性

- 智能意图分流：简单问答走快速回答链路，复杂调研走多智能体研究链路。
- 多智能体协作：规划、网页检索、本地知识库检索、证据判断、分析反思和报告写作分工执行。
- 双源检索增强：支持联网搜索与本地知识库 RAG 并行召回。
- 证据驱动输出：保留检索来源、证据池、引用编号和参考资料列表。
- 记忆增强：支持短期会话记忆、长期结构化记忆和跨轮次上下文注入。
- 流式交互：后端通过 Server-Sent Events 输出执行阶段，前端实时显示任务进度。
- 多种运行方式：支持 CLI、FastAPI 服务、Vue 前端和 Docker Compose。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 后端 API | FastAPI, Uvicorn, SSE |
| 多智能体编排 | LangGraph, LangChain |
| 大模型 | DashScope / 通义千问 ChatTongyi |
| 搜索 | Bocha, Serper, Tavily |
| 本地 RAG | Milvus, BM25, DashScope Embedding / Rerank |
| 记忆系统 | Redis, PostgreSQL, SQLite fallback, Milvus optional |
| 前端 | Vue 3, TypeScript, Vite |
| 测试 | Python 脚本化测试 |
| 部署 | Docker, Docker Compose |

## 项目结构

```text
deep_research/
  app/
    app_main.py                     # FastAPI 应用入口
    backend/
      config/                       # API 服务配置
      middleware/                   # API Key 鉴权、限流
      router/                       # 健康检查与调研接口
      schemas/                      # 请求/响应模型
      service/                      # WorkflowService 生命周期管理
    mult_agents/
      graph.py                      # LangGraph 工作流编排
      main.py                       # CLI 入口与 agent 构建
      config.py                     # config.json 与环境变量加载
      state.py                      # 研究状态定义
      nodes/                        # 各智能体节点实现
      search.py                     # 多搜索后端、缓存、重排
      rag/                          # 本地知识库检索与入库
      memory/                       # 短期/长期记忆系统
  front/agent_front/
    src/App.vue                     # 前端主界面
    vite.config.ts                  # Vite 代理配置
  tests/
    test_memory_system.py           # 记忆系统单元/集成测试
    test_multiturn.py               # 多轮记忆能力测试
  config.json                       # 默认运行配置
  .env.example                      # 环境变量模板
  docker-compose.yml                # 后端与基础设施编排
  Dockerfile                        # 后端镜像构建
  main.py                           # CLI 启动封装
```

## 环境准备

建议使用 Python 3.10+ 和 Node.js 20.19+ 或 22.12+。项目依赖外部大模型与搜索服务，至少需要配置 DashScope API Key；如果使用联网搜索，还需要配置 Bocha 或其他搜索后端的 Key。

```powershell
copy .env.example .env
```

然后在 `.env` 中填写：

```env
DASHSCOPE_API_KEY=sk-your-dashscope-key
BOCHA_API_KEY=sk-your-bocha-key
SERPER_API_KEY=your-serper-key
TAVILY_API_KEY=tvly-your-tavily-key
```

常用配置项：

- `MODEL`：复杂研究节点使用的大模型。
- `SMALL_MODEL`：意图识别、快速回答等轻量节点使用的小模型。
- `MAX_ITERATIONS`：反思补搜的最大轮数。
- `ENABLE_MEMORY`：是否启用记忆系统。
- `REDIS_URL`、`POSTGRES_DSN`、`MILVUS_HOST`：记忆与 RAG 基础设施配置。
- `SEARCH_BACKENDS`：默认搜索后端。
- `SEARCH_FALLBACK_BACKENDS`：默认搜索失败时的候选后端。

## 安装依赖

后端：

```powershell
python -m pip install -r requirements.txt
```

前端：

```powershell
cd front\agent_front
npm install
```

## 启动方式

### 1. 命令行模式

单次提问：

```powershell
python main.py --once "请调研多智能体 DeepResearch 系统的关键技术路线"
```

交互式对话：

```powershell
python main.py
```

命令行内置指令：

- `quit`、`exit`、`退出`：结束会话。
- `/memory` 或 `memory-status`：查看记忆统计。
- `/memory-vacuum`：清理低保留价值记忆。
- `/memory-trace`：查看最近一次记忆轨迹。

### 2. 后端 API

```powershell
uvicorn app_main:app --app-dir app --host 0.0.0.0 --port 8001
```

健康检查：

```powershell
curl http://127.0.0.1:8001/health
```

普通调用：

```powershell
curl -X POST http://127.0.0.1:8001/api/v1/research/run ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"请分析 RAG 与多智能体系统结合的优势\",\"user_id\":\"user01\",\"thread_id\":\"thread01\",\"tenant_id\":\"default_tenant\"}"
```

流式调用：

```powershell
curl -N -X POST http://127.0.0.1:8001/api/v1/research/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"请调研企业级 AI Agent 平台的发展趋势\",\"user_id\":\"user01\",\"thread_id\":\"thread01\",\"tenant_id\":\"default_tenant\"}"
```

### 3. 前端页面

后端默认按上面的方式运行在 `8001` 端口；前端 Vite 代理也指向该端口。

```powershell
cd front\agent_front
npm run dev
```

打开 Vite 输出的本地地址，一般为：

```text
http://127.0.0.1:5173
```

### 4. Docker Compose

```powershell
docker compose up --build
```

该方式会启动后端、Redis、PostgreSQL、Milvus、etcd 和 MinIO。首次运行前请确认 `.env` 中的数据库连接、模型 Key 与搜索 Key 已配置正确。

## API 说明

### `GET /health`

返回服务状态。

响应示例：

```json
{
  "status": "ok",
  "service": "deepresearch-backend"
}
```

### `POST /api/v1/research/run`

同步执行一次研究任务。

请求体：

```json
{
  "query": "请分析多智能体系统在企业知识管理中的应用",
  "user_id": "user01",
  "thread_id": "thread01",
  "tenant_id": "default_tenant",
  "max_iterations": 3,
  "enable_memory": true
}
```

响应体：

```json
{
  "query": "...",
  "user_id": "user01",
  "thread_id": "thread01",
  "tenant_id": "default_tenant",
  "final": "Markdown 格式研究报告"
}
```

### `POST /api/v1/research/stream`

以 SSE 方式返回阶段性事件。事件类型包括：

- `status`：任务状态。
- `phase`：当前执行节点。
- `route`：意图分流结果。
- `final`：最终报告。
- `error`：错误信息。

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

## 本地知识库入库

RAG 入库入口位于：

```text
app/mult_agents/rag/ingest.py
```

默认支持 TXT、Markdown、PDF、DOCX 和常见代码文件。运行前请确认 Milvus 可访问，并在 `.env` 中配置 `DASHSCOPE_API_KEY`、`MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_COLLECTION`。

```powershell
python app\mult_agents\rag\ingest.py
```

## 测试

记忆系统测试：

```powershell
python tests\test_memory_system.py
```

多轮对话记忆测试：

```powershell
python tests\test_multiturn.py
```

前端构建检查：

```powershell
cd front\agent_front
npm run build
```

## 注意事项

- 不要提交真实 `.env`、API Key、数据库密码和本地日志。
- 搜索与模型服务依赖网络和外部服务额度，演示前建议提前验证 Key 是否可用。
- 如果未启用 Redis/PostgreSQL/Milvus，系统会尽量降级到内存或 SQLite fallback，但 RAG 与长期记忆能力会受限。
- 前端开发代理当前指向 `http://127.0.0.1:8001`，如果后端端口不同，需要同步修改 `front/agent_front/vite.config.ts`。
- 当前项目中部分旧注释或前端文案可能出现编码显示异常，建议统一使用 UTF-8 保存源码和终端环境。


