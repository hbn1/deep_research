<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'

type WorkspaceMode = 'research' | 'evals'

type StreamEvent = {
  type: 'status' | 'phase' | 'route' | 'final' | 'error'
  message?: string
  final?: string
  node?: string
  route?: string
}

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
}

type Conversation = {
  id: string
  title: string
  messages: ChatMessage[]
  threadId: string
  loading: boolean
}

type EvalDatasetInfo = {
  id: string
  path: string
  case_count: number
}

type EvalRunCaseSummary = {
  case_id: string
  query: string
  status: string
  score: number
  latency_ms: number
  failed_evaluators: string[]
  suspected_stages: string[]
}

type EvalRunSummary = {
  run_id: string
  dataset_id: string
  status: string
  started_at: string
  completed_at?: string | null
  total_cases: number
  passed_cases: number
  failed_cases: number
  pass_rate: number
  average_score: number
  average_latency_ms: number
  result_dir: string
  cases: EvalRunCaseSummary[]
}

type EvalIssue = {
  stage: string
  evaluator: string
  severity: string
  message: string
  evidence?: string | null
}

type EvalMetric = {
  name: string
  stage: string
  score: number
  passed: boolean
  weight: number
  reason: string
}

type EvalTrace = {
  route: string
  intent: string
  plan: string
  analysis: string
  final: string
  counts: Record<string, number>
  citations: Record<string, string[]>
  search_plan: Record<string, unknown>[]
  source_index: Record<string, unknown>[]
  evidence_pool: Record<string, unknown>[]
  web_evidence: Record<string, unknown>[]
  local_evidence: Record<string, unknown>[]
  findings: Record<string, unknown>[]
  needs_more_research: boolean
  missing_gaps: string[]
  iteration: number
}

type EvalCaseResult = {
  case: {
    id: string
    query: string
    category: string
    expected_route?: string | null
  }
  status: string
  passed: boolean
  score: number
  threshold: number
  latency_ms: number
  suspected_stages: string[]
  failed_evaluators: string[]
  metrics: EvalMetric[]
  issues: EvalIssue[]
  trace: EvalTrace
  error?: string | null
}

type EvalRunDetail = {
  summary: EvalRunSummary
  cases: EvalCaseResult[]
}

type RagDocumentRecord = {
  doc_id: string
  filename: string
  source: string
  tenant_id: string
  user_id: string
  content_type: string
  size_bytes: number
  chunks: number
  collection: string
  stored_path: string
  uploaded_at: string
}

type RagStatus = {
  tenant_id: string
  collection: string
  milvus_host: string
  milvus_port: number
  configured_enabled: boolean
  runtime_initialized: boolean
  stats: Record<string, unknown>
  documents: RagDocumentRecord[]
}

type RagSearchResult = {
  title?: string
  doc_id?: string
  snippet?: string
  score?: number
  metadata?: Record<string, unknown>
}

type EvalStreamEvent =
  | { type: 'run_start'; run_id: string; dataset_id: string; total_cases: number }
  | { type: 'case_start'; run_id: string; case_id: string; index: number; total: number; query: string }
  | { type: 'case_result'; run_id: string; case_id: string; result: EvalCaseResult }
  | { type: 'summary'; run_id: string; summary: EvalRunSummary }
  | { type: 'error'; message: string }

const STORAGE_KEY = 'deepresearch_workspace'
const ADMIN_SESSION_KEY = 'deepresearch_admin_key'
const adminToolsEnabled = import.meta.env.VITE_ENABLE_ADMIN_TOOLS === 'true'

const makeConversation = (): Conversation => ({
  id: `conv-${Date.now()}`,
  title: '新会话',
  messages: [
    {
      id: `m-${Date.now()}`,
      role: 'assistant',
      content: '你好，我是 DeepResearch。你可以直接提问，我会根据问题选择快速回答或深度研究链路。',
    },
  ],
  threadId: `thread_${Date.now()}`,
  loading: false,
})

const loadFromStorage = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const data = JSON.parse(raw)
    if (!data.conversations?.length) return null
    for (const conv of data.conversations as Conversation[]) {
      conv.loading = false
      for (const msg of conv.messages) msg.streaming = false
    }
    return data
  } catch {
    return null
  }
}

const saved = loadFromStorage()
const mode = ref<WorkspaceMode>(adminToolsEnabled && saved?.mode === 'evals' ? 'evals' : 'research')
const userId = ref(saved?.userId ?? 'user01')
const tenantId = ref(saved?.tenantId ?? 'default_tenant')
const adminKey = ref(adminToolsEnabled ? sessionStorage.getItem(ADMIN_SESSION_KEY) ?? '' : '')
const conversations = ref<Conversation[]>(saved?.conversations?.length ? saved.conversations : [makeConversation()])
const activeIndex = ref(Math.min(Math.max(Number(saved?.activeIndex ?? 0), 0), conversations.value.length - 1))
const query = ref('')
const errorMessage = ref('')
const abortController = ref<AbortController | null>(null)
const messageListRef = ref<HTMLElement | null>(null)
const composerRef = ref<HTMLTextAreaElement | null>(null)

const datasets = ref<EvalDatasetInfo[]>([])
const selectedDataset = ref('smoke')
const evalMaxCases = ref(1)
const evalMaxIterations = ref(1)
const evalScoreThreshold = ref(70)
const evalEnableMemory = ref(false)
const evalRuns = ref<EvalRunSummary[]>([])
const selectedRun = ref<EvalRunDetail | null>(null)
const selectedCase = ref<EvalCaseResult | null>(null)
const liveCaseResults = ref<EvalCaseResult[]>([])
const evalRunning = ref(false)
const evalError = ref('')
const evalLog = ref<string[]>([])
const ragFileInputRef = ref<HTMLInputElement | null>(null)
const ragUploading = ref(false)
const ragStatus = ref<RagStatus | null>(null)
const ragStatusMessage = ref('')
const ragSearchQuery = ref('')
const ragSearchResults = ref<RagSearchResult[]>([])

const saveToStorage = () => {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      mode: adminToolsEnabled ? mode.value : 'research',
      conversations: conversations.value,
      activeIndex: activeIndex.value,
      userId: userId.value,
      tenantId: tenantId.value,
    }),
  )
}

watch([mode, conversations, activeIndex, userId, tenantId], saveToStorage, { deep: true })
watch(adminKey, (value) => {
  if (!adminToolsEnabled) return
  const trimmed = value.trim()
  if (trimmed) sessionStorage.setItem(ADMIN_SESSION_KEY, trimmed)
  else sessionStorage.removeItem(ADMIN_SESSION_KEY)
})

const activeConv = computed(() => conversations.value[activeIndex.value])
const runRows = computed(() => evalRuns.value.slice(0, 8))
const caseRows = computed<EvalRunCaseSummary[]>(() => {
  if (selectedRun.value) return selectedRun.value.summary.cases
  return liveCaseResults.value.map((result) => ({
    case_id: result.case.id,
    query: result.case.query,
    status: result.status,
    score: result.score,
    latency_ms: result.latency_ms,
    failed_evaluators: result.failed_evaluators,
    suspected_stages: result.suspected_stages,
  }))
})
const currentSummary = computed<EvalRunSummary | null>(() => selectedRun.value?.summary ?? null)

const starterPrompts = [
  '请总结这个项目的多智能体链路，并指出最容易出问题的节点。',
  'LangGraph vs CrewAI：从架构、工具调用、适用场景三个方面对比。',
  '请把上线 DeepResearch MVP 拆成两周计划，包含验收标准和风险。',
]

const escapeHtml = (value: string): string =>
  value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')

const markdownToHtml = (markdown: string): string => {
  const codeBlocks: string[] = []
  let text = markdown.replace(/```([\s\S]*?)```/g, (_, block) => {
    const index = codeBlocks.length
    codeBlocks.push(`<pre><code>${escapeHtml(String(block).trim())}</code></pre>`)
    return `@@CODE_BLOCK_${index}@@`
  })
  const lines = text.split('\n')
  const out: string[] = []
  let inList = false
  const closeList = () => {
    if (inList) {
      out.push('</ul>')
      inList = false
    }
  }
  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) {
      closeList()
      continue
    }
    if (line.startsWith('# ')) {
      closeList()
      out.push(`<h1>${escapeHtml(line.slice(2))}</h1>`)
      continue
    }
    if (line.startsWith('## ')) {
      closeList()
      out.push(`<h2>${escapeHtml(line.slice(3))}</h2>`)
      continue
    }
    if (line.startsWith('### ')) {
      closeList()
      out.push(`<h3>${escapeHtml(line.slice(4))}</h3>`)
      continue
    }
    if (line.startsWith('- ') || line.startsWith('* ')) {
      if (!inList) {
        out.push('<ul>')
        inList = true
      }
      out.push(`<li>${escapeHtml(line.slice(2))}</li>`)
      continue
    }
    closeList()
    out.push(`<p>${escapeHtml(line)}</p>`)
  }
  closeList()
  let html = out.join('')
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')
  html = html.replace(/\[([^[\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
  html = html.replace(/@@CODE_BLOCK_(\d+)@@/g, (_, idx) => codeBlocks[Number(idx)] || '')
  return html
}

const renderMessageHtml = (message: ChatMessage) => markdownToHtml(message.content || '')

const scrollToBottom = async () => {
  await nextTick()
  const el = messageListRef.value
  if (el) el.scrollTop = el.scrollHeight
}

const isNearBottom = (): boolean => {
  const el = messageListRef.value
  if (!el) return true
  return el.scrollHeight - el.scrollTop - el.clientHeight < 80
}

const smartScroll = async () => {
  if (isNearBottom()) await scrollToBottom()
}

const createNewChat = () => {
  const conv = makeConversation()
  conversations.value.push(conv)
  activeIndex.value = conversations.value.length - 1
  errorMessage.value = ''
  query.value = ''
  nextTick(() => composerRef.value?.focus())
}

const switchConversation = (index: number) => {
  if (!conversations.value[index] || index === activeIndex.value) return
  activeIndex.value = index
  errorMessage.value = ''
  nextTick(() => scrollToBottom())
}

const deleteConversation = (index: number) => {
  if (conversations.value.length <= 1) return
  conversations.value.splice(index, 1)
  if (activeIndex.value >= conversations.value.length) {
    activeIndex.value = conversations.value.length - 1
  }
}

const usePrompt = async (prompt: string) => {
  query.value = prompt
  errorMessage.value = ''
  await nextTick()
  composerRef.value?.focus()
}

const stopResearch = () => {
  abortController.value?.abort()
  abortController.value = null
}

const typewriterRender = async (convIndex: number, messageId: string, fullText: string) => {
  const msg = conversations.value[convIndex]?.messages.find((m) => m.id === messageId)
  if (!msg) return
  let pos = 0
  const total = fullText.length
  return new Promise<void>((resolve) => {
    const timer = setInterval(() => {
      const chunk = total > 3000 ? 8 : total > 1000 ? 5 : 3
      pos += chunk
      if (pos >= total) {
        msg.content = fullText
        msg.streaming = false
        clearInterval(timer)
        resolve()
      } else {
        msg.content = fullText.slice(0, pos)
      }
      if (isNearBottom()) nextTick(() => scrollToBottom())
    }, 12)
  })
}

const runResearch = async () => {
  const userText = query.value.trim()
  const convIndex = activeIndex.value
  const conv = conversations.value[convIndex]
  if (!conv || !userText || conv.loading) return

  if (conv.messages.length <= 1) {
    conv.title = userText.slice(0, 30) + (userText.length > 30 ? '...' : '')
  }

  conv.loading = true
  errorMessage.value = ''
  query.value = ''
  conv.messages.push({ id: `u-${Date.now()}`, role: 'user', content: userText })
  const assistantId = `a-${Date.now()}`
  conv.messages.push({ id: assistantId, role: 'assistant', content: '', streaming: true })

  const updateAssistant = (content: string) => {
    const msg = conversations.value[convIndex]?.messages.find((m) => m.id === assistantId)
    if (msg) msg.content = content
  }

  const controller = new AbortController()
  abortController.value = controller
  await scrollToBottom()

  try {
    const response = await fetch('/api/v1/research/stream', {
      method: 'POST',
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: userText,
        user_id: userId.value.trim() || 'default_user',
        thread_id: conv.threadId,
        tenant_id: tenantId.value.trim() || 'default_tenant',
      }),
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || `请求失败: ${response.status}`)
    }
    if (!response.body) throw new Error('流式响应不可用')

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    const phaseLines: string[] = []

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split('\n\n')
      buffer = parts.pop() || ''
      for (const part of parts) {
        if (!part.startsWith('data: ')) continue
        const jsonText = part.slice(6).trim()
        if (!jsonText) continue
        const event = JSON.parse(jsonText) as StreamEvent
        if (event.type === 'status' || event.type === 'phase' || event.type === 'route') {
          const prefix = event.type === 'phase' && event.node ? `${event.node}: ` : ''
          phaseLines.push(`${prefix}${event.message || ''}`)
          updateAssistant(phaseLines.slice(-6).join('\n'))
          smartScroll()
        }
        if (event.type === 'final') {
          await typewriterRender(convIndex, assistantId, event.final || '已完成，但没有返回正文。')
        }
        if (event.type === 'error') {
          throw new Error(event.message || '服务端执行异常')
        }
      }
    }
  } catch (error) {
    const msg = conversations.value[convIndex]?.messages.find((m) => m.id === assistantId)
    if (error instanceof DOMException && error.name === 'AbortError') {
      if (msg) {
        msg.content = msg.content || '已停止生成。'
        msg.streaming = false
      }
    } else {
      errorMessage.value = error instanceof Error ? error.message : '请求失败'
      if (msg) {
        msg.content = `请求失败：${errorMessage.value}`
        msg.streaming = false
      }
    }
  } finally {
    abortController.value = null
    conv.loading = false
    await smartScroll()
  }
}

const fetchJson = async <T,>(url: string, options?: RequestInit): Promise<T> => {
  const response = await fetch(url, options)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `请求失败: ${response.status}`)
  }
  return response.json() as Promise<T>
}

const adminHeaders = (headers?: HeadersInit): Headers => {
  const next = new Headers(headers)
  const key = adminKey.value.trim()
  if (key) next.set('X-Admin-Key', key)
  return next
}

const loadRagStatus = async () => {
  if (!adminToolsEnabled) return
  ragStatus.value = await fetchJson<RagStatus>(
    `/api/v1/rag/status?tenant_id=${encodeURIComponent(tenantId.value.trim() || 'default_tenant')}`,
    { headers: adminHeaders() },
  )
}

const openRagFilePicker = () => {
  ragFileInputRef.value?.click()
}

const uploadRagDocuments = async (event: Event) => {
  if (!adminToolsEnabled) return
  const input = event.target as HTMLInputElement
  const files = Array.from(input.files || [])
  input.value = ''
  if (!files.length || ragUploading.value) return

  ragUploading.value = true
  ragStatusMessage.value = ''
  ragSearchResults.value = []

  try {
    for (const file of files) {
      const form = new FormData()
      form.append('file', file)
      form.append('tenant_id', tenantId.value.trim() || 'default_tenant')
      form.append('user_id', userId.value.trim() || 'default_user')
      form.append('thread_id', activeConv.value?.threadId || '')
      const response = await fetch('/api/v1/rag/documents', {
        method: 'POST',
        headers: adminHeaders(),
        body: form,
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `上传失败: ${response.status}`)
      }
      const result = (await response.json()) as { document: RagDocumentRecord }
      ragStatusMessage.value = `${result.document.filename} 已入库，${result.document.chunks} 个 chunks`
    }
    await loadRagStatus()
  } catch (error) {
    ragStatusMessage.value = error instanceof Error ? error.message : '上传失败'
  } finally {
    ragUploading.value = false
  }
}

const searchRagDocuments = async () => {
  if (!adminToolsEnabled) return
  const text = ragSearchQuery.value.trim()
  if (!text) return
  try {
    const data = await fetchJson<{ results: RagSearchResult[] }>('/api/v1/rag/search', {
      method: 'POST',
      headers: adminHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        query: text,
        tenant_id: tenantId.value.trim() || 'default_tenant',
        limit: 5,
      }),
    })
    ragSearchResults.value = data.results
    ragStatusMessage.value = data.results.length ? `检索到 ${data.results.length} 条片段` : '没有检索到相关片段'
  } catch (error) {
    ragStatusMessage.value = error instanceof Error ? error.message : '检索失败'
  }
}

const loadEvalDatasets = async () => {
  if (!adminToolsEnabled) return
  const data = await fetchJson<{ datasets: EvalDatasetInfo[] }>('/api/v1/evals/datasets', {
    headers: adminHeaders(),
  })
  datasets.value = data.datasets
  if (!datasets.value.find((item) => item.id === selectedDataset.value) && datasets.value[0]) {
    selectedDataset.value = datasets.value[0].id
  }
}

const loadEvalRuns = async () => {
  if (!adminToolsEnabled) return
  const data = await fetchJson<{ runs: EvalRunSummary[] }>('/api/v1/evals/runs', {
    headers: adminHeaders(),
  })
  evalRuns.value = data.runs
}

const loadEvalRun = async (runId: string) => {
  if (!adminToolsEnabled) return
  selectedRun.value = await fetchJson<EvalRunDetail>(`/api/v1/evals/runs/${encodeURIComponent(runId)}`, {
    headers: adminHeaders(),
  })
  selectedCase.value = selectedRun.value.cases[0] ?? null
}

const loadEvalCase = async (caseId: string) => {
  if (!selectedRun.value) {
    selectedCase.value = liveCaseResults.value.find((item) => item.case.id === caseId) ?? null
    return
  }
  selectedCase.value = await fetchJson<EvalCaseResult>(
    `/api/v1/evals/runs/${encodeURIComponent(selectedRun.value.summary.run_id)}/cases/${encodeURIComponent(caseId)}`,
    { headers: adminHeaders() },
  )
}

const runEval = async () => {
  if (!adminToolsEnabled || evalRunning.value) return
  evalRunning.value = true
  evalError.value = ''
  evalLog.value = []
  liveCaseResults.value = []
  selectedRun.value = null
  selectedCase.value = null

  try {
    const response = await fetch('/api/v1/evals/run/stream', {
      method: 'POST',
      headers: adminHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        dataset_id: selectedDataset.value,
        max_cases: Number(evalMaxCases.value) || 1,
        max_iterations: Number(evalMaxIterations.value) || 1,
        enable_memory: evalEnableMemory.value,
        user_id: userId.value.trim() || 'eval_user',
        tenant_id: tenantId.value.trim() || 'eval_tenant',
        score_threshold: Number(evalScoreThreshold.value) || 70,
      }),
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || `评测启动失败: ${response.status}`)
    }
    if (!response.body) throw new Error('评测流不可用')

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split('\n\n')
      buffer = parts.pop() || ''
      for (const part of parts) {
        if (!part.startsWith('data: ')) continue
        const jsonText = part.slice(6).trim()
        if (!jsonText) continue
        const event = JSON.parse(jsonText) as EvalStreamEvent
        handleEvalEvent(event)
      }
    }
  } catch (error) {
    evalError.value = error instanceof Error ? error.message : '评测失败'
  } finally {
    evalRunning.value = false
    await loadEvalRuns().catch(() => undefined)
  }
}

const handleEvalEvent = (event: EvalStreamEvent) => {
  if (event.type === 'run_start') {
    evalLog.value.push(`Run ${event.run_id} started, ${event.total_cases} cases`)
  }
  if (event.type === 'case_start') {
    evalLog.value.push(`[${event.index}/${event.total}] ${event.case_id}`)
  }
  if (event.type === 'case_result') {
    liveCaseResults.value.push(event.result)
    selectedCase.value = event.result
    evalLog.value.push(`${event.case_id}: ${event.result.status}, score ${event.result.score}`)
  }
  if (event.type === 'summary') {
    evalLog.value.push(`Run ${event.run_id} completed, pass rate ${event.summary.pass_rate.toFixed(1)}%`)
    loadEvalRun(event.run_id).catch(() => undefined)
  }
  if (event.type === 'error') {
    evalError.value = event.message
  }
}

const scoreClass = (score: number) => {
  if (score >= 80) return 'score-good'
  if (score >= 60) return 'score-warn'
  return 'score-bad'
}

const statusClass = (status: string) => (status === 'passed' ? 'status-pass' : 'status-fail')

const formatTime = (value?: string | null) => {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

const stringify = (value: unknown) => JSON.stringify(value ?? {}, null, 2)

const refreshAdminWorkspace = async () => {
  if (!adminToolsEnabled || !adminKey.value.trim()) return
  try {
    await Promise.all([
      loadEvalDatasets(),
      loadEvalRuns(),
      loadRagStatus().catch((error) => {
        ragStatusMessage.value = error instanceof Error ? error.message : 'RAG 状态加载失败'
      }),
    ])
  } catch (error) {
    evalError.value = error instanceof Error ? error.message : '评测数据加载失败'
  }
}

onMounted(async () => {
  if (!adminToolsEnabled) {
    mode.value = 'research'
    return
  }
  await refreshAdminWorkspace()
})
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <strong>DeepResearch</strong>
        <span>Agent Workspace</span>
      </div>

      <div v-if="adminToolsEnabled" class="mode-tabs" role="tablist">
        <button :class="{ active: mode === 'research' }" @click="mode = 'research'">研究</button>
        <button :class="{ active: mode === 'evals' }" @click="mode = 'evals'">评测</button>
      </div>

      <template v-if="mode === 'research'">
        <button class="primary-btn" @click="createNewChat">新建会话</button>
        <div class="conversation-list">
          <button
            v-for="(conv, index) in conversations"
            :key="conv.id"
            class="conversation-item"
            :class="{ active: index === activeIndex }"
            @click="switchConversation(index)"
          >
            <span>{{ conv.title }}</span>
            <i v-if="conv.loading">运行中</i>
          </button>
        </div>
        <div class="prompt-stack">
          <button v-for="item in starterPrompts" :key="item" @click="usePrompt(item)">
            {{ item }}
          </button>
        </div>
        <div v-if="adminToolsEnabled" class="rag-panel">
          <div class="rag-panel-head">
            <strong>RAG 文档</strong>
            <button class="mini-btn" type="button" @click="loadRagStatus">刷新</button>
          </div>
          <input
            ref="ragFileInputRef"
            class="hidden-file"
            type="file"
            accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            multiple
            @change="uploadRagDocuments"
          />
          <button class="ghost-btn upload-btn" :disabled="ragUploading" type="button" @click="openRagFilePicker">
            {{ ragUploading ? '入库中' : '上传 PDF / Word' }}
          </button>
          <div class="rag-meta">
            <span>{{ ragStatus?.collection || '未初始化 collection' }}</span>
            <span>{{ ragStatus?.runtime_initialized ? '已连接' : '待上传初始化' }}</span>
          </div>
          <p v-if="ragStatusMessage" class="rag-message">{{ ragStatusMessage }}</p>
          <div v-if="ragStatus?.documents.length" class="rag-doc-list">
            <span v-for="doc in ragStatus.documents.slice(0, 3)" :key="doc.doc_id">
              {{ doc.filename }} · {{ doc.chunks }} chunks
            </span>
          </div>
          <div class="rag-search">
            <input
              v-model="ragSearchQuery"
              placeholder="检索本地 RAG"
              @keydown.enter.prevent="searchRagDocuments"
            />
            <button class="mini-btn" type="button" @click="searchRagDocuments">查</button>
          </div>
          <div v-if="ragSearchResults.length" class="rag-search-results">
            <span v-for="result in ragSearchResults.slice(0, 2)" :key="`${result.doc_id}-${result.snippet}`">
              {{ result.title || result.doc_id }}：{{ result.snippet }}
            </span>
          </div>
        </div>
      </template>

      <template v-else>
        <label class="field">
          <span>数据集</span>
          <select v-model="selectedDataset">
            <option v-for="item in datasets" :key="item.id" :value="item.id">
              {{ item.id }} ({{ item.case_count }})
            </option>
          </select>
        </label>
        <div class="two-fields">
          <label class="field">
            <span>Case</span>
            <input v-model.number="evalMaxCases" min="1" max="200" type="number" />
          </label>
          <label class="field">
            <span>迭代</span>
            <input v-model.number="evalMaxIterations" min="1" max="6" type="number" />
          </label>
        </div>
        <label class="field">
          <span>阈值</span>
          <input v-model.number="evalScoreThreshold" min="0" max="100" type="number" />
        </label>
        <label class="check-row">
          <input v-model="evalEnableMemory" type="checkbox" />
          <span>启用记忆</span>
        </label>
        <button class="primary-btn" :disabled="evalRunning || !selectedDataset" @click="runEval">
          {{ evalRunning ? '评测中' : '运行评测' }}
        </button>
        <div class="run-list">
          <button
            v-for="run in runRows"
            :key="run.run_id"
            class="run-item"
            :class="{ active: selectedRun?.summary.run_id === run.run_id }"
            @click="loadEvalRun(run.run_id)"
          >
            <span>{{ run.dataset_id }}</span>
            <i>{{ run.average_score.toFixed(0) }}</i>
          </button>
        </div>
      </template>

      <div class="sidebar-footer">
        <label v-if="adminToolsEnabled" class="field">
          <span>Admin Key</span>
          <input
            v-model="adminKey"
            autocomplete="off"
            placeholder="管理密钥"
            type="password"
            @change="refreshAdminWorkspace"
          />
        </label>
        <label class="field">
          <span>User ID</span>
          <input v-model="userId" />
        </label>
        <label class="field">
          <span>Tenant ID</span>
          <input v-model="tenantId" />
        </label>
      </div>
    </aside>

    <main class="workspace">
      <header class="topbar">
        <div>
          <h1>{{ mode === 'research' ? '研究工作台' : 'Agent 评测' }}</h1>
          <p v-if="mode === 'research'">{{ activeConv?.threadId }}</p>
          <p v-else>{{ currentSummary?.run_id || '未选择评测运行' }}</p>
        </div>
        <div class="top-actions">
          <button v-if="mode === 'evals'" class="ghost-btn" @click="loadEvalRuns">刷新</button>
          <button v-if="mode === 'research' && activeConv?.loading" class="danger-btn" @click="stopResearch">停止</button>
        </div>
      </header>

      <section v-if="mode === 'research'" class="research-view">
        <div ref="messageListRef" class="message-list">
          <div
            v-for="message in activeConv?.messages"
            :key="message.id"
            class="message-row"
            :class="`role-${message.role}`"
          >
            <div class="avatar">{{ message.role === 'user' ? '我' : 'AI' }}</div>
            <div
              class="bubble markdown-body"
              :class="{ streaming: message.streaming }"
              v-html="renderMessageHtml(message)"
            ></div>
          </div>
        </div>

        <div class="composer">
          <textarea
            ref="composerRef"
            v-model="query"
            class="composer-input"
            :disabled="activeConv?.loading"
            placeholder="输入问题"
            @keydown.enter.exact.prevent="runResearch"
          />
          <button class="send-btn" :disabled="!query.trim() || activeConv?.loading" @click="runResearch">发送</button>
        </div>
        <p v-if="errorMessage" class="error-line">{{ errorMessage }}</p>
      </section>

      <section v-else class="eval-view">
        <div class="metrics-strip">
          <article>
            <span>总数</span>
            <strong>{{ currentSummary?.total_cases ?? liveCaseResults.length }}</strong>
          </article>
          <article>
            <span>通过率</span>
            <strong>{{ currentSummary ? `${currentSummary.pass_rate.toFixed(1)}%` : '-' }}</strong>
          </article>
          <article>
            <span>平均分</span>
            <strong>{{ currentSummary ? currentSummary.average_score.toFixed(1) : '-' }}</strong>
          </article>
          <article>
            <span>失败</span>
            <strong>{{ currentSummary?.failed_cases ?? liveCaseResults.filter((item) => !item.passed).length }}</strong>
          </article>
        </div>

        <div class="eval-main">
          <section class="case-table">
            <div class="section-head">
              <h2>Case</h2>
              <span>{{ formatTime(currentSummary?.completed_at) }}</span>
            </div>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Case</th>
                    <th>状态</th>
                    <th>分数</th>
                    <th>节点</th>
                    <th>失败项</th>
                    <th>耗时</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="row in caseRows"
                    :key="row.case_id"
                    :class="{ selected: selectedCase?.case.id === row.case_id }"
                    @click="loadEvalCase(row.case_id)"
                  >
                    <td>
                      <strong>{{ row.case_id }}</strong>
                      <span>{{ row.query }}</span>
                    </td>
                    <td><i :class="['status-pill', statusClass(row.status)]">{{ row.status }}</i></td>
                    <td><b :class="scoreClass(row.score)">{{ row.score }}</b></td>
                    <td>{{ row.suspected_stages.join(', ') || '-' }}</td>
                    <td>{{ row.failed_evaluators.join(', ') || '-' }}</td>
                    <td>{{ row.latency_ms }}ms</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <aside class="case-detail">
            <div class="section-head">
              <h2>诊断</h2>
              <span v-if="selectedCase">{{ selectedCase.score }} / {{ selectedCase.threshold }}</span>
            </div>

            <template v-if="selectedCase">
              <div class="detail-block">
                <label>问题</label>
                <p>{{ selectedCase.case.query }}</p>
              </div>

              <div class="detail-block">
                <label>失败原因</label>
                <ul v-if="selectedCase.issues.length" class="issue-list">
                  <li v-for="issue in selectedCase.issues" :key="`${issue.stage}-${issue.evaluator}-${issue.message}`">
                    <strong>{{ issue.stage }}</strong>
                    <span>{{ issue.evaluator }}: {{ issue.message }}</span>
                  </li>
                </ul>
                <p v-else>无</p>
              </div>

              <div class="metric-list">
                <div v-for="metric in selectedCase.metrics" :key="metric.name" class="metric-row">
                  <span>{{ metric.name }}</span>
                  <b :class="metric.passed ? 'score-good' : 'score-bad'">{{ metric.score }}</b>
                  <small>{{ metric.reason }}</small>
                </div>
              </div>

              <details open>
                <summary>Trace</summary>
                <dl class="trace-grid">
                  <dt>route</dt>
                  <dd>{{ selectedCase.trace.route }}</dd>
                  <dt>intent</dt>
                  <dd>{{ selectedCase.trace.intent || '-' }}</dd>
                  <dt>evidence</dt>
                  <dd>{{ selectedCase.trace.counts.evidence_pool ?? 0 }}</dd>
                  <dt>citations</dt>
                  <dd>{{ selectedCase.trace.citations.used?.join(', ') || '-' }}</dd>
                </dl>
              </details>

              <details>
                <summary>最终回答</summary>
                <div class="markdown-body final-preview" v-html="markdownToHtml(selectedCase.trace.final)" />
              </details>

              <details>
                <summary>证据</summary>
                <pre>{{ stringify(selectedCase.trace.evidence_pool) }}</pre>
              </details>
            </template>
            <p v-else class="empty-state">暂无 case</p>
          </aside>
        </div>

        <div class="eval-log">
          <span v-if="evalRunning">Running</span>
          <span v-for="line in evalLog.slice(-5)" :key="line">{{ line }}</span>
        </div>
        <p v-if="evalError" class="error-line">{{ evalError }}</p>
      </section>
    </main>
  </div>
</template>
