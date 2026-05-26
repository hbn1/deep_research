<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'

type StreamEvent = {
  type: 'status' | 'phase' | 'route' | 'final' | 'error'
  message?: string
  final?: string
  node?: string
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

const STORAGE_KEY = 'deepresearch_sessions'

const makeConversation = (): Conversation => ({
  id: `conv-${Date.now()}`,
  title: '新会话',
  messages: [
    {
      id: `m-${Date.now()}`,
      role: 'assistant',
      content: '你好，我是 DeepResearch。你可以直接提问，我会根据意图自动走快速回答或完整研究链路。',
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
    // sanitize: strip transient states
    for (const conv of data.conversations) {
      conv.loading = false
      for (const msg of conv.messages) {
        msg.streaming = false
      }
    }
    return data
  } catch {
    return null
  }
}

const saveToStorage = () => {
  const payload = {
    conversations: conversations.value,
    activeIndex: activeIndex.value,
    userId: userId.value,
    tenantId: tenantId.value,
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
}

const saved = loadFromStorage()
const userId = ref(saved?.userId ?? 'user01')
const tenantId = ref(saved?.tenantId ?? 'default_tenant')
const conversations = ref<Conversation[]>(saved?.conversations ?? [makeConversation()])
const activeIndex = ref(saved?.activeIndex ?? 0)
const query = ref('')
const errorMessage = ref('')
const abortController = ref<AbortController | null>(null)
const messageListRef = ref<HTMLElement | null>(null)
const composerRef = ref<HTMLTextAreaElement | null>(null)

watch([conversations, activeIndex, userId, tenantId], saveToStorage, { deep: true })

const stopResearch = () => {
  abortController.value?.abort()
  abortController.value = null
}

const activeConv = computed(() => conversations.value[activeIndex.value])

const starterPrompts = [
  {
    title: '深度调研',
    prompt: '请调研"企业知识库 Agent 平台"市场，按市场规模、主要竞品、收费模式三部分输出，并在每部分附上可追溯来源链接。',
  },
  {
    title: '方案对比',
    prompt: '我们要做多 Agent 研究助手，请对比"纯大模型直答""RAG 单 Agent""多 Agent 协作"三种方案，给出优缺点、适用场景与推荐结论。',
  },
  {
    title: '知识问答',
    prompt: '请解释这个项目里"意图分流"的作用，以及简单问题和复杂问题分别会走哪条链路。',
  },
  {
    title: '落地计划',
    prompt: '请把"上线一个可用的 DeepResearch MVP"拆成两周计划，按每天输出任务、验收标准和风险点。',
  },
]
const capabilityHighlights = [
  { title: '多智能体编排', desc: '自动完成规划、检索、证据裁判、分析与写作，减少手工研究路径。' },
  { title: '双源检索融合', desc: '网络信息与本地知识库并行召回，输出结论同时保留来源可追溯性。' },
  { title: '会话记忆增强', desc: '跨轮次继承用户偏好与历史任务，持续提升回答一致性和效率。' },
]
const landingMetrics = [
  { label: '执行模式', value: 'Quick + Deep' },
  { label: '检索来源', value: 'Web + Local' },
  { label: '输出风格', value: '结论 + 证据' },
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
    if (line.startsWith('# ')) { closeList(); out.push(`<h1>${escapeHtml(line.slice(2))}</h1>`); continue }
    if (line.startsWith('## ')) { closeList(); out.push(`<h2>${escapeHtml(line.slice(3))}</h2>`); continue }
    if (line.startsWith('### ')) { closeList(); out.push(`<h3>${escapeHtml(line.slice(4))}</h3>`); continue }
    if (line.startsWith('- ') || line.startsWith('* ')) {
      if (!inList) { out.push('<ul>'); inList = true }
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
  if (index === activeIndex.value) return
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
  errorMessage.value = ''
}

const usePrompt = async (prompt: string) => {
  query.value = prompt
  errorMessage.value = ''
  await nextTick()
  composerRef.value?.focus()
}

const applyStarterByIndex = (index: number) => {
  const target = starterPrompts[index]
  if (!target) return
  usePrompt(target.prompt)
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
      if (isNearBottom()) {
        nextTick(() => scrollToBottom())
      }
    }, 12)
  })
}

const runResearch = async () => {
  const userText = query.value.trim()
  const convIndex = activeIndex.value
  const conv = conversations.value[convIndex]
  if (!userText || conv.loading) return

  // auto-title from first user message
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
          const prefix = event.type === 'phase' && event.node ? `${event.node} ` : ''
          phaseLines.push(`${prefix}${event.message || ''}`)
          updateAssistant(phaseLines.slice(-6).join('\n'))
          smartScroll()
        }
        if (event.type === 'final') {
          const finalContent = event.final || '已完成，但未返回正文。'
          await typewriterRender(convIndex, assistantId, finalContent)
        }
        if (event.type === 'error') {
          throw new Error(event.message || '服务端执行异常')
        }
      }
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      const msg = conversations.value[convIndex]?.messages.find((m) => m.id === assistantId)
      if (msg) {
        msg.content = msg.content || '已停止生成。'
        msg.streaming = false
      }
    } else {
      errorMessage.value = error instanceof Error ? error.message : '请求失败'
      const msg = conversations.value[convIndex]?.messages.find((m) => m.id === assistantId)
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
</script>

<template>
  <div class="chat-shell">
    <aside class="chat-sidebar">
      <div class="sidebar-brand">
        <p class="brand-badge">AI Copilot</p>
        <h1>DeepResearch</h1>
        <p class="brand-desc">多智能体研究工作台，支持快速回答与深度调研。</p>
      </div>
      <div class="sidebar-head">
        <button class="new-chat-btn" @click="createNewChat">+ 新建会话</button>
      </div>

      <div class="conv-list">
        <div
          v-for="(conv, index) in conversations"
          :key="conv.id"
          class="conv-item"
          :class="{ 'conv-active': index === activeIndex }"
          @click="switchConversation(index)"
        >
          <span class="conv-title">{{ conv.title }}</span>
          <button
            v-if="conversations.length > 1"
            class="conv-delete"
            @click.stop="deleteConversation(index)"
            title="删除会话"
          >&times;</button>
        </div>
      </div>

      <div class="quick-entry">
        <p class="section-title">推荐起手问题</p>
        <button
          v-for="item in starterPrompts.slice(0, 3)"
          :key="item.title"
          class="quick-entry-btn"
          @click="usePrompt(item.prompt)"
        >
          {{ item.title }}
        </button>
      </div>
      <div class="settings-group">
        <label>User ID</label>
        <input v-model="userId" class="sidebar-input" />
      </div>
      <div class="settings-group">
        <label>Tenant ID</label>
        <input v-model="tenantId" class="sidebar-input" />
      </div>
      <p class="hint-text">当前会话 Thread：{{ activeConv?.threadId }}</p>
    </aside>

    <main class="chat-main">
      <header class="main-header">
        <div>
          <h2>DeepResearch Enterprise Workspace</h2>
          <p>面向业务团队的企业级智能研究台，支持从问题定义到结论落地的完整链路。</p>
        </div>
        <div class="header-tags">
          <span>Evidence-Driven</span>
          <span>Structured Output</span>
          <span>Memory-Powered</span>
        </div>
      </header>

      <div ref="messageListRef" class="message-list">
        <section v-if="activeConv && activeConv.messages.length <= 1" class="onboarding-panel">
          <div class="hero-panel">
            <p class="hero-badge">商业研究 · 策略分析 · 知识问答</p>
            <h3>第一步先讲清目标，再交给 DeepResearch 自动推进</h3>
            <p class="hero-desc">
              推荐提问结构：目标 + 背景约束 + 期望输出。系统会自动选择快速回答或深度研究链路。
            </p>
            <div class="hero-actions">
              <button class="hero-btn primary" @click="applyStarterByIndex(0)">快速开始调研</button>
              <button class="hero-btn" @click="applyStarterByIndex(1)">查看方案对比</button>
            </div>
            <div class="metric-grid">
              <article v-for="item in landingMetrics" :key="item.label">
                <p>{{ item.label }}</p>
                <strong>{{ item.value }}</strong>
              </article>
            </div>
          </div>
          <div class="capability-grid">
            <article v-for="item in capabilityHighlights" :key="item.title" class="capability-card">
              <h4>{{ item.title }}</h4>
              <p>{{ item.desc }}</p>
            </article>
          </div>
          <div class="guide-panel">
            <h4>提问指南</h4>
            <div class="guide-grid">
              <article>
                <h5>1. 说明目标</h5>
                <p>你要解决什么问题、面向谁、希望达到什么结果。</p>
              </article>
              <article>
                <h5>2. 提供上下文</h5>
                <p>给出已知信息、时间范围、数据口径、业务限制。</p>
              </article>
              <article>
                <h5>3. 指定输出</h5>
                <p>例如"表格输出""附来源链接""分点行动清单"。</p>
              </article>
            </div>
          </div>
          <div class="prompt-list">
            <button v-for="item in starterPrompts" :key="item.prompt" class="prompt-chip" @click="usePrompt(item.prompt)">
              {{ item.prompt }}
            </button>
          </div>
        </section>

        <template v-if="activeConv">
          <div
            v-for="message in activeConv.messages"
            :key="message.id"
            class="message-row"
            :class="`role-${message.role}`"
          >
            <div class="avatar">{{ message.role === 'user' ? '你' : 'AI' }}</div>
            <div
              class="bubble markdown-body"
              :class="{ 'bubble-streaming': message.streaming }"
              v-html="renderMessageHtml(message)"
            ></div>
            <span v-if="message.streaming" class="typing-cursor"></span>
          </div>
        </template>
      </div>

      <div class="composer">
        <textarea
          v-model="query"
          ref="composerRef"
          class="composer-input"
          :disabled="activeConv?.loading"
          placeholder="输入你的问题，回车发送（Shift + Enter 换行）"
          @keydown.enter.exact.prevent="runResearch"
        />
        <button v-if="activeConv?.loading" class="stop-btn" @click="stopResearch" title="停止生成">
          <span class="stop-icon"></span>
        </button>
        <button v-else class="send-btn" :disabled="!query.trim()" @click="runResearch">
          发送
        </button>
      </div>
      <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
    </main>
  </div>
</template>
