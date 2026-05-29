import { computed, nextTick, ref, watch } from "vue";

export type StreamEvent = {
  type: "status" | "phase" | "route" | "final" | "error";
  message?: string;
  final?: string;
  node?: string;
  route?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
};

export type Conversation = {
  id: string;
  title: string;
  messages: ChatMessage[];
  threadId: string;
  loading: boolean;
};

const STORAGE_KEY = "deepresearch_sessions";

function makeConversation(): Conversation {
  return {
    id: `conv-${Date.now()}`,
    title: "New Chat",
    messages: [
      {
        id: `m-${Date.now()}`,
        role: "assistant",
        content:
          "Hello, I am DeepResearch. You can ask me anything and I will automatically choose between quick answer or deep research.",
      },
    ],
    threadId: `thread_${Date.now()}`,
    loading: false,
  };
}

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data.conversations?.length) return null;
    for (const conv of data.conversations) {
      conv.loading = false;
      for (const msg of conv.messages) {
        msg.streaming = false;
      }
    }
    return data;
  } catch {
    return null;
  }
}

export function useChat() {
  const saved = loadFromStorage();
  const userId = ref(saved?.userId ?? "user01");
  const tenantId = ref(saved?.tenantId ?? "default_tenant");
  const conversations = ref<Conversation[]>(
    saved?.conversations ?? [makeConversation()]
  );
  const activeIndex = ref(saved?.activeIndex ?? 0);
  const query = ref("");
  const errorMessage = ref("");
  const abortController = ref<AbortController | null>(null);

  watch(
    [conversations, activeIndex, userId, tenantId],
    () => {
      const payload = {
        conversations: conversations.value,
        activeIndex: activeIndex.value,
        userId: userId.value,
        tenantId: tenantId.value,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    },
    { deep: true }
  );

  const activeConv = computed(() => conversations.value[activeIndex.value]);

  function newConversation() {
    const conv = makeConversation();
    conversations.value.unshift(conv);
    activeIndex.value = 0;
  }

  function selectConversation(index: number) {
    activeIndex.value = index;
  }

  function deleteConversation(index: number) {
    if (conversations.value.length <= 1) {
      conversations.value = [makeConversation()];
      activeIndex.value = 0;
      return;
    }
    conversations.value.splice(index, 1);
    if (activeIndex.value >= conversations.value.length) {
      activeIndex.value = conversations.value.length - 1;
    }
  }

  function stopResearch() {
    abortController.value?.abort();
    abortController.value = null;
  }

  async function runResearch() {
    const q = query.value.trim();
    if (!q) return;
    errorMessage.value = "";

    const conv = activeConv.value;
    if (!conv) return;

    query.value = "";
    conv.loading = true;

    const userMsg: ChatMessage = {
      id: `m-${Date.now()}`,
      role: "user",
      content: q,
    };
    conv.messages.push(userMsg);

    const assistantMsg: ChatMessage = {
      id: `m-${Date.now()}-ai`,
      role: "assistant",
      content: "",
      streaming: true,
    };
    conv.messages.push(assistantMsg);

    if (conv.title === "New Chat" || conv.title.startsWith("Conv-")) {
      conv.title = q.slice(0, 40) + (q.length > 40 ? "..." : "");
    }

    const controller = new AbortController();
    abortController.value = controller;

    try {
      const resp = await fetch("/api/v1/research/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          user_id: userId.value,
          thread_id: conv.threadId,
          tenant_id: tenantId.value,
        }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }

      const reader = resp.body?.getReader();
      if (!reader) throw new Error("No reader");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event: StreamEvent = JSON.parse(line.slice(6));
            if (event.type === "final" && event.final) {
              assistantMsg.content = event.final;
            }
          } catch {
            // skip malformed events
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        assistantMsg.content += "\n\n[stopped by user]";
      } else {
        const msg = err instanceof Error ? err.message : String(err);
        errorMessage.value = msg;
        assistantMsg.content = `Error: ${msg}`;
      }
    } finally {
      assistantMsg.streaming = false;
      conv.loading = false;
      abortController.value = null;
    }

    await nextTick();
  }

  function usePrompt(prompt: string) {
    query.value = prompt;
    runResearch();
  }

  return {
    userId,
    tenantId,
    conversations,
    activeIndex,
    activeConv,
    query,
    errorMessage,
    abortController,
    newConversation,
    selectConversation,
    deleteConversation,
    stopResearch,
    runResearch,
    usePrompt,
  };
}

export function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function markdownToHtml(markdown: string): string {
  const codeBlocks: string[] = [];
  let text = markdown.replace(
    /```([\s\S]*?)```/g,
    (_, block) => {
      const index = codeBlocks.length;
      codeBlocks.push(
        `<pre><code>${escapeHtml(String(block).trim())}</code></pre>`
      );
      return `@@CODE_BLOCK_${index}@@`;
    }
  );
  const lines = text.split("\n");
  const out: string[] = [];
  let inList = false;

  const closeList = () => {
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
  };

  for (const line of lines) {
    if (/^@@CODE_BLOCK_\d+@@$/.test(line)) {
      closeList();
      const idx = parseInt(line.match(/\d+/)![0]);
      out.push(codeBlocks[idx]);
      continue;
    }

    if (/^#{1,6}\s/.test(line)) {
      closeList();
      const level = line.match(/^(#+)/)![0].length;
      const content = line.replace(/^#+\s*/, "");
      out.push(`<h${level}>${escapeHtml(content)}</h${level}>`);
      continue;
    }

    if (/^[-*]\s/.test(line)) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${escapeHtml(line.replace(/^[-*]\s/, ""))}</li>`);
      continue;
    }

    if (/^\d+\.\s/.test(line)) {
      if (!inList) {
        out.push("<ol>");
        inList = true;
      }
      out.push(`<li>${escapeHtml(line.replace(/^\d+\.\s/, ""))}</li>`);
      continue;
    }

    closeList();
    if (line.trim()) {
      out.push(`<p>${escapeHtml(line)}</p>`);
    }
  }
  closeList();
  return out.join("\n");
}
