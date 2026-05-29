<script setup lang="ts">
import { useChat } from "./composables/useChat";
import ChatSidebar from "./components/ChatSidebar.vue";
import ChatComposer from "./components/ChatComposer.vue";
import MessageBubble from "./components/MessageBubble.vue";

const {
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
} = useChat();

function handleSend(text: string) {
  query.value = text;
  runResearch();
}

const starterPrompts = [
  {
    title: "Market Research",
    prompt:
      'Research the "Enterprise Knowledge Agent Platform" market, including market size, key competitors, and pricing models.',
  },
  {
    title: "Solution Comparison",
    prompt:
      'Compare "LLM Direct Answer", "RAG + Agent", and "Multi-Agent Collaboration" approaches for building a research assistant.',
  },
  {
    title: "Knowledge Q&A",
    prompt:
      "Explain how intent routing works in this project, and how simple vs complex queries follow different paths.",
  },
];
</script>

<template>
  <div class="app-shell">
    <ChatSidebar
      :conversations="conversations"
      :activeIndex="activeIndex"
      @select="selectConversation"
      @delete="deleteConversation"
      @new="newConversation"
    />

    <main class="chat-main">
      <header class="chat-header">
        <h2>DeepResearch</h2>
        <span>Evidence-Driven | Structured Output | Memory-Powered</span>
      </header>

      <div class="message-list">
        <section
          v-if="activeConv && activeConv.messages.length <= 1"
          class="onboarding"
        >
          <h3>Start by describing your goal, then let DeepResearch run</h3>
          <p>
            Recommended structure: Objective + Background + Expected Output.
            The system will route to Quick Answer or Deep Research automatically.
          </p>
          <div class="starter-grid">
            <button
              v-for="item in starterPrompts"
              :key="item.title"
              class="starter-btn"
              @click="usePrompt(item.prompt)"
            >
              <strong>{{ item.title }}</strong>
              <span>{{ item.prompt.slice(0, 80) }}...</span>
            </button>
          </div>
        </section>

        <template v-if="activeConv">
          <MessageBubble
            v-for="message in activeConv.messages"
            :key="message.id"
            :role="message.role"
            :content="message.content"
            :streaming="message.streaming"
          />
        </template>
      </div>

      <ChatComposer
        :loading="activeConv?.loading ?? false"
        :disabled="activeConv?.loading ?? false"
        @send="handleSend"
        @stop="stopResearch"
      />

      <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
    </main>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  height: 100vh;
  background: #11111b;
  color: #cdd6f4;
  font-family: "Inter", system-ui, sans-serif;
}
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.chat-header {
  padding: 14px 20px;
  border-bottom: 1px solid #313244;
  display: flex;
  align-items: baseline;
  gap: 16px;
  background: #1e1e2e;
}
.chat-header h2 { margin: 0; font-size: 18px; color: #89b4fa; }
.chat-header span { font-size: 12px; color: #6c7086; }
.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}
.onboarding {
  padding: 40px 20px;
  text-align: center;
  max-width: 640px;
  margin: 0 auto;
}
.onboarding h3 { font-size: 20px; margin-bottom: 8px; }
.onboarding p { color: #a6adc8; margin-bottom: 24px; }
.starter-grid {
  display: grid;
  gap: 10px;
}
.starter-btn {
  display: block;
  text-align: left;
  padding: 12px 16px;
  background: #313244;
  border: 1px solid #45475a;
  border-radius: 8px;
  cursor: pointer;
  color: #cdd6f4;
}
.starter-btn:hover { border-color: #89b4fa; }
.starter-btn strong { display: block; margin-bottom: 4px; font-size: 13px; }
.starter-btn span { font-size: 12px; color: #a6adc8; }
.error { color: #f38ba8; padding: 8px 16px; margin: 0; font-size: 13px; }
</style>
