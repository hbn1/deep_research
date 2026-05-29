<script setup lang="ts">
import { markdownToHtml } from "../composables/useChat";

const props = defineProps<{
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}>();

function renderHtml(): string {
  return markdownToHtml(props.content);
}
</script>

<template>
  <div class="message-row" :class="`role-${role}`">
    <div class="avatar">{{ role === "user" ? "You" : "AI" }}</div>
    <div
      class="bubble markdown-body"
      :class="{ 'bubble-streaming': streaming }"
      v-html="renderHtml()"
    ></div>
  </div>
</template>

<style scoped>
.message-row {
  display: flex;
  gap: 10px;
  padding: 12px 20px;
  animation: fadeIn 0.2s ease;
}
.role-user { flex-direction: row-reverse; }
.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 700;
  flex-shrink: 0;
}
.role-user .avatar { background: #89b4fa; color: #1e1e2e; }
.role-assistant .avatar { background: #a6e3a1; color: #1e1e2e; }
.bubble {
  max-width: 80%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
}
.role-user .bubble { background: #89b4fa; color: #1e1e2e; }
.role-assistant .bubble { background: #313244; color: #cdd6f4; }
.bubble-streaming::after {
  content: "";
  display: inline-block;
  width: 6px;
  height: 14px;
  background: #89b4fa;
  animation: blink 0.6s infinite;
  margin-left: 2px;
  vertical-align: middle;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } }
@keyframes blink { 50% { opacity: 0; } }
</style>
