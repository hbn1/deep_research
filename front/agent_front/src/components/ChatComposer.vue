<script setup lang="ts">
import { ref } from "vue";

const props = defineProps<{
  loading: boolean;
  disabled: boolean;
}>();

const emit = defineEmits<{
  send: [text: string];
  stop: [];
}>();

const text = ref("");

function handleSend() {
  const trimmed = text.value.trim();
  if (!trimmed) return;
  emit("send", trimmed);
  text.value = "";
}
</script>

<template>
  <div class="composer">
    <textarea
      v-model="text"
      class="composer-input"
      :disabled="disabled"
      placeholder="Ask anything (Shift+Enter for new line)"
      @keydown.enter.exact.prevent="handleSend"
    />
    <button
      v-if="loading"
      class="stop-btn"
      @click="emit('stop')"
      title="Stop"
    >
      <span class="stop-icon"></span>
    </button>
    <button
      v-else
      class="send-btn"
      :disabled="!text.trim()"
      @click="handleSend"
    >
      Send
    </button>
  </div>
</template>

<style scoped>
.composer {
  display: flex;
  padding: 12px 16px;
  border-top: 1px solid #313244;
  background: #1e1e2e;
  gap: 8px;
}
.composer-input {
  flex: 1;
  resize: none;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #45475a;
  background: #313244;
  color: #cdd6f4;
  font-family: inherit;
  font-size: 14px;
  min-height: 42px;
}
.composer-input:disabled { opacity: 0.5; }
.send-btn, .stop-btn {
  padding: 8px 18px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
}
.send-btn { background: #89b4fa; color: #1e1e2e; }
.send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.stop-btn { background: #f38ba8; color: #1e1e2e; }
.stop-icon::before { content: "\25A0"; font-size: 12px; }
</style>
