<script setup lang="ts">
import { computed } from "vue";
import type { Conversation } from "../composables/useChat";

const props = defineProps<{
  conversations: Conversation[];
  activeIndex: number;
}>();

const emit = defineEmits<{
  select: [index: number];
  delete: [index: number];
  new: [];
}>();

const displayList = computed(() =>
  props.conversations.map((c) => ({
    ...c,
    title: c.title || "New Chat",
  }))
);
</script>

<template>
  <aside class="sidebar">
    <button class="new-chat-btn" @click="emit('new')">+ New Chat</button>
    <div class="conv-list">
      <div
        v-for="(conv, idx) in displayList"
        :key="conv.id"
        class="conv-item"
        :class="{ active: idx === activeIndex }"
        @click="emit('select', idx)"
      >
        <span class="conv-title">{{ conv.title }}</span>
        <button class="conv-del" @click.stop="emit('delete', idx)">x</button>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 220px;
  background: #1e1e2e;
  color: #cdd6f4;
  display: flex;
  flex-direction: column;
  border-right: 1px solid #313244;
}
.new-chat-btn {
  margin: 12px;
  padding: 8px;
  background: #89b4fa;
  color: #1e1e2e;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
}
.conv-list { flex: 1; overflow-y: auto; }
.conv-item {
  padding: 10px 12px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #313244;
}
.conv-item.active { background: #313244; }
.conv-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
}
.conv-del {
  background: none;
  border: none;
  color: #f38ba8;
  cursor: pointer;
  font-size: 14px;
  margin-left: 6px;
}
</style>
