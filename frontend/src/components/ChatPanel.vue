<script setup>
import { nextTick, ref } from 'vue'
import { api, authHeaders } from '../lib/api'
import { useUserStore } from '../stores/user'

const emit = defineEmits(['status'])

const userStore = useUserStore()
const question = ref('')
const sending = ref(false)
const messages = ref([])
const lastAssistantText = ref('')
const chatRef = ref(null)

function scrollToBottom() {
  nextTick(() => {
    if (chatRef.value) {
      chatRef.value.scrollTop = chatRef.value.scrollHeight
    }
  })
}

function addMessage(role, text, error = false) {
  messages.value.push({
    id: `${Date.now()}_${Math.random()}`,
    role,
    text,
    error,
  })
  scrollToBottom()
  return messages.value[messages.value.length - 1]
}

async function sendPlain() {
  if (!question.value.trim() || !userStore.isLoggedIn || sending.value) return
  const q = question.value.trim()
  addMessage('user', q)
  const assistant = addMessage('assistant', '...')
  sending.value = true
  emit('status', '正在请求普通回复')
  try {
    const response = await api.get('/chat', {
      params: {
        memory_id: userStore.memoryId,
        message: q,
      },
      headers: authHeaders(userStore.token),
      responseType: 'text',
    })
    assistant.text = response.data
    lastAssistantText.value = response.data
  } catch (error) {
    assistant.text = error?.response?.data?.detail || error?.message || '普通回复失败'
    assistant.error = true
  } finally {
    question.value = ''
    sending.value = false
    emit('status', '已登录')
    scrollToBottom()
  }
}

async function sendStream() {
  if (!question.value.trim() || !userStore.isLoggedIn || sending.value) return
  const q = question.value.trim()
  addMessage('user', q)
  const assistant = addMessage('assistant', '')
  sending.value = true
  emit('status', '正在接收流式回复')

  try {
    const url = new URL(`${api.defaults.baseURL}/chat-stream`)
    url.searchParams.set('memory_id', userStore.memoryId)
    url.searchParams.set('message', q)

    const response = await fetch(url.toString(), {
      headers: authHeaders(userStore.token),
    })
    if (!response.ok || !response.body) {
      throw new Error(await response.text() || '流式请求失败')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    let answer = ''

    function consumeEvents(text) {
      const normalized = text.replace(/\r\n/g, '\n')
      const chunks = normalized.split(/\n\n/)
      buffer = chunks.pop() || ''
      for (const chunk of chunks) {
        const lines = chunk.split('\n').filter((line) => line.startsWith('data:'))
        const data = lines.length ? lines.map((line) => line.slice(5).replace(/^ /, '')).join('\n') : chunk
        if (!data.trim()) continue
        let payload
        try {
          payload = JSON.parse(data)
        } catch {
          payload = { data }
        }
        if (payload.done) continue
        if (payload.error) throw new Error(payload.error)
        answer += payload.data || ''
        assistant.text = answer
      }
      scrollToBottom()
    }

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      consumeEvents(buffer)
    }
    buffer += decoder.decode()
    if (buffer) consumeEvents(`${buffer}\n\n`)
    lastAssistantText.value = answer
  } catch (error) {
    assistant.text = error?.message || '流式回复失败'
    assistant.error = true
  } finally {
    question.value = ''
    sending.value = false
    emit('status', '已登录')
    scrollToBottom()
  }
}

async function copyLastReply() {
  if (!lastAssistantText.value) {
    emit('status', '暂无可复制内容')
    return
  }
  try {
    await navigator.clipboard.writeText(lastAssistantText.value)
    emit('status', '已复制最后一条回复')
  } catch {
    emit('status', '复制失败')
  }
}

function clearMessages() {
  messages.value = []
  lastAssistantText.value = ''
  emit('status', userStore.isLoggedIn ? '已登录' : '请先登录')
}

defineExpose({
  copyLastReply,
  clearMessages,
})
</script>

<template>
  <section class="workspace">
    <header class="topbar">
      <div>
        <h2>智能问答</h2>
        <small>支持普通问答和流式回复，登录后会自动结合健康档案。</small>
      </div>
    </header>

    <section ref="chatRef" class="chat">
      <div v-if="messages.length === 0" class="empty">
        <h3>今天想确认哪条用药信息？</h3>
        <p>输入药名、症状、人群或剂量，系统会优先围绕药物知识和你的健康档案来回答。</p>
      </div>
      <article
        v-for="message in messages"
        :key="message.id"
        class="message"
        :class="[message.role, { error: message.error }]"
      >
        <div class="meta">{{ message.role === 'user' ? '你' : '助手' }}</div>
        <div class="bubble">{{ message.text }}</div>
      </article>
    </section>

    <section class="composer">
      <form @submit.prevent="sendStream">
        <div class="composer-row">
          <textarea
            v-model="question"
            rows="3"
            :disabled="!userStore.isLoggedIn || sending"
            placeholder="例如：我有胃病，最近在吃阿司匹林，还能吃布洛芬吗？"
          ></textarea>
          <button class="button" type="button" :disabled="!userStore.isLoggedIn || sending" @click="sendPlain">普通发送</button>
          <button class="button primary" type="submit" :disabled="!userStore.isLoggedIn || sending">流式发送</button>
        </div>
        <div class="composer-meta">
          <span>回车换行，Ctrl/Command + Enter 可直接流式发送</span>
          <span class="warn">高风险或复杂情况仍应咨询医生或药师</span>
        </div>
      </form>
    </section>
  </section>
</template>
