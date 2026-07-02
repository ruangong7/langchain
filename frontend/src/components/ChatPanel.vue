<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { api, authHeaders } from '../lib/api'
import { useUserStore } from '../stores/user'

const props = defineProps({
  toolPolicy: {
    type: String,
    default: 'default',
  },
})

const emit = defineEmits(['status', 'conversation-updated'])

const userStore = useUserStore()
const question = ref('')
const sending = ref(false)
const messages = ref([])
const lastAssistantText = ref('')
const chatRef = ref(null)
const loadingHistory = ref(false)

const sessionLabel = computed(() => (userStore.isLoggedIn ? '已登录会话' : '游客会话'))
const idleStatusText = computed(() => (userStore.isLoggedIn ? '已登录' : '游客模式'))

function formatMeta(meta) {
  if (!meta) return ''
  const lines = []
  if (meta.route || meta.intent) {
    lines.push(`路由: ${meta.route || '-'} / 意图: ${meta.intent || '-'}`)
  }
  if (meta.retrieval?.backend) {
    lines.push(`检索: ${meta.retrieval.backend}${meta.retrieval.method ? ` (${meta.retrieval.method})` : ''}`)
  }
  if (Array.isArray(meta.allowed_tool_names) && meta.allowed_tool_names.length) {
    lines.push(`开放工具: ${meta.allowed_tool_names.join('、')}`)
  }
  if (meta.tooling?.used_tools && Array.isArray(meta.tooling.tool_calls) && meta.tooling.tool_calls.length) {
    lines.push(
      `实际调用: ${meta.tooling.tool_calls
        .map((item) => `${item.name}${item.ok ? '' : '（失败）'}`)
        .join('、')}`
    )
  }
  if (meta.tooling?.tool_rounds) {
    lines.push(`工具轮次: ${meta.tooling.tool_rounds}`)
  }
  return lines.join('\n')
}

function buildMetaView(meta) {
  if (!meta) return null
  const badges = []
  const rows = []
  if (meta.route) badges.push(`路由 ${meta.route}`)
  if (meta.intent) badges.push(`意图 ${meta.intent}`)
  if (meta.tool_policy) badges.push(`工具策略 ${meta.tool_policy}`)
  if (meta.retrieval?.backend) {
    rows.push({
      label: '检索',
      value: `${meta.retrieval.backend}${meta.retrieval.method ? ` (${meta.retrieval.method})` : ''}`,
    })
  }
  if (Array.isArray(meta.allowed_tool_names) && meta.allowed_tool_names.length) {
    rows.push({
      label: '开放工具',
      value: meta.allowed_tool_names.join('、'),
    })
  }
  if (meta.tooling?.tool_rounds) {
    rows.push({
      label: '工具轮次',
      value: `${meta.tooling.tool_rounds}${meta.tooling.tool_loop_truncated ? '（已截断）' : ''}`,
    })
  }
  if (meta.tooling?.used_tools && Array.isArray(meta.tooling.tool_calls) && meta.tooling.tool_calls.length) {
    rows.push({
      label: '实际调用',
      value: meta.tooling.tool_calls
        .map((item) => `${item.name}${item.ok ? '' : '（失败）'}`)
        .join('、'),
    })
  }
  const toolCalls = Array.isArray(meta.tooling?.tool_calls) ? meta.tooling.tool_calls : []
  const retrieval = meta.retrieval || {}
  return {
    badges,
    rows,
    toolCalls,
    retrievalSummary: {
      backend: retrieval.backend || '',
      method: retrieval.method || '',
      queries: Array.isArray(retrieval.queries) ? retrieval.queries : [],
      usedChunks: Array.isArray(retrieval.used_chunks) ? retrieval.used_chunks : [],
      topChunks: Array.isArray(retrieval.top_chunks) ? retrieval.top_chunks : [],
    },
  }
}

function formatScore(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '--'
}

function scrollToBottom() {
  nextTick(() => {
    if (chatRef.value) {
      chatRef.value.scrollTop = chatRef.value.scrollHeight
    }
  })
}

function mapHistoryMessage(role, text, meta = null) {
  return {
    id: `${Date.now()}_${Math.random()}`,
    role,
    text,
    error: false,
    meta,
    metaView: buildMetaView(meta),
    metaText: formatMeta(meta),
  }
}

function addMessage(role, text, error = false, meta = null) {
  messages.value.push({
    id: `${Date.now()}_${Math.random()}`,
    role,
    text,
    error,
    meta,
    metaView: buildMetaView(meta),
    metaText: formatMeta(meta),
  })
  scrollToBottom()
  return messages.value[messages.value.length - 1]
}

async function loadHistory() {
  loadingHistory.value = true
  emit('status', '正在读取会话')
  try {
    const response = await api.get('/chat-history', {
      params: {
        memory_id: userStore.memoryId,
        turns: 20,
      },
      headers: authHeaders(userStore.token),
    })
    messages.value = (response.data.messages || []).map((item) => mapHistoryMessage(item.role, item.content, item.meta || null))
    lastAssistantText.value = [...messages.value].reverse().find((item) => item.role === 'assistant')?.text || ''
    emit('status', userStore.isLoggedIn ? '已加载历史会话' : '已加载游客会话')
    emit('conversation-updated')
    scrollToBottom()
  } catch (error) {
    messages.value = []
    emit('status', error?.response?.data?.detail || '读取会话失败')
  } finally {
    loadingHistory.value = false
  }
}

async function sendPlain() {
  if (!question.value.trim() || sending.value) return
  const q = question.value.trim()
  userStore.touchActiveSession(q)
  addMessage('user', q)
  const assistant = addMessage('assistant', '...')
  sending.value = true
  emit('status', '正在请求普通回复')
  try {
    const response = await api.get('/chat', {
      params: {
        memory_id: userStore.memoryId,
        message: q,
        include_meta: true,
        tool_policy: props.toolPolicy,
      },
      headers: authHeaders(userStore.token),
    })
    assistant.text = response.data.answer
    assistant.meta = response.data.meta
    assistant.metaView = buildMetaView(response.data.meta)
    assistant.metaText = formatMeta(response.data.meta)
    lastAssistantText.value = response.data.answer
    emit('conversation-updated')
  } catch (error) {
    assistant.text = error?.response?.data?.detail || error?.message || '普通回复失败'
    assistant.error = true
  } finally {
    question.value = ''
    sending.value = false
    emit('status', idleStatusText.value)
    scrollToBottom()
  }
}

async function sendStream() {
  if (!question.value.trim() || sending.value) return
  const q = question.value.trim()
  userStore.touchActiveSession(q)
  addMessage('user', q)
  const assistant = addMessage('assistant', '')
  sending.value = true
  emit('status', '正在接收流式回复')

  try {
    const url = new URL(`${api.defaults.baseURL}/chat-stream`)
    url.searchParams.set('memory_id', userStore.memoryId)
    url.searchParams.set('message', q)
    url.searchParams.set('tool_policy', props.toolPolicy)

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
    let metaText = ''

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
        if (payload.meta) {
          assistant.meta = payload.meta
          assistant.metaView = buildMetaView(payload.meta)
          assistant.metaText = formatMeta(payload.meta)
          continue
        }
        if (payload.meta_update) {
          assistant.meta = payload.meta_update
          assistant.metaView = buildMetaView(payload.meta_update)
          assistant.metaText = formatMeta(payload.meta_update)
          continue
        }
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
    emit('conversation-updated')
  } catch (error) {
    assistant.text = error?.message || '流式回复失败'
    assistant.error = true
  } finally {
    question.value = ''
    sending.value = false
    emit('status', idleStatusText.value)
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

async function clearConversation() {
  if (sending.value) return
  emit('status', '正在清空会话')
  try {
    await api.delete('/chat-history', {
      params: {
        memory_id: userStore.memoryId,
      },
      headers: authHeaders(userStore.token),
    })
    messages.value = []
    lastAssistantText.value = ''
    emit('status', '当前会话已清空')
    emit('conversation-updated')
  } catch (error) {
    emit('status', error?.response?.data?.detail || '清空会话失败')
  }
}

function clearMessages() {
  messages.value = []
  lastAssistantText.value = ''
  emit('status', userStore.isLoggedIn ? '已登录' : '请先登录')
}

function setDraft(text) {
  question.value = String(text || '')
}

watch(
  () => userStore.memoryId,
  () => {
    messages.value = []
    lastAssistantText.value = ''
    loadHistory()
  },
  { immediate: true }
)

defineExpose({
  copyLastReply,
  clearMessages,
  clearConversation,
  setDraft,
})
</script>

<template>
  <section class="workspace">
    <header class="topbar">
      <div>
        <h2>智能问答</h2>
        <small>{{ sessionLabel }}，{{ loadingHistory ? '正在同步历史消息' : '支持继续之前的对话' }}</small>
      </div>
      <div class="button-row topbar-actions">
        <button class="button" type="button" :disabled="sending || loadingHistory" @click="loadHistory">刷新会话</button>
        <button class="button danger" type="button" :disabled="sending || loadingHistory" @click="clearConversation">清空会话</button>
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
        <div v-if="message.metaView && (message.metaView.badges.length || message.metaView.rows.length)" class="message-status">
          <div v-if="message.metaView.badges.length" class="message-status-badges">
            <span v-for="badge in message.metaView.badges" :key="badge" class="status-chip">{{ badge }}</span>
          </div>
          <div v-if="message.metaView.rows.length" class="message-status-rows">
            <div v-for="row in message.metaView.rows" :key="row.label" class="message-status-row">
              <span class="status-key">{{ row.label }}</span>
              <span class="status-value">{{ row.value }}</span>
            </div>
          </div>
          <div v-if="message.metaView.toolCalls.length" class="tool-call-list">
            <div
              v-for="(call, index) in message.metaView.toolCalls"
              :key="`${call.name}_${index}`"
              class="tool-call-item"
              :class="{ fail: call.ok === false }"
            >
              <div class="tool-call-head">
                <strong>{{ call.name }}</strong>
                <span>{{ call.ok === false ? '失败' : '完成' }}</span>
              </div>
              <div class="tool-call-message">{{ call.message || call.reason || '已执行' }}</div>
            </div>
          </div>
          <div
            v-if="message.metaView.retrievalSummary.usedChunks.length || message.metaView.retrievalSummary.topChunks.length"
            class="trace-panel"
          >
            <div class="trace-row">
              <span class="trace-badge">后端 {{ message.metaView.retrievalSummary.backend || 'legacy_rag' }}</span>
              <span class="trace-badge">方法 {{ message.metaView.retrievalSummary.method || 'hybrid' }}</span>
              <span class="trace-badge">引用 {{ message.metaView.retrievalSummary.usedChunks.length }} 条</span>
              <span class="trace-badge">Top {{ message.metaView.retrievalSummary.topChunks.length }} 条</span>
            </div>
            <div v-if="message.metaView.retrievalSummary.queries.length" class="trace-row">
              检索查询: {{ message.metaView.retrievalSummary.queries.join(' | ') }}
            </div>

            <div v-if="message.metaView.retrievalSummary.usedChunks.length" class="trace-group">
              <div class="trace-title">本次回答实际引用的 chunk</div>
              <div class="trace-list">
                <div
                  v-for="(chunk, index) in message.metaView.retrievalSummary.usedChunks"
                  :key="`used_${message.id}_${chunk.chunk_id || index}`"
                  class="trace-item"
                >
                  <div class="trace-item-head">
                    <strong>{{ chunk.source || '未知来源' }}</strong>
                    <span class="trace-badge">#{{ chunk.chunk_id || 'unknown' }}</span>
                    <span class="trace-badge">排序 {{ chunk.rank ?? '--' }}</span>
                    <span class="trace-score">分数 {{ formatScore(chunk.score) }}</span>
                    <span v-if="chunk.retrieval_sources?.length" class="trace-badge">
                      {{ chunk.retrieval_sources.join(' + ') }}
                    </span>
                  </div>
                  <div v-if="chunk.matched_queries?.length" class="trace-row">
                    命中改写: {{ chunk.matched_queries.join(' / ') }}
                  </div>
                  <div class="trace-preview">{{ chunk.text_preview || '' }}</div>
                </div>
              </div>
            </div>

            <div v-if="message.metaView.retrievalSummary.topChunks.length" class="trace-group">
              <div class="trace-title">分数最高的 chunk</div>
              <div class="trace-list">
                <div
                  v-for="(chunk, index) in message.metaView.retrievalSummary.topChunks"
                  :key="`top_${message.id}_${chunk.chunk_id || index}`"
                  class="trace-item"
                >
                  <div class="trace-item-head">
                    <strong>{{ chunk.source || '未知来源' }}</strong>
                    <span class="trace-badge">#{{ chunk.chunk_id || 'unknown' }}</span>
                    <span class="trace-badge">排序 {{ chunk.rank ?? '--' }}</span>
                    <span class="trace-score">分数 {{ formatScore(chunk.score) }}</span>
                    <span v-if="chunk.retrieval_sources?.length" class="trace-badge">
                      {{ chunk.retrieval_sources.join(' + ') }}
                    </span>
                  </div>
                  <div v-if="chunk.matched_queries?.length" class="trace-row">
                    命中改写: {{ chunk.matched_queries.join(' / ') }}
                  </div>
                  <div class="trace-preview">{{ chunk.text_preview || '' }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </article>
    </section>

    <section class="composer">
      <form @submit.prevent="sendStream">
        <div class="composer-row">
          <textarea
            v-model="question"
            rows="3"
            :disabled="sending"
            placeholder="例如：我有胃病，最近在吃阿司匹林，还能吃布洛芬吗？"
          ></textarea>
          <button class="button" type="button" :disabled="sending" @click="sendPlain">普通发送</button>
          <button class="button primary" type="submit" :disabled="sending">流式发送</button>
        </div>
        <div class="composer-meta">
          <span>{{ userStore.isLoggedIn ? '回车换行，Ctrl/Command + Enter 可直接流式发送' : '游客也可提问，登录后会自动结合健康档案' }}</span>
          <span class="warn">高风险或复杂情况仍应咨询医生或药师</span>
        </div>
      </form>
    </section>
  </section>
</template>
