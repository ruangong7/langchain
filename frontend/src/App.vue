<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import AuthModal from './components/AuthModal.vue'
import ChatPanel from './components/ChatPanel.vue'
import MedicationModal from './components/MedicationModal.vue'
import ProfileModal from './components/ProfileModal.vue'
import { api } from './lib/api'
import { formatRelativeSessionTime, summarizeEffectiveContext, summarizeMedications, summarizeProfile } from './lib/profile'
import { useUserStore } from './stores/user'

const userStore = useUserStore()
const authOpen = ref(false)
const profileOpen = ref(false)
const medicationOpen = ref(false)
const statusText = ref(userStore.isLoggedIn ? '已登录' : '请先登录')
const chatPanelRef = ref(null)
const systemStatus = ref(null)
const loadingSystemStatus = ref(false)
const chatContext = ref(null)
const loadingChatContext = ref(false)
const TOOL_POLICY_STORAGE_KEY = 'med_app_tool_policy'
const toolPolicy = ref(readToolPolicy())
const editingSessionId = ref('')
const editingSessionTitle = ref('')

const profileSummary = computed(() => summarizeProfile(userStore.profile))
const medicationSummary = computed(() => summarizeMedications(userStore.profile?.medications || []))
const effectiveContextSummary = computed(() => summarizeEffectiveContext(chatContext.value?.effective_context))
const sessionItems = computed(() => userStore.sessions || [])
const systemRows = computed(() => {
  const status = systemStatus.value
  if (!status) return []
  const dbCaps = status.database?.capabilities || {}
  return [
    {
      label: '主检索',
      value: status.retrieval?.primary_backend || 'none',
      ok: status.retrieval?.primary_backend !== 'none',
    },
    {
      label: '工具调用',
      value: status.tool_calls_enabled ? `默认开启 (${(status.available_tools || []).length})` : '默认关闭',
      ok: Boolean(status.tool_calls_available),
    },
    {
      label: '个体化',
      value: dbCaps.user_health_profile_table || dbCaps.user_medications_table ? '健康档案可用' : '健康档案未就绪',
      ok: Boolean(dbCaps.user_health_profile_table || dbCaps.user_medications_table),
    },
    {
      label: '会话记忆',
      value: status.memory?.available ? 'Redis 已连接' : '记忆服务不可用',
      ok: Boolean(status.memory?.available),
    },
  ]
})

const toolPolicyOptions = computed(() => {
  const status = systemStatus.value
  const overrideEnabled = Boolean(status?.tool_calls_runtime_override_enabled)
  return [
    { value: 'default', label: '默认' },
    { value: 'force_on', label: '实验开启', disabled: !overrideEnabled || !status?.tool_calls_available },
    { value: 'force_off', label: '强制关闭', disabled: !overrideEnabled },
  ]
})

function readToolPolicy() {
  try {
    const value = localStorage.getItem(TOOL_POLICY_STORAGE_KEY) || 'default'
    return ['default', 'force_on', 'force_off'].includes(value) ? value : 'default'
  } catch {
    return 'default'
  }
}

function persistToolPolicy(value) {
  const normalized = ['default', 'force_on', 'force_off'].includes(value) ? value : 'default'
  toolPolicy.value = normalized
  try {
    localStorage.setItem(TOOL_POLICY_STORAGE_KEY, normalized)
  } catch {
    return
  }
}

function openAuth() {
  authOpen.value = true
}

function closeAuth() {
  authOpen.value = false
}

function requireLogin() {
  if (userStore.isLoggedIn) return true
  authOpen.value = true
  statusText.value = '请先登录'
  return false
}

function openProfile() {
  if (!requireLogin()) return
  profileOpen.value = true
  userStore.loadProfile().catch(() => null)
}

function openMedication() {
  if (!requireLogin()) return
  medicationOpen.value = true
  userStore.loadProfile().catch(() => null)
}

function closeProfile() {
  profileOpen.value = false
}

function closeMedication() {
  medicationOpen.value = false
}

function onAuthSuccess() {
  statusText.value = '登录成功'
}

function onProfileSaved() {
  statusText.value = '健康档案已保存'
  loadChatContext().catch(() => null)
}

function onMedicationSaved() {
  statusText.value = '当前用药已保存'
  loadChatContext().catch(() => null)
}

function onError(message) {
  statusText.value = message
}

function handleLogout() {
  userStore.logout()
  statusText.value = '已退出登录'
}

function applyQuickQuestion(text) {
  chatPanelRef.value?.setDraft(text)
  statusText.value = '已填入问题'
}

async function createConversation() {
  try {
    await userStore.createSession()
    statusText.value = '已新建会话'
  } catch (error) {
    statusText.value = error?.response?.data?.detail || error?.message || '新建会话失败'
  }
}

function selectConversation(sessionId) {
  if (editingSessionId.value) {
    cancelRenameSession()
  }
  userStore.switchSession(sessionId)
  statusText.value = '已切换会话'
}

async function removeConversation(sessionId) {
  try {
    await userStore.deleteSession(sessionId)
    statusText.value = '会话已删除'
  } catch (error) {
    statusText.value = error?.response?.data?.detail || error?.message || '删除会话失败'
  }
}

function startRenameSession(session) {
  editingSessionId.value = session.id
  editingSessionTitle.value = session.title
}

function cancelRenameSession() {
  editingSessionId.value = ''
  editingSessionTitle.value = ''
}

async function submitRenameSession(sessionId) {
  const text = String(editingSessionTitle.value || '').trim()
  if (!text) {
    statusText.value = '会话标题不能为空'
    return
  }
  try {
    await userStore.renameSession(sessionId, text)
    statusText.value = '会话标题已更新'
    cancelRenameSession()
  } catch (error) {
    statusText.value = error?.response?.data?.detail || error?.message || '更新标题失败'
  }
}

async function loadSystemStatus() {
  loadingSystemStatus.value = true
  try {
    const { data } = await api.get('/system/runtime-status')
    systemStatus.value = data
    if (!data.tool_calls_runtime_override_enabled && toolPolicy.value !== 'default') {
      persistToolPolicy('default')
    }
  } catch {
    systemStatus.value = null
  } finally {
    loadingSystemStatus.value = false
  }
}

async function loadChatContext() {
  if (!userStore.memoryId) {
    chatContext.value = null
    return
  }
  loadingChatContext.value = true
  try {
    const { data } = await api.get('/chat-context', {
      params: {
        memory_id: userStore.memoryId,
      },
      headers: userStore.token ? { Authorization: `Bearer ${userStore.token}` } : {},
    })
    chatContext.value = data
  } catch {
    chatContext.value = null
  } finally {
    loadingChatContext.value = false
  }
}

function handleConversationUpdated() {
  loadChatContext().catch(() => null)
}

watch(
  () => [userStore.memoryId, userStore.token],
  () => {
    loadChatContext().catch(() => null)
  },
  { immediate: false }
)

onMounted(() => {
  loadSystemStatus().catch(() => null)
  userStore.loadSessions().catch(() => null)
  loadChatContext().catch(() => null)
  if (userStore.isLoggedIn) {
    userStore.loadProfile().catch(() => null)
  } else {
    statusText.value = '游客模式'
  }
})
</script>

<template>
  <main class="shell">
    <aside class="panel">
      <section class="brand">
        <div class="brand-row">
          <div class="mark">药</div>
          <div>
            <h1>健康用药助手</h1>
          </div>
        </div>
        <p>支持药品问答、相互作用查询，并结合用户自己的健康档案和当前用药做更个体化的提示。</p>
      </section>

      <section class="side-section">
        <div class="section-title">账户状态</div>
        <div class="account-card">
          <div v-if="!userStore.isLoggedIn">
            <div class="identity">
              <div class="avatar">?</div>
              <div>
                <strong>未登录</strong>
                <span>游客也可直接提问，登录后可保存健康档案并进行个体化问答</span>
              </div>
            </div>
            <div class="button-row" style="margin-top: 10px;">
              <button class="button primary" type="button" @click="openAuth">登录 / 注册</button>
            </div>
          </div>

          <div v-else>
            <div class="identity">
              <div class="avatar">
                {{ (userStore.currentUser?.displayName || userStore.currentUser?.username || 'U').slice(0, 1).toUpperCase() }}
              </div>
              <div>
                <strong>{{ userStore.currentUser?.displayName || userStore.currentUser?.username }}</strong>
                <span>个体化用药问答已启用</span>
              </div>
            </div>
            <div class="button-row">
              <button class="button" type="button" @click="openProfile">健康档案</button>
              <button class="button" type="button" @click="openMedication">当前用药</button>
              <button class="button ghost" type="button" @click="handleLogout">退出</button>
            </div>
          </div>
        </div>
      </section>

      <section class="side-section">
        <div class="section-title">档案摘要</div>
        <div class="summary-card">
          <div class="summary-box" :class="{ empty: !userStore.profile }">{{ profileSummary }}</div>
        </div>
      </section>

      <section class="side-section">
        <div class="section-title">当前用药</div>
        <div class="summary-card">
          <div class="summary-box" :class="{ empty: !userStore.profile?.medications?.length }">{{ medicationSummary }}</div>
        </div>
      </section>

      <section class="side-section">
        <div class="section-title">问答背景</div>
        <div class="summary-card">
          <div class="summary-box" :class="{ empty: !chatContext?.effective_context_text }">
            {{ loadingChatContext ? '正在整理当前会话背景...' : effectiveContextSummary }}
          </div>
          <div v-if="chatContext?.effective_context_text" class="runtime-tools">
            {{ chatContext.profile_available ? '含用户档案' : '无用户档案' }} ·
            {{ chatContext.memory_available ? '含会话记忆' : '无会话记忆' }}
          </div>
        </div>
      </section>

      <section class="side-section">
        <div class="section-title">会话列表</div>
        <div class="summary-card">
          <div class="button-row">
            <button class="button" type="button" @click="createConversation">新建会话</button>
          </div>
          <div class="session-list">
            <div
              v-for="session in sessionItems"
              :key="session.id"
              class="session-item"
              :class="{ active: userStore.memoryId === session.id }"
            >
              <button v-if="editingSessionId !== session.id" class="session-item-main" type="button" @click="selectConversation(session.id)">
                <div class="session-item-head">
                  <strong>{{ session.title }}</strong>
                  <small>{{ formatRelativeSessionTime(session.updatedAt) }}</small>
                </div>
                <span>{{ session.preview || '还没有消息' }}</span>
              </button>
              <div v-else class="session-rename">
                <input
                  v-model="editingSessionTitle"
                  class="session-rename-input"
                  type="text"
                  maxlength="32"
                  @keydown.enter.prevent="submitRenameSession(session.id)"
                  @keydown.esc.prevent="cancelRenameSession()"
                />
                <div class="button-row">
                  <button class="button" type="button" @click="submitRenameSession(session.id)">保存</button>
                  <button class="button ghost" type="button" @click="cancelRenameSession()">取消</button>
                </div>
              </div>
              <div class="session-actions">
                <button class="button ghost session-delete" type="button" @click="startRenameSession(session)">改名</button>
                <button class="button ghost session-delete" type="button" @click="removeConversation(session.id)">删除</button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="side-section">
        <div class="section-title">快捷提问</div>
        <div class="chips">
          <button class="chip" type="button" @click="applyQuickQuestion('我有高血压，正在吃缬沙坦，布洛芬能不能吃？')">我有高血压，正在吃缬沙坦，布洛芬能不能吃？</button>
          <button class="chip" type="button" @click="applyQuickQuestion('我对青霉素过敏，现在咳嗽发烧，阿莫西林适合吗？')">我对青霉素过敏，现在咳嗽发烧，阿莫西林适合吗？</button>
          <button class="chip" type="button" @click="applyQuickQuestion('我长期吃二甲双胍，现在又开了布洛芬，需要注意什么？')">我长期吃二甲双胍，现在又开了布洛芬，需要注意什么？</button>
        </div>
      </section>

      <section class="side-section">
        <div class="section-title">系统状态</div>
        <div class="summary-card">
          <div v-if="systemRows.length" class="runtime-card">
            <div v-for="row in systemRows" :key="row.label" class="runtime-row">
              <div class="runtime-label">
                <span class="runtime-dot" :class="{ ok: row.ok, fail: !row.ok }"></span>
                <span>{{ row.label }}</span>
              </div>
              <span class="runtime-value">{{ row.value }}</span>
            </div>
            <div v-if="systemStatus?.available_tools?.length" class="runtime-tools">
              {{ systemStatus.available_tools.join('、') }}
            </div>
            <div class="runtime-policy">
              <div class="tabs">
                <button
                  v-for="option in toolPolicyOptions"
                  :key="option.value"
                  class="tab"
                  :class="{ active: toolPolicy === option.value }"
                  type="button"
                  :disabled="option.disabled"
                  @click="persistToolPolicy(option.value)"
                >
                  {{ option.label }}
                </button>
              </div>
            </div>
          </div>
          <div v-else class="summary-box empty">
            {{ loadingSystemStatus ? '正在读取运行时状态...' : '暂时无法读取系统状态。' }}
          </div>
          <div class="button-row">
            <button class="button" type="button" :disabled="loadingSystemStatus" @click="loadSystemStatus">
              {{ loadingSystemStatus ? '刷新中...' : '刷新状态' }}
            </button>
          </div>
        </div>
      </section>

      <section class="side-section">
        <div class="status">
          <span class="dot"></span>
          <span>{{ statusText }}</span>
        </div>
      </section>
    </aside>

    <ChatPanel
      ref="chatPanelRef"
      :tool-policy="toolPolicy"
      @status="statusText = $event"
      @conversation-updated="handleConversationUpdated"
    />

    <AuthModal :open="authOpen" @close="closeAuth" @success="onAuthSuccess" />
    <ProfileModal :open="profileOpen" @close="closeProfile" @saved="onProfileSaved" @error="onError" />
    <MedicationModal :open="medicationOpen" @close="closeMedication" @saved="onMedicationSaved" @error="onError" />
  </main>
</template>
