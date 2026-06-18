<script setup>
import { computed, onMounted, ref } from 'vue'
import AuthModal from './components/AuthModal.vue'
import ChatPanel from './components/ChatPanel.vue'
import MedicationModal from './components/MedicationModal.vue'
import ProfileModal from './components/ProfileModal.vue'
import { summarizeMedications, summarizeProfile } from './lib/profile'
import { useUserStore } from './stores/user'

const userStore = useUserStore()
const authOpen = ref(false)
const profileOpen = ref(false)
const medicationOpen = ref(false)
const statusText = ref(userStore.isLoggedIn ? '已登录' : '请先登录')

const profileSummary = computed(() => summarizeProfile(userStore.profile))
const medicationSummary = computed(() => summarizeMedications(userStore.profile?.medications || []))

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
}

function onMedicationSaved() {
  statusText.value = '当前用药已保存'
}

function onError(message) {
  statusText.value = message
}

function handleLogout() {
  userStore.logout()
  statusText.value = '已退出登录'
}

onMounted(() => {
  if (userStore.isLoggedIn) {
    userStore.loadProfile().catch(() => null)
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
                <span>登录后可保存健康档案并进行个体化问答</span>
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
        <div class="section-title">快捷提问</div>
        <div class="chips">
          <button class="chip" type="button">我有高血压，正在吃缬沙坦，布洛芬能不能吃？</button>
          <button class="chip" type="button">我对青霉素过敏，现在咳嗽发烧，阿莫西林适合吗？</button>
          <button class="chip" type="button">我长期吃二甲双胍，现在又开了布洛芬，需要注意什么？</button>
        </div>
      </section>

      <section class="side-section">
        <div class="status">
          <span class="dot"></span>
          <span>{{ statusText }}</span>
        </div>
      </section>
    </aside>

    <ChatPanel @status="statusText = $event" />

    <AuthModal :open="authOpen" @close="closeAuth" @success="onAuthSuccess" />
    <ProfileModal :open="profileOpen" @close="closeProfile" @saved="onProfileSaved" @error="onError" />
    <MedicationModal :open="medicationOpen" @close="closeMedication" @saved="onMedicationSaved" @error="onError" />
  </main>
</template>
