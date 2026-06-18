<script setup>
import { computed, reactive, ref } from 'vue'
import { useUserStore } from '../stores/user'

defineProps({
  open: Boolean,
})

const emit = defineEmits(['close', 'success'])

const userStore = useUserStore()
const authMode = ref('login')
const submitting = ref(false)
const errorText = ref('')
const form = reactive({
  username: '',
  password: '',
})

const title = computed(() => (authMode.value === 'register' ? '注册' : '登录'))
const submitText = computed(() => {
  if (submitting.value) return authMode.value === 'register' ? '注册中...' : '登录中...'
  return authMode.value === 'register' ? '注册并登录' : '登录'
})

function resetForm() {
  form.username = ''
  form.password = ''
  errorText.value = ''
  authMode.value = 'login'
}

async function handleSubmit() {
  if (form.username.trim().length < 3) {
    errorText.value = '用户名至少需要 3 个字符'
    return
  }
  if (form.password.length < 6) {
    errorText.value = '密码至少需要 6 个字符'
    return
  }

  submitting.value = true
  errorText.value = ''
  try {
    await userStore.login(form.username.trim().toLowerCase(), form.password, authMode.value)
    emit('success')
    emit('close')
    resetForm()
  } catch (error) {
    errorText.value = error?.response?.data?.detail || error?.message || '登录失败'
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="modal-backdrop" :class="{ open }" aria-hidden="false" @click.self="$emit('close')">
    <section class="modal" role="dialog" aria-modal="true" aria-labelledby="authTitle">
      <div class="modal-head">
        <h3 id="authTitle">{{ title }}</h3>
        <button class="button ghost" type="button" @click="$emit('close')">关闭</button>
      </div>
      <div class="modal-body">
        <div class="tabs">
          <button class="tab" :class="{ active: authMode === 'login' }" type="button" @click="authMode = 'login'">登录</button>
          <button class="tab" :class="{ active: authMode === 'register' }" type="button" @click="authMode = 'register'">注册</button>
        </div>
        <form @submit.prevent="handleSubmit">
          <div class="field">
            <label for="authUsername">用户名</label>
            <input id="authUsername" v-model="form.username" autocomplete="username" />
          </div>
          <div class="field">
            <label for="authPassword">密码</label>
            <input id="authPassword" v-model="form.password" type="password" autocomplete="current-password" />
          </div>
          <button class="button primary" type="submit" :disabled="submitting">{{ submitText }}</button>
          <div v-if="errorText" class="auth-error">{{ errorText }}</div>
        </form>
      </div>
    </section>
  </div>
</template>
