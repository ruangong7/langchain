<script setup>
import { computed, reactive, ref, watch } from 'vue'
import { useUserStore } from '../stores/user'
import { splitList } from '../lib/profile'

const props = defineProps({
  open: Boolean,
})

const emit = defineEmits(['close', 'saved', 'error'])

const userStore = useUserStore()
const submitting = ref(false)
const validationError = ref('')

const ageOptions = Array.from({ length: 118 }, (_, index) => index + 1)
const genderOptions = [
  { label: '男', value: '男' },
  { label: '女', value: '女' },
  { label: '其他', value: '其他' },
]

const form = reactive({
  display_name: '',
  gender: '',
  age: '',
  conditions: '',
  allergies: '',
  notes: '',
})

watch(
  () => [props.open, userStore.profile],
  () => {
    if (!props.open) return
    const profile = userStore.profile || {}
    form.display_name = profile.display_name || userStore.currentUser?.displayName || userStore.currentUser?.username || ''
    form.gender = profile.gender || ''
    form.age = profile.age ?? ''
    form.conditions = (profile.conditions || []).join('、')
    form.allergies = (profile.allergies || []).join('、')
    form.notes = profile.notes || ''
    validationError.value = ''
  },
  { immediate: true },
)

const canSubmit = computed(() => userStore.isLoggedIn)

async function submitProfile() {
  if (!form.display_name.trim()) {
    validationError.value = '请至少填写一个称呼'
    return
  }

  try {
    submitting.value = true
    validationError.value = ''
    const payload = {
      display_name: form.display_name.trim() || userStore.currentUser?.username || '',
      gender: form.gender.trim(),
      age: String(form.age).trim() ? Number(form.age) : null,
      conditions: splitList(form.conditions),
      allergies: splitList(form.allergies),
      medications: userStore.profile?.medications || [],
      notes: form.notes.trim(),
    }
    await userStore.saveProfile(payload)
    emit('saved')
    emit('close')
  } catch (error) {
    emit('error', error?.response?.data?.detail || error?.message || '保存健康档案失败')
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="modal-backdrop" :class="{ open }" aria-hidden="false" @click.self="$emit('close')">
    <section class="modal" role="dialog" aria-modal="true" aria-labelledby="profileTitle">
      <div class="modal-head">
        <h3 id="profileTitle">健康档案</h3>
        <button class="button ghost" type="button" @click="$emit('close')">关闭</button>
      </div>
      <div class="modal-body modal-body-scroll">
        <form class="modal-form" @submit.prevent="submitProfile">
          <div class="modal-form-content">
            <p class="form-tip">这里更适合填写长期稳定的患者设定，当前正在吃的药放在单独的“当前用药”里维护。</p>

            <div class="compact-grid">
              <div class="field">
                <label for="profileName">称呼</label>
                <input id="profileName" v-model="form.display_name" autocomplete="nickname" placeholder="例如：张阿姨" />
              </div>

              <div class="field">
                <label for="profileGender">性别</label>
                <select id="profileGender" v-model="form.gender">
                  <option value="">请选择</option>
                  <option v-for="option in genderOptions" :key="option.value" :value="option.value">
                    {{ option.label }}
                  </option>
                </select>
              </div>

              <div class="field">
                <label for="profileAge">年龄</label>
                <select id="profileAge" v-model="form.age">
                  <option value="">请选择</option>
                  <option v-for="age in ageOptions" :key="age" :value="age">{{ age }} 岁</option>
                </select>
              </div>

              <div class="field field-span-2">
                <label for="profileConditions">基础病</label>
                <textarea
                  id="profileConditions"
                  v-model="form.conditions"
                  rows="3"
                  placeholder="用顿号、逗号或换行分隔，例如：高血压、糖尿病"
                ></textarea>
              </div>

              <div class="field field-span-2">
                <label for="profileAllergies">过敏史</label>
                <textarea
                  id="profileAllergies"
                  v-model="form.allergies"
                  rows="3"
                  placeholder="用顿号、逗号或换行分隔，例如：青霉素、布洛芬"
                ></textarea>
              </div>

              <div class="field field-span-2">
                <label for="profileNotes">备注</label>
                <textarea
                  id="profileNotes"
                  v-model="form.notes"
                  rows="4"
                  placeholder="例如：胃溃疡病史、哺乳期、近期发热、肾功能偏弱等"
                ></textarea>
              </div>
            </div>
          </div>

          <div class="modal-form-actions">
            <div v-if="validationError" class="auth-error">{{ validationError }}</div>
            <div class="button-row">
              <button class="button ghost" type="button" @click="$emit('close')">取消</button>
              <button class="button primary" type="submit" :disabled="!canSubmit || submitting">
                {{ submitting ? '保存中...' : '确定保存' }}
              </button>
            </div>
          </div>
        </form>
      </div>
    </section>
  </div>
</template>
