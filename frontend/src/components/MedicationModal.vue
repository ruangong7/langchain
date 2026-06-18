<script setup>
import { computed, reactive, watch } from 'vue'
import { useUserStore } from '../stores/user'

const props = defineProps({
  open: Boolean,
})

const emit = defineEmits(['close', 'saved', 'error'])

const userStore = useUserStore()

const frequencyOptions = ['每日', '隔日', '每周', '按需']
const timeOptions = ['早餐前', '早餐后', '午餐前', '午餐后', '晚餐前', '晚餐后', '睡前', '空腹']
const countOptions = [1, 2, 3, 4]
const dosageValueOptions = ['0.5', '1', '2', '5', '10', '20', '40', '50', '80', '100', '250', '500']
const dosageUnitOptions = ['mg', 'g', 'ml', '片', '粒', '袋', '支']

const state = reactive({
  rows: [],
  submitting: false,
})

function splitDosage(dosage) {
  const text = String(dosage || '').trim()
  const matched = text.match(/^([\d.]+)\s*([A-Za-z\u4e00-\u9fa5]+)$/)
  if (!matched) {
    return {
      dosage_value: '',
      dosage_unit: 'mg',
    }
  }
  return {
    dosage_value: matched[1],
    dosage_unit: matched[2],
  }
}

function createRow(item = {}) {
  const dosageParts = splitDosage(item.dosage)
  return {
    drug_name: item.drug_name || '',
    dosage_value: dosageParts.dosage_value,
    dosage_unit: dosageParts.dosage_unit,
    purpose: item.purpose || '',
    frequency: item.frequency || '每日',
    times_per_day: item.times_per_day ?? 1,
    administration_time: item.administration_time || '早餐后',
    start_date: item.start_date || '',
    end_date: item.end_date || '',
  }
}

watch(
  () => [props.open, userStore.profile],
  () => {
    if (!props.open) return
    state.rows = (userStore.profile?.medications || []).map((item) => createRow(item))
    if (state.rows.length === 0) addRow()
  },
  { immediate: true },
)

const canSubmit = computed(() => userStore.isLoggedIn)

function addRow() {
  state.rows.push(createRow())
}

function removeRow(index) {
  state.rows.splice(index, 1)
  if (state.rows.length === 0) addRow()
}

function buildDosage(row) {
  const value = String(row.dosage_value || '').trim()
  const unit = String(row.dosage_unit || '').trim()
  if (!value) return ''
  return `${value}${unit}`
}

async function submitMedications() {
  try {
    state.submitting = true
    const medications = state.rows
      .map((item) => ({
        drug_name: String(item.drug_name || '').trim(),
        dosage: buildDosage(item),
        purpose: String(item.purpose || '').trim(),
        frequency: String(item.frequency || '').trim(),
        times_per_day: item.times_per_day ? Number(item.times_per_day) : null,
        administration_time: String(item.administration_time || '').trim(),
        start_date: item.start_date || null,
        end_date: item.end_date || null,
      }))
      .filter((item) => item.drug_name)

    const profile = userStore.profile || {}
    await userStore.saveProfile({
      display_name: profile.display_name || userStore.currentUser?.displayName || userStore.currentUser?.username || '',
      gender: profile.gender || '',
      age: profile.age ?? null,
      conditions: profile.conditions || [],
      allergies: profile.allergies || [],
      medications,
      notes: profile.notes || '',
    })
    emit('saved')
    emit('close')
  } catch (error) {
    emit('error', error?.response?.data?.detail || error?.message || '保存当前用药失败')
  } finally {
    state.submitting = false
  }
}
</script>

<template>
  <div class="modal-backdrop" :class="{ open }" aria-hidden="false" @click.self="$emit('close')">
    <section class="modal modal-wide" role="dialog" aria-modal="true" aria-labelledby="medicationTitle">
      <div class="modal-head">
        <h3 id="medicationTitle">当前用药</h3>
        <button class="button ghost" type="button" @click="$emit('close')">关闭</button>
      </div>
      <div class="modal-body modal-body-scroll">
        <form class="modal-form" @submit.prevent="submitMedications">
          <div class="modal-form-content">
            <p class="form-tip">这里单独维护当前正在使用的药物。时间、频率、次数和剂量都使用固定选择，方便后续做长期记忆。</p>

            <div class="table-tools">
              <button class="button" type="button" @click="addRow">新增一条用药</button>
            </div>

            <div class="medication-list">
              <div v-for="(row, index) in state.rows" :key="index" class="medication-card">
                <div class="medication-card-head">
                  <strong>用药 {{ index + 1 }}</strong>
                  <button class="button ghost danger" type="button" @click="removeRow(index)">删除</button>
                </div>
                <div class="medication-grid">
                  <div class="field">
                    <label>药名</label>
                    <input v-model="row.drug_name" placeholder="例如：缬沙坦" />
                  </div>
                  <div class="field">
                    <label>剂量</label>
                    <div class="inline-pair">
                      <select v-model="row.dosage_value">
                        <option value="">请选择</option>
                        <option v-for="option in dosageValueOptions" :key="option" :value="option">{{ option }}</option>
                      </select>
                      <select v-model="row.dosage_unit">
                        <option v-for="option in dosageUnitOptions" :key="option" :value="option">{{ option }}</option>
                      </select>
                    </div>
                  </div>
                  <div class="field">
                    <label>频率</label>
                    <select v-model="row.frequency">
                      <option v-for="option in frequencyOptions" :key="option" :value="option">{{ option }}</option>
                    </select>
                  </div>
                  <div class="field">
                    <label>次数</label>
                    <select v-model="row.times_per_day">
                      <option v-for="count in countOptions" :key="count" :value="count">{{ count }} 次</option>
                    </select>
                  </div>
                  <div class="field">
                    <label>服用时段</label>
                    <select v-model="row.administration_time">
                      <option v-for="option in timeOptions" :key="option" :value="option">{{ option }}</option>
                    </select>
                  </div>
                  <div class="field">
                    <label>开始日期</label>
                    <input v-model="row.start_date" type="date" />
                  </div>
                  <div class="field">
                    <label>结束日期</label>
                    <input v-model="row.end_date" type="date" />
                  </div>
                  <div class="field field-span-2">
                    <label>用途</label>
                    <input v-model="row.purpose" placeholder="例如：降压、止痛、退烧" />
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="modal-form-actions">
            <div class="button-row">
              <button class="button ghost" type="button" @click="$emit('close')">取消</button>
              <button class="button primary" type="submit" :disabled="!canSubmit || state.submitting">
                {{ state.submitting ? '保存中...' : '确定保存' }}
              </button>
            </div>
          </div>
        </form>
      </div>
    </section>
  </div>
</template>
