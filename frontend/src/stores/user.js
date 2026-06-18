import { defineStore } from 'pinia'
import { api, authHeaders } from '../lib/api'

const STORAGE_KEY = 'med_app_user'

function readStoredUser() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null')
  } catch {
    return null
  }
}

export const useUserStore = defineStore('user', {
  state: () => ({
    currentUser: readStoredUser(),
    loadingProfile: false,
  }),
  getters: {
    isLoggedIn: (state) => Boolean(state.currentUser?.token),
    token: (state) => state.currentUser?.token || '',
    profile: (state) => state.currentUser?.profile || null,
    memoryId: (state) => {
      if (!state.currentUser) return 'user_guest'
      if (state.currentUser.id !== undefined && state.currentUser.id !== null) {
        return `user_id_${String(state.currentUser.id)}`
      }
      return `user_name_${encodeURIComponent(state.currentUser.username || 'guest')}`
    },
  },
  actions: {
    persistUser(user) {
      this.currentUser = user
      if (user) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(user))
      } else {
        localStorage.removeItem(STORAGE_KEY)
      }
    },
    async login(username, password, mode = 'login') {
      const endpoint = mode === 'register' ? '/auth/register' : '/auth/login'
      const { data } = await api.post(endpoint, { username, password })
      this.persistUser({
        ...data.user,
        displayName: data.user?.username || username,
        token: data.token,
        expiresAt: data.expires_at,
        profile: null,
      })
      await this.loadProfile().catch(() => null)
      return data
    },
    logout() {
      this.persistUser(null)
    },
    async loadProfile() {
      if (!this.token) return null
      this.loadingProfile = true
      try {
        const { data } = await api.get('/me/health-profile', {
          headers: authHeaders(this.token),
        })
        this.persistUser({
          ...this.currentUser,
          displayName: data.display_name || this.currentUser?.displayName || this.currentUser?.username,
          profile: data,
        })
        return data
      } finally {
        this.loadingProfile = false
      }
    },
    async saveProfile(payload) {
      const { data } = await api.put('/me/health-profile', payload, {
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders(this.token),
        },
      })
      this.persistUser({
        ...this.currentUser,
        displayName: data.display_name || this.currentUser?.displayName || this.currentUser?.username,
        profile: data,
      })
      return data
    },
  },
})
