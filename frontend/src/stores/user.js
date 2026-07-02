import { defineStore } from 'pinia'
import { api, authHeaders } from '../lib/api'

const STORAGE_KEY = 'med_app_user'
const GUEST_SESSION_KEY = 'med_app_guest_session'
const SESSION_STORAGE_PREFIX = 'med_app_sessions'
const ACTIVE_SESSION_STORAGE_PREFIX = 'med_app_active_session'

function readStoredUser() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null')
  } catch {
    return null
  }
}

function getGuestSessionId() {
  try {
    let value = localStorage.getItem(GUEST_SESSION_KEY) || ''
    if (!value) {
      const randomPart = Math.random().toString(36).slice(2, 10)
      value = `guest_${Date.now().toString(36)}_${randomPart}`
      localStorage.setItem(GUEST_SESSION_KEY, value)
    }
    return value
  } catch {
    return `guest_fallback_${Date.now().toString(36)}`
  }
}

function getScopeKey(user) {
  if (user?.id !== undefined && user?.id !== null) {
    return `user_${String(user.id)}`
  }
  return 'guest'
}

function getSessionStorageKey(user) {
  return `${SESSION_STORAGE_PREFIX}:${getScopeKey(user)}`
}

function getActiveSessionStorageKey(user) {
  return `${ACTIVE_SESSION_STORAGE_PREFIX}:${getScopeKey(user)}`
}

function readStoredJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return fallback
    return JSON.parse(raw)
  } catch {
    return fallback
  }
}

function readSessions(user) {
  const parsed = readStoredJson(getSessionStorageKey(user), [])
  return Array.isArray(parsed) ? parsed : []
}

function readActiveSessionId(user) {
  try {
    return localStorage.getItem(getActiveSessionStorageKey(user)) || ''
  } catch {
    return ''
  }
}

function buildDefaultSessionId(user) {
  if (user?.id !== undefined && user?.id !== null) {
    return `user_id_${String(user.id)}`
  }
  return getGuestSessionId()
}

function buildSessionId(user) {
  const randomPart = Math.random().toString(36).slice(2, 10)
  const timePart = Date.now().toString(36)
  if (user?.id !== undefined && user?.id !== null) {
    return `user_id_${String(user.id)}_session_${timePart}_${randomPart}`
  }
  return `guest_session_${timePart}_${randomPart}`
}

function isDefaultTitle(title = '') {
  return /^新会话(?:\s+\d+)?$/.test(String(title || '').trim()) || String(title || '').trim() === '最近会话'
}

function normalizeSession(session, index = 0, user = null) {
  const id = String(session?.id || '').trim() || (index === 0 ? buildDefaultSessionId(user) : buildSessionId(user))
  const title = String(session?.title || '').trim() || (index === 0 ? '最近会话' : `新会话 ${index + 1}`)
  const updatedAt = Number(session?.updatedAt || Date.now())
  const preview = String(session?.preview || '').trim()
  return { id, title, updatedAt, preview }
}

function ensureSessions(user, sessions = []) {
  const normalized = Array.isArray(sessions) ? sessions.map((item, index) => normalizeSession(item, index, user)) : []
  if (normalized.length > 0) return normalized
  return [
    {
      id: buildDefaultSessionId(user),
      title: '最近会话',
      updatedAt: Date.now(),
      preview: '',
    },
  ]
}

export const useUserStore = defineStore('user', {
  state: () => ({
    currentUser: readStoredUser(),
    loadingProfile: false,
    sessions: ensureSessions(readStoredUser(), readSessions(readStoredUser())),
    activeSessionId: readActiveSessionId(readStoredUser()) || ensureSessions(readStoredUser(), readSessions(readStoredUser()))[0]?.id || '',
  }),
  getters: {
    isLoggedIn: (state) => Boolean(state.currentUser?.token),
    token: (state) => state.currentUser?.token || '',
    profile: (state) => state.currentUser?.profile || null,
    memoryId: (state) => state.activeSessionId || state.sessions[0]?.id || buildDefaultSessionId(state.currentUser),
    activeSession: (state) => state.sessions.find((item) => item.id === state.activeSessionId) || state.sessions[0] || null,
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
    persistSessions() {
      const sessions = ensureSessions(this.currentUser, this.sessions)
      this.sessions = sessions
      if (!sessions.some((item) => item.id === this.activeSessionId)) {
        this.activeSessionId = sessions[0]?.id || ''
      }
      try {
        localStorage.setItem(getSessionStorageKey(this.currentUser), JSON.stringify(this.sessions))
        localStorage.setItem(getActiveSessionStorageKey(this.currentUser), this.activeSessionId || '')
      } catch {
        return
      }
    },
    hydrateSessions() {
      this.sessions = ensureSessions(this.currentUser, readSessions(this.currentUser))
      this.activeSessionId = readActiveSessionId(this.currentUser) || this.sessions[0]?.id || ''
      this.persistSessions()
    },
    async loadSessions() {
      if (!this.isLoggedIn) {
        this.hydrateSessions()
        return this.sessions
      }
      const { data } = await api.get('/me/chat-sessions', {
        headers: authHeaders(this.token),
      })
      const sessions = ensureSessions(this.currentUser, data.sessions || [])
      this.sessions = sessions
      if (!sessions.some((item) => item.id === this.activeSessionId)) {
        this.activeSessionId = sessions[0]?.id || ''
      }
      this.persistSessions()
      return this.sessions
    },
    async createSession(title = '') {
      const trimmedTitle = String(title || '').trim()
      if (this.isLoggedIn) {
        const { data } = await api.post(
          '/me/chat-sessions',
          { title: trimmedTitle || '新会话' },
          {
            headers: {
              'Content-Type': 'application/json',
              ...authHeaders(this.token),
            },
          }
        )
        const session = normalizeSession(data, 0, this.currentUser)
        this.sessions = [session, ...this.sessions.filter((item) => item.id !== session.id)]
        this.activeSessionId = session.id
        this.persistSessions()
        return session
      }
      const session = {
        id: buildSessionId(this.currentUser),
        title: trimmedTitle || `新会话 ${this.sessions.length + 1}`,
        updatedAt: Date.now(),
        preview: '',
      }
      this.sessions = [session, ...this.sessions]
      this.activeSessionId = session.id
      this.persistSessions()
      return session
    },
    switchSession(sessionId) {
      const target = this.sessions.find((item) => item.id === sessionId)
      if (!target) return
      this.activeSessionId = target.id
      this.persistSessions()
    },
    updateSessionMeta(sessionId, patch = {}) {
      const targetId = String(sessionId || '').trim()
      const nextSessions = this.sessions.map((item) => {
        if (item.id !== targetId) return item
        return {
          ...item,
          ...patch,
          title: String(patch.title ?? item.title).trim() || item.title,
          preview: String(patch.preview ?? item.preview).trim(),
          updatedAt: Number(patch.updatedAt || Date.now()),
        }
      })
      const updated = nextSessions.find((item) => item.id === targetId)
      if (!updated) return
      this.sessions = [updated, ...nextSessions.filter((item) => item.id !== targetId)]
      this.persistSessions()
    },
    async renameSession(sessionId, title) {
      const text = String(title || '').trim()
      if (!text) return
      if (this.isLoggedIn) {
        const { data } = await api.patch(
          `/me/chat-sessions/${encodeURIComponent(sessionId)}`,
          { title: text.slice(0, 32) },
          {
            headers: {
              'Content-Type': 'application/json',
              ...authHeaders(this.token),
            },
          }
        )
        this.updateSessionMeta(sessionId, {
          title: data.title,
          updatedAt: data.updated_at,
          preview: data.preview,
        })
        return
      }
      this.updateSessionMeta(sessionId, {
        title: text.slice(0, 32),
        updatedAt: Date.now(),
      })
    },
    touchActiveSession(message = '') {
      const active = this.activeSession
      if (!active) return
      const preview = String(message || '').trim()
      const nextTitle = preview && isDefaultTitle(active.title) ? preview.slice(0, 18) : active.title
      this.updateSessionMeta(active.id, {
        title: nextTitle,
        preview: preview.slice(0, 48),
        updatedAt: Date.now(),
      })
    },
    async deleteSession(sessionId) {
      const targetId = String(sessionId || '').trim()
      if (!targetId) return
      if (this.isLoggedIn) {
        await api.delete(`/me/chat-sessions/${encodeURIComponent(targetId)}`, {
          headers: authHeaders(this.token),
        })
      } else {
        await api.delete('/chat-history', {
          params: {
            memory_id: targetId,
          },
          headers: authHeaders(this.token),
        })
      }
      const remaining = this.sessions.filter((item) => item.id !== targetId)
      this.sessions = ensureSessions(this.currentUser, remaining)
      if (this.activeSessionId === targetId) {
        this.activeSessionId = this.sessions[0]?.id || ''
      }
      this.persistSessions()
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
      await this.loadSessions().catch(() => this.hydrateSessions())
      await this.loadProfile().catch(() => null)
      return data
    },
    logout() {
      this.persistUser(null)
      this.hydrateSessions()
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
