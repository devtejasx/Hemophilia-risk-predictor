import { create } from 'zustand'
import type { Patient, User } from '@/services/api-client'
import { TOKEN_KEY } from '@/services/api'

interface AppState {
  theme: 'light' | 'dark'

  user: User | null
  authChecked: boolean

  patients: Patient[]

  setTheme: (theme: 'light' | 'dark') => void
  setUser: (user: User | null) => void
  setAuthChecked: (checked: boolean) => void
  signOut: () => void
  setPatients: (patients: Patient[]) => void
}

export const useAppStore = create<AppState>((set) => ({
  theme: (localStorage.getItem('theme') as 'light' | 'dark') || 'dark',
  user: null,
  authChecked: false,

  patients: [],

  setTheme: (theme) => {
    localStorage.setItem('theme', theme)
    set({ theme })
  },
  setUser: (user) => set({ user }),
  setAuthChecked: (authChecked) => set({ authChecked }),
  signOut: () => {
    localStorage.removeItem(TOKEN_KEY)
    set({ user: null, patients: [] })
  },
  setPatients: (patients) => set({ patients }),
}))
