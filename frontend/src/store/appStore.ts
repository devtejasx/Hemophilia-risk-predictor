import { create } from 'zustand'
import type { Patient, User } from '@/services/api-client'
import { TOKEN_KEY } from '@/services/api'

interface AppState {
  theme: 'light' | 'dark'
  sidebarOpen: boolean

  user: User | null
  authChecked: boolean

  patients: Patient[]
  currentPatient: Patient | null

  setTheme: (theme: 'light' | 'dark') => void
  toggleSidebar: () => void
  setUser: (user: User | null) => void
  setAuthChecked: (checked: boolean) => void
  signOut: () => void
  setPatients: (patients: Patient[]) => void
  setCurrentPatient: (patient: Patient | null) => void
}

export const useAppStore = create<AppState>((set) => ({
  theme: (localStorage.getItem('theme') as 'light' | 'dark') || 'dark',
  sidebarOpen: true,

  user: null,
  authChecked: false,

  patients: [],
  currentPatient: null,

  setTheme: (theme) => {
    localStorage.setItem('theme', theme)
    set({ theme })
  },
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setUser: (user) => set({ user }),
  setAuthChecked: (authChecked) => set({ authChecked }),
  signOut: () => {
    localStorage.removeItem(TOKEN_KEY)
    set({ user: null, patients: [], currentPatient: null })
  },
  setPatients: (patients) => set({ patients }),
  setCurrentPatient: (currentPatient) => set({ currentPatient }),
}))
