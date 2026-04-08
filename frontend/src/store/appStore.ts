import { create } from 'zustand'

export interface Patient {
  id: string
  name: string
  age: number
  severity: string
  mutation_type: string
  blood_type: string
  email?: string
  phone?: string
  created_at: string
}

export interface PredictionResult {
  risk_score: number
  risk_level: string
  main_factor: string
  confidence: number
  importance: Record<string, number>
}

interface AppState {
  // UI State
  theme: 'light' | 'dark'
  sidebarOpen: boolean
  
  // Patient State
  currentPatient: Patient | null
  patients: Patient[]
  patientsLoading: boolean
  patientsError: string | null
  
  // Prediction State
  lastPrediction: PredictionResult | null
  predictionHistory: PredictionResult[]
  
  // Chat State
  chatMessages: Array<{ role: 'user' | 'assistant'; content: string }>
  selectedChatMode: 'clinical' | 'general' | 'treatment'
  
  // Actions
  setTheme: (theme: 'light' | 'dark') => void
  toggleSidebar: () => void
  setCurrentPatient: (patient: Patient | null) => void
  setPatients: (patients: Patient[]) => void
  setPatientsLoading: (loading: boolean) => void
  setPatientsError: (error: string | null) => void
  addPatient: (patient: Patient) => void
  setLastPrediction: (prediction: PredictionResult | null) => void
  addPredictionToHistory: (prediction: PredictionResult) => void
  addChatMessage: (role: 'user' | 'assistant', content: string) => void
  clearChatHistory: () => void
  setChatMode: (mode: 'clinical' | 'general' | 'treatment') => void
}

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  theme: (localStorage.getItem('theme') as 'light' | 'dark') || 'dark',
  sidebarOpen: true,
  currentPatient: null,
  patients: [],
  patientsLoading: false,
  patientsError: null,
  lastPrediction: null,
  predictionHistory: [],
  chatMessages: [],
  selectedChatMode: 'clinical',

  // Actions
  setTheme: (theme) => {
    localStorage.setItem('theme', theme)
    set({ theme })
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  },

  toggleSidebar: () =>
    set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  setCurrentPatient: (patient) =>
    set({ currentPatient: patient }),

  setPatients: (patients) =>
    set({ patients }),

  setPatientsLoading: (loading) =>
    set({ patientsLoading: loading }),

  setPatientsError: (error) =>
    set({ patientsError: error }),

  addPatient: (patient) =>
    set((state) => ({
      patients: [patient, ...state.patients],
    })),

  setLastPrediction: (prediction) =>
    set({ lastPrediction: prediction }),

  addPredictionToHistory: (prediction) =>
    set((state) => ({
      predictionHistory: [prediction, ...state.predictionHistory],
    })),

  addChatMessage: (role, content) =>
    set((state) => ({
      chatMessages: [
        ...state.chatMessages,
        { role, content },
      ],
    })),

  clearChatHistory: () =>
    set({ chatMessages: [] }),

  setChatMode: (mode) =>
    set({ selectedChatMode: mode }),
}))
