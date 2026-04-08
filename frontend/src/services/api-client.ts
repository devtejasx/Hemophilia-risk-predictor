import apiClient from './api'

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
  updated_at: string
}

export interface PredictionResult {
  risk_score: number
  risk_level: string
  main_factor: string
  confidence: number
  importance: Record<string, number>
  shap_explanation?: Record<string, unknown>
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

export interface AnalyticsData {
  total_patients: number
  high_risk_count: number
  average_risk: number
  severe_cases: number
  severity_distribution: Record<string, number>
}

// Patient APIs
export const patientAPI = {
  getAll: async (limit = 100, skip = 0) => {
    const response = await apiClient.get('/patients', {
      params: { limit, skip },
    })
    return response.data
  },

  getById: async (id: string) => {
    const response = await apiClient.get(`/patients/${id}`)
    return response.data
  },

  create: async (data: Omit<Patient, 'id' | 'created_at' | 'updated_at'>) => {
    const response = await apiClient.post('/patients', data)
    return response.data
  },

  update: async (id: string, data: Partial<Patient>) => {
    const response = await apiClient.put(`/patients/${id}`, data)
    return response.data
  },

  delete: async (id: string) => {
    await apiClient.delete(`/patients/${id}`)
  },
}

// Prediction APIs
export const predictionAPI = {
  predict: async (data: Record<string, unknown>) => {
    const response = await apiClient.post('/predict', data)
    return response.data as PredictionResult
  },

  getHistory: async (patientId: string) => {
    const response = await apiClient.get(`/patients/${patientId}/predictions`)
    return response.data
  },

  savePrediction: async (patientId: string, prediction: PredictionResult) => {
    const response = await apiClient.post(
      `/patients/${patientId}/predictions`,
      prediction
    )
    return response.data
  },

  generateReport: async (patientId: string) => {
    const response = await apiClient.get(`/patients/${patientId}/report`, {
      responseType: 'blob',
    })
    return response.data
  },
}

// Chat APIs
export const chatAPI = {
  sendMessage: async (message: string, context?: Record<string, unknown>) => {
    const response = await apiClient.post('/chat', {
      message,
      context,
    })
    return response.data as ChatMessage
  },

  getHistory: async (conversationId: string) => {
    const response = await apiClient.get(`/chat/history/${conversationId}`)
    return response.data as ChatMessage[]
  },
}

// Analytics APIs
export const analyticsAPI = {
  getDashboard: async () => {
    const response = await apiClient.get('/analytics/dashboard')
    return response.data as AnalyticsData
  },

  getRiskDistribution: async () => {
    const response = await apiClient.get('/analytics/risk-distribution')
    return response.data
  },

  getSeverityDistribution: async () => {
    const response = await apiClient.get('/analytics/severity-distribution')
    return response.data
  },

  getTrends: async (days = 30) => {
    const response = await apiClient.get('/analytics/trends', {
      params: { days },
    })
    return response.data
  },
}

// SHAP APIs
export const shapAPI = {
  getExplanation: async (predictionId: string) => {
    const response = await apiClient.get(`/shap/${predictionId}`)
    return response.data
  },

  comparePredictions: async (ids: string[]) => {
    const response = await apiClient.post('/shap/compare', { ids })
    return response.data
  },
}

export default {
  patientAPI,
  predictionAPI,
  chatAPI,
  analyticsAPI,
  shapAPI,
}
