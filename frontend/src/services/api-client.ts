import apiClient from './api'

/* ------------------------------------------------------------------ *
 * Types mirror backend/schemas.py. Keep the two in step.
 * ------------------------------------------------------------------ */

export interface User {
  id: number
  email: string
  full_name: string
  role: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
  expires_in_minutes: number
}

export interface Patient {
  id: number
  identifier: string
  display_name: string
  notes?: string | null
  created_at: string
  updated_at?: string | null
  prediction_count: number
  latest_probability?: number | null
  latest_risk_category?: string | null
}

/** A CHAMP-compatible F8 variant. Keys are the CHAMP column names. */
export interface GenomicInput {
  'Variant Type': string
  Mechanism: string
  Domain: string
  Subtype: string
  'In Poly A': string
  'Reported Clinical Severity': string
  exon_number?: number | null
  codon_number?: number | null
  is_intron?: boolean
}

export interface Prediction {
  id: number
  patient_id: number
  probability: number
  risk_category: string
  threshold: number
  model_version: string
  preprocessing_version: string
  created_at: string
  interpretation: string
  disclaimer: string
}

export interface Contribution {
  feature: string
  value: unknown
  contribution: number
  direction: 'increases' | 'decreases'
}

export interface MethodExplanation {
  available: boolean
  reason?: string | null
  basis?: string | null
  base_value?: number | null
  local_prediction?: number | null
  contributions: Contribution[]
}

export interface Explanation {
  prediction_id: number
  model_version: string
  unit_of_explanation: string
  shap?: MethodExplanation | null
  lime?: MethodExplanation | null
  disclaimer: string
}

export interface Analytics {
  total_patients: number
  total_predictions: number
  mean_probability?: number | null
  risk_distribution: Record<string, number>
  variant_type_distribution: Record<string, number>
  model_version: string
  note: string
}

/** The vocabulary the model was actually fitted on. */
export interface PredictionSchema {
  model_version: string
  categorical: Record<string, string[]>
  numeric: Record<string, { description: string; required: boolean }>
  boolean: Record<string, { description: string; required: boolean }>
  required: string[]
}

export interface GlobalImportance {
  model_version: string
  available: boolean
  reason?: string
  method?: string
  basis?: string
  n_background?: number
  features?: Array<{ feature: string; importance: number }>
}

/* ------------------------------------------------------------------ */

export const authAPI = {
  register: async (email: string, full_name: string, password: string) => {
    const { data } = await apiClient.post<TokenResponse>('/auth/register', {
      email,
      full_name,
      password,
    })
    return data
  },
  login: async (email: string, password: string) => {
    const { data } = await apiClient.post<TokenResponse>('/auth/login', { email, password })
    return data
  },
  me: async () => {
    const { data } = await apiClient.get<User>('/auth/me')
    return data
  },
}

export const patientAPI = {
  list: async (limit = 100, offset = 0) => {
    const { data } = await apiClient.get<Patient[]>('/patients', { params: { limit, offset } })
    return data
  },
  get: async (id: number) => {
    const { data } = await apiClient.get<Patient>(`/patients/${id}`)
    return data
  },
  create: async (payload: { identifier: string; display_name: string; notes?: string }) => {
    const { data } = await apiClient.post<Patient>('/patients', payload)
    return data
  },
  remove: async (id: number) => {
    await apiClient.delete(`/patients/${id}`)
  },
}

export const predictionAPI = {
  schema: async () => {
    const { data } = await apiClient.get<PredictionSchema>('/predictions/schema')
    return data
  },
  create: async (patientId: number, input: GenomicInput) => {
    const { data } = await apiClient.post<Prediction>(
      `/patients/${patientId}/predictions`,
      input
    )
    return data
  },
  get: async (id: number) => {
    const { data } = await apiClient.get<Prediction>(`/predictions/${id}`)
    return data
  },
  history: async (patientId: number) => {
    const { data } = await apiClient.get<Prediction[]>(`/patients/${patientId}/history`)
    return data
  },
  explanation: async (predictionId: number) => {
    const { data } = await apiClient.get<Explanation>(
      `/predictions/${predictionId}/explanation`
    )
    return data
  },
  globalImportance: async () => {
    const { data } = await apiClient.get<GlobalImportance>('/explanations/global')
    return data
  },
}

export const analyticsAPI = {
  get: async () => {
    const { data } = await apiClient.get<Analytics>('/analytics')
    return data
  },
}
