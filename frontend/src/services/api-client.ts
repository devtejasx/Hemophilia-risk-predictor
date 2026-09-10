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
  latest_risk?: string | null
}

/** The three prediction modes: MMC2 only, MMC3 only, or both. */
export type FeatureSet = 'genomic' | 'clinical' | 'merged'

/**
 * One record to score.
 *
 * `features` is keyed by MMC2/MMC3 source column name. The accepted keys are
 * deliberately not fixed in TypeScript: they are whatever the served model's
 * schema reports for the chosen mode, so a retrained model needs no client
 * change and the two can never disagree.
 */
export interface CaseInput {
  feature_set: FeatureSet
  features: Record<string, string | number>
  mutation_label?: string
}

export interface Prediction {
  id: number
  patient_id: number
  prediction: 0 | 1
  risk: 'Low' | 'High'
  probability: number
  risk_category: string
  threshold: number
  model_version: string
  feature_set: FeatureSet
  preprocessing_version: string
  created_at: string
  interpretation: string
  features: Record<string, string | number>
  mutation_label?: string | null
  disclaimer: string
}

export interface Contribution {
  feature: string
  label: string
  value: unknown
  /** False when the field was left blank; its absence still moved the estimate. */
  supplied: boolean
  contribution: number
  direction: 'increases' | 'decreases' | 'no effect'
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
  feature_set: FeatureSet
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
  mutation_type_distribution: Record<string, number>
  feature_set_distribution: Record<string, number>
  model_version: string
  note: string
}

/** What one prediction mode accepts, straight from the fitted model. */
export interface PredictionSchema {
  model_version: string
  feature_set: FeatureSet
  available_feature_sets: FeatureSet[]
  default_feature_set: FeatureSet
  /** Field -> the values that received their own encoded column. */
  categorical: Record<string, string[]>
  numeric: Record<string, { description: string; required: boolean }>
  /** Field -> human-readable name, so the UI never shows a raw column name. */
  labels: Record<string, string>
  /** Field -> which form section it belongs in. */
  groups: Record<string, 'genomic' | 'clinical'>
  /** Fields where a value the model has not seen is accepted, not rejected. */
  open_vocabulary: string[]
  required: string[]
  optional: string[]
}

export interface GlobalImportance {
  model_version: string
  available: boolean
  reason?: string
  method?: string
  basis?: string
  feature_set?: FeatureSet
  n_background?: number
  features?: Array<{ feature: string; label: string; importance: number }>
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
}

export const predictionAPI = {
  schema: async (featureSet?: FeatureSet) => {
    const { data } = await apiClient.get<PredictionSchema>('/predictions/schema', {
      params: featureSet ? { feature_set: featureSet } : undefined,
    })
    return data
  },
  create: async (patientId: number, input: CaseInput) => {
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
  globalImportance: async (featureSet?: FeatureSet) => {
    const { data } = await apiClient.get<GlobalImportance>('/explanations/global', {
      params: featureSet ? { feature_set: featureSet } : undefined,
    })
    return data
  },
}

export const analyticsAPI = {
  get: async () => {
    const { data } = await apiClient.get<Analytics>('/analytics')
    return data
  },
}
