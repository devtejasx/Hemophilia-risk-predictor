import axios, { AxiosInstance } from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export const TOKEN_KEY = 'auth_token'

const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
})

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY)
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Only bounce to /login on an expired/invalid session, and never while the
    // user is already there (a failed sign-in returns 401 too).
    if (error.response?.status === 401 && window.location.pathname !== '/login') {
      localStorage.removeItem(TOKEN_KEY)
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

/** Pull a readable message out of a FastAPI error body. */
export function errorMessage(error: unknown, fallback = 'Something went wrong.'): string {
  const detail = (error as { response?: { data?: { detail?: unknown } } })?.response?.data
    ?.detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object') {
    const d = detail as { detail?: string; allowed_values?: string[] }
    if (d.detail) {
      return d.allowed_values?.length
        ? `${d.detail} Accepted values: ${d.allowed_values.join(', ')}.`
        : d.detail
    }
  }
  if (Array.isArray(detail) && detail.length) {
    const first = detail[0] as { loc?: string[]; msg?: string }
    const field = first.loc?.slice(-1)[0]
    return field ? `${field}: ${first.msg}` : String(first.msg)
  }
  return fallback
}

export default apiClient
