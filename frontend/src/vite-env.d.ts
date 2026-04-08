import type { defineConfig } from 'vite'

export interface ViteEnv {
  VITE_API_URL: string
  VITE_API_KEY?: string
  VITE_APP_NAME?: string
  VITE_DEBUG?: boolean
}

declare global {
  namespace ImportMeta {
    interface ImportMetaEnv extends ViteEnv {}
  }
}

export {}
