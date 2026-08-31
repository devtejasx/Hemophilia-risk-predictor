import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Brain, Loader2 } from 'lucide-react'
import { authAPI } from '@/services/api-client'
import { errorMessage, TOKEN_KEY } from '@/services/api'
import { useAppStore } from '@/store/appStore'
import { Disclaimer } from '@/components/Disclaimer'

type Mode = 'login' | 'register'

const Login: React.FC = () => {
  const navigate = useNavigate()
  const setUser = useAppStore((s) => s.setUser)

  const [mode, setMode] = useState<Mode>('login')
  const [email, setEmail] = useState('')
  const [fullName, setFullName] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)
    setBusy(true)
    try {
      const token =
        mode === 'login'
          ? await authAPI.login(email, password)
          : await authAPI.register(email, fullName, password)
      localStorage.setItem(TOKEN_KEY, token.access_token)
      setUser(await authAPI.me())
      navigate('/', { replace: true })
    } catch (err) {
      setError(errorMessage(err, 'Sign-in failed. Please try again.'))
    } finally {
      setBusy(false)
    }
  }

  const field =
    'w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-sm text-slate-900 dark:text-white placeholder:text-slate-400'

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950 px-4">
      <div className="w-full max-w-md">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-11 h-11 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg text-slate-900 dark:text-white">
              Hemophilia Inhibitor-Risk
            </h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              CHAMP-based research prototype
            </p>
          </div>
        </div>

        <form
          onSubmit={submit}
          className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-4"
        >
          <h2 className="font-semibold text-slate-900 dark:text-white">
            {mode === 'login' ? 'Sign in' : 'Create an account'}
          </h2>

          {mode === 'register' && (
            <div>
              <label htmlFor="full_name" className="block text-sm mb-1 text-slate-700 dark:text-slate-300">
                Full name
              </label>
              <input
                id="full_name"
                className={field}
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
                autoComplete="name"
              />
            </div>
          )}

          <div>
            <label htmlFor="email" className="block text-sm mb-1 text-slate-700 dark:text-slate-300">
              Email
            </label>
            <input
              id="email"
              type="email"
              className={field}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm mb-1 text-slate-700 dark:text-slate-300">
              Password
            </label>
            <input
              id="password"
              type="password"
              className={field}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            />
            {mode === 'register' && (
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                At least 12 characters.
              </p>
            )}
          </div>

          {error && (
            <p role="alert" className="text-sm text-red-600 dark:text-red-400">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-purple-600 hover:bg-purple-700 disabled:opacity-60 px-4 py-2 text-white text-sm font-medium"
          >
            {busy && <Loader2 className="w-4 h-4 animate-spin" />}
            {mode === 'login' ? 'Sign in' : 'Create account'}
          </button>

          <button
            type="button"
            onClick={() => {
              setMode(mode === 'login' ? 'register' : 'login')
              setError(null)
            }}
            className="w-full text-sm text-purple-600 dark:text-purple-400 hover:underline"
          >
            {mode === 'login'
              ? 'No account? Create one'
              : 'Already have an account? Sign in'}
          </button>
        </form>

        <div className="mt-4">
          <Disclaimer />
        </div>
      </div>
    </div>
  )
}

export default Login
