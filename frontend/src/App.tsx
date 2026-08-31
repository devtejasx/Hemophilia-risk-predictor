import React, { useEffect } from 'react'
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useLocation,
} from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { authAPI } from '@/services/api-client'
import { TOKEN_KEY } from '@/services/api'
import Sidebar from '@/components/Sidebar'
import Login from '@/pages/Login'
import Dashboard from '@/pages/Dashboard'
import Patients from '@/pages/Patients'
import AddPatient from '@/pages/AddPatient'
import PatientDetail from '@/pages/PatientDetail'
import Explanation from '@/pages/Explanation'
import Analytics from '@/pages/Analytics'

function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white">
      <Sidebar />
      <main className="flex-1 min-w-0">{children}</main>
    </div>
  )
}

/** Renders children only for a signed-in user; otherwise redirects to /login. */
function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, authChecked } = useAppStore()
  const location = useLocation()

  if (!authChecked) {
    return (
      <div className="min-h-screen flex items-center justify-center text-slate-500">
        <Loader2 className="w-5 h-5 animate-spin" />
      </div>
    )
  }
  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }
  return <AppLayout>{children}</AppLayout>
}

function App() {
  const { theme, setUser, setAuthChecked, authChecked } = useAppStore()

  // Keep the <html> class in step with the stored theme so Tailwind's `dark:`
  // variants apply. The previous version read the theme but never applied it.
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  // Resolve the stored token to a user once, on load.
  useEffect(() => {
    if (authChecked) return
    const token = localStorage.getItem(TOKEN_KEY)
    if (!token) {
      setAuthChecked(true)
      return
    }
    authAPI
      .me()
      .then(setUser)
      .catch(() => localStorage.removeItem(TOKEN_KEY))
      .finally(() => setAuthChecked(true))
  }, [authChecked, setUser, setAuthChecked])

  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<RequireAuth><Dashboard /></RequireAuth>} />
        <Route path="/patients" element={<RequireAuth><Patients /></RequireAuth>} />
        <Route path="/patients/new" element={<RequireAuth><AddPatient /></RequireAuth>} />
        <Route path="/patients/:id" element={<RequireAuth><PatientDetail /></RequireAuth>} />
        <Route
          path="/explanations/:id"
          element={<RequireAuth><Explanation /></RequireAuth>}
        />
        <Route path="/analytics" element={<RequireAuth><Analytics /></RequireAuth>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  )
}

export default App
