import React, { useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { useAppStore } from '@/store/appStore'
import Sidebar from '@/components/Sidebar'
import Dashboard from '@/pages/Dashboard'
import AddPatient from '@/pages/AddPatient'
import Predictions from '@/pages/Predictions'
import SHAPAnalysis from '@/pages/SHAPAnalysis'
import Chatbot from '@/pages/Chatbot'
import Analytics from '@/pages/Analytics'

function AppLayout({ children }: { children: React.ReactNode }) {
  const { theme } = useAppStore()

  return (
    <div className={theme === 'dark' ? 'dark' : ''}>
      <div className="flex bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white">
        <Sidebar />
        <main className="flex-1">
          {children}
        </main>
      </div>
    </div>
  )
}

function App() {
  const { theme, setTheme } = useAppStore()

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark'
    if (savedTheme) {
      setTheme(savedTheme)
    }
  }, [setTheme])

  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={
            <AppLayout>
              <Dashboard />
            </AppLayout>
          }
        />
        <Route
          path="/add-patient"
          element={
            <AppLayout>
              <AddPatient />
            </AppLayout>
          }
        />
        <Route
          path="/predictions"
          element={
            <AppLayout>
              <Predictions />
            </AppLayout>
          }
        />
        <Route
          path="/shap"
          element={
            <AppLayout>
              <SHAPAnalysis />
            </AppLayout>
          }
        />
        <Route
          path="/chatbot"
          element={
            <AppLayout>
              <Chatbot />
            </AppLayout>
          }
        />
        <Route
          path="/analytics"
          element={
            <AppLayout>
              <Analytics />
            </AppLayout>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  )
}

export default App
