import React from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { Brain, Home, Users, TrendingUp, LogOut, Moon, Sun } from 'lucide-react'
import clsx from 'clsx'
import { useAppStore } from '@/store/appStore'

interface SidebarItem {
  label: string
  icon: React.ReactNode
  path: string
}

const items: SidebarItem[] = [
  { label: 'Dashboard', icon: <Home className="w-5 h-5" />, path: '/' },
  { label: 'Patients', icon: <Users className="w-5 h-5" />, path: '/patients' },
  { label: 'Analytics', icon: <TrendingUp className="w-5 h-5" />, path: '/analytics' },
]

export const Sidebar: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const { sidebarOpen, user, signOut, theme, setTheme } = useAppStore()

  if (!sidebarOpen) return null

  const isActive = (path: string) =>
    path === '/' ? location.pathname === '/' : location.pathname.startsWith(path)

  return (
    <aside className="w-64 shrink-0 bg-slate-50 dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 min-h-screen sticky top-0 flex flex-col">
      <div className="p-6 flex-1">
        <div className="flex items-center gap-2 mb-8">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="font-bold text-slate-900 dark:text-white leading-tight">
              Inhibitor Risk
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              MMC2 + MMC3 research prototype
            </p>
          </div>
        </div>

        <nav className="space-y-1">
          {items.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={clsx(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium',
                isActive(item.path)
                  ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300'
                  : 'text-slate-600 dark:text-slate-300 hover:bg-slate-200/60 dark:hover:bg-slate-800'
              )}
            >
              {item.icon}
              {item.label}
            </Link>
          ))}
        </nav>
      </div>

      <div className="p-4 border-t border-slate-200 dark:border-slate-800 space-y-2">
        <button
          type="button"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-200/60 dark:hover:bg-slate-800"
        >
          {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          {theme === 'dark' ? 'Light mode' : 'Dark mode'}
        </button>

        {user && (
          <>
            <p className="px-3 text-xs text-slate-500 dark:text-slate-400 truncate">
              {user.full_name}
            </p>
            <button
              type="button"
              onClick={() => {
                signOut()
                navigate('/login', { replace: true })
              }}
              className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-200/60 dark:hover:bg-slate-800"
            >
              <LogOut className="w-5 h-5" /> Sign out
            </button>
          </>
        )}
      </div>
    </aside>
  )
}

export default Sidebar
