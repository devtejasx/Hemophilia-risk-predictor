import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  BarChart3,
  Plus,
  Brain,
  MessageCircle,
  TrendingUp,
  Settings,
  Home,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import clsx from 'clsx'

interface SidebarItem {
  label: string
  icon: React.ReactNode
  path: string
  badge?: number
}

const sidebarItems: SidebarItem[] = [
  { label: 'Dashboard', icon: <Home className="w-5 h-5" />, path: '/' },
  { label: 'Add Patient', icon: <Plus className="w-5 h-5" />, path: '/add-patient' },
  { label: 'Predictions', icon: <Brain className="w-5 h-5" />, path: '/predictions' },
  { label: 'SHAP Analysis', icon: <BarChart3 className="w-5 h-5" />, path: '/shap' },
  { label: 'Chatbot', icon: <MessageCircle className="w-5 h-5" />, path: '/chatbot' },
  { label: 'Analytics', icon: <TrendingUp className="w-5 h-5" />, path: '/analytics' },
]

export const Sidebar: React.FC = () => {
  const location = useLocation()
  const { sidebarOpen } = useAppStore()
  const [expandedGroups, setExpandedGroups] = React.useState<string[]>([])

  const toggleGroup = (group: string) => {
    setExpandedGroups((prev) =>
      prev.includes(group)
        ? prev.filter((g) => g !== group)
        : [...prev, group]
    )
  }

  if (!sidebarOpen) return null

  return (
    <aside className="w-64 bg-slate-50 dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 h-screen sticky top-0 overflow-y-auto">
      <div className="p-6">
        <div className="flex items-center gap-2 mb-8">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="font-bold text-slate-900 dark:text-white">Hemophilia</h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">Clinical AI</p>
          </div>
        </div>

        <nav className="space-y-2">
          {sidebarItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={clsx(
                'flex items-center gap-3 px-4 py-3 rounded-lg transition',
                location.pathname === item.path
                  ? 'bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 font-medium'
                  : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
              )}
            >
              {item.icon}
              <span className="flex-1">{item.label}</span>
              {item.badge && (
                <span className="bg-red-500 text-white text-xs font-bold px-2 py-1 rounded-full">
                  {item.badge}
                </span>
              )}
            </Link>
          ))}
        </nav>

        <hr className="my-6 border-slate-200 dark:border-slate-700" />

        <button className="w-full flex items-center gap-3 px-4 py-3 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition">
          <Settings className="w-5 h-5" />
          <span>Settings</span>
        </button>
      </div>
    </aside>
  )
}

export default Sidebar
