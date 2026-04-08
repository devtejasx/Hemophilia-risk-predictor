import React from 'react'
import { Menu, X, Sun, Moon, LogOut } from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import clsx from 'clsx'

export const Navbar: React.FC<{ title?: string }> = ({ title = 'Hemophilia AI' }) => {
  const { theme, setTheme, sidebarOpen, toggleSidebar } = useAppStore()

  return (
    <nav className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 sticky top-0 z-40">
      <div className="px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={toggleSidebar}
            className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition"
          >
            {sidebarOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </button>
          <h1 className="text-xl font-bold text-slate-900 dark:text-white">
            {title}
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition"
          >
            {theme === 'dark' ? (
              <Sun className="w-5 h-5" />
            ) : (
              <Moon className="w-5 h-5" />
            )}
          </button>
          <button className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition">
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
