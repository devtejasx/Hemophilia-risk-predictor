import React from 'react'
import clsx from 'clsx'
import { ArrowUpRight, ArrowDownRight, TrendingUp } from 'lucide-react'

interface MetricCardProps {
  title: string
  value: string | number
  change?: number
  icon?: React.ReactNode
  loading?: boolean
  color?: 'purple' | 'blue' | 'red' | 'green'
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  icon,
  loading = false,
  color = 'purple',
}) => {
  const colorClasses = {
    purple: 'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
    blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400',
    red: 'bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400',
    green: 'bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400',
  }

  return (
    <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">
            {title}
          </p>
          {loading ? (
            <div className="h-8 w-24 bg-slate-200 dark:bg-slate-700 rounded animate-pulse" />
          ) : (
            <p className="text-3xl font-bold text-slate-900 dark:text-white">
              {value}
            </p>
          )}
        </div>
        {icon && <div className={clsx('p-3 rounded-lg', colorClasses[color])}>{icon}</div>}
      </div>
      {change !== undefined && (
        <div className="flex items-center gap-2">
          {change > 0 ? (
            <ArrowUpRight className="w-4 h-4 text-green-500" />
          ) : (
            <ArrowDownRight className="w-4 h-4 text-red-500" />
          )}
          <span
            className={clsx(
              'text-sm font-medium',
              change > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
            )}
          >
            {Math.abs(change)}% from last month
          </span>
        </div>
      )}
    </div>
  )
}

export default MetricCard
