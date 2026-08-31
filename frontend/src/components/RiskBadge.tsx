import React from 'react'
import clsx from 'clsx'

/**
 * Probabilistic risk display. Deliberately shows the probability and the
 * model's own threshold rather than a bare verdict such as "HIGH RISK".
 */
export const RiskBadge: React.FC<{
  probability: number
  category: string
  threshold?: number
  size?: 'sm' | 'lg'
}> = ({ probability, category, threshold, size = 'sm' }) => {
  const elevated = category.toLowerCase().includes('elevated')
  return (
    <div className="flex items-baseline gap-2">
      <span
        className={clsx(
          'font-semibold tabular-nums',
          size === 'lg' ? 'text-4xl' : 'text-base',
          elevated
            ? 'text-amber-600 dark:text-amber-400'
            : 'text-emerald-600 dark:text-emerald-400'
        )}
      >
        {(probability * 100).toFixed(1)}%
      </span>
      <span
        className={clsx(
          'rounded px-2 py-0.5 text-xs font-medium',
          elevated
            ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300'
            : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300'
        )}
      >
        {category}
      </span>
      {threshold !== undefined && (
        <span className="text-xs text-slate-500 dark:text-slate-400 tabular-nums">
          threshold {(threshold * 100).toFixed(1)}%
        </span>
      )}
    </div>
  )
}

export default RiskBadge
