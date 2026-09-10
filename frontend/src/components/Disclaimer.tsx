import React from 'react'
import { Info } from 'lucide-react'
import clsx from 'clsx'

export const DISCLAIMER_TEXT =
  'Research decision-support prototype. This prediction is not intended for standalone diagnosis or treatment decisions.'

/**
 * Persistent research-prototype notice. Present on every screen that shows or
 * produces a model estimate.
 */
export const Disclaimer: React.FC<{ variant?: 'banner' | 'inline'; children?: React.ReactNode }> = ({
  variant = 'inline',
  children,
}) => (
  <div
    role="note"
    className={clsx(
      'flex gap-3 rounded-lg border text-sm',
      'border-amber-300 bg-amber-50 text-amber-900',
      'dark:border-amber-700/60 dark:bg-amber-950/40 dark:text-amber-200',
      variant === 'banner' ? 'p-4' : 'p-3'
    )}
  >
    <Info className="w-4 h-4 mt-0.5 shrink-0" aria-hidden />
    <div>
      <p>{children ?? DISCLAIMER_TEXT}</p>
    </div>
  </div>
)

/**
 * The unit-of-analysis caveat. A row of the training data is one *F8 mutation*:
 * MMC2 describes the mutation itself, MMC3 supplies the clinical records
 * reported for it, and the several records that may name the same mutation are
 * aggregated into that single row. An estimate is therefore attributable to the
 * mutation as the source literature reports it, not to an individual's future.
 */
export const MutationLevelNote: React.FC = () => (
  <p className="text-xs text-slate-500 dark:text-slate-400">
    An estimate describes a mutation, not a person: how often this F8 mutation
    is reported alongside an inhibitor in the source literature. It is not an
    individual patient&rsquo;s probability of developing an inhibitor.
  </p>
)

export default Disclaimer
