import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import {
  predictionAPI,
  type Explanation as ExplanationData,
  type MethodExplanation,
  type Prediction,
} from '@/services/api-client'
import { errorMessage } from '@/services/api'
import RiskBadge from '@/components/RiskBadge'
import { Disclaimer } from '@/components/Disclaimer'

const ContributionBar: React.FC<{ value: number; max: number }> = ({ value, max }) => {
  const width = max === 0 ? 0 : (Math.abs(value) / max) * 100
  const positive = value > 0
  return (
    <div className="flex items-center h-3" aria-hidden>
      <div className="w-1/2 flex justify-end">
        {!positive && (
          <div
            className="h-3 rounded-l bg-emerald-500/70"
            style={{ width: `${width}%` }}
          />
        )}
      </div>
      <div className="w-px h-3 bg-slate-300 dark:bg-slate-600" />
      <div className="w-1/2">
        {positive && (
          <div className="h-3 rounded-r bg-amber-500/70" style={{ width: `${width}%` }} />
        )}
      </div>
    </div>
  )
}

const MethodPanel: React.FC<{ title: string; note: string; data?: MethodExplanation | null }> = ({
  title,
  note,
  data,
}) => {
  if (!data) return null

  if (!data.available) {
    return (
      <div className="rounded-lg border border-slate-200 dark:border-slate-800 p-4">
        <h3 className="font-semibold text-slate-900 dark:text-white mb-1">{title}</h3>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Not available: {data.reason ?? 'unknown reason'}
        </p>
      </div>
    )
  }

  const max = Math.max(...data.contributions.map((c) => Math.abs(c.contribution)), 0)

  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-800 p-4">
      <h3 className="font-semibold text-slate-900 dark:text-white">{title}</h3>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        {note}
        {data.basis ? ` Attribution basis: ${data.basis}.` : ''}
      </p>

      <ul className="space-y-3">
        {data.contributions.map((c) => (
          <li key={c.feature}>
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span className="text-slate-900 dark:text-white">
                {c.label || c.feature}
              </span>
              <span className="text-slate-500 dark:text-slate-400 text-xs text-right">
                {c.supplied ? String(c.value ?? '—') : 'not reported'}
              </span>
            </div>
            <ContributionBar value={c.contribution} max={max} />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 tabular-nums">
              {c.direction === 'increases'
                ? 'Increases'
                : c.direction === 'decreases'
                  ? 'Decreases'
                  : 'No effect on'}{' '}
              the estimate
              {' · '}
              {c.contribution >= 0 ? '+' : ''}
              {c.contribution.toFixed(4)}
            </p>
          </li>
        ))}
      </ul>

      <div className="mt-4 flex items-center gap-4 text-xs text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-2 rounded bg-amber-500/70" /> increases
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-2 rounded bg-emerald-500/70" /> decreases
        </span>
      </div>
    </div>
  )
}

const Explanation: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const predictionId = Number(id)

  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [explanation, setExplanation] = useState<ExplanationData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!Number.isFinite(predictionId)) {
      setError('Invalid prediction reference.')
      setLoading(false)
      return
    }
    Promise.all([
      predictionAPI.get(predictionId),
      predictionAPI.explanation(predictionId),
    ])
      .then(([p, e]) => {
        setPrediction(p)
        setExplanation(e)
      })
      .catch((err) => setError(errorMessage(err, 'Could not load the explanation.')))
      .finally(() => setLoading(false))
  }, [predictionId])

  if (loading) {
    return (
      <div className="p-8 flex items-center gap-2 text-slate-500">
        <Loader2 className="w-4 h-4 animate-spin" /> Computing explanation
      </div>
    )
  }

  if (error || !prediction || !explanation) {
    return (
      <div className="p-8">
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {error ?? 'Explanation unavailable.'}
        </p>
      </div>
    )
  }

  return (
    <div className="p-8 max-w-3xl space-y-6">
      <div>
        <Link
          to={`/patients/${prediction.patient_id}`}
          className="text-sm text-purple-600 dark:text-purple-400 hover:underline"
        >
          Back to patient
        </Link>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white mt-1">
          Why the model produced this estimate
        </h1>
      </div>

      <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
        <RiskBadge
          probability={prediction.probability}
          category={prediction.risk_category}
          threshold={prediction.threshold}
          size="lg"
        />
        <p className="mt-3 text-sm text-slate-700 dark:text-slate-300">
          {prediction.interpretation}
        </p>
      </div>

      <p className="text-sm text-slate-600 dark:text-slate-300">
        {explanation.unit_of_explanation}
      </p>

      <div className="grid gap-4 md:grid-cols-2">
        <MethodPanel
          title="SHAP"
          note="Game-theoretic attribution: how much each field moved this estimate away from the model's average."
          data={explanation.shap}
        />
        <MethodPanel
          title="LIME"
          note="Local linear approximation around this specific input."
          data={explanation.lime}
        />
      </div>

      <Disclaimer variant="banner" />
    </div>
  )
}

export default Explanation
