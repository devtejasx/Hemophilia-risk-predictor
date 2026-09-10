import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Loader2, Dna, History } from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  patientAPI,
  predictionAPI,
  type Patient,
  type Prediction,
} from '@/services/api-client'
import { errorMessage } from '@/services/api'
import RiskBadge from '@/components/RiskBadge'
import { Disclaimer, MutationLevelNote } from '@/components/Disclaimer'
import PredictionForm from '@/components/PredictionForm'

/**
 * One patient: their record, the inhibitor-risk input form, and the full
 * history of estimates recorded against them. Each estimate describes the
 * mutation that was entered, not the patient.
 */
const PatientDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const patientId = Number(id)

  const [patient, setPatient] = useState<Patient | null>(null)
  const [history, setHistory] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  /**
   * The stored estimates in the order they were made. Nothing is interpolated
   * or back-filled: one point per prediction actually recorded, so a gap in the
   * line is a gap in the record.
   */
  const trend = useMemo(
    () =>
      [...history].reverse().map((p, index) => ({
        index: index + 1,
        probability: Number((p.probability * 100).toFixed(1)),
        created_at: p.created_at,
        feature_set: p.feature_set,
      })),
    [history]
  )
  const threshold = history.length ? history[0].threshold * 100 : null

  const refresh = useCallback(async () => {
    const [p, h] = await Promise.all([
      patientAPI.get(patientId),
      predictionAPI.history(patientId),
    ])
    setPatient(p)
    setHistory(h)
  }, [patientId])

  useEffect(() => {
    if (!Number.isFinite(patientId)) {
      setError('Invalid patient reference.')
      setLoading(false)
      return
    }
    refresh()
      .catch((err) => setError(errorMessage(err, 'Could not load this patient.')))
      .finally(() => setLoading(false))
  }, [patientId, refresh])

  if (loading) {
    return (
      <div className="p-8 flex items-center gap-2 text-slate-500">
        <Loader2 className="w-4 h-4 animate-spin" /> Loading
      </div>
    )
  }

  if (error || !patient) {
    return (
      <div className="p-8">
        <p role="alert" className="text-sm text-red-600 dark:text-red-400 mb-4">
          {error ?? 'Patient not found.'}
        </p>
        <Link to="/patients" className="text-purple-600 dark:text-purple-400 hover:underline text-sm">
          Back to patients
        </Link>
      </div>
    )
  }

  return (
    <div className="p-8 max-w-4xl space-y-8">
      <div>
        <Link
          to="/patients"
          className="text-sm text-purple-600 dark:text-purple-400 hover:underline"
        >
          Patients
        </Link>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white mt-1">
          {patient.display_name}
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          {patient.identifier} · added {patient.created_at}
        </p>
        {patient.notes && (
          <p className="mt-3 text-sm text-slate-600 dark:text-slate-300">{patient.notes}</p>
        )}
      </div>

      <Disclaimer variant="banner" />

      <section>
        <h2 className="flex items-center gap-2 font-semibold text-slate-900 dark:text-white mb-1">
          <Dna className="w-4 h-4" /> New risk estimate
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
          Describe the F8 mutation (MMC2), the clinical findings reported for
          it (MMC3), or both. Options are limited to values the model was
          actually fitted on.
        </p>
        <PredictionForm patientId={patientId} onPredicted={refresh} />
      </section>

      <section>
        <h2 className="flex items-center gap-2 font-semibold text-slate-900 dark:text-white mb-1">
          <History className="w-4 h-4" /> Prediction history
        </h2>
        <MutationLevelNote />

        {trend.length >= 2 && threshold !== null && (
          <div className="mt-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
            <h3 className="text-sm font-medium text-slate-900 dark:text-white">
              Estimates over time
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
              Each point is one recorded estimate, in the order it was made. The
              dashed line is the model's decision threshold. Points may come from
              different input modes, so a change between them reflects what was
              entered, not a change in the mutation.
            </p>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trend} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="index" tick={{ fontSize: 11 }} />
                  <YAxis
                    domain={[0, 100]}
                    unit="%"
                    tick={{ fontSize: 11 }}
                    width={48}
                  />
                  <Tooltip
                    formatter={(value: number) => [`${value}%`, 'Estimated risk']}
                    labelFormatter={(index: number) => {
                      const point = trend[Number(index) - 1]
                      return point ? `${point.created_at} · ${point.feature_set}` : ''
                    }}
                  />
                  <ReferenceLine
                    y={threshold}
                    stroke="#94a3b8"
                    strokeDasharray="4 4"
                    label={{
                      value: `threshold ${threshold.toFixed(1)}%`,
                      position: 'insideTopRight',
                      fontSize: 10,
                      fill: '#94a3b8',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="probability"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {trend.length === 1 && (
          <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">
            One estimate recorded. A trend appears once there are at least two.
          </p>
        )}

        {history.length === 0 ? (
          <p className="mt-4 text-sm text-slate-500 dark:text-slate-400">
            No estimates recorded for this patient yet.
          </p>
        ) : (
          <ul className="mt-4 space-y-3">
            {history.map((p) => (
              <li
                key={p.id}
                className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4"
              >
                <div className="flex items-center justify-between gap-4 flex-wrap mb-2">
                  <RiskBadge
                    probability={p.probability}
                    category={p.risk_category}
                    threshold={p.threshold}
                  />
                  <Link
                    to={`/explanations/${p.id}`}
                    className="text-sm text-purple-600 dark:text-purple-400 hover:underline"
                  >
                    View explanation
                  </Link>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  {p.created_at} · {p.feature_set} · model {p.model_version}
                  {p.mutation_label ? ' · ' + p.mutation_label : ''}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}

export default PatientDetail
