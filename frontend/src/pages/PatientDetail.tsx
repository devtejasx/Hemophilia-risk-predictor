import React, { useCallback, useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Loader2, Dna, History } from 'lucide-react'
import {
  patientAPI,
  predictionAPI,
  type Patient,
  type Prediction,
} from '@/services/api-client'
import { errorMessage } from '@/services/api'
import RiskBadge from '@/components/RiskBadge'
import { Disclaimer, VariantLevelNote } from '@/components/Disclaimer'
import GenomicForm from '@/components/GenomicForm'

/**
 * One patient: their record, the genomic input form, and the full history of
 * estimates made for them.
 */
const PatientDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const patientId = Number(id)

  const [patient, setPatient] = useState<Patient | null>(null)
  const [history, setHistory] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

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
          Enter the F8 variant as described in CHAMP. Options are limited to values the
          model was actually trained on.
        </p>
        <GenomicForm patientId={patientId} onPredicted={refresh} />
      </section>

      <section>
        <h2 className="flex items-center gap-2 font-semibold text-slate-900 dark:text-white mb-1">
          <History className="w-4 h-4" /> Prediction history
        </h2>
        <VariantLevelNote />

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
                  {p.created_at} · model {p.model_version}
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
