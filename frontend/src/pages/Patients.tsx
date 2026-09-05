import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Plus, Users, Loader2 } from 'lucide-react'
import { patientAPI, type Patient } from '@/services/api-client'
import { errorMessage } from '@/services/api'
import { useAppStore } from '@/store/appStore'
import RiskBadge from '@/components/RiskBadge'

const Patients: React.FC = () => {
  const { patients, setPatients } = useAppStore()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    patientAPI
      .list()
      .then(setPatients)
      .catch((err) => setError(errorMessage(err, 'Could not load patients.')))
      .finally(() => setLoading(false))
  }, [setPatients])

  return (
    <div className="p-8 max-w-5xl">
      <div className="flex items-center justify-between mb-6 gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Patients</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Records you have created. Only you can see them.
          </p>
        </div>
        <Link
          to="/patients/new"
          className="inline-flex items-center gap-2 rounded-lg bg-purple-600 hover:bg-purple-700 px-4 py-2 text-white text-sm font-medium"
        >
          <Plus className="w-4 h-4" /> Add patient
        </Link>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-slate-500">
          <Loader2 className="w-4 h-4 animate-spin" /> Loading
        </div>
      )}

      {error && (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      )}

      {!loading && !error && patients.length === 0 && (
        <div className="rounded-xl border border-dashed border-slate-300 dark:border-slate-700 p-10 text-center">
          <Users className="w-8 h-8 mx-auto mb-3 text-slate-400" />
          <p className="text-slate-600 dark:text-slate-300 mb-1">No patients yet</p>
          <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
            Add one to record an F8 mutation and run a risk estimate.
          </p>
          <Link
            to="/patients/new"
            className="text-purple-600 dark:text-purple-400 hover:underline text-sm"
          >
            Add your first patient
          </Link>
        </div>
      )}

      <ul className="space-y-3">
        {patients.map((p: Patient) => (
          <li key={p.id}>
            <Link
              to={`/patients/${p.id}`}
              className="block rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 hover:border-purple-400 dark:hover:border-purple-600"
            >
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <div>
                  <p className="font-medium text-slate-900 dark:text-white">
                    {p.display_name}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {p.identifier} · {p.prediction_count} prediction
                    {p.prediction_count === 1 ? '' : 's'}
                  </p>
                </div>
                {p.latest_probability != null && p.latest_risk_category ? (
                  <RiskBadge
                    probability={p.latest_probability}
                    category={p.latest_risk_category}
                  />
                ) : (
                  <span className="text-xs text-slate-400">No estimate yet</span>
                )}
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default Patients
