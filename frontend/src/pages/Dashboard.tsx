import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Users, Activity, Gauge } from 'lucide-react'
import {
  analyticsAPI,
  patientAPI,
  type Analytics,
  type Patient,
} from '@/services/api-client'
import { errorMessage } from '@/services/api'
import MetricCard from '@/components/MetricCard'
import RiskBadge from '@/components/RiskBadge'
import { Disclaimer } from '@/components/Disclaimer'
import { useAppStore } from '@/store/appStore'

const Dashboard: React.FC = () => {
  const user = useAppStore((s) => s.user)
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [recent, setRecent] = useState<Patient[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([analyticsAPI.get(), patientAPI.list(5, 0)])
      .then(([a, p]) => {
        setAnalytics(a)
        setRecent(p)
      })
      .catch((err) => setError(errorMessage(err, 'Could not load the dashboard.')))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="p-8 max-w-5xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
          {user ? `Welcome, ${user.full_name}` : 'Dashboard'}
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Explainable Hemophilia A inhibitor-risk estimates. Genomic information
          from MMC2 is fused with clinical information from MMC3 at the mutation
          level, so an estimate describes a mutation rather than a person.
        </p>
      </div>

      <Disclaimer variant="banner" />

      {error && (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      )}

      <div className="grid gap-4 sm:grid-cols-3">
        <MetricCard
          title="Patients"
          value={analytics?.total_patients ?? 0}
          loading={loading}
          icon={<Users className="w-5 h-5" />}
          color="purple"
        />
        <MetricCard
          title="Estimates run"
          value={analytics?.total_predictions ?? 0}
          loading={loading}
          icon={<Activity className="w-5 h-5" />}
          color="blue"
        />
        <MetricCard
          title="Mean probability"
          value={
            analytics?.mean_probability != null
              ? `${(analytics.mean_probability * 100).toFixed(1)}%`
              : '—'
          }
          loading={loading}
          icon={<Gauge className="w-5 h-5" />}
          color="green"
        />
      </div>

      {analytics && (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {analytics.note} Model version {analytics.model_version}.
        </p>
      )}

      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-slate-900 dark:text-white">Recent patients</h2>
          <Link
            to="/patients"
            className="text-sm text-purple-600 dark:text-purple-400 hover:underline"
          >
            View all
          </Link>
        </div>

        {!loading && recent.length === 0 ? (
          <p className="text-sm text-slate-500 dark:text-slate-400">
            No patients yet.{' '}
            <Link to="/patients/new" className="text-purple-600 dark:text-purple-400 hover:underline">
              Add one
            </Link>
            .
          </p>
        ) : (
          <ul className="space-y-2">
            {recent.map((p) => (
              <li key={p.id}>
                <Link
                  to={`/patients/${p.id}`}
                  className="flex items-center justify-between gap-4 flex-wrap rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3 hover:border-purple-400 dark:hover:border-purple-600"
                >
                  <span className="text-sm text-slate-900 dark:text-white">
                    {p.display_name}
                    <span className="text-slate-400"> · {p.identifier}</span>
                  </span>
                  {p.latest_probability != null && p.latest_risk_category && (
                    <RiskBadge
                      probability={p.latest_probability}
                      category={p.latest_risk_category}
                    />
                  )}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}

export default Dashboard
