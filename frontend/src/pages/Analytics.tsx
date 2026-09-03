import React, { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Loader2 } from 'lucide-react'
import {
  analyticsAPI,
  predictionAPI,
  type Analytics as AnalyticsData,
  type GlobalImportance,
} from '@/services/api-client'
import { errorMessage } from '@/services/api'
import { Disclaimer } from '@/components/Disclaimer'

const PALETTE = ['#8b5cf6', '#0ea5e9', '#14b8a6', '#f59e0b', '#ef4444', '#6366f1']

const Analytics: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null)
  const [importance, setImportance] = useState<GlobalImportance | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([analyticsAPI.get(), predictionAPI.globalImportance()])
      .then(([a, g]) => {
        setData(a)
        setImportance(g)
      })
      .catch((err) => setError(errorMessage(err, 'Could not load analytics.')))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="p-8 flex items-center gap-2 text-slate-500">
        <Loader2 className="w-4 h-4 animate-spin" /> Loading
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-8">
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      </div>
    )
  }

  const riskData = Object.entries(data?.risk_distribution ?? {}).map(([name, value]) => ({
    name,
    value,
  }))
  const mutationData = Object.entries(data?.mutation_type_distribution ?? {}).map(
    ([name, value]) => ({ name, value })
  )
  // Show the human label, not the raw source column name.
  const importanceData = (importance?.features ?? []).map((f) => ({
    name: f.label || f.feature,
    value: Number(f.importance.toFixed(4)),
  }))

  return (
    <div className="p-8 max-w-4xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Analytics</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          {data?.note}
        </p>
      </div>

      <Disclaimer />

      <section>
        <h2 className="font-semibold text-slate-900 dark:text-white mb-1">
          Model feature importance
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
          {importance?.available
            ? `${importance.method}. This describes the model overall, not any individual patient.`
            : `Unavailable: ${importance?.reason ?? 'unknown reason'}`}
        </p>
        {importanceData.length > 0 && (
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importanceData} layout="vertical" margin={{ left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {importanceData.map((_, i) => (
                    <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      <section className="grid gap-8 md:grid-cols-2">
        <div>
          <h2 className="font-semibold text-slate-900 dark:text-white mb-3">
            Your estimates by risk band
          </h2>
          {riskData.length === 0 ? (
            <p className="text-sm text-slate-500 dark:text-slate-400">
              No estimates recorded yet.
            </p>
          ) : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={riskData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div>
          <h2 className="font-semibold text-slate-900 dark:text-white mb-3">
            Mutation types you have assessed
          </h2>
          {mutationData.length === 0 ? (
            <p className="text-sm text-slate-500 dark:text-slate-400">
              No estimates recorded yet.
            </p>
          ) : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={mutationData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

export default Analytics
