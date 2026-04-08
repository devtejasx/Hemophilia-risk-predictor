import React, { useState, useEffect } from 'react'
import { Navbar } from '@/components/Navbar'
import { PatientCard } from '@/components/PatientCard'
import { FormField } from '@/components/FormField'
import {
  RiskDistributionChart,
  TrendChart,
  SeverityDistributionChart,
} from '@/components/Charts'
import { analyticsAPI, patientAPI } from '@/services/api-client'
import { useAppStore } from '@/store/appStore'
import { Download, Filter, BarChart3 } from 'lucide-react'
import clsx from 'clsx'

export const Analytics: React.FC = () => {
  const { setCurrentPatient } = useAppStore()
  const [patients, setPatients] = useState<any[]>([])
  const [analytics, setAnalytics] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [filterSeverity, setFilterSeverity] = useState<string>('all')
  const [filterRisk, setFilterRisk] = useState<string>('all')
  const [sortBy, setSortBy] = useState<string>('recent')

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      setLoading(true)
      const [analyticsData, patientsData] = await Promise.all([
        analyticsAPI.getDashboard(),
        patientAPI.getAll(1000, 0),
      ])
      setAnalytics(analyticsData)
      setPatients(patientsData.data || [])
    } catch (error) {
      console.error('Error fetching analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  const filteredPatients = patients
    .filter((p) => filterSeverity === 'all' || p.severity === filterSeverity)
    .sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name)
      if (sortBy === 'age') return b.age - a.age
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    })

  const riskChartData = [
    { name: 'Low Risk', value: analytics?.low_risk_count || 0 },
    { name: 'Medium Risk', value: analytics?.medium_risk_count || 0 },
    { name: 'High Risk', value: analytics?.high_risk_count || 0 },
  ]

  const severityData = Object.entries(analytics?.severity_distribution || {}).map(
    ([name, count]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      count,
    })
  )

  const trendData = Array.from({ length: 12 }, (_, i) => ({
    month: `Month ${i + 1}`,
    risk: Math.random() * 100,
    patients: Math.floor(Math.random() * 50) + 10,
  }))

  const handleExport = () => {
    const csv = [
      ['Name', 'Age', 'Severity', 'Mutation Type', 'Risk Level'],
      ...filteredPatients.map((p) => [
        p.name,
        p.age,
        p.severity,
        p.mutation_type,
        p.risk_level || 'Pending',
      ]),
    ]
      .map((row) => row.join(','))
      .join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'patients-analytics.csv'
    a.click()
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      <Navbar title="Analytics & Reporting" />

      <div className="p-6">
        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border border-slate-200 dark:border-slate-800">
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Total Patients</p>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">
              {loading ? '-' : patients.length}
            </p>
          </div>
          <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border border-slate-200 dark:border-slate-800">
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">High Risk</p>
            <p className="text-3xl font-bold text-red-600 dark:text-red-400">
              {loading ? '-' : analytics?.high_risk_count || 0}
            </p>
          </div>
          <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border border-slate-200 dark:border-slate-800">
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Avg Risk</p>
            <p className="text-3xl font-bold text-purple-600 dark:text-purple-400">
              {loading ? '-' : (analytics?.average_risk || 0).toFixed(1)}%
            </p>
          </div>
          <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border border-slate-200 dark:border-slate-800">
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Severe Cases</p>
            <p className="text-3xl font-bold text-orange-600 dark:text-orange-400">
              {loading ? '-' : analytics?.severe_cases || 0}
            </p>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="lg:col-span-1">
            <RiskDistributionChart data={riskChartData} loading={loading} />
          </div>
          <div className="lg:col-span-2">
            <TrendChart
              data={trendData}
              title="Risk Trends (12 Months)"
              loading={loading}
            />
          </div>
        </div>

        <SeverityDistributionChart data={severityData} loading={loading} />

        {/* Filters & Export */}
        <div className="mt-8">
          <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <Filter className="w-5 h-5" />
                Patient Cohort Analysis
              </h2>
              <button
                onClick={handleExport}
                className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition"
              >
                <Download className="w-4 h-4" />
                Export CSV
              </button>
            </div>

            {/* Filters */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <FormField
                label="Filter by Severity"
                name="filterSeverity"
                type="select"
                value={filterSeverity}
                onChange={(val) => setFilterSeverity(val)}
                options={[
                  { label: 'All', value: 'all' },
                  { label: 'Mild', value: 'mild' },
                  { label: 'Moderate', value: 'moderate' },
                  { label: 'Severe', value: 'severe' },
                ]}
              />
              <FormField
                label="Sort By"
                name="sortBy"
                type="select"
                value={sortBy}
                onChange={(val) => setSortBy(val)}
                options={[
                  { label: 'Most Recent', value: 'recent' },
                  { label: 'Name (A-Z)', value: 'name' },
                  { label: 'Age (Oldest)', value: 'age' },
                ]}
              />
              <div className="flex items-end">
                <span className="text-sm text-slate-600 dark:text-slate-400">
                  Showing {filteredPatients.length} patient{filteredPatients.length !== 1 ? 's' : ''}
                </span>
              </div>
            </div>

            {/* Patient List */}
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800">
                    <th className="text-left py-3 px-4 text-sm font-semibold text-slate-900 dark:text-white">
                      Name
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-slate-900 dark:text-white">
                      Age
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-slate-900 dark:text-white">
                      Severity
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-slate-900 dark:text-white">
                      Mutation
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-slate-900 dark:text-white">
                      Risk
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-slate-900 dark:text-white">
                      Action
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPatients.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="py-8 text-center text-slate-500">
                        No patients found
                      </td>
                    </tr>
                  ) : (
                    filteredPatients.slice(0, 10).map((patient) => (
                      <tr
                        key={patient.id}
                        className="border-b border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 transition"
                      >
                        <td className="py-3 px-4 text-slate-900 dark:text-white">{patient.name}</td>
                        <td className="py-3 px-4 text-slate-600 dark:text-slate-400">{patient.age}</td>
                        <td className="py-3 px-4">
                          <span
                            className={clsx(
                              'px-3 py-1 rounded-full text-xs font-medium',
                              patient.severity === 'severe'
                                ? 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400'
                                : patient.severity === 'moderate'
                                  ? 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400'
                                  : 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                            )}
                          >
                            {patient.severity}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-slate-600 dark:text-slate-400">
                          {patient.mutation_type}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={clsx(
                              'font-semibold',
                              patient.risk_level === 'high'
                                ? 'text-red-600 dark:text-red-400'
                                : patient.risk_level === 'medium'
                                  ? 'text-yellow-600 dark:text-yellow-400'
                                  : 'text-green-600 dark:text-green-400'
                            )}
                          >
                            {patient.risk_level || 'Pending'}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <button
                            onClick={() => setCurrentPatient(patient)}
                            className="text-purple-600 dark:text-purple-400 hover:underline text-sm font-medium"
                          >
                            View
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Analytics
