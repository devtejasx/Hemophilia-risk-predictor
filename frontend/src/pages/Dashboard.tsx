import React, { useEffect, useState } from 'react'
import { Navbar } from '@/components/Navbar'
import { MetricCard } from '@/components/MetricCard'
import {
  RiskDistributionChart,
  TrendChart,
  SeverityDistributionChart,
} from '@/components/Charts'
import { PatientCard } from '@/components/PatientCard'
import { analyticsAPI, patientAPI } from '@/services/api-client'
import { useAppStore } from '@/store/appStore'
import { BarChart3, Users, AlertTriangle, TrendingUp } from 'lucide-react'

export const Dashboard: React.FC = () => {
  const [analytics, setAnalytics] = useState<any>(null)
  const [recentPatients, setRecentPatients] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const { setCurrentPatient } = useAppStore()

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const [analyticsData, patientsData] = await Promise.all([
          analyticsAPI.getDashboard(),
          patientAPI.getAll(5, 0),
        ])
        setAnalytics(analyticsData)
        setRecentPatients(patientsData.data || [])
      } catch (error) {
        console.error('Error fetching dashboard data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

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

  const trendData = Array.from({ length: 7 }, (_, i) => ({
    day: `Day ${i + 1}`,
    risk: Math.random() * 100,
  }))

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      <Navbar title="Dashboard" />

      <div className="p-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          <MetricCard
            title="Total Patients"
            value={analytics?.total_patients || 0}
            icon={<Users className="w-6 h-6" />}
            loading={loading}
            change={5}
          />
          <MetricCard
            title="High Risk Patients"
            value={analytics?.high_risk_count || 0}
            icon={<AlertTriangle className="w-6 h-6" />}
            color="red"
            loading={loading}
            change={-2}
          />
          <MetricCard
            title="Average Risk Score"
            value={analytics?.average_risk?.toFixed(2) || '0'}
            icon={<BarChart3 className="w-6 h-6" />}
            color="blue"
            loading={loading}
          />
          <MetricCard
            title="Severe Cases"
            value={analytics?.severe_cases || 0}
            icon={<TrendingUp className="w-6 h-6" />}
            color="green"
            loading={loading}
            change={3}
          />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="lg:col-span-1">
            <RiskDistributionChart data={riskChartData} loading={loading} />
          </div>
          <div className="lg:col-span-2">
            <TrendChart
              data={trendData}
              title="Risk Trends (Last 7 Days)"
              loading={loading}
            />
          </div>
        </div>

        <SeverityDistributionChart data={severityData} loading={loading} />

        {/* Recent Patients */}
        <div className="mt-8">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
            Recent Patients
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {loading ? (
              Array.from({ length: 3 }).map((_, i) => (
                <div
                  key={i}
                  className="h-64 bg-slate-200 dark:bg-slate-800 rounded-lg animate-pulse"
                />
              ))
            ) : recentPatients.length === 0 ? (
              <div className="col-span-full text-center py-12">
                <p className="text-slate-500 dark:text-slate-400">
                  No patients found. Start by adding your first patient.
                </p>
              </div>
            ) : (
              recentPatients.map((patient) => (
                <PatientCard
                  key={patient.id}
                  {...patient}
                  onClick={() => setCurrentPatient(patient)}
                />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
