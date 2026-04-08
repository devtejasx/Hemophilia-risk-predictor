import React from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts'

const COLORS = ['#9333ea', '#7e22ce', '#6b21a8', '#581c87', '#a855f7']

interface ChartProps {
  data: any[]
  title?: string
  loading?: boolean
}

export const RiskDistributionChart: React.FC<ChartProps> = ({
  data,
  title = 'Risk Distribution',
  loading,
}) => {
  if (loading) return <div className="h-64 bg-slate-200 dark:bg-slate-800 rounded animate-pulse" />

  return (
    <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
      <h3 className="font-semibold text-slate-900 dark:text-white mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, value }) => `${name}: ${value}`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

export const TrendChart: React.FC<ChartProps> = ({
  data,
  title = 'Trends',
  loading,
}) => {
  if (loading) return <div className="h-64 bg-slate-200 dark:bg-slate-800 rounded animate-pulse" />

  return (
    <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
      <h3 className="font-semibold text-slate-900 dark:text-white mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: 'none',
              borderRadius: '8px',
              color: '#f1f5f9',
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="risk"
            stroke="#9333ea"
            dot={{ fill: '#9333ea' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export const SeverityDistributionChart: React.FC<ChartProps> = ({
  data,
  title = 'Severity Distribution',
  loading,
}) => {
  if (loading) return <div className="h-64 bg-slate-200 dark:bg-slate-800 rounded animate-pulse" />

  return (
    <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
      <h3 className="font-semibold text-slate-900 dark:text-white mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: 'none',
              borderRadius: '8px',
              color: '#f1f5f9',
            }}
          />
          <Bar dataKey="count" fill="#9333ea" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export const FeatureImportanceChart: React.FC<{
  data: Array<{ feature: string; importance: number }>
  title?: string
}> = ({ data, title = 'Feature Importance' }) => {
  return (
    <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
      <h3 className="font-semibold text-slate-900 dark:text-white mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 200, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis type="number" stroke="#94a3b8" />
          <YAxis dataKey="feature" type="category" width={190} stroke="#94a3b8" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: 'none',
              borderRadius: '8px',
              color: '#f1f5f9',
            }}
          />
          <Bar dataKey="importance" fill="#9333ea" radius={[0, 8, 8, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default {
  RiskDistributionChart,
  TrendChart,
  SeverityDistributionChart,
  FeatureImportanceChart,
}
