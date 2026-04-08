import React, { useState } from 'react'
import { Navbar } from '@/components/Navbar'
import { FormField } from '@/components/FormField'
import { MetricCard } from '@/components/MetricCard'
import { FeatureImportanceChart } from '@/components/Charts'
import { predictionAPI } from '@/services/api-client'
import { useAppStore } from '@/store/appStore'
import { AlertTriangle, TrendingUp } from 'lucide-react'
import clsx from 'clsx'

export const Predictions: React.FC = () => {
  const { currentPatient, setLastPrediction } = useAppStore()
  const [prediction, setPrediction] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [formData, setFormData] = useState({
    age: currentPatient?.age || '',
    severity: currentPatient?.severity || 'mild',
    mutation_type: currentPatient?.mutation_type || 'Missense',
    treatment_type: 'Factor VIII',
    bleeding_episodes: '0',
    joint_damage: 'No',
    inhibitor_history: 'No',
  })

  const handlePredict = async () => {
    try {
      setLoading(true)
      setError(null)
      const result = await predictionAPI.predict(formData)
      setPrediction(result)
      setLastPrediction(result)
    } catch (err) {
      setError('Failed to generate prediction. Please try again.')
      console.error('Prediction error:', err)
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return 'text-green-600 dark:text-green-400'
      case 'medium':
        return 'text-yellow-600 dark:text-yellow-400'
      case 'high':
        return 'text-red-600 dark:text-red-400'
      default:
        return 'text-slate-600 dark:text-slate-400'
    }
  }

  const getRiskBgColor = (riskLevel: string) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return 'bg-green-50 dark:bg-green-900/20'
      case 'medium':
        return 'bg-yellow-50 dark:bg-yellow-900/20'
      case 'high':
        return 'bg-red-50 dark:bg-red-900/20'
      default:
        return 'bg-slate-50 dark:bg-slate-900/20'
    }
  }

  const importanceData = prediction?.importance
    ? Object.entries(prediction.importance)
        .map(([feature, importance]) => ({
          feature: feature
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (char) => char.toUpperCase()),
          importance: Number(importance) || 0,
        }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 10)
    : []

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      <Navbar title="Risk Predictions" />

      <div className="p-6 max-w-6xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Input Form */}
          <div className="lg:col-span-1 bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800 h-fit sticky top-20">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-6">
              Patient Data
            </h2>

            <FormField
              label="Age"
              name="age"
              type="number"
              value={formData.age}
              onChange={(val) => setFormData({ ...formData, age: val })}
            />

            <FormField
              label="Severity"
              name="severity"
              type="select"
              value={formData.severity}
              onChange={(val) => setFormData({ ...formData, severity: val })}
              options={[
                { label: 'Mild', value: 'mild' },
                { label: 'Moderate', value: 'moderate' },
                { label: 'Severe', value: 'severe' },
              ]}
            />

            <FormField
              label="Mutation Type"
              name="mutation_type"
              type="select"
              value={formData.mutation_type}
              onChange={(val) => setFormData({ ...formData, mutation_type: val })}
              options={[
                { label: 'Missense', value: 'Missense' },
                { label: 'Nonsense', value: 'Nonsense' },
                { label: 'Frameshift', value: 'Frameshift' },
                { label: 'Inversion', value: 'Inversion' },
                { label: 'Large Deletion', value: 'Large Deletion' },
              ]}
            />

            <FormField
              label="Treatment Type"
              name="treatment_type"
              type="select"
              value={formData.treatment_type}
              onChange={(val) => setFormData({ ...formData, treatment_type: val })}
              options={[
                { label: 'Factor VIII', value: 'Factor VIII' },
                { label: 'Factor IX', value: 'Factor IX' },
                { label: 'Prothrombin Complex', value: 'Prothrombin Complex' },
              ]}
            />

            <FormField
              label="Bleeding Episodes (Last Year)"
              name="bleeding_episodes"
              type="number"
              value={formData.bleeding_episodes}
              onChange={(val) => setFormData({ ...formData, bleeding_episodes: val })}
            />

            <FormField
              label="Joint Damage"
              name="joint_damage"
              type="select"
              value={formData.joint_damage}
              onChange={(val) => setFormData({ ...formData, joint_damage: val })}
              options={[
                { label: 'No', value: 'No' },
                { label: 'Mild', value: 'Mild' },
                { label: 'Moderate', value: 'Moderate' },
                { label: 'Severe', value: 'Severe' },
              ]}
            />

            <FormField
              label="Inhibitor History"
              name="inhibitor_history"
              type="select"
              value={formData.inhibitor_history}
              onChange={(val) => setFormData({ ...formData, inhibitor_history: val })}
              options={[
                { label: 'No', value: 'No' },
                { label: 'Transient', value: 'Transient' },
                { label: 'Persistent', value: 'Persistent' },
              ]}
            />

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full mt-6 px-6 py-3 bg-purple-500 text-white font-medium rounded-lg hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {loading ? 'Generating Prediction...' : 'Generate Prediction'}
            </button>
          </div>

          {/* Results */}
          <div className="lg:col-span-2">
            {error && (
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 mb-6">
                <p className="text-red-700 dark:text-red-400">{error}</p>
              </div>
            )}

            {!prediction && !loading && (
              <div className="bg-white dark:bg-slate-900 rounded-lg p-12 border border-slate-200 dark:border-slate-800 text-center">
                <Brain className="w-12 h-12 text-slate-400 dark:text-slate-600 mx-auto mb-4" />
                <p className="text-slate-600 dark:text-slate-400">
                  Enter patient data and click "Generate Prediction" to see risk analysis
                </p>
              </div>
            )}

            {loading && (
              <div className="bg-white dark:bg-slate-900 rounded-lg p-12 border border-slate-200 dark:border-slate-800">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto"></div>
                <p className="text-center mt-4 text-slate-600 dark:text-slate-400">
                  Analyzing patient data...
                </p>
              </div>
            )}

            {prediction && (
              <div className="space-y-6">
                {/* Risk Score */}
                <div
                  className={clsx(
                    'rounded-lg p-8 border-2',
                    getRiskBgColor(prediction.risk_level)
                  )}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <p className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">
                        INHIBITOR DEVELOPMENT RISK
                      </p>
                      <p className={clsx('text-5xl font-bold', getRiskColor(prediction.risk_level))}>
                        {(prediction.risk_score * 100).toFixed(1)}%
                      </p>
                    </div>
                    <AlertTriangle className={clsx('w-12 h-12', getRiskColor(prediction.risk_level))} />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className={clsx('text-lg font-bold', getRiskColor(prediction.risk_level))}>
                      {prediction.risk_level.toUpperCase()} RISK
                    </span>
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Key Finding */}
                <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
                  <h3 className="font-semibold text-slate-900 dark:text-white mb-2">
                    Primary Risk Factor
                  </h3>
                  <p className="text-slate-700 dark:text-slate-300">
                    {prediction.main_factor}
                  </p>
                </div>

                {/* Recommendations */}
                <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
                  <h3 className="font-semibold text-slate-900 dark:text-white mb-4">
                    Clinical Recommendations
                  </h3>
                  <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                    {prediction.risk_level === 'high' && (
                      <>
                        <li>• Consider inhibitor titer screening every 3-6 months</li>
                        <li>• Evaluate immune tolerance therapy (ITT)</li>
                        <li>• Monitor for breakthrough bleeding episodes</li>
                        <li>• Consider prophylactic bypass therapy</li>
                      </>
                    )}
                    {prediction.risk_level === 'medium' && (
                      <>
                        <li>• Regular inhibitor monitoring every 6-12 months</li>
                        <li>• Continue current prophylaxis regimen</li>
                        <li>• Educate patient on warning signs</li>
                        <li>• Schedule follow-up in 3 months</li>
                      </>
                    )}
                    {prediction.risk_level === 'low' && (
                      <>
                        <li>• Continue current treatment plan</li>
                        <li>• Annual inhibitor screening sufficient</li>
                        <li>• Maintain regular clinical monitoring</li>
                        <li>• Patient education on prevention</li>
                      </>
                    )}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Feature Importance */}
        {prediction && importanceData.length > 0 && (
          <div className="mt-6">
            <FeatureImportanceChart data={importanceData} />
          </div>
        )}
      </div>
    </div>
  )
}

import { Brain } from 'lucide-react'

export default Predictions
