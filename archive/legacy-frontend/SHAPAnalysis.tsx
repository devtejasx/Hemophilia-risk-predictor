import React, { useState, useEffect } from 'react'
import { Navbar } from '@/components/Navbar'
import { FeatureImportanceChart } from '@/components/Charts'
import { shapAPI, predictionAPI } from '@/services/api-client'
import { useAppStore } from '@/store/appStore'
import { BarChart3, ZoomIn, Eye, EyeOff } from 'lucide-react'
import clsx from 'clsx'

type ViewMode = 'basic' | 'advanced' | 'detailed'

export const SHAPAnalysis: React.FC = () => {
  const { lastPrediction } = useAppStore()
  const [viewMode, setViewMode] = useState<ViewMode>('basic')
  const [loading, setLoading] = useState(false)
  const [shaplaExplanation, setSHAPExplanation] = useState<any>(null)

  useEffect(() => {
    if (lastPrediction) {
      loadExplanation()
    }
  }, [lastPrediction])

  const loadExplanation = async () => {
    try {
      setLoading(true)
      if (lastPrediction?.shap_explanation) {
        setSHAPExplanation(lastPrediction.shap_explanation)
      }
    } catch (error) {
      console.error('Error loading SHAP explanation:', error)
    } finally {
      setLoading(false)
    }
  }

  if (!lastPrediction) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
        <Navbar title="SHAP Analysis" />
        <div className="flex items-center justify-center h-96 px-6">
          <div className="text-center">
            <BarChart3 className="w-16 h-16 text-slate-400 dark:text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
              No Prediction Available
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              Generate a prediction first to see SHAP analysis
            </p>
          </div>
        </div>
      </div>
    )
  }

  const importanceData = lastPrediction.importance
    ? Object.entries(lastPrediction.importance)
        .map(([feature, importance]) => ({
          feature: feature
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (char) => char.toUpperCase()),
          importance: Number(importance) || 0,
        }))
        .sort((a, b) => b.importance - a.importance)
    : []

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      <Navbar title="SHAP Explainability Analysis" />

      <div className="p-6 max-w-6xl mx-auto">
        {/* View Mode Tabs */}
        <div className="flex gap-2 mb-6 bg-white dark:bg-slate-900 p-1 rounded-lg border border-slate-200 dark:border-slate-800 w-fit">
          {(['basic', 'advanced', 'detailed'] as ViewMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              className={clsx(
                'px-4 py-2 rounded font-medium transition',
                viewMode === mode
                  ? 'bg-purple-500 text-white'
                  : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
              )}
            >
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>

        {/* Basic View - Feature Importance Only */}
        {viewMode === 'basic' && (
          <div className="space-y-6">
            <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
              <div className="mb-6">
                <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
                  Feature Importance
                </h2>
                <p className="text-slate-600 dark:text-slate-400">
                  Shows which patient features most influenced the risk prediction
                </p>
              </div>

              <div className="space-y-3">
                {importanceData.slice(0, 5).map((item) => (
                  <div key={item.feature}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-slate-900 dark:text-white font-medium">
                        {item.feature}
                      </span>
                      <span className="text-purple-600 dark:text-purple-400 font-bold">
                        {(item.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-purple-500 to-purple-700 h-full rounded-full transition-all"
                        style={{ width: `${Math.min(item.importance * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border border-blue-200 dark:border-blue-800">
                <p className="text-sm text-blue-600 dark:text-blue-400 font-medium mb-2">
                  PREDICTION
                </p>
                <p className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                  {(lastPrediction.risk_score * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border border-purple-200 dark:border-purple-800">
                <p className="text-sm text-purple-600 dark:text-purple-400 font-medium mb-2">
                  RISK LEVEL
                </p>
                <p className="text-2xl font-bold text-purple-700 dark:text-purple-300">
                  {lastPrediction.risk_level}
                </p>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border border-green-200 dark:border-green-800">
                <p className="text-sm text-green-600 dark:text-green-400 font-medium mb-2">
                  CONFIDENCE
                </p>
                <p className="text-2xl font-bold text-green-700 dark:text-green-300">
                  {(lastPrediction.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Advanced View - Full Feature Chart */}
        {viewMode === 'advanced' && (
          <div className="space-y-6">
            <FeatureImportanceChart data={importanceData} />

            <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
              <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-4">
                Feature Breakdown
              </h3>

              <div className="space-y-4">
                {importanceData.map((item, idx) => (
                  <div
                    key={item.feature}
                    className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-purple-500 text-white flex items-center justify-center font-bold text-sm">
                        {idx + 1}
                      </div>
                      <span className="text-slate-900 dark:text-white">{item.feature}</span>
                    </div>
                    <span className="text-purple-600 dark:text-purple-400 font-bold">
                      {(item.importance * 100).toFixed(2)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
              <h3 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">
                What This Means
              </h3>
              <p className="text-blue-800 dark:text-blue-300 text-sm">
                The features are ranked by their contribution to the prediction. The higher the
                importance, the more that feature influenced the risk calculation. Use this to
                understand which clinical factors are driving the predicted risk.
              </p>
            </div>
          </div>
        )}

        {/* Detailed View - Full Analysis with Explanations */}
        {viewMode === 'detailed' && (
          <div className="space-y-6">
            <FeatureImportanceChart data={importanceData} />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
                  Risk Interpretation
                </h3>
                <div className="space-y-3 text-slate-700 dark:text-slate-300">
                  <p>
                    <span className="font-semibold">Score:</span> {(lastPrediction.risk_score * 100).toFixed(1)}%
                  </p>
                  <p>
                    <span className="font-semibold">Level:</span> {lastPrediction.risk_level}
                  </p>
                  <p>
                    <span className="font-semibold">Confidence:</span> {(lastPrediction.confidence * 100).toFixed(1)}%
                  </p>
                  <p>
                    <span className="font-semibold">Primary Factor:</span> {lastPrediction.main_factor}
                  </p>
                </div>
              </div>

              <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
                  Model Information
                </h3>
                <div className="space-y-3 text-slate-700 dark:text-slate-300 text-sm">
                  <p>
                    <span className="font-semibold">Model Type:</span> XGBoost Classifier
                  </p>
                  <p>
                    <span className="font-semibold">Explanation Method:</span> SHAP (SHapley Additive exPlanations)
                  </p>
                  <p>
                    <span className="font-semibold">Features Analyzed:</span> {importanceData.length}
                  </p>
                  <p>
                    <span className="font-semibold">Trained On:</span> 5+ years of patient data
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-slate-900 rounded-lg p-6 border border-slate-200 dark:border-slate-800">
              <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
                Top Contributing Features
              </h3>
              <div className="space-y-4">
                {importanceData.slice(0, 3).map((item, idx) => (
                  <div key={item.feature} className="border-l-4 border-purple-500 pl-4 py-2">
                    <div className="flex justify-between items-baseline mb-2">
                      <span className="font-semibold text-slate-900 dark:text-white">
                        {idx + 1}. {item.feature}
                      </span>
                      <span className="text-sm text-slate-600 dark:text-slate-400">
                        Impact: {(item.importance * 100).toFixed(2)}%
                      </span>
                    </div>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      This feature has a significant influence on the inhibitor risk prediction.
                      Consider monitoring this factor closely during clinical follow-up.
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-6">
              <h3 className="font-semibold text-amber-900 dark:text-amber-200 mb-2">
                Clinical Guidance
              </h3>
              <p className="text-amber-800 dark:text-amber-300 text-sm mb-3">
                Use this SHAP analysis to understand which patient characteristics are most predictive
                of inhibitor development. This can help guide:
              </p>
              <ul className="text-sm text-amber-800 dark:text-amber-300 space-y-1">
                <li>• Risk stratification for new patients</li>
                <li>• Targeted monitoring intensity</li>
                <li>• Treatment plan adjustments</li>
                <li>• Patient counseling priorities</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default SHAPAnalysis
