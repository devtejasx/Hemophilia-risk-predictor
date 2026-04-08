import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Navbar } from '@/components/Navbar'
import { FormField } from '@/components/FormField'
import { patientAPI } from '@/services/api-client'
import { useAppStore } from '@/store/appStore'
import { ArrowRight, CheckCircle } from 'lucide-react'
import clsx from 'clsx'

type Step = 'demographics' | 'clinical' | 'treatment' | 'history' | 'review'

export const AddPatient: React.FC = () => {
  const navigate = useNavigate()
  const { addPatient } = useAppStore()
  const [step, setStep] = useState<Step>('demographics')
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})

  const [formData, setFormData] = useState({
    // Demographics
    name: '',
    age: '',
    blood_type: 'O+',
    email: '',
    phone: '',

    // Clinical
    severity: 'mild',
    mutation_type: 'Missense',
    family_history: '',
    onset_age: '',

    // Treatment
    treatment_type: 'Factor VIII',
    treatment_frequency: 'Prophylactic',
    inhibitor_history: 'No',

    // History
    bleeding_episodes: '',
    joint_damage: 'No',
    notes: '',
  })

  const steps: Array<{ id: Step; label: string; icon: string }> = [
    { id: 'demographics', label: 'Demographics', icon: '👤' },
    { id: 'clinical', label: 'Clinical', icon: '🏥' },
    { id: 'treatment', label: 'Treatment', icon: '💊' },
    { id: 'history', label: 'History', icon: '📋' },
    { id: 'review', label: 'Review', icon: '✓' },
  ]

  const validateStep = (): boolean => {
    const newErrors: Record<string, string> = {}

    if (step === 'demographics') {
      if (!formData.name.trim()) newErrors.name = 'Name is required'
      if (!formData.age) newErrors.age = 'Age is required'
      if (parseInt(formData.age) < 0 || parseInt(formData.age) > 150) {
        newErrors.age = 'Invalid age'
      }
      if (formData.email && !formData.email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
        newErrors.email = 'Invalid email'
      }
    }

    if (step === 'clinical') {
      if (!formData.mutation_type.trim()) newErrors.mutation_type = 'Mutation type is required'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleNext = () => {
    if (!validateStep()) return

    const stepOrder: Step[] = ['demographics', 'clinical', 'treatment', 'history', 'review']
    const currentIndex = stepOrder.indexOf(step)
    if (currentIndex < stepOrder.length - 1) {
      setStep(stepOrder[currentIndex + 1])
    }
  }

  const handleBack = () => {
    const stepOrder: Step[] = ['demographics', 'clinical', 'treatment', 'history', 'review']
    const currentIndex = stepOrder.indexOf(step)
    if (currentIndex > 0) {
      setStep(stepOrder[currentIndex - 1])
    }
  }

  const handleSubmit = async () => {
    if (!validateStep()) return

    try {
      setLoading(true)
      const newPatient = await patientAPI.create(formData)
      addPatient(newPatient)
      navigate('/predictions')
    } catch (error) {
      console.error('Error adding patient:', error)
      setErrors({ submit: 'Failed to add patient' })
    } finally {
      setLoading(false)
    }
  }

  const updateField = (field: string, value: string | number) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    if (errors[field]) {
      setErrors((prev) => {
        const newErrors = { ...prev }
        delete newErrors[field]
        return newErrors
      })
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      <Navbar title="Add Patient" />

      <div className="max-w-2xl mx-auto p-6">
        {/* Progress Indicator */}
        <div className="mb-8">
          <div className="flex justify-between mb-4">
            {steps.map((s, idx) => (
              <div
                key={s.id}
                className="flex flex-col items-center flex-1"
              >
                <div
                  className={clsx(
                    'w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg mb-2 transition',
                    s.id === step
                      ? 'bg-purple-500 text-white ring-4 ring-purple-200 dark:ring-purple-900'
                      : steps.findIndex((st) => st.id === step) > idx
                        ? 'bg-green-500 text-white'
                        : 'bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                  )}
                >
                  {steps.findIndex((st) => st.id === step) > idx ? (
                    <CheckCircle className="w-6 h-6" />
                  ) : (
                    s.icon
                  )}
                </div>
                <p className="text-sm font-medium text-slate-900 dark:text-white">
                  {s.label}
                </p>
              </div>
            ))}
          </div>
          <div className="h-1 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-purple-700 transition-all duration-300"
              style={{
                width: `${((steps.findIndex((s) => s.id === step) + 1) / steps.length) * 100}%`,
              }}
            />
          </div>
        </div>

        {/* Form Content */}
        <div className="bg-white dark:bg-slate-900 rounded-lg p-8 border border-slate-200 dark:border-slate-800 mb-6">
          {step === 'demographics' && (
            <div>
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
                Patient Demographics
              </h2>
              <FormField
                label="Full Name"
                name="name"
                value={formData.name}
                onChange={(val) => updateField('name', val)}
                error={errors.name}
                required
              />
              <div className="grid grid-cols-2 gap-4">
                <FormField
                  label="Age"
                  name="age"
                  type="number"
                  value={formData.age}
                  onChange={(val) => updateField('age', val)}
                  error={errors.age}
                  required
                />
                <FormField
                  label="Blood Type"
                  name="blood_type"
                  type="select"
                  value={formData.blood_type}
                  onChange={(val) => updateField('blood_type', val)}
                  options={[
                    { label: 'O+', value: 'O+' },
                    { label: 'O-', value: 'O-' },
                    { label: 'A+', value: 'A+' },
                    { label: 'A-', value: 'A-' },
                    { label: 'B+', value: 'B+' },
                    { label: 'B-', value: 'B-' },
                    { label: 'AB+', value: 'AB+' },
                    { label: 'AB-', value: 'AB-' },
                  ]}
                />
              </div>
              <FormField
                label="Email"
                name="email"
                type="email"
                value={formData.email}
                onChange={(val) => updateField('email', val)}
                error={errors.email}
              />
              <FormField
                label="Phone"
                name="phone"
                type="tel"
                value={formData.phone}
                onChange={(val) => updateField('phone', val)}
              />
            </div>
          )}

          {step === 'clinical' && (
            <div>
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
                Clinical Information
              </h2>
              <FormField
                label="Severity"
                name="severity"
                type="select"
                value={formData.severity}
                onChange={(val) => updateField('severity', val)}
                options={[
                  { label: 'Mild', value: 'mild' },
                  { label: 'Moderate', value: 'moderate' },
                  { label: 'Severe', value: 'severe' },
                ]}
                required
              />
              <FormField
                label="Mutation Type"
                name="mutation_type"
                type="select"
                value={formData.mutation_type}
                onChange={(val) => updateField('mutation_type', val)}
                options={[
                  { label: 'Missense', value: 'Missense' },
                  { label: 'Nonsense', value: 'Nonsense' },
                  { label: 'Frameshift', value: 'Frameshift' },
                  { label: 'Inversion', value: 'Inversion' },
                  { label: 'Large Deletion', value: 'Large Deletion' },
                ]}
                error={errors.mutation_type}
                required
              />
              <FormField
                label="Family History"
                name="family_history"
                type="textarea"
                value={formData.family_history}
                onChange={(val) => updateField('family_history', val)}
                placeholder="Describe any family history of hemophilia or bleeding disorders"
              />
              <FormField
                label="Age of Onset"
                name="onset_age"
                type="number"
                value={formData.onset_age}
                onChange={(val) => updateField('onset_age', val)}
              />
            </div>
          )}

          {step === 'treatment' && (
            <div>
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
                Treatment Information
              </h2>
              <FormField
                label="Treatment Type"
                name="treatment_type"
                type="select"
                value={formData.treatment_type}
                onChange={(val) => updateField('treatment_type', val)}
                options={[
                  { label: 'Factor VIII', value: 'Factor VIII' },
                  { label: 'Factor IX', value: 'Factor IX' },
                  { label: 'Prothrombin Complex', value: 'Prothrombin Complex' },
                  { label: 'DDAVP', value: 'DDAVP' },
                ]}
              />
              <FormField
                label="Treatment Frequency"
                name="treatment_frequency"
                type="select"
                value={formData.treatment_frequency}
                onChange={(val) => updateField('treatment_frequency', val)}
                options={[
                  { label: 'On-Demand', value: 'On-Demand' },
                  { label: 'Prophylactic', value: 'Prophylactic' },
                  { label: 'Gene Therapy', value: 'Gene Therapy' },
                ]}
              />
              <FormField
                label="Inhibitor History"
                name="inhibitor_history"
                type="select"
                value={formData.inhibitor_history}
                onChange={(val) => updateField('inhibitor_history', val)}
                options={[
                  { label: 'No', value: 'No' },
                  { label: 'Transient', value: 'Transient' },
                  { label: 'Persistent', value: 'Persistent' },
                ]}
              />
            </div>
          )}

          {step === 'history' && (
            <div>
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
                Medical History
              </h2>
              <FormField
                label="Bleeding Episodes (Last Year)"
                name="bleeding_episodes"
                type="number"
                value={formData.bleeding_episodes}
                onChange={(val) => updateField('bleeding_episodes', val)}
              />
              <FormField
                label="Joint Damage"
                name="joint_damage"
                type="select"
                value={formData.joint_damage}
                onChange={(val) => updateField('joint_damage', val)}
                options={[
                  { label: 'No', value: 'No' },
                  { label: 'Mild', value: 'Mild' },
                  { label: 'Moderate', value: 'Moderate' },
                  { label: 'Severe', value: 'Severe' },
                ]}
              />
              <FormField
                label="Additional Notes"
                name="notes"
                type="textarea"
                value={formData.notes}
                onChange={(val) => updateField('notes', val)}
                placeholder="Any additional clinical notes or observations"
              />
            </div>
          )}

          {step === 'review' && (
            <div>
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
                Review Patient Information
              </h2>
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg">
                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Name</p>
                    <p className="font-semibold text-slate-900 dark:text-white">
                      {formData.name}
                    </p>
                  </div>
                  <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg">
                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Age</p>
                    <p className="font-semibold text-slate-900 dark:text-white">
                      {formData.age}
                    </p>
                  </div>
                  <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg">
                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">Severity</p>
                    <p className="font-semibold text-slate-900 dark:text-white">
                      {formData.severity}
                    </p>
                  </div>
                  <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg">
                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">
                      Mutation Type
                    </p>
                    <p className="font-semibold text-slate-900 dark:text-white">
                      {formData.mutation_type}
                    </p>
                  </div>
                </div>
                {errors.submit && (
                  <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-4 rounded-lg">
                    <p className="text-red-700 dark:text-red-400">{errors.submit}</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Navigation Buttons */}
        <div className="flex gap-4">
          <button
            onClick={handleBack}
            disabled={step === 'demographics'}
            className="px-6 py-2 border-2 border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            Back
          </button>
          {step !== 'review' ? (
            <button
              onClick={handleNext}
              className="ml-auto px-6 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition flex items-center gap-2"
            >
              Next <ArrowRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="ml-auto px-6 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {loading ? 'Adding Patient...' : 'Add Patient'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default AddPatient
