import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { patientAPI } from '@/services/api-client'
import { errorMessage } from '@/services/api'

/**
 * Patient identity only.
 *
 * An earlier version of this form collected around twenty clinical fields
 * (blood type, joint damage score, adherence, HLA typing). None of them exist
 * in the dataset or reach the model, so collecting them implied a clinical
 * model that does not exist. The predictive input is the F8 mutation and the
 * clinical findings reported for it, captured on the prediction screen.
 */
const AddPatient: React.FC = () => {
  const navigate = useNavigate()
  const [identifier, setIdentifier] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [notes, setNotes] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)
    setBusy(true)
    try {
      const patient = await patientAPI.create({
        identifier,
        display_name: displayName,
        notes: notes || undefined,
      })
      navigate(`/patients/${patient.id}`)
    } catch (err) {
      setError(errorMessage(err, 'Could not create the patient.'))
    } finally {
      setBusy(false)
    }
  }

  const field =
    'w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-sm text-slate-900 dark:text-white'

  return (
    <div className="p-8 max-w-xl">
      <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-1">Add patient</h1>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
        Identity only. The F8 mutation and the clinical findings reported for it
        are entered when you run an estimate.
      </p>

      <form
        onSubmit={submit}
        className="space-y-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6"
      >
        <div>
          <label
            htmlFor="identifier"
            className="block text-sm mb-1 text-slate-700 dark:text-slate-300"
          >
            Reference
          </label>
          <input
            id="identifier"
            className={field}
            value={identifier}
            onChange={(e) => setIdentifier(e.target.value)}
            placeholder="e.g. MRN-1042"
            required
          />
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Your own identifier. Must be unique within your account.
          </p>
        </div>

        <div>
          <label
            htmlFor="display_name"
            className="block text-sm mb-1 text-slate-700 dark:text-slate-300"
          >
            Display name
          </label>
          <input
            id="display_name"
            className={field}
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            required
          />
        </div>

        <div>
          <label
            htmlFor="notes"
            className="block text-sm mb-1 text-slate-700 dark:text-slate-300"
          >
            Notes <span className="text-slate-400">(optional)</span>
          </label>
          <textarea
            id="notes"
            className={field}
            rows={3}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        </div>

        {error && (
          <p role="alert" className="text-sm text-red-600 dark:text-red-400">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="inline-flex items-center gap-2 rounded-lg bg-purple-600 hover:bg-purple-700 disabled:opacity-60 px-4 py-2 text-white text-sm font-medium"
        >
          {busy && <Loader2 className="w-4 h-4 animate-spin" />} Create patient
        </button>
      </form>
    </div>
  )
}

export default AddPatient
