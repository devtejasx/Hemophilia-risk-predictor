import React, { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import {
  predictionAPI,
  type GenomicInput,
  type Prediction,
  type PredictionSchema,
} from '@/services/api-client'
import { errorMessage } from '@/services/api'
import RiskBadge from '@/components/RiskBadge'

/**
 * CHAMP genomic input.
 *
 * The select options come from GET /api/predictions/schema, which reports the
 * category vocabulary the preprocessor was actually fitted on. The UI therefore
 * cannot offer a value the model has never seen, and the backend rejects one
 * anyway if it somehow arrives.
 */
const GenomicForm: React.FC<{ patientId: number; onPredicted?: () => void }> = ({
  patientId,
  onPredicted,
}) => {
  const [schema, setSchema] = useState<PredictionSchema | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [exon, setExon] = useState('')
  const [codon, setCodon] = useState('')
  const [isIntron, setIsIntron] = useState(false)

  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [schemaError, setSchemaError] = useState<string | null>(null)

  useEffect(() => {
    predictionAPI
      .schema()
      .then((s) => {
        setSchema(s)
        // Preselect nothing: an empty select forces a deliberate choice.
        setValues(Object.fromEntries(Object.keys(s.categorical).map((k) => [k, ''])))
      })
      .catch((err) =>
        setSchemaError(
          errorMessage(err, 'The prediction model is unavailable right now.')
        )
      )
  }, [])

  const complete = useMemo(
    () => schema?.required.every((field) => values[field]) ?? false,
    [schema, values]
  )

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!schema) return
    setError(null)
    setBusy(true)
    setResult(null)
    try {
      const payload = {
        ...(values as unknown as GenomicInput),
        exon_number: exon === '' ? null : Number(exon),
        codon_number: codon === '' ? null : Number(codon),
        is_intron: isIntron,
      }
      const prediction = await predictionAPI.create(patientId, payload)
      setResult(prediction)
      onPredicted?.()
    } catch (err) {
      setError(errorMessage(err, 'The estimate could not be produced.'))
    } finally {
      setBusy(false)
    }
  }

  if (schemaError) {
    return (
      <p role="alert" className="text-sm text-red-600 dark:text-red-400">
        {schemaError}
      </p>
    )
  }

  if (!schema) {
    return (
      <div className="flex items-center gap-2 text-slate-500 text-sm">
        <Loader2 className="w-4 h-4 animate-spin" /> Loading model schema
      </div>
    )
  }

  const control =
    'w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-sm text-slate-900 dark:text-white'

  return (
    <form
      onSubmit={submit}
      className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-4"
    >
      <div className="grid gap-4 sm:grid-cols-2">
        {Object.entries(schema.categorical).map(([field, options]) => (
          <div key={field}>
            <label
              htmlFor={field}
              className="block text-sm mb-1 text-slate-700 dark:text-slate-300"
            >
              {field}
            </label>
            <select
              id={field}
              className={control}
              value={values[field] ?? ''}
              onChange={(e) => setValues((v) => ({ ...v, [field]: e.target.value }))}
              required
            >
              <option value="">Select</option>
              {options.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
        ))}

        <div>
          <label htmlFor="exon_number" className="block text-sm mb-1 text-slate-700 dark:text-slate-300">
            Exon / intron number <span className="text-slate-400">(optional)</span>
          </label>
          <input
            id="exon_number"
            type="number"
            min={0}
            max={200}
            className={control}
            value={exon}
            onChange={(e) => setExon(e.target.value)}
          />
        </div>

        <div>
          <label htmlFor="codon_number" className="block text-sm mb-1 text-slate-700 dark:text-slate-300">
            Codon number <span className="text-slate-400">(optional)</span>
          </label>
          <input
            id="codon_number"
            type="number"
            min={0}
            max={10000}
            className={control}
            value={codon}
            onChange={(e) => setCodon(e.target.value)}
          />
        </div>
      </div>

      <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300">
        <input
          type="checkbox"
          checked={isIntron}
          onChange={(e) => setIsIntron(e.target.checked)}
          className="rounded border-slate-300 dark:border-slate-700"
        />
        Variant lies in an intron rather than an exon
      </label>

      <p className="text-xs text-slate-500 dark:text-slate-400">
        Optional numeric fields are imputed with the training median when left blank.
        Model version {schema.model_version}.
      </p>

      {error && (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={busy || !complete}
        className="inline-flex items-center gap-2 rounded-lg bg-purple-600 hover:bg-purple-700 disabled:opacity-50 px-4 py-2 text-white text-sm font-medium"
      >
        {busy && <Loader2 className="w-4 h-4 animate-spin" />} Estimate risk
      </button>

      {result && (
        <div className="mt-2 rounded-lg border border-slate-200 dark:border-slate-800 p-4 bg-slate-50 dark:bg-slate-950">
          <RiskBadge
            probability={result.probability}
            category={result.risk_category}
            threshold={result.threshold}
            size="lg"
          />
          <p className="mt-3 text-sm text-slate-700 dark:text-slate-300">
            {result.interpretation}
          </p>
          <Link
            to={`/explanations/${result.id}`}
            className="inline-block mt-3 text-sm text-purple-600 dark:text-purple-400 hover:underline"
          >
            Why did the model say this?
          </Link>
        </div>
      )}
    </form>
  )
}

export default GenomicForm
