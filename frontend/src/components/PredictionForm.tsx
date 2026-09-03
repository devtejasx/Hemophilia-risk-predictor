import React, { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Loader2, ChevronDown, ChevronRight } from 'lucide-react'
import {
  predictionAPI,
  type FeatureSet,
  type Prediction,
  type PredictionSchema,
} from '@/services/api-client'
import { errorMessage } from '@/services/api'
import RiskBadge from '@/components/RiskBadge'

/**
 * The inhibitor-risk input form.
 *
 * Nothing about the fields is hardcoded here. GET /api/predictions/schema
 * reports, for the chosen prediction mode, which fields the served model
 * accepts, a human-readable label for each, whether it belongs to the clinical
 * or the genomic block, whether it is required, and which values it was fitted
 * on. The UI therefore cannot offer a value the model has never seen, and it
 * cannot drift out of step with a retrained model.
 *
 * Fields the schema marks `open_vocabulary` are identifiers or measurements
 * written as text (HGVS notation, an activity reading). Those get a free-text
 * input with the known values as suggestions, because a new patient's mutation
 * has a notation the model has not seen. Everything else is a strict select.
 */

const MODES: { id: FeatureSet; label: string; blurb: string }[] = [
  {
    id: 'merged',
    label: 'Mutation + clinical',
    blurb: 'Uses both blocks. Discriminates best; recommended when you have both.',
  },
  {
    id: 'genomic',
    label: 'Mutation only',
    blurb: 'Only the F8 variant description. Use before assay results are back.',
  },
  {
    id: 'clinical',
    label: 'Clinical only',
    blurb: 'Only the reported clinical record and assay values.',
  },
]

const SECTIONS: { group: 'clinical' | 'genomic'; title: string; blurb: string }[] = [
  {
    group: 'clinical',
    title: 'Patient / clinical information',
    blurb: 'What the clinical record reports for this case.',
  },
  {
    group: 'genomic',
    title: 'Genomic / mutation information',
    blurb: 'How the F8 variant is described.',
  },
]

const control =
  'w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-sm text-slate-900 dark:text-white'

const PredictionForm: React.FC<{ patientId: number; onPredicted?: () => void }> = ({
  patientId,
  onPredicted,
}) => {
  const [mode, setMode] = useState<FeatureSet>('merged')
  const [schema, setSchema] = useState<PredictionSchema | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [showOptional, setShowOptional] = useState(false)

  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [schemaError, setSchemaError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setSchema(null)
    setSchemaError(null)
    predictionAPI
      .schema(mode)
      .then((s) => {
        if (cancelled) return
        setSchema(s)
        // Keep anything the previous mode already collected that this mode also
        // asks for, so switching modes does not wipe the form.
        setValues((previous) => {
          const fields = Object.keys(s.groups)
          return Object.fromEntries(fields.map((f) => [f, previous[f] ?? '']))
        })
      })
      .catch((err) => {
        if (!cancelled) {
          setSchemaError(
            errorMessage(err, 'The prediction model is unavailable right now.')
          )
        }
      })
    return () => {
      cancelled = true
    }
  }, [mode])

  const missingRequired = useMemo(
    () => (schema?.required ?? []).filter((field) => !values[field]?.trim()),
    [schema, values]
  )

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!schema) return
    setError(null)
    setBusy(true)
    setResult(null)
    try {
      // Blank optional fields are omitted, not sent as "". The model records an
      // unmeasured field as unmeasured rather than guessing a value for it.
      const features = Object.fromEntries(
        Object.entries(values).filter(([, v]) => v.trim() !== '')
      )
      const prediction = await predictionAPI.create(patientId, {
        feature_set: mode,
        features,
        mutation_label: values['mut_syn']?.trim() || undefined,
      })
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

  const isOpen = new Set(schema.open_vocabulary)
  const isRequired = new Set(schema.required)

  const field = (name: string) => {
    const label = schema.labels[name] ?? name
    const options = schema.categorical[name]
    const numeric = schema.numeric[name]
    const listId = `${name}-options`

    return (
      <div key={name}>
        <label
          htmlFor={name}
          className="block text-sm mb-1 text-slate-700 dark:text-slate-300"
        >
          {label}
          {!isRequired.has(name) && (
            <span className="text-slate-400"> (optional)</span>
          )}
        </label>

        {numeric ? (
          <input
            id={name}
            type="number"
            step="any"
            className={control}
            value={values[name] ?? ''}
            onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))}
            required={isRequired.has(name)}
          />
        ) : isOpen.has(name) ? (
          <>
            <input
              id={name}
              type="text"
              list={listId}
              className={control}
              value={values[name] ?? ''}
              onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))}
              required={isRequired.has(name)}
            />
            <datalist id={listId}>
              {(options ?? []).slice(0, 200).map((option) => (
                <option key={option} value={option} />
              ))}
            </datalist>
          </>
        ) : (
          <select
            id={name}
            className={control}
            value={values[name] ?? ''}
            onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))}
            required={isRequired.has(name)}
          >
            <option value="">Select</option>
            {(options ?? []).map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        )}

        {numeric?.description && (
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            {numeric.description}
          </p>
        )}
      </div>
    )
  }

  const inGroup = (group: string, optional: boolean) =>
    Object.keys(schema.groups).filter(
      (name) =>
        schema.groups[name] === group && isRequired.has(name) !== optional
    )

  const optionalCount = Object.keys(schema.groups).filter(
    (name) => !isRequired.has(name)
  ).length

  return (
    <form
      onSubmit={submit}
      className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-6"
    >
      <fieldset>
        <legend className="text-sm font-medium text-slate-900 dark:text-white mb-2">
          What information do you have?
        </legend>
        <div className="grid gap-2 sm:grid-cols-3">
          {MODES.filter((m) => schema.available_feature_sets.includes(m.id)).map(
            (m) => (
              <label
                key={m.id}
                className={`cursor-pointer rounded-lg border p-3 text-sm ${
                  mode === m.id
                    ? 'border-purple-500 bg-purple-50 dark:bg-purple-950/40'
                    : 'border-slate-200 dark:border-slate-800'
                }`}
              >
                <input
                  type="radio"
                  name="feature_set"
                  className="sr-only"
                  checked={mode === m.id}
                  onChange={() => setMode(m.id)}
                />
                <span className="block font-medium text-slate-900 dark:text-white">
                  {m.label}
                </span>
                <span className="block text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  {m.blurb}
                </span>
              </label>
            )
          )}
        </div>
      </fieldset>

      {SECTIONS.map((section) => {
        const required = inGroup(section.group, false)
        if (required.length === 0) return null
        return (
          <section key={section.group}>
            <h3 className="text-sm font-medium text-slate-900 dark:text-white">
              {section.title}
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
              {section.blurb}
            </p>
            <div className="grid gap-4 sm:grid-cols-2">{required.map(field)}</div>
          </section>
        )
      })}

      {optionalCount > 0 && (
        <section>
          <button
            type="button"
            onClick={() => setShowOptional((s) => !s)}
            aria-expanded={showOptional}
            className="flex items-center gap-1 text-sm font-medium text-slate-900 dark:text-white"
          >
            {showOptional ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
            Additional details ({optionalCount})
          </button>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Most of these are not reported for most cases. Anything you leave
            blank is recorded as not measured — the model is trained on records
            with the same gaps.
          </p>
          {showOptional && (
            <div className="mt-4 space-y-6">
              {SECTIONS.map((section) => {
                const optional = inGroup(section.group, true)
                if (optional.length === 0) return null
                return (
                  <div key={section.group}>
                    <h4 className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">
                      {section.title}
                    </h4>
                    <div className="grid gap-4 sm:grid-cols-2">
                      {optional.map(field)}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </section>
      )}

      <div className="space-y-3">
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Model {schema.model_version}. Options are limited to values the model
          was fitted on; fields you can type into accept a value it has not seen.
        </p>

        {error && (
          <p role="alert" className="text-sm text-red-600 dark:text-red-400">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy || missingRequired.length > 0}
          className="inline-flex items-center gap-2 rounded-lg bg-purple-600 hover:bg-purple-700 disabled:opacity-50 px-4 py-2 text-white text-sm font-medium"
        >
          {busy && <Loader2 className="w-4 h-4 animate-spin" />} Estimate
          inhibitor risk
        </button>
      </div>

      {result && (
        <div className="rounded-lg border border-slate-200 dark:border-slate-800 p-4 bg-slate-50 dark:bg-slate-950">
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

export default PredictionForm
