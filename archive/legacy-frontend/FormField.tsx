import React from 'react'
import clsx from 'clsx'

interface FormFieldProps {
  label: string
  name: string
  type?: 'text' | 'number' | 'email' | 'tel' | 'select' | 'textarea'
  value: string | number
  onChange: (value: string | number) => void
  options?: Array<{ label: string; value: string }>
  required?: boolean
  error?: string
  placeholder?: string
  disabled?: boolean
}

export const FormField: React.FC<FormFieldProps> = ({
  label,
  name,
  type = 'text',
  value,
  onChange,
  options,
  required = false,
  error,
  placeholder,
  disabled = false,
}) => {
  return (
    <div className="mb-4">
      <label className="block text-sm font-medium text-slate-900 dark:text-white mb-2">
        {label}
        {required && <span className="text-red-500">*</span>}
      </label>

      {type === 'select' ? (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={clsx(
            'w-full px-4 py-2 rounded-lg border-2 transition focus:outline-none',
            error
              ? 'border-red-500 bg-red-50 dark:bg-red-900/10'
              : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 focus:border-purple-500'
          )}
        >
          <option value="">Select {label.toLowerCase()}</option>
          {options?.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      ) : type === 'textarea' ? (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          rows={4}
          className={clsx(
            'w-full px-4 py-2 rounded-lg border-2 transition focus:outline-none resize-none',
            error
              ? 'border-red-500 bg-red-50 dark:bg-red-900/10'
              : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 focus:border-purple-500'
          )}
        />
      ) : (
        <input
          type={type}
          value={value}
          onChange={(e) => onChange(type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className={clsx(
            'w-full px-4 py-2 rounded-lg border-2 transition focus:outline-none',
            error
              ? 'border-red-500 bg-red-50 dark:bg-red-900/10'
              : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 focus:border-purple-500'
          )}
        />
      )}

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  )
}

export default FormField
