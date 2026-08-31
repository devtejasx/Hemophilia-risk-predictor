import React from 'react'
import { User, Mail, Phone, MapPin, AlertCircle } from 'lucide-react'
import clsx from 'clsx'

interface PatientCardProps {
  id: string
  name: string
  age: number
  severity: string
  mutation_type: string
  blood_type: string
  email?: string
  phone?: string
  risk_level?: 'low' | 'medium' | 'high'
  onClick?: () => void
  isSelected?: boolean
}

const severityColors = {
  mild: 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400',
  moderate: 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400',
  severe: 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400',
}

const riskColors = {
  low: 'text-green-600 dark:text-green-400',
  medium: 'text-yellow-600 dark:text-yellow-400',
  high: 'text-red-600 dark:text-red-400',
}

export const PatientCard: React.FC<PatientCardProps> = ({
  id,
  name,
  age,
  severity,
  mutation_type,
  blood_type,
  email,
  phone,
  risk_level,
  onClick,
  isSelected = false,
}) => {
  return (
    <div
      onClick={onClick}
      className={clsx(
        'bg-white dark:bg-slate-900 rounded-lg p-5 border-2 transition cursor-pointer hover:shadow-lg',
        isSelected
          ? 'border-purple-500 dark:border-purple-400 shadow-lg'
          : 'border-slate-200 dark:border-slate-800 hover:border-purple-200 dark:hover:border-purple-800'
      )}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-purple-400 to-purple-600 rounded-full flex items-center justify-center">
            <User className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-900 dark:text-white">{name}</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400">{age} years old</p>
          </div>
        </div>
        {risk_level && (
          <span className={clsx('text-lg font-bold', riskColors[risk_level])}>
            {risk_level.toUpperCase()}
          </span>
        )}
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        <span
          className={clsx(
            'text-xs font-medium px-3 py-1 rounded-full',
            severityColors[severity as keyof typeof severityColors] ||
              'bg-gray-100 dark:bg-gray-800'
          )}
        >
          {severity}
        </span>
        <span className="text-xs font-medium px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400">
          {mutation_type}
        </span>
        <span className="text-xs font-medium px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
          {blood_type}
        </span>
      </div>

      {(email || phone) && (
        <div className="space-y-1 text-sm text-slate-600 dark:text-slate-400">
          {email && (
            <div className="flex items-center gap-2">
              <Mail className="w-4 h-4" />
              <span>{email}</span>
            </div>
          )}
          {phone && (
            <div className="flex items-center gap-2">
              <Phone className="w-4 h-4" />
              <span>{phone}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default PatientCard
