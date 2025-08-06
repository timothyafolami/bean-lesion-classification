import React from 'react'
import { CheckCircle, AlertCircle, Loader2, Upload, Zap, Clock } from 'lucide-react'
import { cn, formatBytes, formatDuration } from '../lib/utils'
import { UploadProgress } from '../store/useAppStore'

interface ProgressTrackerProps {
  uploadProgress: UploadProgress[]
  isUploading: boolean
  className?: string
}

export const ProgressTracker: React.FC<ProgressTrackerProps> = ({
  uploadProgress,
  isUploading,
  className,
}) => {
  if (uploadProgress.length === 0 && !isUploading) {
    return null
  }

  const completedCount = uploadProgress.filter(p => p.status === 'completed').length
  const errorCount = uploadProgress.filter(p => p.status === 'error').length
  const processingCount = uploadProgress.filter(p => p.status === 'processing').length
  const uploadingCount = uploadProgress.filter(p => p.status === 'uploading').length

  const overallProgress = uploadProgress.length > 0 
    ? (completedCount / uploadProgress.length) * 100 
    : 0

  return (
    <div className={cn('card p-6 space-y-4', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">
          Upload Progress
        </h3>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          {isUploading && <Loader2 className="h-4 w-4 animate-spin" />}
          <span>
            {completedCount} of {uploadProgress.length} completed
          </span>
        </div>
      </div>

      {/* Overall Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Overall Progress</span>
          <span className="font-medium text-gray-900">
            {overallProgress.toFixed(0)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className="bg-primary-600 h-3 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
      </div>

      {/* Status Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="flex items-center gap-2">
          <Upload className="h-4 w-4 text-secondary-600" />
          <div className="text-sm">
            <p className="font-medium text-gray-900">{uploadingCount}</p>
            <p className="text-gray-600">Uploading</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-blue-600" />
          <div className="text-sm">
            <p className="font-medium text-gray-900">{processingCount}</p>
            <p className="text-gray-600">Processing</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <CheckCircle className="h-4 w-4 text-primary-600" />
          <div className="text-sm">
            <p className="font-medium text-gray-900">{completedCount}</p>
            <p className="text-gray-600">Completed</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-danger-600" />
          <div className="text-sm">
            <p className="font-medium text-gray-900">{errorCount}</p>
            <p className="text-gray-600">Failed</p>
          </div>
        </div>
      </div>

      {/* Individual File Progress */}
      <div className="space-y-3 max-h-64 overflow-y-auto">
        {uploadProgress.map((progress) => (
          <ProgressItem key={progress.fileId} progress={progress} />
        ))}
      </div>
    </div>
  )
}

interface ProgressItemProps {
  progress: UploadProgress
}

const ProgressItem: React.FC<ProgressItemProps> = ({ progress }) => {
  const getStatusIcon = () => {
    switch (progress.status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-primary-600" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-danger-600" />
      case 'uploading':
        return <Upload className="h-4 w-4 text-secondary-600" />
      case 'processing':
        return <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />
      default:
        return <Clock className="h-4 w-4 text-gray-400" />
    }
  }

  const getStatusColor = () => {
    switch (progress.status) {
      case 'completed':
        return 'text-primary-600'
      case 'error':
        return 'text-danger-600'
      case 'uploading':
        return 'text-secondary-600'
      case 'processing':
        return 'text-blue-600'
      default:
        return 'text-gray-500'
    }
  }

  const getProgressBarColor = () => {
    switch (progress.status) {
      case 'completed':
        return 'bg-primary-600'
      case 'error':
        return 'bg-danger-600'
      case 'uploading':
        return 'bg-secondary-600'
      case 'processing':
        return 'bg-blue-600'
      default:
        return 'bg-gray-400'
    }
  }

  const getStatusText = () => {
    switch (progress.status) {
      case 'pending':
        return 'Waiting...'
      case 'uploading':
        return `Uploading... ${progress.progress}%`
      case 'processing':
        return 'Processing...'
      case 'completed':
        return 'Completed'
      case 'error':
        return progress.error || 'Failed'
      default:
        return 'Unknown'
    }
  }

  return (
    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
      {/* Status Icon */}
      <div className="flex-shrink-0">
        {getStatusIcon()}
      </div>

      {/* File Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <p className="text-sm font-medium text-gray-900 truncate">
            {progress.fileName}
          </p>
          <span className={cn('text-xs font-medium', getStatusColor())}>
            {getStatusText()}
          </span>
        </div>

        {/* Progress Bar */}
        {(progress.status === 'uploading' || progress.status === 'processing') && (
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div
              className={cn('h-1.5 rounded-full transition-all duration-300', getProgressBarColor())}
              style={{ 
                width: `${progress.status === 'processing' ? 100 : progress.progress}%` 
              }}
            />
          </div>
        )}

        {/* Error Message */}
        {progress.status === 'error' && progress.error && (
          <p className="text-xs text-danger-600 mt-1">
            {progress.error}
          </p>
        )}
      </div>
    </div>
  )
}