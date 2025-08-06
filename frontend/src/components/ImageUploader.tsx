import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, X, Image as ImageIcon, AlertCircle, CheckCircle } from 'lucide-react'
import { cn, formatBytes, generateId, isValidImageFile, getImageDimensions } from '../lib/utils'

export interface UploadedFile {
  id: string
  file: File
  preview: string
  status: 'pending' | 'uploading' | 'completed' | 'error'
  progress: number
  error?: string
  dimensions?: { width: number; height: number }
}

interface ImageUploaderProps {
  onFilesSelected: (files: UploadedFile[]) => void
  maxFiles?: number
  maxFileSize?: number
  acceptedTypes?: string[]
  disabled?: boolean
  className?: string
}

const DEFAULT_MAX_FILES = 10
const DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB
const DEFAULT_ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onFilesSelected,
  maxFiles = DEFAULT_MAX_FILES,
  maxFileSize = DEFAULT_MAX_FILE_SIZE,
  acceptedTypes = DEFAULT_ACCEPTED_TYPES,
  disabled = false,
  className,
}) => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [errors, setErrors] = useState<string[]>([])

  const processFiles = useCallback(async (files: File[]) => {
    const newErrors: string[] = []
    const validFiles: UploadedFile[] = []

    // Check total file count
    if (uploadedFiles.length + files.length > maxFiles) {
      newErrors.push(`Maximum ${maxFiles} files allowed`)
      setErrors(newErrors)
      return
    }

    for (const file of files) {
      // Validate file type
      if (!isValidImageFile(file) || !acceptedTypes.includes(file.type)) {
        newErrors.push(`${file.name}: Invalid file type. Supported: JPEG, PNG, WebP`)
        continue
      }

      // Validate file size
      if (file.size > maxFileSize) {
        newErrors.push(`${file.name}: File too large. Maximum ${formatBytes(maxFileSize)}`)
        continue
      }

      // Create preview URL
      const preview = URL.createObjectURL(file)
      
      try {
        // Get image dimensions
        const dimensions = await getImageDimensions(file)
        
        const uploadedFile: UploadedFile = {
          id: generateId(),
          file,
          preview,
          status: 'pending',
          progress: 0,
          dimensions,
        }
        
        validFiles.push(uploadedFile)
      } catch (error) {
        newErrors.push(`${file.name}: Failed to process image`)
        URL.revokeObjectURL(preview)
      }
    }

    if (newErrors.length > 0) {
      setErrors(newErrors)
    } else {
      setErrors([])
    }

    if (validFiles.length > 0) {
      const updatedFiles = [...uploadedFiles, ...validFiles]
      setUploadedFiles(updatedFiles)
      onFilesSelected(updatedFiles)
    }
  }, [uploadedFiles, maxFiles, maxFileSize, acceptedTypes, onFilesSelected])

  const onDrop = useCallback((acceptedFiles: File[]) => {
    processFiles(acceptedFiles)
  }, [processFiles])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'image/*': acceptedTypes.map(type => type.replace('image/', '.')),
    },
    maxSize: maxFileSize,
    disabled,
    multiple: true,
  })

  const removeFile = useCallback((fileId: string) => {
    setUploadedFiles(prev => {
      const file = prev.find(f => f.id === fileId)
      if (file) {
        URL.revokeObjectURL(file.preview)
      }
      const updated = prev.filter(f => f.id !== fileId)
      onFilesSelected(updated)
      return updated
    })
  }, [onFilesSelected])

  const clearAll = useCallback(() => {
    uploadedFiles.forEach(file => {
      URL.revokeObjectURL(file.preview)
    })
    setUploadedFiles([])
    setErrors([])
    onFilesSelected([])
  }, [uploadedFiles, onFilesSelected])

  // Cleanup preview URLs on unmount
  React.useEffect(() => {
    return () => {
      uploadedFiles.forEach(file => {
        URL.revokeObjectURL(file.preview)
      })
    }
  }, [])

  return (
    <div className={cn('space-y-4', className)}>
      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={cn(
          'border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer',
          isDragActive && !isDragReject && 'border-primary-500 bg-primary-50',
          isDragReject && 'border-danger-500 bg-danger-50',
          !isDragActive && !isDragReject && 'border-gray-300 hover:border-gray-400',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center gap-4">
          <div className={cn(
            'p-3 rounded-full',
            isDragActive && !isDragReject && 'bg-primary-100',
            isDragReject && 'bg-danger-100',
            !isDragActive && !isDragReject && 'bg-gray-100'
          )}>
            <Upload className={cn(
              'h-8 w-8',
              isDragActive && !isDragReject && 'text-primary-600',
              isDragReject && 'text-danger-600',
              !isDragActive && !isDragReject && 'text-gray-400'
            )} />
          </div>
          
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              {isDragActive ? 'Drop images here' : 'Upload Images'}
            </h3>
            <p className="text-gray-600 mb-4">
              {isDragActive 
                ? 'Release to upload your images'
                : 'Drag and drop your bean leaf images here, or click to browse'
              }
            </p>
            
            {!isDragActive && (
              <button
                type="button"
                className="btn-primary"
                disabled={disabled}
              >
                Choose Files
              </button>
            )}
          </div>
        </div>
        
        <div className="mt-4 text-sm text-gray-500">
          <p>Supports JPEG, PNG, WebP up to {formatBytes(maxFileSize)}</p>
          <p>Maximum {maxFiles} files</p>
        </div>
      </div>

      {/* Error Messages */}
      {errors.length > 0 && (
        <div className="bg-danger-50 border border-danger-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-danger-600 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h4 className="text-sm font-medium text-danger-800 mb-2">
                Upload Errors
              </h4>
              <ul className="text-sm text-danger-700 space-y-1">
                {errors.map((error, index) => (
                  <li key={index}>• {error}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* File List */}
      {uploadedFiles.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-lg font-medium text-gray-900">
              Selected Images ({uploadedFiles.length})
            </h4>
            <button
              onClick={clearAll}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Clear All
            </button>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {uploadedFiles.map((uploadedFile) => (
              <FilePreview
                key={uploadedFile.id}
                uploadedFile={uploadedFile}
                onRemove={removeFile}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

interface FilePreviewProps {
  uploadedFile: UploadedFile
  onRemove: (fileId: string) => void
}

const FilePreview: React.FC<FilePreviewProps> = ({ uploadedFile, onRemove }) => {
  const { id, file, preview, status, progress, error, dimensions } = uploadedFile

  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-primary-600" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-danger-600" />
      default:
        return <ImageIcon className="h-4 w-4 text-gray-400" />
    }
  }

  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'border-primary-200 bg-primary-50'
      case 'error':
        return 'border-danger-200 bg-danger-50'
      case 'uploading':
        return 'border-secondary-200 bg-secondary-50'
      default:
        return 'border-gray-200 bg-white'
    }
  }

  return (
    <div className={cn('card p-4 relative', getStatusColor())}>
      {/* Remove Button */}
      <button
        onClick={() => onRemove(id)}
        className="absolute top-2 right-2 p-1 rounded-full bg-white shadow-sm hover:bg-gray-50 transition-colors"
      >
        <X className="h-4 w-4 text-gray-400" />
      </button>

      {/* Image Preview */}
      <div className="aspect-square mb-3 rounded-lg overflow-hidden bg-gray-100">
        <img
          src={preview}
          alt={file.name}
          className="w-full h-full object-cover"
        />
      </div>

      {/* File Info */}
      <div className="space-y-2">
        <div className="flex items-start gap-2">
          {getStatusIcon()}
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">
              {file.name}
            </p>
            <p className="text-xs text-gray-500">
              {formatBytes(file.size)}
              {dimensions && (
                <span> • {dimensions.width}×{dimensions.height}</span>
              )}
            </p>
          </div>
        </div>

        {/* Progress Bar */}
        {status === 'uploading' && (
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-secondary-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        {/* Error Message */}
        {status === 'error' && error && (
          <p className="text-xs text-danger-600">{error}</p>
        )}

        {/* Status Text */}
        <p className="text-xs text-gray-500">
          {status === 'pending' && 'Ready to upload'}
          {status === 'uploading' && `Uploading... ${progress}%`}
          {status === 'completed' && 'Upload complete'}
          {status === 'error' && 'Upload failed'}
        </p>
      </div>
    </div>
  )
}