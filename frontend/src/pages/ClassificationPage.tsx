import React, { useState, useCallback } from 'react'
import { Upload, Image as ImageIcon, BarChart3, Wifi, WifiOff, RefreshCw } from 'lucide-react'
import { ImageUploader, UploadedFile } from '../components/ImageUploader'
import { ResultsDisplay } from '../components/ResultsDisplay'
import { ProgressTracker } from '../components/ProgressTracker'
import { LoadingState } from '../components/LoadingSpinner'
import { ToastContainer, useToast } from '../components/Toast'
import { useResultsState, useUploadState } from '../store/useAppStore'
import { useFileUpload, useConnectionStatus, useRetry } from '../hooks/useApi'
import { ClassificationResult } from '../lib/api'
import { generateId } from '../lib/utils'

export const ClassificationPage: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<UploadedFile[]>([])
  
  const { results, addResults, selectedResult, setSelectedResult } = useResultsState()
  const { uploadProgress, isUploading, setUploadProgress, updateUploadProgress, setIsUploading } = useUploadState()
  const { uploadSingle, uploadBatch } = useFileUpload()
  const { isOnline, isOffline, health } = useConnectionStatus()
  const { retryAll } = useRetry()
  const toast = useToast()

  const handleFilesSelected = useCallback((files: UploadedFile[]) => {
    setSelectedFiles(files)
  }, [])

  const processFiles = useCallback(async (files: UploadedFile[]) => {
    if (files.length === 0) return

    setIsUploading(true)
    
    // Initialize progress tracking
    const progressItems = files.map(file => ({
      fileId: file.id,
      fileName: file.file.name,
      progress: 0,
      status: 'pending' as const,
    }))
    setUploadProgress(progressItems)

    try {
      if (files.length === 1) {
        // Single file upload
        const file = files[0]
        
        // Update progress to uploading
        updateUploadProgress(file.id, { status: 'uploading', progress: 50 })
        
        const result = await uploadSingle.mutateAsync({
          file: file.file,
          returnProbabilities: true,
        })
        
        // Update progress to processing
        updateUploadProgress(file.id, { status: 'processing', progress: 75 })
        
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 500))
        
        // Complete
        updateUploadProgress(file.id, { status: 'completed', progress: 100 })
        
        // Add result with filename
        const resultWithFilename = {
          ...result.result,
          filename: file.file.name,
        }
        addResults([resultWithFilename])
        
      } else {
        // Batch upload
        const fileList = files.map(f => f.file)
        
        // Update all to uploading
        files.forEach(file => {
          updateUploadProgress(file.id, { status: 'uploading', progress: 30 })
        })
        
        const result = await uploadBatch.mutateAsync({
          files: fileList,
          returnProbabilities: true,
        })
        
        // Update all to processing
        files.forEach(file => {
          updateUploadProgress(file.id, { status: 'processing', progress: 70 })
        })
        
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        // Complete all
        files.forEach(file => {
          updateUploadProgress(file.id, { status: 'completed', progress: 100 })
        })
        
        // Add results with filenames
        const resultsWithFilenames = result.results.map((res, index) => ({
          ...res,
          filename: files[index]?.file.name || `image_${index + 1}`,
        }))
        addResults(resultsWithFilenames)
      }
      
      toast.success(
        'Classification Complete',
        `Successfully processed ${files.length} image${files.length > 1 ? 's' : ''}`
      )
      
    } catch (error: any) {
      // Mark all as failed
      files.forEach(file => {
        updateUploadProgress(file.id, {
          status: 'error',
          progress: 0,
          error: error.message || 'Classification failed',
        })
      })
      
      toast.error(
        'Classification Failed',
        error.response?.data?.detail || error.message || 'Please try again'
      )
    } finally {
      setIsUploading(false)
    }
  }, [uploadSingle, uploadBatch, updateUploadProgress, setUploadProgress, setIsUploading, addResults, toast])

  const handleClassify = useCallback(() => {
    const pendingFiles = selectedFiles.filter(f => f.status === 'pending')
    if (pendingFiles.length > 0) {
      processFiles(pendingFiles)
    }
  }, [selectedFiles, processFiles])

  const handleResultSelect = useCallback((result: ClassificationResult) => {
    setSelectedResult(result)
    setSelectedResult(result)
  }, [setSelectedResult])

  return (
    <div className="space-y-8">
      {/* Toast Container */}
      <ToastContainer toasts={toast.toasts} onClose={toast.removeToast} />
      
      {/* Connection Status */}
      {isOffline && (
        <div className="bg-danger-50 border border-danger-200 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <WifiOff className="h-5 w-5 text-danger-600" />
            <div className="flex-1">
              <h4 className="text-sm font-medium text-danger-800">
                Connection Lost
              </h4>
              <p className="text-sm text-danger-700">
                Unable to connect to the classification service. Please check your connection.
              </p>
            </div>
            <button
              onClick={retryAll}
              className="btn-secondary text-sm flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Retry
            </button>
          </div>
        </div>
      )}

      {isOnline && health && (
        <div className="bg-primary-50 border border-primary-200 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <Wifi className="h-5 w-5 text-primary-600" />
            <div className="flex-1">
              <h4 className="text-sm font-medium text-primary-800">
                Service Online
              </h4>
              <p className="text-sm text-primary-700">
                Classification service is running and ready to process images.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900">
          Image Classification
        </h1>
        <p className="mt-2 text-gray-600">
          Upload bean leaf images to detect diseases using AI
        </p>
      </div>

      {/* Upload Area */}
      <ImageUploader
        onFilesSelected={handleFilesSelected}
        maxFiles={50}
        disabled={isUploading || isOffline}
      />

      {/* Classify Button */}
      {selectedFiles.length > 0 && (
        <div className="flex justify-center">
          <button
            onClick={handleClassify}
            disabled={isUploading || isOffline || selectedFiles.filter(f => f.status === 'pending').length === 0}
            className="btn-primary text-lg px-8 py-3 flex items-center gap-2"
          >
            <BarChart3 className="h-5 w-5" />
            {isUploading ? 'Classifying...' : `Classify ${selectedFiles.filter(f => f.status === 'pending').length} Image${selectedFiles.filter(f => f.status === 'pending').length !== 1 ? 's' : ''}`}
          </button>
        </div>
      )}

      {/* Progress Tracker */}
      <ProgressTracker
        uploadProgress={uploadProgress}
        isUploading={isUploading}
      />

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-4">
            <ImageIcon className="h-8 w-8 text-primary-600" />
            <h3 className="text-lg font-semibold text-gray-900">
              Single & Batch Processing
            </h3>
          </div>
          <p className="text-gray-600">
            Upload single images for quick analysis or multiple images for batch processing 
            with optimized performance.
          </p>
        </div>

        <div className="card p-6">
          <div className="flex items-center gap-3 mb-4">
            <BarChart3 className="h-8 w-8 text-primary-600" />
            <h3 className="text-lg font-semibold text-gray-900">
              Detailed Analysis
            </h3>
          </div>
          <p className="text-gray-600">
            Get confidence scores, probability distributions, and image quality 
            metrics for comprehensive disease assessment.
          </p>
        </div>
      </div>

      {/* Results Display */}
      <ResultsDisplay
        results={results}
        selectedResult={selectedResult}
        onResultSelect={handleResultSelect}
      />
    </div>
  )
}