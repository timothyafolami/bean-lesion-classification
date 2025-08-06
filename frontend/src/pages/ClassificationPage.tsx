import React from 'react'
import { Upload, Image as ImageIcon, BarChart3 } from 'lucide-react'

export const ClassificationPage: React.FC = () => {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900">
          Image Classification
        </h1>
        <p className="mt-2 text-gray-600">
          Upload bean leaf images to detect diseases using AI
        </p>
      </div>

      {/* Upload Area - Placeholder */}
      <div className="card p-8">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
          <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Upload Images
          </h3>
          <p className="text-gray-600 mb-4">
            Drag and drop your bean leaf images here, or click to browse
          </p>
          <button className="btn-primary">
            Choose Files
          </button>
          <p className="text-sm text-gray-500 mt-2">
            Supports JPEG, PNG, WebP up to 10MB
          </p>
        </div>
      </div>

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

      {/* Results Area - Placeholder */}
      <div className="card p-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Classification Results
        </h3>
        <div className="text-center py-12 text-gray-500">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 text-gray-300" />
          <p>Results will appear here after uploading images</p>
        </div>
      </div>
    </div>
  )
}