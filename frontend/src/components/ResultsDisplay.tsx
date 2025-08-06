import React, { useState, useMemo } from 'react'
import { 
  BarChart3, 
  Download, 
  Eye, 
  Filter, 
  SortAsc, 
  SortDesc, 
  Image as ImageIcon,
  Clock,
  Zap,
  CheckCircle,
  AlertTriangle,
  Info
} from 'lucide-react'
import { ClassificationResult } from '../lib/api'
import { cn, formatPercentage, formatDuration, formatBytes } from '../lib/utils'

interface ResultsDisplayProps {
  results: ClassificationResult[]
  onResultSelect?: (result: ClassificationResult) => void
  selectedResult?: ClassificationResult | null
  className?: string
}

type SortField = 'confidence' | 'class_name' | 'processing_time' | 'filename'
type SortDirection = 'asc' | 'desc'
type FilterClass = 'all' | 'healthy' | 'angular_leaf_spot' | 'bean_rust'

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  results,
  onResultSelect,
  selectedResult,
  className,
}) => {
  const [sortField, setSortField] = useState<SortField>('confidence')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [filterClass, setFilterClass] = useState<FilterClass>('all')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')

  // Filter and sort results
  const filteredAndSortedResults = useMemo(() => {
    let filtered = results

    // Apply class filter
    if (filterClass !== 'all') {
      filtered = filtered.filter(result => result.class_name === filterClass)
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue: any
      let bValue: any

      switch (sortField) {
        case 'confidence':
          aValue = a.confidence
          bValue = b.confidence
          break
        case 'class_name':
          aValue = a.class_name
          bValue = b.class_name
          break
        case 'processing_time':
          aValue = a.processing_time
          bValue = b.processing_time
          break
        case 'filename':
          aValue = a.filename || ''
          bValue = b.filename || ''
          break
        default:
          return 0
      }

      if (typeof aValue === 'string') {
        aValue = aValue.toLowerCase()
        bValue = bValue.toLowerCase()
      }

      if (sortDirection === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0
      }
    })

    return filtered
  }, [results, filterClass, sortField, sortDirection])

  // Calculate statistics
  const stats = useMemo(() => {
    if (results.length === 0) return null

    const classCount = results.reduce((acc, result) => {
      acc[result.class_name] = (acc[result.class_name] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const avgConfidence = results.reduce((sum, result) => sum + result.confidence, 0) / results.length
    const avgProcessingTime = results.reduce((sum, result) => sum + result.processing_time, 0) / results.length

    return {
      total: results.length,
      classCount,
      avgConfidence,
      avgProcessingTime,
    }
  }, [results])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('desc')
    }
  }

  const exportResults = () => {
    const dataStr = JSON.stringify(results, null, 2)
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr)
    
    const exportFileDefaultName = `classification_results_${new Date().toISOString().split('T')[0]}.json`
    
    const linkElement = document.createElement('a')
    linkElement.setAttribute('href', dataUri)
    linkElement.setAttribute('download', exportFileDefaultName)
    linkElement.click()
  }

  const exportCSV = () => {
    const headers = ['filename', 'class_name', 'confidence', 'processing_time']
    const csvContent = [
      headers.join(','),
      ...results.map(result => [
        result.filename || '',
        result.class_name,
        result.confidence.toFixed(4),
        result.processing_time.toFixed(4)
      ].join(','))
    ].join('\n')

    const dataUri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvContent)
    const exportFileDefaultName = `classification_results_${new Date().toISOString().split('T')[0]}.csv`
    
    const linkElement = document.createElement('a')
    linkElement.setAttribute('href', dataUri)
    linkElement.setAttribute('download', exportFileDefaultName)
    linkElement.click()
  }

  if (results.length === 0) {
    return (
      <div className={cn('card p-8 text-center', className)}>
        <BarChart3 className="h-12 w-12 mx-auto mb-4 text-gray-300" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          No Results Yet
        </h3>
        <p className="text-gray-600">
          Upload and classify images to see results here
        </p>
      </div>
    )
  }

  return (
    <div className={cn('space-y-6', className)}>
      {/* Statistics Overview */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="card p-4">
            <div className="flex items-center gap-3">
              <ImageIcon className="h-8 w-8 text-primary-600" />
              <div>
                <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
                <p className="text-sm text-gray-600">Total Images</p>
              </div>
            </div>
          </div>
          
          <div className="card p-4">
            <div className="flex items-center gap-3">
              <CheckCircle className="h-8 w-8 text-primary-600" />
              <div>
                <p className="text-2xl font-bold text-gray-900">
                  {formatPercentage(stats.avgConfidence)}
                </p>
                <p className="text-sm text-gray-600">Avg Confidence</p>
              </div>
            </div>
          </div>
          
          <div className="card p-4">
            <div className="flex items-center gap-3">
              <Clock className="h-8 w-8 text-primary-600" />
              <div>
                <p className="text-2xl font-bold text-gray-900">
                  {formatDuration(stats.avgProcessingTime * 1000)}
                </p>
                <p className="text-sm text-gray-600">Avg Time</p>
              </div>
            </div>
          </div>
          
          <div className="card p-4">
            <div className="flex items-center gap-3">
              <BarChart3 className="h-8 w-8 text-primary-600" />
              <div>
                <p className="text-2xl font-bold text-gray-900">
                  {Object.keys(stats.classCount).length}
                </p>
                <p className="text-sm text-gray-600">Classes Found</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Class Distribution */}
      {stats && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Class Distribution
          </h3>
          <div className="space-y-3">
            {Object.entries(stats.classCount).map(([className, count]) => {
              const percentage = (count / stats.total) * 100
              const colorClass = getClassColor(className)
              
              return (
                <div key={className} className="flex items-center gap-4">
                  <div className="w-24 text-sm font-medium text-gray-700 capitalize">
                    {className.replace('_', ' ')}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div
                          className={cn('h-2 rounded-full', colorClass)}
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                      <span className="text-sm text-gray-600 w-12">
                        {count}
                      </span>
                    </div>
                  </div>
                  <div className="text-sm text-gray-500 w-12">
                    {percentage.toFixed(1)}%
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <div className="flex flex-wrap gap-2">
          {/* Filter */}
          <select
            value={filterClass}
            onChange={(e) => setFilterClass(e.target.value as FilterClass)}
            className="input text-sm"
          >
            <option value="all">All Classes</option>
            <option value="healthy">Healthy</option>
            <option value="angular_leaf_spot">Angular Leaf Spot</option>
            <option value="bean_rust">Bean Rust</option>
          </select>

          {/* Sort */}
          <select
            value={`${sortField}-${sortDirection}`}
            onChange={(e) => {
              const [field, direction] = e.target.value.split('-')
              setSortField(field as SortField)
              setSortDirection(direction as SortDirection)
            }}
            className="input text-sm"
          >
            <option value="confidence-desc">Confidence (High to Low)</option>
            <option value="confidence-asc">Confidence (Low to High)</option>
            <option value="class_name-asc">Class Name (A-Z)</option>
            <option value="class_name-desc">Class Name (Z-A)</option>
            <option value="processing_time-asc">Processing Time (Fast to Slow)</option>
            <option value="processing_time-desc">Processing Time (Slow to Fast)</option>
            <option value="filename-asc">Filename (A-Z)</option>
            <option value="filename-desc">Filename (Z-A)</option>
          </select>

          {/* View Mode */}
          <div className="flex rounded-lg border border-gray-300 overflow-hidden">
            <button
              onClick={() => setViewMode('grid')}
              className={cn(
                'px-3 py-2 text-sm',
                viewMode === 'grid' 
                  ? 'bg-primary-600 text-white' 
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              )}
            >
              Grid
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={cn(
                'px-3 py-2 text-sm border-l border-gray-300',
                viewMode === 'list' 
                  ? 'bg-primary-600 text-white' 
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              )}
            >
              List
            </button>
          </div>
        </div>

        {/* Export */}
        <div className="flex gap-2">
          <button
            onClick={exportCSV}
            className="btn-secondary text-sm flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            Export CSV
          </button>
          <button
            onClick={exportResults}
            className="btn-secondary text-sm flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            Export JSON
          </button>
        </div>
      </div>

      {/* Results */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">
            Results ({filteredAndSortedResults.length})
          </h3>
        </div>

        {viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredAndSortedResults.map((result, index) => (
              <ResultCard
                key={`${result.filename}-${index}`}
                result={result}
                isSelected={selectedResult === result}
                onClick={() => onResultSelect?.(result)}
              />
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {filteredAndSortedResults.map((result, index) => (
              <ResultRow
                key={`${result.filename}-${index}`}
                result={result}
                isSelected={selectedResult === result}
                onClick={() => onResultSelect?.(result)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

interface ResultCardProps {
  result: ClassificationResult
  isSelected: boolean
  onClick: () => void
}

const ResultCard: React.FC<ResultCardProps> = ({ result, isSelected, onClick }) => {
  const classColor = getClassColor(result.class_name)
  
  return (
    <div
      onClick={onClick}
      className={cn(
        'card p-4 cursor-pointer transition-all hover:shadow-md',
        isSelected && 'ring-2 ring-primary-500'
      )}
    >
      <div className="space-y-3">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">
              {result.filename || 'Unknown'}
            </p>
            <p className="text-xs text-gray-500">
              {formatDuration(result.processing_time * 1000)}
            </p>
          </div>
          <span className={cn(
            'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium',
            getClassBadgeColor(result.class_name)
          )}>
            {result.class_name.replace('_', ' ')}
          </span>
        </div>

        {/* Confidence */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm text-gray-600">Confidence</span>
            <span className="text-sm font-medium text-gray-900">
              {formatPercentage(result.confidence)}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={cn('h-2 rounded-full', classColor)}
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
        </div>

        {/* Probabilities */}
        {result.probabilities && (
          <div className="space-y-1">
            <p className="text-xs text-gray-500">All Probabilities</p>
            {Object.entries(result.probabilities).map(([className, probability]) => (
              <div key={className} className="flex justify-between text-xs">
                <span className="text-gray-600 capitalize">
                  {className.replace('_', ' ')}
                </span>
                <span className="text-gray-900">
                  {formatPercentage(probability)}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Validation Info */}
        {result.validation_info && (
          <div className="pt-2 border-t border-gray-100">
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Info className="h-3 w-3" />
              <span>
                {result.validation_info.width}×{result.validation_info.height}
                {result.validation_info.file_size && (
                  <span> • {formatBytes(result.validation_info.file_size)}</span>
                )}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

interface ResultRowProps {
  result: ClassificationResult
  isSelected: boolean
  onClick: () => void
}

const ResultRow: React.FC<ResultRowProps> = ({ result, isSelected, onClick }) => {
  return (
    <div
      onClick={onClick}
      className={cn(
        'card p-4 cursor-pointer transition-all hover:shadow-sm',
        isSelected && 'ring-2 ring-primary-500'
      )}
    >
      <div className="flex items-center gap-4">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 truncate">
            {result.filename || 'Unknown'}
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          <span className={cn(
            'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium',
            getClassBadgeColor(result.class_name)
          )}>
            {result.class_name.replace('_', ' ')}
          </span>
          
          <div className="text-right">
            <p className="text-sm font-medium text-gray-900">
              {formatPercentage(result.confidence)}
            </p>
            <p className="text-xs text-gray-500">
              {formatDuration(result.processing_time * 1000)}
            </p>
          </div>
          
          <Eye className="h-4 w-4 text-gray-400" />
        </div>
      </div>
    </div>
  )
}

// Utility functions
const getClassColor = (className: string): string => {
  switch (className) {
    case 'healthy':
      return 'bg-primary-600'
    case 'angular_leaf_spot':
      return 'bg-secondary-600'
    case 'bean_rust':
      return 'bg-danger-600'
    default:
      return 'bg-gray-600'
  }
}

const getClassBadgeColor = (className: string): string => {
  switch (className) {
    case 'healthy':
      return 'bg-primary-100 text-primary-800'
    case 'angular_leaf_spot':
      return 'bg-secondary-100 text-secondary-800'
    case 'bean_rust':
      return 'bg-danger-100 text-danger-800'
    default:
      return 'bg-gray-100 text-gray-800'
  }
}