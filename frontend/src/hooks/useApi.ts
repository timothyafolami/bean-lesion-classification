import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, ClassificationResult, HealthResponse, ModelInfoResponse, ClassesResponse } from '../lib/api'
import { useToast } from '../components/Toast'

// Health check hook
export const useHealthCheck = () => {
  return useQuery<HealthResponse>({
    queryKey: ['health'],
    queryFn: () => api.health.check().then(res => res.data),
    refetchInterval: 30000, // Refetch every 30 seconds
    retry: 3,
    staleTime: 10000, // 10 seconds
  })
}

// Model info hook
export const useModelInfo = () => {
  return useQuery<ModelInfoResponse>({
    queryKey: ['modelInfo'],
    queryFn: () => api.health.modelInfo().then(res => res.data),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 2,
  })
}

// Classes hook
export const useClasses = () => {
  return useQuery<ClassesResponse>({
    queryKey: ['classes'],
    queryFn: () => api.predict.classes().then(res => res.data),
    staleTime: 10 * 60 * 1000, // 10 minutes
    retry: 2,
  })
}

// Single prediction hook
export const useSinglePrediction = () => {
  const toast = useToast()
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ file, returnProbabilities = true }: { file: File; returnProbabilities?: boolean }) =>
      api.predict.single(file, returnProbabilities).then(res => res.data),
    onSuccess: (data) => {
      toast.success('Classification Complete', `Predicted: ${data.result.class_name}`)
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['performanceStats'] })
    },
    onError: (error: any) => {
      const message = error.response?.data?.detail || error.message || 'Classification failed'
      toast.error('Classification Failed', message)
    },
  })
}

// Batch prediction hook
export const useBatchPrediction = () => {
  const toast = useToast()
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ files, returnProbabilities = true }: { files: File[]; returnProbabilities?: boolean }) =>
      api.predict.batch(files, returnProbabilities).then(res => res.data),
    onSuccess: (data) => {
      toast.success(
        'Batch Classification Complete', 
        `Processed ${data.batch_size} images in ${data.total_processing_time.toFixed(2)}s`
      )
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['performanceStats'] })
    },
    onError: (error: any) => {
      const message = error.response?.data?.detail || error.message || 'Batch classification failed'
      toast.error('Batch Classification Failed', message)
    },
  })
}

// Performance stats hook
export const usePerformanceStats = () => {
  return useQuery({
    queryKey: ['performanceStats'],
    queryFn: () => api.predict.performanceStats().then(res => res.data),
    refetchInterval: 60000, // Refetch every minute
    retry: 2,
    staleTime: 30000, // 30 seconds
  })
}

// Benchmark hook
export const useBenchmark = () => {
  const toast = useToast()

  return useMutation({
    mutationFn: () => api.predict.benchmark().then(res => res.data),
    onSuccess: () => {
      toast.success('Benchmark Complete', 'Performance benchmark completed successfully')
    },
    onError: (error: any) => {
      const message = error.response?.data?.detail || error.message || 'Benchmark failed'
      toast.error('Benchmark Failed', message)
    },
  })
}

// Preprocessing info hook
export const usePreprocessingInfo = () => {
  return useQuery({
    queryKey: ['preprocessingInfo'],
    queryFn: () => api.predict.preprocessingInfo().then(res => res.data),
    staleTime: 10 * 60 * 1000, // 10 minutes
    retry: 2,
  })
}

// Custom hook for handling file uploads with progress
export const useFileUpload = () => {
  const toast = useToast()

  const uploadSingle = useMutation({
    mutationFn: ({ file, returnProbabilities = true }: { file: File; returnProbabilities?: boolean }) =>
      api.predict.single(file, returnProbabilities).then(res => res.data),
    onError: (error: any) => {
      const message = error.response?.data?.detail || error.message || 'Upload failed'
      toast.error('Upload Failed', message)
    },
  })

  const uploadBatch = useMutation({
    mutationFn: ({ files, returnProbabilities = true }: { files: File[]; returnProbabilities?: boolean }) =>
      api.predict.batch(files, returnProbabilities).then(res => res.data),
    onError: (error: any) => {
      const message = error.response?.data?.detail || error.message || 'Batch upload failed'
      toast.error('Batch Upload Failed', message)
    },
  })

  return {
    uploadSingle,
    uploadBatch,
    isUploading: uploadSingle.isPending || uploadBatch.isPending,
  }
}

// Hook for retry logic
export const useRetry = () => {
  const queryClient = useQueryClient()

  const retryQuery = (queryKey: string | string[]) => {
    queryClient.invalidateQueries({ queryKey: Array.isArray(queryKey) ? queryKey : [queryKey] })
    queryClient.refetchQueries({ queryKey: Array.isArray(queryKey) ? queryKey : [queryKey] })
  }

  const retryAll = () => {
    queryClient.invalidateQueries()
    queryClient.refetchQueries()
  }

  return {
    retryQuery,
    retryAll,
  }
}

// Connection status hook
export const useConnectionStatus = () => {
  const { data: health, error, isLoading } = useHealthCheck()

  const isOnline = !error && health?.status === 'healthy'
  const isOffline = !!error
  
  return {
    isOnline,
    isOffline,
    isLoading,
    health,
    error,
  }
}