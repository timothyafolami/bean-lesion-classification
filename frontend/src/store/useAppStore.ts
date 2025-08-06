import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { ClassificationResult } from '../lib/api'

export interface UploadProgress {
  fileId: string
  fileName: string
  progress: number
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error'
  error?: string
}

export interface AppState {
  // Upload state
  uploadProgress: UploadProgress[]
  isUploading: boolean
  
  // Results state
  results: ClassificationResult[]
  selectedResult: ClassificationResult | null
  
  // UI state
  sidebarOpen: boolean
  theme: 'light' | 'dark'
  
  // Actions
  setUploadProgress: (progress: UploadProgress[]) => void
  updateUploadProgress: (fileId: string, updates: Partial<UploadProgress>) => void
  setIsUploading: (uploading: boolean) => void
  addResult: (result: ClassificationResult) => void
  addResults: (results: ClassificationResult[]) => void
  setSelectedResult: (result: ClassificationResult | null) => void
  clearResults: () => void
  setSidebarOpen: (open: boolean) => void
  setTheme: (theme: 'light' | 'dark') => void
  reset: () => void
}

const initialState = {
  uploadProgress: [],
  isUploading: false,
  results: [],
  selectedResult: null,
  sidebarOpen: false,
  theme: 'light' as const,
}

export const useAppStore = create<AppState>()(
  devtools(
    (set, get) => ({
      ...initialState,

      setUploadProgress: (progress) =>
        set({ uploadProgress: progress }, false, 'setUploadProgress'),

      updateUploadProgress: (fileId, updates) =>
        set(
          (state) => ({
            uploadProgress: state.uploadProgress.map((item) =>
              item.fileId === fileId ? { ...item, ...updates } : item
            ),
          }),
          false,
          'updateUploadProgress'
        ),

      setIsUploading: (uploading) =>
        set({ isUploading: uploading }, false, 'setIsUploading'),

      addResult: (result) =>
        set(
          (state) => ({
            results: [result, ...state.results],
          }),
          false,
          'addResult'
        ),

      addResults: (results) =>
        set(
          (state) => ({
            results: [...results, ...state.results],
          }),
          false,
          'addResults'
        ),

      setSelectedResult: (result) =>
        set({ selectedResult: result }, false, 'setSelectedResult'),

      clearResults: () =>
        set({ results: [], selectedResult: null }, false, 'clearResults'),

      setSidebarOpen: (open) =>
        set({ sidebarOpen: open }, false, 'setSidebarOpen'),

      setTheme: (theme) => {
        set({ theme }, false, 'setTheme')
        // Update document class for theme
        document.documentElement.classList.toggle('dark', theme === 'dark')
      },

      reset: () => set(initialState, false, 'reset'),
    }),
    {
      name: 'bean-lesion-app-store',
    }
  )
)

// Selectors for better performance
export const useUploadState = () =>
  useAppStore((state) => ({
    uploadProgress: state.uploadProgress,
    isUploading: state.isUploading,
    setUploadProgress: state.setUploadProgress,
    updateUploadProgress: state.updateUploadProgress,
    setIsUploading: state.setIsUploading,
  }))

export const useResultsState = () =>
  useAppStore((state) => ({
    results: state.results,
    selectedResult: state.selectedResult,
    addResult: state.addResult,
    addResults: state.addResults,
    setSelectedResult: state.setSelectedResult,
    clearResults: state.clearResults,
  }))

export const useUIState = () =>
  useAppStore((state) => ({
    sidebarOpen: state.sidebarOpen,
    theme: state.theme,
    setSidebarOpen: state.setSidebarOpen,
    setTheme: state.setTheme,
  }))