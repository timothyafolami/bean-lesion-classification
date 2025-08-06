import React from 'react'
import { Link } from 'react-router-dom'
import { Camera, Zap, Shield, BarChart3, ArrowRight } from 'lucide-react'

const features = [
  {
    name: 'AI-Powered Detection',
    description: 'Advanced machine learning models trained on thousands of bean leaf images for accurate disease classification.',
    icon: Zap,
  },
  {
    name: 'Real-time Analysis',
    description: 'Get instant results with our optimized ONNX inference engine supporting both single and batch processing.',
    icon: BarChart3,
  },
  {
    name: 'Reliable Results',
    description: 'High-accuracy classification with confidence scores and detailed validation information.',
    icon: Shield,
  },
]

const diseases = [
  {
    name: 'Healthy',
    description: 'Normal, disease-free bean leaves',
    color: 'bg-primary-100 text-primary-800',
  },
  {
    name: 'Angular Leaf Spot',
    description: 'Caused by Pseudocercospora griseola',
    color: 'bg-secondary-100 text-secondary-800',
  },
  {
    name: 'Bean Rust',
    description: 'Caused by Uromyces appendiculatus',
    color: 'bg-danger-100 text-danger-800',
  },
]

export const HomePage: React.FC = () => {
  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="text-center">
        <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
          Bean Lesion
          <span className="text-primary-600"> Classification</span>
        </h1>
        <p className="mt-6 text-lg leading-8 text-gray-600 max-w-2xl mx-auto">
          AI-powered system for detecting and classifying bean leaf diseases. 
          Upload images of bean leaves to get instant, accurate disease diagnosis.
        </p>
        <div className="mt-10 flex items-center justify-center gap-x-6">
          <Link
            to="/classify"
            className="btn-primary flex items-center gap-2 text-lg px-6 py-3"
          >
            <Camera className="h-5 w-5" />
            Start Classification
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-12">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900">
            Why Choose Our System?
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Built with cutting-edge technology for reliable plant disease detection
          </p>
        </div>
        
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => {
            const Icon = feature.icon
            return (
              <div key={feature.name} className="card p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-600">
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900">
                    {feature.name}
                  </h3>
                </div>
                <p className="text-gray-600">
                  {feature.description}
                </p>
              </div>
            )
          })}
        </div>
      </div>

      {/* Disease Types Section */}
      <div className="py-12">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900">
            Detectable Conditions
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Our system can identify the following bean leaf conditions
          </p>
        </div>
        
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {diseases.map((disease) => (
            <div key={disease.name} className="card p-6">
              <div className="flex items-center gap-3 mb-3">
                <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${disease.color}`}>
                  {disease.name}
                </span>
              </div>
              <p className="text-gray-600">
                {disease.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* CTA Section */}
      <div className="card p-8 text-center bg-gradient-to-r from-primary-50 to-primary-100">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Ready to Get Started?
        </h2>
        <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
          Upload your bean leaf images and get instant AI-powered disease classification 
          with detailed confidence scores and analysis.
        </p>
        <Link
          to="/classify"
          className="btn-primary inline-flex items-center gap-2"
        >
          <Camera className="h-5 w-5" />
          Start Classifying Now
        </Link>
      </div>
    </div>
  )
}