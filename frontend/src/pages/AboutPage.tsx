import React from 'react'
import { Brain, Cpu, Zap, Shield, Github, ExternalLink } from 'lucide-react'

const technologies = [
  {
    name: 'EfficientNet-B0',
    description: 'State-of-the-art convolutional neural network architecture optimized for image classification.',
    icon: Brain,
  },
  {
    name: 'ONNX Runtime',
    description: 'High-performance inference engine with hardware acceleration support for fast predictions.',
    icon: Cpu,
  },
  {
    name: 'FastAPI',
    description: 'Modern, fast web framework for building APIs with automatic documentation generation.',
    icon: Zap,
  },
  {
    name: 'React + TypeScript',
    description: 'Modern frontend framework with type safety for robust user interface development.',
    icon: Shield,
  },
]

const metrics = [
  { label: 'Model Accuracy', value: '94.2%' },
  { label: 'Inference Time', value: '<100ms' },
  { label: 'Supported Formats', value: 'JPEG, PNG, WebP' },
  { label: 'Max File Size', value: '10MB' },
]

export const AboutPage: React.FC = () => {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900">
          About Bean Lesion Classification
        </h1>
        <p className="mt-4 text-lg text-gray-600 max-w-3xl mx-auto">
          An AI-powered system for detecting and classifying bean leaf diseases using 
          advanced machine learning techniques and modern web technologies.
        </p>
      </div>

      {/* Overview */}
      <div className="card p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          Project Overview
        </h2>
        <div className="prose prose-gray max-w-none">
          <p className="text-gray-600 leading-relaxed">
            This system leverages deep learning to automatically identify diseases in bean plant leaves. 
            The model was trained on a comprehensive dataset of bean leaf images and can distinguish 
            between healthy leaves and two common diseases: Angular Leaf Spot and Bean Rust.
          </p>
          <p className="text-gray-600 leading-relaxed mt-4">
            The application provides both single image classification and batch processing capabilities, 
            making it suitable for both individual farmers and agricultural researchers who need to 
            process multiple images efficiently.
          </p>
        </div>
      </div>

      {/* Technologies */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-8 text-center">
          Technologies Used
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {technologies.map((tech) => {
            const Icon = tech.icon
            return (
              <div key={tech.name} className="card p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-600">
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900">
                    {tech.name}
                  </h3>
                </div>
                <p className="text-gray-600">
                  {tech.description}
                </p>
              </div>
            )
          })}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="card p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
          Performance Metrics
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {metrics.map((metric) => (
            <div key={metric.label} className="text-center">
              <div className="text-2xl font-bold text-primary-600 mb-2">
                {metric.value}
              </div>
              <div className="text-sm text-gray-600">
                {metric.label}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Disease Information */}
      <div className="card p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          Detectable Diseases
        </h2>
        <div className="space-y-6">
          <div className="border-l-4 border-primary-500 pl-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Healthy Leaves
            </h3>
            <p className="text-gray-600">
              Normal, disease-free bean leaves showing typical green coloration and healthy tissue structure.
            </p>
          </div>
          
          <div className="border-l-4 border-secondary-500 pl-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Angular Leaf Spot
            </h3>
            <p className="text-gray-600">
              Caused by <em>Pseudocercospora griseola</em>, this disease appears as angular, water-soaked 
              lesions that become brown and necrotic, often surrounded by a yellow halo.
            </p>
          </div>
          
          <div className="border-l-4 border-danger-500 pl-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Bean Rust
            </h3>
            <p className="text-gray-600">
              Caused by <em>Uromyces appendiculatus</em>, this fungal disease produces small, reddish-brown 
              pustules on the leaf surface, which can lead to premature leaf drop.
            </p>
          </div>
        </div>
      </div>

      {/* Links */}
      <div className="card p-8 text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          Learn More
        </h2>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <a
            href="#"
            className="btn-primary inline-flex items-center gap-2"
          >
            <Github className="h-5 w-5" />
            View Source Code
          </a>
          <a
            href="#"
            className="btn-secondary inline-flex items-center gap-2"
          >
            <ExternalLink className="h-5 w-5" />
            API Documentation
          </a>
        </div>
      </div>
    </div>
  )
}