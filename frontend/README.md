# Bean Lesion Classification Frontend

A modern React frontend for the Bean Lesion Classification system, built with TypeScript, Vite, and Tailwind CSS.

## Features

- 🎨 **Modern UI**: Clean, responsive design with Tailwind CSS
- 🔧 **TypeScript**: Full type safety and better developer experience
- ⚡ **Vite**: Fast development server and optimized builds
- 🎯 **React Query**: Efficient data fetching and caching
- 🗃️ **Zustand**: Lightweight state management
- 🛡️ **Error Boundaries**: Graceful error handling
- 📱 **Responsive**: Works on desktop, tablet, and mobile devices

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **React Router** - Client-side routing
- **React Query** - Data fetching
- **Zustand** - State management
- **Axios** - HTTP client
- **Lucide React** - Icons

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create environment file:
   ```bash
   cp .env.example .env
   ```

4. Update the `.env` file with your API URL:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

### Development

Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Building for Production

Build the app for production:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── ErrorBoundary.tsx
│   └── Layout.tsx
├── lib/                # Utilities and configurations
│   ├── api.ts          # API client and types
│   └── utils.ts        # Helper functions
├── pages/              # Page components
│   ├── HomePage.tsx
│   ├── ClassificationPage.tsx
│   └── AboutPage.tsx
├── store/              # State management
│   └── useAppStore.ts  # Zustand store
├── App.tsx             # Main app component
├── main.tsx           # App entry point
└── index.css          # Global styles
```

## API Integration

The frontend communicates with the FastAPI backend through a configured Axios client. The API client includes:

- Automatic request/response interceptors
- Error handling
- TypeScript types for all endpoints
- File upload support for image classification

### Available API Endpoints

- `GET /health/` - Health check
- `GET /health/model` - Model information
- `POST /predict/single` - Single image classification
- `POST /predict/batch` - Batch image classification
- `GET /predict/classes` - Available classes
- `GET /predict/preprocessing-info` - Preprocessing information
- `GET /predict/performance-stats` - Performance statistics

## State Management

The app uses Zustand for state management with the following stores:

- **Upload State**: File upload progress and status
- **Results State**: Classification results and selected items
- **UI State**: Sidebar, theme, and other UI preferences

## Styling

The app uses Tailwind CSS with a custom design system:

- **Colors**: Primary (green), secondary (yellow), danger (red)
- **Components**: Reusable button, card, and input styles
- **Animations**: Fade-in, slide-up, and pulse animations
- **Responsive**: Mobile-first responsive design

## Error Handling

Comprehensive error handling includes:

- **Error Boundaries**: Catch and display React errors gracefully
- **API Error Handling**: Automatic retry and user-friendly error messages
- **Form Validation**: Client-side validation for file uploads
- **Network Errors**: Offline detection and retry mechanisms

## Performance Optimizations

- **Code Splitting**: Automatic route-based code splitting
- **Image Optimization**: Lazy loading and responsive images
- **Caching**: React Query for efficient data caching
- **Bundle Optimization**: Vite's optimized production builds

## Development Guidelines

### Code Style

- Use TypeScript for all new code
- Follow React best practices and hooks patterns
- Use functional components with hooks
- Implement proper error boundaries
- Write descriptive component and function names

### File Organization

- Group related components in folders
- Use index files for clean imports
- Separate business logic from UI components
- Keep components small and focused

### State Management

- Use Zustand for global state
- Keep local state in components when possible
- Use React Query for server state
- Implement proper loading and error states

## Contributing

1. Follow the existing code style and patterns
2. Add TypeScript types for new features
3. Test components in different screen sizes
4. Update documentation for new features
5. Ensure accessibility best practices

## License

This project is part of the Bean Lesion Classification system.