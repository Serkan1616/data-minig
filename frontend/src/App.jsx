import { useState } from 'react'
import './App.css'

// Apple-style animation keyframes will be defined in App.css
function App() {
  const [selectedFiles, setSelectedFiles] = useState({
    dataset1: null,
    dataset2: null
  })
  const [results, setResults] = useState(null)
  const [currentImageIndex, setCurrentImageIndex] = useState(0)

  // Example images - you can replace these with your actual images
  const images = [
    {
      src: '/comparison1.png',
      title: 'Supervised vs Semi-Supervised Learning Comparison',
      description: 'Visual comparison of how both learning methods work with labeled and unlabeled data'
    },
    {
      src: '/accuracy.png',
      title: 'Accuracy Comparison',
      description: 'Performance metrics and accuracy comparison between methods'
    },
    {
      src: '/workflow.png',
      title: 'Learning Workflow',
      description: 'Step-by-step visualization of the learning process'
    }
  ]

  const nextImage = () => {
    setCurrentImageIndex((prev) => (prev + 1) % images.length)
  }

  const previousImage = () => {
    setCurrentImageIndex((prev) => (prev - 1 + images.length) % images.length)
  }

  const handleFileChange = (e, datasetNum) => {
    const file = e.target.files[0]
    setSelectedFiles(prev => ({
      ...prev,
      [`dataset${datasetNum}`]: file
    }))
  }

  const handleRun = () => {
    // TODO: Implement the run logic here
    console.log('Running analysis with files:', selectedFiles)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 font-sans">
      <div className="h-screen flex flex-col">
        <h1 className="text-4xl font-medium text-gray-800 py-6 text-center animate-fade-in bg-white/50 backdrop-blur-sm sticky top-0 z-10 border-b border-gray-200/50">
          Supervised vs Semi-Supervised Learning
        </h1>
        
        <div className="flex-1 flex flex-col md:flex-row gap-6 p-6 lg:p-8 overflow-hidden">
          {/* Left side - File inputs */}
          <div className="w-full md:w-1/3 space-y-6 bg-white rounded-2xl p-8 shadow-soft animate-slide-up">
            <div className="space-y-6">
              <div className="transition-all duration-300 hover:translate-y-[-2px]">
                <label className="block text-sm font-medium text-gray-700 mb-3">Dataset 1</label>
                <div className="relative group">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => handleFileChange(e, 1)}
                    className="w-full p-4 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all duration-300 outline-none file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  />
                </div>
              </div>
              
              <div className="transition-all duration-300 hover:translate-y-[-2px]">
                <label className="block text-sm font-medium text-gray-700 mb-3">Dataset 2</label>
                <div className="relative group">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => handleFileChange(e, 2)}
                    className="w-full p-4 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all duration-300 outline-none file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  />
                </div>
              </div>
            </div>
            
            <button
              onClick={handleRun}
              className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-4 px-6 rounded-xl text-lg font-medium hover:from-blue-700 hover:to-blue-800 transition-all duration-300 transform hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 shadow-lg"
            >
              Analyze
            </button>
          </div>

          {/* Right side - Results display */}
          <div className="w-full md:w-2/3 flex-1 overflow-hidden">
            <div className="bg-white rounded-2xl p-8 shadow-soft h-full animate-slide-up">
              <h2 className="text-2xl font-medium text-gray-800 mb-6 text-center">
                Analysis Results
              </h2>
              
              <div className="flex justify-between items-center mb-6">
                <button 
                  onClick={previousImage}
                  className="p-3 hover:bg-gray-100 rounded-full transition-all duration-300 group"
                >
                  <svg className="w-6 h-6 text-gray-600 group-hover:text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <h3 className="text-lg font-medium text-gray-700">{images[currentImageIndex].title}</h3>
                <button 
                  onClick={nextImage}
                  className="p-3 hover:bg-gray-100 rounded-full transition-all duration-300 group"
                >
                  <svg className="w-6 h-6 text-gray-600 group-hover:text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
              
              <div className="h-[400px] flex flex-col items-center justify-center rounded-xl bg-gray-50 p-8 relative overflow-hidden">
                {results ? (
                  <pre className="text-sm text-gray-700">{JSON.stringify(results, null, 2)}</pre>
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center">
                    <div className="relative w-full h-full">
                      <img 
                        src={images[currentImageIndex].src} 
                        alt={images[currentImageIndex].title}
                        className="absolute inset-0 w-full h-full object-contain transform transition-all duration-500 animate-fade-in"
                      />
                    </div>
                    <p className="text-gray-600 mt-4 text-center text-sm">
                      {images[currentImageIndex].description}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App