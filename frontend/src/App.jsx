import { useState } from "react";
import "./App.css";
import axios from "axios";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [plotPaths, setPlotPaths] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [error, setError] = useState(null);

  const nextImage = () => {
    setCurrentImageIndex((prev) => (prev + 1) % plotPaths.length);
  };

  const previousImage = () => {
    setCurrentImageIndex(
      (prev) => (prev - 1 + plotPaths.length) % plotPaths.length
    );
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setPlotPaths([]); // Reset previous plots
    setError(null);
  };

  const handleRun = async () => {
    if (!selectedFile) {
      alert("Please upload a CSV file.");
      return;
    }

    setLoading(true);
    setError(null);
    setPlotPaths([]);
    setCurrentImageIndex(0);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(
        "http://localhost:5000/analyze",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      if (response.data.status === "ok") {
        const formattedPaths = response.data.plots.map((path) =>
          path.replace("./static", "http://localhost:5000/static")
        );
        setPlotPaths(formattedPaths);
      } else {
        setError("Analysis failed.");
      }
    } catch (err) {
      console.error(err);
      setError("Server error or invalid file format.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 font-sans">
      <div className="h-screen flex flex-col">
        <h1 className="text-4xl font-medium text-gray-800 py-6 text-center animate-fade-in bg-white/50 backdrop-blur-sm sticky top-0 z-10 border-b border-gray-200/50">
          Supervised vs Semi-Supervised Learning
        </h1>

        <div className="flex-1 flex flex-col md:flex-row gap-6 p-6 lg:p-8 overflow-hidden">
          {/* File Upload Panel */}
          <div className="w-full md:w-1/3 space-y-6 bg-white rounded-2xl p-8 shadow-soft animate-slide-up">
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Upload CSV Dataset
            </label>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="w-full p-4 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 file:bg-blue-50 file:text-blue-700 transition-all"
            />
            <button
              onClick={handleRun}
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-4 px-6 rounded-xl text-lg font-medium hover:from-blue-700 hover:to-blue-800 transition-all transform hover:scale-[1.02] shadow-lg"
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>
            {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
          </div>

          {/* Results Panel */}
          <div className="w-full md:w-2/3 flex-1 overflow-hidden">
            <div className="bg-white rounded-2xl p-8 shadow-soft h-full animate-slide-up">
              <h2 className="text-2xl font-medium text-gray-800 mb-6 text-center">
                Analysis Results
              </h2>

              {plotPaths.length > 0 && (
                <>
                  <div className="flex justify-between items-center mb-6">
                    <button
                      onClick={previousImage}
                      className="p-3 hover:bg-gray-100 rounded-full"
                    >
                      <svg
                        className="w-6 h-6 text-gray-600"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M15 19l-7-7 7-7"
                        />
                      </svg>
                    </button>
                    <span className="text-gray-600 text-sm">
                      {currentImageIndex + 1} / {plotPaths.length}
                    </span>
                    <button
                      onClick={nextImage}
                      className="p-3 hover:bg-gray-100 rounded-full"
                    >
                      <svg
                        className="w-6 h-6 text-gray-600"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 5l7 7-7 7"
                        />
                      </svg>
                    </button>
                  </div>

                  <div className="h-[400px] flex flex-col items-center justify-center bg-gray-50 rounded-xl overflow-hidden">
                    <img
                      src={plotPaths[currentImageIndex]}
                      alt={`Plot ${currentImageIndex + 1}`}
                      className="max-h-full max-w-full object-contain transition-all duration-500"
                    />
                  </div>
                </>
              )}

              {!plotPaths.length && !loading && (
                <div className="text-center text-gray-500">
                  Upload a CSV file and run analysis to see results.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
