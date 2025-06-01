import { useState } from "react";
import "./App.css";
import axios from "axios";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [plotPaths, setPlotPaths] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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
      <div className="min-h-screen flex flex-col">
        {" "}
        {/* Changed from h-screen to min-h-screen */}
        <h1 className="text-4xl font-medium text-gray-800 py-6 text-center animate-fade-in bg-white/50 backdrop-blur-sm sticky top-0 z-10 border-b border-gray-200/50">
          Supervised vs Semi-Supervised Learning
        </h1>
        <div className="flex-1 flex flex-col md:flex-row gap-6 p-6 lg:p-8">
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
          <div className="w-full md:w-2/3 flex-1">
            <div className="bg-white rounded-2xl p-8 shadow-soft animate-slide-up">
              <h2 className="text-2xl font-medium text-gray-800 mb-6 text-center">
                Analysis Results
              </h2>

              {plotPaths.length > 0 && (
                <div className="space-y-8">
                  {/* Supervised Learning Section */}
                  <div>
                    <h3 className="text-xl font-medium text-gray-700 mb-4 border-b pb-2">
                      Supervised Learning Analysis
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {plotPaths.slice(0, 2).map((path, index) => (
                        <div
                          key={index}
                          className="bg-gray-50 rounded-xl p-4 overflow-hidden"
                        >
                          <img
                            src={path}
                            alt={`Supervised Plot ${index + 1}`}
                            className="w-full h-auto object-contain"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Semi-Supervised Learning Section */}
                  <div>
                    <h3 className="text-xl font-medium text-gray-700 mb-4 border-b pb-2">
                      Semi-Supervised Learning Analysis
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {plotPaths.slice(2, 4).map((path, index) => (
                        <div
                          key={index + 2}
                          className="bg-gray-50 rounded-xl p-4 overflow-hidden"
                        >
                          <img
                            src={path}
                            alt={`Semi-Supervised Plot ${index + 1}`}
                            className="w-full h-auto object-contain"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Final Comparison Section */}
                  <div>
                    <h3 className="text-xl font-medium text-gray-700 mb-4 border-b pb-2">
                      Final Comparison
                    </h3>
                    <div className="bg-gray-50 rounded-xl p-4 overflow-hidden">
                      <img
                        src={plotPaths[4]}
                        alt="Final Comparison"
                        className="w-full h-auto object-contain"
                      />
                    </div>
                  </div>
                </div>
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
