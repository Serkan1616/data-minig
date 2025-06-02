import { useState } from "react";
import "./App.css";
import axios from "axios";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Purpose from "./components/Purpose"; // Purpose bileşenini import et

function Implementation() {
  const [selectedDataset, setSelectedDataset] = useState("");
  const [plotPaths, setPlotPaths] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);

  const datasets = {
    iris: {
      name: "Iris Dataset (Supervised Model SVM  Semi-supervised Model Label Propagation)",
      description: (
        <div>
          <p className="mb-4">
            The Iris flower dataset is a classical multivariate dataset
            introduced by Ronald Fisher in 1936. It is one of the most popular
            test cases for statistical classification techniques in machine
            learning. The dataset contains measurements from three Iris species
            (Iris setosa, Iris virginica, and Iris versicolor), with 50 samples
            from each species.
          </p>

          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200 mb-4">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Column Name
                  </th>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Unit
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Sepal Length
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Length of the sepal
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">cm</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Sepal Width
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Width of the sepal
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">cm</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Petal Length
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Length of the petal
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">cm</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Petal Width
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Width of the petal
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">cm</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Species
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Type of Iris flower
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Categorical
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="mt-4">
            Based on Fisher's linear discriminant model, this dataset became a
            typical test case for many statistical classification techniques in
            machine learning. Two of the three species were collected in the
            Gaspé Peninsula "all from the same pasture, and picked on the same
            day and measured at the same time by the same person with the same
            apparatus".
          </p>
        </div>
      ),
      file: "iris.csv",
    },
    titanic: {
      name: "Titanic Dataset (Supervised Model Random Forest Semi-supervised Model Self-training Logistic)",
      description: (
        <div>
          <p className="mb-4">
            The RMS Titanic was a British passenger liner that sank in the North
            Atlantic Ocean on April 15, 1912, after striking an iceberg during
            her maiden voyage from Southampton to New York City. This dataset
            contains passenger information including survival status, providing
            a classic binary classification problem.
          </p>

          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200 mb-4">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Variable
                  </th>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Values
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Survival
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Did the passenger survive?
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    0 = No, 1 = Yes
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Pclass
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Ticket class
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    1 = 1st, 2 = 2nd, 3 = 3rd
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Sex
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Gender of passenger
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    male, female
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Age
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Age in years
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">Numeric</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Fare
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Passenger fare
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">Numeric</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Embarked
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Port of Embarkation
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    C = Cherbourg, Q = Queenstown, S = Southampton
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    SibSp
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Number of siblings/spouses aboard
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Numeric (0 or more)
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Parch
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Number of parents/children aboard
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Numeric (0 or more)
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Ticket
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Ticket number
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">String</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="mt-4">
            This dataset is particularly interesting for machine learning as it
            combines categorical and numerical features, contains missing
            values, and presents a real-world binary classification problem. The
            challenge is to predict survival based on passenger attributes,
            making it perfect for comparing supervised and semi-supervised
            approaches.
          </p>
        </div>
      ),
      file: "train.csv",
    },
    creditcard: {
      name: "Credit Card Fraud Detection Dataset (Supervised Model XGBoost  Semi-supervised Model Autoencoder )",
      description: (
        <div>
          <p className="mb-4">
            Credit card fraud is a significant concern in financial services,
            with fraudsters continuously developing new methods to commit
            fraudulent transactions. This dataset contains transactions made by
            credit cards in September 2013, where we have 492 frauds out of
            284,807 transactions. The dataset is highly unbalanced, with frauds
            accounting for only 0.172% of all transactions.
          </p>

          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200 mb-4">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Feature
                  </th>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Time
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Seconds elapsed between each transaction
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">Numeric</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Amount
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Transaction amount
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">Numeric</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    V1-V28
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    PCA transformed features for confidentiality
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">Numeric</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Class
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    1 for fraudulent transactions, 0 otherwise
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">Binary</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="mt-4">
            Due to confidentiality issues, the original features have been
            transformed using PCA transformation. The only features that have
            not been transformed are 'Time' and 'Amount'. This dataset presents
            a perfect use case for semi-supervised learning, as labeled fraud
            cases are rare and expensive to obtain in real-world scenarios.
          </p>
        </div>
      ),
      file: "creditcard.csv",
    },
    wine: {
      name: "Wine Quality Dataset (Supervised Model RandomForestClassifier Semi-supervised Model Self-training )",
      description: (
        <div>
          <p className="mb-4">
            The Wine Quality dataset is based on variants of the Portuguese
            "Vinho Verde" wine. This unique wine is produced in the Minho region
            of Portugal and has been a protected designation of origin since
            1908. The dataset includes both physicochemical test results and
            quality ratings from wine experts, making it an excellent case for
            studying the relationship between objective measurements and
            subjective quality assessments.
          </p>

          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200 mb-4">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Feature
                  </th>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Unit/Type
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Fixed Acidity
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Most acids involved with wine or fixed or nonvolatile
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    g(tartaric acid)/L
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Volatile Acidity
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    The amount of acetic acid in wine
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    g(acetic acid)/L
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Citric Acid
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Found in small quantities, citric acid can add freshness
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">g/L</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Residual Sugar
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Amount of sugar remaining after fermentation
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">g/L</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Chlorides
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Amount of salt in the wine
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    g(sodium chloride)/L
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Free/Total Sulfur Dioxide
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Prevents microbial growth and wine oxidation
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">mg/L</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Density
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    The density of water is close to that of water
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">g/cm³</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    pH
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Describes how acidic or basic a wine is
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    0-14 scale
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Sulphates
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    A wine additive that contributes to SO2 levels
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    g(potassium sulphate)/L
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Alcohol
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    The percent alcohol content of the wine
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">% vol.</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    Quality
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Quality score given by wine experts
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    Score between 0-10
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="mt-4">
            This dataset presents a multi-class classification challenge where
            the goal is to predict wine quality based on physicochemical tests.
            Due to privacy and logistic issues, only these physicochemical and
            sensory variables are available (e.g., there is no data about grape
            types, wine brand, wine selling price, etc.). This makes it an
            interesting case for comparing supervised and semi-supervised
            approaches, especially when dealing with limited labeled data.
          </p>
        </div>
      ),
      file: "winequalityN.csv",
    },
  };

  const handleDatasetChange = (e) => {
    setSelectedDataset(e.target.value);
    setPlotPaths([]);
    setError(null);
  };

  const handleRun = async () => {
    if (!selectedDataset) {
      alert("Please select a dataset.");
      return;
    }

    setLoading(true);
    setError(null);
    setPlotPaths([]);

    try {
      const response = await axios.post(
        "http://localhost:5000/analyze",
        { dataset: datasets[selectedDataset].file },
        {
          headers: { "Content-Type": "application/json" },
        }
      );
      if (response.data.status === "ok") {
        const formattedPaths = response.data.plots.map((path) =>
          path.replace("./uploads/", "http://localhost:5000/static/")
        );
        setPlotPaths(formattedPaths);
      } else {
        setError("Analysis failed.");
      }
    } catch (err) {
      console.error(err);
      setError("Server error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 font-sans">
      <div className="min-h-screen flex flex-col">
        <div className="flex-1 flex flex-col md:flex-row gap-6 p-6 lg:p-8">
          {/* Dataset Selection Panel */}
          <div className="w-full md:w-1/3 space-y-6 bg-gray-800 rounded-2xl p-8 shadow-xl border border-gray-700">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Select Dataset
              </label>
              <select
                value={selectedDataset}
                onChange={handleDatasetChange}
                className="w-full p-4 bg-gray-700 border-2 border-gray-600 rounded-xl text-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 transition-all"
              >
                <option value="">Choose a dataset...</option>
                <option value="iris">Iris Dataset</option>
                <option value="titanic">Titanic Dataset</option>
                <option value="creditcard">Credit Card Fraud Detection</option>
                <option value="wine">Wine Quality Dataset</option>
              </select>
            </div>

            <button
              onClick={handleRun}
              disabled={loading || !selectedDataset}
              className="w-full bg-gradient-to-r from-indigo-600 to-indigo-700 text-white py-4 px-6 rounded-xl text-lg font-medium 
                hover:from-indigo-700 hover:to-indigo-800 transition-all transform hover:scale-[1.02] shadow-lg 
                disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Analyzing...
                </div>
              ) : (
                "Analyze"
              )}
            </button>
            {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
          </div>

          {/* Results Panel */}
          <div className="w-full md:w-2/3 flex-1">
            <div className="bg-gray-800 rounded-2xl p-8 shadow-xl border border-gray-700">
              {!plotPaths.length && selectedDataset && (
                <div className="prose prose-invert">
                  <h2 className="text-2xl font-medium text-gray-100 mb-4">
                    {datasets[selectedDataset].name}
                  </h2>
                  <div className="text-gray-300">
                    {datasets[selectedDataset].description}
                  </div>
                </div>
              )}

              {plotPaths.length > 0 && (
                <div className="space-y-8">
                  <h2 className="text-2xl font-medium text-gray-100 mb-6 text-center">
                    Analysis Results
                  </h2>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Metric cards with consistent styling */}
                    <div className="bg-gray-700 rounded-xl p-6 shadow-lg border border-gray-600">
                      <h3 className="text-lg font-medium text-gray-200 mb-4 border-b border-gray-600 pb-2">
                        Accuracy Analysis
                      </h3>
                      <img
                        src={plotPaths[0]}
                        alt="Accuracy Plot"
                        className="w-full rounded-lg cursor-pointer hover:opacity-90 transition-opacity"
                        onClick={() => setSelectedImage(plotPaths[0])}
                      />
                    </div>
                    <div className="bg-gray-700 rounded-xl p-6 shadow-lg border border-gray-600">
                      <h3 className="text-lg font-medium text-gray-200 mb-4 border-b border-gray-600 pb-2">
                        F1 Score Analysis
                      </h3>
                      <img
                        src={plotPaths[1]}
                        alt="F1 Score Plot"
                        className="w-full rounded-lg cursor-pointer hover:opacity-90 transition-opacity"
                        onClick={() => setSelectedImage(plotPaths[1])}
                      />
                    </div>
                    <div className="bg-gray-700 rounded-xl p-6 shadow-lg border border-gray-600">
                      <h3 className="text-lg font-medium text-gray-200 mb-4 border-b border-gray-600 pb-2">
                        Precision Analysis
                      </h3>
                      <img
                        src={plotPaths[2]}
                        alt="Precision Plot"
                        className="w-full rounded-lg cursor-pointer hover:opacity-90 transition-opacity"
                        onClick={() => setSelectedImage(plotPaths[2])}
                      />
                    </div>
                    <div className="bg-gray-700 rounded-xl p-6 shadow-lg border border-gray-600">
                      <h3 className="text-lg font-medium text-gray-200 mb-4 border-b border-gray-600 pb-2">
                        Recall Analysis
                      </h3>
                      <img
                        src={plotPaths[3]}
                        alt="Recall Plot"
                        className="w-full rounded-lg cursor-pointer hover:opacity-90 transition-opacity"
                        onClick={() => setSelectedImage(plotPaths[3])}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Image Modal with improved styling */}
              {selectedImage && (
                <div
                  className="fixed inset-0 bg-black bg-opacity-75 backdrop-blur-sm flex items-center justify-center z-50 p-4"
                  onClick={() => setSelectedImage(null)}
                >
                  <div className="relative max-w-4xl w-full">
                    <button
                      className="absolute -top-10 right-0 text-white hover:text-gray-300 text-xl"
                      onClick={() => setSelectedImage(null)}
                    >
                      ✕
                    </button>
                    <img
                      src={selectedImage}
                      alt="Enlarged Plot"
                      className="w-full rounded-lg shadow-2xl"
                      onClick={(e) => e.stopPropagation()}
                    />
                  </div>
                </div>
              )}

              {!plotPaths.length && !selectedDataset && (
                <div className="text-center text-gray-400">
                  Select a dataset and run analysis to see results.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 font-sans">
        <nav className="bg-gray-800 shadow-xl border-b border-gray-700">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex justify-between h-16">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-semibold text-gray-100">
                  Supervised vs Semi-Supervised Learning
                </h1>
              </div>

              <div className="flex items-center space-x-8">
                <Link
                  to="/purpose"
                  className="text-gray-300 hover:text-indigo-400 inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-indigo-400 transition-all"
                >
                  Purpose
                </Link>
                <Link
                  to="/"
                  className="text-gray-300 hover:text-indigo-400 inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-indigo-400 transition-all"
                >
                  Implementation
                </Link>
              </div>
            </div>
          </div>
        </nav>

        <Routes>
          <Route path="/purpose" element={<Purpose />} />
          <Route path="/" element={<Implementation />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
