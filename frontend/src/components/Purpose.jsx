const Purpose = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 font-sans">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="bg-gray-800 rounded-2xl p-8 shadow-xl border border-gray-700">
          <h2 className="text-3xl font-bold text-gray-100 mb-8">
            Project Purpose
          </h2>

          <div className="space-y-8">
            <div className="prose prose-invert max-w-none">
              <p className="text-gray-300 mb-6">
                Our goal is to investigate when semi-supervised learning is more
                effective than supervised learning. To answer this question, we
                will test both methods on different datasets under various
                conditions.
              </p>

              <h3 className="text-xl font-semibold text-gray-100 mt-8 mb-4">
                Experimental Scenarios
              </h3>

              <div className="bg-gray-700 rounded-lg shadow-lg border border-gray-600 overflow-hidden mb-8">
                <table className="min-w-full">
                  <thead className="bg-gray-600">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                        Scenario
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                        Labeled Data Ratio
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                        Purpose
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-500">
                    <tr>
                      <td className="px-6 py-4 text-sm text-gray-100">1</td>
                      <td className="px-6 py-4 text-sm text-gray-100">1%</td>
                      <td className="px-6 py-4 text-sm text-gray-100">
                        Is semi-supervised advantageous?
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 text-sm text-gray-100">2</td>
                      <td className="px-6 py-4 text-sm text-gray-100">5%</td>
                      <td className="px-6 py-4 text-sm text-gray-100">
                        Is semi-supervised still better?
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 text-sm text-gray-100">3</td>
                      <td className="px-6 py-4 text-sm text-gray-100">
                        10%-20%
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-100">
                        Similar performance range?
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 text-sm text-gray-100">4</td>
                      <td className="px-6 py-4 text-sm text-gray-100">
                        40%-60%
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-100">
                        Does supervised start to lead?
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 text-sm text-gray-100">5</td>
                      <td className="px-6 py-4 text-sm text-gray-100">100%</td>
                      <td className="px-6 py-4 text-sm text-gray-100">
                        Is supervised clearly better?
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <h3 className="text-xl font-semibold text-gray-100 mt-8 mb-4">
                Datasets Used
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-700 rounded-lg p-6 border border-gray-600">
                  <h4 className="font-medium text-gray-100 mb-2">
                    Iris Dataset (150 samples)
                  </h4>
                  <p className="text-gray-300">
                    Classic classification problem with three iris flower
                    species.
                  </p>
                </div>

                <div className="bg-gray-700 rounded-lg p-6 border border-gray-600">
                  <h4 className="font-medium text-gray-100 mb-2">
                    Titanic Dataset (1000 samples)
                  </h4>
                  <p className="text-gray-300">
                    Predicting survival on the Titanic based on passenger
                    features.
                  </p>
                </div>

                <div className="bg-gray-700 rounded-lg p-6 border border-gray-600">
                  <h4 className="font-medium text-gray-100 mb-2">
                    Credit Card Fraud Detection (500 samples)
                  </h4>
                  <p className="text-gray-300">
                    Detecting fraud in credit card transactions, an example of
                    an imbalanced dataset.
                  </p>
                </div>

                <div className="bg-gray-700 rounded-lg p-6 border border-gray-600">
                  <h4 className="font-medium text-gray-100 mb-2">
                    Wine Quality Dataset (3000 samples)
                  </h4>
                  <p className="text-gray-300">
                    Predicting wine quality based on physicochemical properties.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Purpose;
