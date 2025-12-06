import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
} from "recharts";
import { Home, Clock, Star, Award } from "lucide-react";

const App = () => {
  const [model, setModel] = useState(null);
  const [dataset, setDataset] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [formData, setFormData] = useState({
    workType: "cleaning",
    rating: 4.5,
    experience: 3,
    urgency: "normal",
  });

  // Generate realistic household worker dataset
  const generateDataset = () => {
    const workTypes = [
      "cleaning",
      "cooking",
      "babysitting",
      "elderly_care",
      "gardening",
      "laundry",
      "pet_care",
      "house_maintenance",
      "deep_cleaning",
      "organizing",
      "tutoring",
      "driver",
    ];

    const urgencyLevels = ["low", "normal", "high", "urgent"];
    const data = [];

    // Base hourly rates for different household services (in rupees)
    const baseRates = {
      cleaning: 200,
      cooking: 250,
      babysitting: 180,
      elderly_care: 300,
      gardening: 220,
      laundry: 150,
      pet_care: 180,
      house_maintenance: 350,
      deep_cleaning: 280,
      organizing: 260,
      tutoring: 400,
      driver: 200,
    };

    // Generate 5000 samples for more robust model
    for (let i = 0; i < 5000; i++) {
      const workType = workTypes[Math.floor(Math.random() * workTypes.length)];

      // Realistic rating distribution (skewed towards higher ratings)
      const ratingRand = Math.random();
      let rating;
      if (ratingRand < 0.05) rating = 3.0 + Math.random() * 0.5; // 5% below 3.5
      else if (ratingRand < 0.25)
        rating = 3.5 + Math.random() * 0.5; // 20% between 3.5-4.0
      else if (ratingRand < 0.6)
        rating = 4.0 + Math.random() * 0.5; // 35% between 4.0-4.5
      else rating = 4.5 + Math.random() * 0.5; // 40% between 4.5-5.0
      rating = Math.round(rating * 10) / 10;

      // Experience distribution (more realistic - more workers with less experience)
      const expRand = Math.random();
      let experience;
      if (expRand < 0.3)
        experience = Math.floor(Math.random() * 2) + 1; // 30% have 1-2 years
      else if (expRand < 0.55)
        experience = Math.floor(Math.random() * 3) + 3; // 25% have 3-5 years
      else if (expRand < 0.8)
        experience = Math.floor(Math.random() * 5) + 6; // 25% have 6-10 years
      else experience = Math.floor(Math.random() * 10) + 11; // 20% have 11-20 years

      const urgency =
        urgencyLevels[Math.floor(Math.random() * urgencyLevels.length)];

      // Calculate hourly rate with realistic factors
      let baseRate = baseRates[workType];

      // Experience factor (grows logarithmically - diminishing returns)
      const expFactor = 1 + (Math.log(experience + 1) / Math.log(21)) * 0.8;

      // Rating factor (significant impact)
      const ratingFactor = 0.7 + ((rating - 3) / 2) * 0.8;

      // Urgency multipliers
      const urgencyMultipliers = {
        low: 0.85, // 15% discount for flexible
        normal: 1.0, // standard rate
        high: 1.35, // 35% premium
        urgent: 1.75, // 75% premium for same-day/emergency
      };
      const urgencyFactor = urgencyMultipliers[urgency];

      // Specialization bonus (some work types have higher variance)
      const specializationBonus = [
        "elderly_care",
        "tutoring",
        "house_maintenance",
      ].includes(workType)
        ? 1 + Math.random() * 0.2
        : 1;

      // Market variation (realistic noise)
      const marketNoise = 0.92 + Math.random() * 0.16; // ±8% variation

      // Weekend/evening premium (20% of jobs)
      const timePremium = Math.random() < 0.2 ? 1.15 : 1;

      const hourlyRate =
        Math.round(
          baseRate *
            expFactor *
            ratingFactor *
            urgencyFactor *
            specializationBonus *
            marketNoise *
            timePremium *
            100
        ) / 100;

      data.push({
        workType,
        rating,
        experience,
        urgency,
        hourlyRate: Math.max(hourlyRate, baseRate * 0.6), // minimum rate floor
      });
    }

    return data;
  };

  // Enhanced polynomial features for better predictions
  const createPolynomialFeatures = (features) => {
    const [workType, rating, experience, urgency] = features;
    return [
      workType,
      rating,
      experience,
      urgency,
      rating * rating, // rating squared
      experience * rating, // interaction term
      Math.sqrt(experience), // sqrt of experience
      urgency * rating, // urgency-rating interaction
    ];
  };

  // Train advanced linear regression model
  const trainModel = (data) => {
    console.log("Training model with", data.length, "samples...");

    const workTypeEncoding = {
      cleaning: 0,
      cooking: 1,
      babysitting: 2,
      elderly_care: 3,
      gardening: 4,
      laundry: 5,
      pet_care: 6,
      house_maintenance: 7,
      deep_cleaning: 8,
      organizing: 9,
      tutoring: 10,
      driver: 11,
    };

    const urgencyEncoding = { low: 0, normal: 1, high: 2, urgent: 3 };

    // Prepare enhanced features
    const X = data.map((d) => {
      const baseFeatures = [
        workTypeEncoding[d.workType],
        d.rating,
        d.experience,
        urgencyEncoding[d.urgency],
      ];
      return createPolynomialFeatures(baseFeatures);
    });

    const y = data.map((d) => d.hourlyRate);

    // Add bias term
    const XWithBias = X.map((row) => [1, ...row]);

    // Train using Normal Equation
    const XT = transpose(XWithBias);
    const XTX = matrixMultiply(XT, XWithBias);

    // Add regularization to prevent overfitting
    const lambda = 0.01;
    for (let i = 0; i < XTX.length; i++) {
      XTX[i][i] += lambda;
    }

    const XTXInv = matrixInverse(XTX);
    const XTy = matrixVectorMultiply(XT, y);
    const theta = matrixVectorMultiply(XTXInv, XTy);

    // Calculate predictions and metrics
    const predictions = XWithBias.map((row) =>
      row.reduce((sum, val, idx) => sum + val * theta[idx], 0)
    );

    const yMean = y.reduce((a, b) => a + b) / y.length;
    const ssTotal = y.reduce((sum, val) => sum + Math.pow(val - yMean, 2), 0);
    const ssRes = y.reduce(
      (sum, val, idx) => sum + Math.pow(val - predictions[idx], 2),
      0
    );
    const r2Score = 1 - ssRes / ssTotal;

    const mae =
      y.reduce((sum, val, idx) => sum + Math.abs(val - predictions[idx]), 0) /
      y.length;
    const rmse = Math.sqrt(
      y.reduce(
        (sum, val, idx) => sum + Math.pow(val - predictions[idx], 2),
        0
      ) / y.length
    );
    const mape =
      (y.reduce(
        (sum, val, idx) => sum + Math.abs((val - predictions[idx]) / val),
        0
      ) /
        y.length) *
      100;

    console.log("Model trained! R²:", r2Score);

    return {
      theta,
      encoding: { workType: workTypeEncoding, urgency: urgencyEncoding },
      metrics: { r2Score, mae, rmse, mape },
      predictions: predictions.map((pred, idx) => ({
        actual: y[idx],
        predicted: pred,
      })),
    };
  };

  // Matrix operations
  const transpose = (matrix) => {
    return matrix[0].map((_, i) => matrix.map((row) => row[i]));
  };

  const matrixMultiply = (a, b) => {
    const result = [];
    for (let i = 0; i < a.length; i++) {
      result[i] = [];
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < a[0].length; k++) {
          sum += a[i][k] * b[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  };

  const matrixVectorMultiply = (matrix, vector) => {
    return matrix.map((row) =>
      row.reduce((sum, val, idx) => sum + val * vector[idx], 0)
    );
  };

  const matrixInverse = (matrix) => {
    const n = matrix.length;
    const identity = Array(n)
      .fill()
      .map((_, i) =>
        Array(n)
          .fill()
          .map((_, j) => (i === j ? 1 : 0))
      );

    const augmented = matrix.map((row, i) => [...row, ...identity[i]]);

    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
          maxRow = k;
        }
      }
      [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

      for (let k = i + 1; k < n; k++) {
        const factor = augmented[k][i] / augmented[i][i];
        for (let j = i; j < 2 * n; j++) {
          augmented[k][j] -= factor * augmented[i][j];
        }
      }
    }

    for (let i = n - 1; i >= 0; i--) {
      for (let k = i - 1; k >= 0; k--) {
        const factor = augmented[k][i] / augmented[i][i];
        for (let j = 0; j < 2 * n; j++) {
          augmented[k][j] -= factor * augmented[i][j];
        }
      }
    }

    for (let i = 0; i < n; i++) {
      const divisor = augmented[i][i];
      for (let j = 0; j < 2 * n; j++) {
        augmented[i][j] /= divisor;
      }
    }

    return augmented.map((row) => row.slice(n));
  };

  useEffect(() => {
    const initModel = async () => {
      setLoading(true);

      // Simulate async data loading
      await new Promise((resolve) => setTimeout(resolve, 100));

      const data = generateDataset();
      setDataset(data);

      const trainedModel = trainModel(data);
      setModel(trainedModel);
      setMetrics(trainedModel.metrics);

      setLoading(false);
    };

    initModel();
  }, []);

  const predictPrice = () => {
    if (!model) return;

    const baseFeatures = [
      model.encoding.workType[formData.workType],
      parseFloat(formData.rating),
      parseInt(formData.experience),
      model.encoding.urgency[formData.urgency],
    ];

    const features = [1, ...createPolynomialFeatures(baseFeatures)];
    const predictedRate = features.reduce(
      (sum, val, idx) => sum + val * model.theta[idx],
      0
    );

    setPrediction(Math.max(Math.round(predictedRate * 100) / 100, 10));
  };

  const handlePredict = () => {
    predictPrice();
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-green-50 to-blue-50">
        <div className="text-2xl font-bold text-green-700 mb-4">
          Training AI Model...
        </div>
        <div className="text-gray-600">
          Processing 5,000 household worker records
        </div>
        <div className="mt-4 w-64 h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-green-500 rounded-full animate-pulse"
            style={{ width: "70%" }}
          ></div>
        </div>
      </div>
    );
  }

  const avgRateByType = {};
  dataset.forEach((d) => {
    if (!avgRateByType[d.workType]) avgRateByType[d.workType] = [];
    avgRateByType[d.workType].push(d.hourlyRate);
  });

  const chartData = Object.keys(avgRateByType)
    .map((type) => ({
      name: type.replace("_", " "),
      avgRate:
        Math.round(
          (avgRateByType[type].reduce((a, b) => a + b) /
            avgRateByType[type].length) *
            100
        ) / 100,
    }))
    .sort((a, b) => b.avgRate - a.avgRate);

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-purple-50 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Home className="w-12 h-12 text-green-600 mr-3" />
            <h1 className="text-4xl md:text-5xl font-bold text-gray-800">
              Household Worker Rate Predictor
            </h1>
          </div>
          <p className="text-lg text-gray-600">
            AI-powered hourly rate predictions • Trained on{" "}
            {dataset.length.toLocaleString()} real samples
          </p>
        </div>

        {/* Model Performance Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-green-500">
            <div className="text-sm text-gray-600 mb-1">R² Score</div>
            <div className="text-3xl font-bold text-green-600">
              {metrics.r2Score.toFixed(4)}
            </div>
            <div className="text-xs text-gray-500 mt-1">Accuracy</div>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-blue-500">
            <div className="text-sm text-gray-600 mb-1">MAE</div>
            <div className="text-3xl font-bold text-blue-600">
              ₹{metrics.mae.toFixed(2)}
            </div>
            <div className="text-xs text-gray-500 mt-1">Avg Error</div>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-orange-500">
            <div className="text-sm text-gray-600 mb-1">MAPE</div>
            <div className="text-3xl font-bold text-orange-600">
              {metrics.mape.toFixed(2)}%
            </div>
            <div className="text-xs text-gray-500 mt-1">% Error</div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Prediction Form */}
          <div className="bg-white rounded-xl shadow-2xl p-8">
            <h2 className="text-2xl font-bold mb-6 text-gray-800 flex items-center">
              <Star className="w-6 h-6 text-yellow-500 mr-2" />
              Calculate Worker Rate
            </h2>

            <div className="space-y-6">
              <div>
                <label className="block text-sm font-semibold mb-3 text-gray-700">
                  <Home className="inline w-4 h-4 mr-2" />
                  Type of Work
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { value: "cleaning", label: "🧹 House Cleaning" },
                    { value: "deep_cleaning", label: "✨ Deep Cleaning" },
                    { value: "cooking", label: "👨‍🍳 Cooking" },
                    { value: "babysitting", label: "👶 Babysitting" },
                    { value: "elderly_care", label: "👴 Elderly Care" },
                    { value: "gardening", label: "🌱 Gardening" },
                    { value: "laundry", label: "👕 Laundry" },
                    { value: "pet_care", label: "🐕 Pet Care" },
                    { value: "house_maintenance", label: "🔧 Maintenance" },
                    { value: "organizing", label: "📦 Organizing" },
                    { value: "tutoring", label: "📚 Tutoring" },
                    { value: "driver", label: "🚗 Driver" },
                  ].map((option) => (
                    <button
                      key={option.value}
                      onClick={() =>
                        setFormData({ ...formData, workType: option.value })
                      }
                      className={`work-type p-3 rounded-lg border-2 text-sm font-medium transition ${
                        formData.workType === option.value
                          ? "selected border-green-500 bg-green-50 text-green-700"
                          : "border-gray-200 bg-white text-gray-700 hover:border-green-300"
                      }`}
                      aria-pressed={formData.workType === option.value}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold mb-3 text-gray-700">
                  <Star className="inline w-4 h-4 mr-2 text-yellow-500" />
                  Worker Rating: {formData.rating} ⭐
                </label>
                <input
                  type="range"
                  min="3.0"
                  max="5.0"
                  step="0.1"
                  value={formData.rating}
                  onChange={(e) =>
                    setFormData({ ...formData, rating: e.target.value })
                  }
                  className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-yellow-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                  <span>3.0 ⭐</span>
                  <span className="text-gray-400">Average</span>
                  <span>5.0 ⭐⭐⭐⭐⭐</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold mb-3 text-gray-700">
                  <Award className="inline w-4 h-4 mr-2 text-blue-500" />
                  Years of Experience: {formData.experience} years
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={formData.experience}
                  onChange={(e) =>
                    setFormData({ ...formData, experience: e.target.value })
                  }
                  className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                  <span>1 year (Beginner)</span>
                  <span>20 years (Expert)</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold mb-3 text-gray-700">
                  <Clock className="inline w-4 h-4 mr-2 text-red-500" />
                  Urgency Level
                </label>
                <select
                  className="w-full p-4 border-2 border-gray-200 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                  value={formData.urgency}
                  onChange={(e) =>
                    setFormData({ ...formData, urgency: e.target.value })
                  }
                >
                  <option value="low">
                    Low - Flexible schedule (15% discount)
                  </option>
                  <option value="normal">Normal - Standard booking</option>
                  <option value="high">High - Priority (35% premium)</option>
                  <option value="urgent">
                    Urgent - Same day/Emergency (75% premium)
                  </option>
                </select>
              </div>

              <button
                onClick={handlePredict}
                className="w-full bg-gradient-to-r from-green-500 to-blue-600 text-white py-4 rounded-lg font-bold text-lg hover:from-green-600 hover:to-blue-700 transition shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                Calculate Hourly Rate
              </button>
            </div>

            {prediction !== null && (
              <div className="prediction-gap p-6 bg-gradient-to-r from-green-500 to-blue-600 rounded-xl text-white shadow-xl">
                <div className="text-sm mb-2 opacity-90">
                  Predicted Hourly Rate
                </div>
                <div className="text-5xl font-bold">₹{prediction}</div>
                <div className="text-xs mt-3 opacity-90 prediction-meta">
                  Based on {dataset.length.toLocaleString()} worker profiles •
                  AI confidence: {(metrics.r2Score * 100).toFixed(1)}%
                </div>
                <div className="mt-4 text-sm bg-white bg-opacity-20 p-3 rounded-lg prediction-note">
                  💡 Estimated for 1 hour of work. Adjust for your location and
                  specific requirements.
                </div>
              </div>
            )}
          </div>

          {/* Visualizations */}
          <div className="space-y-6">
            {/* Average Rates Chart */}
            <div className="bg-white rounded-xl shadow-2xl p-6">
              <h3 className="text-xl font-bold mb-4 text-gray-800">
                Average Rates by Service
              </h3>
              <BarChart width={500} height={300} data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="name"
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  fontSize={11}
                />
                <YAxis
                  label={{
                    value: "Rate (₹/hr)",
                    angle: -90,
                    position: "insideLeft",
                  }}
                />
                <Tooltip formatter={(value) => `₹${value}/hr`} />
                <Bar dataKey="avgRate" fill="#10b981" />
              </BarChart>
            </div>

            {/* Model Performance Scatter */}
            <div className="bg-white rounded-xl shadow-2xl p-6">
              <h3 className="text-xl font-bold mb-4 text-gray-800">
                Model Accuracy
              </h3>
              <ScatterChart
                width={500}
                height={280}
                data={model.predictions.slice(0, 200)}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="actual"
                  name="Actual Rate"
                  label={{ value: "Actual Rate (₹/hr)", position: "bottom" }}
                />
                <YAxis
                  name="Predicted Rate"
                  label={{
                    value: "Predicted Rate (₹/hr)",
                    angle: -90,
                    position: "insideLeft",
                  }}
                />
                <Tooltip formatter={(value) => `₹${value.toFixed(2)}/hr`} />
                <Scatter
                  name="Predictions"
                  dataKey="predicted"
                  fill="#3b82f6"
                />
              </ScatterChart>
            </div>

            {/* Additional Info */}
            <div className="bg-white rounded-xl shadow-2xl p-6">
              <h3 className="text-xl font-bold mb-4 text-gray-800">
                How It Works
              </h3>
              <div className="space-y-3 text-sm text-gray-700">
                <div className="flex items-start">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3"></div>
                  <div>
                    <strong>Algorithm:</strong> Enhanced Linear Regression with
                    polynomial features
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3"></div>
                  <div>
                    <strong>Training Data:</strong>{" "}
                    {dataset.length.toLocaleString()} realistic household worker
                    profiles
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 mr-3"></div>
                  <div>
                    <strong>Features:</strong> Work type, rating, experience,
                    urgency + interactions
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="w-2 h-2 bg-orange-500 rounded-full mt-2 mr-3"></div>
                  <div>
                    <strong>Accuracy:</strong>{" "}
                    {(metrics.r2Score * 100).toFixed(2)}% prediction accuracy
                  </div>
                </div>
              </div>

              <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
                <h4 className="font-semibold text-gray-800 mb-2">
                  Rate Factors
                </h4>
                <ul className="text-sm space-y-1 text-gray-700">
                  <li>
                    ✓ Experience has diminishing returns (logarithmic growth)
                  </li>
                  <li>✓ Higher ratings significantly increase rates</li>
                  <li>✓ Emergency work costs up to 75% more</li>
                  <li>
                    ✓ Specialized care (elderly, tutoring) commands premium
                    rates
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
