# 📊 Semi-Supervised vs Supervised ML Comparison Platform

This project is a **React + Tailwind** web application with a **Python (Flask) backend** designed to compare the performance of **supervised** and **semi-supervised** machine learning models across various datasets.

It demonstrates how **semi-supervised learning** can perform comparably (or better) than supervised learning when labeled data is limited.

---

## 🎯 Project Goals

- 🔍 Compare **supervised** vs **semi-supervised** ML models.
- ⚖️ Visualize model performance across **4 datasets**.
- 🧠 Explore the potential of **self-training** with scikit-learn and other semi-supervised techniques.
- 💻 Provide an interactive front-end to upload datasets and view results.

---

## 🛠️ Tech Stack

| Layer          | Technology                    |
|----------------|-------------------------------|
| Frontend       | React, Tailwind CSS           |
| Backend        | Python, Flask                 |
| ML Libraries   | scikit-learn, pandas, NumPy   |
| Semi-Supervised Models | SelfTrainingClassifier, LabelPropagation, LabelSpreading |

---

## 📁 Datasets Used

The platform includes evaluation on 4 datasets:

1. **Iris Dataset**
2. **Credit Card Fraud Detection**
3. **Titanic Dataset**
4. **Wine Dataset**

Each dataset is processed with different label ratios (1%, 5%, 20%, 100%) to test how supervised vs semi-supervised learning performs under low-label conditions.

---

## 🚀 Features

- 📤 Upload custom CSV datasets via the frontend.
- 📈 Automatically triggers training and evaluation on both supervised and semi-supervised models.
- 🖼️ Displays performance metrics and matplotlib/seaborn plots in the UI.
- 🧪 Supports multiple dataset-specific analysis modules for clean architecture.

---

## 🧪 How It Works

1. Upload a CSV file from the React frontend.
2. Backend detects the dataset type and runs appropriate analysis using `scikit-learn`.
3. Both supervised and semi-supervised models are trained with varying labeled data percentages.
4. Accuracy, precision, recall, and F1-scores are computed and plotted.
5. Results are sent back and displayed on the frontend.

---

## 🔧 Screenshots
![image](https://github.com/user-attachments/assets/4ee34b64-9d8c-4f0e-aa92-63cd182f3f5f)

![image](https://github.com/user-attachments/assets/5a203b9b-bf45-4898-96f7-f72696ed2b4e)

![image](https://github.com/user-attachments/assets/11305fc9-883e-4f91-a138-670b9974d391)

