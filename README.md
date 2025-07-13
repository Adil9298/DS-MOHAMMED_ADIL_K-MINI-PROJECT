# 🚴‍♂️ Bike Rental Demand Prediction and Anomaly Detection

This project builds machine learning models to predict bike rental demand using daily and hourly historical data based on environmental and seasonal factors. It also explores anomaly detection by identifying days with unusual rental patterns, potentially linked to external events (e.g., Hurricane Sandy).

---

## 📌 Objective

To predict the number of bikes rented using machine learning models trained on weather, season, and temporal data, and detect anomalies caused by significant events.

---

## 🧾 Dataset Description

The dataset contains both **daily** and **hourly** rental records collected in Washington D.C. It includes:

| Feature       | Description |
|---------------|-------------|
| `dteday`      | Date |
| `season`      | 1: Spring, 2: Summer, 3: Fall, 4: Winter |
| `yr`          | 0: 2011, 1: 2012 |
| `mnth`        | Month |
| `hr`          | Hour *(hourly data only)* |
| `holiday`     | Is holiday (1: Yes) |
| `weekday`     | Day of week (0: Sunday) |
| `workingday`  | Is working day |
| `weathersit`  | Weather condition |
| `temp`        | Normalized temperature |
| `atemp`       | Normalized "feels like" temperature |
| `hum`         | Normalized humidity |
| `windspeed`   | Normalized wind speed |
| `casual`      | Count of casual users *(dropped for modeling)* |
| `registered`  | Count of registered users *(dropped for modeling)* |
| `cnt`         | Total count of rentals (target) |

---

## 🧠 Machine Learning Models

### ✅ Regression Models:
- Linear Regression
- Support Vector Regression (SVR)
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- MLP Regressor

### ✅ Classification Models (optional anomaly detection):
- Logistic Regression
- Decision Tree
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boost Classifier
- AdaBoost Classifier
- Naive Bayes
- MLP Classifier

---

## 📊 Evaluation Metrics

### Regression:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

### Classification:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve

---

## 🔍 Project Pipeline

1. **Data Collection**: Import and inspect daily/hourly datasets.
2. **Data Preprocessing**: Clean, handle duplicates, outliers, skewness.
3. **EDA**: Visualizations, correlation analysis.
4. **Feature Engineering**: One-hot encoding, new features (e.g., weekend flag).
5. **Model Training**: Apply various ML algorithms for regression/classification.
6. **Hyperparameter Tuning**: GridSearchCV for model optimization.
7. **Model Evaluation**: Compare models using R², RMSE, etc.
8. **Model Saving**: Save best model using `joblib`.
9. **Deployment (optional)**: Package model with FastAPI & Docker.

---

## 📦 How to Run

1. Clone the repo:
git clone https://github.com/Adil9298/DS-MOHAMMED_ADIL_K-MINI-PROJECT.git
