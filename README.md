# 🌧️ Developed Data-Driven Machine Learning Models for Accurate Rainfall Prediction

This project focuses on predicting rainfall using multiple machine learning algorithms. The goal is to forecast whether it will rain **tomorrow** based on today’s weather data like temperature, humidity, wind speed, pressure, and more.

---

## 🎯 Project Objective

To build a predictive model using machine learning that forecasts whether it will rain tomorrow, using features such as:

- Temperature
- Humidity
- Wind Speed
- Atmospheric Pressure
- Rain Today

---

## 📊 Dataset Description

- **Source:** Kaggle / Meteorological Department Dataset  
- **Format:** CSV  
- **Rows:** 14,000+ entries  
- **Target Variable:** `RainTomorrow` (Yes/No)

### 🔑 Features:

`Date`, `Location`, `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`,  
`WindGustDir`, `WindGustSpeed`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Temp3pm`,  
`RainToday`, `RainTomorrow`

---

## 💻 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn (for EDA & Visualizations)  
- Google Colab

---

## 🧹 Data Preprocessing

- Removed null/missing values  
- Label encoded categorical variables  
- Converted `RainToday` and `RainTomorrow` to binary (Yes → 1, No → 0)  
- Normalized continuous variables  
- Feature selection based on correlation matrix

---

## 🔍 Model Building

The following models were trained and evaluated:

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Naive Bayes  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- **XGBoost (Best Performer)**  
- Artificial Neural Networks (ANN)

---

## 📏 Evaluation Metrics

Models were evaluated on:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- ROC Curve & AUC Score

---

## ✅ Results & Insights

![Results](https://github.com/user-attachments/assets/7302906d-11ec-4df8-93ff-14c060063c26)

📈 **Best Performing Model: `XGBoost`**

- **Accuracy:** 92.65%  
- **F1 Score (True class):** 0.91  
- Outperformed all other models including ANN and Random Forest

### 🧠 Additional Analysis

- Model performance using selected features  
- PCA-transformed data performance  
- Trade-off between model complexity and accuracy

---

## 📌 Conclusion

Machine Learning models, especially XGBoost, show high potential in **predicting rainfall** using weather datasets. This project also explores **feature engineering**, **model comparison**, and **PCA** for dimensionality reduction — helping build robust predictive systems for meteorology.

---

> 🔗 Let me know if you'd like to turn this into a web app (e.g., Streamlit UI) or write a paper on it — we can go all the way pro! 😎
