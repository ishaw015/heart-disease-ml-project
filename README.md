# Heart Disease Prediction ML Project

## Objective
This project predicts the likelihood of heart disease using machine learning models trained on a real cardiovascular dataset.

---

## Dataset
Heart Disease Dataset (UCI Repository + Kaggle)

- 918 patient records
- 11 medical features
- Binary classification:
  - 1 → Heart Disease
  - 0 → No Heart Disease

---

## Features Used

- Age
- Sex
- ChestPainType
- RestingBP
- Cholesterol
- FastingBS
- RestingECG
- MaxHR
- ExerciseAngina
- Oldpeak
- ST_Slope

---

## Project Workflow

1. Exploratory Data Analysis (EDA)
2. Data preprocessing pipeline
3. Feature encoding and scaling
4. Model training
5. Model evaluation
6. Best model selection
7. Model saving and inference

---

## Models Trained

- Logistic Regression
- Random Forest
- XGBoost

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Best Model

Random Forest achieved the best ROC-AUC score.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib

---

## Project Structure

```text
data/
models/
notebooks/
reports/
src/