import pytest
import pandas as pd
import os
import joblib
from src.preprocess import build_preprocessor


# ── Test 1: Cleaned data file exists ─────────────────────────────────────────
def test_data_file_exists():
    assert os.path.exists("data/heart_clean.csv"), \
        "heart_clean.csv not found in data/"


# ── Test 2: Correct columns and non-empty ────────────────────────────────────
def test_data_columns_and_shape():
    df = pd.read_csv("data/heart_clean.csv")
    expected_cols = [
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
        "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
        "Oldpeak", "ST_Slope", "HeartDisease"
    ]
    assert list(df.columns) == expected_cols, \
        f"Column mismatch. Got: {list(df.columns)}"
    assert df.shape[0] > 0, "Dataset is empty"
    assert df.shape[1] == 12, "Expected 12 columns"


# ── Test 3: No missing values ─────────────────────────────────────────────────
def test_no_missing_values():
    df = pd.read_csv("data/heart_clean.csv")
    missing = df.isnull().sum().sum()
    assert missing == 0, f"Found {missing} missing values"


# ── Test 4: Target column is binary ──────────────────────────────────────────
def test_target_is_binary():
    df = pd.read_csv("data/heart_clean.csv")
    unique_vals = set(df["HeartDisease"].unique())
    assert unique_vals == {0, 1}, \
        f"HeartDisease should only contain 0 and 1, got: {unique_vals}"


# ── Test 5: No duplicate rows ─────────────────────────────────────────────────
def test_no_duplicates():
    df = pd.read_csv("data/heart_clean.csv")
    duplicates = df.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate rows"


# ── Test 6: Preprocessor builds without error ────────────────────────────────
def test_preprocessor_builds():
    preprocessor = build_preprocessor()
    assert preprocessor is not None, \
        "build_preprocessor() returned None"


# ── Test 7: Preprocessor transforms data correctly ───────────────────────────
def test_preprocessor_transforms():
    df = pd.read_csv("data/heart_clean.csv")
    X = df.drop("HeartDisease", axis=1)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == X.shape[0], \
        "Row count changed after preprocessing"
    assert X_transformed.shape[1] > X.shape[1], \
        "Expected more columns after OneHotEncoding"


# ── Test 8: Model file exists ─────────────────────────────────────────────────
def test_model_file_exists():
    assert os.path.exists("models/best_model.pkl"), \
        "best_model.pkl not found in models/"


# ── Test 9: Model loads and predicts correctly ────────────────────────────────
def test_model_loads_and_predicts():
    model = joblib.load("models/best_model.pkl")
    sample = pd.DataFrame([{
        "Age": 54,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 130,
        "Cholesterol": 250,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 140,
        "ExerciseAngina": "Y",
        "Oldpeak": 1.5,
        "ST_Slope": "Flat"
    }])
    prediction = model.predict(sample)
    assert prediction[0] in [0, 1], \
        f"Prediction must be 0 or 1, got: {prediction[0]}"


# ── Test 10: Model probability is valid ──────────────────────────────────────
def test_model_probability_range():
    model = joblib.load("models/best_model.pkl")
    sample = pd.DataFrame([{
        "Age": 54,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 130,
        "Cholesterol": 250,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 140,
        "ExerciseAngina": "Y",
        "Oldpeak": 1.5,
        "ST_Slope": "Flat"
    }])
    proba = model.predict_proba(sample)[0]
    assert 0.0 <= proba[1] <= 1.0, \
        f"Probability out of range: {proba[1]}"