from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ── Initialize app ────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-powered REST API for heart disease prediction using Random Forest",
    version="1.0.0"
)

# ── Load model ────────────────────────────────────────────────
model = joblib.load("models/best_model.pkl")

# ── Input schema ──────────────────────────────────────────────
class PatientData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Random Forest",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(patient: PatientData):
    # Convert input to dataframe
    input_df = pd.DataFrame([patient.dict()])
    
    # Get prediction and probability
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    
    return {
        "prediction": prediction,
        "label": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "probability": round(probability, 4),
        "risk_level": (
            "High" if probability >= 0.7
            else "Medium" if probability >= 0.4
            else "Low"
        )
    }