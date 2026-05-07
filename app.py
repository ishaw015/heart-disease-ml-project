import streamlit as st
import requests

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("❤️ Heart Disease Prediction")
st.markdown("Enter patient details below to predict heart disease risk.")
st.divider()

# ── Input form ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=54)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting BP", min_value=0, max_value=300, value=130)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=250)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120mg/dl", [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", min_value=0, max_value=300, value=140)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=10.0, value=1.5)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

# ── Predict button ────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):
    payload = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )
        result = response.json()

        st.divider()

        # ── Result display ────────────────────────────────────
        if result["prediction"] == 1:
            st.error(f"🚨 {result['label']}")
        else:
            st.success(f"✅ {result['label']}")

        # ── Metrics ───────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", result["prediction"])
        col2.metric("Probability", f"{result['probability']*100:.1f}%")
        col3.metric("Risk Level", result["risk_level"])

        # ── Risk bar ──────────────────────────────────────────
        st.markdown("#### Risk Probability")
        st.progress(result["probability"])

    except Exception as e:
        st.error(f"API Error: {e}. Make sure FastAPI is running!")