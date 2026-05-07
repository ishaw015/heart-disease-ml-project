import joblib
import pandas as pd

# load trained model
model = joblib.load('../models/best_model.pkl')

# sample patient data
sample_data = pd.DataFrame({

    'Age': [45],
    'Sex': ['M'],
    'ChestPainType': ['ASY'],
    'RestingBP': [130],
    'Cholesterol': [250],
    'FastingBS': [0],
    'RestingECG': ['Normal'],
    'MaxHR': [150],
    'ExerciseAngina': ['Y'],
    'Oldpeak': [1.5],
    'ST_Slope': ['Flat']

})

# prediction
prediction = model.predict(sample_data)

# probability
probability = model.predict_proba(sample_data)

print("Prediction:", prediction[0])

print("Heart Disease Probability:", probability[0][1])