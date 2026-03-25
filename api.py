from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load models
rf = joblib.load("rf.pkl")
xgb = joblib.load("xgb.pkl")
columns = joblib.load("columns.pkl")

@app.get("/")
def home():
    return {"message": "Hemophilia Risk API Running"}

@app.get("/predict")
def predict(age: int, dose: int, exposure: int):
    
    # Create input
    data = {
        "mutation_type": "intron22",
        "exon": 22,
        "severity": "severe",
        "age_first_treatment": age,
        "dose_intensity": dose,
        "exposure_days": exposure
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    for col in columns:
        if col not in df:
            df[col] = 0

    df = df[columns]

    p1 = rf.predict_proba(df)[0][1]
    p2 = xgb.predict_proba(df)[0][1]

    risk = (p1 + p2) / 2

    return {"risk_score": float(risk)}