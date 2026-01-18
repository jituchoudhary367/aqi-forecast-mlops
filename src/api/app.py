from fastapi import FastAPI
import joblib
import pandas as pd
import json
from pathlib import Path

app = FastAPI(title="AQI Forecast API")

MODEL_PATH = Path("models/aqi_model.pkl")
DATASET_PATH = Path("data/processed/dataset.csv")


def load_model():
    return joblib.load(MODEL_PATH)


@app.get("/")
def health():
    return {"status": "ok", "message": "AQI Forecast API running"}


@app.get("/predict")
def predict():
    df = pd.read_csv(DATASET_PATH)
    latest = df.sort_values("timestamp").iloc[-1]

    FEATURES = ["temp", "humidity", "pressure", "wind_speed", "pm2_5", "pm10"]
    X = latest[FEATURES].to_frame().T

    model = load_model()
    pred = model.predict(X)[0]

    return {
        "predicted_aqi_next_day": float(pred),
        "latest_features": latest[FEATURES].to_dict()
    }
