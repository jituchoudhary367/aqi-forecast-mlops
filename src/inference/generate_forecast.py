import pandas as pd
import joblib
import json
from datetime import datetime


FEATURES = ["temp", "humidity", "pressure", "wind_speed", "pm2_5", "pm10"]


def main():
    # Load latest row from dataset
    df = pd.read_csv("data/processed/dataset.csv")
    latest = df.sort_values("timestamp").iloc[-1]

    X = latest[FEATURES].to_frame().T

    model = joblib.load("models/aqi_model.pkl")

    pred = model.predict(X)[0]

    forecast = {
        "timestamp": datetime.utcnow().isoformat(),
        "predicted_aqi_next_day": float(pred),
    }

    today = datetime.utcnow().strftime("%Y-%m-%d")
    path = f"forecasts/forecast_{today}.json"

    with open(path, "w") as f:
        json.dump(forecast, f, indent=2)

    print("Forecast saved:", path)
    print(forecast)


if __name__ == "__main__":
    main()
