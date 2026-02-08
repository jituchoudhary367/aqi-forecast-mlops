import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path


def main():
    # Load the full dataset to calculate temporal features
    dataset_path = Path("data/processed/dataset.csv")
    
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found. Run build_dataset.py first.")
        return
    
    df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    
    print(f"Loaded dataset with {len(df)} records")
    
    # Calculate temporal features (MUST match training features exactly)
    df["pm2_5_lag1"] = df["pm2_5"].shift(1)
    df["pm2_5_roll3"] = df["pm2_5"].rolling(3).mean()
    df["pm10_roll3"] = df["pm10"].rolling(3).mean()
    
    # Get the latest row with valid features
    latest = df.iloc[-1]
    
    # Check if we have enough history for rolling features
    if pd.isna(latest["pm2_5_roll3"]):
        print("Error: Not enough historical data (need at least 3 days)")
        return
    
    # Feature list MUST match prepare_data.py exactly
    FEATURES = [
        "temp", "humidity", "pressure", "wind_speed",
        "pm2_5", "pm10",
        "pm2_5_lag1", "pm2_5_roll3", "pm10_roll3"
    ]
    
    # Prepare input for model
    X = latest[FEATURES].to_frame().T
    
    print(f"Input features:\n{X.to_dict('records')[0]}")
    
    # Load model and predict
    model_path = Path("models/aqi_model.pkl")
    if not model_path.exists():
        print(f"Error: {model_path} not found. Train model first.")
        return
        
    model = joblib.load(model_path)
    pred = model.predict(X)[0]
    
    # Create forecast output
    forecast = {
        "timestamp": datetime.utcnow().isoformat(),
        "predicted_aqi_next_day": float(pred),
        "input_timestamp": str(latest["timestamp"]),
    }
    
    # Save forecast
    today = datetime.utcnow().strftime("%Y-%m-%d")
    forecast_dir = Path("forecasts")
    forecast_dir.mkdir(exist_ok=True)
    
    path = forecast_dir / f"forecast_{today}.json"
    
    with open(path, "w") as f:
        json.dump(forecast, f, indent=2)
    
    print(f"✓ Forecast saved: {path}")
    print(f"✓ Predicted AQI: {pred:.2f}")


if __name__ == "__main__":
    main()
