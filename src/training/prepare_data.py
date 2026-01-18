import pandas as pd
from pathlib import Path

# Use the dataset that contains all available history (backfill + new)
RAW_PATH = Path("data/processed/dataset.csv") 
OUT_PATH = Path("data/processed/train_dataset.csv")


def main():
    if not RAW_PATH.exists():
        print(f"Error: {RAW_PATH} not found. Run backfill_data.py first.")
        return

    df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    print(f"Loading data from {RAW_PATH}, shape: {df.shape}")

    # --- temporal features ---
    # Lag 1: Yesterday's PM2.5
    df["pm2_5_lag1"] = df["pm2_5"].shift(1)
    
    # Rolling stats: Mean of last 3 days
    df["pm2_5_roll3"] = df["pm2_5"].rolling(3).mean()
    df["pm10_roll3"] = df["pm10"].rolling(3).mean()

    # target = next-day AQI
    # We want to predict tomorrow's AQI using today's features
    df["target_aqi"] = df["aqi"].shift(-1)

    # Drop rows with NaN (first few rows due to rolling, last row due to target shift)
    df = df.dropna()

    features = [
        "temp", "humidity", "pressure", "wind_speed",
        "pm2_5", "pm10",
        "pm2_5_lag1", "pm2_5_roll3", "pm10_roll3"
    ]

    final = df[features + ["target_aqi"]]
    final.to_csv(OUT_PATH, index=False)

    print("Training dataset saved:", OUT_PATH, "Shape:", final.shape)
    print("Columns:", final.columns.tolist())


if __name__ == "__main__":
    main()
