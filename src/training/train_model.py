import pandas as pd
from pathlib import Path
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

DATA_PATH = Path("data/processed/train_dataset.csv")
MODEL_PATH = Path("models/aqi_model.pkl")


def main():
    print("Starting model training...")
    
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded: {df.shape}")

    X = df.drop(columns=["target_aqi"])
    y = df["target_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"✅ Model trained successfully. RMSE: {rmse:.4f}")

    # Save model
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
