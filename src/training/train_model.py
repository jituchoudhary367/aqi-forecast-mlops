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
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target_aqi"])
    y = df["target_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Check if running in CI/CD environment
    is_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"RMSE: {rmse:.4f}")

    # Only use MLflow locally, skip in CI to avoid database issues
    if not is_ci:
        try:
            import mlflow
            import mlflow.sklearn
            
            mlflow.set_tracking_uri("file:./mlruns")
            
            with mlflow.start_run():
                mlflow.log_param("n_estimators", 300)
                mlflow.log_param("max_depth", 12)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(model, "model")
                print("MLflow tracking: enabled")
        except Exception as e:
            print(f"MLflow tracking skipped: {e}")
    else:
        print("MLflow tracking: disabled (CI environment)")

    # Save model to file (works in both CI and local)
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
