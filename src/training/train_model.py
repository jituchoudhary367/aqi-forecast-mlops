import pandas as pd
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
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

    # Configure MLflow for CI/CD environment
    # Use sqlite backend to avoid filesystem path issues in GitHub Actions
    is_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"
    
    if is_ci:
        # In CI, use a simple sqlite database
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    else:
        # Locally, use the default ./mlruns directory
        mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 12)
        mlflow.log_metric("rmse", rmse)
        
        # Only log model artifacts if not in CI (to avoid path issues)
        if not is_ci:
            mlflow.sklearn.log_model(model, "model")

        print("RMSE:", rmse)

        MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(model, MODEL_PATH)
        print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
