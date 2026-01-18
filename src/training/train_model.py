import pandas as pd
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

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
        mlflow.sklearn.log_model(model, "model")

        print("RMSE:", rmse)

        MODEL_PATH.parent.mkdir(exist_ok=True, parents=True) # Ensure parents exist
        joblib.dump(model, MODEL_PATH)
        print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
