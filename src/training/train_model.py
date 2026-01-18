import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn


FEATURES = ["temp", "humidity", "pressure", "wind_speed", "pm2_5", "pm10"]
TARGET = "target_aqi"


def main():
    df = pd.read_csv("data/processed/train_dataset.csv")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, "models/aqi_model.pkl")

        print("RMSE:", rmse)
        print("Model saved to models/aqi_model.pkl")


if __name__ == "__main__":
    main()
