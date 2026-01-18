import pandas as pd
import requests
from datetime import datetime, timedelta


LAT = 23.0225
LON = 72.5714


def fetch_weather_history(start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,pressure_msl_mean,wind_speed_10m_mean",
        "timezone": "UTC"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def main():
    end = datetime.utcnow().date()
    start = end - timedelta(days=30)

    weather = fetch_weather_history(start.isoformat(), end.isoformat())

    df = pd.DataFrame({
        "timestamp": weather["daily"]["time"],
        "temp": weather["daily"]["temperature_2m_mean"],
        "humidity": weather["daily"]["relative_humidity_2m_mean"],
        "pressure": weather["daily"]["pressure_msl_mean"],
        "wind_speed": weather["daily"]["wind_speed_10m_mean"],
    })

    # fake AQI proxy using PM relationship (temporary)
    df["pm2_5"] = df["temp"] * 1.5
    df["pm10"] = df["temp"] * 2.5
    df["aqi"] = (df["pm2_5"] / 10).astype(int)

    df.to_csv("data/raw/backfill.csv", index=False)
    df.to_csv("data/processed/dataset.csv", index=False)

    print("Backfilled rows:", len(df))


if __name__ == "__main__":
    main()
