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
    # Scale PM2.5 to be more realistic (e.g. 20-100 range)
    df["pm2_5"] = df["temp"] * 2.0 + df["wind_speed"] * 1.5 + 20
    df["pm10"] = df["pm2_5"] * 1.8
    
    # Calculate AQI (approximate)
    # Real AQI calculation is complex, here we use a simple linear proxy for the demo
    # AQI ~ PM2.5 * 3 (roughly) for indian context often works as a naive proxy
    df["aqi"] = (df["pm2_5"] * 3).astype(int) 
    # Clamp to realistic range
    df["aqi"] = df["aqi"].clip(lower=20, upper=500)

    df.to_csv("data/raw/backfill.csv", index=False)
    df.to_csv("data/processed/dataset.csv", index=False)

    print("Backfilled rows:", len(df))


if __name__ == "__main__":
    main()
