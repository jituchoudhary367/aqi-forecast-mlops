import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT")
LON = os.getenv("LON")

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"


def fetch_weather():
    params = {"lat": LAT, "lon": LON, "appid": API_KEY, "units": "metric"}
    r = requests.get(WEATHER_URL, params=params)
    r.raise_for_status()
    return r.json()


def fetch_pollution():
    params = {"lat": LAT, "lon": LON, "appid": API_KEY}
    r = requests.get(POLLUTION_URL, params=params)
    r.raise_for_status()
    return r.json()


def main():
    weather = fetch_weather()
    pollution = fetch_pollution()

    record = {
        "timestamp": datetime.utcnow(),
        "temp": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": weather["wind"]["speed"],
        "pm2_5": pollution["list"][0]["components"]["pm2_5"],
        "pm10": pollution["list"][0]["components"]["pm10"],
        "aqi": pollution["list"][0]["main"]["aqi"],
    }

    df = pd.DataFrame([record])

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    path = f"data/raw/{date_str}.csv"
    df.to_csv(path, index=False)

    print("Saved:", path)


if __name__ == "__main__":
    main()
