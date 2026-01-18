import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="AQI Forecaster", layout="centered")

st.title("🌍 Air Quality Forecast Dashboard")

st.write("Predicting next-day AQI using weather and pollution data.")

if st.button("Get Forecast"):
    try:
        r = requests.get(API_URL, timeout=5)
        data = r.json()

        st.subheader("📈 Predicted AQI (Next Day)")
        st.metric("AQI", round(data["predicted_aqi_next_day"], 2))

        st.subheader("🌦 Latest Conditions")
        st.json(data["latest_features"])

    except Exception as e:
        st.error(f"API not reachable: {e}")
