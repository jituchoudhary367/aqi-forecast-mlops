import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- Configuration & Safety ---
st.set_page_config(page_title="AQI Forecaster", layout="centered", page_icon="🌬️")

# Paths (adapt to where files land in Docker/HF)
MODEL_PATH = Path("models/aqi_model.pkl")
DATASET_PATH = Path("data/processed/dataset.csv")

# AQI Mapping: Class -> (Label, Representative AQI Value)
AQI_INFO = {
    1: ("Good", 25),              # Range: 0-50
    2: ("Satisfactory", 75),      # Range: 51-100
    3: ("Moderate", 150),         # Range: 101-200
    4: ("Poor", 250),             # Range: 201-300
    5: ("Severe", 400),           # Range: 301-500
}

# --- Load Resources ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        return None

def load_data():
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return None

# --- UI Layout ---
st.title("🌍 Air Quality Forecast Dashboard")
st.markdown("**Predicting next-day AQI using weather and historical data.**")

# Load model/data
model = load_resources()
df = load_data()

if model is None or df is None:
    st.error("🚨 System Error: Model or Data file not found.")
    st.stop()

# --- Main Logic ---
# --- Main Logic ---
if st.button("Generate Forecast", type="primary"):
    with st.spinner("Analyzing atmospheric conditions..."):
        try:
            # 1. Preprocess Data (Calculate temporal features)
            # We need to sort and calculate features on the whole dataframe to get the correct rolling values for the latest entry
            df = df.sort_values("timestamp")
            
            # Create temporal features (same as training)
            df["pm2_5_lag1"] = df["pm2_5"].shift(1)
            df["pm2_5_roll3"] = df["pm2_5"].rolling(3).mean()
            df["pm10_roll3"] = df["pm10"].rolling(3).mean()
            
            # Get latest row with valid rolling features
            latest = df.iloc[-1]
            
            # Check for NaN (if not enough history)
            if pd.isna(latest["pm2_5_roll3"]):
                st.error("Not enough historical data to generate forecast (need at least 3 days).")
                st.stop()

            # 2. Select features (Must match model training features EXACTLY)
            FEATURES = [
                "temp", "humidity", "pressure", "wind_speed",
                "pm2_5", "pm10",
                "pm2_5_lag1", "pm2_5_roll3", "pm10_roll3"
            ]
            
            input_data = latest[FEATURES].to_frame().T
            
            # 3. Predict (Model now outputs actual AQI value, e.g., 145.6)
            predicted_aqi = model.predict(input_data)[0]
            
            # 4. Map AQI Value to Label (Standard Limits)
            # 0-50: Good, 51-100: Satisfactory, 101-200: Moderate, 201-300: Poor, 301-400: Very Poor, 401+: Severe
            aqi_val = int(round(predicted_aqi))
            
            if aqi_val <= 50:
                label = "Good"
                color = "green"
            elif aqi_val <= 100:
                label = "Satisfactory"
                color = "#9acd32" # yellowgreen
            elif aqi_val <= 200:
                label = "Moderate"
                color = "orange"
            elif aqi_val <= 300:
                label = "Poor"
                color = "red"
            elif aqi_val <= 400:
                label = "Very Poor"
                color = "#8b0000" # darkred
            else:
                label = "Severe"
                color = "#800000" # maroon
            
            # --- Display Results ---
            st.divider()
            
            # Primary Metric
            st.subheader("📉 Tomorrow's Forecast")

            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                <h2 style="margin:0;">{label}</h2>
                <p style="margin:0; font-size: 2rem; font-weight: bold;">AQI {aqi_val}</p>
                <p style="margin:0; opacity: 0.9;">Predicted Index</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Context: Latest Conditions
            st.divider()
            st.subheader("📡 Based on Current Conditions")
            col1, col2, col3 = st.columns(3)
            col1.metric("PM 2.5", f"{latest['pm2_5']:.1f}")
            col2.metric("Temp", f"{latest['temp']:.1f} °C")
            col3.metric("Humidity", f"{latest['humidity']:.1f}%")
            
            with st.expander("Show detailed features"):
                st.write(latest[FEATURES])
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
