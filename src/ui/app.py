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
if st.button("Generate Forecast", type="primary"):
    with st.spinner("Analyzing atmospheric conditions..."):
        try:
            # 1. Get latest data row
            latest = df.sort_values("timestamp").iloc[-1]
            
            # 2. Select features (Must match model training features exactly)
            FEATURES = ["temp", "humidity", "pressure", "wind_speed", "pm2_5", "pm10"]
            input_data = latest[FEATURES].to_frame().T
            
            # 3. Predict
            raw_pred = model.predict(input_data)[0]
            
            # 4. Map Prediction to Human-Readable Label
            # Clamp value between 1 and 5 just in case
            pred_class = int(round(raw_pred))
            pred_class = max(1, min(5, pred_class))
            
            label, approx_aqi = AQI_INFO.get(pred_class, ("Unknown", 0))

            # --- Display Results ---
            st.divider()
            
            # Primary Metric
            st.subheader("📉 Tomorrow's Forecast")
            
            # Color logic for visual impact
            color_map = {
                1: "green", 
                2: "#9acd32", # yellowgreen
                3: "orange", 
                4: "red", 
                5: "#8b0000" # darkred
            }
            color = color_map.get(pred_class, "grey")

            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                <h2 style="margin:0;">{label}</h2>
                <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">Expect AQI ~ {approx_aqi}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"Model Raw Output Class: {pred_class} (mapped from {raw_pred:.2f})")

            # Context: Latest Conditions
            st.divider()
            st.subheader("📡 Based on Current Conditions")
            col1, col2, col3 = st.columns(3)
            col1.metric("PM 2.5", f"{latest['pm2_5']:.1f}")
            col2.metric("Temp", f"{latest['temp']:.1f} °C")
            col3.metric("Humidity", f"{latest['humidity']:.1f}%")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
