---
title: AQI Forecast MLOps
emoji: 🌬️
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# AQI Forecast MLOps

This project implements an end-to-end MLOps pipeline for AQI forecasting.
It exposes a FastAPI backend for predictions and a Streamlit UI for visualization.

## Running Locally

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   bash start.sh
   ```
