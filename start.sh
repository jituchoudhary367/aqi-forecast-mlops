echo "Starting FastAPI..."
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit..."
streamlit run src/ui/app.py --server.port 7860 --server.address 0.0.0.0
