# Dashboard setup

1. Install Python 3.10+.
2. In this folder run `pip install -r requirements.txt`.
3. Run `streamlit run app.py`.
4. Open the local URL shown in the terminal (usually http://localhost:8501).

The app includes synthetic demo mode and CSV upload. It is not yet connected to ESP32 or the camera. The pattern index is illustrative, not a trained or clinically validated depression-risk model.

CSV columns required: `timestamp`, `heart_rate_bpm`, `eda_us`, `activity_index`. Optional: `signal_quality` (0–1), `facial_feature_score` (0–1).
