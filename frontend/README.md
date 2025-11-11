# Streamlit Frontend

## Quick Start

1. Install dependencies (create a venv if you prefer):
   ```powershell
   python -m pip install streamlit
   ```
2. Launch the dashboard from the project root:
   ```powershell
   streamlit run frontend/streamlit_app.py
   ```
3. The app reads the latest model metadata in `models/main_model_meta.json` and the stratified preview predictions in `data/processed/lol_15min_preview_predictions.csv`.

## What You Get

- Headline hold-out metrics and preview metrics side-by-side.
- Ranked feature weight summary to support the Explainable AI narrative.
- Interactive table of the classroom preview sample with win probabilities.
- Per-match breakdown of feature contributions so you can walk the teacher through *why* the model likes blue or red.
