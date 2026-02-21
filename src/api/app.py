"""
app.py
------
Flask backend for the Mobile Price Predictor.
Loads the saved LightGBM model and encoders to provide real-time estimates.

Usage:
    python src/api/app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add the project root to sys.path so we can import modules if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

app = Flask(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
PROCESSED_FILE = os.path.join("data", "processed", "mobile_data_processed.csv")
MODELS_DIR = os.path.join("outputs", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.joblib")

# ─── Global Variables for Model/Encoders ──────────────────────────────────────
model = None
encoders = None

def load_resources():
    global model, encoders
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        return True
    return False

def get_spec_mapping():
    """Analyze the processed data to create a brand -> specs mapping."""
    if not os.path.exists(PROCESSED_FILE):
        return {}
    
    df = pd.read_csv(PROCESSED_FILE)
    mapping = {}
    
    for brand in df["brand"].unique():
        brand_df = df[df["brand"] == brand]
        mapping[brand] = {
            "versions": sorted(brand_df["brand_version"].dropna().unique().tolist()),
            "storages": sorted([int(x) for x in brand_df["storage_gb"].dropna().unique() if x > 0])
        }
    return mapping

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if encoders is None:
        if not load_resources():
            return "<h1>Model files not found. Please run 'python src/modeling/train_model.py' first.</h1>"
    
    spec_mapping = get_spec_mapping()
    
    options = {
        "brands": sorted(encoders["brand"].classes_.tolist()),
        "conditions": sorted(encoders["condition"].classes_.tolist()),
        "locations": sorted(encoders["location"].classes_.tolist()),
        "mapping": spec_mapping
    }
    return render_template("index.html", options=options)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or encoders is None:
        return jsonify({"status": "error", "message": "Model or encoders not loaded"}), 500
    
    try:
        data = request.json or {}
        
        # 1. Extraction with defaults
        brand         = data.get("brand", "Other")
        brand_version = data.get("brand_version", "Generic")
        condition     = data.get("condition", "Used")
        location      = data.get("location", "Colombo")
        
        # Safe numeric conversion
        try:
            storage_gb = float(data.get("storage_gb", 0))
        except (ValueError, TypeError):
            storage_gb = 0.0
            
        is_brand_new = 1 if condition == "Brand New" else 0
        title_length = 25

        # 2. Enhanced safe_encode logic
        def safe_encode(encoder_name, value):
            if encoder_name not in encoders:
                return 0
            
            le = encoders[encoder_name]
            val_str = str(value).strip() if value else "Unknown"
            
            # Direct match
            if val_str in le.classes_:
                return le.transform([val_str])[0]
            
            # Fallback search
            for fallback in [val_str.upper(), val_str.title(), "Other", "Unknown", "Generic"]:
                if fallback in le.classes_:
                    return le.transform([fallback])[0]
            
            return 0 # Final fallback to first class index

        brand_enc   = safe_encode("brand", brand)
        version_enc = safe_encode("brand_version", brand_version)
        cond_enc    = safe_encode("condition", condition)
        loc_enc     = safe_encode("location", location)

        # 3. Predict (XGBoost expects 2D array)
        features = np.array([[brand_enc, version_enc, cond_enc, loc_enc, is_brand_new, title_length, storage_gb]])
        prediction_log = model.predict(features)[0]
        
        # Result conversion
        prediction_price = np.expm1(prediction_log)
        
        return jsonify({
            "status": "success",
            "prediction": int(round(prediction_price, -1))
        })

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"PREDICTION ERROR:\n{error_msg}")
        return jsonify({
            "status": "error", 
            "message": "Internal processing error. Check server logs."
        }), 400

if __name__ == "__main__":
    if load_resources():
        print(f"Model loaded successfully. Starting server at http://127.0.0.1:5000")
        app.run(debug=True, port=5000)
    else:
        print("ERROR: Could not load model files. Run training first.")
