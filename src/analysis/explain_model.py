"""
explain_model.py
----------------
Generates Explainable AI (XAI) plots using SHAP.
This satisfies the assignment requirement for Section 4 (SHAP/LIME).

Usage:
    python src/analysis/explain_model.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────
PROCESSED_FILE = os.path.join("data", "processed", "mobile_data_processed.csv")
MODELS_DIR     = os.path.join("outputs", "models")
MODEL_PATH     = os.path.join(MODELS_DIR, "best_model.joblib")
ENCODERS_PATH  = os.path.join(MODELS_DIR, "encoders.joblib")
OUTPUT_DIR     = os.path.join("outputs", "analysis")
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_shap_analysis():
    print("Starting SHAP Explainability Analysis...")

    # 1. Load Resources
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        print(f"Error: Model files not found at {MODELS_DIR}. Run training first.")
        return

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    df = pd.read_csv(PROCESSED_FILE)

    # 2. Prepare Feature Data (Same as train_model.py)
    features = ["brand", "brand_version", "condition", "location", "is_brand_new", "title_length", "storage_gb"]
    
    # We need the numeric encoded version for SHAP
    X_enc = df.copy()
    X_enc["is_brand_new"] = (X_enc["condition"] == "Brand New").astype(int)
    X_enc["title_length"] = X_enc["title"].str.len()
    X_enc["brand_version"] = X_enc["brand_version"].fillna("Generic")
    X_enc["storage_gb"]    = X_enc["storage_gb"].fillna(0)

    for col in ["brand", "brand_version", "condition", "location"]:
        le = encoders[col]
        X_enc[col] = le.transform(X_enc[col].fillna("Unknown").astype(str))

    X = X_enc[features]

    # 3. Create SHAP Explainer
    # For XGBoost/LightGBM, we use TreeExplainer
    print("Calculating SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 4. Global Explainability: Summary Plot
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"))
    plt.close()

    # 5. Local Explainability: Waterfall plot for a specific sample
    # Let's pick a high-end sample (e.g., an iPhone or expensive model)
    sample_idx = 0 
    print(f"Generating Local Explanation for sample index {sample_idx}...")
    
    plt.figure(figsize=(12, 4))
    # Note: For TreeExplainer, shap_values is sometimes a list for multi-class, 
    # but for regression it's a 2D array.
    if isinstance(shap_values, list):
        val = shap_values[0][sample_idx]
    else:
        val = shap_values[sample_idx]
        
    shap.plots.bar(shap.Explanation(
        values=val, 
        base_values=explainer.expected_value, 
        data=X.iloc[sample_idx], 
        feature_names=features
    ), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_local_explanation.png"))
    plt.close()

    print(f"Analysis complete! Plots saved to '{OUTPUT_DIR}'")
    print(f"   1. Global View: shap_summary.png")
    print(f"   2. Local View:  shap_local_explanation.png")

if __name__ == "__main__":
    run_shap_analysis()
