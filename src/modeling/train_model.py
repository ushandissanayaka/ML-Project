"""
train_model.py
--------------
Train a mobile phone price prediction model using the processed dataset.
Upgraded version with XGBoost, LightGBM, and Log Transformation.

Reads  : data/processed/mobile_data_processed.csv
Outputs: outputs/metrics/model_results.txt
         outputs/plots/actual_vs_predicted.png
         outputs/plots/feature_importance.png
         outputs/plots/residuals.png

Usage:
    python src/modeling/train_model.py
"""

import logging
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import lightgbm as lgb

# ─── Configuration ────────────────────────────────────────────────────────────
PROCESSED_FILE = os.path.join("data", "processed", "mobile_data_processed.csv")
METRICS_FILE   = os.path.join("outputs", "metrics", "model_results.txt")
MODELS_DIR     = os.path.join("outputs", "models")
PLOTS_DIR      = os.path.join("outputs", "plots")
LOG_DIR        = "logs"
RANDOM_STATE   = 42
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "train_model.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─── Feature preparation ──────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """Select and encode features for the model."""
    # Removed ram_gb
    features = ["brand", "brand_version", "condition", "location", "is_brand_new", "title_length", "storage_gb"]
    target   = "price"

    # Keep only rows where all features and target exist
    subset = df[features + [target]].copy()
    subset = subset.dropna(subset=[target])

    # Fill missing Storage and Brand Version
    subset["storage_gb"] = subset["storage_gb"].fillna(0)
    subset["brand_version"] = subset["brand_version"].fillna("Generic")

    # Label-encode categorical columns
    cat_cols = ["brand", "brand_version", "condition", "location"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        subset[col] = le.fit_transform(subset[col].fillna("Unknown").astype(str))
        encoders[col] = le

    X = subset[features]
    y = subset[target].astype(float)
    
    # Log transformation for target (Price) to handle outliers and scale
    y_log = np.log1p(y)
    
    return X, y, y_log, encoders


# ─── Model training ───────────────────────────────────────────────────────────

def train_and_evaluate(X, y_true, y_log):
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Get the original prices for the test set evaluation
    y_test_true = np.expm1(y_test_log)

    models = {
        "Ridge Regression":          Ridge(alpha=1.0),
        "Random Forest":             RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":                    xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE),
        "LightGBM":                  lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE, verbose=-1),
    }

    results = {}
    best_model = None
    best_r2_log = -float("inf")

    for name, model in models.items():
        logger.info(f"Training {name} ...")
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        
        # Convert predictions back from log scale
        y_pred_true = np.expm1(y_pred_log)

        # Metrics on original scale
        mae   = mean_absolute_error(y_test_true, y_pred_true)
        rmse  = root_mean_squared_error(y_test_true, y_pred_true)
        r2    = r2_score(y_test_true, y_pred_true)
        
        # Metrics on log scale (for model selection)
        r2_log = r2_score(y_test_log, y_pred_log)

        results[name] = {
            "model": model, 
            "y_pred": y_pred_true, 
            "MAE": mae, 
            "RMSE": rmse, 
            "R2": r2,
            "R2_log": r2_log
        }
        logger.info(f"  {name}: Original MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.4f} (Log-R2={r2_log:.4f})")

        if r2_log > best_r2_log:
            best_r2_log = r2_log
            best_model = name

    return results, best_model, X_test, y_test_true, X_train.columns.tolist()


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, y_pred, model_name: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="none", s=20, color="#4C72B0")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Price (LKR)")
    ax.set_ylabel("Predicted Price (LKR)")
    ax.set_title(f"Actual vs Predicted (Improved) — {model_name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {path}")


def plot_residuals(y_test, y_pred, model_name: str):
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, edgecolors="none", s=20, color="#DD8452")
    ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted Price (LKR)")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residual Plot (Improved) — {model_name}")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "residuals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {path}")


def plot_feature_importance(model, feature_names: list, model_name: str):
    # Try different ways models store importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        logger.info(f"{model_name} doesn't have internal importance — skipping plot")
        return

    indices = np.argsort(importances)[::-1]
    names   = [feature_names[i] for i in indices]
    vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], vals[::-1], color="#55A868")
    ax.set_xlabel("Importance Scale")
    ax.set_title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Plot saved: {path}")


# ─── Save metrics ─────────────────────────────────────────────────────────────

def save_metrics(results: dict, best_model: str):
    lines = ["=" * 60, "  UPGRADED MOBILE PRICE MODEL — RESULTS (with Log Transform)", "=" * 60, ""]
    for name, res in results.items():
        marker = "  * BEST" if name == best_model else ""
        lines += [
            f"  Model : {name}{marker}",
            f"  MAE   : {res['MAE']:>12,.0f} LKR",
            f"  RMSE  : {res['RMSE']:>12,.0f} LKR",
            f"  R2    : {res['R2']:>12.4f}",
            f"  Log-R2: {res['R2_log']:>12.4f}",
            "",
        ]
    lines.append("=" * 60)
    text = "\n".join(lines)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    logger.info(f"Metrics saved to '{METRICS_FILE}'")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_FILE):
        logger.error(f"Processed file not found: '{PROCESSED_FILE}'")
        logger.error("Run preprocess_data.py first.")
        sys.exit(1)

    logger.info(f"Loading '{PROCESSED_FILE}' ...")
    df = pd.read_csv(PROCESSED_FILE)
    logger.info(f"  {len(df):,} records loaded")

    X, y_true, y_log, encoders = prepare_features(df)
    logger.info(f"  Feature matrix: {X.shape}")

    results, best_model, X_test, y_test_true, feature_names = train_and_evaluate(X, y_true, y_log)

    best = results[best_model]
    plot_actual_vs_predicted(y_test_true, best["y_pred"], best_model)
    plot_residuals(y_test_true, best["y_pred"], best_model)
    plot_feature_importance(best["model"], feature_names, best_model)

    save_metrics(results, best_model)

    # ── Save best model and encoders for the web interface ──────────
    logger.info(f"Saving '{best_model}' as the representative project model...")
    joblib.dump(best["model"], os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.joblib"))
    logger.info(f"Model and encoders saved to '{MODELS_DIR}'")

    print(f"\nDone. Upgraded Training complete. Best model: {best_model}")
    print(f"Model saved to '{MODELS_DIR}' for the web interface.")
