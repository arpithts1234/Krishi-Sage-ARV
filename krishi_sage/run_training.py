"""
run_training.py
---------------
Master script to:
  1. Load / generate the dataset
  2. Clean data
  3. Add engineered features
  4. Time-based train/test split
  5. Train Random Forest + XGBoost with hyperparameter tuning
  6. Evaluate both models
  7. Save best model, plots, and feature importance

Run:
    python run_training.py
"""

import sys
import os

# Fix emoji printing on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Make src/ importable
SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC)
DATA = os.path.join(os.path.dirname(__file__), "data")
sys.path.insert(0, DATA)

from data_loader import load_data
from preprocessing import clean_data, random_split
from feature_engineering import add_features
from train import train_models
from evaluate import evaluate_model, compare_models, feature_importance_plot

import pickle, os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def main():
    print("=" * 60)
    print("  🌾  CROP YIELD PREDICTION SYSTEM")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_data()

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    df = clean_data(df)
    print(f"   After cleaning: {len(df)} rows")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    df = add_features(df)
    print("   Features added: Year_Trend, Humidity_Norm, Temp_Rain_Interaction, Rain_Squared")

    # ── 4. Random split (70/15/15) ────────────────────────────────────────────
    train_df, val_df, test_df = random_split(df)

    # ── 5. Train models ───────────────────────────────────────────────────────
    # n_iter=15 keeps tuning fast (~2-3 min). Increase to 30+ for a final run.
    models = train_models(train_df, val_df)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, test_df, model_name=name)

    # ── 7. Compare & pick best ────────────────────────────────────────────────
    best_name = compare_models(results)

    # ── 8. Feature importance ─────────────────────────────────────────────────
    best_model = models[best_name]
    feature_importance_plot(best_model, model_name=best_name)

    # Save "best" model alias so the Streamlit app can load it
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(best_path, "wb") as f:
        pickle.dump({"name": best_name, "model": best_model}, f)
    print(f"\n💾  Best model ({best_name}) saved → {best_path}")

    print("\n✅  Training complete! You can now launch the Streamlit app:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    main()
