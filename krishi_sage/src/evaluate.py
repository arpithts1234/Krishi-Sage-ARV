"""
src/evaluate.py
---------------
Evaluates trained models on the test set.

Metrics explained:
  RMSE (Root Mean Squared Error)
       How far off (on average) are our predictions?
       In the same unit as Yield (kg/ha).
       Lower = better.

  MAE (Mean Absolute Error)
       Average absolute difference between predicted and actual yield.
       More intuitive than RMSE; less sensitive to outliers.
       Lower = better.

  R² Score (R-squared / Coefficient of Determination)
       ─── SIMPLE EXPLANATION FOR VIVA ───
       Imagine you had no model and just predicted the AVERAGE yield
       for every farm. R² tells you how much BETTER your model is
       compared to that "dummy average" predictor.

       R² = 1.0  → perfect prediction
       R² = 0.0  → no better than guessing the average
       R² < 0    → even worse than guessing the average!

       Example: R² = 0.88 means your model explains 88 % of the
       variation in yield that the simple average cannot explain.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving files
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from feature_engineering import FEATURE_COLS, TARGET_COL

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(PLOTS_DIR, exist_ok=True)


def evaluate_model(model, test_df: pd.DataFrame, model_name: str = "Model"):
    """
    Returns a dict with RMSE, MAE, R2 and prints a summary.
    Also saves two plots: actual vs predicted, and residuals.
    """
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n📊  {model_name} Evaluation")
    print(f"   RMSE : {rmse:>10.2f} kg/ha")
    print(f"   MAE  : {mae:>10.2f} kg/ha")
    print(f"   R²   : {r2:>10.4f}")

    # ── Plot 1: Actual vs Predicted ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.3, s=10, color="#2196F3")
    mn, mx = y_test.min(), y_test.max()
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Yield (kg/ha)")
    axes[0].set_ylabel("Predicted Yield (kg/ha)")
    axes[0].set_title(f"{model_name} – Actual vs Predicted")
    axes[0].legend()

    # ── Plot 2: Residuals ────────────────────────────────────────────────────
    residuals = y_test.values - y_pred
    axes[1].hist(residuals, bins=50, color="#4CAF50", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual (Actual – Predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{model_name} – Residual Distribution")

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_eval.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"   📈  Plot saved → {plot_path}")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "y_pred": y_pred}


def compare_models(results: dict):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 50)
    print("  MODEL COMPARISON")
    print("=" * 50)
    print(f"  {'Model':<20} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
    print("-" * 50)
    for name, m in results.items():
        print(f"  {name:<20} {m['RMSE']:>10.2f} {m['MAE']:>10.2f} {m['R2']:>8.4f}")
    print("=" * 50)

    # Best by RMSE
    best = min(results, key=lambda k: results[k]["RMSE"])
    print(f"\n  🏆  Best model (lowest RMSE): {best}")
    return best


def feature_importance_plot(model, model_name: str = "XGBoost"):
    """
    Saves a horizontal bar chart of top-20 feature importances.
    Works for both RandomForest and XGBoost pipelines.
    """
    try:
        estimator = model.named_steps["model"]
        importances = estimator.feature_importances_

        # Get feature names AFTER the preprocessor transforms them
        preprocessor = model.named_steps["preprocessor"]
        try:
            ohe_names = (preprocessor
                         .named_transformers_["cat"]
                         .named_steps["encoder"]
                         .get_feature_names_out(["Crop", "Season", "State"]))
        except Exception:
            ohe_names = []

        from feature_engineering import FEATURE_COLS
        CAT_COLS = ["Crop", "Season", "State"]
        NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]
        all_names = list(NUM_COLS) + list(ohe_names)

        if len(all_names) != len(importances):
            all_names = [f"f{i}" for i in range(len(importances))]

        feat_df = (pd.DataFrame({"Feature": all_names, "Importance": importances})
                     .sort_values("Importance", ascending=False)
                     .head(20))

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color="#FF9800")
        ax.set_xlabel("Importance Score")
        ax.set_title(f"{model_name} – Top 20 Feature Importances")
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"{model_name.lower()}_feature_importance.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"   📊  Feature importance plot saved → {path}")

    except Exception as e:
        print(f"   ⚠️  Could not generate feature importance: {e}")
