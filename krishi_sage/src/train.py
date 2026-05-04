"""
src/train.py
------------
Builds sklearn Pipelines for:
  - Random Forest Regressor
  - XGBoost Regressor

Each pipeline:
  1. Imputes missing values
  2. Encodes categorical columns
  3. Scales numerical columns
  4. Fits the model

Hyperparameter tuning is done via RandomizedSearchCV.
Best model is selected by lowest RMSE on the test set and saved to disk.
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from xgboost import XGBRegressor

from feature_engineering import FEATURE_COLS, TARGET_COL

# ── Where to save models ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Column categories ─────────────────────────────────────────────────────────
CAT_COLS = ["Crop", "Season", "State"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]


def build_preprocessor():
    """
    ColumnTransformer that handles:
      - Numerical: impute with median → standard scale
      - Categorical: impute with 'missing' → one-hot encode
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUM_COLS),
        ("cat", cat_pipeline, CAT_COLS),
    ])

    return preprocessor


def build_rf_pipeline():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])


def build_xgb_pipeline():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric="rmse",
        )),
    ])


# ── Hyperparameter search spaces ──────────────────────────────────────────────
RF_PARAM_DIST = {
    "model__n_estimators":      [100, 200, 300],
    "model__max_depth":         [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf":  [1, 2, 4],
    "model__max_features":      ["sqrt", "log2", 0.8],
}

XGB_PARAM_DIST = {
    "model__n_estimators":  [100, 200, 300],
    "model__max_depth":     [3, 5, 7, 9],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample":     [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__reg_alpha":     [0, 0.1, 0.5],
    "model__reg_lambda":    [1, 1.5, 2],
}


def tune_and_train(pipeline, param_dist, X_train, y_train,
                   n_iter=15, cv=3, name="model"):
    """
    Runs RandomizedSearchCV and returns the best fitted pipeline.
    """
    print(f"\n🔍  Tuning {name} with {n_iter} iterations …")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_cv_rmse = -search.best_score_
    print(f"   ✅  {name} best CV-RMSE = {best_cv_rmse:.2f}")
    print(f"   Best params: {search.best_params_}")

    return search.best_estimator_


def train_models(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Full training workflow using explicit 15% Validation set via PredefinedSplit.
    Returns dict with both trained pipelines.
    """
    # ── Combine train and val for PredefinedSplit ──
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    X_combined = combined_df[FEATURE_COLS]
    y_combined = combined_df[TARGET_COL]

    # -1 means 'do not use for validation' (it's training data)
    #  0 means 'use for validation in the first (and only) split'
    test_fold = np.concatenate([
        np.full(len(train_df), -1),
        np.full(len(val_df), 0)
    ])
    cv_split = PredefinedSplit(test_fold)

    rf_pipeline  = build_rf_pipeline()
    xgb_pipeline = build_xgb_pipeline()

    rf_best  = tune_and_train(rf_pipeline,  RF_PARAM_DIST,  X_combined, y_combined, cv=cv_split, name="RandomForest")
    xgb_best = tune_and_train(xgb_pipeline, XGB_PARAM_DIST, X_combined, y_combined, cv=cv_split, name="XGBoost")

    # Save both
    for name, model in [("random_forest", rf_best), ("xgboost", xgb_best)]:
        path = os.path.join(MODEL_DIR, f"{name}_pipeline.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"💾  Saved → {path}")

    return {"RandomForest": rf_best, "XGBoost": xgb_best}


def load_model(name: str = "xgboost"):
    """Load a saved pipeline from disk."""
    path = os.path.join(MODEL_DIR, f"{name}_pipeline.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}. Run train.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)
