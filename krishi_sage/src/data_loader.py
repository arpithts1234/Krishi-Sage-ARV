"""
src/data_loader.py
------------------
Loads the crop dataset (generates it first if not found).
Handles basic validation and returns a clean DataFrame.
"""

import os
import pandas as pd
import sys

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "crop_data.csv")


def load_data() -> pd.DataFrame:
    """
    Load the crop dataset from CSV.
    If the file is missing, auto-generate it.
    """
    if not os.path.exists(DATA_PATH):
        print("📦  Dataset not found – generating now …")
        # Add data/ folder to path and run the generator
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        sys.path.insert(0, data_dir)
        from generate_dataset import generate
        generate()

    df = pd.read_csv(DATA_PATH)

    # Basic sanity check
    required = ["Crop_Year", "State", "Crop", "Season",
                "Area", "Temperature", "Rainfall", "Humidity", "Yield"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print(f"✅  Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df
