"""
src/feature_engineering.py
---------------------------
Creates new features BEFORE passing data to the pipeline.

Each feature is explained with the agri-science reason behind it.
"""

import pandas as pd
import numpy as np


# Columns the model will actually USE as inputs
FEATURE_COLS = [
    "Area",
    "Temperature",
    "Rainfall",
    "Humidity",
    "Crop",
    "Season",
    "State",
    "Year_Trend",
    "Humidity_Norm",
    "Temp_Rain_Interaction",
    "Rain_Squared",
]

TARGET_COL = "Yield"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features.

    1. Year_Trend
       WHY: Crop yields improve over time due to better seeds, fertilizers,
            and farming techniques. A linear trend captures this.

    2. Humidity_Norm
       WHY: Humidity is on a 0–100 scale. Normalising to 0–1 makes it
            easier for tree models and also comparable to other features.

    3. Temp_Rain_Interaction
       WHY: Temperature × Rainfall together determine evapotranspiration.
            A warm + rainy season is very different from warm + dry.
            This single product captures the combined climatic effect.

    4. Rain_Squared
       WHY: The relationship between rainfall and yield is non-linear –
            too little OR too much rain both hurt. A squared term lets
            linear components of the model capture this U-shape.
    """

    df = df.copy()

    # 1. Year trend (2001 = 0, 2002 = 1, …)
    df["Year_Trend"] = df["Crop_Year"] - 2001

    # 2. Humidity normalised (0–1)
    df["Humidity_Norm"] = df["Humidity"] / 100.0

    # 3. Temperature × Rainfall interaction
    df["Temp_Rain_Interaction"] = df["Temperature"] * df["Rainfall"]

    # 4. Rainfall squared (captures non-linear effect)
    df["Rain_Squared"] = df["Rainfall"] ** 2

    return df
