"""
src/preprocessing.py
---------------------
Handles:
  - Dropping rows with missing target (Yield)
  - Capping extreme outliers in Yield
  - Time-based train/test split (avoids data leakage!)

WHY TIME-BASED SPLIT?
  If we split randomly, the model could see future data during training.
  In agriculture, you'd always train on past years and predict future years.
  This mimics the real-world scenario.
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where Yield is missing or negative."""
    df = df.dropna(subset=["Yield"])
    df = df[df["Yield"] > 0].reset_index(drop=True)

    # Cap extreme outliers: clip Yield above 99.5th percentile
    upper = df["Yield"].quantile(0.995)
    df["Yield"] = df["Yield"].clip(upper=upper)

    return df


from sklearn.model_selection import train_test_split

def random_split(df: pd.DataFrame, random_state: int = 42):
    """
    RANDOM SPLIT (70/15/15)
    -----------------------
    Train:      70%
    Validation: 15%
    Test:       15%
    """
    # First, split off 15% for the Test set. 85% remains.
    df_temp, test_df = train_test_split(df, test_size=0.15, random_state=random_state)
    
    # Second, split the remaining 85% to get exactly 15% of the TOTAL for validation.
    val_size_ratio = 15 / 85
    train_df, val_df = train_test_split(df_temp, test_size=val_size_ratio, random_state=random_state)

    print(f"🔀  Train: {len(train_df)} rows  (~70%)")
    print(f"🔀  Val  : {len(val_df)} rows  (~15%)")
    print(f"🔀  Test : {len(test_df)} rows  (~15%)")
    
    return train_df, val_df, test_df
