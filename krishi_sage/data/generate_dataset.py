"""
generate_dataset.py
--------------------
Generates a realistic synthetic dataset modeled after the
Government of India Open Data Portal – Crop Production dataset.

The original dataset has columns:
  State_Name, District_Name, Crop_Year, Season, Crop, Area, Production

We EXTEND it with weather features (Temperature, Rainfall, Humidity)
and compute Yield = Production / Area as the TARGET.
We deliberately do NOT include Production as an input feature
to avoid data leakage.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── Indian States ──────────────────────────────────────────────────────────────
STATES = [
    "Uttar Pradesh", "Punjab", "Haryana", "Madhya Pradesh",
    "Maharashtra", "Andhra Pradesh", "Karnataka", "Tamil Nadu",
    "Rajasthan", "West Bengal", "Bihar", "Odisha",
    "Gujarat", "Chhattisgarh", "Telangana"
]

# ── Crops with realistic base-yield ranges (kg/hectare) ───────────────────────
CROP_CONFIG = {
    "Rice":        {"base_yield": 2200, "sd": 400,  "seasons": ["Kharif"]},
    "Wheat":       {"base_yield": 2800, "sd": 350,  "seasons": ["Rabi"]},
    "Maize":       {"base_yield": 2000, "sd": 300,  "seasons": ["Kharif", "Rabi"]},
    "Sugarcane":   {"base_yield": 65000,"sd": 8000, "seasons": ["Whole Year"]},
    "Cotton":      {"base_yield": 450,  "sd": 80,   "seasons": ["Kharif"]},
    "Groundnut":   {"base_yield": 1200, "sd": 200,  "seasons": ["Kharif", "Rabi"]},
    "Soyabean":    {"base_yield": 1000, "sd": 180,  "seasons": ["Kharif"]},
    "Bajra":       {"base_yield": 900,  "sd": 150,  "seasons": ["Kharif"]},
    "Jowar":       {"base_yield": 850,  "sd": 130,  "seasons": ["Kharif", "Rabi"]},
    "Potato":      {"base_yield": 19000,"sd": 3000, "seasons": ["Rabi"]},
    "Mustard":     {"base_yield": 1100, "sd": 160,  "seasons": ["Rabi"]},
    "Sunflower":   {"base_yield": 750,  "sd": 120,  "seasons": ["Kharif", "Rabi"]},
    "Turmeric":    {"base_yield": 5500, "sd": 700,  "seasons": ["Kharif"]},
    "Chickpea":    {"base_yield": 900,  "sd": 140,  "seasons": ["Rabi"]},
    "Lentil":      {"base_yield": 850,  "sd": 120,  "seasons": ["Rabi"]},
}

SEASONS = ["Kharif", "Rabi", "Whole Year"]

# ── Weather profiles per state ─────────────────────────────────────────────────
STATE_WEATHER = {
    "Uttar Pradesh":  {"temp": 28, "rain": 900,  "hum": 65},
    "Punjab":         {"temp": 24, "rain": 700,  "hum": 60},
    "Haryana":        {"temp": 26, "rain": 650,  "hum": 58},
    "Madhya Pradesh": {"temp": 30, "rain": 1100, "hum": 68},
    "Maharashtra":    {"temp": 29, "rain": 1200, "hum": 70},
    "Andhra Pradesh": {"temp": 32, "rain": 950,  "hum": 72},
    "Karnataka":      {"temp": 27, "rain": 1000, "hum": 75},
    "Tamil Nadu":     {"temp": 31, "rain": 1150, "hum": 78},
    "Rajasthan":      {"temp": 33, "rain": 400,  "hum": 42},
    "West Bengal":    {"temp": 28, "rain": 1600, "hum": 80},
    "Bihar":          {"temp": 27, "rain": 1100, "hum": 72},
    "Odisha":         {"temp": 29, "rain": 1500, "hum": 76},
    "Gujarat":        {"temp": 31, "rain": 750,  "hum": 60},
    "Chhattisgarh":   {"temp": 28, "rain": 1300, "hum": 74},
    "Telangana":      {"temp": 31, "rain": 950,  "hum": 68},
}

YEARS = list(range(2001, 2022))   # 2001 – 2021  (21 years)
N_RECORDS = 15000


def generate():
    rows = []
    for _ in range(N_RECORDS):
        state   = np.random.choice(STATES)
        crop    = np.random.choice(list(CROP_CONFIG.keys()))
        cfg     = CROP_CONFIG[crop]
        season  = np.random.choice(cfg["seasons"])
        year    = np.random.choice(YEARS)
        area    = np.random.uniform(50, 5000)         # hectares

        # ── Weather (state-based with noise) ──────────────────────────────────
        w = STATE_WEATHER[state]
        temperature = w["temp"] + np.random.normal(0, 2.5)
        rainfall    = w["rain"] + np.random.normal(0, 150)
        humidity    = np.clip(w["hum"]  + np.random.normal(0, 6), 20, 100)

        # ── Weather effect on yield ───────────────────────────────────────────
        # Optimal rain band: 600–1400 mm => bonus; extremes => penalty
        rain_effect = 1 - 0.0003 * abs(rainfall - 1000)
        rain_effect = max(0.5, min(rain_effect, 1.2))

        # Optimal temp: 25–30 °C
        temp_effect = 1 - 0.01 * max(0, abs(temperature - 27) - 5)
        temp_effect = max(0.6, min(temp_effect, 1.1))

        # Humidity: higher is generally better (up to 75 %)
        hum_effect  = 0.8 + 0.004 * min(humidity, 75)
        hum_effect  = max(0.7, min(hum_effect, 1.15))

        # Mild technology trend: +0.5 % per year after 2001
        trend       = 1 + 0.005 * (year - 2001)

        base   = cfg["base_yield"] * rain_effect * temp_effect * hum_effect * trend
        yield_ = max(0, np.random.normal(base, cfg["sd"]))   # kg / hectare

        rows.append({
            "Crop_Year":   year,
            "State":       state,
            "Crop":        crop,
            "Season":      season,
            "Area":        round(area, 2),
            "Temperature": round(temperature, 1),
            "Rainfall":    round(rainfall, 1),
            "Humidity":    round(humidity, 1),
            "Yield":       round(yield_, 2),     # TARGET – kg/hectare
        })

    df = pd.DataFrame(rows)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "crop_data.csv")
    df.to_csv(out_path, index=False)
    print(f"✅  Dataset saved → {out_path}  ({len(df)} rows)")
    print(df.describe())
    return df


if __name__ == "__main__":
    generate()
