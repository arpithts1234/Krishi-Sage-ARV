# 🌾 Crop Yield Prediction System

> Predict expected crop yield (kg/hectare) **before harvest** using environmental
> and agricultural features. Built with scikit-learn, XGBoost, and Streamlit.

---

## 📁 Project Structure

```
krishi_sage/
├── data/
│   ├── generate_dataset.py     # Synthetic dataset generator
│   └── crop_data.csv           # Auto-generated on first run
├── src/
│   ├── data_loader.py          # Loads / validates the dataset
│   ├── preprocessing.py        # Cleans data + time-based split
│   ├── feature_engineering.py  # Adds interaction features
│   ├── train.py                # Pipeline + hyperparameter tuning
│   └── evaluate.py             # RMSE / MAE / R² + plots
├── models/                     # Saved .pkl models + plots
├── app.py                      # Streamlit web app
├── run_training.py             # One-click training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python run_training.py
```
This will:
- Generate the dataset (15,000 rows)
- Engineer features
- Tune Random Forest and XGBoost
- Save models to `models/`
- Print RMSE / MAE / R² for both models

### 3. Launch the web app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 🔬 Research Gap Addressed

| Problem in existing systems | Our solution |
|---|---|
| Data leakage (using Production as feature) | Only pre-harvest inputs used |
| Random train-test split | Time-based split: train on past, test on future |
| No weather interaction | `Temperature × Rainfall` interaction feature |
| Missing non-linear weather effects | `Rainfall²` captures U-shaped yield response |

---

## 📊 Features Used

| Feature | Description |
|---|---|
| Crop, Season, State | Categorical identifiers |
| Area | Farm area in hectares |
| Temperature | Average growing-season temp (°C) |
| Rainfall | Annual rainfall (mm) |
| Humidity | Average humidity (%) |
| Year_Trend | Years since 2001 (technology trend) |
| Humidity_Norm | Humidity ÷ 100 |
| Temp_Rain_Interaction | Temperature × Rainfall |
| Rain_Squared | Rainfall² (non-linear effect) |

**Target:** `Yield` (kg/hectare)

---

## 📏 Metrics

- **RMSE** – Root Mean Squared Error (kg/ha)
- **MAE** – Mean Absolute Error (kg/ha)
- **R²** – Proportion of yield variance explained by the model

> R² in simple words: "How much better is our model than just predicting the average yield every time?"

---

## 🧑‍💻 Author
Built as a final-year ML project demonstrating leakage-free, temporally-validated
crop yield prediction for Indian agriculture.
