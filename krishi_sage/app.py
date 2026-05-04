


"""
app.py
------
Crop Yield Prediction – Streamlit Web Application
Run: streamlit run app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from groq import Groq
from audio_recorder_streamlit import audio_recorder
import base64

# ── Make src/ importable ──────────────────────────────────────────────────────
SRC  = os.path.join(os.path.dirname(__file__), "src")
DATA = os.path.join(os.path.dirname(__file__), "data")
sys.path.insert(0, SRC)
sys.path.insert(0, DATA)

from feature_engineering import FEATURE_COLS
from chatbot import transcribe_audio, get_chat_response, text_to_audio

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
HISTORY_FILE = os.path.join(DATA, "search_history.csv")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Prediction 🌾",
    page_icon="🌾",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 2.6rem; font-weight: 800; color: #2E7D32; }
    .sub-title   { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-box  { background: #E8F5E9; border-radius: 12px; padding: 16px;
                   text-align: center; }
    .metric-val  { font-size: 2.2rem; font-weight: 700; color: #1B5E20; }
    .metric-lbl  { font-size: 0.85rem; color: #388E3C; }
    .info-box    { background: #FFF9C4; border-left: 5px solid #F9A825;
                   padding: 12px 16px; border-radius: 6px; font-size: 0.92rem; color: #333333; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    if not os.path.exists(best_path):
        return None, None
    with open(best_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["name"]


model, model_name = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌾 Crop Yield Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">AI-Powered Crop Yield Prediction System &nbsp;|&nbsp; '
    'Predict expected yield (kg/hectare) BEFORE harvest</p>',
    unsafe_allow_html=True,
)

# ── Sidebar – About ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/India_gate_banner_sunrise.jpg/320px-India_gate_banner_sunrise.jpg", use_column_width=True)
    st.markdown("## About Crop Yield Prediction")
    st.markdown("""
**Crop Yield Prediction** uses machine learning (Random Forest + XGBoost) to predict
how much crop yield (kg per hectare) a farmer can expect, given:

- 🌡️ Weather conditions
- 🌧️ Rainfall and humidity
- 🌱 Crop type & growing season
- 📐 Farm area

**Key design choices:**
- No data leakage (Production not used as input)
- Time-based train/test split (trains on past, tests on future years)
- Engineered features for agri-climate interactions
""")

    if model_name:
        st.success(f"✅ Active model: **{model_name}**")
    else:
        st.error("⚠️ No trained model found. Run `python run_training.py` first.")

# ── Main content ──────────────────────────────────────────────────────────────
if model is None:
    st.warning(
        "🚨 No trained model found.\n\n"
        "Please run `python run_training.py` from the project root, then reload this page."
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab_rec, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict Yield", 
    "🌟 Best Crop Recommendation",
    "📊 Model Insights", 
    "📖 How It Works", 
    "📜 Search History",
    "💬 AI Assistant"
])

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 – PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Farm & Weather Details")

    col1, col2, col3 = st.columns(3)

    CROPS = [
        "Rice", "Wheat", "Maize", "Sugarcane", "Cotton",
        "Groundnut", "Soyabean", "Bajra", "Jowar", "Potato",
        "Mustard", "Sunflower", "Turmeric", "Chickpea", "Lentil",
    ]
    SEASONS = ["Kharif", "Rabi", "Whole Year"]
    STATES = [
        "Uttar Pradesh", "Punjab", "Haryana", "Madhya Pradesh",
        "Maharashtra", "Andhra Pradesh", "Karnataka", "Tamil Nadu",
        "Rajasthan", "West Bengal", "Bihar", "Odisha",
        "Gujarat", "Chhattisgarh", "Telangana",
    ]

    with col1:
        crop     = st.selectbox("🌱 Crop", CROPS)
        season   = st.selectbox("📅 Season", SEASONS)
        state    = st.selectbox("📍 State", STATES)

    with col2:
        area        = st.number_input("📐 Area (hectares)", min_value=1.0, max_value=10000.0, value=500.0, step=50.0)
        temperature = st.slider("🌡️ Avg Temperature (°C)", 10.0, 45.0, 28.0, 0.5)
        crop_year   = st.number_input("📆 Crop Year", min_value=2001, max_value=2030, value=2024)

    with col3:
        rainfall = st.slider("🌧️ Annual Rainfall (mm)", 100.0, 3000.0, 900.0, 10.0)
        humidity = st.slider("💧 Avg Humidity (%)", 20.0, 100.0, 65.0, 1.0)

        # Live feature preview
        st.markdown("**Derived Features (auto-computed)**")
        st.caption(f"Year Trend: {crop_year - 2001}")
        st.caption(f"Humidity Norm: {humidity/100:.2f}")
        st.caption(f"Temp × Rain: {temperature * rainfall:,.0f}")

    st.divider()

    if st.button("🌾 Predict Yield", type="primary", use_container_width=True):
        # Build input row
        input_dict = {
            "Area":                  area,
            "Temperature":           temperature,
            "Rainfall":              rainfall,
            "Humidity":              humidity,
            "Crop":                  crop,
            "Season":                season,
            "State":                 state,
            "Year_Trend":            crop_year - 2001,
            "Humidity_Norm":         humidity / 100.0,
            "Temp_Rain_Interaction": temperature * rainfall,
            "Rain_Squared":          rainfall ** 2,
        }
        input_df = pd.DataFrame([input_dict])[FEATURE_COLS]

        with st.spinner("Running prediction …"):
            predicted_yield = float(model.predict(input_df)[0])
            predicted_production = predicted_yield * area

        # ── Save History ───────────────────────────────────────────────────
        history_record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Crop": crop,
            "Season": season,
            "State": state,
            "Area (ha)": area,
            "Temp (°C)": temperature,
            "Rainfall (mm)": rainfall,
            "Humidity (%)": humidity,
            "Predicted Yield (kg/ha)": round(predicted_yield, 2),
            "Total Production (t)": round(predicted_production / 1000, 2)
        }
        history_df = pd.DataFrame([history_record])
        if os.path.exists(HISTORY_FILE):
            history_df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
        else:
            history_df.to_csv(HISTORY_FILE, index=False)

        # ── Display results ────────────────────────────────────────────────
        st.markdown("### 🎯 Prediction Results")
        
        # Baseline averages for UI deltas (dummy national averages for demonstration)
        baselines = {
            "Rice": 2600, "Wheat": 3200, "Maize": 2500, "Sugarcane": 70000, 
            "Cotton": 500, "Potato": 20000, "Lentil": 1000, "Sunflower": 1200,
            "Soyabean": 1500, "Jowar": 1000, "Mustard": 1500, "Chickpea": 1000,
            "Groundnut": 1800, "Bajra": 1200, "Turmeric": 5000
        }
        baseline = baselines.get(crop, 2000) # default fallback
        yield_delta = predicted_yield - baseline
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Predicted Yield (kg/ha)", f"{predicted_yield:,.0f}", f"{yield_delta:+,.0f} vs avg")
        with r2:
            st.metric("Total Production (tonnes)", f"{predicted_production/1000:,.1f}")
        with r3:
            st.metric("Farm Area (ha)", f"{area:,.0f}")
            
        st.success("Prediction generated successfully!")
        
        # ── Agri-Advisor Insights ──────────────────────────────────────────
        st.markdown("### 🧑‍🌾 Agri-Advisor Insights")
        advisor_cols = st.columns(3)
        
        with advisor_cols[0]:
            if rainfall < 500:
                st.error("💧 **Water Stress:** Low rainfall detected. Consider drip irrigation or drought-resistant seed variants.")
            elif rainfall > 1500:
                st.error("🌧️ **Heavy Rain:** Ensure proper field drainage to prevent waterlogging and root rot.")
            else:
                st.success("🌦️ **Optimal Rain:** Rainfall is within a generally healthy range.")
                
        with advisor_cols[1]:
            if temperature > 35:
                st.error("🔥 **Heat Stress:** High temperatures may affect grain filling. Ensure adequate soil moisture.")
            elif temperature < 15:
                st.error("❄️ **Cold Stress:** Risk of frost damage. Consider protective measures if in vulnerable stages.")
            else:
                st.success("🌡️ **Optimal Temp:** Temperature conditions look favorable.")
                
        with advisor_cols[2]:
            if humidity > 80:
                st.error("🍄 **Fungal Risk:** High humidity increases disease risks. Consider preventive fungicide applications.")
            elif humidity < 40:
                st.error("🌵 **Dry Air:** Low humidity might increase evapotranspiration. Keep an eye on watering schedules.")
            else:
                st.success("💧 **Optimal Humidity:** Moisture levels are balanced.")

        # ── Info box ────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="info-box">
        🌾 Based on the inputs provided, <strong>{crop}</strong> grown in <strong>{state}</strong>
        during the <strong>{season}</strong> season in <strong>{crop_year}</strong>
        is expected to yield approximately <strong>{predicted_yield:,.0f} kg per hectare</strong>.
        Over your <strong>{area:,.0f} hectare</strong> farm, total production should be around
        <strong>{predicted_production/1000:,.1f} tonnes</strong>.
        </div>
        """, unsafe_allow_html=True)

        # ── Bar chart of input factors ──────────────────────────────────────
        st.markdown("#### Input Factor Summary")
        factor_df = pd.DataFrame({
            "Factor":  ["Temperature (°C)", "Rainfall (mm÷10)", "Humidity (%)"],
            "Value":   [temperature,         rainfall / 10,       humidity],
        })
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(factor_df["Factor"], factor_df["Value"], color=["#EF5350", "#42A5F5", "#66BB6A"], edgecolor="black")
        ax.set_ylabel("Value")
        ax.set_title("Weather Inputs (Rainfall ÷ 10 for scale)")
        st.pyplot(fig)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  TAB REC – BEST CROP RECOMMENDATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_rec:
    st.subheader("Enter Farm & Weather Details")

    rc1, rc2, rc3 = st.columns(3)

    with rc1:
        rec_season   = st.selectbox("📅 Season", SEASONS, key="rec_season")
        rec_state    = st.selectbox("📍 State", STATES, key="rec_state")

    with rc2:
        rec_area        = st.number_input("📐 Area (hectares)", min_value=1.0, max_value=10000.0, value=500.0, step=50.0, key="rec_area")
        rec_temperature = st.slider("🌡️ Avg Temperature (°C)", 10.0, 45.0, 28.0, 0.5, key="rec_temp")
        rec_crop_year   = st.number_input("📆 Crop Year", min_value=2001, max_value=2030, value=2024, key="rec_year")

    with rc3:
        rec_rainfall = st.slider("🌧️ Annual Rainfall (mm)", 100.0, 3000.0, 900.0, 10.0, key="rec_rain")
        rec_humidity = st.slider("💧 Avg Humidity (%)", 20.0, 100.0, 65.0, 1.0, key="rec_hum")

    st.divider()

    if st.button("🌟 Recommend Best Crop", type="primary", use_container_width=True):
        st.markdown("### 🌟 Top Recommendations")
        st.caption("Given your specific temperature, area, and rainfall, these crops will yield the highest returns compared to their national averages:")
        
        rec_input_dict = {
            "Area":                  rec_area,
            "Temperature":           rec_temperature,
            "Rainfall":              rec_rainfall,
            "Humidity":              rec_humidity,
            "Season":                rec_season,
            "State":                 rec_state,
            "Year_Trend":            rec_crop_year - 2001,
            "Humidity_Norm":         rec_humidity / 100.0,
            "Temp_Rain_Interaction": rec_temperature * rec_rainfall,
            "Rain_Squared":          rec_rainfall ** 2,
        }
        
        baselines = {
            "Rice": 2600, "Wheat": 3200, "Maize": 2500, "Sugarcane": 70000, 
            "Cotton": 500, "Potato": 20000, "Lentil": 1000, "Sunflower": 1200,
            "Soyabean": 1500, "Jowar": 1000, "Mustard": 1500, "Chickpea": 1000,
            "Groundnut": 1800, "Bajra": 1200, "Turmeric": 5000
        }
        all_crops = list(baselines.keys())
        crop_scores = []
        
        with st.spinner("Analyzing all crops..."):
            for c in all_crops:
                c_input = rec_input_dict.copy()
                c_input["Crop"] = c
                c_df = pd.DataFrame([c_input])[FEATURE_COLS]
                c_pred = float(model.predict(c_df)[0])
                c_base = baselines.get(c, 2000)
                score = (c_pred - c_base) / c_base * 100
                crop_scores.append((c, c_pred, score))
                
        crop_scores.sort(key=lambda x: x[2], reverse=True)
        top_3 = crop_scores[:3]
        
        rec_cols = st.columns(3)
        for idx, (rec_crop, rec_yield, rec_score) in enumerate(top_3):
            with rec_cols[idx]:
                medal = ["🥇", "🥈", "🥉"][idx]
                st.metric(f"{medal} {rec_crop}", f"{rec_yield:,.0f} kg/ha", f"{rec_score:+.1f}% vs baseline")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 – MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Model Insights – {model_name}")

    img_map = {
        "eval":    os.path.join(MODEL_DIR, f"{model_name.lower()}_eval.png"),
        "fi":      os.path.join(MODEL_DIR, f"{model_name.lower()}_feature_importance.png"),
    }

    for label, path in img_map.items():
        if os.path.exists(path):
            st.image(path, use_column_width=True)
        else:
            st.info(f"📌 {label} plot not found. Run `python run_training.py` to generate it.")

    # Metric cards from saved pkl
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    if os.path.exists(best_path):
        st.markdown("#### Model trained successfully ✅")
        st.markdown("Evaluation metrics are printed in the terminal during training. "
                    "Re-run `python run_training.py` and check the console output.")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 – HOW IT WORKS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("How Crop Yield Prediction Works")

    st.markdown("""
### 🔬 Research Gap

Most existing crop prediction tools suffer from:
- **Data Leakage**: Using `Production` as a feature when `Yield = Production / Area`
- **Random splits**: Training on future data, testing on past → inflated accuracy
- **No weather interactions**: Treating temperature and rainfall independently

This system fixes all three.

---

### 🧪 Feature Engineering

| Feature | Formula | Why |
|---|---|---|
| Year_Trend | `Crop_Year - 2001` | Captures yield improvements over time (better seeds, tech) |
| Humidity_Norm | `Humidity / 100` | Normalises to 0–1 for model stability |
| Temp_Rain_Interaction | `Temperature × Rainfall` | Evapotranspiration proxy – captures combined climate effect |
| Rain_Squared | `Rainfall²` | Both drought and flood hurt yield – non-linear relationship |

---

### ⏳ Data Split Strategy

```
Train:      70%  (For model learning)
Validation: 15%  (For hyperparameter tuning)
Test:       15%  (Held-out for final evaluation)
```

We use a strict random split to evaluate the model on a completely unseen 15% of the data, ensuring robust evaluation and validation.

---

### 📏 Evaluation Metrics

| Metric | Meaning |
|---|---|
| **RMSE** | Average prediction error in kg/ha. Lower = better. |
| **MAE** | Average absolute error. Less sensitive to outliers. |
| **R²** | How much variation the model explains vs just using the mean. 1.0 = perfect. |

> **R² in one sentence (for viva):**
> "R² tells us how much better our model is compared to simply predicting the average yield for every farm."

---

### 🏗️ Full Pipeline

```
Raw CSV → Clean → Feature Engineering → ColumnTransformer
         (impute / scale / encode) → RandomForest / XGBoost → Predict Yield
```
""")

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4 – SEARCH HISTORY
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📜 Recent Predictions & Analytics")
    
    if os.path.exists(HISTORY_FILE):
        try:
            history_data = pd.read_csv(HISTORY_FILE)
            # Parse datetime so Streamlit line_chart can handle it natively
            history_data["Timestamp"] = pd.to_datetime(history_data["Timestamp"])
            history_data = history_data.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)
            
            # --- Interactive Dashboard ---
            st.markdown("##### 📈 Analytics Overview")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.caption("Yield predictions over time")
                # Set index to Timestamp for a clean time-series plot
                chart_data = history_data.set_index("Timestamp")[["Predicted Yield (kg/ha)"]]
                st.line_chart(chart_data)
            with col_chart2:
                st.caption("Predictions by Crop")
                crop_counts = history_data["Crop"].value_counts()
                st.bar_chart(crop_counts)
                
            st.markdown("##### 🗄️ Raw Data")
            st.dataframe(history_data, use_container_width=True)
            
            # --- Download & Clear Actions ---
            col_action1, col_action2 = st.columns([1, 4])
            with col_action1:
                csv_data = history_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name='crop_yield_history.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            with col_action2:
                if st.button("🗑️ Clear History", type="secondary"):
                    os.remove(HISTORY_FILE)
                    st.rerun()
        except Exception as e:
            st.error(f"Error loading history: {e}")
    else:
        st.info("No search history found. Go to 'Predict Yield' to make your first prediction!")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 5 – AI ASSISTANT
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("💬 AI Agricultural Assistant (Hindi / English)")
    st.write("Ask anything about crops, yield predictions, weather, etc.")
    
    # Embedded Groq API Key
    import os
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    groq_api_key = GROQ_API_KEY
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for msg in st.session_state.chat_history:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "audio" in msg and msg["audio"]:
                    st.audio(msg["audio"], format="audio/mp3")
                
    if groq_api_key:
        client = Groq(api_key=groq_api_key)
        
        # Audio input
        st.write("🎙️ **Voice Input**")
        audio_bytes = audio_recorder(text="Click to record, click again to stop", icon_size="2x")
        
        user_input = st.chat_input("Type your question here...")
        audio_text = None
        
        # Process audio if new
        if audio_bytes and ("last_audio_bytes" not in st.session_state or st.session_state.last_audio_bytes != audio_bytes):
            st.session_state.last_audio_bytes = audio_bytes
            with st.spinner("Transcribing audio..."):
                audio_text = transcribe_audio(client, audio_bytes)
                if not audio_text or "Error" in audio_text:
                    st.error(audio_text or "Could not transcribe audio.")
                    audio_text = None
                
        final_input = user_input or audio_text
        
        if final_input:
            msg_user = {"role": "user", "content": final_input}
            st.session_state.chat_history.append(msg_user)
            with st.chat_message("user"):
                st.markdown(final_input)
                
            messages = [{"role": "system", "content": "You are a helpful agricultural AI assistant named Krishi Sage AI. You can speak Hindi and English. Keep responses concise and helpful. If the user asks in Hindi, reply in Hindi. If user asks in English, reply in English."}]
            for msg in st.session_state.chat_history:
                if msg["role"] != "system":
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
            with st.spinner("Thinking..."):
                response_text = get_chat_response(client, messages)
                
            with st.spinner("Generating audio response..."):
                response_audio = text_to_audio(response_text)
                
            msg_data = {"role": "assistant", "content": response_text}
            if response_audio:
                msg_data["audio"] = response_audio
                
            st.session_state.chat_history.append(msg_data)
            
            with st.chat_message("assistant"):
                st.markdown(response_text)
                if response_audio:
                    # using HTML to auto-play audio is tricky in Streamlit, but st.audio has autoplay parameter in newer versions
                    st.audio(response_audio, format="audio/mp3", autoplay=True)
    else:
        st.warning("Please enter your Groq API Key to chat.")
