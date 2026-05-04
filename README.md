# 🌾 Krishi Sage ARV

An intelligent crop recommendation and yield prediction system powered by Machine Learning and an interactive AI chatbot. This project helps farmers and agricultural enthusiasts make data-driven decisions for better productivity and sustainability.

---

## 🚀 Features

* 🌱 **Crop Recommendation System**
  Suggests the most suitable crops based on environmental and soil conditions.

* 📊 **Yield Prediction**
  Predicts crop yield using trained ML models.

* 🤖 **AI Chatbot Integration**
  Provides agricultural assistance and answers queries using an LLM-powered chatbot.

* 📈 **Model Evaluation & Visualization**
  Includes performance metrics and feature importance graphs.

* 🎙️ **Voice Input Support**
  Users can interact with the system using voice input.

* 🌐 **Interactive Web App**
  Built with Streamlit for a clean and user-friendly interface.

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Frontend/UI:** Streamlit
* **Machine Learning:** Scikit-learn, XGBoost
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib
* **AI Integration:** Groq API (LLM chatbot)

---

## 🤖 Machine Learning Models Used

* Random Forest Regressor
* XGBoost Regressor

These models are trained on agricultural datasets to predict crop yield and optimize recommendations.

---

## 📁 Project Structure

```
Krishi-Sage-ARV/
│
├── krishi_sage/
│   ├── app.py                  # Main Streamlit app
│   ├── data/                  # Dataset files
│   ├── models/                # Trained ML models (excluded in repo)
│   ├── src/                   # Core ML pipeline modules
│   ├── run_training.py        # Model training script
│   └── requirements.txt       # Dependencies
│
├── crop_data.csv              # Sample dataset
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/arpithts1234/Krishi-Sage-ARV.git
cd Krishi-Sage-ARV
```

### 2️⃣ Install dependencies

```bash
pip install -r krishi_sage/requirements.txt
```

### 3️⃣ Set environment variables

Create a `.env` file and add:

```
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run krishi_sage/app.py
```

---

## 📊 Output

* Crop recommendations based on user input
* Predicted yield values
* Model evaluation graphs
* Chatbot-based agricultural guidance

---

## 🔐 Security Note

API keys are not stored in the code. Environment variables are used for secure handling of sensitive data.

---

## 💡 Future Enhancements

* 📱 Mobile app integration
* 🌍 Real-time weather API integration
* 📡 IoT-based soil data input
* 📊 Advanced deep learning models

---

## 👨‍💻 Author

**Arpit Raj Verma**
B.Tech CSE | Machine Learning Enthusiast

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!
