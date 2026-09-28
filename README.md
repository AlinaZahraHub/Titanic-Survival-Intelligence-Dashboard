# 🚢 Titanic Survival Intelligence Dashboard

An interactive, end-to-end Machine Learning web application that predicts passenger survival probabilities for the historic Titanic disaster. Built using Python, Scikit-Learn, and Streamlit, this application features an optimized Random Forest classifier paired with a sleek, custom-styled dark-mode dashboard interface.

---

## 🌟 Live Demo & Overview
This project demonstrates a complete end-to-end Machine Learning pipeline—progressing from raw historical data exploration and rigorous feature engineering to hyperparameter tuning and modern web deployment.

---

## 🚀 Key Features
* **Advanced Feature Engineering:** Extracted high-impact predictive features such as `FamilySize`, `IsAlone` status, and refined passenger `Titles` from raw name strings.
* **Optimized Machine Learning Model:** Powered by a **Random Forest Classifier**, tuned using `GridSearchCV` to ensure optimal performance and robustness.
* **Interactive Dark-Mode UI:** Designed with custom CSS styling and a glassmorphism aesthetic, offering real-time probability breakdowns, risk metrics, and intuitive sidebar controls.
* **Model Persistence:** Saved models and categorical encoders (`.pkl`) are seamlessly loaded at runtime for instant, low-latency predictions.

---

## 🛠️ Tech Stack & Libraries
* **Programming Language:** Python 3.12
* **Machine Learning & Data Science:** Scikit-Learn, Pandas, NumPy
* **Web Framework & UI:** Streamlit (Custom CSS & HTML Integration)
* **Development Tools:** Git, Visual Studio Code, Jupyter Notebook

---

## 📂 Project Structure
```text
Titanic-Dataset/
│
├── app.py                 # Main Streamlit web application script
├── titanic_model.pkl      # Trained Random Forest classifier
├── sex_encoder.pkl        # Label encoder for passenger gender
├── embarked_encoder.pkl   # Label encoder for embarkation port
├── title_encoder.pkl      # Label encoder for extracted titles
├── requirements.txt       # Project dependencies for deployment
└── README.md              # Project documentation

```

---

## ⚙️ Local Installation & Setup

To run this project locally on your machine, follow these steps:

1. **Clone the Repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/titanic-survival-intelligence-dashboard.git](https://github.com/YOUR_USERNAME/titanic-survival-intelligence-dashboard.git)
cd titanic-survival-intelligence-dashboard

```


2. **Install Dependencies:**
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt

```


3. **Run the Streamlit App:**
```bash
python -m streamlit run app.py

```



---

## 📊 Machine Learning Pipeline Steps

1. **Problem Definition:** Binary classification task (Predicting survival: `0` = Did Not Survive, `1` = Survived).
2. **Exploratory Data Analysis (EDA):** Uncovering historical patterns (e.g., "women and children first" protocol, class-based evacuation access).
3. **Data Preprocessing & Cleaning:** Handling missing values, encoding categorical variables, and scaling features.
4. **Model Training & Validation:** Training baseline models, evaluating performance metrics, and tuning hyperparameters.
5. **Deployment:** Building an interactive user-facing dashboard via Streamlit Cloud.

---

## 👤 Author

**Alina Zahra**

*Undergraduate Computer Science Student | AI & Frontend Development Enthusiast*

[GitHub Profile](https://www.google.com/search?q=https://github.com/YOUR_USERNAME&utm_source=gemini)
