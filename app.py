import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Configuration
st.set_page_config(
    page_title="Titanic Survival Intelligence Dashboard", 
    page_icon="🚢", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Complete Dark Theme CSS with Fixed Dropdown Menu Background
st.markdown("""
    <style>
    /* Global Theme & Background Overrides */
    .stApp {
        background: linear-gradient(135deg, #0d1b1e 0%, #050b0d 100%);
        color: #ffffff !important;
    }
    
    /* Remove Top White Header Bar & Footer */
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    
    /* Fix Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 100% !important;
    }
    
    /* Sidebar Styling & Text Color */
    section[data-testid="stSidebar"] {
        background-color: #081418;
        border-right: 1px solid rgba(0, 150, 136, 0.2);
    }
    
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    
    /* Input Fields Background Dark & Text White */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
        background-color: #0d1b1e !important;
        color: #ffffff !important;
        border: 1px solid rgba(77, 182, 172, 0.4) !important;
    }
    
    /* Dropdown Popover & Listbox Background Fix (Making it Black/Dark) */
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], div[role="listbox"] {
        background-color: #0d1b1e !important;
        color: #ffffff !important;
        border: 1px solid rgba(77, 182, 172, 0.4) !important;
    }
    
    div[data-baseweb="option"], div[role="option"], li[role="option"], span {
        background-color: #0d1b1e !important;
        color: #ffffff !important;
    }
    
    /* Hover effect for dropdown items */
    div[data-baseweb="option"]:hover, div[role="option"]:hover, li[role="option"]:hover {
        background-color: #004d40 !important;
        color: #ffffff !important;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #4db6ac !important;
        font-weight: 700;
    }
    
    p, span, label, div {
        color: #ffffff;
    }
    
    /* Custom Styled Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00897b 0%, #004d40 100%);
        color: #ffffff !important;
        font-size: 16px;
        font-weight: 600;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(77, 182, 172, 0.4);
        box-shadow: 0 4px 15px rgba(0, 137, 123, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00a896 0%, #00695c 100%);
        border-color: #4db6ac;
        box-shadow: 0 6px 20px rgba(0, 168, 150, 0.5);
    }
    
    /* Metric Labels and Values Visibility */
    [data-testid="stMetricLabel"] p {
        color: #b2dfdb !important;
    }
    [data-testid="stMetricValue"] div {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Saved Model and Encoders
@st.cache_resource
def load_artifacts():
    with open('titanic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('sex_encoder.pkl', 'rb') as file:
        sex_encoder = pickle.load(file)
    with open('embarked_encoder.pkl', 'rb') as file:
        embarked_encoder = pickle.load(file)
    with open('title_encoder.pkl', 'rb') as file:
        title_encoder = pickle.load(file)
    return model, sex_encoder, embarked_encoder, title_encoder

model, sex_encoder, embarked_encoder, title_encoder = load_artifacts()

# 4. Header Section
st.title("🚢 Titanic Survival Intelligence Dashboard")
st.markdown("Explore historical maritime analytics powered by machine learning algorithms to evaluate survival probabilities.")
st.markdown("<hr style='border-color: rgba(77, 182, 172, 0.2);'>", unsafe_allow_html=True)

# 5. Sidebar Layout for Inputs
st.sidebar.header("⚙️ Passenger Configuration")
st.sidebar.markdown("Modify the parameters below to run real-time predictions.")

pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], format_func=lambda x: f"Class {x}")
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age (Years)", 0.42, 80.0, 28.0)
sibsp = st.sidebar.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)
parch = st.sidebar.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=6, value=0)
fare = st.sidebar.number_input("Ticket Fare ($)", min_value=0.0, max_value=512.33, value=32.20)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"], format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
title = st.sidebar.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev"])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.sidebar.button("🚀 Analyze Survival")

# 6. Main Content Area & Prediction Logic
if predict_btn:
    # Feature Engineering
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    # Safe Encoding
    try:
        sex_encoded = sex_encoder.transform([sex])[0]
    except:
        sex_encoded = 0

    try:
        embarked_encoded = embarked_encoder.transform([embarked])[0]
    except:
        embarked_encoded = 0

    try:
        title_encoded = title_encoder.transform([title])[0]
    except:
        title_encoded = 0

    input_data = pd.DataFrame([[
        pclass, sex_encoded, age, sibsp, parch, fare, 
        embarked_encoded, family_size, is_alone, title_encoded
    ]], columns=[
        'Pclass', 'Sex_Encoded', 'Age', 'SibSp', 'Parch', 'Fare', 
        'Embarked_Encoded', 'FamilySize', 'IsAlone', 'Title_Encoded'
    ])

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("📊 Diagnostic & Prediction Results")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 🎯 Final Outcome")
        if prediction == 1:
            st.success("### 🟢 SURVIVED")
            st.markdown("This profile correlates with high-priority evacuation protocols (e.g., upper class deck proximity or female priority).")
        else:
            st.error("### 🔴 DID NOT SURVIVE")
            st.markdown("This profile correlates with restricted lower-deck access or high-risk statistical weights.")

    with col2:
        st.markdown("### 📈 Probability Breakdown")
        surv_prob = prediction_proba[1] * 100
        not_surv_prob = prediction_proba[0] * 100
        
        st.metric(label="Calculated Survival Likelihood", value=f"{surv_prob:.2f}%")
        st.metric(label="Risk Likelihood", value=f"{not_surv_prob:.2f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(prediction_proba[1]))
else:
    st.markdown("""
        <div style="background: rgba(13, 27, 30, 0.85); border: 1px solid rgba(77, 182, 172, 0.3); padding: 30px; border-radius: 16px; text-align: center;">
            <h3>👋 Welcome to the Interactive Dashboard</h3>
            <p style="color: #b2dfdb; font-size: 16px;">Select your desired passenger attributes from the sidebar control panel and click <b>'Analyze Survival'</b> to generate real-time predictive insights.</p>
        </div>
    """, unsafe_allow_html=True)