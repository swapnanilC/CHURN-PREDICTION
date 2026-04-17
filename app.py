import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import model_from_json

# Page config
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# Styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ================= MODEL LOADING =================
with open("model.json", "r") as json_file:
    model = model_from_json(json_file.read())

model.load_weights("model.weights.h5")

# ✅ IMPORTANT FIX: compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ================= LOAD PREPROCESSING =================
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ================= UI =================
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer will churn based on their profile.")

st.sidebar.header("🧾 Customer Details")

geography = st.sidebar.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('⚧ Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('🎂 Age', 18, 92)
balance = st.sidebar.number_input('💰 Balance', min_value=0.0)
credit_score = st.sidebar.number_input('💳 Credit Score', min_value=300, max_value=900)
estimated_salary = st.sidebar.number_input('💼 Estimated Salary', min_value=0.0)
tenure = st.sidebar.slider('📅 Tenure (Years)', 0, 10)
num_of_products = st.sidebar.slider('📦 Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('⚡ Active Member', [0, 1])

predict_btn = st.sidebar.button("🔍 Predict Churn")

# ================= PREDICTION =================
if predict_btn:

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encoding
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scaling
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled, verbose=0)
    prediction_proba = float(prediction[0][0])

    # ================= OUTPUT =================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Churn Probability")
        st.progress(prediction_proba)

        st.metric(
            label="Probability",
            value=f"{prediction_proba:.2f}"
        )

    with col2:
        st.subheader("📊 Prediction Result")

        if prediction_proba > 0.5:
            st.error("⚠️ High Risk: Customer likely to churn")
        else:
            st.success("✅ Low Risk: Customer likely to stay")

    st.markdown("---")
    st.subheader("🔍 Interpretation")

    if prediction_proba > 0.75:
        st.write("Very high chance of churn. Immediate action recommended.")
    elif prediction_proba > 0.5:
        st.write("Moderate risk. Consider retention strategies.")
    else:
        st.write("Customer is stable.")