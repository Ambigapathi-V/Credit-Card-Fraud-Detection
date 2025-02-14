import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set Streamlit page config
st.set_page_config(page_title="Credit Fraud Prediction", layout="wide")

# Load model and scaler
@st.cache_resource
def load_joblib_model(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

scaler = load_joblib_model("artifacts/scaler.joblib")
model = load_joblib_model("artifacts/random_forest_model.joblib")

if not scaler or not model:
    st.error("Model or scaler could not be loaded. Please check file paths.")
    st.stop()

# User input feature names
user_input_features = ['V12', 'V17', 'V14', 'V10', 'V11', 'V16', 'V9', 'V4', 'V18', 'V21']

st.title("Credit Card Fraud Detection")

# Sidebar: User choice
with st.sidebar:
    st.header("Input Method")
    input_choice = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

if input_choice == "Manual Input":
    st.subheader("Enter Transaction Details")
    
    # Grid layout for input fields
    cols = st.columns(3)  # Three-column layout for better arrangement
    user_input = {}
    
    for idx, feature in enumerate(user_input_features):
        col_idx = idx % 3  # Alternate between the three columns
        with cols[col_idx]:
            user_input[feature] = st.slider(f"{feature}", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    
    input_df = pd.DataFrame([user_input], columns=user_input_features)
    scaled_input = scaler.transform(input_df)
    
    if st.button("Predict", use_container_width=True):
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0, 1]
        prediction_text = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
        prediction_color = "#ff4b4b" if prediction == 1 else "#4caf50"
        st.markdown(
            f"<h2 style='text-align: center; color: white; background-color: {prediction_color}; padding: 10px;'>"
            f"Prediction: {prediction_text} (Probability: {probability:.2f})</h2>",
            unsafe_allow_html=True
        )

elif input_choice == "Upload CSV":
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if not set(user_input_features).issubset(df.columns):
            st.error("Uploaded file does not contain the required features.")
        else:
            df_selected = df[user_input_features]
            scaled_data = scaler.transform(df_selected)
            df["Prediction"] = model.predict(scaled_data)
            df["Probability"] = model.predict_proba(scaled_data)[:, 1]
            df["Prediction"] = df["Prediction"].map({0: "Legitimate", 1: "Fraudulent"})
            
            st.write(df)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "fraud_predictions.csv", "text/csv", use_container_width=True)

# Feature Importance
with st.sidebar:
    if st.button("Show Feature Importance", use_container_width=True):
        st.subheader("Feature Importance")
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": user_input_features, "Importance": feature_importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=importance_df["Importance"], y=importance_df["Feature"], palette="coolwarm", ax=ax)
        ax.set_xlabel("Feature Importance Score")
        ax.set_ylabel("Feature Name")
        ax.set_title("Feature Importance (Random Forest)")
        st.pyplot(fig)
