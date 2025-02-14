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

# Page Header
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>Credit Fraud Prediction</h1>
    <p style='text-align: center;'>Enter transaction details below or upload a CSV file to check for fraud.</p>
""", unsafe_allow_html=True)

# User choice: Upload CSV or Manual Input
input_choice = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

if input_choice == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not set(user_input_features).issubset(df.columns):
            st.error("Uploaded file does not contain the required features.")
        else:
            with st.spinner("Processing file..."):
                df_selected = df[user_input_features]
                try:
                    scaled_data = scaler.transform(df_selected)
                    predictions = model.predict(scaled_data)
                    probabilities = model.predict_proba(scaled_data)[:, 1]

                    df_selected["Prediction"] = predictions
                    df_selected["Probability"] = probabilities
                    df_selected["Prediction"] = df_selected["Prediction"].map({0: "Legitimate", 1: "Fraudulent"})

                    # Extract first prediction result for display
                    prediction_text = "Fraudulent Transaction" if predictions[0] == 1 else "Legitimate Transaction"
                    prediction_color = "#ff4b4b" if predictions[0] == 1 else "#4caf50"

                    st.markdown(
                        f"<h2 style='text-align: center; color: white; background-color: {prediction_color}; padding: 10px;'>"
                        f"Prediction: {prediction_text} (Probability: {probabilities[0]:.2f})</h2>",
                        unsafe_allow_html=True
                        )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop()
                            
                    
    

          

            # Feature Importance Visualization
            if hasattr(model, "feature_importances_"):
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
else:
    with st.sidebar:
        st.header("User Inputs")
        user_input = {feature: st.slider(f"{feature}", min_value=-10.0, max_value=10.0, value=0.0, step=0.1) for feature in user_input_features}

    # Convert user inputs to DataFrame
    input_df = pd.DataFrame([user_input], columns=user_input_features)

    if not isinstance(scaler, StandardScaler):
        st.error("Scaler is not a valid StandardScaler object.")
        st.stop()

    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Prediction Button
    if st.button("Predict", use_container_width=True):
        with st.spinner("Making predictions..."):
            try:
                prediction = model.predict(scaled_input)[0]
                probability = model.predict_proba(scaled_input)[0, 1]
                prediction_text = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
                prediction_color = "#ff4b4b" if prediction == 1 else "#4caf50"

                st.markdown(
                    f"<h2 style='text-align: center; color: white; background-color: {prediction_color}; padding: 10px;'>"
                    f"Prediction: {prediction_text} (Probability: {probability:.2f})</h2>",
                    unsafe_allow_html=True
                )

                # Feature Importance Visualization
                if hasattr(model, "feature_importances_"):
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
            except Exception as e:
                st.error(f"Error during prediction: {e}")
