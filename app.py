
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score
)

# --- 1. Load Model and Data ---
# Load the pre-trained model
model = joblib.load('netflix_churn_model.pkl')

# Load feature importance data
feature_importance = pd.read_csv('feature_importance.csv')

# Hardcoding evaluation metrics from the notebook's last run for display
# In a real-world scenario, these would typically be stored or calculated from a validation set.
accuracy = 0.8870
roc_auc = 0.9639
mcc = 0.7744
cohen_kappa = 0.7739
specificity = 0.8692
fpr = 0.1308
fnr = 0.0954

# Confusion Matrix from notebook: [[432  65], [ 48 455]]
cm_notebook = np.array([[432,  65], [ 48, 455]])

# Dummy y_test and y_pred to generate classification report, 
# as we don't have the actual test set available in the app. 
# This is for display purposes, assuming the report is static based on training evaluation.
# In a production app, evaluation reports would be pre-generated or run on a separate test set.
y_test_dummy = np.concatenate([np.zeros(cm_notebook[0,0] + cm_notebook[0,1]), np.ones(cm_notebook[1,0] + cm_notebook[1,1])])
y_pred_dummy = np.concatenate([np.zeros(cm_notebook[0,0]), np.ones(cm_notebook[0,1]), np.zeros(cm_notebook[1,0]), np.ones(cm_notebook[1,1])])
y_pred_proba_dummy = np.random.rand(len(y_test_dummy)) # Placeholder, not used for report directly

# --- 2. Streamlit Page Configuration ---
st.set_page_config(
    page_title="Netflix Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Netflix Customer Churn Prediction")
st.write("Use the sidebar to input customer data and predict churn probability.")

# --- 3. Display Metrics Function ---
def display_metrics(
    accuracy_val, roc_auc_val, mcc_val, cohen_kappa_val, specificity_val, fpr_val, fnr_val, 
    y_test_dummy, y_pred_dummy, cm_notebook
):
    st.subheader("Model Performance Metrics")
    st.markdown("--- ")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Accuracy", value=f"{accuracy_val:.4f}")
        st.metric(label="ROC-AUC Score", value=f"{roc_auc_val:.4f}")
    with col2:
        st.metric(label="Matthews Corr. Coeff.", value=f"{mcc_val:.4f}")
        st.metric(label="Cohen's Kappa", value=f"{cohen_kappa_val:.4f}")
    with col3:
        st.metric(label="Specificity", value=f"{specificity_val:.4f}")
        st.metric(label="False Positive Rate", value=f"{fpr_val:.4f}")
        st.metric(label="False Negative Rate", value=f"{fnr_val:.4f}")

    st.markdown("--- ")
    st.write("### Classification Report")
    # Generate classification report using dummy values
    report = classification_report(y_test_dummy, y_pred_dummy, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.write("### Confusion Matrix")
    st.write(cm_notebook)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_notebook, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted No Churn', 'Predicted Churn'],
                yticklabels=['Actual No Churn', 'Actual Churn'], ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_ylabel("Actual Label")
    ax_cm.set_xlabel("Predicted Label")
    st.pyplot(fig_cm)

# --- 4. Sidebar for User Input ---
st.sidebar.header("Customer Input Features")

# Define feature options (must match what the model was trained on)
subscription_types = ["Basic", "Standard", "Premium"]
genders = ["Female", "Male", "Other"]
regions = [
    "North America", "Europe", "Asia",
    "South America", "Africa", "Oceania"
]
devices = ["Laptop", "Smartphone", "Tablet", "Smart TV", "Gaming Console", "Desktop"]
payment_methods = [
    "Credit Card", "Debit Card", "PayPal",
    "Bank Transfer", "Gift Card", "Crypto"
]
favorite_genres = [
    "Action", "Comedy", "Drama", "Sci-Fi",
    "Horror", "Romance", "Documentary", "Animation"
]

# Input widgets
age = st.sidebar.slider("Age", 18, 80, 30)
watch_hours = st.sidebar.slider("Watch Hours (per month)", 0.0, 50.0, 15.0)
last_login_days = st.sidebar.slider("Days Since Last Login", 0, 90, 7)
monthly_fee = st.sidebar.slider("Monthly Fee", 5.0, 50.0, 15.0)
number_of_profiles = st.sidebar.slider("Number of Profiles", 1, 5, 1)
avg_watch_time_per_day = st.sidebar.slider("Avg Watch Time (hours/day)", 0.0, 5.0, 1.0)

gender = st.sidebar.selectbox("Gender", options=genders)
subscription_type = st.sidebar.selectbox("Subscription Type", options=subscription_types)
region = st.sidebar.selectbox("Region", options=regions)
device = st.sidebar.selectbox("Device", options=devices)
payment_method = st.sidebar.selectbox("Payment Method", options=payment_methods)
favorite_genre = st.sidebar.selectbox("Favorite Genre", options=favorite_genres)

# Create DataFrame from user input
user_input = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "subscription_type": [subscription_type],
    "watch_hours": [watch_hours],
    "last_login_days": [last_login_days],
    "monthly_fee": [monthly_fee],
    "number_of_profiles": [number_of_profiles],
    "avg_watch_time_per_day": [avg_watch_time_per_day],
    "region": [region],
    "device": [device],
    "payment_method": [payment_method],
    "favorite_genre": [favorite_genre]
})

# --- 5. Prediction Button ---
st.sidebar.markdown("--- ")
if st.sidebar.button("Predict Churn"):
    st.subheader("Prediction Result")
    try:
        prediction = model.predict(user_input)[0]
        prediction_proba = model.predict_proba(user_input)[:, 1][0]

        churn_status = "Churn" if prediction == 1 else "No Churn"
        churn_color = "red" if prediction == 1 else "green"

        st.markdown(f"<h3 style='color: {churn_color};'>Prediction: {churn_status}</h3>", unsafe_allow_html=True)
        st.write(f"Churn Probability: {prediction_proba:.2%}")

        if prediction == 1:
            st.warning("This customer is predicted to churn. Consider retention strategies.")
        else:
            st.success("This customer is predicted to not churn.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure all input fields are correctly filled.")


# --- 6. Display Model Evaluation and Feature Importance ---
st.header("Model Overview and Diagnostics")

display_metrics(
    accuracy, roc_auc, mcc, cohen_kappa, specificity, fpr, fnr,
    y_test_dummy, y_pred_dummy, cm_notebook
)

st.subheader("Feature Importance")

# Plot 1: Absolute Feature Importance
fig1, ax1 = plt.subplots(figsize=(10, 7))
sns.barplot(
    data=feature_importance.head(35),
    x="AbsCoefficient",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False,
    ax=ax1
)
ax1.set_title("Top Features Influencing Netflix Customer Churn (Absolute Impact)")
ax1.set_xlabel("Impact Strength (|Coefficient|)")
ax1.set_ylabel("Feature")
ax1.grid(axis="x", linestyle="--", alpha=0.6)
st.pyplot(fig1)

# Plot 2: Coefficient Impact Visualization
top_features = feature_importance.head(35).sort_values("Coefficient") # Sort for better visualization
fig2, ax2 = plt.subplots(figsize=(10, 7))
sns.barplot(
    data=top_features,
    x="Coefficient",
    y="Feature",
    hue="Coefficient", # Use hue for coloring based on coefficient value
    palette="coolwarm", # Use a diverging colormap
    dodge=False,
    legend=False,
    ax=ax2
)
ax2.axvline(0, color="black", linestyle="--")
ax2.set_title("Feature Impact on Netflix Customer Churn")
ax2.set_xlabel("Coefficient Value (Negative = Reduce Churn, Positive = Increase Churn)")
ax2.set_ylabel("Feature")
ax2.grid(axis="x", linestyle="--", alpha=0.6)
st.pyplot(fig2)
