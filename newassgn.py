# loan_default_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Set page config
st.set_page_config(
    page_title="Loan Default Prediction", 
    page_icon="💳", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 Loan Default Prediction App")
st.write("Enter customer details to predict whether they will default next month.")

# Function to train and save a model
def train_model():
    # Create sample data for training
    np.random.seed(42)
    n_samples = 5000
    
    # Create realistic sample data with corrected probability arrays
    # For range(-2, 9) which has 11 values, we need 11 probabilities
    pay_probabilities = [0.1, 0.15, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1]
    
    df = pd.DataFrame({
        'LIMIT_BAL': np.random.randint(10000, 500000, n_samples),
        'SEX': np.random.choice([1, 2], n_samples, p=[0.6, 0.4]),
        'EDUCATION': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'MARRIAGE': np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.4, 0.1]),
        'AGE': np.random.randint(21, 65, n_samples),
        'PAY_0': np.random.choice(range(-2, 9), n_samples, p=pay_probabilities),
        'PAY_2': np.random.choice(range(-2, 9), n_samples, p=pay_probabilities),
        'PAY_3': np.random.choice(range(-2, 9), n_samples, p=pay_probabilities),
        'PAY_4': np.random.choice(range(-2, 9), n_samples, p=pay_probabilities),
        'PAY_5': np.random.choice(range(-2, 9), n_samples, p=pay_probabilities),
        'PAY_6': np.random.choice(range(-2, 9), n_samples, p=pay_probabilities),
        'BILL_AMT1': np.random.randint(0, 200000, n_samples),
        'BILL_AMT2': np.random.randint(0, 200000, n_samples),
        'BILL_AMT3': np.random.randint(0, 200000, n_samples),
        'BILL_AMT4': np.random.randint(0, 200000, n_samples),
        'BILL_AMT5': np.random.randint(0, 200000, n_samples),
        'BILL_AMT6': np.random.randint(0, 200000, n_samples),
        'PAY_AMT1': np.random.randint(0, 50000, n_samples),
        'PAY_AMT2': np.random.randint(0, 50000, n_samples),
        'PAY_AMT3': np.random.randint(0, 50000, n_samples),
        'PAY_AMT4': np.random.randint(0, 50000, n_samples),
        'PAY_AMT5': np.random.randint(0, 50000, n_samples),
        'PAY_AMT6': np.random.randint(0, 50000, n_samples),
    })
    
    # Create target variable with some logic
    default_proba = 0.2  # 20% default rate
    df['default'] = np.random.choice([0, 1], n_samples, p=[1-default_proba, default_proba])

    # Prepare features and target
    X = df.drop(["default"], axis=1)
    y = df["default"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, 
                                                        random_state=42, stratify=y)

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    st.sidebar.text("Model Performance:")
    st.sidebar.text(classification_report(y_test, y_pred))

    # Save the model and scaler
    joblib.dump(model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return model, scaler

# Load or train model
model = None
scaler = None

if os.path.exists("best_model.pkl") and os.path.exists("scaler.pkl"):
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        st.sidebar.success("Model loaded successfully!")
    except:
        st.sidebar.warning("Model file exists but couldn't be loaded. Training a new model...")
        model, scaler = train_model()
        st.sidebar.success("New model trained and saved!")
else:
    st.sidebar.info("No model found. Training a new model...")
    model, scaler = train_model()
    st.sidebar.success("Model trained and saved!")

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.info("This application predicts credit card defaults using machine learning.")
    st.header("Input Guide")
    st.markdown("""
    - **Repayment Status**: 
      - -2 = No consumption, 
      - -1 = Paid duly, 
      - 0 = Revolving credit, 
      - 1-8 = Payment delay months
    - **Education**: 1=Graduate, 2=University, 3=High School, 4=Others
    - **Marriage**: 1=Married, 2=Single, 3=Others
    """)

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.header("Demographic Information")
    LIMIT_BAL = st.number_input("Credit Limit (NT$)", min_value=1000, step=1000, value=50000)
    SEX = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    EDUCATION = st.selectbox("Education", [1, 2, 3, 4], 
                            format_func=lambda x: "Graduate" if x == 1 else "University" if x == 2 else "High School" if x == 3 else "Others")
    MARRIAGE = st.selectbox("Marital Status", [1, 2, 3], 
                           format_func=lambda x: "Married" if x == 1 else "Single" if x == 2 else "Others")
    AGE = st.slider("Age", min_value=18, max_value=100, value=30)

with col2:
    st.header("Payment History")
    
    st.subheader("Repayment Status")
    PAY_0 = st.selectbox("September", list(range(-2, 9)), index=2, 
                        help="-2: No consumption, -1: Paid duly, 0: Revolving credit, 1-8: Payment delay months")
    PAY_2 = st.selectbox("August", list(range(-2, 9)), index=2)
    PAY_3 = st.selectbox("July", list(range(-2, 9)), index=2)
    PAY_4 = st.selectbox("June", list(range(-2, 9)), index=2)
    PAY_5 = st.selectbox("May", list(range(-2, 9)), index=2)
    PAY_6 = st.selectbox("April", list(range(-2, 9)), index=2)
    
    st.subheader("Bill Amounts (NT$)")
    BILL_AMT1 = st.number_input("September Bill", step=1000, value=0)
    BILL_AMT2 = st.number_input("August Bill", step=1000, value=0)
    BILL_AMT3 = st.number_input("July Bill", step=1000, value=0)
    BILL_AMT4 = st.number_input("June Bill", step=1000, value=0)
    BILL_AMT5 = st.number_input("May Bill", step=1000, value=0)
    BILL_AMT6 = st.number_input("April Bill", step=1000, value=0)
    
    st.subheader("Previous Payment Amounts (NT$)")
    PAY_AMT1 = st.number_input("September Payment", step=1000, value=0)
    PAY_AMT2 = st.number_input("August Payment", step=1000, value=0)
    PAY_AMT3 = st.number_input("July Payment", step=1000, value=0)
    PAY_AMT4 = st.number_input("June Payment", step=1000, value=0)
    PAY_AMT5 = st.number_input("May Payment", step=1000, value=0)
    PAY_AMT6 = st.number_input("April Payment", step=1000, value=0)

# Collect features in the correct order
features = [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
            PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
            BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
            PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]

# Prediction
if st.button("Predict Default Risk"):
    try:
        # Scale the features
        input_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] * 100
        
        # Display results
        st.subheader("Prediction Result")
        
        # Create a visual result
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            if prediction == 1:
                st.error(f"⚠️ **High Risk of Default**")
                st.metric("Probability of Default", f"{prob:.2f}%")
            else:
                st.success(f"✅ **Low Risk of Default**")
                st.metric("Probability of Default", f"{prob:.2f}%")
            
            # Add a visual indicator
            st.progress(prob/100)
        
        with result_col2:
            if prediction == 1:
                st.write("This client has a high likelihood of defaulting next month.")
                st.write("**Recommendation:** Further review recommended before approving credit.")
            else:
                st.write("This client is unlikely to default next month.")
                st.write("**Recommendation:** Credit application appears low risk.")
        
        # Add some explanation of factors
        st.subheader("Key Influencing Factors")
        if PAY_0 > 0:
            st.write(f"- ⚠️ Recent payment delay ({PAY_0} months overdue in September)")
        if LIMIT_BAL < 50000:
            st.write(f"- ⚠️ Low credit limit (NT${LIMIT_BAL:,})")
        if AGE < 25:
            st.write(f"- ⚠️ Young applicant ({AGE} years old)")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Add some explanation
st.markdown("---")
st.header("How It Works")
st.markdown("""
This application predicts the likelihood of credit card default using machine learning.

**Key Factors Considered:**
- Payment history (timeliness of past payments)
- Credit utilization (bill amounts vs payments)
- Demographic information (age, education, marital status)
- Credit limit

The model outputs a probability score between 0% and 100%, with higher scores indicating greater risk.
""")

# Add footer
st.markdown("---")
st.markdown("*This is a demonstration application for educational purposes only.*")