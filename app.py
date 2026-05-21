import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import smtplib
import random
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import json
import os

from pymongo import MongoClient
from bson import ObjectId

# ============================================
# MongoDB Connection
# ============================================
MONGO_URI = "mongodb+srv://yashkarekar241_db_user:AM55A2FDQVp8tJU3@cluster0.zyligrf.mongodb.net/?retryWrites=true&w=majority"
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    st.error(f"❌ MongoDB connection failed: {e}")
    st.stop()

db = client["insfra"]
user_collection = db["emp information"]
analysis_collection = db["fraud"]
claims_status_collection = db["claims status"]
employee_collection = db["admin12"]

# ============================================
# Email configuration
# ============================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "yashkarekar241@gmail.com"
APP_PASSWORD = "shsl ftcw vtnn odrs"

# ============================================
# Employee database (MongoDB)
# ============================================
DEFAULT_EMPLOYEES = {
    "EMP001": {
        "name": "Yash Smith",
        "department": "Claims Department",
        "registered_email": "john.smith@example.com",
        "password": "pass123"
    },
    "EMP002": {
        "name": "Anuj Johnson",
        "department": "Underwriting",
        "registered_email": "sarah.j@example.com",
        "password": "pass123"
    },
    "EMP003": {
        "name": "Michael Chen",
        "department": "Fraud Investigation",
        "registered_email": "michael.chen@example.com",
        "password": "pass123"
    }
}

def load_employees_from_mongo():
    employees = {}
    for doc in employee_collection.find():
        emp_id = doc.get("employee_id")
        if emp_id:
            employees[emp_id] = {
                "name": doc.get("name"),
                "department": doc.get("department"),
                "registered_email": doc.get("registered_email"),
                "password": doc.get("password")
            }
    if not employees:
        for emp_id, data in DEFAULT_EMPLOYEES.items():
            employee_collection.insert_one({
                "employee_id": emp_id,
                **data
            })
            employees[emp_id] = data
    return employees

def save_employee_to_mongo(employee_id, employee_data):
    employee_collection.update_one(
        {"employee_id": employee_id},
        {"$set": employee_data},
        upsert=True
    )

def delete_employee_from_mongo(employee_id):
    employee_collection.delete_one({"employee_id": employee_id})

# ============================================
# Authentication functions
# ============================================
def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(receiver_email, otp):
    try:
        message = MIMEMultipart()
        message["From"] = SENDER_EMAIL
        message["To"] = receiver_email
        message["Subject"] = "Your OTP Code - Insurance Fraud Detection System"
        body = f"""
        Hello,
        Your One-Time Password (OTP) for authentication is: {otp}
        This OTP is valid for 5 minutes.
        Do not share this code with anyone.
        Best regards,
        Insurance Fraud Detection System
        """
        message.attach(MIMEText(body, "plain"))
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(message)
        return True
    except Exception as e:
        st.error(f"Failed to send OTP: {str(e)}")
        return False

def verify_employee(employee_id, password, db):
    if employee_id not in db:
        return False, "Invalid Employee ID"
    if db[employee_id]["password"] != password:
        return False, "Incorrect password"
    return True, "OK"

# ============================================
# Feature definitions
# ============================================
FEATURES = {
    'Vehicle': [
        'age', 'claim_amount', 'policy_tenure', 'previous_claims', 'vehicle_age',
        'time_to_report', 'witness_present', 'police_report', 'incident_severity',
        'repair_cost_estimate', 'claim_to_vehicle_value_ratio', 'accident_location_type',
        'policy_upgrade_recent', 'customer_income'
    ],
    'Home': [
        'age', 'claim_amount', 'policy_tenure', 'previous_claims', 'property_age',
        'coverage_increase_recent', 'time_to_report', 'fire_or_theft_type',
        'forced_entry_sign', 'damage_severity', 'claim_to_property_value_ratio',
        'number_of_high_value_items_claimed', 'customer_income'
    ],
    'Life': [
        'age', 'policy_tenure', 'claim_amount', 'annual_income',
        'sum_assured_to_income_ratio', 'medical_history_flag', 'cause_of_death_type',
        'death_location', 'beneficiary_change_recent', 'number_of_policies',
        'time_between_policy_and_death', 'policy_lapsed'
    ]
}

CATEGORICAL_FEATURES = {
    'Vehicle': ['witness_present', 'police_report', 'incident_severity', 'accident_location_type', 'policy_upgrade_recent'],
    'Home': ['fire_or_theft_type', 'forced_entry_sign', 'damage_severity', 'coverage_increase_recent'],
    'Life': ['medical_history_flag', 'cause_of_death_type', 'death_location', 
             'beneficiary_change_recent', 'policy_lapsed']
}

# ============================================
# Session state initialization
# ============================================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'auth_step' not in st.session_state:
    st.session_state.auth_step = 1
if 'otp_data' not in st.session_state:
    st.session_state.otp_data = {}
if 'employee_db' not in st.session_state:
    st.session_state.employee_db = load_employees_from_mongo()

if 'nav_page' not in st.session_state:
    st.session_state.nav_page = "Dashboard"
if 'history' not in st.session_state:
    st.session_state.history = []
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'uploaded_type' not in st.session_state:
    st.session_state.uploaded_type = None
if 'uploaded_predictions' not in st.session_state:
    st.session_state.uploaded_predictions = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scalers' not in st.session_state:
    st.session_state.scalers = {}
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = {}
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = {}
if 'optimal_thresholds' not in st.session_state:
    st.session_state.optimal_thresholds = {}
if 'all_sample_data' not in st.session_state:
    st.session_state.all_sample_data = None
if 'insurance_types' not in st.session_state:
    st.session_state.insurance_types = ['Vehicle', 'Home', 'Life']
if 'claim_lookup' not in st.session_state:
    st.session_state.claim_lookup = {}
if 'lookup_target' not in st.session_state:
    st.session_state.lookup_target = None
if 'sample_claims_display' not in st.session_state:
    st.session_state.sample_claims_display = []
if 'claim_approvals' not in st.session_state:
    st.session_state.claim_approvals = {}
if 'low_risk_threshold' not in st.session_state:
    st.session_state.low_risk_threshold = 0.3
if 'high_risk_threshold' not in st.session_state:
    st.session_state.high_risk_threshold = 0.7
if 'allow_missing_values' not in st.session_state:
    st.session_state.allow_missing_values = True

# Classification threshold is now always equal to low_risk_threshold
# (any probability above low_risk_threshold is considered FRAUD)
st.session_state.classification_threshold = st.session_state.low_risk_threshold

# ============================================
# Helper: sync classification threshold with low threshold
# ============================================
def sync_classification_threshold():
    """Make classification threshold always equal to low risk threshold."""
    st.session_state.classification_threshold = st.session_state.low_risk_threshold

# ============================================
# Authentication Page (Only Employee Login)
# ============================================
def show_auth_page():
    st.markdown("""
    <style>
    .main > div {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 85vh;
    }
    .auth-container {
        max-width: 380px;
        margin: 0 auto;
        padding: 1.2rem 1.5rem;
        background: #2d2d2d;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        border: 1px solid #444;
    }
    .auth-header {
        text-align: center;
        color: #4da6ff;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .stTextInput > div > div > input {
        padding: 0.5rem 0.75rem;
        font-size: 0.9rem;
        background-color: #1e2a3a;
        color: white;
        border-radius: 6px;
    }
    .stButton > button {
        padding: 0.4rem 1rem;
        font-size: 0.9rem;
        border-radius: 6px;
        background: linear-gradient(135deg, #1a3a5f 0%, #0a1a2f 100%);
        color: white;
        border: 1px solid #2a4a6f;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a4a6f 0%, #1a3a5f 100%);
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        if st.session_state.auth_step == 1:
            st.markdown('<h3 class="auth-header">Employee Login</h3>', unsafe_allow_html=True)
            employee_id = st.text_input("Employee ID", placeholder="EMP001", key="emp_id")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="emp_pass")
            email = st.text_input("Email Address", placeholder="your@email.com", key="emp_email")
            
            if st.button("Send OTP", type="primary", use_container_width=True):
                if not employee_id or not password or not email:
                    st.error("Please fill in all fields")
                else:
                    valid, msg = verify_employee(employee_id, password, st.session_state.employee_db)
                    if valid:
                        otp = generate_otp()
                        st.session_state.otp_data = {
                            'employee_id': employee_id,
                            'email': email,
                            'otp': otp,
                            'timestamp': time.time(),
                            'expires_at': time.time() + 300
                        }
                        with st.spinner("Sending OTP..."):
                            if send_otp_email(email, otp):
                                st.session_state.auth_step = 2
                                st.success("OTP sent!")
                                st.rerun()
                    else:
                        st.error(f"❌ {msg}")
        elif st.session_state.auth_step == 2:
            st.markdown('<h3 class="auth-header">Verify OTP</h3>', unsafe_allow_html=True)
            st.info(f"📧 OTP sent to: {st.session_state.otp_data['email']}")
            otp_input = st.text_input("Enter 6-digit OTP", placeholder="123456", max_chars=6)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Verify", type="primary", use_container_width=True):
                    if not otp_input or len(otp_input) != 6:
                        st.error("Enter valid 6-digit OTP")
                    else:
                        current_time = time.time()
                        if current_time > st.session_state.otp_data['expires_at']:
                            st.error("OTP expired. Request new one.")
                            st.session_state.auth_step = 1
                            st.rerun()
                        elif otp_input == st.session_state.otp_data['otp']:
                            emp_id = st.session_state.otp_data['employee_id']
                            emp_info = st.session_state.employee_db.get(emp_id, {})
                            st.session_state.authenticated = True
                            st.session_state.user_type = "employee"
                            st.session_state.current_user = {
                                'id': emp_id,
                                'email': st.session_state.otp_data['email'],
                                'name': emp_info.get('name', 'Unknown'),
                                'department': emp_info.get('department', 'Unknown')
                            }
                            user_collection.insert_one({
                                "employee_id": emp_id,
                                "email": st.session_state.otp_data['email'],
                                "name": emp_info.get('name'),
                                "department": emp_info.get('department'),
                                "login_time": datetime.now()
                            })
                            st.session_state.auth_step = 1
                            st.session_state.otp_data = {}
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid OTP")
            with col_b:
                if st.button("Back", use_container_width=True):
                    st.session_state.auth_step = 1
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# Page config
# ============================================
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Check authentication
# ============================================
if not st.session_state.authenticated:
    show_auth_page()
    st.stop()

# ============================================
# Helper functions
# ============================================
def detect_id_column(df):
    possible = ['policy_id', 'claim_id', 'id', 'policy_number', 'claim_number']
    for col in possible:
        if col in df.columns:
            return col
    return None

def add_to_lookup(df, source='uploaded'):
    id_col = detect_id_column(df)
    if id_col:
        for _, row in df.iterrows():
            claim_id = str(row[id_col])
            if claim_id not in st.session_state.claim_lookup:
                st.session_state.claim_lookup[claim_id] = {
                    'insurance_type': row.get('insurance_type', 'Unknown'),
                    'data': row.to_dict(),
                    'source': source
                }

def rebuild_sample_lookup():
    uploaded_entries = {k: v for k, v in st.session_state.claim_lookup.items() if v.get('source') == 'uploaded'}
    st.session_state.claim_lookup = uploaded_entries
    if st.session_state.all_sample_data is not None:
        for _, row in st.session_state.all_sample_data.iterrows():
            if 'claim_id' in row and pd.notna(row['claim_id']):
                st.session_state.claim_lookup[str(row['claim_id'])] = {
                    'insurance_type': row['insurance_type'],
                    'data': row.to_dict(),
                    'source': 'sample'
                }

def refresh_sample_display():
    display_list = []
    if st.session_state.all_sample_data is not None:
        for ins_type in st.session_state.insurance_types:
            type_data = st.session_state.all_sample_data[
                st.session_state.all_sample_data['insurance_type'] == ins_type
            ].copy()
            fraud_samples = type_data[type_data['is_fraud'] == 1].sample(n=min(3, len(type_data)), random_state=42)
            legit_samples = type_data[type_data['is_fraud'] == 0].sample(n=min(3, len(type_data)), random_state=42)
            samples = pd.concat([fraud_samples, legit_samples]).head(6)
            for _, row in samples.iterrows():
                prob = row['fraud_probability']
                low_thresh = st.session_state.low_risk_threshold
                high_thresh = st.session_state.high_risk_threshold
                if prob > high_thresh:
                    risk = "High"
                    color = "#FF4444"
                elif prob > low_thresh:
                    risk = "Medium"
                    color = "#FFA500"
                else:
                    risk = "Low"
                    color = "#00C851"
                display_list.append({
                    'claim_id': row['claim_id'],
                    'insurance_type': ins_type,
                    'fraud_probability': prob,
                    'risk': risk,
                    'color': color
                })
    uploaded_claims = {}
    for claim_id, info in st.session_state.claim_lookup.items():
        if info.get('source') == 'uploaded':
            ins_type = info['insurance_type']
            if ins_type not in uploaded_claims:
                uploaded_claims[ins_type] = []
            if len(uploaded_claims[ins_type]) < 3:
                prob = info['data'].get('fraud_probability', info['data'].get('Predicted_Fraud_Probability', 0.5))
                low_thresh = st.session_state.low_risk_threshold
                high_thresh = st.session_state.high_risk_threshold
                if prob > high_thresh:
                    risk = "High"
                    color = "#FF4444"
                elif prob > low_thresh:
                    risk = "Medium"
                    color = "#FFA500"
                else:
                    risk = "Low"
                    color = "#00C851"
                uploaded_claims[ins_type].append({
                    'claim_id': claim_id,
                    'insurance_type': ins_type,
                    'fraud_probability': prob,
                    'risk': risk,
                    'color': color
                })
    for claims in uploaded_claims.values():
        display_list.extend(claims)
    st.session_state.sample_claims_display = display_list

# ============================================
# Data generation and model training
# ============================================
def generate_sample_data(insurance_type, n_samples=2000):
    np.random.seed(42)
    data = {}
    if insurance_type == 'Vehicle':
        data['age'] = np.random.randint(18, 70, n_samples)
        data['claim_amount'] = np.random.exponential(80000, n_samples).clip(10000, 4000000)
        data['policy_tenure'] = np.random.randint(1, 20, n_samples)
        data['previous_claims'] = np.random.poisson(0.5, n_samples).clip(0, 5)
        data['vehicle_age'] = np.random.randint(0, 15, n_samples)
        data['time_to_report'] = np.random.poisson(5, n_samples).clip(0, 30)
        data['witness_present'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
        data['police_report'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6])
        data['incident_severity'] = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.5, 0.3, 0.2])
        data['repair_cost_estimate'] = np.random.uniform(5000, 150000, n_samples)
        data['claim_to_vehicle_value_ratio'] = np.random.uniform(0.1, 1.5, n_samples)
        data['accident_location_type'] = np.random.choice(['Urban', 'Rural', 'Highway'], n_samples)
        data['policy_upgrade_recent'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8])
        data['customer_income'] = np.random.normal(600000, 200000, n_samples).clip(300000, 1500000)
    elif insurance_type == 'Home':
        data['age'] = np.random.randint(18, 70, n_samples)
        data['claim_amount'] = np.random.exponential(120000, n_samples).clip(20000, 8000000)
        data['policy_tenure'] = np.random.randint(1, 20, n_samples)
        data['previous_claims'] = np.random.poisson(0.4, n_samples).clip(0, 4)
        data['property_age'] = np.random.randint(0, 50, n_samples)
        data['coverage_increase_recent'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85])
        data['time_to_report'] = np.random.poisson(4, n_samples).clip(0, 30)
        data['fire_or_theft_type'] = np.random.choice(['Fire', 'Theft', 'Burglary', 'Other'], n_samples)
        data['forced_entry_sign'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8])
        data['damage_severity'] = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2])
        data['claim_to_property_value_ratio'] = np.random.uniform(0.05, 1.0, n_samples)
        data['number_of_high_value_items_claimed'] = np.random.poisson(0.5, n_samples).clip(0, 5)
        data['customer_income'] = np.random.normal(700000, 250000, n_samples).clip(350000, 2000000)
    elif insurance_type == 'Life':
        data['age'] = np.random.randint(18, 70, n_samples)
        data['policy_tenure'] = np.random.randint(1, 30, n_samples)
        data['claim_amount'] = np.random.exponential(500000, n_samples).clip(100000, 50000000)
        data['annual_income'] = np.random.normal(500000, 200000, n_samples).clip(200000, 2000000)
        data['sum_assured_to_income_ratio'] = np.random.uniform(1, 20, n_samples)
        data['medical_history_flag'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
        data['cause_of_death_type'] = np.random.choice(['Natural', 'Accident', 'Suicide', 'Other'], n_samples)
        data['death_location'] = np.random.choice(['Hospital', 'Home', 'Other'], n_samples)
        data['beneficiary_change_recent'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9])
        data['number_of_policies'] = np.random.poisson(1.5, n_samples).clip(1, 5)
        data['time_between_policy_and_death'] = np.random.poisson(5, n_samples).clip(0, 30)
        data['policy_lapsed'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9])
    df = pd.DataFrame(data)
    fraud_prob = np.zeros(n_samples)
    if insurance_type == 'Vehicle':
        fraud_prob += (df['claim_amount'] / 4000000) * 0.3
        fraud_prob += (df['previous_claims'] / 5) * 0.15
        fraud_prob += ((30 - df['time_to_report']) / 30) * 0.1
        fraud_prob += (df['incident_severity'] == 'High') * 0.1
        fraud_prob += (df['witness_present'] == 'No') * 0.05
        fraud_prob += (df['police_report'] == 'No') * 0.05
        fraud_prob += ((df['claim_to_vehicle_value_ratio'] > 1.2) * 0.1)
        fraud_prob += ((df['claim_to_vehicle_value_ratio'] > 1.0) & (df['police_report'] == 'No')) * 0.1
    elif insurance_type == 'Home':
        fraud_prob += (df['claim_amount'] / 8000000) * 0.25
        fraud_prob += (df['previous_claims'] / 4) * 0.2
        fraud_prob += ((30 - df['time_to_report']) / 30) * 0.1
        fraud_prob += (df['damage_severity'] == 'High') * 0.1
        fraud_prob += (df['forced_entry_sign'] == 'No') * 0.05
        fraud_prob += ((df['claim_to_property_value_ratio'] > 0.8) & (df['fire_or_theft_type'] == 'Theft')) * 0.15
    elif insurance_type == 'Life':
        fraud_prob += (df['claim_amount'] / 50000000) * 0.35
        fraud_prob += ((df['time_between_policy_and_death'] < 2) * 0.35)
        fraud_prob += (df['beneficiary_change_recent'] == 'Yes') * 0.25
        fraud_prob += (df['medical_history_flag'] == 'Yes') * 0.1
        fraud_prob += ((df['sum_assured_to_income_ratio'] > 10) & (df['cause_of_death_type'] == 'Accident')) * 0.25
        fraud_prob += ((df['policy_lapsed'] == 'Yes') & (df['time_between_policy_and_death'] < 3)) * 0.3
    fraud_prob = np.clip(fraud_prob + np.random.normal(0, 0.1, n_samples), 0, 1)
    df['is_fraud'] = (fraud_prob > 0.4).astype(int)
    df['fraud_probability'] = fraud_prob
    df['insurance_type'] = insurance_type
    prefix = insurance_type[0].upper()
    df['claim_id'] = [f"{prefix}{i:06d}" for i in range(1, n_samples+1)]
    return df

def train_fraud_model_advanced(df, insurance_type):
    feature_cols = FEATURES[insurance_type]
    categorical_cols = CATEGORICAL_FEATURES[insurance_type]
    df_processed = df.copy()
    label_encoders = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
    X = df_processed[feature_cols]
    y = df_processed['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    gb = GradientBoostingClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        gb, param_distributions=param_grid, n_iter=20,
        cv=5, scoring='f1', random_state=42, n_jobs=-1
    )
    random_search.fit(X_train_scaled, y_train)
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        y_pred = (y_pred_proba > thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    y_pred_default = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_default)
    return best_model, accuracy, feature_cols, scaler, label_encoders, best_thresh

@st.cache_resource(show_spinner="Training fraud detection models...")
def train_all_models():
    models = {}
    scalers = {}
    label_encoders = {}
    accuracy = {}
    optimal_thresholds = {}
    feature_columns = {}
    all_samples = []
    for ins_type in st.session_state.insurance_types:
        with st.spinner(f"Training model for {ins_type} insurance..."):
            sample = generate_sample_data(ins_type, 2500)
            all_samples.append(sample)
            model, acc, features, scaler, encoders, thresh = train_fraud_model_advanced(sample, ins_type)
            models[ins_type] = model
            accuracy[ins_type] = acc
            feature_columns[ins_type] = features
            scalers[ins_type] = scaler
            label_encoders[ins_type] = encoders
            optimal_thresholds[ins_type] = thresh
    all_sample_data = pd.concat(all_samples, ignore_index=True)
    return models, scalers, label_encoders, accuracy, optimal_thresholds, feature_columns, all_sample_data

if not st.session_state.models:
    models, scalers, label_encoders, accuracy, optimal_thresholds, feature_columns, all_sample_data = train_all_models()
    st.session_state.models = models
    st.session_state.scalers = scalers
    st.session_state.label_encoders = label_encoders
    st.session_state.accuracy = accuracy
    st.session_state.optimal_thresholds = optimal_thresholds
    st.session_state.feature_columns = feature_columns
    st.session_state.all_sample_data = all_sample_data
    rebuild_sample_lookup()
    refresh_sample_display()

# ============================================
# Prediction functions
# ============================================
def preprocess_input(df, insurance_type):
    df_processed = df.copy()
    feature_cols = st.session_state.feature_columns[insurance_type]
    categorical_cols = CATEGORICAL_FEATURES[insurance_type]
    encoders = st.session_state.label_encoders[insurance_type]
    missing_cols = [col for col in feature_cols if col not in df_processed.columns]
    if missing_cols:
        if st.session_state.allow_missing_values:
            for col in missing_cols:
                if col in categorical_cols:
                    df_processed[col] = -1
                else:
                    df_processed[col] = 0
        else:
            st.error(f"Missing required columns for {insurance_type}: {missing_cols}")
            return None
    for col in categorical_cols:
        if col in df_processed.columns:
            le = encoders[col]
            df_processed[col] = df_processed[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    X = df_processed[feature_cols]
    return X

def predict_fraud(input_df, insurance_type):
    if insurance_type not in st.session_state.models:
        st.error(f"No model trained for {insurance_type} insurance.")
        return None, None
    model = st.session_state.models[insurance_type]
    scaler = st.session_state.scalers[insurance_type]
    X = preprocess_input(input_df, insurance_type)
    if X is None:
        return None, None
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    # Classification threshold is now equal to low_risk_threshold
    thresh = st.session_state.classification_threshold
    predictions = (probabilities > thresh).astype(int)
    return predictions, probabilities

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-header {
        font-size: 2rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        padding: 0.75rem;
        background: linear-gradient(135deg, #1a3a5f 0%, #0a1a2f 100%);
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        border: 1px solid #2a4a6f;
    }
    .section-header {
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
        padding: 0.4rem 1rem;
        background: linear-gradient(135deg, #1a3a5f 0%, #0a1a2f 100%);
        border-radius: 8px;
        border: 1px solid #2a4a6f;
    }
    .sub-header {
        font-size: 1.1rem;
        margin-bottom: 0.6rem;
        padding: 0.3rem 1rem;
        background: linear-gradient(135deg, #1a3a5f 0%, #0a1a2f 100%);
        border-radius: 6px;
        border: 1px solid #2a4a6f;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #0f1a24 100%);
        padding: 0.6rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #4da6ff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        margin-bottom: 0.5rem;
    }
    .metric-card label {
        color: #a0c4ff !important;
    }
    .metric-card div {
        color: white !important;
    }
    .user-card {
        background: linear-gradient(135deg, #1a3a5f 0%, #0a1a2f 100%);
        padding: 0.75rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #4da6ff;
        border: 1px solid #2a4a6f;
    }
    .insurance-type-card {
        background: #1e2a3a;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #4da6ff;
        border: 1px solid #2a4a6f;
    }
    .nav-container {
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
        margin: 0.5rem 0;
    }
    .nav-btn {
        background: transparent;
        border: none;
        color: #b0c4de;
        text-align: left;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        width: 100%;
        border-left: 3px solid transparent;
    }
    .nav-btn:hover {
        background: rgba(74, 144, 226, 0.15);
        color: white;
        border-left-color: #4da6ff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1a3a5f 0%, #0a1a2f 100%);
        color: white;
        border: 1px solid #2a4a6f;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2a4a6f 0%, #1a3a5f 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .dataframe {
        background-color: #1e2a3a !important;
        color: white !important;
        border: 1px solid #2a4a6f;
    }
    .dataframe th {
        background-color: #0f1a24 !important;
        color: #4da6ff !important;
    }
    .dataframe td {
        background-color: #1e2a3a !important;
        color: white !important;
    }
    .stTextInput input, .stSelectbox div, .stMultiSelect div {
        background-color: #1e2a3a !important;
        border-color: #2a4a6f !important;
        color: white !important;
    }
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    .js-plotly-plot .plotly .bg {
        fill: #1e2a3a !important;
    }
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.markdown(f"""
    <div class="user-card">
        <h4 style="color: white; margin: 0;">👤 {st.session_state.current_user['name']}</h4>
        <p style="margin: 2px 0; font-size: 0.8rem; color: #a0c4ff;">ID: {st.session_state.current_user['id']}</p>
        <p style="margin: 2px 0; font-size: 0.8rem; color: #a0c4ff;">Type: {st.session_state.user_type.title()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: white; text-align: center;'>🔍 Insurance Fraud Detection</h3>", unsafe_allow_html=True)
    
    if st.session_state.user_type == "employee":
        with st.expander("🔐 Switch to Admin (Admin Login)", expanded=False):
            st.caption("Enter admin credentials to switch role")
            admin_id_sw = st.text_input("Admin ID", key="sidebar_admin_id", placeholder="ADMIN")
            admin_pass_sw = st.text_input("Password", type="password", key="sidebar_admin_pass", placeholder="admin123")
            if st.button("Login as Admin", key="sidebar_admin_btn", use_container_width=True):
                if admin_id_sw == "ADMIN" and admin_pass_sw == "admin123":
                    st.session_state.user_type = "admin"
                    st.session_state.current_user = {
                        'id': 'ADMIN',
                        'name': 'Administrator',
                        'department': 'System Admin'
                    }
                    st.success("Switched to Admin mode!")
                    st.rerun()
                else:
                    st.error("Invalid admin credentials")
    
    if st.session_state.user_type == "admin":
        pages = [
            ("📊", "Dashboard"),
            ("📤", "Upload Data"),
            ("🧪", "Prediction"),
            ("✅", "Claim Approval"),
            ("🔎", "Search"),
            ("📈", "Analytics"),
            ("📜", "History"),
            ("⚙️", "Settings"),
            ("👥", "Manage Employees")
        ]
    else:
        pages = [
            ("📊", "Dashboard"),
            ("📤", "Upload Data"),
            ("🧪", "Prediction"),
            ("✅", "Claim Approval"),
            ("🔎", "Search"),
            ("📈", "Analytics"),
            ("📜", "History"),
            ("⚙️", "Settings")
        ]
    
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    for icon, page in pages:
        if st.button(
            f"{icon} {page}",
            key=f"nav_{page}",
            use_container_width=True,
            type="secondary" if st.session_state.nav_page != page else "primary"
        ):
            st.session_state.nav_page = page
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div class='insurance-type-card'>", unsafe_allow_html=True)
    st.write("**Filter by Insurance Type:**")
    selected_insurance = st.multiselect(
        "Select types to focus on:",
        options=st.session_state.insurance_types,
        default=st.session_state.insurance_types,
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.info("Detect fraudulent insurance claims using machine learning")
    
    st.markdown("---")
    if st.button("🚪 Logout", type="secondary", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_type = None
        st.session_state.current_user = None
        st.rerun()

# ============================================
# Dashboard page
# ============================================
if st.session_state.nav_page == "Dashboard":
    st.markdown("<h1 class='main-header'>Insurance Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    dashboard_data = st.session_state.all_sample_data.copy()
    
    if 'Predicted_Fraud_Probability' not in dashboard_data.columns:
        dashboard_data['Predicted_Fraud_Probability'] = dashboard_data['fraud_probability']
        
    def get_prediction(row):
        prob = row['fraud_probability']
        # Now uses classification_threshold which equals low_risk_threshold
        return 'Fraud' if prob > st.session_state.classification_threshold else 'Legitimate'
    
    dashboard_data['Prediction'] = dashboard_data.apply(get_prediction, axis=1)
    
    if st.session_state.uploaded_data is not None:
        uploaded = st.session_state.uploaded_data.copy()
        ins_type = st.session_state.uploaded_type
        if 'Predicted_Fraud_Probability' not in uploaded.columns:
            with st.spinner("Analyzing uploaded data..."):
                preds, probs = predict_fraud(uploaded, ins_type)
                if preds is not None:
                    uploaded['Predicted_Fraud_Probability'] = probs
                    uploaded['Prediction'] = ['Fraud' if p == 1 else 'Legitimate' for p in preds]
        if 'fraud_probability' not in uploaded.columns:
            uploaded['fraud_probability'] = uploaded['Predicted_Fraud_Probability']
        if 'is_fraud' not in uploaded.columns:
            uploaded['is_fraud'] = (uploaded['Prediction'] == 'Fraud').astype(int)
        if 'insurance_type' in uploaded.columns:
            dashboard_data = pd.concat([dashboard_data, uploaded], ignore_index=True)
        else:
            uploaded['insurance_type'] = ins_type
            dashboard_data = pd.concat([dashboard_data, uploaded], ignore_index=True)

    filtered_data = dashboard_data[dashboard_data['insurance_type'].isin(selected_insurance)].copy()
    
    if filtered_data.empty:
        st.warning("No data matches the selected insurance types.")
        st.stop()

    low_thresh = st.session_state.low_risk_threshold
    high_thresh = st.session_state.high_risk_threshold
    total_claims = len(filtered_data)
    fraudulent_claims = (filtered_data['Prediction'] == 'Fraud').sum()
    suspected_cases = ((filtered_data['Predicted_Fraud_Probability'] > low_thresh) & 
                       (filtered_data['Predicted_Fraud_Probability'] < high_thresh)).sum()
    high_risk_alerts = (filtered_data['Predicted_Fraud_Probability'] > high_thresh).sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Claims", f"{total_claims:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Fraudulent Claims", f"{fraudulent_claims:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Suspected Cases", f"{suspected_cases:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("High Risk Alerts", f"{high_risk_alerts:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Fraud Analysis by Insurance Type</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fraud_by_type = filtered_data.groupby('insurance_type').agg({
            'Prediction': lambda x: (x == 'Fraud').sum(),
            'claim_id': 'count'
        }).rename(columns={'Prediction': 'fraud_count', 'claim_id': 'total_count'})
        
        fraud_by_type['fraud_rate'] = (fraud_by_type['fraud_count'] / fraud_by_type['total_count'] * 100).round(1)
        fraud_by_type = fraud_by_type.reset_index()
        
        fig = px.bar(fraud_by_type, x='insurance_type', y='fraud_rate',
                    title='Fraud Rate by Insurance Type (%)',
                    color='fraud_rate',
                    color_continuous_scale='RdYlGn_r',
                    text='fraud_rate')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(
            xaxis_title="Insurance Type",
            yaxis_title="Fraud Rate (%)",
            plot_bgcolor='#1e2a3a',
            paper_bgcolor='#1e2a3a',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fraud_dist = filtered_data.groupby(['insurance_type', 'Prediction']).size().reset_index(name='count')
        
        fig = px.bar(fraud_dist, x='insurance_type', y='count', color='Prediction',
                    title='Claims Distribution by Type and Status',
                    color_discrete_map={'Fraud': '#FF4444', 'Legitimate': '#00C851'},
                    barmode='group')
        fig.update_layout(
            xaxis_title="Insurance Type",
            yaxis_title="Number of Claims",
            plot_bgcolor='#1e2a3a',
            paper_bgcolor='#1e2a3a',
            font=dict(color='white'),
            legend_title_text='Status'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Fraud Rate Over Time</div>", unsafe_allow_html=True)

    if 'claim_date' in filtered_data.columns:
        filtered_data['claim_date'] = pd.to_datetime(filtered_data['claim_date'])
        monthly = filtered_data.set_index('claim_date').resample('M').agg({
            'claim_amount': 'sum',
            'Predicted_Fraud_Probability': 'mean',
            'Prediction': lambda x: (x == 'Fraud').sum()
        }).reset_index()
        monthly['total_claims'] = filtered_data.set_index('claim_date').resample('M').size().values
        monthly['fraud_rate'] = (monthly['Prediction'] / monthly['total_claims'] * 100)
        
        months = monthly['claim_date'].dt.strftime('%b %Y')
        fraud_rate = monthly['fraud_rate'].fillna(0)
        total_claims_monthly = monthly['claim_amount'].fillna(0)
    else:
        months = pd.date_range(start='2024-01-01', end='2024-08-01', freq='MS').strftime('%b %Y').tolist()
        fraud_rate = [2.1, 2.8, 2.5, 3.2, 3.5, 4.0, 4.5, 3.8]
        total_claims_monthly = [1100, 1250, 1180, 1350, 1420, 1380, 1550, 1480]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=fraud_rate, mode='lines+markers', name='Fraud Rate',
                             line=dict(color='#00FFFF', width=3), yaxis='y'))
    fig.add_trace(go.Scatter(x=months, y=total_claims_monthly, mode='lines+markers', name='Total Claims',
                             line=dict(color='#FFA500', width=3), yaxis='y2'))
    fig.update_layout(
        xaxis=dict(title='Month'),
        yaxis=dict(title='Fraud Rate (%)', side='left', color='#00FFFF'),
        yaxis2=dict(title='Total Claims (₹)', overlaying='y', side='right', color='#FFA500'),
        plot_bgcolor='#1e2a3a', 
        paper_bgcolor='#1e2a3a', 
        font=dict(color='white'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Recent Suspicious Claims</div>", unsafe_allow_html=True)

    suspicious = filtered_data[filtered_data['Prediction'] == 'Fraud'].head(4)
    if not suspicious.empty:
        display_data = []
        for _, row in suspicious.iterrows():
            prob = row['Predicted_Fraud_Probability']
            risk_level = 'High' if prob > high_thresh else ('Medium' if prob > low_thresh else 'Low')
            display_data.append({
                'Claim ID': row.get('claim_id', 'N/A'),
                'Insurance Type': row['insurance_type'],
                'Amount': f"₹{row['claim_amount']:,.0f}",
                'Risk Score': f"{prob:.1%}",
                'Risk Level': risk_level
            })
        
        df_table = pd.DataFrame(display_data)
        
        def color_risk(val):
            if val == 'High':
                return 'background-color: #ff444420; color: #ff4444'
            elif val == 'Medium':
                return 'background-color: #ffa50020; color: #ffa500'
            return ''
        
        styled_df = df_table.style.map(color_risk, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No suspicious claims found in filtered data.")

    st.markdown("<div class='sub-header'>Key Fraud Indicators by Insurance Type</div>", unsafe_allow_html=True)
    
    for ins_type in selected_insurance:
        type_data = filtered_data[filtered_data['insurance_type'] == ins_type]
        if not type_data.empty:
            with st.expander(f"📊 {ins_type} Insurance Indicators", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                fraud_count = (type_data['Prediction'] == 'Fraud').sum()
                total_type = len(type_data)
                fraud_rate_type = (fraud_count / total_type * 100) if total_type > 0 else 0
                
                with col1:
                    st.metric("Fraud Rate", f"{fraud_rate_type:.1f}%")
                with col2:
                    avg_claim = type_data['claim_amount'].mean()
                    st.metric("Avg Claim Amount", f"₹{avg_claim:,.0f}")
                with col3:
                    if ins_type == 'Vehicle':
                        late_reports = (type_data['time_to_report'] > 7).sum()
                        st.metric("Late Reports (>7 days)", f"{late_reports}")
                    elif ins_type == 'Home':
                        forced_entry = (type_data['forced_entry_sign'] == 'Yes').sum() if 'forced_entry_sign' in type_data.columns else 0
                        st.metric("Forced Entry Claims", f"{forced_entry}")
                    elif ins_type == 'Life':
                        recent_policies = (type_data['time_between_policy_and_death'] < 2).sum() if 'time_between_policy_and_death' in type_data.columns else 0
                        st.metric("Recent Policies (<2 yrs)", f"{recent_policies}")
                with col4:
                    if ins_type == 'Vehicle':
                        high_severity = (type_data['incident_severity'] == 'High').sum() if 'incident_severity' in type_data.columns else 0
                        st.metric("High Severity", f"{high_severity}")
                    elif ins_type == 'Home':
                        high_value = (type_data['number_of_high_value_items_claimed'] > 3).sum() if 'number_of_high_value_items_claimed' in type_data.columns else 0
                        st.metric("High Value Items", f"{high_value}")
                    elif ins_type == 'Life':
                        beneficiary_changes = (type_data['beneficiary_change_recent'] == 'Yes').sum() if 'beneficiary_change_recent' in type_data.columns else 0
                        st.metric("Beneficiary Changes", f"{beneficiary_changes}")

# ============================================
# Upload Data Page
# ============================================
elif st.session_state.nav_page == "Upload Data":
    st.markdown("<h1 class='main-header'>Upload Insurance Claims Data</h1>", unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is not None:
        st.success(f"✅ Data already loaded: **{st.session_state.uploaded_type}** insurance, {len(st.session_state.uploaded_data)} records.")
        
        with st.expander("Show currently loaded data", expanded=False):
            st.dataframe(st.session_state.uploaded_data.head(10))
            st.caption("Showing first 10 rows. Full data is stored in session.")
        
        if st.button("Replace with new file", type="secondary"):
            st.session_state.uploaded_data = None
            st.session_state.uploaded_type = None
            st.session_state.uploaded_predictions = None
            st.session_state.claim_approvals = {}
            st.rerun()
        
        st.markdown("---")
        st.subheader("Or upload a new file")
    
    uploaded_file = st.file_uploader("Drag and drop file here", type=['csv'], key="file_uploader")
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            
            if 'insurance_type' not in df_uploaded.columns:
                st.error("Uploaded file must contain an 'insurance_type' column.")
                st.stop()
            
            unique_types = df_uploaded['insurance_type'].unique()
            if len(unique_types) > 1:
                st.error("Uploaded file contains multiple insurance types. Please upload data for only one type.")
                st.stop()
            
            ins_type = unique_types[0]
            if ins_type not in st.session_state.insurance_types:
                st.error(f"Unknown insurance type: {ins_type}. Allowed: {st.session_state.insurance_types}")
                st.stop()
            
            required_cols = FEATURES[ins_type]
            missing = [col for col in required_cols if col not in df_uploaded.columns]
            if missing and not st.session_state.allow_missing_values:
                st.error(f"Missing required columns for {ins_type} insurance: {missing}. Enable 'Allow missing values' in Settings to proceed.")
                st.stop()
            
            if 'is_fraud' in df_uploaded.columns:
                df_uploaded['is_fraud'] = df_uploaded['is_fraud'].astype(int)
            
            st.success("File uploaded and validated successfully!")
            st.session_state.uploaded_data = df_uploaded
            st.session_state.uploaded_type = ins_type
            st.session_state.uploaded_predictions = None
            st.session_state.claim_approvals = {}

            add_to_lookup(df_uploaded, source='uploaded')
            refresh_sample_display()
            
            st.info(f"**Uploaded file:** {uploaded_file.name}")
            
            policy_id_col = detect_id_column(df_uploaded)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records Uploaded", len(df_uploaded))
            with col2:
                st.metric("Insurance Type", ins_type)
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            if policy_id_col:
                st.markdown(f"**Policy/Claim ID column detected:** `{policy_id_col}`")
                st.markdown("**Sample Policy/Claim IDs:**")
                sample_ids = df_uploaded[policy_id_col].head(5).tolist()
                st.write(sample_ids)
                st.success("✅ These IDs are now available for lookup in the 'Search' page!")
            else:
                st.warning("No policy/claim ID column found. Claims from this file cannot be looked up by ID.")
            
            st.markdown("<div class='section-header'>Data Preview</div>", unsafe_allow_html=True)
            st.dataframe(df_uploaded.head(10))
            
            with st.expander("View Full Uploaded Data"):
                st.dataframe(df_uploaded)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        if st.session_state.uploaded_data is None:
            st.info("Please upload a CSV file containing insurance claims data.")

# ============================================
# Prediction Page (formerly Test Data)
# ============================================
elif st.session_state.nav_page == "Prediction":
    st.markdown("<h1 class='main-header'>Predict Fraud on Insurance Claims</h1>", unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is None:
        st.warning("Please upload data first in the 'Upload Data' section")
        st.stop()

    test_data = st.session_state.uploaded_data.copy()
    ins_type = st.session_state.uploaded_type
    
    predictions, probabilities = predict_fraud(test_data, ins_type)
    if predictions is None:
        st.stop()
    
    results_df = test_data.copy()
    results_df['Predicted_Fraud_Probability'] = probabilities
    results_df['Prediction'] = ['Fraud' if p == 1 else 'Legitimate' for p in predictions]
    
    id_col = detect_id_column(results_df)
    if id_col is None:
        id_col = '_temp_id'
        results_df[id_col] = [f"row_{i}" for i in range(len(results_df))]
    
    for idx, row in results_df.iterrows():
        claim_id = str(row[id_col])
        if claim_id in st.session_state.claim_lookup:
            st.session_state.claim_lookup[claim_id]['predicted_fraud_probability'] = float(row['Predicted_Fraud_Probability'])
            st.session_state.claim_lookup[claim_id]['prediction'] = row['Prediction']
        else:
            st.session_state.claim_lookup[claim_id] = {
                'insurance_type': ins_type,
                'data': row.to_dict(),
                'source': 'uploaded',
                'predicted_fraud_probability': float(row['Predicted_Fraud_Probability']),
                'prediction': row['Prediction']
            }
    
    employee_id = st.session_state.current_user['id']
    timestamp = datetime.now()
    
    inserted_count = 0
    for idx, row in results_df.iterrows():
        features_dict = {}
        for col in FEATURES[ins_type]:
            if col in row:
                val = row[col]
                if pd.isna(val):
                    val = None
                elif isinstance(val, (np.integer, np.int64)):
                    val = int(val)
                elif isinstance(val, (np.floating, np.float64)):
                    val = float(val)
                features_dict[col] = val
        
        doc = {
            "employee_id": employee_id,
            "insurance_type": ins_type,
            "timestamp": timestamp,
            "claim_id": str(row[id_col]),
            "fraud_probability": float(row['Predicted_Fraud_Probability']),
            "prediction": row['Prediction'],
            "features": features_dict
        }
        if 'is_fraud' in row:
            doc['actual_fraud'] = int(row['is_fraud'])
        
        try:
            result = analysis_collection.insert_one(doc)
            inserted_count += 1
        except Exception as e:
            st.error(f"Failed to insert claim {row[id_col]}: {str(e)}")
    
    if inserted_count > 0:
        st.success(f"✅ {inserted_count} claim analysis records saved to database.")
    else:
        st.warning("No records were saved to the database. Check errors above.")
    
    has_actual = 'is_fraud' in results_df.columns
    if has_actual:
        results_df['is_fraud'] = results_df['is_fraud'].astype(int)
        results_df['Actual'] = ['Fraud' if f == 1 else 'Legitimate' for f in results_df['is_fraud']]
        results_df['Correct'] = results_df['Prediction'] == results_df['Actual']
    
    history_entry = {
        'timestamp': datetime.now(),
        'type': 'Batch Analysis',
        'data_source': 'Uploaded',
        'num_records': len(results_df),
        'fraud_count': sum(predictions),
        'fraud_rate': f"{(sum(predictions)/len(results_df)*100):.1f}%",
        'insurance_type': ins_type
    }
    st.session_state.history.append(history_entry)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Claims", len(results_df))
    with col2:
        fraud_count = sum(predictions)
        st.metric("Predicted Fraud", fraud_count)
    with col3:
        if has_actual:
            actual_fraud = results_df['is_fraud'].sum()
            st.metric("Actual Fraud", actual_fraud)
        else:
            st.metric("Insurance Type", ins_type)
    with col4:
        if has_actual:
            accuracy = sum(results_df['Correct']) / len(results_df)
            st.metric("Accuracy", f"{accuracy:.2%}")
        else:
            st.metric("Fraud Rate", f"{(fraud_count/len(results_df)*100):.1f}%")
    
    st.markdown("<div class='section-header'>Claims Data with Predictions</div>", unsafe_allow_html=True)
    display_cols = ['age', 'claim_amount', 'insurance_type', 'Predicted_Fraud_Probability', 'Prediction']
    if has_actual:
        display_cols.extend(['Actual', 'Correct'])
    st.dataframe(results_df[display_cols].head(20))
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(results_df, x='Predicted_Fraud_Probability', nbins=20,
                          title='Predicted Fraud Probability Distribution',
                          color_discrete_sequence=['#FF4500'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if has_actual:
            correct_counts = results_df['Correct'].value_counts()
            fig = px.pie(values=correct_counts.values, 
                         names=correct_counts.index.map({True: 'Correct', False: 'Incorrect'}),
                        title='Prediction Accuracy',
                        color_discrete_sequence=['#00FF00', '#FF4500'])
        else:
            prediction_counts = results_df['Prediction'].value_counts()
            fig = px.pie(values=prediction_counts.values, names=prediction_counts.index,
                        title='Prediction Distribution',
                        color_discrete_sequence=['#00FFFF', '#FF00FF'])
        st.plotly_chart(fig, use_container_width=True)
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Analysis Results as CSV",
        data=csv,
        file_name=f"fraud_analysis_{ins_type}.csv",
        mime="text/csv",
    )

# ============================================
# Claim Approval Page
# ============================================
elif st.session_state.nav_page == "Claim Approval":
    st.markdown("<h1 class='main-header'>Claim Approval & Reassessment</h1>", unsafe_allow_html=True)

    if st.session_state.uploaded_data is None:
        st.warning("Please upload data first in the 'Upload Data' section.")
        st.stop()

    if st.session_state.uploaded_predictions is None:
        with st.spinner("Running fraud detection model on uploaded data..."):
            preds, probs = predict_fraud(st.session_state.uploaded_data, st.session_state.uploaded_type)
            if preds is not None:
                pred_df = pd.DataFrame({
                    'Predicted_Fraud': preds,
                    'Predicted_Fraud_Probability': probs
                })
                st.session_state.uploaded_predictions = pred_df
            else:
                st.error("Failed to generate predictions. Please check your data.")
                st.stop()

    df = st.session_state.uploaded_data.copy()
    df = pd.concat([df, st.session_state.uploaded_predictions], axis=1)

    id_col = detect_id_column(df)
    if id_col is None:
        st.warning("No claim/policy ID column found. Using row index as temporary ID. Changes will not persist if data is reordered.")
        df['temp_id'] = [f"row_{i}" for i in range(len(df))]
        id_col = 'temp_id'
    else:
        df[id_col] = df[id_col].astype(str)

    claim_ids = df[id_col].tolist()
    existing_statuses = {}
    for claim_id in claim_ids:
        doc = claims_status_collection.find_one({
            "employee_id": st.session_state.current_user['id'],
            "claim_id": claim_id
        })
        if doc:
            existing_statuses[claim_id] = doc.get("status", "Pending")
        else:
            existing_statuses[claim_id] = "Pending"

    st.session_state.claim_approvals.update(existing_statuses)
    for cid in claim_ids:
        if cid not in st.session_state.claim_approvals:
            st.session_state.claim_approvals[cid] = "Pending"

    df['Approval Status'] = df[id_col].map(st.session_state.claim_approvals)

    display_cols = [id_col, 'insurance_type', 'claim_amount', 
                    'Predicted_Fraud_Probability', 'Prediction', 'Approval Status']
    display_cols = [c for c in display_cols if c in df.columns]
    display_df = df[display_cols].copy()

    if 'claim_amount' in display_df.columns:
        display_df['claim_amount'] = display_df['claim_amount'].apply(lambda x: f"₹{x:,.0f}")
    if 'Predicted_Fraud_Probability' in display_df.columns:
        display_df['Predicted_Fraud_Probability'] = display_df['Predicted_Fraud_Probability'].apply(lambda x: f"{x:.1%}")

    st.markdown("### Review and update claim approval status")
    st.caption("Select a status for each claim. Changes are saved automatically to the database.")

    edited_df = st.data_editor(
        display_df,
        column_config={
            "Approval Status": st.column_config.SelectboxColumn(
                "Approval Status",
                help="Current approval status",
                width="medium",
                options=["Pending", "Approved", "Reassessment"],
                required=True
            )
        },
        disabled=[id_col, 'insurance_type', 'claim_amount', 'Predicted_Fraud_Probability', 'Prediction'],
        hide_index=True,
        use_container_width=True,
        key="claim_approval_editor"
    )

    if edited_df is not None:
        for idx, row in edited_df.iterrows():
            cid = row[id_col]
            new_status = row['Approval Status']
            old_status = st.session_state.claim_approvals.get(cid)
            if new_status != old_status:
                st.session_state.claim_approvals[cid] = new_status
                prob_value = 0.0
                if 'Predicted_Fraud_Probability' in df.columns:
                    prob_series = df.loc[df[id_col] == cid, 'Predicted_Fraud_Probability']
                    if not prob_series.empty:
                        prob_value = float(prob_series.iloc[0])
                doc = {
                    "employee_id": st.session_state.current_user['id'],
                    "claim_id": cid,
                    "insurance_type": st.session_state.uploaded_type,
                    "status": new_status,
                    "updated_at": datetime.now(),
                    "fraud_probability": prob_value
                }
                claims_status_collection.update_one(
                    {"employee_id": st.session_state.current_user['id'], "claim_id": cid},
                    {"$set": doc},
                    upsert=True
                )
        st.success("✅ Approval statuses updated in database.")

    st.markdown("### Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Claims", len(df))
    with col2:
        approved = sum(1 for s in st.session_state.claim_approvals.values() if s == "Approved")
        st.metric("Approved", approved)
    with col3:
        pending = sum(1 for s in st.session_state.claim_approvals.values() if s == "Pending")
        st.metric("Pending", pending)
    with col4:
        reassess = sum(1 for s in st.session_state.claim_approvals.values() if s == "Reassessment")
        st.metric("Reassessment", reassess)

    filter_status = st.multiselect("Filter by approval status", 
                                    options=["Pending", "Approved", "Reassessment"],
                                    default=[])
    if filter_status:
        filtered_ids = [cid for cid, status in st.session_state.claim_approvals.items() if status in filter_status]
        filtered_df = df[df[id_col].isin(filtered_ids)]
        filtered_display = filtered_df[display_cols].copy()
        if 'claim_amount' in filtered_display.columns:
            filtered_display['claim_amount'] = filtered_display['claim_amount'].apply(lambda x: f"₹{x:,.0f}")
        if 'Predicted_Fraud_Probability' in filtered_display.columns:
            filtered_display['Predicted_Fraud_Probability'] = filtered_display['Predicted_Fraud_Probability'].apply(lambda x: f"{x:.1%}")
        st.dataframe(filtered_display, use_container_width=True, hide_index=True)

# ============================================
# Search Page
# ============================================
elif st.session_state.nav_page == "Search":
    st.markdown("<h1 class='main-header'>Individual Claim Assessment</h1>", unsafe_allow_html=True)
    
    st.markdown("## 🔎 Search Claim by ID")
    st.info("Enter a Claim ID from the sample data or from an uploaded file.")
    
    col1, col2 = st.columns([3,1])
    with col1:
        lookup_id = st.text_input("Claim ID", key="lookup_id_input", placeholder="e.g., V000001 or any uploaded ID")
    with col2:
        analyze_clicked = st.button("Analyze", use_container_width=True)
    
    if analyze_clicked and lookup_id:
        if lookup_id.strip() in st.session_state.claim_lookup:
            st.session_state.lookup_target = lookup_id.strip()
            st.rerun()
        else:
            st.error(f"Claim ID '{lookup_id}' not found.")
    
    if 'lookup_target' in st.session_state and st.session_state.lookup_target:
        target_id = st.session_state.lookup_target
        if target_id in st.session_state.claim_lookup:
            result = st.session_state.claim_lookup[target_id]
            
            if 'predicted_fraud_probability' in result and 'prediction' in result:
                probability = result['predicted_fraud_probability']
                prediction = result['prediction']
            else:
                input_df = pd.DataFrame([result['data']])
                pred, prob = predict_fraud(input_df, result['insurance_type'])
                if pred is None:
                    st.error("Failed to get prediction for this claim.")
                    st.stop()
                probability = prob[0]
                prediction = 'Fraud' if pred[0] == 1 else 'Legitimate'
                result['predicted_fraud_probability'] = probability
                result['prediction'] = prediction
            
            low_thresh = st.session_state.low_risk_threshold
            high_thresh = st.session_state.high_risk_threshold
            if probability > high_thresh:
                risk_level = "High"
            elif probability > low_thresh:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            st.markdown("---")
            st.markdown(f"### Prediction Result for Claim **{target_id}**")
            
            col1, col2 = st.columns(2)
            with col1:
                if risk_level == "High":
                    st.error(f"🚨 **HIGH RISK** – Fraud Probability: {probability:.2%}")
                elif risk_level == "Medium":
                    st.warning(f"⚠️ **MEDIUM RISK** – Fraud Probability: {probability:.2%}")
                else:
                    st.success(f"✅ **LOW RISK** – Fraud Probability: {probability:.2%}")
                
                st.markdown(f"**Insurance Type:** {result['insurance_type']}")
                st.markdown(f"**Source:** {result['source'].title()} Data")
                
                ins_type = result['insurance_type']
                all_data = result['data']
                base_cols = ['claim_id', 'insurance_type', 'is_fraud', 'fraud_probability']
                relevant_features = FEATURES.get(ins_type, [])
                display_keys = base_cols + relevant_features
                filtered_details = {k: all_data[k] for k in display_keys if k in all_data}
                
                with st.expander("View Full Claim Details"):
                    st.json(filtered_details)
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#FF4500"},
                        'steps': [
                            {'range': [0, low_thresh*100], 'color': "#00FF00"},
                            {'range': [low_thresh*100, high_thresh*100], 'color': "#FFFF00"},
                            {'range': [high_thresh*100, 100], 'color': "#FF0000"}],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': high_thresh*100}}))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            
            history_entry = {
                'timestamp': datetime.now(),
                'type': 'Single Claim',
                'claim_id': target_id,
                'fraud_probability': probability,
                'risk_level': risk_level,
                'insurance_type': result['insurance_type'],
                'source': result['source']
            }
            st.session_state.history.append(history_entry)
        else:
            st.error(f"Claim ID '{target_id}' not found in lookup.")
        
        del st.session_state.lookup_target

# ============================================
# Analytics Page
# ============================================
elif st.session_state.nav_page == "Analytics":
    st.markdown("<h1 class='main-header'>Advanced Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is None:
        st.warning("Please upload data first in the 'Upload Data' section")
        st.stop()
    
    analytics_data = st.session_state.uploaded_data.copy()
    ins_type = st.session_state.uploaded_type
    
    if st.session_state.uploaded_predictions is None:
        with st.spinner("Running fraud detection model on uploaded data..."):
            preds, probs = predict_fraud(analytics_data, ins_type)
            if preds is not None:
                pred_df = pd.DataFrame({
                    'Predicted_Fraud': preds,
                    'Predicted_Fraud_Probability': probs
                })
                st.session_state.uploaded_predictions = pred_df
            else:
                st.error("Failed to generate predictions. Please check your data.")
                st.stop()
    
    analytics_with_pred = pd.concat([analytics_data, st.session_state.uploaded_predictions], axis=1)
    
    graph_options = ["Claim Amount Distribution", "Age Distribution", "Correlation Heatmap"]
    graph_type = st.selectbox("Select Graph Type", graph_options)
    
    if graph_type == "Claim Amount Distribution":
        if 'claim_amount' in analytics_with_pred.columns:
            fig = px.histogram(analytics_with_pred, x='claim_amount', nbins=20,
                              title='Claim Amount Distribution',
                              color_discrete_sequence=['#00FFFF'])
            fig.update_layout(xaxis_title="Claim Amount (₹)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Column 'claim_amount' not found in the uploaded data.")
    
    elif graph_type == "Age Distribution":
        if 'age' in analytics_with_pred.columns:
            fig = px.histogram(analytics_with_pred, x='age', nbins=15,
                              title='Age Distribution',
                              color_discrete_sequence=['#FFA500'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Column 'age' not found in the uploaded data.")
    
    elif graph_type == "Correlation Heatmap":
        numeric_data = analytics_with_pred.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            corr_matrix = numeric_data.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title='Correlation Matrix Heatmap',
                           color_continuous_scale='Plasma')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric data available for correlation heatmap.")
    
    st.markdown("<div class='section-header'>Advanced Insights</div>", unsafe_allow_html=True)
    
    cat_cols = [c for c in CATEGORICAL_FEATURES.get(ins_type, []) if c in analytics_with_pred.columns]
    if cat_cols:
        selected_cat_dist = st.selectbox("Select categorical column for distribution", cat_cols, key='cat_dist')
        cat_counts = analytics_with_pred[selected_cat_dist].value_counts().reset_index()
        cat_counts.columns = [selected_cat_dist, 'count']
        fig = px.bar(cat_counts, x=selected_cat_dist, y='count',
                    title=f'Distribution of {selected_cat_dist.replace("_", " ").title()}',
                    color='count', color_continuous_scale='Plasma')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns available for insights.")
        fig = px.histogram(analytics_with_pred, x='Predicted_Fraud_Probability', nbins=20,
                          title='Distribution of Predicted Fraud Probabilities',
                          color_discrete_sequence=['#FF4500'])
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# History Page (MODIFIED: Removed View Claim Details)
# ============================================
elif st.session_state.nav_page == "History":
    st.markdown("<h1 class='main-header'>Analysis History</h1>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No analysis history available.")
    else:
        for entry in reversed(st.session_state.history):
            st.markdown("<div class='history-item'>", unsafe_allow_html=True)
            st.write(f"**{entry['type']} Analysis** - {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            if entry['type'] == 'Batch Analysis':
                st.write(f"Data Source: {entry['data_source']} | Records: {entry['num_records']} | Fraud Count: {entry['fraud_count']} | Fraud Rate: {entry['fraud_rate']}")
                st.write(f"Insurance Type: {entry.get('insurance_type', 'Unknown')}")
            else:
                st.write(f"Risk Level: {entry['risk_level']} | Fraud Probability: {entry['fraud_probability']:.2%}")
                st.write(f"Insurance Type: {entry['insurance_type']}")
                if 'source' in entry:
                    st.write(f"Source: {entry['source'].title()} Data")
                if 'claim_id' in entry:
                    st.write(f"Claim ID: {entry['claim_id']}")
                # Removed the expander with "View Claim Details"
            st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# ============================================
# Settings Page (Removed Fraud Classification Threshold)
# ============================================
elif st.session_state.nav_page == "Settings":
    st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
    
    st.markdown("### Risk Thresholds")
    st.caption("Adjust the probability thresholds that define Low, Medium, and High risk levels.")
    
    col1, col2 = st.columns(2)
    with col1:
        new_low = st.number_input(
            "Low Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.low_risk_threshold,
            step=0.05,
            format="%.2f",
            help="Probabilities below this value are considered Low Risk and also Legitimate."
        )
    with col2:
        new_high = st.number_input(
            "High Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.high_risk_threshold,
            step=0.05,
            format="%.2f",
            help="Probabilities above this value are considered High Risk."
        )
    
    if new_low >= new_high:
        st.error("Low Risk Threshold must be less than High Risk Threshold. Please adjust.")
    else:
        st.session_state.low_risk_threshold = new_low
        st.session_state.high_risk_threshold = new_high
        # Update classification threshold to match low risk threshold
        sync_classification_threshold()
    
    st.markdown("---")
    st.markdown("### Prediction Behavior")
    
    allow_missing = st.checkbox(
        "Allow missing values in claims",
        value=st.session_state.allow_missing_values,
        help="If enabled, missing required columns will be filled with defaults (0 for numeric, -1 for categorical) and extra columns will be ignored."
    )
    st.session_state.allow_missing_values = allow_missing
    
    st.markdown("---")
    st.markdown("### MongoDB Connection Test")
    if st.button("Test MongoDB Connection", type="secondary"):
        try:
            client.admin.command('ping')
            test_doc = {"test": "connection", "timestamp": datetime.now()}
            result = analysis_collection.insert_one(test_doc)
            st.success(f"✅ Connection successful! Test document inserted with ID: {result.inserted_id}")
        except Exception as e:
            st.error(f"❌ MongoDB error: {e}")
    
    st.info("Changes are saved automatically and applied across all pages.")
    
    st.markdown("### Current Risk & Fraud Zones")
    st.write(f"**Fraud Classification:** Any claim with probability **> {st.session_state.low_risk_threshold:.0%}** is marked as **FRAUD** (Low Risk Threshold = cut-off).")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, st.session_state.low_risk_threshold, st.session_state.high_risk_threshold, 1],
        y=[1, 1, 1, 1],
        mode='lines+markers',
        line=dict(width=0),
        marker=dict(size=20, color=['green', 'yellow', 'red'], symbol='square'),
        text=['Legitimate / Low', 'Medium', 'High'],
        hoverinfo='text'
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Fraud Probability"),
        yaxis=dict(showticklabels=False, showgrid=False, range=[0, 2]),
        plot_bgcolor='#1e2a3a',
        paper_bgcolor='#1e2a3a',
        font=dict(color='white'),
        height=150
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# Admin Page: Manage Employees (with Activity)
# ============================================
elif st.session_state.nav_page == "Manage Employees" and st.session_state.user_type == "admin":
    st.markdown("<h1 class='main-header'>👥 Manage Employees</h1>", unsafe_allow_html=True)
    
    st.markdown("### Current Employees")
    if st.session_state.employee_db:
        df_emp = pd.DataFrame.from_dict(st.session_state.employee_db, orient='index')
        df_emp.reset_index(inplace=True)
        df_emp.rename(columns={'index': 'Employee ID'}, inplace=True)
        st.dataframe(df_emp, use_container_width=True)
    else:
        st.info("No employees found.")
    
    st.markdown("---")
    st.markdown("### Add New Employee")
    with st.form("add_employee_form"):
        new_id = st.text_input("Employee ID")
        new_name = st.text_input("Full Name")
        new_dept = st.text_input("Department")
        new_email = st.text_input("Email Address")
        new_pass = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Add Employee")
        if submitted:
            if not new_id or not new_name or not new_dept or not new_email or not new_pass:
                st.error("All fields are required.")
            elif new_id in st.session_state.employee_db:
                st.error(f"Employee ID {new_id} already exists.")
            else:
                employee_data = {
                    "name": new_name,
                    "department": new_dept,
                    "registered_email": new_email,
                    "password": new_pass
                }
                save_employee_to_mongo(new_id, employee_data)
                st.session_state.employee_db[new_id] = employee_data
                st.success(f"Employee {new_id} added successfully.")
                st.rerun()
    
    st.markdown("---")
    st.markdown("### Remove Employee")
    remove_id = st.selectbox("Select Employee ID to remove", options=list(st.session_state.employee_db.keys()))
    if st.button("Remove Employee", type="secondary"):
        if remove_id in st.session_state.employee_db:
            delete_employee_from_mongo(remove_id)
            del st.session_state.employee_db[remove_id]
            st.success(f"Employee {remove_id} removed.")
            st.rerun()
        else:
            st.error("Employee not found.")
    
    st.markdown("---")
    st.markdown("### 📊 Employee Activity Overview")
    st.caption("View the analysis and claim status updates performed by each employee.")
    
    all_analyses = list(analysis_collection.find({}))
    all_status_updates = list(claims_status_collection.find({}))
    
    employee_activity = []
    for emp_id, emp_data in st.session_state.employee_db.items():
        emp_name = emp_data.get('name', emp_id)
        analyses = [doc for doc in all_analyses if doc.get('employee_id') == emp_id]
        analyses_count = len(analyses)
        status_updates = [doc for doc in all_status_updates if doc.get('employee_id') == emp_id]
        status_count = len(status_updates)
        all_timestamps = []
        for doc in analyses:
            ts = doc.get('timestamp')
            if ts:
                all_timestamps.append(ts)
        for doc in status_updates:
            ts = doc.get('updated_at')
            if ts:
                all_timestamps.append(ts)
        latest_activity = max(all_timestamps) if all_timestamps else None
        employee_activity.append({
            'Employee ID': emp_id,
            'Name': emp_name,
            'Department': emp_data.get('department', ''),
            'Analyses Performed': analyses_count,
            'Status Updates': status_count,
            'Latest Activity': latest_activity.strftime('%Y-%m-%d %H:%M:%S') if latest_activity else 'Never'
        })
    
    if employee_activity:
        activity_df = pd.DataFrame(employee_activity)
        st.dataframe(activity_df, use_container_width=True, hide_index=True)
        
        selected_emp = st.selectbox("Select an employee to view detailed activity", 
                                     options=[emp['Employee ID'] for emp in employee_activity])
        if selected_emp:
            st.markdown(f"#### Detailed Activity for {selected_emp}")
            emp_analyses = [doc for doc in all_analyses if doc.get('employee_id') == selected_emp]
            if emp_analyses:
                st.markdown("**Fraud Analyses**")
                analyses_df = pd.DataFrame(emp_analyses)
                display_cols = ['timestamp', 'insurance_type', 'claim_id', 'fraud_probability', 'prediction']
                analyses_df = analyses_df[[c for c in display_cols if c in analyses_df.columns]]
                if 'timestamp' in analyses_df.columns:
                    analyses_df['timestamp'] = pd.to_datetime(analyses_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(analyses_df, use_container_width=True)
            else:
                st.info("No fraud analyses performed by this employee.")
            
            emp_updates = [doc for doc in all_status_updates if doc.get('employee_id') == selected_emp]
            if emp_updates:
                st.markdown("**Claim Status Updates**")
                updates_df = pd.DataFrame(emp_updates)
                display_cols = ['updated_at', 'claim_id', 'insurance_type', 'status', 'fraud_probability']
                updates_df = updates_df[[c for c in display_cols if c in updates_df.columns]]
                if 'updated_at' in updates_df.columns:
                    updates_df['updated_at'] = pd.to_datetime(updates_df['updated_at']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(updates_df, use_container_width=True)
            else:
                st.info("No claim status updates made by this employee.")
    else:
        st.info("No activity records found. Employees need to perform analyses or update claim statuses to see activity here.")
    
    st.markdown("---")
    st.caption("Changes are saved to MongoDB (collection 'admin'). Employee activity is derived from the 'fraud' and 'claims status' collections.")

# ============================================
# Footer
# ============================================
st.markdown("---")
st.caption(f"Insurance Fraud Detection System | Logged in as: {st.session_state.current_user['name']} | Built with Streamlit and Machine Learning")