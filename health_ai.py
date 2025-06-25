# Health AI - Complete Streamlit Application
# Run with: streamlit run health_ai.py

import streamlit as st
import sqlite3
import altair as alt
import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any
import random
import langchain
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
import hashlib
import concurrent.futures

# Configuration
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
WATSONX_APIKEY = "v2pjUHPAm5HfLBGpVN5T1-DHjM3tbAVIzyQmEUhMS_0v"
WATSONX_PROJECT_ID = "443a8ed4-48fd-4630-ab81-e5a5480dd375"
WATSONX_MODEL_ID = "ibm/granite-3-2-8b-instruct"

# CSS Styling
def load_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-title {
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        color: white;
        text-align: center;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: white;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .treatment-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: white;
    }
    
    .health-tip {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Database Management
class DatabaseManager:
    def __init__(self, db_path="health_ai.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                contact TEXT,
                medical_history TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Chat history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                message TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)
        
        # Health metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                metric_type TEXT,
                value REAL,
                unit TEXT,
                recorded_date DATE,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                symptoms TEXT,
                prediction TEXT,
                confidence REAL,
                recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'patient', -- 'admin' or 'patient'
                admin_id INTEGER, -- For patients, references admin's user id
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (admin_id) REFERENCES users (id)
            )
        """)
        
        # Admin-patient chat table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_patient_chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id INTEGER,
                patient_id INTEGER,
                sender_role TEXT, -- 'admin' or 'patient'
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (admin_id) REFERENCES users (id),
                FOREIGN KEY (patient_id) REFERENCES users (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_patient(self, name, age, gender, contact, medical_history=""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patients (name, age, gender, contact, medical_history)
            VALUES (?, ?, ?, ?, ?)
        """, (name, age, gender, contact, medical_history))
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return patient_id
    
    def get_all_patients(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM patients ORDER BY created_at DESC", conn)
        conn.close()
        return df
    
    def add_chat_message(self, patient_id, message, response):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (patient_id, message, response)
            VALUES (?, ?, ?)
        """, (patient_id, message, response))
        conn.commit()
        conn.close()
    
    def get_chat_history(self, patient_id):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT message, response, timestamp 
            FROM chat_history 
            WHERE patient_id = ? 
            ORDER BY timestamp ASC
        """, conn, params=(patient_id,))
        conn.close()
        return df
    
    def add_health_metric(self, patient_id, metric_type, value, unit, recorded_date):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO health_metrics (patient_id, metric_type, value, unit, recorded_date)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_id, metric_type, value, unit, recorded_date))
        conn.commit()
        conn.close()
    
    def get_health_metrics(self, patient_id):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM health_metrics 
            WHERE patient_id = ? 
            ORDER BY recorded_date DESC
        """, conn, params=(patient_id,))
        conn.close()
        return df
    
    def add_prediction(self, patient_id, symptoms, prediction, confidence, recommendations):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (patient_id, symptoms, prediction, confidence, recommendations)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_id, symptoms, prediction, confidence, recommendations))
        conn.commit()
        conn.close()
    
    def get_predictions(self, patient_id):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM predictions 
            WHERE patient_id = ? 
            ORDER BY created_at DESC
        """, conn, params=(patient_id,))
        conn.close()
        return df
    
    def add_user(self, username, password, role='patient', admin_id=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        try:
            cursor.execute("""
                INSERT INTO users (username, password_hash, role, admin_id)
                VALUES (?, ?, ?, ?)
            """, (username, password_hash, role, admin_id))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def validate_user(self, username, password):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute("""
            SELECT * FROM users WHERE username = ? AND password_hash = ?
        """, (username, password_hash))
        user = cursor.fetchone()
        conn.close()
        return user is not None

    def get_user(self, username):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        return user

    def get_admins(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE role = 'admin'")
        admins = cursor.fetchall()
        conn.close()
        return admins

    def get_patients_for_admin(self, admin_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE role = 'patient' AND admin_id = ?", (admin_id,))
        patients = cursor.fetchall()
        conn.close()
        return patients

    def add_admin_patient_message(self, admin_id, patient_id, sender_role, message):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO admin_patient_chat (admin_id, patient_id, sender_role, message)
            VALUES (?, ?, ?, ?)
        """, (admin_id, patient_id, sender_role, message))
        conn.commit()
        conn.close()

    def get_admin_patient_chat(self, admin_id, patient_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sender_role, message, timestamp FROM admin_patient_chat
            WHERE admin_id = ? AND patient_id = ?
            ORDER BY timestamp ASC
        """, (admin_id, patient_id))
        chat = cursor.fetchall()
        conn.close()
        return chat

# AI Integration Class
class HealthAI:
    def __init__(self):
        try:
            self.llm = WatsonxLLM(
                model_id=WATSONX_MODEL_ID,
                url=WATSONX_URL,
                apikey=WATSONX_APIKEY,
                project_id=WATSONX_PROJECT_ID,
                params={
                    "decoding_method": "greedy",
                    "max_new_tokens": 500,
                    "temperature": 0.7
                }
            )
            self.chat_template = self._create_chat_template()
            self.prediction_template = self._create_prediction_template()
            self.treatment_template = self._create_treatment_template()
        except Exception as e:
            st.error(f"Error initializing AI: {str(e)}")
            self.llm = None
    
    def _create_chat_template(self):
        return PromptTemplate(
            input_variables=["user_question", "chat_history"],
            template="""You are a Health AI assistant providing medical information and guidance. 
            
IMPORTANT: You are not a replacement for professional medical advice. Always recommend consulting healthcare professionals for serious concerns.

Previous conversation context:
{chat_history}

Current question: {user_question}

Provide helpful, accurate health information while being empathetic and clear. Include relevant health tips and recommendations for a healthy lifestyle.

Response:"""
        )
    
    def _create_prediction_template(self):
        return PromptTemplate(
            input_variables=["symptoms", "patient_info"],
            template="""You are a medical AI assistant analyzing symptoms to provide potential health insights.

Patient Information: {patient_info}
Reported Symptoms: {symptoms}

Based on the symptoms provided, analyze and provide:
1. Most likely conditions (with confidence levels)
2. Recommended immediate actions
3. When to seek medical attention
4. Lifestyle recommendations

IMPORTANT: This is for informational purposes only. Always recommend consulting a healthcare professional for proper diagnosis.

Please format your response as JSON with the following structure:
{{
    "primary_conditions": [
        {{"condition": "condition_name", "confidence": 0.8, "description": "brief_description"}}
    ],
    "recommendations": ["recommendation1", "recommendation2"],
    "urgency_level": "low/medium/high",
    "next_steps": "detailed_next_steps"
}}

Response:"""
        )
    
    def _create_treatment_template(self):
        return PromptTemplate(
            input_variables=["condition", "patient_info"],
            template="""You are a medical AI assistant providing treatment plan guidance.

Patient Information: {patient_info}
Diagnosed/Suspected Condition: {condition}

Create a comprehensive treatment plan including:
1. General treatment approaches
2. Lifestyle modifications
3. Dietary recommendations
4. Exercise guidelines
5. Follow-up recommendations

IMPORTANT: This is educational information only. Professional medical supervision is required for all treatments.

Please format your response as JSON:
{{
    "treatment_plan": {{
        "medications": ["general_medication_types"],
        "lifestyle_changes": ["change1", "change2"],
        "diet_recommendations": ["diet1", "diet2"],
        "exercise_plan": "exercise_guidelines",
        "follow_up": "follow_up_schedule",
        "precautions": ["precaution1", "precaution2"]
    }}
}}

Response:"""
        )
    
    def generate_chat_response(self, message: str, chat_history: str = "") -> str:
        if not self.llm:
            return "AI service is currently unavailable. Please try again later."
        
        try:
            prompt = self.chat_template.format(
                user_question=message,
                chat_history=chat_history
            )
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Sorry, there was an error processing your request: {str(e)}"
    
    def predict_condition(self, symptoms: str, patient_info: str) -> Dict:
        if not self.llm:
            return {"error": "AI service unavailable"}
        
        try:
            prompt = self.prediction_template.format(
                symptoms=symptoms,
                patient_info=patient_info
            )
            response = self.llm(prompt)
            
            # Try to parse JSON response
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                # If JSON parsing fails, return structured response
                return {
                    "primary_conditions": [
                        {"condition": "Analysis provided", "confidence": 0.7, "description": response}
                    ],
                    "recommendations": ["Consult healthcare professional"],
                    "urgency_level": "medium",
                    "next_steps": "Please consult with a healthcare provider for proper diagnosis"
                }
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def generate_treatment_plan(self, condition: str, patient_info: str) -> Dict:
        if not self.llm:
            return {"error": "AI service unavailable"}
        
        try:
            prompt = self.treatment_template.format(
                condition=condition,
                patient_info=patient_info
            )
            response = self.llm(prompt)
            
            # Try to parse JSON response
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                # If JSON parsing fails, return structured response
                return {
                    "treatment_plan": {
                        "medications": ["Consult doctor for prescription"],
                        "lifestyle_changes": ["Follow healthcare provider guidance"],
                        "diet_recommendations": ["Maintain balanced diet"],
                        "exercise_plan": "Light exercise as tolerated",
                        "follow_up": "Regular check-ups with healthcare provider",
                        "precautions": ["Monitor symptoms", "Seek immediate care if symptoms worsen"]
                    }
                }
        except Exception as e:
            return {"error": f"Treatment plan error: {str(e)}"}

# Data Generation Utilities
def generate_sample_health_data(patient_id: int, days: int = 30):
    """Generate sample health metrics for demonstration"""
    db = DatabaseManager()
    
    metrics = [
        {"type": "blood_pressure_systolic", "unit": "mmHg", "range": (110, 140)},
        {"type": "blood_pressure_diastolic", "unit": "mmHg", "range": (70, 90)},
        {"type": "heart_rate", "unit": "bpm", "range": (60, 100)},
        {"type": "weight", "unit": "kg", "range": (60, 90)},
        {"type": "blood_sugar", "unit": "mg/dL", "range": (80, 120)},
        {"type": "temperature", "unit": "°F", "range": (97, 99)}
    ]
    
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        for metric in metrics:
            value = round(random.uniform(metric["range"][0], metric["range"][1]), 1)
            db.add_health_metric(patient_id, metric["type"], value, metric["unit"], date)

# Streamlit App
def main():
    st.set_page_config(
        page_title="Health AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()

    # Initialize database and AI
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()

    if 'ai' not in st.session_state:
        st.session_state.ai = HealthAI()

    # --- LOGIN CHECK ---
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        login_signup_page()
        return

    # --- Show admin ID in sidebar if admin ---
    if st.session_state.get("user_role") == "admin":
        st.sidebar.markdown(
            f"""
            <div style="background:#f3e5f5;padding:1em;border-radius:8px;margin-bottom:1em;">
                <strong>Your Admin ID:</strong> <span style="color:#6a1b9a;font-size:1.2em;">{st.session_state.get("user_id")}</span><br>
                <small>Share this ID with patients for signup.</small>
            </div>
            """, unsafe_allow_html=True
        )

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🏥 Health AI</h1>
        <p class="main-subtitle">Your Intelligent Healthcare Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "👤 Patient Management", "💬 Chat Assistant", 
         "🔮 Disease Prediction", "📋 Treatment Plans", "📊 Health Analytics", "📑 Report Analysis", "🗨️ Admin-Patient Chat"]
    )
    
    # Page Routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "👤 Patient Management":
        show_patient_management()
    elif page == "💬 Chat Assistant":
        show_chat_assistant()
    elif page == "🔮 Disease Prediction":
        show_disease_prediction()
    elif page == "📋 Treatment Plans":
        show_treatment_plans()
    elif page == "📊 Health Analytics":
        show_health_analytics()
    elif page == "📑 Report Analysis":
        show_report_analysis()
    elif page == "🗨️ Admin-Patient Chat":
        show_admin_patient_chat()

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

def show_home_page():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 Welcome to Health AI</h3>
            <p>Your comprehensive healthcare assistant powered by IBM Granite AI. Get personalized health insights, chat with AI, predict potential conditions, and manage your health data all in one place.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>💬 AI Chat Assistant</h4>
            <p>Ask health questions and get intelligent responses</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🔮 Disease Prediction</h4>
            <p>Analyze symptoms to predict potential conditions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📋 Treatment Plans</h4>
            <p>Generate personalized treatment recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Health Analytics</h4>
            <p>Visualize and track your health metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="health-tip">
            <h4>💡 Daily Health Tip</h4>
            <p>Stay hydrated! Aim for 8 glasses of water daily to maintain optimal health and energy levels.</p>
        </div>
        """, unsafe_allow_html=True)

def show_patient_management():
    st.header("👤 Patient Management")
    
    tab1, tab2 = st.tabs(["Add New Patient", "View Patients"])
    
    with tab1:
        st.subheader("Add New Patient")
        
        with st.form("add_patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name*")
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                contact = st.text_input("Contact Number")
            
            medical_history = st.text_area("Medical History (Optional)")
            
            submitted = st.form_submit_button("Add Patient")
            
            if submitted and name:
                patient_id = st.session_state.db.add_patient(name, age, gender, contact, medical_history)
                st.success(f"Patient '{name}' added successfully! ID: {patient_id}")
                
                # Generate sample data for demo
                if st.checkbox("Generate sample health data for demo"):
                    generate_sample_health_data(patient_id)
                    st.info("Sample health data generated for the last 30 days")
    
    with tab2:
        st.subheader("All Patients")
        patients_df = st.session_state.db.get_all_patients()
        
        if not patients_df.empty:
            st.dataframe(patients_df, use_container_width=True)
        else:
            st.info("No patients found. Add a patient to get started!")

def show_chat_assistant():
    if 'ai' not in st.session_state or st.session_state.ai is None:
        st.session_state.ai = HealthAI()
        
    st.header("💬 Health Chat Assistant")
    
    # Patient selection
    patients_df = st.session_state.db.get_all_patients()
    
    if patients_df.empty:
        st.warning("Please add a patient first in the Patient Management section.")
        return
    
    patient_options = {f"{row['name']} (ID: {row['id']})": row['id'] 
                      for _, row in patients_df.iterrows()}
    
    selected_patient = st.selectbox("Select Patient", list(patient_options.keys()))
    patient_id = patient_options[selected_patient]
    
    # Chat interface
    st.subheader(f"Chat with {selected_patient.split(' (')[0]}")
    
    # Display chat history
    chat_history = st.session_state.db.get_chat_history(patient_id)
    
    chat_container = st.container()
    with chat_container:
        if not chat_history.empty:
            for _, row in chat_history.iterrows():
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>Patient:</strong> {row['message']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>Health AI:</strong> {row['response']}
                </div>
                """, unsafe_allow_html=True)
    
    # Message input
    user_message = st.text_input("Ask a health question:", key="chat_input")
    
    if st.button("Send") and user_message:
        with st.spinner("AI is thinking..."):
            # Get chat context
            chat_context = ""
            if not chat_history.empty:
                recent_chats = chat_history.tail(3)
                chat_context = "\n".join([f"User: {row['message']}\nAI: {row['response']}" 
                                        for _, row in recent_chats.iterrows()])
            
            # Generate AI response
            ai_response = st.session_state.ai.generate_chat_response(user_message, chat_context)
            
            # Save to database
            st.session_state.db.add_chat_message(patient_id, user_message, ai_response)
            
            # Refresh the page to show new message
            st.rerun()

def show_disease_prediction():
    if 'ai' not in st.session_state or st.session_state.ai is None:
        st.session_state.ai = HealthAI()
        
    st.header("🔮 Disease Prediction System")
    
    # Patient selection
    patients_df = st.session_state.db.get_all_patients()
    
    if patients_df.empty:
        st.warning("Please add a patient first in the Patient Management section.")
        return
    
    patient_options = {f"{row['name']} (ID: {row['id']})": row['id'] 
                      for _, row in patients_df.iterrows()}
    
    selected_patient = st.selectbox("Select Patient", list(patient_options.keys()))
    patient_id = patient_options[selected_patient]
    
    # Get patient info
    patient_info = patients_df[patients_df['id'] == patient_id].iloc[0]
    patient_details = f"Age: {patient_info['age']}, Gender: {patient_info['gender']}, Medical History: {patient_info['medical_history']}"
    
    st.subheader("Symptom Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Symptom input
        symptoms = st.text_area(
            "Describe symptoms in detail:",
            placeholder="e.g., Fever for 3 days, headache, body aches, fatigue...",
            height=100
        )
        
        if st.button("Analyze Symptoms", type="primary"):
            if symptoms:
                with st.spinner("Analyzing symptoms..."):
                    prediction_result = st.session_state.ai.predict_condition(symptoms, patient_details)
                    
                    if "error" not in prediction_result:
                        # Save prediction
                        st.session_state.db.add_prediction(
                            patient_id, symptoms, 
                            str(prediction_result.get("primary_conditions", [])),
                            prediction_result.get("urgency_level", "medium"),
                            str(prediction_result.get("recommendations", []))
                        )
                        
                        # Display results
                        st.success("Analysis Complete!")
                        
                        # Primary conditions
                        if "primary_conditions" in prediction_result:
                            st.subheader("Potential Conditions")
                            for condition in prediction_result["primary_conditions"]:
                                confidence = condition.get("confidence", 0) * 100
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <h4>{condition.get('condition', 'Unknown')}</h4>
                                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                    <p>{condition.get('description', '')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Recommendations
                        if "recommendations" in prediction_result:
                            st.subheader("Recommendations")
                            render_ai_response_lines("\n".join(prediction_result["recommendations"]))
                        
                        # Urgency level
                        urgency = prediction_result.get("urgency_level", "medium")
                        urgency_color = {"low": "green", "medium": "orange", "high": "red"}
                        st.markdown(f"""
                        <div style="background-color: {urgency_color.get(urgency, 'gray')}; 
                                   color: white; padding: 1rem; border-radius: 5px; text-align: center;">
                            <strong>Urgency Level: {urgency.upper()}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Next steps
                        if "next_steps" in prediction_result:
                            st.subheader("Next Steps")
                            render_ai_response_lines(str(prediction_result["next_steps"]))
                    else:
                        st.error(prediction_result["error"])
            else:
                st.warning("Please describe your symptoms.")
    
    with col2:
        st.subheader("Patient Info")
        st.write(f"**Name:** {patient_info['name']}")
        st.write(f"**Age:** {patient_info['age']}")
        st.write(f"**Gender:** {patient_info['gender']}")
        if patient_info['medical_history']:
            st.write(f"**Medical History:** {patient_info['medical_history']}")
    
    # Previous predictions
    st.subheader("Previous Predictions")
    predictions_df = st.session_state.db.get_predictions(patient_id)
    
    if not predictions_df.empty:
        for _, pred in predictions_df.iterrows():
            with st.expander(f"Prediction from {pred['created_at'][:10]}"):
                st.write(f"**Symptoms:** {pred['symptoms']}")
                st.write(f"**Confidence:** {pred['confidence']}")
                st.write(f"**Recommendations:** {pred['recommendations']}")
    else:
        st.info("No previous predictions found.")

def show_treatment_plans():
    if 'ai' not in st.session_state or st.session_state.ai is None:
        st.session_state.ai = HealthAI()
        
    st.header("📋 Treatment Plan Generator")
    
    # Patient selection
    patients_df = st.session_state.db.get_all_patients()
    
    if patients_df.empty:
        st.warning("Please add a patient first in the Patient Management section.")
        return
    
    patient_options = {f"{row['name']} (ID: {row['id']})": row['id'] 
                      for _, row in patients_df.iterrows()}
    
    selected_patient = st.selectbox("Select Patient", list(patient_options.keys()))
    patient_id = patient_options[selected_patient]
    
    # Get patient info
    patient_info = patients_df[patients_df['id'] == patient_id].iloc[0]
    patient_details = f"Age: {patient_info['age']}, Gender: {patient_info['gender']}, Medical History: {patient_info['medical_history']}"
    
    st.subheader("Generate Treatment Plan")
    
    condition = st.text_input(
        "Enter condition/diagnosis:",
        placeholder="e.g., Common Cold, Hypertension, Diabetes Type 2..."
    )
    
    if st.button("Generate Treatment Plan", type="primary"):
        if condition:
            with st.spinner("Generating treatment plan..."):
                treatment_plan = st.session_state.ai.generate_treatment_plan(condition, patient_details)
                
                if "error" not in treatment_plan and "treatment_plan" in treatment_plan:
                    plan = treatment_plan["treatment_plan"]
                    
                    st.success("Treatment Plan Generated!")
                    
                    # Medications
                    if "medications" in plan:
                        st.markdown("""
                        <div class="treatment-section">
                            <h4>💊 Medications</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for med in plan["medications"]:
                            st.write(f"• {med}")
                    
                    # Lifestyle changes
                    if "lifestyle_changes" in plan:
                        st.markdown("""
                        <div class="treatment-section">
                            <h4>🏃‍♂️ Lifestyle Changes</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for change in plan["lifestyle_changes"]:
                            st.write(f"• {change}")
                    
                    # Diet recommendations
                    if "diet_recommendations" in plan:
                        st.markdown("""
                        <div class="treatment-section">
                            <h4>🥗 Diet Recommendations</h4>
                                    </div>
                        """, unsafe_allow_html=True)
                        for diet in plan["diet_recommendations"]:
                            st.write(f"• {diet}")
                    
                    # Exercise plan
                    if "exercise_plan" in plan:
                        st.markdown("""
                        <div class="treatment-section">
                            <h4>🏋️‍♀️ Exercise Plan</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        render_ai_response_lines(str(plan["exercise_plan"]))
                    
                    # Follow-up
                    if "follow_up" in plan:
                        st.markdown("""
                        <div class="treatment-section">
                            <h4>📅 Follow-up Schedule</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        render_ai_response_lines(str(plan["follow_up"]))
                    
                    # Precautions
                    if "precautions" in plan:
                        st.markdown("""
                        <div class="treatment-section">
                            <h4>⚠️ Precautions</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for precaution in plan["precautions"]:
                            st.write(f"• {precaution}")
                    
                    # Disclaimer
                    st.warning("⚠️ This treatment plan is for informational purposes only. Please consult with a qualified healthcare professional before starting any treatment.")
                    
                else:
                    st.error("Unable to generate treatment plan. Please try again.")
        else:
            st.warning("Please enter a condition or diagnosis.")

def show_health_analytics():
    if 'ai' not in st.session_state or st.session_state.ai is None:
        st.session_state.ai = HealthAI()
        
    st.header("📊 Health Analytics Dashboard")
    
    # Patient selection
    patients_df = st.session_state.db.get_all_patients()
    
    if patients_df.empty:
        st.warning("Please add a patient first in the Patient Management section.")
        return
    
    patient_options = {f"{row['name']} (ID: {row['id']})": row['id'] 
                      for _, row in patients_df.iterrows()}
    
    selected_patient = st.selectbox("Select Patient", list(patient_options.keys()))
    patient_id = patient_options[selected_patient]
    
    # Get health metrics
    metrics_df = st.session_state.db.get_health_metrics(patient_id)
    
    if metrics_df.empty:
        st.info("No health data found for this patient. Generate sample data in Patient Management.")
        return
    
    # Convert recorded_date to datetime
    metrics_df['recorded_date'] = pd.to_datetime(metrics_df['recorded_date'])
    
    # Metrics Overview
    st.subheader("Health Metrics Overview")
    
    # Get latest metrics for summary
    latest_metrics = metrics_df.groupby('metric_type').apply(
        lambda x: x.loc[x['recorded_date'].idxmax()]
    ).reset_index(drop=True)
    
    # Display metric cards
    metric_cols = st.columns(3)
    for i, (_, metric) in enumerate(latest_metrics.iterrows()):
        if i < 6:  # Show first 6 metrics
            with metric_cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{metric['metric_type'].replace('_', ' ').title()}</h4>
                    <h2>{metric['value']} {metric['unit']}</h2>
                    <p>Last recorded: {metric['recorded_date'].strftime('%Y-%m-%d')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Time Series Charts
    st.subheader("Health Trends")
    
    # Blood Pressure Chart
    bp_data = metrics_df[metrics_df['metric_type'].isin(['blood_pressure_systolic', 'blood_pressure_diastolic'])]
    if not bp_data.empty:
        bp_chart = alt.Chart(bp_data).mark_line(point=True).encode(
            x=alt.X('recorded_date:T', title='Date'),
            y=alt.Y('value:Q', title='mmHg'),
            color=alt.Color('metric_type:N', title='Metric'),
            tooltip=['recorded_date', 'metric_type', 'value']
        ).properties(
            title='Blood Pressure Trends',
            width=600,
            height=300
        )
        st.altair_chart(bp_chart, use_container_width=True)
    
    # Heart Rate Chart
    hr_data = metrics_df[metrics_df['metric_type'] == 'heart_rate']
    if not hr_data.empty:
        hr_chart = alt.Chart(hr_data).mark_line(point=True, color='red').encode(
            x=alt.X('recorded_date:T', title='Date'),
            y=alt.Y('value:Q', title='BPM'),
            tooltip=['recorded_date', 'value']
        ).properties(
            title='Heart Rate Trends',
            width=600,
            height=300
        )
        st.altair_chart(hr_chart, use_container_width=True)
    
    # Weight and Blood Sugar Charts
    col1, col2 = st.columns(2)
    
    with col1:
        weight_data = metrics_df[metrics_df['metric_type'] == 'weight']
        if not weight_data.empty:
            weight_chart = alt.Chart(weight_data).mark_line(point=True, color='green').encode(
                x=alt.X('recorded_date:T', title='Date'),
                y=alt.Y('value:Q', title='kg'),
                tooltip=['recorded_date', 'value']
            ).properties(
                title='Weight Trends',
                width=300,
                height=250
            )
            st.altair_chart(weight_chart, use_container_width=True)
    
    with col2:
        bs_data = metrics_df[metrics_df['metric_type'] == 'blood_sugar']
        if not bs_data.empty:
            bs_chart = alt.Chart(bs_data).mark_line(point=True, color='orange').encode(
                x=alt.X('recorded_date:T', title='Date'),
                y=alt.Y('value:Q', title='mg/dL'),
                tooltip=['recorded_date', 'value']
            ).properties(
                title='Blood Sugar Trends',
                width=300,
                height=250
            )
            st.altair_chart(bs_chart, use_container_width=True)
    
    # Health Insights with AI
    st.subheader("AI Health Insights")
    
    if st.button("Generate Health Insights", type="primary"):
        with st.spinner("Analyzing health data..."):
            # Prepare health data summary for AI analysis
            health_summary = generate_health_summary(metrics_df)
            patient_info = patients_df[patients_df['id'] == patient_id].iloc[0]
            
            # Create insights prompt
            insights_prompt = f"""
            Analyze the following health data for a patient and provide insights:
            
            Patient: {patient_info['name']}, Age: {patient_info['age']}, Gender: {patient_info['gender']}
            
            Health Data Summary:
            {health_summary}
            
            Provide:
            1. Overall health assessment
            2. Areas of concern (if any)
            3. Positive trends
            4. Recommendations for improvement
            5. Suggested follow-up actions
            
            Format as clear, actionable insights for the patient.
            """
            
            try:
                insights = st.session_state.ai.llm(insights_prompt)
                st.markdown("""
                <div class="treatment-section">
                    <h4>🧠 AI Health Analysis</h4>
                """, unsafe_allow_html=True)
                render_ai_response_lines(insights)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Unable to generate insights: {str(e)}")
    
    # Add new health metric
    st.subheader("Add New Health Metric")
    
    with st.form("add_metric_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric_type = st.selectbox(
                "Metric Type",
                ["blood_pressure_systolic", "blood_pressure_diastolic", "heart_rate", 
                 "weight", "blood_sugar", "temperature", "other"]
            )
            
            if metric_type == "other":
                metric_type = st.text_input("Custom Metric Type")
        
        with col2:
            value = st.number_input("Value", min_value=0.0, step=0.1)
            unit = st.text_input("Unit", value="units")
        
        with col3:
            recorded_date = st.date_input("Date", value=datetime.now().date())
        
        submitted = st.form_submit_button("Add Metric")
        
        if submitted and metric_type and value:
            st.session_state.db.add_health_metric(
                patient_id, metric_type, value, unit, recorded_date.strftime('%Y-%m-%d')
            )
            st.success("Health metric added successfully!")
            st.rerun()

def generate_health_summary(metrics_df):
    """Generate a summary of health metrics for AI analysis"""
    summary = []
    
    for metric_type in metrics_df['metric_type'].unique():
        metric_data = metrics_df[metrics_df['metric_type'] == metric_type]
        latest_value = metric_data.iloc[-1]['value']
        avg_value = metric_data['value'].mean()
        trend = "increasing" if metric_data['value'].iloc[-1] > metric_data['value'].iloc[0] else "decreasing"
        
        summary.append(f"{metric_type}: Latest={latest_value}, Average={avg_value:.1f}, Trend={trend}")
    
    return "\n".join(summary)

# Enhanced AI Class Methods (Additional functionality)
def enhance_ai_responses():
    """Add additional AI response enhancement methods"""
    
    def get_health_recommendations(self, patient_data: Dict) -> str:
        """Generate personalized health recommendations"""
        prompt = f"""
        Based on the following patient data, provide personalized health recommendations:
        
        Patient Information: {patient_data}
        
        Provide specific, actionable recommendations for:
        1. Diet and nutrition
        2. Exercise and physical activity
        3. Lifestyle modifications
        4. Preventive care
        5. Risk factors to monitor
        
        Keep recommendations practical and achievable.
        """
        
        try:
            return self.llm(prompt)
        except Exception as e:
            return f"Unable to generate recommendations: {str(e)}"
    
    def analyze_symptoms_detailed(self, symptoms: List[str], duration: str, severity: str) -> Dict:
        """Enhanced symptom analysis with duration and severity"""
        prompt = f"""
        Analyze the following symptoms with additional context:
        
        Symptoms: {', '.join(symptoms)}
        Duration: {duration}
        Severity: {severity}
        
        Provide a detailed analysis including:
        1. Most likely conditions with confidence scores
        2. Red flags that require immediate attention
        3. Home care recommendations
        4. When to seek medical care
        5. Questions for healthcare provider
        6. Any additional relevant information
        """
        
        try:
            response = self.llm(prompt)
            return {"analysis": response, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    # Add these methods to the HealthAI class
    HealthAI.get_health_recommendations = get_health_recommendations
    HealthAI.analyze_symptoms_detailed = analyze_symptoms_detailed

# Initialize enhanced AI functionality
enhance_ai_responses()

# Additional utility functions
def export_patient_data(patient_id: int) -> str:
    """Export patient data to JSON format"""
    db = DatabaseManager()
    
    # Get all patient data
    patients_df = db.get_all_patients()
    patient_info = patients_df[patients_df['id'] == patient_id].to_dict('records')[0]
    
    metrics_df = db.get_health_metrics(patient_id)
    chat_history = db.get_chat_history(patient_id)
    predictions = db.get_predictions(patient_id)
    
    export_data = {
        "patient_info": patient_info,
        "health_metrics": metrics_df.to_dict('records'),
        "chat_history": chat_history.to_dict('records'),
        "predictions": predictions.to_dict('records'),
        "export_date": datetime.now().isoformat()
    }
    
    return json.dumps(export_data, indent=2)

def calculate_health_score(metrics_df: pd.DataFrame) -> int:
    """Calculate overall health score based on metrics"""
    if metrics_df.empty:
        return 0
    
    score = 100
    latest_metrics = metrics_df.groupby('metric_type').last()
    
    # Check each metric against normal ranges
    ranges = {
        'blood_pressure_systolic': (90, 120),
        'blood_pressure_diastolic': (60, 80),
        'heart_rate': (60, 100),
        'blood_sugar': (70, 140),
        'temperature': (97, 99)
    }
    
    for metric_type, (min_val, max_val) in ranges.items():
        if metric_type in latest_metrics.index:
            value = latest_metrics.loc[metric_type, 'value']
            if value < min_val or value > max_val:
                score -= 10
    
    return max(0, score)

# Add footer with additional information
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Health AI</strong> - Powered by IBM Granite AI</p>
        <p>⚠️ This application is for informational purposes only. Always consult healthcare professionals for medical advice.</p>
        <p>© 2025 Health AI. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

def make_links_clickable(text):
    import re
    # Convert plain URLs to clickable links
    url_pattern = r'(https?://[^\s]+)'
    return re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)

def render_ai_response_lines(ai_text):
    """Render AI response line by line as bullet points."""
    for line in str(ai_text).splitlines():
        if line.strip():
            st.markdown(f"- {line.strip()}")

def show_report_analysis():
    if 'ai' not in st.session_state or st.session_state.ai is None:
        st.session_state.ai = HealthAI()

    # Check if AI is properly initialized
    if not hasattr(st.session_state.ai, "llm") or st.session_state.ai.llm is None:
        st.error("AI service is currently unavailable. Please check your AI configuration or try again later.")
        return

    st.header("📑 Health Report Upload & AI Analysis")

    # Patient selection
    patients_df = st.session_state.db.get_all_patients()
    if patients_df.empty:
        st.warning("Please add a patient first in the Patient Management section.")
        return

    patient_options = {f"{row['name']} (ID: {row['id']})": row['id'] 
                      for _, row in patients_df.iterrows()}
    selected_patient = st.selectbox("Select Patient", list(patient_options.keys()))
    patient_id = patient_options[selected_patient]

    st.subheader("Upload Health Report")
    uploaded_file = st.file_uploader("Upload a health report (PDF, image, or text)", type=["pdf", "txt", "png", "jpg", "jpeg"])

    if uploaded_file:
        # Extract text from file
        import io
        report_text = ""
        if uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    report_text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Could not read PDF: {e}")
        elif uploaded_file.type.startswith("image/"):
            try:
                import pytesseract
                from PIL import Image
                image = Image.open(uploaded_file)
                report_text = pytesseract.image_to_string(image)
            except Exception as e:
                st.error(f"Could not read image: {e}")
        else:
            report_text = uploaded_file.read().decode("utf-8")

        # Limit report_text length to avoid LLM timeout
        max_chars = 10000
        if len(report_text) > max_chars:
            st.warning(f"Report is too long. Only the first {max_chars} characters will be analyzed for faster response.")
            report_text = report_text[:max_chars]

        if report_text.strip():
            st.subheader("Extracted Report Text")
            st.text_area("Report Content", report_text, height=200)

            if st.button("Analyze Report with AI"):
                with st.spinner("Analyzing report (may take up to 60 seconds)..."):
                    prompt = f"""
                    Analyze the following health report for the patient and provide:
                    1. Key findings and summary
                    2. Any abnormal values or concerns

                    Report:
                    {report_text}
                    """
                    ai = st.session_state.ai  # <-- Get AI object in main thread
                    def ai_call():
                        return ai.llm(prompt)
                    ai_result = None
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(ai_call)
                            ai_result = future.result(timeout=90)
                        st.markdown("""
                        <div class="treatment-section">
                            <h4>🧠 AI Report Analysis</h4>
                        """, unsafe_allow_html=True)
                        render_ai_response_lines(ai_result)
                        st.markdown("</div>", unsafe_allow_html=True)
                        # Optional: Try to extract tabular data and plot
                        import re
                        import pandas as pd
                        table_match = re.search(r"((?:[A-ZaZ0-9_ ]+\t?)+\n(?:[A-ZaZ0-9_.\- ]+\t?)+\n(?:.+\n)+)", ai_result)
                        if table_match:
                            table_text = table_match.group(1)
                            df = pd.read_csv(io.StringIO(table_text), sep="\t")
                            st.subheader("Extracted Data Chart")
                            st.line_chart(df)
                    except concurrent.futures.TimeoutError:
                        st.error("AI analysis timed out. Please try with a shorter report or try again later.")
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
        else:
            st.warning("No text could be extracted from the uploaded file.")

def login_signup_page():
    st.title("🔐 Login or Sign Up")
    option = st.radio("Select option", ["Login", "Sign Up"])
    db = st.session_state.db

    if option == "Sign Up":
        role = st.selectbox("Sign up as", ["patient", "admin"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        admin_id = None
        if role == "patient":
            admins = db.get_admins()
            admin_options = ["None"] + [f"{a[1]} (ID: {a[0]})" for a in admins]
            selected_admin = st.selectbox("Assign to Admin (optional)", admin_options)
            if selected_admin != "None":
                admin_id = int(selected_admin.split("ID: ")[1].replace(")", ""))
        if st.button("Sign Up"):
            if username and password:
                if db.add_user(username, password, role, admin_id):
                    st.success("Account created! Please log in.")
                else:
                    st.error("Username already exists.")
            else:
                st.warning("Please enter all required fields.")
    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username and password:
                if db.validate_user(username, password):
                    user = db.get_user(username)
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_role = user[3]  # role
                    st.session_state.user_id = user[0]    # id
                    st.success("Login successful!")
                    st.rerun()  # <-- use st.rerun() instead of st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")
            else:
                st.warning("Please enter both username and password.")

def show_admin_patient_chat():
    db = st.session_state.db
    user_role = st.session_state.user_role
    user_id = st.session_state.user_id

    st.header("🗨️ Admin-Patient Chat")

    if user_role == "admin":
        patients = db.get_patients_for_admin(user_id)
        if not patients:
            st.info("No patients assigned to you yet.")
            return
        patient_options = [f"{p[1]} (ID: {p[0]})" for p in patients]
        selected_patient = st.selectbox("Select Patient", patient_options)
        patient_id = int(selected_patient.split("ID: ")[1].replace(")", ""))
        chat = db.get_admin_patient_chat(user_id, patient_id)
        st.subheader(f"Chat with Patient: {selected_patient}")
        for sender, msg, ts in chat:
            st.markdown(f"**{sender.capitalize()}** ({ts}): {msg}")
        msg = st.text_input("Send a message to patient:")
        if st.button("Send Message"):
            if msg:
                db.add_admin_patient_message(user_id, patient_id, "admin", msg)
                st.rerun()
    elif user_role == "patient":
        user = db.get_user(st.session_state.username)
        admin_id = user[4]
        if not admin_id:
            st.info("You are not assigned to any admin.")
            return
        chat = db.get_admin_patient_chat(admin_id, user_id)
        st.subheader("Chat with Your Admin")
        for sender, msg, ts in chat:
            st.markdown(f"**{sender.capitalize()}** ({ts}): {msg}")
        msg = st.text_input("Send a message to admin:")
        if st.button("Send Message"):
            if msg:
                db.add_admin_patient_message(admin_id, user_id, "patient", msg)
                st.rerun()

# Run the application
if __name__ == "__main__":
    main()
    show_footer
