"""
Clinical AI Chatbot - Complete Usage Guide & Examples

This file contains:
1. Quick start examples
2. Integration patterns
3. Streamlit integration
4. API integration
5. Advanced usage patterns
"""

# ============================================================================
# EXAMPLE 1: BASIC CHATBOT USAGE
# ============================================================================

"""
BASIC USAGE - Single Conversation

from clinical_ai_chatbot import ClinicalAIChatbot, PatientContext
from chatbot_config import ChatbotConfig

# Initialize
chatbot = ClinicalAIChatbot()
config = ChatbotConfig()

# Create patient
patient = PatientContext(
    patient_id="PAT-2026-001",
    age=35,
    gender="M",
    diagnosis="Hemophilia A",
    severity="moderate",
    treatment_history=["Factor VIII replacement"],
    current_medications=["Factor VIII concentrate"],
    allergies=["Penicillin"]
)

# Start session
conversation_id = chatbot.start_session(patient)

# Get recommendations
question = "What is the optimal prophylaxis strategy?"
response = chatbot.get_clinical_recommendations(question)
print(response)

# End session
summary = chatbot.end_session()
print(summary)
"""


# ============================================================================
# EXAMPLE 2: RISK ASSESSMENT
# ============================================================================

"""
RISK ASSESSMENT WORKFLOW

from clinical_ai_chatbot import ClinicalAIChatbot, PatientContext

chatbot = ClinicalAIChatbot()

patient = PatientContext(
    patient_id="PAT-2026-001",
    age=35,
    gender="M",
    diagnosis="Hemophilia A",
    severity="moderate",
    treatment_history=["Factor VIII replacement", "Prophylaxis"],
    current_medications=["Factor VIII", "Aspirin"],
    allergies=[]
)

chatbot.start_session(patient)

# Comprehensive risk assessment
assessment_data = {
    "key_risk_factors": [
        "Inhibitor development risk",
        "Joint damage progression",
        "Medication non-adherence"
    ],
    "severity_score": 65,
    "comorbidities": ["HIV positive"],
    "lab_abnormalities": ["Low factor VIII activity"]
}

risk_explanation = chatbot.generate_risk_explanation(assessment_data)
print("RISK ASSESSMENT RESULT:")
print(risk_explanation)

chatbot.end_session()
"""


# ============================================================================
# EXAMPLE 3: MONITORING DATA ANALYSIS
# ============================================================================

"""
MONITORING DATA ANALYSIS

from clinical_ai_chatbot import ClinicalAIChatbot, PatientContext

chatbot = ClinicalAIChatbot()

patient = PatientContext(
    patient_id="PAT-2026-001",
    age=35,
    gender="M",
    diagnosis="Hemophilia A",
    severity="moderate",
    treatment_history=["Factor VIII replacement", "Prophylaxis"],
    current_medications=["Factor VIII", "Aspirin"],
    allergies=[]
)

chatbot.start_session(patient)

# Current monitoring data
monitoring_data = {
    "lab_results": {
        "factor_viii_activity": "40%",
        "inhibitor_level": "2 BU/mL",
        "hemoglobin": "14.2 g/dL",
        "platelet_count": "245K",
        "INR": "1.0"
    },
    "vital_signs": {
        "blood_pressure": "120/80 mmHg",
        "heart_rate": "72 bpm",
        "temperature": "37.2°C",
        "respiratory_rate": "16/min"
    },
    "symptoms": ["Mild joint swelling", "Easy bruising"],
    "medication_adherence": 95,
    "adverse_events": []
}

analysis = chatbot.analyze_monitoring_data(monitoring_data)
print("MONITORING ANALYSIS:")
print(analysis)

# Save clinical note
chatbot.db.save_clinical_note(
    patient_id=patient.patient_id,
    conversation_id=chatbot.conversation_id,
    note_type="monitoring_assessment",
    risk_score=65.0,
    monitoring_plan="Continue current regimen. Recheck labs in 3 months."
)

chatbot.end_session()
"""


# ============================================================================
# EXAMPLE 4: STREAMLIT INTEGRATION
# ============================================================================

STREAMLIT_EXAMPLE = """
# streamlit_app_chatbot.py

import streamlit as st
from clinical_ai_chatbot import ClinicalAIChatbot, PatientContext
from clinical_chatbot_integration import (
    ResponseFormatter, ConversationAnalytics, ConversationExporter
)
from chatbot_config import get_chatbot_instance, ChatbotConfig, print_configuration_status

st.set_page_config(
    page_title="Clinical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & Header
st.title("🏥 Clinical AI Chatbot - Hemophilia Support")
st.markdown("*Powered by GPT-4 with Clinical Decision Support*")

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    if st.button("Show Configuration Status"):
        print_configuration_status()
    
    st.divider()
    
    st.subheader("Patient Information")
    patient_id = st.text_input("Patient ID", value="PAT-2026-001")
    age = st.number_input("Age", min_value=1, max_value=120, value=35)
    gender = st.selectbox("Gender", ["M", "F", "Other"])
    
    st.subheader("Clinical Data")
    severity = st.selectbox("Severity", ["mild", "moderate", "severe"])
    diagnosis = st.text_input("Diagnosis", value="Hemophilia A")
    
    treatment_history = st.text_area(
        "Treatment History (comma-separated)",
        value="Factor VIII replacement,Prophylaxis therapy"
    ).split(",")
    
    current_meds = st.text_area(
        "Current Medications (comma-separated)",
        value="Factor VIII,Aspirin"
    ).split(",")
    
    allergies = st.text_area(
        "Allergies (comma-separated)",
        value="Penicillin"
    ).split(",")

# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

col1, col2 = st.columns([3, 1], gap="medium")

with col1:
    st.subheader("💬 Clinical Chat")
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = get_chatbot_instance()
    
    if "conversation_id" not in st.session_state:
        patient = PatientContext(
            patient_id=patient_id,
            age=age,
            gender=gender,
            diagnosis=diagnosis,
            severity=severity,
            treatment_history=treatment_history,
            current_medications=current_meds,
            allergies=allergies
        )
        st.session_state.chatbot.start_session(patient)
        st.session_state.conversation_id = st.session_state.chatbot.conversation_id
        st.success(f"✅ Session started - Patient: {patient_id}")
    
    # Chat input
    user_input = st.text_input(
        "Your clinical question:",
        placeholder="Ask for recommendations, analysis, or clinical guidance..."
    )
    
    if user_input:
        with st.spinner("🤖 Thinking..."):
            try:
                response = st.session_state.chatbot.chat(user_input)
                st.info(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    st.subheader("🎯 Quick Actions")
    
    if st.button("📊 Risk Analysis"):
        st.session_state.action = "risk_analysis"
    
    if st.button("💊 Get Recommendations"):
        st.session_state.action = "recommendations"
    
    if st.button("📈 Monitor Data"):
        st.session_state.action = "monitoring"
    
    if st.button("📝 Conversation History"):
        st.session_state.action = "history"

# ============================================================
# DYNAMIC ACTIONS
# ============================================================

if "action" in st.session_state:
    action = st.session_state.action
    
    if action == "risk_analysis":
        st.divider()
        st.subheader("🔴 Risk Analysis")
        
        risk_factors = st.multiselect(
            "Select risk factors",
            ["Inhibitor risk", "Joint damage", "Viral transmission", "Non-adherence"]
        )
        
        severity_score = st.slider("Severity Score", 0, 100, 65)
        
        if st.button("Generate Risk Assessment"):
            assessment_data = {
                "key_risk_factors": risk_factors,
                "severity_score": severity_score,
                "comorbidities": [],
                "lab_abnormalities": []
            }
            
            with st.spinner("Analyzing..."):
                risk_explanation = st.session_state.chatbot.generate_risk_explanation(assessment_data)
                st.success(risk_explanation)
    
    elif action == "recommendations":
        st.divider()
        st.subheader("💊 Treatment Recommendations")
        
        question = st.text_area(
            "Clinical question",
            value="What is the optimal prophylaxis strategy?"
        )
        
        if st.button("Get Recommendations"):
            with st.spinner("Generating recommendations..."):
                recommendations = st.session_state.chatbot.get_clinical_recommendations(question)
                st.success(recommendations)
    
    elif action == "monitoring":
        st.divider()
        st.subheader("📈 Monitoring Data Analysis")
        
        col_labs, col_vitals = st.columns(2)
        
        with col_labs:
            st.write("**Lab Results**")
            factor_viii = st.number_input("Factor VIII %", value=40)
            inhibitor = st.number_input("Inhibitor Level (BU/mL)", value=2.0)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", value=14.2)
        
        with col_vitals:
            st.write("**Vital Signs**")
            bp = st.text_input("Blood Pressure", value="120/80 mmHg")
            hr = st.number_input("Heart Rate (bpm)", value=72)
            temp = st.number_input("Temperature (°C)", value=37.2)
        
        adherence = st.slider("Medication Adherence %", 0, 100, 95)
        symptoms = st.text_area("Symptoms (comma-separated)", value="Mild joint swelling,Easy bruising")
        
        if st.button("Analyze Monitoring Data"):
            monitoring_data = {
                "lab_results": {
                    "factor_viii_activity": f"{factor_viii}%",
                    "inhibitor_level": f"{inhibitor} BU/mL",
                    "hemoglobin": f"{hemoglobin} g/dL"
                },
                "vital_signs": {
                    "blood_pressure": bp,
                    "heart_rate": f"{hr} bpm",
                    "temperature": f"{temp}°C"
                },
                "symptoms": [s.strip() for s in symptoms.split(",")] if symptoms else [],
                "medication_adherence": adherence,
                "adverse_events": []
            }
            
            with st.spinner("Analyzing..."):
                analysis = st.session_state.chatbot.analyze_monitoring_data(monitoring_data)
                st.success(analysis)
    
    elif action == "history":
        st.divider()
        st.subheader("📝 Conversation History")
        
        history = st.session_state.chatbot.db.get_conversation_history(
            st.session_state.conversation_id
        )
        
        for msg in history:
            if msg["role"] == "user":
                st.write(f"👤 **Patient**: {msg['content']}")
            else:
                st.write(f"🤖 **AI**: {msg['content']}")
                st.divider()

# ============================================================
# FOOTER & END SESSION
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 View Analytics"):
        analytics = ConversationAnalytics()
        stats = analytics.get_conversation_stats(patient_id)
        st.json(stats)

with col2:
    export_format = st.selectbox("Export As", ["JSON", "CSV", "Text"])
    if st.button("📥 Export Conversation"):
        exporter = ConversationExporter()
        if export_format == "JSON":
            data = exporter.export_to_json(st.session_state.conversation_id)
        elif export_format == "CSV":
            data = exporter.export_to_csv(st.session_state.conversation_id, "temp.csv")
        else:
            data = exporter.export_text_summary(st.session_state.conversation_id)
        
        st.download_button(
            label=f"Download {export_format}",
            data=data,
            file_name=f"conversation_{st.session_state.conversation_id}.{export_format.lower()}",
            mime="text/plain"
        )

with col3:
    if st.button("🏥 End Session"):
        summary = st.session_state.chatbot.end_session()
        st.success(f"✅ Session ended. Total messages: {summary['total_messages']}")
        del st.session_state.conversation_id
"""


# ============================================================================
# EXAMPLE 5: FLASK/FASTAPI INTEGRATION
# ============================================================================

FASTAPI_EXAMPLE = """
# fastapi_chatbot.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from clinical_ai_chatbot import ClinicalAIChatbot, PatientContext
from chatbot_config import get_chatbot_instance

app = FastAPI(title="Clinical AI Chatbot API")

# ============================================================
# MODELS
# ============================================================

class PatientData(BaseModel):
    patient_id: str
    age: int
    gender: str
    diagnosis: str
    severity: str
    treatment_history: List[str]
    current_medications: List[str]
    allergies: List[str]

class ChatMessage(BaseModel):
    patient_id: str
    message: str
    conversation_id: Optional[int] = None

class RiskAssessmentRequest(BaseModel):
    patient_data: PatientData
    key_risk_factors: List[str]
    severity_score: int
    comorbidities: List[str]
    lab_abnormalities: List[str]

class MonitoringAnalysisRequest(BaseModel):
    patient_data: PatientData
    lab_results: dict
    vital_signs: dict
    symptoms: List[str]
    medication_adherence: int
    adverse_events: List[str]

# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    \"\"\"Chat endpoint\"\"\"
    try:
        chatbot = get_chatbot_instance()
        
        # Start session if needed
        if not message.conversation_id:
            patient = PatientContext(
                patient_id=message.patient_id,
                age=35, gender="M", diagnosis="Hemophilia A", severity="moderate",
                treatment_history=[], current_medications=[], allergies=[]
            )
            chatbot.start_session(patient)
            conversation_id = chatbot.conversation_id
        else:
            conversation_id = message.conversation_id
        
        response = chatbot.chat(message.message)
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "response": response
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/risk-analysis")
async def risk_analysis_endpoint(request: RiskAssessmentRequest):
    \"\"\"Risk analysis endpoint\"\"\"
    try:
        chatbot = get_chatbot_instance()
        
        patient = PatientContext(**request.patient_data.dict())
        chatbot.start_session(patient)
        
        assessment_data = {
            "key_risk_factors": request.key_risk_factors,
            "severity_score": request.severity_score,
            "comorbidities": request.comorbidities,
            "lab_abnormalities": request.lab_abnormalities
        }
        
        risk_explanation = chatbot.generate_risk_explanation(assessment_data)
        summary = chatbot.end_session()
        
        return {
            "status": "success",
            "risk_explanation": risk_explanation,
            "session_summary": summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/monitoring-analysis")
async def monitoring_endpoint(request: MonitoringAnalysisRequest):
    \"\"\"Monitoring data analysis endpoint\"\"\"
    try:
        chatbot = get_chatbot_instance()
        
        patient = PatientContext(**request.patient_data.dict())
        chatbot.start_session(patient)
        
        monitoring_data = {
            "lab_results": request.lab_results,
            "vital_signs": request.vital_signs,
            "symptoms": request.symptoms,
            "medication_adherence": request.medication_adherence,
            "adverse_events": request.adverse_events
        }
        
        analysis = chatbot.analyze_monitoring_data(monitoring_data)
        summary = chatbot.end_session()
        
        return {
            "status": "success",
            "analysis": analysis,
            "session_summary": summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    \"\"\"Health check\"\"\"
    try:
        chatbot = get_chatbot_instance()
        return {"status": "healthy", "chatbot": "ready"}
    except:
        return {"status": "unhealthy"}

# Run with: uvicorn fastapi_chatbot:app --reload
"""


# ============================================================================
# EXAMPLE 6: QUICK START PATTERNS
# ============================================================================

QUICK_START_EXAMPLES = """
# Quick Start Patterns

# 1. Quick Risk Analysis (No full session)
from chatbot_config import quick_risk_analysis

patient_data = {
    "patient_id": "PAT-001",
    "age": 35,
    "gender": "M",
    "severity": "moderate"
}

assessment = {
    "key_risk_factors": ["Inhibitor risk", "Joint damage"],
    "severity_score": 65
}

risk_text = quick_risk_analysis(patient_data, assessment)
print(risk_text)


# 2. Quick Recommendations
from chatbot_config import quick_recommendations

recommendations = quick_recommendations(
    patient_data,
    "What prophylaxis strategy is best?"
)
print(recommendations)


# 3. Quick Monitoring Analysis
from chatbot_config import quick_monitoring_analysis

monitoring = {
    "lab_results": {"factor_viii_activity": "40%"},
    "vital_signs": {"blood_pressure": "120/80"},
    "symptoms": [],
    "medication_adherence": 95,
    "adverse_events": []
}

analysis = quick_monitoring_analysis(patient_data, monitoring)
print(analysis)


# 4. Full Session with Multiple Operations
from chatbot_config import get_chatbot_instance
from clinical_ai_chatbot import PatientContext

chatbot = get_chatbot_instance()

patient = PatientContext(
    patient_id="PAT-001",
    age=35, gender="M",
    diagnosis="Hemophilia A",
    severity="moderate",
    treatment_history=["Factor VIII"],
    current_medications=["Factor VIII"],
    allergies=[]
)

# Start
convo_id = chatbot.start_session(patient)

# Multiple operations
risk = chatbot.generate_risk_explanation({"severity_score": 65})
recs = chatbot.get_clinical_recommendations("Best prophylaxis?")
analysis = chatbot.analyze_monitoring_data({})

# Chat
response = chatbot.chat("What should I monitor?")

# End
summary = chatbot.end_session()
"""


# ============================================================================
# REQUIREMENTS
# ============================================================================

REQUIREMENTS = """
# requirements.txt for Clinical AI Chatbot

# Core
openai>=0.27.0
python-dotenv>=0.21.0

# For Streamlit integration
streamlit>=1.20.0
streamlit-chat>=0.0.1  # Optional: for better chat UI

# For FastAPI integration (optional)
fastapi>=0.95.0
uvicorn>=0.21.0
pydantic>=1.10.0

# Data processing
pandas>=1.5.0
numpy>=1.24.0

# Database (usually included)
sqlite3  # Built-in

# Utilities
requests>=2.28.0
"""


# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================

SETUP_INSTRUCTIONS = """
# Clinical AI Chatbot Setup Instructions

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
# Copy example and edit
cp .env.example .env

# Add your OpenAI API key
# OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

## Step 3: Initialize Configuration

```python
from chatbot_config import setup_environment, load_environment, print_configuration_status

setup_environment()
load_environment()
print_configuration_status()
```

## Step 4: Choose Integration

### Option A: Standalone Python Script
```python
from chatbot_config import quick_recommendations

result = quick_recommendations(patient_data, "Your question")
```

### Option B: Streamlit App
```bash
streamlit run streamlit_app_chatbot.py
```

### Option C: FastAPI Server
```bash
uvicorn fastapi_chatbot:app --reload
```

## Step 5: Run Demo

```bash
python clinical_ai_chatbot.py
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `CHATBOT_DB`: Path to SQLite database (default: clinical_conversations.db)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING)

## File Structure

```
project/
├── clinical_ai_chatbot.py           # Main chatbot
├── clinical_chatbot_integration.py  # Integration helpers
├── chatbot_config.py               # Configuration
├── streamlit_app_chatbot.py        # Streamlit app
├── fastapi_chatbot.py              # FastAPI server
├── requirements.txt                # Dependencies
└── clinical_conversations.db       # SQLite database (auto-created)
```
"""


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("📚 EXAMPLES GUIDE")
    print("="*70)
    print("\nChoose an example to view:")
    print("1. Basic Usage")
    print("2. Risk Assessment")
    print("3. Monitoring Analysis")
    print("4. Streamlit Integration")
    print("5. FastAPI Integration")
    print("6. Quick Start Patterns")
    print("7. Requirements")
    print("8. Setup Instructions")
