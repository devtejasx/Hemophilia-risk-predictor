# 🏥 Medical AI System - Multi-Page Streamlit Application

**Production-Ready Medical AI Interface with Risk Prediction, SHAP Explainability, and Clinical Decision Support**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pages Guide](#pages-guide)
- [Backend Integration](#backend-integration)
- [Components](#components)
- [Usage Examples](#usage-examples)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

A comprehensive, production-ready Streamlit application for medical AI decision support. The system provides:

- **Multi-page interface** with 5 specialized pages
- **State management** for seamless cross-page data flow
- **SHAP explainability** for model predictions
- **AI chatbot integration** for clinical Q&A
- **Analytics dashboard** for system monitoring
- **Patient management** with CRUD operations
- **Clinical visualizations** with Plotly charts

### Key Metrics
- **10 files created** (3,500+ lines of code)
- **5 fully functional pages**
- **30+ UI components**
- **10+ chart types**
- **15+ API endpoints**

---

## ✨ Features

### Patient Management
- ✅ Create new patient records with comprehensive clinical data
- ✅ Load existing patients from database
- ✅ Quick assessment for rapid risk evaluation
- ✅ Edit and update patient information
- ✅ View patient history and trends

### Risk Assessment
- ✅ ML model integration for risk prediction
- ✅ Real-time risk scoring (0-100%)
- ✅ Color-coded risk levels (Low → Critical)
- ✅ Confidence scores
- ✅ Clinical recommendations based on risk

### Explainability
- ✅ SHAP waterfall plots showing feature contributions
- ✅ Feature importance rankings
- ✅ Plain English explanations
- ✅ Visual interpretation guides
- ✅ Red/blue indicators for increasing/decreasing risk

### Historical Analysis
- ✅ Trend visualization over time
- ✅ Statistical analysis (mean, median, percentiles)
- ✅ Peak/dip detection
- ✅ Date range filtering
- ✅ Pagination for large datasets
- ✅ Export to CSV/JSON/Excel

### AI Chatbot
- ✅ Conversational interface
- ✅ Context-aware responses (uses loaded patient)
- ✅ Pre-configured question templates
- ✅ Chat history with timestamps
- ✅ Integration with clinical AI assistant

### Analytics Dashboard
- ✅ System-wide statistics
- ✅ Risk distribution analysis
- ✅ Patient cohort analysis
- ✅ Trend analysis by time period
- ✅ System health monitoring

### UI/UX
- ✅ Professional dark theme
- ✅ Medical color scheme (risk indicators)
- ✅ Responsive design
- ✅ Loading indicators
- ✅ Error handling and feedback
- ✅ Icon-based navigation

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           STREAMLIT MEDICAL AI INTERFACE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │            streamlit_medical_app.py                  │ │
│  │  (Main entry point, routing, page management)       │ │
│  └──────────────────────────────────────────────────────┘ │
│                           │                                │
│           ┌───────────────┼───────────────┐               │
│           ▼               ▼               ▼               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Page 1     │  │   Page 2     │  │   Page 3     │   │
│  │ Patient Form │  │   Results    │  │   History    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│           │               │               │               │
│           └───────────────┼───────────────┘               │
│                           │                                │
│                    ┌──────▼──────┐                        │
│                    │StateManager  │                        │
│                    │(Session      │                        │
│                    │State)        │                        │
│                    └──────────────┘                        │
│                           │                                │
│           ┌───────────────┼───────────────┐               │
│           ▼               ▼               ▼               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Page 4     │  │   Page 5     │  │UI Components │   │
│  │  Chatbot     │  │  Dashboard   │  │& Utilities   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│           │               │               │               │
│           └───────────────┼───────────────┘               │
│                           │                                │
│                    ┌──────▼──────────────┐               │
│                    │BackendClient        │               │
│                    │(FastAPI Comm)       │               │
│                    └──────┬──────────────┘               │
└─────────────────────────┼─────────────────────────────────┘
                          │
                ┌─────────▼─────────┐
                │  FastAPI Backend  │
                │  (ML Model,       │
                │  Database,        │
                │  Chatbot)         │
                └───────────────────┘
```

### Module Breakdown

#### Core Modules

**1. streamlit_medical_app.py** (120 lines)
- Multi-page routing
- Sidebar navigation with option_menu
- Session state initialization
- Backend health monitoring
- User role selection

**2. streamlit_utils/state_manager.py** (200+ lines)
- StateManager class for centralized state
- Patient data management
- Prediction storage
- Chat history tracking
- Backend connection checks

**3. streamlit_utils/ui_components.py** (400+ lines)
- setup_page_config() - Configuration
- apply_custom_styling() - Dark theme CSS
- UI helpers (cards, spinners, messages)
- Form components
- Layout builders

**4. streamlit_utils/backend_client.py** (350+ lines)
- BackendClient class for API communication
- Health checks
- Prediction endpoints
- Patient CRUD operations
- Chat integration
- Analytics queries

**5. streamlit_utils/plotly_charts.py** (600+ lines)
- MedicalCharts class
- Risk gauge, trend lines, time series
- Feature importance, waterfall plots
- Risk distribution, correlation heatmaps
- Medical visualization utilities

#### Page Modules

**6. streamlit_pages/patient_form.py** (350+ lines)
- New patient form with validation
- Patient selection from database
- Quick assessment workflow

**7. streamlit_pages/results_dashboard.py** (350+ lines)
- Risk score display with gauge
- SHAP explainability
- Trend analysis
- Clinical recommendations

**8. streamlit_pages/patient_history.py** (400+ lines)
- Records table with filtering
- Risk timeline visualization
- Statistical analysis
- Data export functionality

**9. streamlit_pages/ai_chatbot.py** (250+ lines)
- Chat interface
- Message history
- Example questions
- Context awareness

**10. streamlit_pages/doctor_dashboard.py** (450+ lines)
- System overview
- Risk distribution analysis
- Trend analysis
- Cohort analysis
- System status

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- FastAPI backend (for full functionality)

### Step 1: Install Dependencies

```bash
pip install streamlit streamlit-option-menu plotly pandas requests numpy
```

### Step 2: Optional Dependencies

```bash
# For Excel export support
pip install openpyxl

# For environment variables
pip install python-dotenv
```

### Step 3: Verify Installation

```bash
streamlit --version
```

---

## 🎬 Quick Start

### 1. Start FastAPI Backend

```bash
# Terminal 1 - Start your FastAPI backend
python api.py
# Backend should be available at http://localhost:8000
```

### 2. Run Streamlit App

```bash
# Terminal 2 - Start Streamlit
cd path/to/app
streamlit run streamlit_medical_app.py
```

### 3. Access Application

Open your browser to: `http://localhost:8501`

### 4. Test Workflow

1. ✅ Go to **🔬 Patient Form**
2. ✅ Select **"New Patient"** tab
3. ✅ Fill in a test patient
4. ✅ Click **"Save Patient"**
5. ✅ Go to **📊 Results**
6. ✅ Click **"Generate Prediction"**
7. ✅ Review risk score and SHAP plots

---

## 📄 Pages Guide

### Page 1: 🔬 Patient Form

**Purpose**: Create new patients or load existing ones

**Tabs**:
1. **New Patient**
   - Personal information (name, age, gender)
   - Medical history (diagnosis, severity, comorbidities)
   - Clinical measurements (vitals: BP, HR, O2, temp)
   - Laboratory values (glucose, cholesterol, creatinine, etc.)
   - Notes and observations
   - Form validation and submission

2. **Select Existing**
   - Load patient from database
   - View patient summary
   - Set as current patient

3. **Quick Assessment**
   - Minimal fields for rapid evaluation
   - Direct prediction without full form
   - Ideal for urgent cases

### Page 2: 📊 Results Dashboard

**Purpose**: Display predictions and explanations

**Tabs**:
1. **Risk Score**
   - Risk gauge (0-100%)
   - Color-coded severity
   - Model accuracy metrics
   - Confidence score

2. **Explainability**
   - SHAP waterfall plot
   - Feature importance ranking
   - Red (risk increase) vs Blue (risk decrease)
   - Interpretation guide

3. **Trends**
   - Historical predictions
   - Trend line chart
   - Peak/dip analysis
   - Statistical summaries

4. **Recommendations**
   - Priority-based actions
   - Clinical interventions
   - Follow-up schedule
   - Severity-specific suggestions

### Page 3: 📋 Patient History

**Purpose**: Analyze historical patient data

**Tabs**:
1. **Records Table**
   - Filtered historical records
   - Date range filter
   - Sort options
   - Pagination

2. **Risk Timeline**
   - Time series chart
   - Notable events
   - Trend indicators

3. **Analysis**
   - Risk distribution
   - Descriptive statistics
   - Trend analysis
   - Risk improvement/decline

4. **Export**
   - Download CSV
   - Download JSON
   - Download Excel
   - Export summary

### Page 4: 💬 AI Chatbot

**Purpose**: Interactive Q&A with clinical AI

**Features**:
- Real-time chat interface
- Context-aware (uses loaded patient)
- Message history with timestamps
- Pre-configured example questions
- Chat history clear button

**Example Questions**:
- "What is the patient's current medical status?"
- "What are the main risk factors?"
- "What clinical interventions are recommended?"
- "What trends do you see?"

### Page 5: 👨‍⚕️ Doctor Dashboard

**Purpose**: System-wide analytics

**Tabs**:
1. **Overview**
   - Total patients metrics
   - System health status
   - Today's summary
   - Key indicators

2. **Risk Distribution**
   - Pie chart of risk levels
   - Risk breakdown by level
   - Action items

3. **Trends**
   - Select time period (week/month/quarter)
   - Predictions over time
   - Average risk trends

4. **Cohort Analysis**
   - By Age Group
   - By Diagnosis
   - By Risk Level
   - By Gender

5. **System Status**
   - Backend services
   - Database connection
   - ML model status
   - Application info

---

## 🔌 Backend Integration

### Required API Endpoints

The backend must provide these endpoints:

```python
# Health & Status
GET /api/health                      # -> {"status": "ok"}
GET /api/status                      # -> Status details

# Predictions
POST /api/predict                    # Input: patient_data -> risk_score
POST /api/predict/batch              # Batch predictions
POST /api/explain                    # -> SHAP explanation

# Patient Management
GET /api/patients                    # -> List of patients
GET /api/patients/{id}               # -> Single patient
POST /api/patients                   # Create patient
PUT /api/patients/{id}               # Update patient
GET /api/patients/{id}/history       # Patient history

# Chat/Chatbot
POST /api/chat                       # Send message
GET /api/chat/history                # Chat history

# Analytics
GET /api/analytics/dashboard         # System stats
GET /api/analytics/risk-distribution # Risk data
GET /api/analytics/trends            # Trends

# Model Info
GET /api/model/info                  # Model metadata
GET /api/model/features              # Feature importance
```

### Expected Response Formats

```json
// Prediction Response
{
  "risk_score": 0.65,
  "risk_label": "HIGH",
  "confidence": 0.92,
  "model_name": "RF-Ensemble",
  "model_version": "1.0.0",
  "model_accuracy": 0.89,
  "model_auc": 0.91,
  "recommendations": [
    {
      "priority": "HIGH",
      "action": "Schedule follow-up",
      "reason": "Risk is elevated"
    }
  ]
}

// SHAP Response
{
  "shap_values": [0.15, -0.05, 0.12, ...],
  "feature_names": ["Age", "BP_Systolic", "BMI", ...],
  "base_value": 0.5,
  "prediction_value": 0.65
}

// Patient Response
{
  "patient_id": "P001",
  "first_name": "John",
  "last_name": "Doe",
  "age": 52,
  "gender": "Male",
  "diagnosis": "Hypertension",
  "severity": "Moderate",
  "vitals": {...},
  "measurements": {...},
  "labs": {...},
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Configuration

Backend URL (edit in `backend_client.py`):
```python
BASE_URL = "http://localhost:8000"
API_PREFIX = f"{BASE_URL}/api"
TIMEOUT = 30  # seconds
```

---

## 🧩 Components

### StateManager

Centralized session state management:

```python
state = StateManager()

# Patient management
state.set_current_patient(patient_id, patient_data)
patient = state.get_current_patient()
state.clear_current_patient()

# Predictions
state.set_prediction_results(predictions)
predictions = state.get_prediction_results()

# Chat
state.add_chat_message("user", "Hello")
history = state.get_chat_history()
state.clear_chat_history()

# Filters
state.set_date_range(start_date, end_date)
start, end = state.get_date_range()

# General
state.set("key", value)
value = state.get("key", default)
state.reset_session()
```

### UI Components

```python
from streamlit_utils.ui_components import *

# Configuration
setup_page_config()
apply_custom_styling()

# Headers
create_header("Title", "Subtitle")
create_tabs(["Tab 1", "Tab 2"])

# Cards
metric_card("Title", "Value", delta="10%", icon="📊")
risk_card(0.65, "HIGH")

# Forms
form_section("Section", "📋")
required_input("Label", "key", input_type="text")

# Messages
show_success("Success message!")
show_error("Error message")
show_warning("Warning message")
show_info("Info message")

# Charts
fig = MedicalCharts.risk_gauge(0.65)
fig = MedicalCharts.trend_line(dates, values)
fig = MedicalCharts.feature_importance(features, scores)
```

### Backend Client

```python
from streamlit_utils.backend_client import get_backend_client

backend = get_backend_client()

# Health
if backend.health_check():
    print("Backend online")

# Predictions
result = backend.predict(patient_data)
explanation = backend.get_explainability(patient_data)

# Patients
patients = backend.list_patients()
patient = backend.get_patient(patient_id)
new_patient = backend.create_patient(patient_data)
backend.update_patient(patient_id, updates)

# History
history = backend.get_patient_history(patient_id)

# Chat
response = backend.send_chat_message(message, patient_id)

# Analytics
stats = backend.get_dashboard_stats()
risks = backend.get_risk_distribution()
trends = backend.get_trends("month")
```

---

## 💻 Usage Examples

### Example 1: Create and Predict

```python
# 1. Go to 🔬 Patient Form
# 2. New Patient tab
# 3. Fill form:
patient_data = {
    "patient_id": "P001",
    "first_name": "John",
    "last_name": "Doe",
    "age": 52,
    "gender": "Male",
    "diagnosis": "Hypertension",
    "severity": "Moderate",
    "vitals": {
        "systolic_bp": 150,
        "diastolic_bp": 95,
        "heart_rate": 85,
        "oxygen_saturation": 98,
        "temperature": 37.0,
    },
    "measurements": {
        "bmi": 28.5,
        "glucose": 125,
        "cholesterol": 220,
    },
}
# 4. Save Patient
# 5. Go to 📊 Results
# 6. Generate Prediction
```

### Example 2: Analyze Patient History

```python
# 1. Load patient from Patient Form
# 2. Go to 📋 History
# 3. Set date range to "Last 30 days"
# 4. View Records table
# 5. Click Trends tab
# 6. Analyze timeline
# 7. Export as CSV
```

### Example 3: Ask Questions

```python
# 1. Load/create patient
# 2. Go to 💬 Chatbot
# 3. Ask: "What is this patient's current risk level?"
# 4. Read AI response
# 5. Ask: "What factors contribute most?"
# 6. Get explanation
```

---

## 🎨 Customization

### Change Color Scheme

Edit `streamlit_utils/plotly_charts.py`:

```python
COLORS = {
    "risk_critical": "#ff3333",    # Change colors here
    "risk_high": "#ff9900",
    "risk_moderate": "#ffcc00",
    "risk_low": "#00cc33",
    "primary": "#00d4ff",
    "secondary": "#0099cc",
    "accent": "#ff6b6b",
}
```

### Change Dark Theme

Edit `streamlit_utils/ui_components.py` CSS section

### Change Layout

Edit `streamlit_medical_app.py` sidebar and page_map

### Add New Page

1. Create `streamlit_pages/new_page.py`:
```python
def render():
    create_header("Title", "Subtitle")
    # Add your content
```

2. Import in `streamlit_medical_app.py`:
```python
from streamlit_pages import new_page
```

3. Add to `page_map`:
```python
page_map = {
    "🔬 Patient Form": patient_form.render,
    "📊 Results": results_dashboard.render,
    "📋 History": patient_history.render,
    "💬 Chatbot": ai_chatbot.render,
    "👨‍⚕️ Dashboard": doctor_dashboard.render,
    "🆕 New Page": new_page.render,  # Add here
}
```

---

## 🔧 Troubleshooting

### Backend Connection Failed
- [ ] Is backend running? `curl http://localhost:8000/api/health`
- [ ] Is port 8000 in use?
- [ ] Check firewall settings
- [ ] Try different URL in backend_client.py

### Patient Data Not Saving
- [ ] Is backend running?
- [ ] Check form validation (fill all required fields)
- [ ] Check database connection in backend
- [ ] Look at backend logs for errors

### SHAP Plots Not Showing
- [ ] Click "Generate SHAP Explanation" button
- [ ] Wait for computation
- [ ] Check backend has SHAP package
- [ ] Verify explain endpoint works

### Chat Not Responding
- [ ] Is chatbot service running?
- [ ] Check backend chat endpoint
- [ ] Verify patient is loaded
- [ ] Check API key if using external AI

### Slow Performance
- [ ] Limit number of records
- [ ] Reduce pagination size
- [ ] Check network speed
- [ ] Cache data appropriately
- [ ] Profile with @ st.cache_resource

---

## 📊 Monitoring & Logging

Check system status:
```python
backend = get_backend_client()
status = backend.get_status()
print(f"API: {status.get('api')}")
print(f"DB: {status.get('database')}")
print(f"Model: {status.get('model')}")
```

---

## 🚀 Deployment

### Local Development
```bash
streamlit run streamlit_medical_app.py
```

### Production (Streamlit Cloud)
1. Push to GitHub
2. Deploy via Streamlit Cloud
3. Set secrets for backend URL

### Production (Docker)
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_medical_app.py"]
```

---

## 📞 Support

- Check logs: Streamlit debug mode
- Backend logs: Check FastAPI server output
- Browser console: Check for JavaScript errors

---

## 📄 License

Medical AI System - 2024

---

## ✅ Version Info

- **App Version**: 1.0.0
- **Streamlit**: 1.25.0+
- **Python**: 3.8+
- **Plotly**: 5.0+
- **Status**: Production Ready
