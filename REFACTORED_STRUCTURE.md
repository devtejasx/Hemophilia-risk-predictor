# Hemophilia AI Platform - Refactored Structure

## 🎯 Overview

Professional, scalable Streamlit multi-page application featuring:
- **6 Feature Pages** with independent functionality
- **Reusable Components** for consistent UI
- **Business Logic Layer** with services
- **Database Abstraction** for data operations
- **Session State Management** for data sharing
- **Professional Theming** with dark/light mode

---

## 📁 Project Structure

```
capstone/
│
├── app_refactored.py          ⭐ MAIN ENTRY POINT (Streamlit app)
│
├── pages/                      📄 Multi-page features
│   ├── __init__.py
│   ├── 1_Dashboard.py          📊 Overview & metrics
│   ├── 2_Add_Patient.py        👤 Patient registration
│   ├── 3_Predictions.py        🔮 Risk assessment
│   ├── 4_SHAP_Explainability.py 🧠 Model interpretation
│   ├── 5_Chatbot.py            🤖 AI assistant
│   └── 6_Analytics.py          📈 Advanced analytics
│
├── components/                 🎨 UI Components (Reusable)
│   ├── __init__.py
│   ├── navbar.py              Sidebar & navigation
│   ├── cards.py               Info, metric, patient cards
│   └── charts.py              Plotly, matplotlib charts
│
├── services/                   ⚙️ Business Logic
│   ├── __init__.py
│   ├── ml_service.py          Model predictions & SHAP
│   ├── api_client.py          Backend API communication
│   └── chatbot_service.py     LLM & chat logic
│
├── utils/                      🛠️ Utilities
│   ├── __init__.py
│   ├── helpers.py             Helper functions
│   └── session_state.py       State management
│
├── database/                   💾 Data Layer
│   ├── __init__.py
│   └── db.py                  Database abstraction
│
├── styles/                     🎨 Theming
│   ├── __init__.py
│   └── css.py                 CSS & styling
│
└── assets/                     📦 Static files

```

---

## 🚀 Running the App

### Option 1: Main Entry Point (Recommended)
```bash
streamlit run app_refactored.py
```

This launches the main page with navigation to all 6 pages.

### Option 2: Run Individual Pages
```bash
streamlit run pages/1_Dashboard.py
streamlit run pages/3_Predictions.py
streamlit run pages/4_SHAP_Explainability.py
```

---

## 📄 Pages Overview

### 1. **Dashboard** (`pages/1_Dashboard.py`)
- Real-time statistics
- Patient count, risk distribution
- Recent predictions
- Key metrics cards
- Trend charts

**Key Components:**
- Metric cards (patients, predictions, alerts)
- Risk distribution chart
- Patient timeline
- Status indicators

---

### 2. **Add Patient** (`pages/2_Add_Patient.py`)
- Patient registration form
- Clinical data entry
- Family history tracking
- Treatment parameters
- Form validation
- Database save

**Key Fields:**
- Demographics (name, age, DOB, ethnicity)
- Blood type, HLA typing
- Hemophilia severity
- Family history
- Current treatments
- Medical history

---

### 3. **Predictions** (`pages/3_Predictions.py`)
- Risk assessment form
- ML model predictions
- Ensemble results (RF + XGBoost)
- Risk score visualization
- Confidence metrics
- Recommendations

**Features:**
- Input form for patient data
- Real-time prediction
- Risk gauge chart
- Feature importance display
- Treatment recommendations
- PDF report generation

---

### 4. **SHAP Explainability** (`pages/4_SHAP_Explainability.py`)
- Model explanation interface
- Basic vs. Advanced modes
- Waterfall plots
- Force plots
- Dependence plots
- Feature contribution analysis

**Views:**
- Basic: Simple risk explanation
- Advanced: Detailed SHAP analysis
- Comparison: Multiple patients
- Report generation

---

### 5. **Chatbot** (`pages/5_Chatbot.py`)
- Clinical AI assistant
- Multi-turn conversation
- Mode switching (clinical, general, treatment)
- Chat history
- Real-time responses
- Export conversations

**Modes:**
- `clinical_assistant`: Medical insights
- `general_qa`: General questions
- `treatment_recommendations`: Treatment advice

---

### 6. **Analytics** (`pages/6_Analytics.py`)
- Advanced data analysis
- Patient cohorts
- Treatment outcomes
- Risk distribution
- Interactive filters
- Export functionality

**Features:**
- Severity distribution
- Risk level breakdown
- Age cohort analysis
- Treatment response metrics
- Trend analysis

---

## 🎨 Components Guide

### Navbar (`components/navbar.py`)
```python
from components.navbar import show_sidebar, show_page_header

# In any page:
show_sidebar()           # Display navigation sidebar
show_page_header(        # Show page title
    title="Page Title",
    subtitle="Optional subtitle",
    icon="📊"
)
```

### Cards (`components/cards.py`)
```python
from components.cards import metric_card, info_card, patient_card, empty_state

metric_card(
    label="Total Patients",
    value="125",
    delta="+12 this month",
    icon="👥"
)

patient_card(
    name="John Doe",
    age=35,
    severity="Severe",
    risk_score=72.5,
    patient_id="P001"
)

empty_state(
    icon="📭",
    title="No Data",
    message="Add patients to see statistics"
)
```

### Charts (`components/charts.py`)
```python
from components.charts import (
    plot_risk_gauge,
    plot_feature_importance,
    plot_patient_metrics
)

plot_risk_gauge(risk_score=65.5)

plot_feature_importance({
    "Mutation Type": 0.35,
    "Severity": 0.25,
    "Age": 0.20,
    "Dose": 0.15,
    "Exposure": 0.05
})
```

---

## ⚙️ Services Guide

### ML Service (`services/ml_service.py`)
```python
from services.ml_service import MLService

ml = MLService()

# Make prediction
result = ml.predict(feature_vector)
# Returns: {
#     "risk_score": 0.65,
#     "rf_score": 0.63,
#     "xgb_score": 0.67,
#     "main_factor": "Mutation Type"
# }

# Get SHAP explanation
shap_data = ml.explain_prediction(features)
```

### Chatbot Service (`services/chatbot_service.py`)
```python
from services.chatbot_service import ChatbotService

chatbot = ChatbotService(mode="clinical_assistant")

response = chatbot.get_response(
    user_message="What are treatment options?",
    context=patient_data
)
```

### API Client (`services/api_client.py`)
```python
from services.api_client import APIClient

api = APIClient(base_url="http://localhost:8000")

patients = api.get_patients()
prediction = api.predict(patient_data)
report = api.generate_report(patient_id)
```

---

## 🛠️ Utilities Guide

### Session State (`utils/session_state.py`)
```python
from utils.session_state import (
    init_session_state,
    get_session_var,
    set_session_var,
    get_patient_data,
    set_patient_data
)

# Initialize (call once in main app)
init_session_state()

# Get/set variables across pages
user = get_session_var("user_name")
set_session_var("current_patient", patient_dict)

# Patient-specific
current_patient = get_patient_data()
set_patient_data(new_patient)
```

### Helpers (`utils/helpers.py`)
```python
from utils.helpers import (
    format_number,
    format_percentage,
    get_risk_level,
    calculate_age,
    validate_email
)

age = calculate_age("2000-01-15")  # 24
risk_label = get_risk_level(0.65)  # "🟡 MODERATE"
formatted = format_percentage(0.875)  # "87.5%"
```

---

## 💾 Database Layer

### DB Module (`database/db.py`)
```python
from database.db import get_database

db = get_database()

# Patient operations
patients = db.get_patients(limit=100)
patient = db.get_patient(patient_id)
db.save_patient(patient_data)
db.delete_patient(patient_id)

# Predictions
predictions = db.get_predictions(patient_id)
db.save_prediction(prediction_data)
```

---

## 🎨 Styling

### Theme Configuration (`styles/css.py`)
```python
from styles.css import apply_theme, get_risk_color, get_risk_label

# Apply in main app (already done in app_refactored.py)
apply_theme()

# Use colors and labels
color = get_risk_color(0.75)  # "#dc3545"
label = get_risk_label(0.75)  # "HIGH RISK"
```

---

## 📊 Data Flow

### Adding a Patient
```
Add Patient Page
    ↓
Validate Form Input
    ↓
DB Save (database/db.py)
    ↓
Session State Update
    ↓
Success Notification
```

### Making a Prediction
```
Predictions Page
    ↓
Input Patient Data
    ↓
ML Service (services/ml_service.py)
    ↓
Get Risk Score + SHAP
    ↓
Display Results + Explanation
    ↓
Save to DB & History
```

### Using Chat
```
Chatbot Page
    ↓
User Message
    ↓
Chatbot Service (services/chatbot_service.py)
    ↓
LLM Response + Context
    ↓
Display in Chat UI
    ↓
Save Conversation History
```

---

## 🔐 Session State Variables

All session variables are initialized in `utils/session_state.py`:

```
Authentication:
- authenticated (bool)
- user_id, user_name, user_role

Patient Data:
- current_patient (dict)
- patient_form_data (dict)
- patient_list (list)

Predictions:
- last_prediction (dict)
- prediction_history (list)
- shap_explanation (dict)

Chat:
- chat_history (list)
- chat_mode (str)

UI:
- theme (str: "dark" or "light")
- show_advanced (bool)
- view_mode (str: "grid" or "list")
```

---

## 🚀 Adding a New Feature

### 1. Create Page File
```python
# pages/7_NewFeature.py
import streamlit as st
from components.navbar import show_sidebar, show_page_header
from utils.session_state import init_session_state

st.set_page_config(page_title="New Feature", layout="wide")

init_session_state()
show_sidebar()
show_page_header("🆕 New Feature", "Description")

# Your feature code here
```

### 2. Update Sidebar
In `components/navbar.py`, add to navigation options:
```python
options=[
    ...,
    "New Feature"
]
```

### 3. Use Existing Services
```python
from services.ml_service import MLService
from database.db import get_database

ml = MLService()
db = get_database()
```

---

## 📝 Best Practices

### ✅ DO:
- ✅ Use `init_session_state()` in each page
- ✅ Import components from `components/`
- ✅ Use helpers from `utils/helpers.py`
- ✅ Leverage services in `services/`
- ✅ Call `show_sidebar()` for navigation
- ✅ Use consistent styling

### ❌ DON'T:
- ❌ Import from `app.py` in pages
- ❌ Duplicate utility functions
- ❌ Direct database calls without `db.py`
- ❌ Inconsistent error handling
- ❌ Missing session state initialization

---

## 🔧 Configuration

### Environment Variables
Create `.env`:
```
DATABASE_URL=mongodb://localhost:27017
API_URL=http://localhost:8000
OPENAI_API_KEY=your_key_here
SHAP_ENABLED=true
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Message")
logger.error("Error")
```

---

## 📦 Dependencies

Core:
- `streamlit>=1.28`
- `pandas>=2.0`
- `numpy>=1.24`

ML/Data:
- `scikit-learn>=1.3`
- `xgboost>=2.0`
- `shap>=0.43`

Visualization:
- `plotly>=5.17`
- `matplotlib>=3.7`

API/Chat:
- `requests>=2.31`
- `openai>=1.3`

Database:
- `pymongo>=4.5`
- `sqlalchemy>=2.0`

---

## 🎓 Examples

### Example 1: Display Patient Data
```python
from database.db import get_database
from components.cards import patient_card

db = get_database()
patients = db.get_patients()

for patient in patients:
    patient_card(
        name=patient["name"],
        age=patient["age"],
        severity=patient["severity"],
        risk_score=patient.get("risk_score", 0),
        patient_id=patient["_id"]
    )
```

### Example 2: Accept Form Input & Save
```python
from utils.session_state import set_session_var, get_session_var
from database.db import get_database

form_data = {
    "name": st.text_input("Name"),
    "age": st.number_input("Age", 0, 100),
    "severity": st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
}

if st.button("Save Patient"):
    db = get_database()
    db.save_patient(form_data)
    st.success("Patient saved!")
```

### Example 3: Get Predictions with SHAP
```python
from services.ml_service import MLService
from components.charts import plot_risk_gauge, plot_feature_importance

ml = MLService()

features = {...patient data...}
result = ml.predict(features)

col1, col2 = st.columns(2)
with col1:
    plot_risk_gauge(result["risk_score"] * 100)
with col2:
    plot_feature_importance(result.get("importance", {}))
```

---

## 📞 Support

For issues or questions:
1. Check existing pages for patterns
2. Review documentation in `styles/`, `services/`, `utils/`
3. Use `st.write()` for debugging
4. Check logs in terminal

---

## ✅ Checklist for Completion

- [x] Project structure created
- [x] Multi-page Streamlit setup
- [x] Components library
- [x] Services layer
- [x] Session state management
- [x] Database abstraction
- [x] Styling system
- [x] 6 Feature pages
- [x] Documentation
- [ ] Deploy to production

---

**Last Updated:** 2024
**Version:** 3.0 (Refactored)
