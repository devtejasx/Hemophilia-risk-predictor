# 🏥 Hemophilia AI Platform - Complete Setup Guide

## 📋 Overview

This is an advanced AI-powered clinical intelligence platform for hemophilia management featuring:
- **Real GPT-4 Powered Chatbot** for clinical decision support
- **Doctor Dashboard** with comprehensive analytics and patient management
- **SQL Database** for persistent data storage
- **ML-based Risk Prediction** using Random Forest and XGBoost
- **Multi-page Streamlit Application**

---

## 🚀 Installation & Setup

### 1. **Install Required Packages**

```bash
pip install -r requirements.txt
```

**Key packages:**
- `streamlit==1.28.1` - Web framework
- `openai==1.3.5` - GPT-4 API client
- `python-dotenv==1.0.0` - Environment variables
- `pandas==2.0.3` - Data manipulation
- `scikit-learn==1.3.2` - Machine learning
- `plotly`, `matplotlib` - Visualization
- `reportlab==4.0.7` - PDF generation

### 2. **Configure OpenAI API Key**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_actual_openai_api_key_here
```

**To get an API key:**
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy and paste it into `.env`

### 3. **Initialize Database**

The database is automatically initialized on first run. It creates:
- `patients` - Patient demographic and clinical data
- `conversations` - Chatbot conversation history
- `doctor_notes` - Clinical notes from doctors
- `monitoring_records` - Lab and monitoring data
- `treatment_history` - Treatment logs
- `dashboard_analytics` - System analytics

---

## 📱 Application Pages

### 1. **📋 Patient Form**
- Comprehensive patient intake form
- 6 sections covering demographics, genetics, treatment, medical history, health status, and lifestyle
- ML-based risk prediction using trained models
- Automatic PDF report generation

### 2. **📊 Results**
- Detailed risk analysis and predictions
- Feature importance visualization using SHAP
- Clinical interpretation and recommendations
- Export capabilities

### 3. **📈 History**
- Complete patient database management
- Filtering by severity, mutation, and risk
- Statistical analysis and trends
- CSV export functionality

### 4. **🤖 Chatbot (NEW - GPT-4 POWERED)**
- Real-time AI medical assistant powered by GPT-4
- Context-aware responses based on patient data
- Three specialized buttons:
  - **Clinical Recommendations** - Generates comprehensive treatment plans
  - **Inhibitor Risk Analysis** - Detailed genetic and immunological analysis
  - **Monitoring Data Analysis** - Analytics on patient monitoring records
- Doctor note integration
- Conversation history stored in database

### 5. **🏥 Doctor Dashboard (NEW)**

#### **Tab 1: Patient Directory**
- View all registered patients
- Risk stratification with color-coded indicators
- Detailed patient profiles
- Doctor note management
- Add/edit clinical notes

#### **Tab 2: Clinical Notes**
- Browse notes by category (General, Inhibitor, Treatment, Monitoring)
- Filter by patient or severity
- Organized chronological display

#### **Tab 3: Analytics & Trends**
- Risk score distribution histogram
- Severity breakdown pie chart
- Mutation type analysis
- Adherence vs Risk scatter plot

#### **Tab 4: Search & Filter**
- Search by patient name
- Filter by risk level, mutation type, or severity
- Quick statistics for each filter

#### **Tab 5: Utilities**
- Export all patient data to CSV
- Generate system-wide reports
- Database refresh options

---

## 🤖 GPT-4 Chatbot Features

### How It Works
1. Analyzes patient clinical context (mutation, severity, exposure, risk)
2. Retrieves relevant conversation history
3. Sends patient context + conversation + new message to GPT-4
4. Returns clinical decision support recommendations
5. Stores all conversations in database for continuity of care

### Example Interactions
- **"What's the prognosis for this patient?"** → Analyzes risk factors and provides outlook
- **"Should we change treatment?"** → Reviews current dose, adherence, risk level, recommends adjustments
- **"What monitoring should we do?"** → Creates personalized monitoring schedule
- **"Generate clinical recommendations"** → Produces comprehensive care plan

### Clinical Safety Features
- System prompt emphasizes evidence-based medicine
- Includes disclaimers that AI supplements, not replaces, clinical judgment
- Maintains conversation history for accountability
- All responses stored in database for audit trail

---

## 💾 Database Operations

### Python API Usage

```python
from database import *

# Add a patient
patient_data = {
    'Name': 'John Doe',
    'Age': 25,
    'Severity': 'Severe',
    'Mutation': 'Intron22',
    'Risk_Score': 0.75
}
patient_id = add_patient(patient_data)

# Retrieve patient
patient = get_patient(patient_id)

# Get all patients
all_patients = get_all_patients()

# Add conversation
add_conversation(patient_id, "user message", "gpt response", "general")

# Get conversation history
history = get_conversation_history(patient_id, limit=50)

# Add doctor note
add_doctor_note(patient_id, "Dr. Smith", "Note content", "Treatment", "Important")

# Get doctor notes
notes = get_doctor_notes(patient_id)

# Get dashboard stats
stats = get_dashboard_stats()
```

### Database File
- Location: `hemophilia_clinic.db`
- Format: SQLite3
- Automatically created on first run
- Can be backed up like any regular file

---

## 🤖 GPT Chatbot API Usage

```python
from gpt_chatbot import *

# Generate response with patient context
response = create_gpt_response(
    user_message="What's the treatment plan?",
    patient_context=patient_data,
    conversation_history=past_messages
)

# Get clinical recommendations
recommendations = get_clinical_recommendations(patient_data)

# Analyze monitoring data
analysis = analyze_monitoring_data(patient_data, monitoring_records)

# Generate risk explanation
risk_explanation = generate_inhibitor_risk_explanation(
    patient_data,
    risk_factors_dict
)
```

---

## 🔧 Configuration

### Environment Variables (.env)
```env
OPENAI_API_KEY=sk-...               # Required for GPT features
APP_NAME=Hemophilia AI Platform     # Optional
DEBUG=False                          # Optional
LOG_LEVEL=INFO                       # Optional
```

### Modify Model Behavior
Edit `gpt_chatbot.py` `SYSTEM_PROMPT` to customize AI personality and guidelines.

---

## 📊 Feature Importance & Risk Factors

The ML models consider:
- **Mutation Type** (Strongest factor) - Intron22 has highest inhibitor risk
- **Severity** - Severe cases require more intensive management
- **Exposure Days** - Early exposures carry highest risk
- **Age** - Younger patients may have different risk profiles
- **Family History** - Genetic predisposition
- **Treatment Adherence** - Compliance affects outcomes
- **Baseline Factor Level** - Clinical marker
- **Immunological factors** - Infections, vaccinations
- **Lifestyle factors** - Physical activity, stress

---

## 🚨 Important Notes

### API Key Security
- Never commit `.env` file to version control
- Use `.gitignore` to exclude it
- Keep your API key private
- Regenerate if accidentally exposed

### OpenAI Costs
- GPT-4 is more expensive than GPT-3.5
- App has fallback to GPT-3.5 if GPT-4 unavailable
- Monitor your API usage at https://platform.openai.com/account/usage

### Database Backup
```bash
# Backup database
cp hemophilia_clinic.db hemophilia_clinic_backup_$(date +%Y%m%d).db

# Export to CSV
# Use Dashboard → Utilities → Export All Patients
```

---

## 🎯 Clinical Workflow

### Typical Use Case

1. **Patient Intake** (Patient Form)
   - Enter comprehensive patient data
   - Run ML risk prediction
   - View initial assessment

2. **Analysis** (Results Page)
   - Review feature importance
   - Understand risk drivers
   - Generate PDF report

3. **Clinical Consultation** (Chatbot)
   - Ask GPT-4 for clinical recommendations
   - Get inhibitor risk analysis
   - Review monitoring protocols

4. **Documentation** (Doctor Dashboard)
   - Add clinical notes
   - Track monitoring data
   - Review patient history

5. **Analytics** (Doctor Dashboard Analytics)
   - Identify trends
   - Compare populations
   - Export for research

---

## 🐛 Troubleshooting

### API Key Issues
```
Error: OpenAI API key not configured
Solution: Create .env file with OPENAI_API_KEY=your_key
```

### Database Lock
```
Error: database is locked
Solution: Close other instances accessing the database, restart app
```

### Missing Packages
```
Error: ModuleNotFoundError: No module named 'streamlit'
Solution: pip install -r requirements.txt
```

### GPT Timeout
```
Error: API request timeout
Solution: Check internet connection, RobustError likely network issue
```

---

## 📚 File Structure

```
.
├── app.py                    # Main Streamlit application
├── database.py               # Database operations and schema
├── gpt_chatbot.py           # GPT-4 integration
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
├── .env                     # Environment variables (create this)
├── hemophilia_clinic.db     # SQLite database (auto-created)
├── rf.pkl                   # Random Forest model
├── xgb.pkl                  # XGBoost model
├── columns.pkl              # Feature column names
├── patients.csv             # Historical patient CSV
├── clinical.csv             # Clinical data CSV
├── genomic.csv              # Genomic data CSV
└── README.md                # This file
```

---

## 🚀 Running the Application

### Start the app
```bash
streamlit run app.py
```

### Access in browser
- Local: `http://localhost:8501`
- Remote: Use `streamlit config` to customize

### For production deployment
```bash
# Using Streamlit Cloud, AWS, GCP, Azure, etc.
# See: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
```

---

## 📞 Support & Documentation

- **Streamlit Docs**: https://docs.streamlit.io/
- **OpenAI Docs**: https://platform.openai.com/docs/
- **SQLite Docs**: https://www.sqlite.org/docs.html
- **Scikit-learn**: https://scikit-learn.org/stable/

---

## ✅ Checklist for First-Time Setup

- [ ] Python 3.8+ installed
- [ ] Requirements installed (`pip install -r requirements.txt`)
- [ ] OpenAI account created (https://platform.openai.com)
- [ ] API key generated and added to `.env`
- [ ] `.env` file created in project root
- [ ] `.env` added to `.gitignore`
- [ ] Models trained (rf.pkl, xgb.pkl, columns.pkl exist)
- [ ] Run `streamlit run app.py`
- [ ] Test chatbot with sample patient
- [ ] Verify database created (hemophilia_clinic.db)
- [ ] Test Doctor Dashboard features

---

## 🎉 You're All Set!

Your hemophilia AI clinical platform is now ready for use. Start by:
1. Creating a test patient in Patient Form
2. Running risk prediction
3. Chatting with GPT-4 in Chatbot
4. Exploring Doctor Dashboard

Enjoy your advanced AI-powered clinical intelligence system!
