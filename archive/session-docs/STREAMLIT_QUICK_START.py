"""
STREAMLIT MEDICAL APP - QUICK START GUIDE

Complete multi-page medical AI interface with:
- Patient Form (new patient entry & selection)
- Results Dashboard (risk predictions & SHAP explanations)
- Patient History (historical records & trends)
- AI Chatbot (conversational interface)
- Doctor Dashboard (system analytics)
"""

# ============================================================================
# INSTALLATION & SETUP
# ============================================================================

## Step 1: Install Dependencies

```bash
pip install streamlit streamlit-option-menu plotly pandas requests numpy
```

Optional (for additional features):
```bash
pip install openpyxl python-dotenv streamlit-lottie
```

## Step 2: Ensure FastAPI Backend is Running

The Streamlit app requires a FastAPI backend running on `http://localhost:8000`.

If you don't have a backend running yet:
```bash
# Terminal window 1: Start the FastAPI backend
python api.py  # or your FastAPI app file
```

## Step 3: Run the Streamlit App

```bash
# Terminal window 2: Start Streamlit
streamlit run streamlit_medical_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

# ============================================================================
# APP STRUCTURE
# ============================================================================

```
Medical AI System/
├── streamlit_medical_app.py          # Main entry point (no changes needed)
├── streamlit_pages/                  # Page implementations
│   ├── __init__.py
│   ├── patient_form.py               # Patient input/selection
│   ├── results_dashboard.py          # Risk & SHAP visualizations
│   ├── patient_history.py            # Historical data & trends
│   ├── ai_chatbot.py                 # Chat interface
│   └── doctor_dashboard.py           # Analytics dashboard
└── streamlit_utils/                  # Utility modules
    ├── __init__.py
    ├── state_manager.py              # Session state management
    ├── ui_components.py              # Reusable UI components
    ├── backend_client.py             # FastAPI communication
    └── plotly_charts.py              # Medical visualizations
```

# ============================================================================
# PAGE GUIDE
# ============================================================================

## 1️⃣ Patient Form (🔬 Patient Form)

### New Patient Tab
- Enter comprehensive patient information
- Fields: Personal info, Medical history, Vitals, Labs, Notes
- Validation ensures all required fields are complete
- Option to save to backend or preview data

### Select Existing Patient Tab
- Load previously created patients from backend
- View patient summary
- Click "Load Patient" to work with that patient

### Quick Assessment Tab
- Minimal fields for rapid risk assessment
- Age, Gender, Vital signs, Basic measurements
- Direct prediction without full form

## 2️⃣ Results Dashboard (📊 Results)

### Risk Score Tab
- Large risk gauge visualization (0-100%)
- Color-coded risk levels:
  - 🟢 Low (0-40%)
  - 🟡 Moderate (40-60%)
  - 🟠 High (60-75%)
  - 🔴 Critical (>75%)
- Model accuracy and version info
- Confidence score

### Explainability Tab
- SHAP waterfall plot showing feature contributions
- Feature importance bar chart (top 15 features)
- Red bars = increase risk, Blue bars = decrease risk
- Plain English explanations

### Trends Tab
- Historical predictions for patient
- Risk trend line chart
- Peak/dip analysis
- Statistics (average, highest, change)

### Recommendations Tab
- Priority-based clinical recommendations
- Different action items based on risk level
- Treatment suggestions
- Follow-up schedule

## 3️⃣ Patient History (📋 History)

### Records Table Tab
- All historical records for patient
- Filtering by: Record type, Date range, Sort order
- Pagination with customizable page size
- Summary statistics

### Risk Timeline Tab
- Line chart of risk scores over time
- Notable events (highest/lowest risk)
- Timeline analysis

### Analysis Tab
- Risk distribution pie chart
- Statistical breakdowns
- Trend indicators (improving/worsening)
- Percentile analysis

### Export Tab
- Download as CSV
- Download as JSON
- Download as Excel
- Get export summary info

## 4️⃣ AI Chatbot (💬 Chatbot)

### Chat Tab
- Message history display
- Type questions about patient
- Get AI-powered responses
- Clear chat history option

### Examples Tab
- Patient information questions
- Clinical analysis questions
- Historical analysis questions
- Click any example to auto-send

Pre-configured questions:
- "What is the patient's current medical status?"
- "What are the main risk factors?"
- "What clinical interventions are recommended?"
- "What trends do you see in history?"

## 5️⃣ Doctor Dashboard (👨‍⚕️ Doctor Dashboard)

### Overview Tab
- Key system metrics (total patients, high-risk count, avg risk)
- System health status (API, database, model)
- Today's summary (new patients, predictions, alerts)

### Risk Distribution Tab
- Pie chart of patient risk distribution
- Risk level breakdown with percentages
- Action items for each risk level

### Trends Tab
- Select time period (Week/Month/Quarter)
- Predictions per day chart
- Average risk trend line
- Key statistics

### Cohort Analysis Tab
- By Age Group
- By Diagnosis
- By Risk Level
- By Gender
- Detailed cohort tables

### System Status Tab
- API Server status
- ML Model status
- Database connection
- Application info
- Refresh button

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

**User Role Selector** (Top of sidebar)
- Select: Clinician, Patient, Administrator
- Controls dashboard visibility

**Current Patient** (Sidebar display)
- Shows active patient ID
- Updated when you load/create patient

**Backend Status** (Sidebar indicator)
- 🟢 Online - Backend responding
- 🔴 Offline - Cannot reach backend

# ============================================================================
# KEY FEATURES
# ============================================================================

### Session State Management
- Patient data persists across page navigation
- Prediction results cached in memory
- Chat history maintained throughout session
- Filters and date ranges remembered

### Professional UI
- Dark theme with cyan accents
- Medical color scheme (risk indicators)
- Responsive design
- Loading spinners and progress indicators

### Backend Integration
- Automatic health checks
- Error handling with user feedback
- Timeout management (30 seconds)
- Batch operations support

### Data Visualizations
- Risk gauge with color zones
- Trend time series
- SHAP feature contributions
- Risk distribution charts
- Correlation heatmaps
- Feature importance plots

### Medical Features
- Risk stratification
- Vital signs management
- Laboratory value tracking
- Comorbidity tracking
- Severity assessment
- Clinical recommendations

# ============================================================================
# USAGE WORKFLOWS
# ============================================================================

## Workflow 1: Assess New Patient
1. Navigate to 🔬 Patient Form
2. Select "New Patient" tab
3. Fill in all fields
4. Click "Save Patient"
5. Navigate to 📊 Results
6. Click "Generate Prediction"
7. Review risk score and SHAP explanations

## Workflow 2: Review Existing Patient
1. Navigate to 🔬 Patient Form
2. Select "Select Existing" tab
3. Choose patient from list
4. Navigate to 📊 Results
5. View risk score and trends

## Workflow 3: Analyze Patient History
1. Ensure patient is loaded
2. Navigate to 📋 History
3. Apply filters (date range, record type)
4. View records table with pagination
5. Analyze trends in "Trends" tab
6. Export data if needed

## Workflow 4: Ask Questions About Patient
1. Ensure patient is loaded
2. Navigate to 💬 Chatbot
3. Type question or click example
4. View AI response
5. Follow-up questions will maintain context

## Workflow 5: View System Analytics
1. Set user role to Clinician/Administrator
2. Navigate to 👨‍⚕️ Doctor Dashboard
3. View Overview tab for key metrics
4. Analyze risk distribution
5. Review trends and cohorts
6. Check system status

# ============================================================================
# CUSTOMIZATION
# ============================================================================

### Change Backend URL
Edit `streamlit_utils/backend_client.py`:
```python
backend = get_backend_client(base_url="http://new-url:port")
```

### Change Colors
Edit `streamlit_utils/plotly_charts.py`:
```python
COLORS = {
    "risk_critical": "#your-color",
    "risk_high": "#your-color",
    ...
}
```

### Modify Page Timeout
Edit `streamlit_utils/backend_client.py`:
```python
self.timeout = 60  # Change from 30 seconds
```

### Add New Page
1. Create `streamlit_pages/new_page.py`
2. Implement `render()` function
3. Import in `streamlit_medical_app.py`
4. Add to page_map dictionary
5. Add to sidebar options

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

### Backend Connection Issues
- [ ] Is FastAPI backend running? (`python api.py`)
- [ ] Is it on localhost:8000? (check backend settings)
- [ ] Any firewall blocking localhost?
- [ ] Try `curl http://localhost:8000/api/health`

### Missing Patient Data
- [ ] Did you save the patient? (click Save Patient button)
- [ ] Is backend running? (check status indicator)
- [ ] Try selecting from existing patients list

### SHAP Explanation Not Showing
- [ ] Click "Generate SHAP Explanation" button
- [ ] Wait for computation to complete
- [ ] Check that prediction was generated first

### Export Not Working
- [ ] Make sure records are loaded
- [ ] Check that openpyxl is installed (for Excel)
- [ ] Try CSV export as fallback

### Session State Issues
- [ ] Clear browser cache and refresh
- [ ] Reload page (F5)
- [ ] Try incognito/private browsing
- [ ] Click "Clear Chat History" (if in chatbot)

# ============================================================================
# API ENDPOINTS REQUIRED
# ============================================================================

The backend should implement these endpoints:

```
Health Check
- GET /api/health

Predictions
- POST /api/predict            # Single prediction
- POST /api/predict/batch      # Batch predictions

Explainability
- POST /api/explain            # SHAP explanation

Patient Management
- GET /api/patients            # List all patients
- GET /api/patients/{id}       # Get specific patient
- POST /api/patients           # Create patient
- PUT /api/patients/{id}       # Update patient
- GET /api/patients/{id}/history  # Patient history

Chat
- POST /api/chat               # Send message to chatbot
- GET /api/chat/history        # Get chat history

Analytics
- GET /api/analytics/dashboard    # System statistics
- GET /api/analytics/risk-distribution  # Risk data
- GET /api/analytics/trends       # Trend data

Model Info
- GET /api/model/info          # Model information
- GET /api/model/features      # Feature importance
```

# ============================================================================
# KEYBOARD SHORTCUTS
# ============================================================================

- `R` - Rerun page
- `C` - Open command palette
- `K` - Keyboard help
- `Ctrl+P` - Page options
- `F5` - Full reload

# ============================================================================
# TIPS & TRICKS
# ============================================================================

1. **Quick Assessment**: Use the Quick Assessment tab for rapid predictions
2. **Dark Mode**: Already enabled by default with medical theme
3. **Data Export**: Export to CSV for use in Excel/R/Python
4. **Chat Context**: Chatbot automatically uses loaded patient data
5. **History Analysis**: Use date range filters for specific periods
6. **Bulk Import**: Create multiple patients in quick succession
7. **Dashboard Refresh**: Click refresh button to get latest stats
8. **Sidebar Collapse**: Click sidebar toggle to expand workspace

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

- Patient list cached (limit 100 by default)
- Backend responses cached with @st.cache_resource
- Chart rendering optimized for large datasets
- Pagination reduces data transfer
- Lazy loading on navigation

# ============================================================================
# SUPPORT & DOCUMENTATION
# ============================================================================

For detailed information on:
- **Backend Integration**: See backend_client.py docstrings
- **Component Usage**: See ui_components.py docstrings
- **State Management**: See state_manager.py docstrings
- **Chart Types**: See plotly_charts.py docstrings

# ============================================================================
# VERSION INFORMATION
# ============================================================================

- App Version: 1.0.0
- Streamlit: 1.25.0+
- Python: 3.8+
- Plotly: 5.0+
- FastAPI: 0.95+ (backend requirement)

# ============================================================================
# END OF QUICK START GUIDE
# ============================================================================
"""

print(__doc__)
