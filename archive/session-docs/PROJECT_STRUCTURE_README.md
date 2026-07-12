# Project README and Setup Guide

# 🏥 Hemophilia AI Platform - Professional Multi-Page Streamlit App

A production-ready, scalable Streamlit application for clinical risk assessment and patient management in hemophilia care.

## 📁 Project Structure

```
hemophilia-ai/
│
├── app.py                          # Main entry point - home page
│
├── pages/                          # Multi-page app (auto-discovered by Streamlit)
│   ├── 1_Dashboard.py             # Patient overview & metrics
│   ├── 2_Add_Patient.py           # Patient data entry form
│   ├── 3_Predictions.py           # ML risk assessment
│   ├── 4_SHAP_Explainability.py   # Model interpretability
│   ├── 5_Chatbot.py               # AI assistant interface
│   └── 6_Analytics.py             # Advanced analytics & reporting
│
├── components/                     # Reusable UI components
│   ├── __init__.py
│   ├── navbar.py                  # Sidebar navigation & branding
│   ├── cards.py                   # UI card components
│   └── charts.py                  # Visualization utilities
│
├── services/                       # Business logic & external services
│   ├── __init__.py
│   ├── ml_service.py              # ML model predictions
│   ├── chatbot_service.py         # AI chatbot logic
│   └── api_client.py              # Backend API client
│
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── helpers.py                 # General helper functions
│   └── session_state.py           # Cross-page state management
│
├── database/                       # Data persistence
│   ├── __init__.py
│   └── db.py                     # SQLite database wrapper
│
├── styles/                         # Styling & theming
│   ├── __init__.py
│   └── css.py                    # Centralized CSS theming
│
└── assets/                         # Static files, images, etc.
```

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone or navigate to project directory
cd hemophilia-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install streamlit pandas numpy matplotlib scikit-learn xgboost joblib shap plotly seaborn requests
```

### 2. **Run the App**

```bash
# Start the Streamlit server
streamlit run app.py

# Open browser to http://localhost:8501
```

### 3. **Access Features**

- **📊 Dashboard**: View patient overview and key metrics
- **👤 Add Patient**: Enter new patient clinical data
- **🔮 Predictions**: Get ML-powered risk assessments
- **🧠 SHAP**: Understand model decisions
- **🤖 Chatbot**: Ask clinical questions
- **📈 Analytics**: Deep data analysis and reporting

## 🛠️ Architecture Overview

### Frontend Layer (Streamlit)
- **app.py**: Entry point & home page
- **pages/**: 6 dedicated pages for different features
- **components/**: Reusable UI elements (navbar, cards, charts)

### Business Logic Layer
- **services/**: ML inference, chatbot, API calls
- **database/**: Data persistence layer

### Utilities Layer
- **utils/session_state.py**: Cross-page state management
- **utils/helpers.py**: Common helper functions
- **styles/css.py**: Centralized theming

## 📊 Page Descriptions

### 1_Dashboard.py
- **Purpose**: Patient overview and key metrics
- **Features**:
  - Total patient count
  - Average risk scores
  - High-risk patient identification
  - Recent patients table
  - Severity and risk distribution charts

### 2_Add_Patient.py
- **Purpose**: Patient data entry and management
- **Features**:
  - 16-field patient form
  - Automatic risk calculation
  - CSV persistence
  - Validation and error handling
  - Severity guidelines

### 3_Predictions.py
- **Purpose**: ML-powered risk assessment
- **Features**:
  - Ensemble predictions (RF + XGB)
  - Risk gauge visualization
  - Feature importance analysis
  - Clinical recommendations
  - Risk-based alerts

### 4_SHAP_Explainability.py
- **Purpose**: Model interpretability
- **Features**:
  - **Basic View**: Feature importance chart + top factors
  - **Advanced View**: 
    - Summary plot
    - Waterfall plot (prediction breakdown)
    - Dependence plot (feature relationships)

### 5_Chatbot.py
- **Purpose**: AI-powered clinical assistant
- **Features**:
  - Natural language Q&A
  - Conversation history
  - Quick command buttons
  - Multi-turn dialogue support

### 6_Analytics.py
- **Purpose**: Advanced analytics and reporting
- **Features**:
  - Dynamic filtering (severity, risk range, age)
  - Summary statistics
  - Data visualizations
  - CSV export functionality
  - Responsive data tables

## 🔐 Session State Management

The app uses Streamlit's `st.session_state` for cross-page data sharing:

```python
st.session_state = {
    'authenticated': bool,
    'user': dict,
    'theme': 'dark' | 'light',
    'patient_data': dict,          # Shared across pages
    'prediction_result': dict,       # Shared across pages
    'shap_view': 'Basic' | 'Advanced',
    'chat_history': list,
    'analytics_filters': dict,
}
```

## 🎨 Styling & Theming

### Dark Mode (Default)
- Primary: `#00d4ff` (cyan)
- Secondary: `#0099ff` (blue)
- Background: `#0a0e27` (dark blue)
- Text: `#ffffff` (white)

### Light Mode
- Primary: `#0066cc` (blue)
- Background: `#ffffff` (white)
- Text: `#333333` (dark gray)

### Customize Theme
Edit `styles/css.py`:
```python
def apply_theme(theme: str = "dark") -> None:
    # Modify CSS variables in :root selector
```

## 🤖 ML Models

### Integrated Models
1. **Random Forest (rf.pkl)**
   - Feature count: Dynamic (based on training data)
   - Output: Probability score (0-1)

2. **XGBoost (xgb.pkl)**
   - Feature count: Dynamic
   - Output: Risk score (0-1)

### Ensemble Approach
- **Ensemble Score** = (RF_score + XGB_score) / 2
- **Confidence** = Model-based uncertainty

### Load Custom Models
1. Place model files in project root:
   - `rf.pkl` (Random Forest)
   - `xgb.pkl` (XGBoost)
   - `columns.pkl` (Feature names)

2. Update `services/ml_service.py`:
```python
def _load_models(self) -> None:
    # Models auto-loaded from current directory
    self.models['rf'] = joblib.load("rf.pkl")
    self.models['xgb'] = joblib.load("xgb.pkl")
```

## 🗄️ Database

### SQLite Tables
1. **patients** - Patient demographics & clinical data
2. **predictions** - Model predictions & history
3. **chat_history** - Chatbot conversation logs

### Initialize DB
```python
from database.db import get_database

db = get_database()
patients = db.get_patients()
```

## 🔧 Configuration

### Environment Variables (Optional)
Create `.env` file:
```
API_BASE_URL=http://localhost:8000
LOG_LEVEL=INFO
DATABASE_PATH=hemophilia.db
```

### Streamlit Config
File: `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#00d4ff"
backgroundColor = "#0a0e27"
secondaryBackgroundColor = "#1a1f3a"
textColor = "#ffffff"
font = "sans serif"

[client]
showErrorDetails = true

[logger]
level = "info"
```

## 📦 Dependencies

```
streamlit==1.28+
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
xgboost>=1.5.0
joblib>=1.0.0
shap>=0.40.0
plotly>=5.0.0
seaborn>=0.11.0
requests>=2.25.0
python-dotenv>=0.19.0
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production (Streamlit Cloud)
1. Push to GitHub
2. Connect GitHub repo to Streamlit Cloud
3. Deploy with auto-sync

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

```bash
# Build & run
docker build -t hemophilia-app .
docker run -p 8501:8501 hemophilia-app
```

## 📝 Development Guide

### Adding a New Page
1. Create `pages/N_PageName.py`
2. Import components from their modules:
```python
from components.navbar import show_page_header
from components.cards import metric_card
```
3. Use session state for data sharing:
```python
from utils.session_state import get_session_var, set_session_var
```
4. Streamlit automatically adds to sidebar nav

### Adding a New Component
1. Create function in `components/cards.py` or new file
2. Use reusable HTML/CSS styling:
```python
def my_component(title: str) -> None:
    st.markdown(f"""
    <div class='custom-card'>
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)
```

### Adding a New Service
1. Create `services/service_name.py`
2. Implement service class with public methods
3. Provide singleton getter:
```python
def get_service() -> ServiceClass:
    global _service
    if _service is None:
        _service = ServiceClass()
    return _service
```
4. Import in pages:
```python
from services.service_name import get_service
service = get_service()
```

## 🐛 Troubleshooting

### Pages Not Showing in Sidebar
- Ensure files in `pages/` directory start with number: `1_Name.py`
- Check that `st.set_page_config()` is in `app.py` (first command)

### Slow Page Loading
- Update cache time or disable cache:
```python
@st.cache_resource(ttl=3600)  # 1 hour
def load_data():
    ...
```

### CSS Not Applied
- Clear Streamlit cache: `streamlit cache clear`
- Restart app: Ctrl+C then `streamlit run app.py`
- Check CSS syntax in `styles/css.py`

### Model Not Loading
- Verify model files exist: `rf.pkl`, `xgb.pkl`, `columns.pkl`
- Check file paths in `services/ml_service.py`
- Use fallback prediction if models unavailable

## 📚 API Reference

### Session State Functions
```python
# Initialize all defaults
from utils.session_state import init_session_state

# Get/set variables
from utils.session_state import get_session_var, set_session_var

# Update specific data
from utils.session_state import update_patient_data, update_prediction_result

# Theme management
from utils.session_state import toggle_theme
```

### Component Functions
```python
from components.cards import metric_card, info_card, patient_card
from components.navbar import show_page_header
from components.charts import plot_risk_gauge, plot_feature_importance
```

### Service Functions
```python
from services.ml_service import get_ml_service
from services.chatbot_service import get_chatbot_service
from database.db import get_database
```

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes following project structure
3. Test locally: `streamlit run app.py`
4. Commit & push:  `git push origin feature/your-feature`
5. Create pull request

## 📋 Best Practices

1. **Modular Design**: Each page is independent
2. **Reusable Components**: Use components/ for UI  
3. **Session State**: Share data via st.session_state
4. **Error Handling**: Graceful fallbacks for missing data
5. **Logging**: Use logger for debugging
6. **Documentation**: Add docstrings to functions
7. **Performance**: Cache expensive operations
8. **Styling**: Centralized CSS in styles/

## 📄 License

MIT License - See LICENSE file

## 🆘 Support

For issues and questions:
- Check troubleshooting section
- Review existing GitHub issues
- Create new issue with details

## 📞 Contact

- Email: support@hemophilia-ai.com
- GitHub: https://github.com/your-repo

---

**Version**: 3.0  
**Last Updated**: April 2024  
**Status**: Production Ready ✅
