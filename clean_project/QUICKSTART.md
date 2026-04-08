# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Prerequisites
- Python 3.9+
- pip or conda
- Virtual environment (recommended)

### Option 1: Fastest (Pre-configured)

```bash
# Navigate to project
cd clean_project

# Copy environment template
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

✅ App opens at: http://localhost:8501

### Option 2: With Virtual Environment

```bash
# Create environment
cd clean_project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install & run
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

### Option 3: Docker (Recommended for Production)

```bash
# Simple version
docker-compose up

# Production build
docker build -t hemophilia-app .
docker run -p 8501:8501 hemophilia-app
```

✅ App opens at: http://localhost:8501

## 📝 First Steps

### 1. Create an Account
- Click "Register"
- Choose username, email, password
- Click "Register"

### 2. Add a Patient
- Go to Dashboard
- Fill in patient information:
  - Name
  - Age
  - Gender
  - Clotting factor %
  - Activity level
  - Treatment compliance %
  - Recent bleeds
- Click "Submit"

### 3. Get Risk Prediction
- Go to Predictions
- Select the patient you added
- Click "Calculate Risk"
- View the risk score and explanation

### 4. Ask Chat Questions
- Go to Chat
- Ask: "How do I manage bleeds?"
- Get clinical guidance

### 5. View Analytics
- Go to Analytics
- See patient population insights
- Monitor key metrics

## 🎮 Key Features

### Dashboard
- KPI metrics at a glance
- Add new patients quickly
- View patient list
- Quick access to all features

### Predictions
- Select any patient
- Get AI risk assessment
- See feature contributions
- Understand why (SHAP explanation)

### Chat
- Ask clinical questions
- Get evidence-based replies
- Integrated patient context
- Clinical knowledge base

### Analytics
- Risk distribution charts
- Patient statistics
- Trend analysis
- Aggregate metrics

### Settings
- Dark mode toggle
- Account management
- Logout option

## 🛠️ Configuration

### .env Variables (Most Important)

```bash
# Database
DATABASE_URL=sqlite:///app_data.db

# Optional: External services
OPENAI_API_KEY=your-key-here  # For advanced chat
ENVIRONMENT=development       # or production
```

See `.env.example` for complete list.

## 📂 Project Structure (Quick)

```
clean_project/
├── app.py              ← Run this!
├── components/         ← UI widgets
├── services/          ← Business logic
├── utils/             ← Helpers
├── database.py        ← Data storage
├── config.py          ← Configuration
└── requirements.txt   ← Dependencies
```

Full details: See `PROJECT_STRUCTURE.md`

## 🔍 Common Tasks

### Add Different Patient Types

```python
# In a Python script
from services import predict_risk

# Create patient data
patient = {
    "age": 35,
    "clotting_factor": 45,
    "activity_level": 6,
    "compliance": 0.75,
    "bleeds": 1,
    "hospitalization": False,
}

# Get prediction
result = predict_risk(patient)
print(f"Risk: {result['risk_score']:.1%}")
```

### Use Chat Programmatically

```python
from services import get_response

response = get_response("What factors affect hemophilia?")
print(response)
```

### Access Database

```python
from database import get_database

db = get_database()
patients = db.get_patients(user_id=1)

for patient in patients:
    print(f"{patient['name']}: {patient['age']} years")
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Make sure you're in correct directory
cd clean_project

# Reinstall dependencies
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print(sys.path)"
```

### Issue: "Port 8501 already in use"
```bash
# Change port in command
streamlit run app.py --server.port 8502

# Or kill existing process (Mac/Linux)
lsof -ti:8501 | xargs kill -9
```

### Issue: "Database locked"
```bash
# Delete and recreate database
rm app_data.db
python -c "from database import get_database; get_database()"
```

### Issue: "Chart not showing"
```bash
# Check plotly is installed
pip install --upgrade plotly

# Restart streamlit
# Ctrl+C then streamlit run app.py
```

## 📚 Documentation Files

| File | Purpose | Read Time |
|------|---------|-----------|
| README.md | Complete guide | 10 min |
| REFACTORING_SUMMARY.md | What changed | 5 min |
| PROJECT_STRUCTURE.md | Code organization | 15 min |
| This file | Quick start | 3 min |

## 🎯 Next Steps

### For Users
1. ✅ Get app running (this guide)
2. Add your patients
3. Generate predictions
4. Ask questions via chat
5. Monitor analytics

### For Developers
1. Understand structure (PROJECT_STRUCTURE.md)
2. Explore components/ (UI code)
3. Explore services/ (Business logic)
4. Read docstrings in each module
5. Modify and extend as needed

### For Deployment
1. Use provided Docker setup
2. Configure .env for production
3. Set up database backup
4. Enable logging
5. Monitor performance

## 💡 Tips & Tricks

### Streamlit Features
- Press `R` to rerun
- Press `C` to clear cache
- Use Ctrl+Shift+P for options
- Hold Shift-click to select range

### Faster Development
```bash
# Use streamlit watch for auto-reload
streamlit run app.py --logger.level=warning
```

### Dark Mode
- Toggle in Settings (bottom left)
- Automatically saves preference

### Export Data
- Chat history: Exported automatically to database
- Predictions: Saved with timestamp
- Patients: Stored in database

## 📞 Getting Help

### Documentation
1. Check README.md (comprehensive)
2. Check PROJECT_STRUCTURE.md (architecture)
3. Read function docstrings (in code)

### Debug Info
```bash
# Show streamlit config
streamlit config show

# Check Python version
python --version

# Verify imports
python -c "from clean_project import *"
```

### Common Solutions
- **App won't start**: Check port availability
- **Data not saving**: Check database file permissions
- **Predictions failing**: Check that patient data is valid
- **Chat not responding**: Check OpenAI key if configured

## 🎓 Learning Resources

### Understanding the Code
1. Start with `app.py` - see overall flow
2. Look at `components/` - see how UI is built
3. Look at `services/` - understand business logic
4. Look at `utils/` - see helper patterns

### Python Concepts Used
- Classes and methods
- Decorators (for caching)
- Generators (for iteration)
- Context managers (for database)
- Type hints (for documentation)

### Streamlit Concepts
- `st.session_state` - persistent state
- `st.cache_data` - memoization
- `@st.session_state` - state access
- `st.columns()` - layout
- `st.form()` - grouped inputs

## ⚡ Performance Tips

### For Faster Load Times
```bash
# Run with minimal logging
streamlit run app.py --logger.level=error

# Use production config
export ENVIRONMENT=production
```

### For Faster Development
```bash
# Keep only essential imports at top
# Use lazy imports inside functions
# Run without full dataset initially

# Example: Load smaller dataset for testing
if ENVIRONMENT == "development":
    SAMPLE_SIZE = 100
else:
    SAMPLE_SIZE = 10000
```

## 🔐 Security Notes

### In Development
- ✅ .env.example provided
- ✅ Passwords are validated
- ✅ Session tokens used
- ⚠️ SECRET_KEY in code (change in .env)

### For Production
- ✅ Use strong SECRET_KEY
- ✅ Enable HTTPS
- ✅ Use environment variables
- ✅ Regular security updates
- ✅ Monitor access logs

See `config.py` for security settings.

## 📊 Data

### Sample Patient Data
```python
sample_patient = {
    "name": "John Doe",
    "age": 45,
    "gender": "Male",
    "clotting_factor": 50,
    "activity_level": 5,
    "compliance": 0.8,
    "bleeds": 2,
    "hospitalization": False,
    "notes": "Compliant with treatment"
}
```

### Risk Score Interpretation
- **0.0-0.4**: Low risk ✅
- **0.4-0.7**: Medium risk ⚠️
- **0.7-1.0**: High risk 🔴

## 🎉 Success!

You're now running the clean, modular Hemophilia Clinical Decision Support System!

### What You Have
✓ Fully functional web application
✓ AI predictions for risk assessment
✓ Clinical chatbot for guidance
✓ Secure data storage
✓ Beautiful dashboard
✓ Production-ready code

### What You Can Do
✓ Add unlimited patients
✓ Get risk predictions
✓ Ask clinical questions
✓ Monitor analytics
✓ Customize for your needs
✓ Deploy to production

### Next?
- Read full documentation
- Customize colors/settings
- Add your own patient data
- Extend with new features
- Deploy to cloud

---

**Need help?** Check README.md or PROJECT_STRUCTURE.md

**Want to contribute?** Follow the modular structure and add in appropriate location.

**Ready to deploy?** See DEPLOYMENT.md (in parent README)

**Happy coding!** 🚀
