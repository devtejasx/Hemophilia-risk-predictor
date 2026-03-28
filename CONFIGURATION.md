# ⚙️ Configuration & Customization Guide

## 🔧 Environment Variables

### Required Variables
```env
OPENAI_API_KEY=sk-...  # Your OpenAI API key (required for chatbot)
```

### Optional Variables
```env
# Application Settings
APP_NAME=Hemophilia AI Platform
DEBUG=False
LOG_LEVEL=INFO

# Database
DB_PATH=hemophilia_clinic.db
DB_BACKUP=enabled

# OpenAI Settings
GPT_MODEL=gpt-4              # or create a different model preference
GPT_FALLBACK_MODEL=gpt-3.5-turbo
GPT_TEMPERATURE=0.7
GPT_MAX_TOKENS=2000
```

---

## 🤖 Customize GPT Behavior

### Edit System Prompt
File: `gpt_chatbot.py`

```python
SYSTEM_PROMPT = """You are Dr. Hemophilia, an expert AI medical assistant specializing in hemophilia and bleeding disorders.
...
"""
```

**Modify to:**
- Change personality (formal vs casual)
- Add specific guidelines
- Include institutional protocols
- Specify output format preferences

### Adjust Response Tone
```python
# In gpt_chatbot.py create_gpt_response()
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7,        # Lower = more focused, Higher = more creative
    max_tokens=2000,        # Lower = shorter responses
    top_p=0.9              # Lower = more deterministic
)
```

### Change Model
```python
# Use different model
model="gpt-3.5-turbo"      # Faster, cheaper
model="gpt-4-turbo"        # More capable
```

---

## 💾 Database Customization

### Change Database Location
File: `database.py`

```python
DB_PATH = "hemophilia_clinic.db"  # Change this path
```

### Add New Table
```python
# In init_database()
c.execute('''
    CREATE TABLE IF NOT EXISTS new_table (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        field_name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id) REFERENCES patients(id)
    )
''')
```

### Add New Database Function
```python
# File: database.py
def add_custom_record(patient_id, data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO new_table (patient_id, field_name)
        VALUES (?, ?)
    ''', (patient_id, data))
    
    conn.commit()
    conn.close()
```

---

## 🎨 Customize UI

### Change Colors
File: `app.py`

```python
# Customize risk colors
emoji_map = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MODERATE": "🟡",
    "LOW": "🟢"
}
```

### Modify Header
```python
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>Your Custom Title</h1>
        <p>Your custom subtitle</p>
    </div>
""", unsafe_allow_html=True)
```

### Add Custom CSS
```python
st.markdown("""
    <style>
        .your-class {
            color: #your-color;
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)
```

---

## 📊 Customize Dashboard

### Add New Dashboard Metric
File: `app.py` in "Doctor Dashboard" section

```python
with metric_col5:
    st.metric("New Metric", value, "description")
```

### Add New Chart
```python
# In Dashboard → Analytics tab
fig, ax = plt.subplots(figsize=(10, 5))
# Your custom chart code
st.pyplot(fig)
```

### Add New Search Filter
```python
# In Doctor Dashboard → Search tab
elif search_type == "New Filter":
    # Your filter logic
    filtered = [p for p in patients_list if p['field'] == value]
```

---

## 📋 Customize Patient Form

### Add New Form Fields
File: `app.py` in "Patient Form" section

```python
with col1:
    new_field = st.slider("New Field", min_val, max_val, default)
```

### Reorganize Form Sections
```python
# Simply reorder the st.markdown sections
st.markdown("### 🏥 New Section Title")
```

### Change Field Validation
```python
if predict_btn:
    if not name:
        st.error("❌ Please enter patient name")
    # Add your custom validations
```

---

## 🚀 Performance Optimization

### Database Optimization
```python
# Add index for faster searches
c.execute('CREATE INDEX idx_patient_name ON patients(name)')
c.execute('CREATE INDEX idx_risk_score ON patients(risk_score)')
```

### Cache Database Queries
```python
@st.cache_resource
def get_all_patients_cached():
    return get_all_patients()
```

### Limit Dashboard Data
```python
# In Doctor Dashboard, limit to recent patients
c.execute('''
    SELECT * FROM patients 
    ORDER BY created_at DESC 
    LIMIT 1000  # Limit displayed records
''')
```

---

## 🔌 Integrate External APIs

### Add EHR Integration
```python
# File: integrations/ehr.py
def sync_with_epic_ehr(patient_id):
    """Sync patient data with Epic EHR"""
    patient = get_patient(patient_id)
    # API call to Epic
    pass
```

### Add Lab Result Import
```python
def import_lab_results(file_path):
    """Import lab results from CSV"""
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        add_monitoring_record(
            row['patient_id'],
            row['test_type'],
            row['result_value'],
            row['result_status']
        )
```

---

## 📈 Add Custom Analytics

### New Chart Type
```python
# In Doctor Dashboard → Analytics
if tab == "Analytics":
    # Add your custom chart
    fig, ax = plt.subplots()
    # Your visualization code
    st.pyplot(fig)
```

### Export Custom Report
```python
def generate_custom_report(patients_list):
    """Generate specialized report"""
    df = pd.DataFrame(patients_list)
    # Custom processing
    df.to_excel("custom_report.xlsx")
    return "custom_report.xlsx"
```

---

## 💬 Customize Chatbot Behavior

### Change Clinical Focus
```python
# Add to SYSTEM_PROMPT
"Focus on: [Your priority areas]"
"Available guidelines: [Your institutional guidelines]"
```

### Add Structured Output
```python
prompt = f"""
Provide response in this format:
1. Clinical Assessment:
2. Recommendations:
3. Monitoring Plan:
4. Follow-up:
"""
```

### Implement RAG (Retrieval Augmented Generation)
```python
def gpt_with_documents(user_query, documents):
    """Use GPT with document context"""
    context = "\n".join(documents)
    prompt = f"Context: {context}\n\nQuestion: {user_query}"
    # Send to GPT with context
```

---

## 🔐 Security Customization

### Add Authentication
```python
# File: auth.py
def check_login():
    if 'logged_in' not in st.session_state:
        st.error("Please log in")
        st.stop()
```

### Audit Logging
```python
def log_action(user, action, patient_id):
    """Log all important actions"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"{user} - {action} - patient {patient_id}")
```

### Data Encryption
```python
from cryptography.fernet import Fernet

def encrypt_patient_data(data):
    """Encrypt sensitive patient data"""
    key = Fernet.generate_key()
    cipher = Fernet(key)
    encrypted = cipher.encrypt(data.encode())
    return encrypted
```

---

## 📱 Mobile Responsiveness

### Adjust Layout for Mobile
```python
# Different layouts for different screen sizes
import streamlit as st

if st.session_state.get('mobile'):
    col1, col2 = st.columns(1)  # Stack instead of side-by-side
else:
    col1, col2 = st.columns(2)  # Side by side
```

---

## 🧪 Development Environment

### Enable Debug Mode
```env
DEBUG=True
LOG_LEVEL=DEBUG
```

### Add Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug(f"Patient data: {patient_data}")
```

### Run Tests
```bash
pytest tests/
pytest tests/ -v  # Verbose
pytest tests/ --cov  # With coverage
```

---

## 📦 Version Management

### Update Requirements
```bash
# Update all packages
pip install -r requirements.txt --upgrade

# Save new versions
pip freeze > requirements.txt
```

### Database Migration
```python
def migrate_v1_to_v2():
    """Migrate from v1 schema to v2"""
    # Backup old database
    import shutil
    shutil.copy('hemophilia_clinic.db', 'hemophilia_clinic_v1_backup.db')
    
    # Add new columns or tables
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('ALTER TABLE patients ADD COLUMN new_field TEXT')
    conn.commit()
    conn.close()
```

---

## 🎓 Best Practices

### Always Backup Database
```bash
cp hemophilia_clinic.db hemophilia_clinic_backup_$(date +%s).db
```

### Test Changes on Development
```bash
# Use test database
DB_PATH = "test_hemophilia.db"
python tests/test_app.py
```

### Document Custom Changes
```python
# Always add comments
# Custom change by [Your Name] - [Date]
# Reason: [Why this change was needed]
# Impact: [What this affects]
```

### Version Control
```bash
# Keep .env out of git
echo ".env" >> .gitignore

# Commit documentation
git add *.md
git commit -m "Update documentation"
```

---

## 📞 Getting Help

- Check documentation in README.md
- Review code comments
- Check examples.py
- Test on small dataset first
- Keep backups before major changes

---

## ✅ Configuration Checklist

- [ ] Set OpenAI API key in .env
- [ ] Test database operations
- [ ] Customize system prompt if needed
- [ ] Test dashboard features
- [ ] Backup original database
- [ ] Document any customizations
- [ ] Test with sample patient
- [ ] Verify all pages load correctly

