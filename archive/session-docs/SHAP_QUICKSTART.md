# SHAP Explainability & PDF Reports - Quick Start

## ⚡ 5-Minute Setup

### 1. Install Dependencies

```bash
pip install shap reportlab plotly pandas numpy scikit-learn joblib streamlit
```

### 2. Verify Model Files

```bash
# Should exist:
ls -la rf.pkl                    # Or your model file
ls -la background_data.pkl       # Optional, for SHAP
```

### 3. Run Streamlit App

```bash
cd pages/
streamlit run shap_explainability.py
```

Visit: `http://localhost:8501`

---

## 🎯 Common Tasks

### Task 1: Get Prediction with SHAP Explanation

```python
from backend.services.prediction import PredictionService
import numpy as np

# Initialize
service = PredictionService("rf.pkl", explainability_enabled=True)

# Make prediction
features = np.array([[14.0, 7.5, 250.0, 90, 1, 0, 0, 0]])
result = service.predict_with_explanation(features)

# Access results
print(f"Risk Score: {result['prediction']}")
print(f"Top Factors: {result['explanation']['top_positive_contributors']}")
```

### Task 2: Generate PDF Report

```python
# After prediction
pdf_bytes, report_data = service.generate_full_report(
    patient_data={"patient_id": "P001", "name": "John Doe"},
    features=features,
    include_visualizations=True
)

# Save PDF
with open("report.pdf", "wb") as f:
    f.write(pdf_bytes)
```

### Task 3: Visualize Feature Importance

```python
from backend.ui_components import ExplainabilityUI
import streamlit as st

# In Streamlit app
explanation = result['explanation']
ExplainabilityUI.display_feature_importance(
    explanation['feature_contributions']
)
```

### Task 4: Batch Process Patients

```python
# CSV file with patient data
df = pd.read_csv("patients.csv")

# Get predictions
results = []
for idx, row in df.iterrows():
    features = row[['hemoglobin', 'white_cells', 'platelets', ...]].values
    result = service.predict_with_explanation(features)
    results.append(result)

# Export
df_results = pd.DataFrame([
    {
        "patient_id": r.get("patient_id"),
        "risk_score": r["prediction"]
    }
    for r in results
])
df_results.to_csv("results.csv")
```

---

## 📊 What You'll See

### In Streamlit UI:
- 🎨 Colorful risk gauge (Red/Orange/Green)
- 📈 Feature importance bar chart
- 📋 Contributing factors table
- ✅ Clinical recommendations
- 📥 Download PDF button

### In Generated Reports:
- Patient demographics
- Risk assessment with visual indicators
- Key contributing factors table
- Clinical recommendations
- SHAP visualization plots
- Professional formatting

---

## 🔍 Understanding SHAP Values

Each feature gets a **SHAP value** that shows how much it contributes to the prediction:

```
Prediction = Base Value + Sum(SHAP Values)

Example:
50% (base) + 20% (inhibitor) - 8% (platelets) + 13% (other) = 75% risk
```

**Positive SHAP**: Feature increases risk ↑
**Negative SHAP**: Feature decreases risk ↓

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `backend/services/explainability.py` | SHAP engine |
| `backend/services/reports.py` | PDF generation |
| `backend/services/prediction.py` | Orchestration |
| `backend/ui_components.py` | Streamlit components |
| `pages/shap_explainability.py` | Main UI |

---

## ✅ Troubleshooting

**Q: SHAP not explaining my predictions?**
A: Ensure model is tree-based (XGBoost, RandomForest) or provide background data

**Q: PDF looks ugly?**
A: Check ReportLab installation: `pip install --upgrade reportlab`

**Q: Slow predictions?**
A: SHAP calculation is normal (~200ms). Use batch processing for multiple patients

**Q: Missing visualizations?**
A: Matplotlib might need backend config. Add to top of script:
```python
import matplotlib
matplotlib.use('Agg')
```

---

## 🚀 Next Steps

1. ✅ Run Streamlit app and test with sample patient
2. ✅ Generate a PDF report
3. ✅ Upload CSV file for batch predictions
4. ✅ Review SHAP explanations
5. ✅ Integrate into your workflow

---

## 💡 Pro Tips

### Tip 1: Set Feature Names for Better Readability
```python
service.set_feature_names([
    "Hemoglobin", "White Blood Cells", "Platelets",
    "Treatment Adherence", "Recent Bleeds", ...
])
```

### Tip 2: Cache Model for Performance
```python
@st.cache_resource
def load_service():
    return PredictionService("rf.pkl")

service = load_service()
```

### Tip 3: Export Explanations as JSON
```python
service.export_explanation_as_json(
    explanation,
    "explanation.json"
)
```

### Tip 4: Generate Comparison Across Patients
```python
cohort = service.generate_cohort_analysis(
    features_list, patient_ids
)
print(f"Average Risk: {cohort['average_risk']:.1%}")
```

---

## 📞 Quick Reference

### API Endpoints (if using FastAPI)
- `POST /predict` - Single prediction with SHAP
- `POST /batch_predict` - Batch predictions
- `POST /generate_report` - Generate PDF report
- `GET /feature_importance` - Global feature importance

### CLI Commands
```bash
# Run Streamlit
streamlit run pages/shap_explainability.py

# Run with specific model
streamlit run pages/shap_explainability.py -- --model custom.pkl

# Enable debug logging
streamlit run pages/shap_explainability.py --logger.level=debug
```

---

## 🎓 Learning Resources

- **SHAP Documentation**: https://shap.readthedocs.io/
- **ReportLab Guide**: https://www.reportlab.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Model Interpretability**: https://christophm.github.io/interpretable-ml-book/

---

**Need help?** Check the full guide: `SHAP_EXPLAINABILITY_GUIDE.md`
