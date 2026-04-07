# Integration Guide: Using SHAP Explainability in Your Application

## 🔗 Quick Integration

### Option 1: Use Standalone Streamlit Page (Easiest)

The system includes a complete standalone Streamlit page ready to use:

```bash
cd pages/
streamlit run shap_explainability.py
```

Visit: `http://localhost:8501`

---

### Option 2: Integrate Into Existing App

Add this to your main Streamlit app:

```python
# In your app.py or main page
import streamlit as st
from backend.services.prediction import PredictionService
from backend.ui_components import ExplainabilityUI

# Initialize (do once)
@st.cache_resource
def load_prediction_service():
    return PredictionService("rf.pkl", explainability_enabled=True)

service = load_prediction_service()

# Use in your pages/logic
def show_prediction_section():
    st.subheader("Patient Risk Prediction")
    
    # Get patient features (from your form)
    features = get_patient_features()  # Your existing function
    
    # Generate prediction with SHAP
    result = service.predict_with_explanation(features)
    
    # Display using UI components
    ExplainabilityUI.display_risk_score(
        result["prediction"],
        result["clinical_summary"]["risk_level"]
    )
    
    ExplainabilityUI.display_feature_importance(
        result["explanation"]["feature_contributions"]
    )
    
    ExplainabilityUI.display_clinical_summary(
        result["clinical_summary"]
    )
```

---

## 🔨 Integration Patterns

### Pattern 1: Add to Prediction Page

```python
# pages/predictions.py or existing prediction module
import streamlit as st
from backend.services.prediction import PredictionService
from backend.ui_components import ExplainabilityUI, ReportUI

def main():
    st.title("Patient Risk Prediction")
    
    # Setup
    service = st.session_state.get("prediction_service")
    if not service:
        service = PredictionService("rf.pkl")
        st.session_state.prediction_service = service
    
    # Input form (your existing form)
    with st.form("patient_form"):
        hemoglobin = st.slider("Hemoglobin", 10.0, 16.0, 14.0)
        wbc = st.slider("WBC", 4.0, 10.0, 7.5)
        # ... more fields ...
        
        submitted = st.form_submit_button("Predict & Explain")
    
    if submitted:
        # Make prediction
        features = np.array([[hemoglobin, wbc, ...]])
        result = service.predict_with_explanation(features)
        
        # Display SHAP explanation
        col1, col2 = st.columns([1, 2])
        with col1:
            ExplainabilityUI.display_risk_score(
                result["prediction"],
                result["clinical_summary"]["risk_level"]
            )
        
        with col2:
            ExplainabilityUI.display_feature_importance(
                result["explanation"]["feature_contributions"]),
                max_features=8
            )
        
        # Generate report
        st.markdown("---")
        if ReportUI.show_report_generation_form()["submitted"]:
            pdf_bytes, _ = service.generate_full_report(
                patient_data={...},
                features=features,
                include_visualizations=True
            )
            
            st.download_button(
                "📥 Download Report",
                pdf_bytes,
                "report.pdf",
                "application/pdf"
            )

if __name__ == "__main__":
    main()
```

---

### Pattern 2: Add to Dashboard

```python
# pages/dashboard.py
import streamlit as st
import pandas as pd
from backend.services.prediction import PredictionService
from backend.ui_components import ExplainabilityUI

def main():
    st.title("Clinical Dashboard")
    
    service = st.session_state.get("prediction_service")
    if not service:
        service = PredictionService("rf.pkl")
        st.session_state.prediction_service = service
    
    # Patient selector
    patient_id = st.selectbox("Select Patient", load_patient_list())
    patient_data = load_patient_data(patient_id)
    
    # Get latest prediction
    result = service.predict_with_explanation(patient_data["features"])
    
    # Display dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Risk Score", f"{result['prediction']:.1%}")
    with col2:
        st.metric("Risk Level", result["clinical_summary"]["risk_level"])
    with col3:
        if result.get("prediction_proba"):
            st.metric("Confidence", f"{max(result['prediction_proba']):.1%}")
    
    # SHAP explanation
    st.subheader("Contributing Factors")
    ExplainabilityUI.display_feature_importance(
        result["explanation"]["feature_contributions"]
    )
    
    # Trends
    st.subheader("Risk Trend")
    trend_data = load_patient_trends(patient_id)
    if trend_data:
        ExplainabilityUI.display_trend_analysis(trend_data)
```

---

### Pattern 3: Add to Batch Analysis

```python
# pages/batch_analysis.py
import streamlit as st
import pandas as pd
from backend.services.prediction import PredictionService

def main():
    st.title("Batch Analysis")
    
    service = st.session_state.get("prediction_service")
    if not service:
        service = PredictionService("rf.pkl")
        st.session_state.prediction_service = service
    
    # Upload CSV
    uploaded_file = st.file_uploader("Upload patient CSV")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Run batch predictions
        if st.button("Run Predictions"):
            with st.spinner("Processing..."):
                # Extract features
                feature_cols = [col for col in df.columns 
                              if col.lower() not in ['patient_id', 'name']]
                features_array = df[feature_cols].values
                
                # Batch predict
                results = service.batch_predict_with_explanations(
                    features_array,
                    sample_size=len(df)  # Or limit for performance
                )
                
                # Display results
                results_df = pd.DataFrame([
                    {
                        "patient_id": row["patient_id"],
                        "risk_score": r["prediction"],
                        "risk_level": r["clinical_summary"]["risk_level"]
                    }
                    for r, row in zip(results, df.iterrows())
                ])
                
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "results.csv",
                    "text/csv"
                )
```

---

## 📊 Integration with Existing Code

### Integration Point 1: Clinical Assistant Module

```python
# In clinical_assistant.py or gpt_chatbot.py
from backend.services.prediction import PredictionService

def enhanced_risk_analysis(patient_id, patient_features):
    """
    Existing function enhanced with SHAP explanations
    """
    # Your existing analysis
    existing_result = existing_risk_function(patient_features)
    
    # NEW: Add SHAP explanation
    prediction_service = get_prediction_service()
    shap_result = prediction_service.predict_with_explanation(patient_features)
    
    # Combine results
    enhanced_result = {
        **existing_result,
        "shap_explanation": shap_result["explanation"],
        "clinical_summary": shap_result["clinical_summary"],
        "contributing_factors": shap_result["explanation"]["feature_contributions"]
    }
    
    return enhanced_result
```

### Integration Point 2: Database Module

```python
# In database.py
from backend.services.prediction import PredictionService

def save_prediction_with_explanation(patient_id, prediction_result):
    """Save prediction and SHAP explanation to database"""
    
    # Save basic prediction
    save_prediction(patient_id, prediction_result["prediction"])
    
    # NEW: Save SHAP explanation
    save_explanation(
        patient_id,
        prediction_result["explanation"],
        prediction_result["clinical_summary"]
    )
    
    # NEW: Save report
    pdf_bytes, report_data = generate_report(prediction_result)
    save_report(patient_id, pdf_bytes, report_data)
```

### Integration Point 3: Evaluation Module

```python
# In evaluation.py
from backend.services.prediction import PredictionService
from backend.services.explainability import ExplainabilityService

class ModelEvaluator:
    """Enhanced evaluator with explainability"""
    
    def evaluate_with_explanations(self, X_test, y_test):
        """
        Existing evaluation + SHAP feature importance
        """
        # Your existing evaluation
        basic_metrics = self._evaluate_model(X_test, y_test)
        
        # NEW: Add SHAP feature importance
        explainer = ExplainabilityService(self.model, background_data=X_test)
        feature_importance = explainer.get_feature_importance(X_test)
        
        return {
            **basic_metrics,
            "shap_feature_importance": feature_importance["feature_importance"]
        }
```

---

## 🔗 Setup Steps

### Step 1: Ensure Dependencies

```bash
pip install shap reportlab plotly pandas numpy scikit-learn streamlit --upgrade
```

### Step 2: Organize Files

```
your_project/
├── app.py (main Streamlit app)
├── pages/
│   └── shap_explainability.py (new page)
├── backend/
│   ├── services/
│   │   ├── explainability.py (new)
│   │   ├── reports.py (new)
│   │   ├── prediction.py (new)
│   │   └── ... existing files
│   ├── ui_components.py (new)
│   └── ... existing files
└── ... other files
```

### Step 3: Create __init__.py

```python
# backend/__init__.py
from .services.prediction import PredictionService
from .services.explainability import ExplainabilityService
from .services.reports import ClinicalReportGenerator
from .ui_components import ExplainabilityUI, ReportUI

__all__ = [
    'PredictionService',
    'ExplainabilityService',
    'ClinicalReportGenerator',
    'ExplainabilityUI',
    'ReportUI'
]
```

### Step 4: Update Main App

```python
# app.py (if not using standalone page)
import streamlit as st
from pages.shap_explainability import main as shap_main

def main():
    st.set_page_config(
        page_title="Clinical Decision Support",
        layout="wide"
    )
    
    # Navigation
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Predictions", "SHAP Analysis", "Batch"]
    )
    
    if page == "SHAP Analysis":
        shap_main()  # Use new page
    else:
        # Your existing pages

if __name__ == "__main__":
    main()
```

---

## ✅ Verification Checklist

After integration:

- [ ] Dependencies installed: `pip list | grep shap`
- [ ] Services import correctly: `python -c "from backend.services.prediction import PredictionService"`
- [ ] Model file exists: `ls -la rf.pkl`
- [ ] UI components load: `python -c "from backend.ui_components import ExplainabilityUI"`
- [ ] Streamlit page runs: `streamlit run pages/shap_explainability.py`
- [ ] Single prediction works
- [ ] PDF report generates
- [ ] Charts display properly
- [ ] Batch processing works

---

## 🧪 Quick Test

Run this to verify everything works:

```python
import numpy as np
from backend.services.prediction import PredictionService

# Initialize
service = PredictionService("rf.pkl", explainability_enabled=True)

# Test prediction
features = np.array([[14.0, 7.5, 250.0, 90, 1, 0, 0, 0]])
result = service.predict_with_explanation(features)

# Verify
assert "prediction" in result
assert "explanation" in result
assert "clinical_summary" in result

print("✅ All systems operational!")
print(f"Risk Score: {result['prediction']:.1%}")
```

---

## 🚀 Next Steps

1. **Run It**
   ```bash
   streamlit run pages/shap_explainability.py
   ```

2. **Generate Sample Predictions**
   - Use the web interface
   - Download PDF reports

3. **Integrate Into Your Workflow**
   - Add to existing pages
   - Connect to database
   - Set up batch processing

4. **Customize**
   - Adjust feature names
   - Modify risk thresholds
   - Brand PDF reports

---

## 📞 Common Integration Questions

**Q: How do I add this to my existing prediction page?**
A: Use Pattern 1 above - just call `service.predict_with_explanation()` and display results with UI components.

**Q: Can I use different models?**
A: Yes! Just provide different model path to PredictionService. Works with any sklearn/xgboost model.

**Q: How do I customize the SHAP explanations?**
A: Modify `ExplainabilityService.generate_clinical_explanation()` to customize risk thresholds and recommendations.

**Q: How do I customize the PDF reports?**
A: Modify `ClinicalReportGenerator` methods to change layout, styling, or sections.

**Q: Can I run this without Streamlit?**
A: Yes! Use `PredictionService` directly in any Python app for API/CLI/batch usage.

---

## 📚 Reference Files

- `SHAP_QUICKSTART.md` - 5-minute setup
- `SHAP_EXPLAINABILITY_GUIDE.md` - Full documentation
- `SHAP_EXAMPLES.py` - Code examples
- `ARCHITECTURE.md` - System design
- `IMPLEMENTATION_COMPLETE.md` - Project summary

---

**Ready to integrate? Start with one of the patterns above and customize as needed!**
