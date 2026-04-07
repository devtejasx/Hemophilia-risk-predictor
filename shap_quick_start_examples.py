#!/usr/bin/env python3
"""
SHAP Explainability - Quick Start Examples
Runnable examples showing how to use SHAP for model explainability
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from shap_explainability import (
    SHAPExplainer, SHAPVisualizer, SHAPInterpreter,
    display_shap_dashboard, display_feature_importance, explain_individual_prediction
)


# ============================================================================
# EXAMPLE 1: Standalone Python Script
# ============================================================================

def example_standalone():
    """
    Use SHAP without Streamlit - pure Python
    Best for: Batch analysis, scripts, CLI tools
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Standalone Python Usage")
    print("="*70)
    
    try:
        # Load model
        model = joblib.load("rf.pkl")
        columns = joblib.load("columns.pkl")
        
        # Create patient data
        patient = {
            "mutation_type": "missense",
            "exon": 5,
            "severity": "moderate",
            "age_first_treatment": 5,
            "dose_intensity": 2500,
            "exposure_days": 100
        }
        
        # Convert to DataFrame
        X = pd.DataFrame([patient])
        X = pd.get_dummies(X, columns=['mutation_type', 'severity'])
        for col in columns:
            if col not in X:
                X[col] = 0
        X = X[columns]
        
        # Initialize SHAP
        explainer = SHAPExplainer(model, columns)
        explanation = explainer.explain_prediction(X)
        
        if explanation:
            # Print results
            print(f"\n✅ Prediction: {explanation['predictions'][0]:.1%} risk")
            print(f"✅ Base Value: {explanation['base_value']:.3f}")
            
            # Get interpretation
            interpreter = SHAPInterpreter()
            interp = interpreter.interpret_prediction(explanation)
            print(f"\n📊 Assessment: {interp['prediction_phrase']}")
            print(f"Key factors:")
            for factor in interp['key_factors'][:3]:
                print(f"  - {factor}")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")


# ============================================================================
# EXAMPLE 2: Streamlit Dashboard
# ============================================================================

def example_streamlit_dashboard():
    """
    Full Streamlit dashboard with SHAP analysis
    Best for: Clinical web interface
    
    HOW TO RUN:
    1. Save this as streamlit_shap_demo.py
    2. Run: streamlit run streamlit_shap_demo.py
    """
    code = '''
import streamlit as st
import joblib
import pandas as pd
from shap_explainability import SHAPExplainer, display_shap_dashboard

st.set_page_config(page_title="SHAP Analysis", layout="wide")
st.title("🧠 SHAP Model Explainability Dashboard")

# Load models
@st.cache_resource
def load_data():
    model = joblib.load("rf.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

model, columns = load_data()

# Sidebar for inputs
with st.sidebar:
    st.markdown("## Patient Input")
    disability = st.radio("Disease Severity", ["Mild", "Moderate", "Severe"])
    exposure_days = st.slider("Exposure Days", 0, 300, 100)
    baseline_factor = st.slider("Baseline Factor %", 5, 100, 50)

# Create prediction
severity_map = {"Mild": "mild", "Moderate": "moderate", "Severe": "severe"}
X = pd.DataFrame([{
    "mutation_type": "missense",
    "exon": 5,
    "severity": severity_map[disability],
    "age_first_treatment": 5,
    "dose_intensity": 2500,
    "exposure_days": exposure_days
}])

X = pd.get_dummies(X, columns=['mutation_type', 'severity'])
for col in columns:
    if col not in X:
        X[col] = 0
X = X[columns]

# Get SHAP analysis
explainer = SHAPExplainer(model, columns)
explanation = explainer.explain_prediction(X)

if explanation:
    # Display prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", f"{explanation['predictions'][0]:.1%}")
    with col2:
        risk_level = "HIGH" if explanation["predictions"][0] > 0.6 else "MODERATE" if explanation["predictions"][0] > 0.4 else "LOW"
        st.metric("Risk Level", risk_level)
    with col3:
        st.metric("Base Value", f"{explanation['base_value']:.3f}")
    
    # Display SHAP dashboard
    display_shap_dashboard(explanation, model_type="Random Forest")
'''
    
    print("EXAMPLE 2: Streamlit Dashboard")
    print("\nPaste this code into a new file and run:")
    print("\n" + code)
    print("\n\nThen run: streamlit run your_file.py")


# ============================================================================
# EXAMPLE 3: Integration into Existing Streamlit App
# ============================================================================

def example_integration():
    """
    How to add SHAP to existing Streamlit app
    Best for: Adding to current applications
    """
    code = '''
# In your existing Streamlit app, after showing prediction:

from shap_explainability import SHAPExplainer, SHAPVisualizer

if st.checkbox("🧠 Show SHAP Analysis"):
    try:
        # Load model
        model = joblib.load("rf.pkl")
        columns = joblib.load("columns.pkl")
        
        # Create SHAP explainer
        explainer = SHAPExplainer(model, columns)
        
        # Get explanation for patient
        explanation = explainer.explain_prediction(patient_features_df)
        
        if explanation:
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Summary", "Waterfall", "Force"])
            
            with tab1:
                fig = SHAPVisualizer.plot_summary(explanation)
                st.pyplot(fig)
            
            with tab2:
                fig = SHAPVisualizer.plot_waterfall(explanation)
                st.pyplot(fig)
            
            with tab3:
                fig = SHAPVisualizer.plot_force(explanation)
                st.pyplot(fig)
    
    except Exception as e:
        st.error(f"SHAP analysis failed: {str(e)}")
'''
    
    print("EXAMPLE 3: Integration Pattern")
    print("\n" + code)


# ============================================================================
# EXAMPLE 4: Batch Analysis for Multiple Patients
# ============================================================================

def example_batch_analysis():
    """
    Analyze multiple patients and compare
    Best for: Population analysis, reports
    """
    code = '''
import pandas as pd
from shap_explainability import SHAPExplainer

# Load model
model = joblib.load("rf.pkl")
columns = joblib.load("columns.pkl")

# Load patient data
patient_df = pd.read_csv("patients.csv")

# Create explainer
explainer = SHAPExplainer(model, columns)

# Analyze each patient
results = []
for idx, row in patient_df.iterrows():
    # Prepare features
    X = pd.DataFrame([row])
    X = pd.get_dummies(X)
    for col in columns:
        if col not in X:
            X[col] = 0
    X = X[columns]
    
    # Get SHAP explanation
    explanation = explainer.explain_prediction(X)
    if explanation:
        prediction = explanation['predictions'][0]
        risk_level = "HIGH" if prediction > 0.6 else "MODERATE" if prediction > 0.4 else "LOW"
        
        results.append({
            "patient_id": row.get("patient_id"),
            "name": row.get("name"),
            "risk": prediction,
            "risk_level": risk_level
        })

# Display results
result_df = pd.DataFrame(results).sort_values("risk", ascending=False)
print(result_df)

# Export
result_df.to_csv("shap_analysis_results.csv", index=False)
'''
    
    print("EXAMPLE 4: Batch Analysis")
    print("\n" + code)


# ============================================================================
# EXAMPLE 5: Clinical Decision Support Workflow
# ============================================================================

def example_clinical_workflow():
    """
    Using SHAP in clinical decision-making
    Best for: Healthcare applications
    """
    code = '''
# Clinical workflow example:

# Step 1: Get patient data and prediction
patient_id = "P12345"
patient_data = load_patient_data(patient_id)
features = prepare_features(patient_data)

# Step 2: Generate prediction
model = load_model()
prediction_probability = model.predict_proba(features)[0][1]

# Step 3: Generate SHAP explanation
explainer = SHAPExplainer(model, feature_names)
explanation = explainer.explain_prediction(features)

# Step 4: Interpret for clinical use
interpreter = SHAPInterpreter()
interpretation = interpreter.interpret_prediction(
    explanation,
    risk_threshold=0.5,
    context="clinical"
)

# Step 5: Generate clinical report
print(f"Patient {patient_id}:")
print(f"Risk Score: {prediction_probability:.1%}")
print(f"Risk Level: {interpretation['risk_level']}")
print(f"Assessment: {interpretation['prediction_phrase']}")
print("Key Factors:")
for factor in interpretation['key_factors']:
    print(f"  • {factor}")
print(f"Clinical Assessment: {interpretation['overall_assessment']}")

# Step 6: Store in EHR
save_to_ehr(
    patient_id,
    prediction_probability,
    interpretation
)

# Step 7: Clinical team reviews
# (Notification sent to clinical team for validation)
'''
    
    print("EXAMPLE 5: Clinical Decision Support")
    print("\n" + code)


# ============================================================================
# EXAMPLE 6: Feature Importance Comparison
# ============================================================================

def example_feature_comparison():
    """
    Compare features across different models
    Best for: Model validation, research
    """
    code = '''
import pandas as pd
from shap_explainability import SHAPExplainer

# Load both models
rf_model = joblib.load("rf.pkl")
xgb_model = joblib.load("xgb.pkl")
columns = joblib.load("columns.pkl")

# Create data for analysis (background set)
X = pd.read_csv("training_data.csv")

# Get importance from both models
rf_explainer = SHAPExplainer(rf_model, columns, "random_forest")
xgb_explainer = SHAPExplainer(xgb_model, columns, "xgboost")

rf_importance = rf_explainer.get_feature_importance(X, top_n=10)
xgb_importance = xgb_explainer.get_feature_importance(X, top_n=10)

# Merge for comparison
comparison = rf_importance.merge(
    xgb_importance,
    on="Feature",
    suffixes=("_RF", "_XGB")
)
comparison["Difference"] = abs(comparison["Importance_RF"] - comparison["Importance_XGB"])
comparison = comparison.sort_values("Difference", ascending=False)

print("Feature Importance Comparison (RF vs XGBoost):")
print(comparison)

# Visualize
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(comparison))
ax.bar([i-0.2 for i in x], comparison["Importance_RF"], width=0.4, label="Random Forest")
ax.bar([i+0.2 for i in x], comparison["Importance_XGB"], width=0.4, label="XGBoost")
ax.set_xticks(x)
ax.set_xticklabels(comparison["Feature"], rotation=45, ha="right")
ax.set_ylabel("Feature Importance")
ax.set_title("Model Comparison: Feature Importance")
ax.legend()
plt.tight_layout()
plt.show()
'''
    
    print("EXAMPLE 6: Feature Comparison")
    print("\n" + code)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "SHAP EXPLAINABILITY - QUICK START EXAMPLES".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    print("""
This file contains 6 runnable examples showing how to use SHAP:

1. Standalone Python (no web framework)
2. Full Streamlit Dashboard
3. Integration into Existing App
4. Batch Analysis for Multiple Patients
5. Clinical Decision Support Workflow
6. Feature Importance Comparison

Choose the example that matches your use case!
""")
    
    # Show examples
    example_standalone()
    example_streamlit_dashboard()
    example_integration()
    example_batch_analysis()
    example_clinical_workflow()
    example_feature_comparison()
    
    print("\n" + "#"*70)
    print("Ready to use! Copy any example code and adapt for your needs.")
    print("#"*70 + "\n")
