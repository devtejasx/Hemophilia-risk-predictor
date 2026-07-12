"""
SHAP Examples & Implementation Patterns
========================================

Comprehensive examples showing how to use the SHAP explainability system
in various scenarios.
"""

# Example 1: Basic Prediction with SHAP Explanation
# ===================================================

def example_basic_prediction():
    """Basic example: single patient prediction with SHAP explanation."""
    
    import numpy as np
    from backend.services.prediction import PredictionService
    
    # Initialize prediction service
    service = PredictionService(
        model_path="rf.pkl",
        explainability_enabled=True,
        background_data_path="background_data.pkl"
    )
    
    # Set feature names for interpretability
    feature_names = [
        "Hemoglobin", "WBC", "Platelets", "Treatment Adherence",
        "Bleeds (Month)", "Inhibitor Screen", "Previous Surgery",
        "Transfusions"
    ]
    service.set_feature_names(feature_names)
    
    # Patient features
    patient_features = np.array([[
        14.0,  # Hemoglobin (g/dL)
        7.5,   # White blood cells (K/uL)
        250.0, # Platelets (K/uL)
        90,    # Treatment adherence (%)
        1,     # Bleeds in past month
        0,     # Inhibitor screen (negative)
        0,     # No previous surgery
        0      # No transfusions
    ]])
    
    # Get prediction with explanation
    result = service.predict_with_explanation(patient_features, feature_names)
    
    # Access results
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Risk Score: {result['prediction']:.1%}")
    print(f"Risk Level: {result['clinical_summary']['risk_level']}")
    print(f"\nTop Risk-Increasing Factors:")
    for factor in result['explanation']['top_positive_contributors'][:3]:
        print(f"  • {factor['feature']}: {factor['contribution']:.4f}")
    print(f"\nClinical Recommendations:")
    for rec in result['clinical_summary']['recommendations']:
        print(f"  ✓ {rec}")
    
    return result


# Example 2: Generate PDF Report with Visualizations
# ===================================================

def example_pdf_report_generation():
    """Generate complete clinical PDF report."""
    
    import numpy as np
    from backend.services.prediction import PredictionService
    from pathlib import Path
    
    service = PredictionService("rf.pkl", explainability_enabled=True)
    service.set_feature_names([
        "Hemoglobin", "WBC", "Platelets", "Treatment Adherence",
        "Bleeds (Month)", "Inhibitor Screen", "Previous Surgery", "Transfusions"
    ])
    
    # Patient data
    patient_data = {
        "patient_id": "HEM-2026-001",
        "name": "John Doe",
        "date_of_birth": "1980-05-15",
        "age": 45,
        "gender": "Male",
        "diagnosis": "Hemophilia A"
    }
    
    # Features
    features = np.array([[14.0, 7.5, 250.0, 90, 1, 0, 0, 0]])
    
    # Generate report
    output_path = "reports/patient_HEM2026001.pdf"
    pdf_bytes, report_data = service.generate_full_report(
        patient_data=patient_data,
        features=features,
        include_trends=True,
        include_visualizations=True,
        output_path=output_path
    )
    
    if pdf_bytes:
        print(f"✅ Report generated: {output_path}")
        print(f"   File size: {len(pdf_bytes) / 1024:.1f} KB")
    else:
        print("❌ Report generation failed")
    
    return pdf_bytes


# Example 3: Batch Processing Multiple Patients
# ===============================================

def example_batch_processing():
    """Process multiple patients from CSV."""
    
    import pandas as pd
    import numpy as np
    from backend.services.prediction import PredictionService
    
    # Load patient data
    df = pd.read_csv("patients.csv")
    print(f"Processing {len(df)} patients...")
    
    service = PredictionService("rf.pkl", explainability_enabled=True)
    
    # Extract features and IDs
    feature_columns = ['hemoglobin', 'wbc', 'platelets', 'adherence', 
                       'bleeds', 'inhibitor', 'surgery', 'transfusions']
    features_list = [df.loc[i, feature_columns].values for i in range(len(df))]
    patient_ids = df['patient_id'].tolist()
    
    # Batch predict
    results = service.batch_predict_with_explanations(
        features=np.array(features_list),
        sample_size=None  # Process all
    )
    
    # Aggregate results
    summary = {
        "total": len(results),
        "high_risk": 0,
        "moderate_risk": 0,
        "low_risk": 0,
        "predictions": []
    }
    
    for patient_id, result in zip(patient_ids, results):
        risk_score = result['prediction']
        risk_level = result['clinical_summary']['risk_level']
        
        summary['predictions'].append({
            "patient_id": patient_id,
            "risk_score": risk_score,
            "risk_level": risk_level
        })
        
        if risk_level == "HIGH":
            summary["high_risk"] += 1
        elif risk_level == "MODERATE":
            summary["moderate_risk"] += 1
        else:
            summary["low_risk"] += 1
    
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    print(f"Total Patients: {summary['total']}")
    print(f"🔴 High Risk: {summary['high_risk']}")
    print(f"🟡 Moderate Risk: {summary['moderate_risk']}")
    print(f"🟢 Low Risk: {summary['low_risk']}")
    
    return summary


# Example 4: Cohort Analysis
# ===========================

def example_cohort_analysis():
    """Analyze risk patterns across cohort."""
    
    import numpy as np
    from backend.services.prediction import PredictionService
    
    service = PredictionService("rf.pkl", explainability_enabled=True)
    
    # Generate sample cohort
    np.random.seed(42)
    cohort_size = 50
    features_list = [
        np.random.uniform(
            [10, 4, 150, 70, 0, 0, 0, 0],  # min
            [16, 10, 400, 100, 5, 1, 1, 10]  # max
        )
        for _ in range(cohort_size)
    ]
    patient_ids = [f"P{i:04d}" for i in range(cohort_size)]
    
    # Cohort analysis
    cohort_analysis = service.generate_cohort_analysis(
        features_list=features_list,
        patient_ids=patient_ids
    )
    
    print("\n" + "="*50)
    print("COHORT ANALYSIS")
    print("="*50)
    print(f"Total Patients: {cohort_analysis['total_patients']}")
    print(f"Average Risk: {cohort_analysis['average_risk']:.1%}")
    print(f"Median Risk: {cohort_analysis['median_risk']:.1%}")
    print(f"Risk Range: {cohort_analysis['min_risk']:.1%} - {cohort_analysis['max_risk']:.1%}")
    print(f"\nRisk Distribution:")
    print(f"  🔴 High Risk: {cohort_analysis['high_risk_count']}")
    print(f"  🟡 Moderate Risk: {cohort_analysis['moderate_risk_count']}")
    print(f"  🟢 Low Risk: {cohort_analysis['low_risk_count']}")
    
    return cohort_analysis


# Example 5: Feature Importance Analysis
# =======================================

def example_feature_importance():
    """Analyze global feature importance."""
    
    import numpy as np
    from backend.services.prediction import PredictionService
    import pandas as pd
    
    service = PredictionService("rf.pkl", explainability_enabled=True)
    
    # Generate sample data
    np.random.seed(42)
    sample_size = 100
    background_data = np.random.uniform(
        [10, 4, 150, 70, 0, 0, 0, 0],
        [16, 10, 400, 100, 5, 1, 1, 10],
        (sample_size, 8)
    )
    
    # Get feature importance
    importance = service.get_feature_importance(background_data)
    
    if "feature_importance" in importance:
        df_importance = pd.DataFrame(importance["feature_importance"])
        
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE RANKING")
        print("="*50)
        print(df_importance.to_string(index=False))
        
        # Top 3 features
        print(f"\nTop 3 Most Important Features:")
        for idx, row in df_importance.head(3).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    return importance


# Example 6: Streamlit Integration
# =================================

def example_streamlit_integration():
    """Example integration in Streamlit app."""
    
    import streamlit as st
    import numpy as np
    from backend.services.prediction import PredictionService
    from backend.ui_components import ExplainabilityUI
    
    # Initialize session
    if "service" not in st.session_state:
        st.session_state.service = PredictionService("rf.pkl")
    
    service = st.session_state.service
    
    # Input form
    st.title("🔮 Clinical Risk Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        hemoglobin = st.slider("Hemoglobin (g/dL)", 10.0, 16.0, 14.0)
        wbc = st.slider("White Blood Cells (K/uL)", 4.0, 10.0, 7.5)
    with col2:
        platelets = st.slider("Platelets (K/uL)", 150, 400, 250)
        adherence = st.slider("Treatment Adherence (%)", 0, 100, 90)
    
    # Make prediction
    if st.button("Generate Prediction"):
        features = np.array([[hemoglobin, wbc, platelets, adherence, 1, 0, 0, 0]])
        result = service.predict_with_explanation(features)
        
        # Display results
        if "error" not in result:
            risk_score = result["prediction"]
            risk_level = result["clinical_summary"]["risk_level"]
            
            # Risk gauge
            ExplainabilityUI.display_risk_score(risk_score, risk_level)
            
            # Feature importance
            ExplainabilityUI.display_feature_importance(
                result["explanation"]["feature_contributions"]
            )
            
            # Clinical summary
            ExplainabilityUI.display_clinical_summary(
                result["clinical_summary"]
            )
        else:
            st.error(result["error"])


# Example 7: Export Results
# ==========================

def example_export_results():
    """Export predictions and explanations."""
    
    import json
    import numpy as np
    from backend.services.prediction import PredictionService
    
    service = PredictionService("rf.pkl", explainability_enabled=True)
    
    # Make prediction
    features = np.array([[14.0, 7.5, 250.0, 90, 1, 0, 0, 0]])
    result = service.predict_with_explanation(features)
    
    # Export as JSON
    export_path = "prediction_result.json"
    service.export_explanation_as_json(
        result["explanation"],
        export_path
    )
    print(f"✅ Exported to {export_path}")
    
    # Export predictions
    import pandas as pd
    df = pd.DataFrame([{
        "patient_id": "P001",
        "risk_score": result["prediction"],
        "risk_level": result["clinical_summary"]["risk_level"],
        "confidence": result.get("prediction_proba", [0, 0])[1] if result.get("prediction_proba") else None
    }])
    
    df.to_csv("predictions.csv", index=False)
    print("✅ Exported to predictions.csv")
    
    return df


# Example 8: Custom Clinical Interpretation
# ==========================================

def example_custom_interpretation():
    """Generate custom clinical interpretation."""
    
    from backend.services.explainability import ExplainabilityService
    import numpy as np
    import joblib
    
    # Load model
    model = joblib.load("rf.pkl")
    
    # Initialize explainer
    explainer = ExplainabilityService(model)
    explainer.set_feature_names([
        "Hemoglobin", "WBC", "Platelets", "Treatment Adherence",
        "Bleeds", "Inhibitor", "Surgery", "Transfusions"
    ])
    
    # Get explanation
    features = np.array([[14.0, 7.5, 250.0, 90, 1, 1, 0, 0]])
    explanation = explainer.explain_prediction(features)
    
    # Custom interpretation
    clinical_summary = explainer.generate_clinical_explanation(
        explanation,
        risk_threshold=0.5
    )
    
    print("\n" + "="*50)
    print("CLINICAL INTERPRETATION")
    print("="*50)
    print(f"Risk Level: {clinical_summary['risk_level']}")
    print(f"Description: {clinical_summary['risk_description']}")
    print(f"\nKey Risk Factors:")
    for factor in clinical_summary['key_risk_factors'][:5]:
        print(f"  • {factor['factor']}")
        print(f"    - Impact: {factor['impact']}")
        print(f"    - Value: {factor['current_value']:.2f}")
    print(f"\nRecommendations:")
    for rec in clinical_summary['recommendations']:
        print(f"  ✓ {rec}")
    
    return clinical_summary


# Main execution
# ==============

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY EXAMPLES")
    print("="*60)
    
    # Run examples (comment out as needed)
    print("\n1. Basic Prediction with SHAP")
    # example_basic_prediction()
    
    print("\n2. PDF Report Generation")
    # example_pdf_report_generation()
    
    print("\n3. Batch Processing")
    # example_batch_processing()
    
    print("\n4. Cohort Analysis")
    # example_cohort_analysis()
    
    print("\n5. Feature Importance")
    # example_feature_importance()
    
    # Uncomment below to see results
    # example_basic_prediction()
    # example_feature_importance()
    # example_cohort_analysis()
