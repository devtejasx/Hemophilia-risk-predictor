"""
SHAP Explainability & Reporting Module
=======================================

Streamlit page for model predictions with SHAP explanations and PDF report generation.
Provides complete end-to-end prediction and interpretation workflow.

Features:
- Interactive prediction interface
- SHAP-based model explanations
- Automatic clinical report generation
- Feature importance visualization
- Prediction comparison and cohort analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import joblib

# Import custom modules
from backend.services.prediction import PredictionService
from backend.ui_components import ExplainabilityUI, ReportUI

logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "prediction_service" not in st.session_state:
        st.session_state.prediction_service = None
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "last_report_data" not in st.session_state:
        st.session_state.last_report_data = None


def load_models():
    """Load prediction models and data."""
    try:
        # Try to load the pre-trained model
        model_path = "rf.pkl"  # Adjust path as needed
        
        if Path(model_path).exists():
            try:
                service = PredictionService(
                    model_path=model_path,
                    explainability_enabled=True
                )
                if service.model is not None:
                    st.session_state.prediction_service = service
                    return service
            except Exception as load_error:
                logger.error(f"Failed to load service with model: {load_error}")
        
        # Model not found or failed to load, use mock
        st.warning(f"⚠️ Model file not found or failed to load: {model_path}")
        st.info("Creating a mock model for demonstration purposes...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            
            # Create a simple mock model
            mock_model = RandomForestClassifier(n_estimators=50, random_state=42)
            
            # Create a mock service - pass empty string to avoid None errors
            from backend.services.prediction import PredictionService as PS
            service = PS.__new__(PS)  # Create instance without __init__
            service.model = mock_model
            service.explainability_enabled = True
            service.explainer = None
            service.background_data = None
            service.feature_names = []
            
            st.session_state.prediction_service = service
            st.success("✅ Mock model ready for demo")
            return service
        except Exception as mock_error:
            st.error(f"Could not create mock model: {str(mock_error)}")
            logger.error(f"Failed to create mock model: {str(mock_error)}")
            return None
            
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        logger.error(f"Failed to load models: {str(e)}")
        
        # Try fallback to mock model
        try:
            from sklearn.ensemble import RandomForestClassifier
            from backend.services.prediction import PredictionService as PS
            
            st.info("Using mock model for demonstration...")
            mock_model = RandomForestClassifier(n_estimators=50, random_state=42)
            
            service = PS.__new__(PS)  # Create instance without __init__
            service.model = mock_model
            service.explainability_enabled = True
            service.explainer = None
            service.background_data = None
            service.feature_names = []
            
            st.session_state.prediction_service = service
            return service
        except Exception as fallback_error:
            st.error(f"Fallback failed: {str(fallback_error)}")
            logger.error(f"Fallback model creation failed: {str(fallback_error)}")
            return None


def render_prediction_input_form() -> Optional[Dict[str, Any]]:
    """
    Render form for prediction input.
    
    Returns:
        Dictionary with input data or None
    """
    st.subheader("Enter Patient Data for Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID", value="P001")
            first_name = st.text_input("First Name", value="John")
            last_name = st.text_input("Last Name", value="Doe")
            age = st.number_input("Age", min_value=0, max_value=150, value=45)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            diagnosis = st.text_input("Diagnosis", value="Hemophilia A")
            date_of_birth = st.date_input("Date of Birth")
        
        st.markdown("---")
        
        # Clinical features
        st.markdown("### Clinical Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", value=14.0, step=0.1)
            white_cells = st.number_input("White Blood Cells (K/uL)", value=7.5, step=0.1)
            platelets = st.number_input("Platelets (K/uL)", value=250.0, step=1.0)
        
        with col2:
            treatment_adherence = st.slider("Treatment Adherence (%)", 0, 100, 90)
            bleeds_past_month = st.number_input("Bleeds in Past Month", value=1, min_value=0)
            inhibitor_screen = st.selectbox("Inhibitor Screening Result", ["Negative", "Positive", "Not Tested"])
        
        with col3:
            previous_surgery = st.selectbox("Previous Surgery", ["Yes", "No"])
            transfusions = st.number_input("Number of Transfusions", value=0, min_value=0)
            genetic_mutation = st.text_input("Genetic Mutation", value="F8:c.6079G>A")
        
        st.markdown("---")
        
        submitted = st.form_submit_button("🔮 Generate Prediction & Explanation")
        
        if submitted:
            # Convert inhibitor screening
            inhibitor_map = {"Negative": 0, "Positive": 1, "Not Tested": 0.5}
            surgery_map = {"Yes": 1, "No": 0}
            
            # Create feature array (order matters!)
            features = np.array([[
                hemoglobin,
                white_cells,
                platelets,
                treatment_adherence,
                bleeds_past_month,
                inhibitor_map[inhibitor_screen],
                surgery_map[previous_surgery],
                transfusions,
                age,
                45  # average age for scaling
            ]])
            
            patient_data = {
                "patient_id": patient_id,
                "name": f"{first_name} {last_name}",
                "date_of_birth": date_of_birth.isoformat(),
                "age": age,
                "gender": gender,
                "diagnosis": diagnosis
            }
            
            return {
                "patient_data": patient_data,
                "features": features,
                "input_data": {
                    "hemoglobin": hemoglobin,
                    "white_cells": white_cells,
                    "platelets": platelets,
                    "treatment_adherence": treatment_adherence,
                    "bleeds_past_month": bleeds_past_month,
                    "inhibitor_screen": inhibitor_screen,
                    "previous_surgery": previous_surgery,
                    "transfusions": transfusions,
                    "genetic_mutation": genetic_mutation
                }
            }
    
    return None


def render_prediction_results(prediction_result: Dict[str, Any]) -> None:
    """
    Render prediction results with explanations.
    
    Args:
        prediction_result: Prediction dictionary from service
    """
    if "error" in prediction_result:
        st.error(f"Prediction Error: {prediction_result['error']}")
        return
    
    # Risk Score Display
    risk_score = prediction_result.get("prediction", 0)
    clinical_summary = prediction_result.get("clinical_summary", {})
    risk_level = clinical_summary.get("risk_level", "UNKNOWN")
    
    ExplainabilityUI.display_risk_score(risk_score, risk_level)
    
    st.markdown("---")
    
    # Clinical Interpretation
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        ExplainabilityUI.display_clinical_summary(clinical_summary)
    
    with col2:
        if prediction_result.get("prediction_proba"):
            ExplainabilityUI.display_prediction_confidence(
                prediction_result["prediction_proba"],
                int(risk_score > 0.5)
            )
    
    st.markdown("---")
    
    # Feature Importance
    explanation = prediction_result.get("explanation", {})
    if explanation and "feature_contributions" in explanation:
        ExplainabilityUI.display_feature_importance(
            explanation["feature_contributions"],
            max_features=10
        )
        
        st.markdown("---")
        
        # Top Factors
        ExplainabilityUI.display_top_factors(
            explanation.get("top_positive_contributors", []),
            explanation.get("top_negative_contributors", [])
        )


def render_report_generation(
    prediction_data: Dict[str, Any],
    prediction_result: Dict[str, Any]
) -> None:
    """
    Render report generation section.
    
    Args:
        prediction_data: Patient prediction data
        prediction_result: Prediction result from service
    """
    st.subheader("📊 Clinical Report Generation")
    
    # Report customization form
    options = ReportUI.show_report_generation_form()
    
    if options["submitted"] and st.session_state.prediction_service:
        try:
            with st.spinner("Generating report..."):
                pdf_bytes, report_data = st.session_state.prediction_service.generate_full_report(
                    patient_data=prediction_data["patient_data"],
                    features=prediction_data["features"],
                    include_trends=options["include_trends"],
                    include_visualizations=options["include_visualizations"]
                )
                
                if pdf_bytes:
                    st.success("✅ Report generated successfully!")
                    
                    # Store in session
                    st.session_state.last_report_data = report_data
                    
                    # Download button
                    filename = f"report_{prediction_data['patient_data']['patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf"
                    )
                    
                    # Show preview
                    with st.expander("View Report Preview"):
                        ReportUI.show_report_preview(report_data)
                else:
                    st.error("Failed to generate report")
        
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            logger.error(f"Report generation error: {str(e)}")


def render_batch_analysis() -> None:
    """Render batch prediction and analysis section."""
    st.subheader("📈 Batch Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with patient data",
        type="csv",
        help="CSV should contain patient features in columns"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} patients")
            
            if st.button("Run Batch Predictions"):
                with st.spinner("Running batch predictions..."):
                    # Extract features (assuming standard column names)
                    feature_columns = df.drop(
                        [col for col in df.columns if col in ['patient_id', 'name', 'diagnosis']],
                        axis=1
                    )
                    
                    patient_ids = df.get('patient_id', [f"P{i:03d}" for i in range(len(df))])
                    
                    results = st.session_state.prediction_service.generate_cohort_analysis(
                        features_list=[feature_columns.iloc[i].values for i in range(len(feature_columns))],
                        patient_ids=patient_ids,
                        feature_names=list(feature_columns.columns)
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Patients", results["total_patients"])
                    with col2:
                        st.metric("Average Risk", f"{results['average_risk']:.1%}")
                    with col3:
                        st.metric("High Risk Count", results["high_risk_count"])
                    
                    st.markdown("---")
                    
                    # Risk distribution
                    st.markdown("### Risk Distribution")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "🔴 High Risk",
                            results["high_risk_count"],
                            delta=f"{results['high_risk_count']/results['total_patients']:.1%}"
                        )
                    with col2:
                        st.metric(
                            "🟡 Moderate Risk",
                            results["moderate_risk_count"],
                            delta=f"{results['moderate_risk_count']/results['total_patients']:.1%}"
                        )
                    with col3:
                        st.metric(
                            "🟢 Low Risk",
                            results["low_risk_count"],
                            delta=f"{results['low_risk_count']/results['total_patients']:.1%}"
                        )
                    
                    # Export results
                    st.markdown("---")
                    results_df = pd.DataFrame(results["predictions"])
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=results_df.to_csv(index=False),
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="SHAP Explainability & Clinical Reports",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🔬 Clinical AI System")
        st.markdown("---")
        
        page = st.radio(
            "Select Page",
            ["Individual Prediction", "Batch Analysis", "Feature Importance", "About"]
        )
        
        st.markdown("---")
        st.markdown("### Model Information")
        if st.session_state.prediction_service:
            st.success("✅ Model loaded")
        else:
            st.warning("⚠️ Model not loaded")
    
    # Main content
    if page == "Individual Prediction":
        st.title("🔮 Individual Prediction with SHAP Explanations")
        st.markdown("Enter patient data to generate predictions with detailed explanations")
        
        # Load models if not already loaded
        if not st.session_state.prediction_service:
            with st.spinner("Loading models..."):
                load_models()
        
        if st.session_state.prediction_service:
            # Input form
            prediction_data = render_prediction_input_form()
            
            if prediction_data:
                # Generate prediction
                with st.spinner("Generating prediction and explanation..."):
                    prediction_result = st.session_state.prediction_service.predict_with_explanation(
                        prediction_data["features"]
                    )
                    st.session_state.last_prediction = prediction_result
                
                # Display results
                render_prediction_results(prediction_result)
                
                st.markdown("---")
                
                # Report generation
                render_report_generation(prediction_data, prediction_result)
        else:
            st.error("Could not load prediction model. Please check the model file path.")
    
    elif page == "Batch Analysis":
        st.title("📈 Batch Prediction & Analysis")
        st.markdown("Upload a CSV file to run predictions on multiple patients")
        
        if not st.session_state.prediction_service:
            with st.spinner("Loading models..."):
                load_models()
        
        if st.session_state.prediction_service:
            render_batch_analysis()
        else:
            st.error("Could not load prediction model.")
    
    elif page == "Feature Importance":
        st.title("📊 Global Feature Importance")
        st.markdown("View average feature contributions across predictions")
        
        if st.session_state.prediction_service and st.session_state.prediction_service.explainability_enabled:
            try:
                importance = st.session_state.prediction_service.get_feature_importance()
                
                if "error" not in importance:
                    importance_list = importance.get("feature_importance", [])
                    
                    if importance_list:
                        df_importance = pd.DataFrame(importance_list)
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Features Analyzed", len(df_importance))
                        with col2:
                            st.metric("Top Feature", df_importance.iloc[0]["feature"])
                        
                        st.markdown("---")
                        
                        # Chart
                        ExplainabilityUI.display_feature_importance(
                            importance_list,
                            max_features=min(15, len(importance_list))
                        )
                        
                        # Table
                        st.dataframe(df_importance, use_container_width=True)
                else:
                    st.error(importance.get("error", "Error calculating feature importance"))
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Explainability service not currently enabled")
    
    elif page == "About":
        st.title("📚 About This System")
        
        st.markdown("""
        ## SHAP Explainability & Clinical Reporting System
        
        ### Features
        - **SHAP Explanations**: Understand model predictions through Shapley values
        - **Clinical Reports**: Generate professional PDF reports with interpretations
        - **Feature Analysis**: Visualize feature importance and contributions
        - **Batch Processing**: Handle multiple predictions simultaneously
        - **Risk Assessment**: Integrated clinical risk evaluation
        
        ### Technology Stack
        - **ML Model**: XGBoost / Random Forest
        - **Explainability**: SHAP (SHapley Additive exPlanations)
        - **Reporting**: ReportLab for PDF generation
        - **Visualization**: Plotly, Matplotlib
        - **Interface**: Streamlit
        
        ### How to Use
        1. Navigate to "Individual Prediction" for single patient analysis
        2. Enter patient demographics and clinical data
        3. System generates prediction with SHAP explanation
        4. Generate professional PDF report
        5. Use "Batch Analysis" for multiple patients
        
        ### Clinical Considerations
        - This system is for decision support only
        - Always validate with clinical judgment
        - Consult healthcare professionals for diagnosis
        - Follow all relevant clinical protocols
        """)


if __name__ == "__main__":
    main()
