"""
Results Dashboard Page - Risk predictions and SHAP explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_utils.state_manager import StateManager
from streamlit_utils.ui_components import (
    create_header, show_success, show_error, show_warning,
    risk_card, metric_card, create_tabs, loading_spinner
)
from streamlit_utils.backend_client import get_backend_client
from streamlit_utils.plotly_charts import MedicalCharts, create_empty_chart


def render():
    """Render results dashboard page"""
    
    create_header("📊 Results Dashboard", "Risk predictions and AI explanations")
    
    # Get clients
    state = StateManager()
    backend = get_backend_client()
    
    # Check if patient is loaded
    current_patient = state.get_current_patient()
    
    if not current_patient or not current_patient.get("patient_id"):
        st.warning("⚠️  Please load a patient from the Patient Form first")
        return
    
    # Display current patient
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👤 Patient ID", current_patient.get("patient_id", "N/A"))
    with col2:
        st.metric("📅 Age", f"{current_patient.get('age', 'N/A')} years")
    with col3:
        st.metric("⚕️ Diagnosis", current_patient.get("diagnosis", "N/A"))
    
    st.divider()
    
    # Get prediction or run new prediction
    prediction = state.get_prediction_results()
    
    if not prediction:
        st.info("📌 Run prediction first to see results")
        
        if st.button("🔮 Generate Prediction", use_container_width=True, key="run_prediction"):
            with st.spinner("Running ML model prediction..."):
                prediction = backend.predict(current_patient)
            
            if prediction:
                state.set_prediction_results(prediction)
                st.rerun()
            else:
                show_error("Failed to get prediction. Check backend connection.")
        return
    
    # ========================================================================
    # TABS FOR DIFFERENT VIEWS
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = create_tabs(["Risk Score", "Explainability", "Trends", "Recommendations"])
    
    # ========================================================================
    # TAB 1: RISK SCORE
    # ========================================================================
    
    with tab1:
        st.markdown("### 🎯 Risk Assessment")
        st.divider()
        
        # Extract risk information
        risk_score = prediction.get("risk_score", 0.0)
        risk_label = prediction.get("risk_label", "UNKNOWN")
        confidence = prediction.get("confidence", 0.0)
        
        # Display risk gauge
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_gauge = MedicalCharts.risk_gauge(risk_score, "Risk Assessment")
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.markdown("#### Assessment Details")
            st.metric("Risk Score", f"{risk_score:.0%}")
            st.metric("Risk Level", risk_label)
            st.metric("Confidence", f"{confidence:.0%}")
            
            # Risk interpretation
            if risk_score > 0.75:
                st.error("🔴 **CRITICAL RISK** - Immediate intervention required")
            elif risk_score > 0.6:
                st.warning("🟠 **HIGH RISK** - Close monitoring recommended")
            elif risk_score > 0.4:
                st.warning("🟡 **MODERATE RISK** - Follow-up recommended")
            else:
                st.success("🟢 **LOW RISK** - Routine monitoring")
        
        # Model metadata
        st.divider()
        st.markdown("#### Model Information")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", prediction.get("model_name", "N/A"))
        with col2:
            st.metric("Model Version", prediction.get("model_version", "N/A"))
        with col3:
            st.metric("Accuracy", f"{prediction.get('model_accuracy', 0):.1%}")
        with col4:
            st.metric("AUC Score", f"{prediction.get('model_auc', 0):.3f}")
    
    # ========================================================================
    # TAB 2: EXPLAINABILITY
    # ========================================================================
    
    with tab2:
        st.markdown("### 🔍 AI Explanation (SHAP)")
        st.markdown("Understand which factors influenced the prediction")
        st.divider()
        
        # Get SHAP explanation
        if st.button("📊 Generate SHAP Explanation", key="gen_shap"):
            with st.spinner("Computing SHAP values..."):
                shap_data = backend.get_explainability(current_patient)
            
            if shap_data:
                state.set(f"{current_patient['patient_id']}_shap", shap_data)
            else:
                show_error("Failed to generate SHAP explanation")
        
        # Try to get stored SHAP data
        shap_data = state.get(f"{current_patient['patient_id']}_shap", None)
        
        if shap_data:
            # Extract SHAP values
            shap_values = shap_data.get("shap_values", [])
            feature_names = shap_data.get("feature_names", [])
            base_value = shap_data.get("base_value", 0.5)
            prediction_value = prediction.get("risk_score", 0.5)
            
            if shap_values and feature_names:
                # Create waterfall plot
                st.markdown("#### Feature Impact on Prediction")
                fig_waterfall = MedicalCharts.shap_waterfall(
                    feature_names=feature_names[:10],  # Top 10
                    shap_values=shap_values[:10],
                    base_value=base_value,
                    prediction_value=prediction_value,
                    title="SHAP Waterfall - Feature Contributions"
                )
                st.plotly_chart(fig_waterfall, use_container_width=True)
                
                # Feature importance bar chart
                st.markdown("#### Feature Importance (Absolute SHAP)")
                abs_shap = [abs(v) for v in shap_values]
                fig_importance = MedicalCharts.feature_importance(
                    features=feature_names[:15],
                    importance=abs_shap[:15],
                    title="Top 15 Contributing Features"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Interpretation
                st.markdown("#### How to Read This")
                st.markdown("""
                - **Waterfall Plot**: Shows how each feature pushes the prediction from the base value
                - **Red bars**: Features that increase risk
                - **Blue bars**: Features that decrease risk
                - **Height**: Magnitude of impact
                
                **Example**: If Age has a large red bar, it means Age is a strong risk factor for this patient
                """)
            else:
                st.info("SHAP data not yet generated. Click the button above.")
        else:
            st.info("Click 'Generate SHAP Explanation' above to see feature contributions")
    
    # ========================================================================
    # TAB 3: TRENDS
    # ========================================================================
    
    with tab3:
        st.markdown("### 📈 Risk Trends")
        st.markdown("Historical predictions for this patient")
        st.divider()
        
        # Fetch historical predictions
        if st.button("📊 Load History", key="load_history"):
            with st.spinner("Fetching patient history..."):
                history = backend.get_patient_history(current_patient.get("patient_id"))
            
            if history:
                state.set(f"{current_patient['patient_id']}_history", history)
        
        history = state.get(f"{current_patient['patient_id']}_history", None)
        
        if history and len(history) > 0:
            # Convert to dataframe
            df_history = pd.DataFrame(history)
            
            if "date" in df_history.columns and "risk_score" in df_history.columns:
                # Sort by date
                df_history["date"] = pd.to_datetime(df_history["date"])
                df_history = df_history.sort_values("date")
                
                # Trend chart
                fig_trend = MedicalCharts.time_series(
                    data={
                        "Risk Score": df_history["risk_score"].tolist(),
                        "Baseline": [0.5] * len(df_history),
                    },
                    dates=[d.strftime("%Y-%m-%d") for d in df_history["date"]],
                    title="Risk Score Trend"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Statistics
                st.markdown("#### Trend Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current", f"{df_history['risk_score'].iloc[-1]:.1%}")
                with col2:
                    st.metric("Average", f"{df_history['risk_score'].mean():.1%}")
                with col3:
                    st.metric("Highest", f"{df_history['risk_score'].max():.1%}")
                with col4:
                    trend_change = df_history['risk_score'].iloc[-1] - df_history['risk_score'].iloc[0]
                    st.metric("Change", f"{trend_change:+.1%}")
                
                # History table
                st.markdown("#### Full History")
                st.dataframe(
                    df_history[["date", "risk_score", "risk_label"]].sort_values("date", ascending=False),
                    use_container_width=True
                )
            else:
                st.warning("Historical data format not recognized")
        else:
            st.info("Click 'Load History' to view historical predictions")
    
    # ========================================================================
    # TAB 4: RECOMMENDATIONS
    # ========================================================================
    
    with tab4:
        st.markdown("### 💊 Clinical Recommendations")
        st.divider()
        
        recommendations = prediction.get("recommendations", [])
        
        if recommendations:
            for idx, rec in enumerate(recommendations, 1):
                if isinstance(rec, dict):
                    priority = rec.get("priority", "INFO")
                    action = rec.get("action", "")
                    reason = rec.get("reason", "")
                    
                    if priority == "CRITICAL":
                        st.error(f"🔴 **Action {idx}: {action}**\n{reason}")
                    elif priority == "HIGH":
                        st.warning(f"🟠 **Action {idx}: {action}**\n{reason}")
                    elif priority == "MODERATE":
                        st.warning(f"🟡 **Action {idx}: {action}**\n{reason}")
                    else:
                        st.info(f"ℹ️ **Action {idx}: {action}**\n{reason}")
                else:
                    st.info(f"• {rec}")
        else:
            # Generate default recommendations based on risk
            st.markdown("#### Suggested Actions")
            
            risk_score = prediction.get("risk_score", 0.0)
            
            if risk_score > 0.75:
                st.error("🔴 **CRITICAL PRIORITY**")
                st.markdown("""
                - Immediate physician consultation required
                - Consider urgent intervention
                - Monitor vital signs continuously
                - Update treatment plan
                """)
            elif risk_score > 0.6:
                st.warning("🟠 **HIGH PRIORITY**")
                st.markdown("""
                - Schedule follow-up appointment within 1 week
                - Increase monitoring frequency
                - Review current medications
                - Consider additional diagnostic tests
                """)
            elif risk_score > 0.4:
                st.warning("🟡 **MODERATE PRIORITY**")
                st.markdown("""
                - Schedule routine follow-up
                - Monitor key indicators
                - Lifestyle modifications recommended
                - Consider preventive measures
                """)
            else:
                st.success("🟢 **ROUTINE MONITORING**")
                st.markdown("""
                - Continue regular check-ups
                - Maintain current treatment
                - Healthy lifestyle recommended
                - Follow routine preventive care
                """)
    
    # ========================================================================
    # BOTTOM ACTION BUTTONS
    # ========================================================================
    
    st.divider()
    st.markdown("#### Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Results", use_container_width=True):
            st.success("Results saved to patient record")
    
    with col2:
        if st.button("📋 Generate Report", use_container_width=True):
            st.info("Report generation coming soon...")
    
    with col3:
        if st.button("➡️ Go to Results", use_container_width=True):
            st.info("Navigation feature coming soon")
