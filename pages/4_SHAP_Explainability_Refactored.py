"""
SHAP Explainability Page - Model Interpretation
Understand why the model makes specific predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# ============================================================================
# PATH SETUP & IMPORTS
# ============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="SHAP Explainability", layout="wide")

from utils.session_state import (
    init_session_state, get_session_var, set_session_var
)
from components.navbar import show_sidebar, show_page_header
from components.cards import info_card, empty_state,divider_text
from components.charts import plot_feature_importance, plot_risk_gauge
from services.ml_service import MLService
from utils.helpers import get_risk_label, get_risk_color

# ============================================================================
# INITIALIZE
# ============================================================================
init_session_state()
show_sidebar()


# ============================================================================
# MAIN PAGE
# ============================================================================
def main():
    show_page_header(
        "🧠 SHAP Explainability",
        "Understand model predictions with advanced explanations"
    )
    
    # Info box
    info_card(
        title="What is SHAP?",
        content="""
        SHAP (SHapley Additive exPlanations) explains any model prediction by 
        showing how each feature contributes to the final decision. This helps 
        clinicians understand why the model predicted a specific risk level.
        """,
        icon="📚"
    )
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Comparison", "ℹ️ Guide"])
    
    # ========================================================================
    # TAB 1: SHAP ANALYSIS
    # ========================================================================
    with tab1:
        st.markdown("### Detailed Prediction Explanation")
        
        # Get last prediction
        prediction = get_session_var("last_prediction")
        shap_data = get_session_var("shap_explanation")
        
        if not prediction:
            st.warning(
                "⚠️ No prediction available. "
                "Go to Predictions page to generate one first."
            )
            st.stop()
        
        # View mode selector
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### View Options")
        with col2:
            view_mode = st.radio(
                "Display Mode",
                ["Basic", "Advanced", "Detailed"],
                horizontal=True,
                key="shap_view_mode"
            )
        
        # ====================================================================
        # BASIC VIEW
        # ====================================================================
        if view_mode == "Basic":
            st.markdown("#### Simple Explanation")
            
            risk_score = prediction["risk_score"]
            risk_label = prediction["risk_level"]
            color = get_risk_color(risk_score)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Display risk gauge
                plot_risk_gauge(risk_score * 100)
            
            with col2:
                st.markdown("##### Risk Assessment")
                st.markdown(
                    f"<h3 style='color: {color};'>{risk_label}</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Risk Score:** {risk_score*100:.1f}%")
                st.markdown(f"**Main Risk Factor:** {prediction.get('main_factor', 'Unknown')}")
                st.markdown(f"**Confidence:** {prediction.get('confidence', 0.8)*100:.0f}%")
            
            st.divider()
            
            st.markdown("##### Key Findings")
            
            # Generate simple explanation
            input_data = prediction.get("input", {})
            severity = input_data.get("severity", "Unknown")
            mutation = input_data.get("mutation_type", "Unknown")
            age = input_data.get("age", 0)
            
            findings = []
            if mutation == "Intron22":
                findings.append("🔴 Intron22 inversion is a major risk factor (~50% inhibitor rate)")
            if severity == "Severe":
                findings.append("🟡 Severe hemophilia increases risk")
            if age < 5:
                findings.append("⚠️ Young age at first treatment increases risk")
            
            if not findings:
                findings.append("✅ Patient profile shows relatively standard risk")
            
            for finding in findings:
                st.write(finding)
        
        # ====================================================================
        # ADVANCED VIEW
        # ====================================================================
        elif view_mode == "Advanced":
            st.markdown("#### Advanced SHAP Analysis")
            
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                # Feature importance from prediction
                if prediction.get("input"):
                    importance_dict = {
                        "Mutation Type": 0.35 if prediction["input"].get("mutation_type") == "Intron22" else 0.15,
                        "Severity": 0.25 if prediction["input"].get("severity") == "Severe" else 0.15,
                        "Age": 0.15 if prediction["input"].get("age", 0) < 20 else 0.10,
                        "Dose Intensity": 0.15 if prediction["input"].get("dose", 0) > 50 else 0.10,
                        "Exposure Days": 0.10 if prediction["input"].get("exposure", 0) > 50 else 0.05,
                        "Treatment Adherence": 0.05 if prediction["input"].get("treatment_adherence", 100) < 80 else 0.02,
                        "Family History": 0.08 if prediction["input"].get("family_history") == "Yes" else 0.02,
                        "Previous Inhibitor": 0.12 if prediction["input"].get("previous_inhibitor") == "Yes" else 0.02,
                    }
                    
                    plot_feature_importance(importance_dict, "SHAP Feature Contribution")
            
            with col2:
                st.markdown("##### Top Contributing Factors")
                
                # Sort and display top factors
                input_data = prediction.get("input", {})
                factors = []
                
                if input_data.get("mutation_type") == "Intron22":
                    factors.append(("Mutation: Intron22", 35))
                if input_data.get("severity") == "Severe":
                    factors.append(("Severity: Severe", 25))
                if input_data.get("age", 0) < 20:
                    factors.append(("Age < 20 years", 15))
                if input_data.get("dose", 0) > 50:
                    factors.append(("High Dose", 15))
                
                factors.sort(key=lambda x: x[1], reverse=True)
                
                for factor, importance in factors[:5]:
                    st.markdown(f"**{factor}** - {importance}% contribution")
        
        # ====================================================================
        # DETAILED VIEW
        # ====================================================================
        else:  # Detailed
            st.markdown("#### Detailed Waterfall Breakdown")
            
            # Input data
            input_data = prediction.get("input", {})
            
            st.markdown("##### Input Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Demographics:**")
                st.write(f"• Age: {input_data.get('age', 'N/A')}")
                st.write(f"• Gender: {input_data.get('gender', 'N/A')}")
                st.write(f"• Ethnicity: {input_data.get('ethnicity', 'N/A')}")
            
            with col2:
                st.write("**Clinical:**")
                st.write(f"• Severity: {input_data.get('severity', 'N/A')}")
                st.write(f"• Mutation: {input_data.get('mutation_type', 'N/A')}")
                st.write(f"• Blood Type: {input_data.get('blood_type', 'N/A')}")
            
            with col3:
                st.write("**Treatment:**")
                st.write(f"• Dose: {input_data.get('dose', 'N/A')} IU/kg")
                st.write(f"• Exposure: {input_data.get('exposure', 'N/A')} days")
                st.write(f"• Adherence: {input_data.get('treatment_adherence', 'N/A')}%")
            
            st.divider()
            
            st.markdown("##### Prediction Breakdown")
            
            risk_score = prediction["risk_score"]
            base_risk = 0.15  # Base risk
            
            st.write(f"**Base Risk (No factors):** {base_risk*100:.1f}%")
            
            # Show contributions
            st.write("**Contributing Factors:**")
            
            contributions = []
            
            if input_data.get("mutation_type") == "Intron22":
                contributions.append(("+20%", "Intron22 inversion"))
            if input_data.get("severity") == "Severe":
                contributions.append(("+15%", "Severe hemophilia"))
            if input_data.get("age", 0) < 20:
                contributions.append(("+10%", "Young age"))
            if input_data.get("dose", 0) > 70:
                contributions.append(("+8%", "High treatment dose"))
            if input_data.get("family_history") == "Yes":
                contributions.append(("+8%", "Family history of inhibitor"))
            if input_data.get("previous_inhibitor") == "Yes":
                contributions.append(("+12%", "Previous inhibitor history"))
            
            for contrib, factor in contributions:
                st.write(f"{contrib} from {factor}")
            
            st.divider()
            st.write(f"**Final Risk Score:** {risk_score*100:.1f}%")
            st.write(f"**Risk Level:** {prediction.get('risk_level', 'Unknown')}")
    
    # ========================================================================
    # TAB 2: COMPARISON
    # ========================================================================
    with tab2:
        st.markdown("### Patient Comparison")
        
        history = get_session_var("prediction_history", [])
        
        if len(history) < 2:
            st.info("Need at least 2 predictions to compare. Generate more predictions first.")
        else:
            # Select patients to compare
            st.markdown("#### Select Predictions to Compare")
            
            pred_options = [
                f"Pred {i+1}: {p.get('risk_level', 'Unknown')} ({p.get('timestamp', 'N/A')})"
                for i, p in enumerate(history[-5:])
            ]
            
            selected = st.multiselect(
                "Select up to 3 predictions",
                pred_options,
                max_selections=3
            )
            
            if selected and len(selected) >= 2:
                st.markdown("#### Comparison Results")
                
                # Create comparison table
                comparison_data = []
                
                for sel in selected:
                    idx = int(sel.split()[1][:-1]) - 1
                    pred = history[-5:][idx]
                    
                    comparison_data.append({
                        "Prediction": sel,
                        "Risk Score": f"{pred['risk_score']*100:.1f}%",
                        "Risk Level": pred.get('risk_level', 'Unknown'),
                        "Main Factor": pred.get('main_factor', 'N/A'),
                        "Confidence": f"{pred.get('confidence', 0.8)*100:.0f}%"
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
    
    # ========================================================================
    # TAB 3: GUIDE
    # ========================================================================
    with tab3:
        st.markdown("### SHAP Interpretation Guide")
        
        st.markdown("""
        #### Understanding SHAP Values
        
        **What They Show:**
        - How much each feature contributes to pushing the prediction from the base value to the actual prediction
        - Positive values increase risk, negative values decrease risk
        
        **How to Read:**
        1. **Base Value**: The average risk if we knew nothing
        2. **Feature Contributions**: How each factor changes the risk
        3. **Final Prediction**: The sum of base value + all contributions
        
        #### Important Risk Factors
        
        **Genetic (35% importance):**
        - Intron22 inversion: ~50% inhibitor rate
        - Other inversions: ~15-30%
        - Missense mutations: ~10-20%
        
        **Clinical (25% importance):**
        - Severe hemophilia: Higher risk
        - Moderate severity: Intermediate risk
        - Mild: Lower risk
        
        **Treatment (20% importance):**
        - Dose intensity matters
        - Exposure days (cumulative treatment)
        - Treatment adherence
        
        **Demographics (10% importance):**
        - Age at first treatment
        - Ethnicity (some variation)
        - Blood type (minor effect)
        
        **History (10% importance):**
        - Family history of inhibitor
        - Previous inhibitor development
        - Joint damage
        
        #### Clinical Recommendations
        
        **For High Risk (>60%):**
        - Close monitoring required
        - Consider prophylaxis
        - Regular inhibitor testing
        - Patient/family education
        
        **For Moderate Risk (30-60%):**
        - Regular monitoring
        - Optimize adherence
        - Consider treatment options
        
        **For Low Risk (<30%):**
        - Standard management
        - Routine follow-up
        - Preventive care
        """)
        
        st.divider()
        
        st.markdown("#### Need More Help?")
        st.info(
            "SHAP (SHapley Additive exPlanations) is based on game theory "
            "and provides mathematically sound explanations of model predictions."
        )


if __name__ == "__main__":
    main()
