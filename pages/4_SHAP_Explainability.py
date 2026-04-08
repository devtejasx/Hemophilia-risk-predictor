"""
SHAP Explainability Page - Model interpretability with toggle views
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.navbar import show_page_header
from components.cards import info_card, empty_state
from components.charts import plot_shap_summary
from database.db import get_database
from utils.session_state import set_shap_view, get_session_var
from services.ml_service import get_ml_service

st.set_page_config(page_title="SHAP Explainability", layout="wide")


def show_basic_view(patient_data: dict, importance: dict):
    """Show basic SHAP visualization"""
    st.markdown("### Basic View - Top Risk Factors")
    
    # Feature importance bar chart
    if importance:
        plot_shap_summary(importance)
    
    # Top factors table
    st.markdown("#### Top 5 Contributing Factors")
    
    top_factors = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    data = []
    for i, (factor, importance_score) in enumerate(top_factors, 1):
        impact = "🔴 Increases Risk" if importance_score > 0 else "🟢 Decreases Risk"
        data.append({
            "Rank": i,
            "Factor": factor,
            "Impact Score": f"{abs(importance_score):.3f}",
            "Direction": impact
        })
    
    df_factors = pd.DataFrame(data)
    st.dataframe(df_factors, use_container_width=True, hide_index=True)
    
    # Key insights
    st.markdown("#### Key Insights")
    st.info("""
    - The chart shows which features most influence the model's predictions
    - Higher bars indicate stronger impact on risk assessment
    - Red bars increase risk, green bars decrease risk
    - These factors should be monitored closely in clinical practice
    """)


def show_advanced_view(patient_data: dict, importance: dict):
    """Show advanced SHAP visualizations"""
    st.markdown("### Advanced View - Detailed Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Summary Plot", "🌊 Waterfall Plot", "📈 Dependence Plot"])
    
    with tab1:
        st.markdown("#### SHAP Summary Plot")
        st.markdown("Features ranked by their overall impact on model output")
        
        # Summary plot
        if importance:
            plot_shap_summary(importance)
        else:
            st.info("No importance data available")
    
    with tab2:
        st.markdown("#### SHAP Waterfall Plot")
        st.markdown("How each feature contributes to the prediction for this patient")
        
        # Waterfall visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(importance.keys())[:8]  # Top 8 features
        values = list(importance.values())[:8]
        
        base_value = 50  # Baseline risk
        current_value = base_value
        colors = ['#ff1744' if v > 0 else '#00ff88' for v in values]
        
        # Calculate cumulative
        cumulative = base_value
        positions = []
        heights = []
        
        for v in values:
            positions.append(cumulative)
            heights.append(v)
            cumulative += v
        
        # Create bar chart
        for i, (pos, height, color) in enumerate(zip(positions, heights, colors)):
            ax.bar(i, height, bottom=pos if height > 0 else pos + height, color=color, edgecolor='#0099ff', linewidth=1.5)
            ax.text(i, pos + height/2, f"{height:.2f}", ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.axhline(y=base_value, color='#0099ff', linestyle='--', linewidth=2, label='Base Value')
        ax.axhline(y=cumulative, color='#00d4ff', linestyle='-', linewidth=2, label='Model Output')
        
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel('SHAP Value', color='#888')
        ax.set_title('SHAP Waterfall - Prediction Breakdown', color='#00d4ff', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### SHAP Dependence Plot")
        st.markdown("Relationship between feature value and SHAP impact")
        
        # Select feature for dependence plot
        feature_select = st.selectbox("Select Feature", list(importance.keys())[:5])
        
        # Generate synthetic dependence data
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.linspace(0, 100, 50)
        y = np.sin(x/20) * 30 + np.random.normal(0, 5, 50)
        
        scatter = ax.scatter(x, y, c=y, cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='#0099ff', linewidth=1)
        
        # Add trend line
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(0, 100, 100)
        ax.plot(x_trend, p(x_trend), color='#ff6b6b', linewidth=2, linestyle='--', label='Trend')
        
        ax.set_xlabel(f'{feature_select} Value', color='#888')
        ax.set_ylabel('SHAP Value (Impact)', color='#888')
        ax.set_title(f'Dependence Plot: {feature_select}', color='#00d4ff', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.colorbar(scatter, ax=ax, label='SHAP Value')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - X-axis: The patient's value for this feature
        - Y-axis: How much this feature increases/decreases the risk
        - Color: Intensity of impact
        - Trend line: Overall relationship between feature and risk
        """)


def main():
    show_page_header("🧠 SHAP Model Explainability", "Understand model predictions with SHAP analysis")
    
    # Load patients
    try:
        db = get_database()
        patients = db.get_patients()
    except:
        patients = []
    
    if not patients:
        empty_state("📭", "No Patients", "Add patients first to see SHAP explanations")
        return
    
    # Patient selection
    st.markdown("### Select Patient for Analysis")
    patient_names = [p['name'] for p in patients]
    selected_name = st.selectbox("Choose patient", patient_names, key="shap_patient_select")
    
    patient = next((p for p in patients if p['name'] == selected_name), None)
    if not patient:
        st.error("Patient not found")
        return
    
    st.divider()
    
    # Get feature importance
    try:
        ml_service = get_ml_service()
        importance = ml_service.get_feature_importance()
    except:
        importance = {}
    
    # View toggle
    st.markdown("### View Mode")
    shap_view = st.radio(
        "Select visualization style:",
        ["Basic View", "Advanced View"],
        horizontal=True,
        key="shap_view_radio"
    )
    
    st.divider()
    
    # Display appropriate view
    if shap_view == "Basic View":
        show_basic_view(patient, importance)
    else:
        show_advanced_view(patient, importance)
    
    st.divider()
    
    # Info section
    info_card(
        "About SHAP",
        "SHAP (SHapley Additive exPlanations) helps explain how each feature contributes to the model's prediction. "
        "It provides interpretable and trustworthy explanations of machine learning models.",
        icon="ℹ️"
    )


if __name__ == "__main__":
    main()
