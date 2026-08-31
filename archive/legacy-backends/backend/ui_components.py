"""
Streamlit UI Components for Model Explainability
=================================================

Reusable Streamlit components for displaying SHAP explanations,
feature importance, and model predictions with visualizations.

Features:
- SHAP explanation displays
- Interactive feature importance charts
- Risk score visualization
- Prediction confidence indicators
- Clinical recommendation display
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import logging
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


class ExplainabilityUI:
    """
    Streamlit UI components for model explainability and interpretation.
    """
    
    @staticmethod
    def display_risk_score(
        risk_score: float,
        risk_level: str,
        show_gauge: bool = True
    ) -> None:
        """
        Display risk score with visual indicator.
        
        Args:
            risk_score: Prediction score (0-1)
            risk_level: Risk classification (HIGH, MODERATE, LOW)
            show_gauge: Whether to show gauge chart
        """
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Color mapping
            color_map = {
                "HIGH": "#DC143C",      # Crimson
                "MODERATE": "#FF8C00",  # Orange
                "LOW": "#28A745"        # Green
            }
            
            color = color_map.get(risk_level, "#999999")
            
            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                ">
                    <h3 style="margin: 0; font-size: 32px;">{risk_score:.1%}</h3>
                    <p style="margin: 5px 0 0 0; font-size: 18px;">{risk_level} RISK</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            if show_gauge:
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score * 100,
                    title={'text': "Risk Percentage"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 33], 'color': "#D3F3D3"},
                            {'range': [33, 67], 'color': "#FFF3CD"},
                            {'range': [67, 100], 'color': "#F8D7DA"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))
                
                fig.update_xaxes(visible=False)
                fig.update_layout(height=250, width=250)
                st.plotly_chart(fig, use_container_width=False)
    
    @staticmethod
    def display_feature_importance(
        contributions: List[Dict[str, Any]],
        max_features: int = 10,
        chart_type: str = "bar"
    ) -> None:
        """
        Display feature importance/contributions.
        
        Args:
            contributions: List of contribution dictionaries
            max_features: Maximum features to display
            chart_type: "bar" or "horizontal"
        """
        st.subheader("Key Contributing Factors")
        
        if not contributions:
            st.info("No contributing factors available")
            return
        
        # Prepare data
        df_contrib = pd.DataFrame(contributions[:max_features])
        df_contrib['direction'] = df_contrib['contribution'].apply(
            lambda x: "Increases Risk" if x > 0 else "Decreases Risk"
        )
        
        # Sort by absolute contribution
        df_contrib = df_contrib.sort_values('abs_contribution', ascending=False)
        
        # Create chart
        fig = px.bar(
            df_contrib,
            x='abs_contribution',
            y='feature',
            color='direction',
            color_discrete_map={
                "Increases Risk": "#DC143C",
                "Decreases Risk": "#28A745"
            },
            orientation='h',
            title='Feature Impact on Prediction',
            labels={
                'abs_contribution': 'Impact Magnitude',
                'feature': 'Feature',
                'direction': 'Direction'
            }
        )
        
        fig.update_layout(
            height=max(300, 30 * max_features),
            showlegend=True,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display as table
        with st.expander("View Detailed Impact Table"):
            display_df = df_contrib[[
                'feature', 'value', 'contribution', 'direction'
            ]].copy()
            display_df.columns = ['Feature', 'Current Value', 'Impact', 'Direction']
            display_df['Impact'] = display_df['Impact'].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(display_df, use_container_width=True)
    
    @staticmethod
    def display_top_factors(
        top_positive: List[Dict[str, Any]],
        top_negative: List[Dict[str, Any]]
    ) -> None:
        """
        Display top positive and negative contributing factors.
        
        Args:
            top_positive: Top risk-increasing factors
            top_negative: Top risk-decreasing factors
        """
        st.subheader("Top Influencing Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔴 Risk Increasing Factors")
            if top_positive:
                for i, factor in enumerate(top_positive[:5], 1):
                    st.markdown(f"""
                    **{i}. {factor['feature']}**
                    - Impact: {factor['contribution']:.4f}
                    - Current Value: {factor['value']:.2f}
                    """)
            else:
                st.info("No risk-increasing factors")
        
        with col2:
            st.markdown("### 🟢 Risk Decreasing Factors")
            if top_negative:
                for i, factor in enumerate(top_negative[:5], 1):
                    st.markdown(f"""
                    **{i}. {factor['feature']}**
                    - Impact: {factor['contribution']:.4f}
                    - Current Value: {factor['value']:.2f}
                    """)
            else:
                st.info("No risk-decreasing factors")
    
    @staticmethod
    def display_shap_visualization(
        image_bytes: Optional[bytes],
        title: str
    ) -> None:
        """
        Display SHAP plot visualization.
        
        Args:
            image_bytes: PNG image bytes
            title: Plot title
        """
        if image_bytes:
            st.subheader(title)
            st.image(image_bytes, use_column_width=True)
        else:
            st.warning(f"Could not generate {title}")
    
    @staticmethod
    def display_clinical_summary(clinical_summary: Dict[str, Any]) -> None:
        """
        Display clinical interpretation.
        
        Args:
            clinical_summary: Clinical interpretation dictionary
        """
        if not clinical_summary:
            return
        
        st.subheader("Clinical Interpretation")
        
        # Risk level with color
        risk_level = clinical_summary.get("risk_level", "UNKNOWN")
        color_map = {
            "HIGH": "🔴",
            "MODERATE": "🟡",
            "LOW": "🟢"
        }
        
        st.markdown(
            f"{color_map.get(risk_level, '⚫')} **{risk_level} RISK** - "
            f"{clinical_summary.get('risk_description', 'No description')}"
        )
        
        # Recommendations
        recommendations = clinical_summary.get("recommendations", [])
        if recommendations:
            st.subheader("Clinical Recommendations")
            for rec in recommendations:
                st.markdown(f"• {rec}")
    
    @staticmethod
    def display_trend_analysis(trend_data: Dict[str, Any]) -> None:
        """
        Display longitudinal trend analysis.
        
        Args:
            trend_data: Trend information dictionary
        """
        if not trend_data:
            return
        
        st.subheader("Risk Trend Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Risk",
                f"{trend_data.get('average_risk', 0):.1%}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Maximum Risk",
                f"{trend_data.get('max_risk', 0):.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Minimum Risk",
                f"{trend_data.get('min_risk', 0):.1%}",
                delta=None
            )
        
        # Trend direction
        trend_direction = trend_data.get("trend_direction", "stable").upper()
        
        color_map = {
            "INCREASING": "#DC143C",
            "DECREASING": "#28A745",
            "STABLE": "#4A90E2"
        }
        
        st.markdown(
            f"<h4 style='color: {color_map.get(trend_direction, '#999')}'>Trend: {trend_direction}</h4>",
            unsafe_allow_html=True
        )
    
    @staticmethod
    def display_prediction_confidence(
        prediction_proba: Optional[List[float]],
        prediction_class: int
    ) -> None:
        """
        Display prediction confidence levels.
        
        Args:
            prediction_proba: Class probabilities
            prediction_class: Predicted class
        """
        if not prediction_proba or len(prediction_proba) < 2:
            return
        
        st.subheader("Prediction Confidence")
        
        # Create confidence chart
        classes = ["Low Risk", "High Risk"]
        probabilities = prediction_proba
        
        fig = go.Figure(data=[
            go.Bar(
                x=classes,
                y=probabilities,
                marker_color=['#28A745', '#DC143C']
            )
        ])
        
        fig.update_layout(
            title="Model Confidence Levels",
            yaxis_title="Probability",
            xaxis_title="Prediction Class",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def display_model_comparison(
        explanations: List[Dict[str, Any]],
        patient_ids: List[str]
    ) -> None:
        """
        Compare explanations across multiple patients.
        
        Args:
            explanations: List of explanation dictionaries
            patient_ids: List of patient identifiers
        """
        st.subheader("Comparative Analysis")
        
        # Extract risk scores
        risk_scores = [e.get("prediction", 0) for e in explanations]
        
        df_compare = pd.DataFrame({
            'Patient': patient_ids,
            'Risk Score': risk_scores
        })
        
        fig = px.bar(
            df_compare,
            x='Patient',
            y='Risk Score',
            color='Risk Score',
            color_continuous_scale=['green', 'yellow', 'red'],
            title='Risk Score Comparison'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def generate_download_button(
        data: bytes,
        filename: str,
        file_type: str,
        button_label: str = "Download"
    ) -> None:
        """
        Generate download button for file.
        
        Args:
            data: File bytes
            filename: Filename for download
            file_type: MIME type (pdf, csv, png, etc.)
            button_label: Button label
        """
        st.download_button(
            label=button_label,
            data=data,
            file_name=filename,
            mime=f"application/{file_type}"
        )
    
    @staticmethod
    def display_patient_comparison_heatmap(
        patients_data: List[Dict[str, Any]],
        top_features: int = 10
    ) -> None:
        """
        Display heatmap comparing top features across patients.
        
        Args:
            patients_data: List of patient prediction data
            top_features: Number of top features to display
        """
        st.subheader("Feature Importance Heatmap")
        
        # Prepare data
        all_contributions = []
        patient_names = []
        
        for patient in patients_data:
            patient_names.append(patient.get("patient_id", "Unknown"))
            contrib = patient.get("explanation", {}).get("feature_contributions", [])[:top_features]
            all_contributions.append([c.get("abs_contribution", 0) for c in contrib])
        
        # Create heatmap
        if all_contributions:
            feature_names = [f.get("feature", f"F{i}") 
                           for f in patients_data[0].get("explanation", {}).get("feature_contributions", [])[:top_features]]
            
            fig = go.Figure(data=go.Heatmap(
                z=all_contributions,
                y=patient_names,
                x=feature_names,
                colorscale='Viridis'
            ))
            
            fig.update_layout(
                height=300,
                title="Feature Importance Across Patients"
            )
            st.plotly_chart(fig, use_container_width=True)


class ReportUI:
    """
    Streamlit UI components for report generation and display.
    """
    
    @staticmethod
    def show_report_generation_form() -> Dict[str, Any]:
        """
        Display form for report customization.
        
        Returns:
            Dictionary with selected options
        """
        st.subheader("Generate Clinical Report")
        
        with st.form("report_options"):
            col1, col2 = st.columns(2)
            
            with col1:
                include_trends = st.checkbox(
                    "Include Trend Analysis",
                    value=True,
                    help="Include longitudinal trend data"
                )
                include_visualizations = st.checkbox(
                    "Include Visualizations",
                    value=True,
                    help="Include SHAP plots"
                )
            
            with col2:
                include_recommendations = st.checkbox(
                    "Include Recommendations",
                    value=True,
                    help="Include clinical recommendations"
                )
                include_demographics = st.checkbox(
                    "Include Demographics",
                    value=True,
                    help="Include patient demographic information"
                )
            
            submitted = st.form_submit_button("Generate Report")
        
        return {
            "include_trends": include_trends,
            "include_visualizations": include_visualizations,
            "include_recommendations": include_recommendations,
            "include_demographics": include_demographics,
            "submitted": submitted
        }
    
    @staticmethod
    def show_report_preview(report_data: Dict[str, Any]) -> None:
        """
        Display report preview in Streamlit.
        
        Args:
            report_data: Report data dictionary
        """
        st.subheader("Report Preview")
        
        # Patient section
        with st.expander("Patient Information"):
            if "patient_data" in report_data:
                st.write(report_data["patient_data"])
        
        # Prediction section
        with st.expander("Risk Assessment"):
            if "prediction_data" in report_data:
                st.write(report_data["prediction_data"])
        
        # Explanation section
        with st.expander("Feature Analysis"):
            if "explanation_data" in report_data:
                st.write(report_data["explanation_data"])
        
        # Clinical summary
        with st.expander("Clinical Summary"):
            if "clinical_summary" in report_data:
                st.write(report_data["clinical_summary"])
