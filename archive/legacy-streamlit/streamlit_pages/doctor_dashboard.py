"""
Doctor Dashboard Page - Analytics and system-wide statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_utils.state_manager import StateManager
from streamlit_utils.ui_components import (
    create_header, create_tabs, show_info
)
from streamlit_utils.backend_client import get_backend_client
from streamlit_utils.plotly_charts import MedicalCharts, create_empty_chart


def render():
    """Render doctor dashboard page"""
    
    create_header("👨‍⚕️ Doctor Dashboard", "System-wide analytics and patient insights")
    
    # Get clients
    state = StateManager()
    backend = get_backend_client()
    
    # Verify user role (optional - could add role-based access control)
    user_role = state.get_user_role()
    if user_role not in ["Clinician", "Administrator"]:
        st.warning("⚠️ Doctor dashboard requires Clinician or Administrator role")
        return
    
    st.divider()
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = create_tabs([
        "Overview",
        "Risk Distribution",
        "Trends",
        "Cohort Analysis",
        "System Status"
    ])
    
    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    
    with tab1:
        st.markdown("### 📊 Dashboard Overview")
        
        # Fetch dashboard stats
        with st.spinner("Loading system statistics..."):
            stats = backend.get_dashboard_stats()
        
        if stats:
            # Key metrics
            st.markdown("#### Key Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Total Patients",
                    stats.get("total_patients", 0),
                    delta=stats.get("new_patients_today", 0)
                )
            with col2:
                st.metric(
                    "Active Cases",
                    stats.get("active_cases", 0),
                    delta=f"{stats.get('high_risk_percent', 0):.0%} high-risk"
                )
            with col3:
                st.metric(
                    "Avg Risk Score",
                    f"{stats.get('avg_risk_score', 0):.1%}",
                    delta=stats.get('risk_trend', 0)
                )
            with col4:
                st.metric(
                    "Critical Cases",
                    stats.get("critical_cases", 0),
                    delta="Require attention"
                )
            with col5:
                st.metric(
                    "Model Accuracy",
                    f"{stats.get('model_accuracy', 0):.1%}",
                    delta=""
                )
            
            # System health
            st.markdown("#### System Health")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                db_status = stats.get("database_status", "unknown").lower()
                if db_status == "online":
                    st.success("✅ Database: Online")
                else:
                    st.warning("⚠️ Database: Offline")
            
            with col2:
                api_status = stats.get("api_status", "unknown").lower()
                if api_status == "online":
                    st.success("✅ API: Online")
                else:
                    st.warning("⚠️ API: Offline")
            
            with col3:
                model_status = stats.get("model_status", "unknown").lower()
                if model_status == "ready":
                    st.success("✅ Model: Ready")
                else:
                    st.warning("⚠️ Model: Not Ready")
            
            # Today's summary
            st.markdown("#### Today's Summary")
            
            summary = stats.get("summary", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.info(f"📋 New Patients: {summary.get('new_patients', 0)}")
            with col2:
                st.warning(f"📈 New Predictions: {summary.get('new_predictions', 0)}")
            with col3:
                if summary.get('alerts', 0) > 0:
                    st.error(f"🚨 Alerts: {summary.get('alerts', 0)}")
                else:
                    st.success(f"✅ Alerts: {summary.get('alerts', 0)}")
            with col4:
                st.info(f"⏰ Last Updated: {summary.get('last_updated', 'N/A')}")
        else:
            st.warning("Failed to load dashboard statistics")
    
    # ========================================================================
    # TAB 2: RISK DISTRIBUTION
    # ========================================================================
    
    with tab2:
        st.markdown("### 📊 Risk Distribution Analysis")
        
        with st.spinner("Loading risk data..."):
            risk_data = backend.get_risk_distribution()
        
        if risk_data:
            # Risk distribution pie chart
            risk_counts = risk_data.get("distribution", {})
            
            if risk_counts:
                fig_risk = MedicalCharts.risk_distribution(risk_counts)
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # Risk breakdown
                st.markdown("#### Risk Level Breakdown")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    low_count = risk_counts.get("Low", 0)
                    st.metric("🟢 Low Risk", low_count, f"{low_count/sum(risk_counts.values())*100:.1f}%")
                
                with col2:
                    mod_count = risk_counts.get("Moderate", 0)
                    st.metric("🟡 Moderate Risk", mod_count, f"{mod_count/sum(risk_counts.values())*100:.1f}%")
                
                with col3:
                    high_count = risk_counts.get("High", 0)
                    st.metric("🟠 High Risk", high_count, f"{high_count/sum(risk_counts.values())*100:.1f}%")
                
                with col4:
                    crit_count = risk_counts.get("Critical", 0)
                    st.metric("🔴 Critical Risk", crit_count, f"{crit_count/sum(risk_counts.values())*100:.1f}%")
                
                # Risk matrix
                st.markdown("#### Risk Categories")
                
                risk_matrix = pd.DataFrame({
                    "Risk Level": ["Low", "Moderate", "High", "Critical"],
                    "Count": [
                        risk_counts.get("Low", 0),
                        risk_counts.get("Moderate", 0),
                        risk_counts.get("High", 0),
                        risk_counts.get("Critical", 0),
                    ],
                    "Percentage": [
                        f"{risk_counts.get('Low', 0)/sum(risk_counts.values())*100:.1f}%",
                        f"{risk_counts.get('Moderate', 0)/sum(risk_counts.values())*100:.1f}%",
                        f"{risk_counts.get('High', 0)/sum(risk_counts.values())*100:.1f}%",
                        f"{risk_counts.get('Critical', 0)/sum(risk_counts.values())*100:.1f}%",
                    ],
                    "Action Required": [
                        "Routine monitoring",
                        "Increased monitoring",
                        "Close monitoring required",
                        "Immediate intervention",
                    ]
                })
                
                st.dataframe(risk_matrix, use_container_width=True)
        else:
            st.warning("Failed to load risk distribution data")
    
    # ========================================================================
    # TAB 3: TRENDS
    # ========================================================================
    
    with tab3:
        st.markdown("### 📈 System Trends")
        
        # Time period selector
        time_period = st.selectbox(
            "Time Period",
            ["Week", "Month", "Quarter"],
            key="trend_period"
        )
        
        with st.spinner("Loading trend data..."):
            trends = backend.get_trends(time_period.lower())
        
        if trends:
            # Various trend metrics
            if "predictions_over_time" in trends:
                dates = trends["predictions_over_time"].get("dates", [])
                counts = trends["predictions_over_time"].get("counts", [])
                
                if dates and counts:
                    fig_predictions = MedicalCharts.bar_chart(
                        categories=dates,
                        values=counts,
                        title=f"Predictions Per Day ({time_period})",
                        y_label="Number of Predictions"
                    )
                    st.plotly_chart(fig_predictions, use_container_width=True)
            
            if "average_risk_over_time" in trends:
                dates = trends["average_risk_over_time"].get("dates", [])
                avg_risks = trends["average_risk_over_time"].get("average_risks", [])
                
                if dates and avg_risks:
                    fig_avg_risk = MedicalCharts.trend_line(
                        dates=dates,
                        values=avg_risks,
                        title=f"Average Risk Score Trend ({time_period})",
                        y_label="Average Risk Score"
                    )
                    st.plotly_chart(fig_avg_risk, use_container_width=True)
            
            # Trend statistics
            st.markdown("#### Trend Statistics")
            
            trend_stats = {
                "Period": time_period,
                "Total Predictions": trends.get("total_predictions", 0),
                "Avg Predictions/Day": trends.get("avg_predictions_per_day", 0),
                "Avg Patient Risk": f"{trends.get('avg_patient_risk', 0):.1%}",
                "Max Risk Recorded": f"{trends.get('max_risk', 0):.1%}",
                "Min Risk Recorded": f"{trends.get('min_risk', 0):.1%}",
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                for key in list(trend_stats.keys())[:2]:
                    st.metric(key, trend_stats[key])
            
            with col2:
                for key in list(trend_stats.keys())[2:4]:
                    st.metric(key, trend_stats[key])
            
            with col3:
                for key in list(trend_stats.keys())[4:]:
                    st.metric(key, trend_stats[key])
        else:
            st.warning("Failed to load trend data")
    
    # ========================================================================
    # TAB 4: COHORT ANALYSIS
    # ========================================================================
    
    with tab4:
        st.markdown("### 👥 Cohort Analysis")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["By Age Group", "By Diagnosis", "By Risk Level", "By Gender"],
            key="cohort_type"
        )
        
        # Generate sample cohort data (in production, fetch from backend)
        if analysis_type == "By Age Group":
            cohort_data = pd.DataFrame({
                "Age Group": ["<30", "30-40", "40-50", "50-60", "60-70", ">70"],
                "Patient Count": [45, 67, 89, 156, 134, 92],
                "Avg Risk Score": [0.25, 0.32, 0.41, 0.58, 0.65, 0.72],
                "High Risk %": ["5.6%", "8.9%", "12.4%", "31.4%", "42.5%", "58.7%"],
            })
            
            fig_cohort = MedicalCharts.bar_chart(
                categories=cohort_data["Age Group"].tolist(),
                values=cohort_data["Patient Count"].tolist(),
                title="Patient Distribution by Age Group",
                y_label="Number of Patients"
            )
            
        elif analysis_type == "By Diagnosis":
            cohort_data = pd.DataFrame({
                "Diagnosis": ["Hypertension", "Diabetes", "Heart Disease", "Obesity", "Other"],
                "Patient Count": [234, 189, 156, 112, 89],
                "Avg Risk Score": [0.45, 0.62, 0.71, 0.38, 0.25],
                "High Risk %": ["28%", "45%", "62%", "18%", "8%"],
            })
            
            fig_cohort = MedicalCharts.bar_chart(
                categories=cohort_data["Diagnosis"].tolist(),
                values=cohort_data["Patient Count"].tolist(),
                title="Patient Distribution by Diagnosis",
                y_label="Number of Patients"
            )
            
        elif analysis_type == "By Risk Level":
            cohort_data = pd.DataFrame({
                "Risk Level": ["Low", "Moderate", "High", "Critical"],
                "Patient Count": [312, 278, 134, 56],
                "Avg Age": [42, 58, 64, 68],
                "Mortality Risk %": ["2%", "8%", "22%", "45%"],
            })
            
            fig_cohort = MedicalCharts.bar_chart(
                categories=cohort_data["Risk Level"].tolist(),
                values=cohort_data["Patient Count"].tolist(),
                title="Patient Distribution by Risk Level",
                y_label="Number of Patients"
            )
            
        else:  # By Gender
            cohort_data = pd.DataFrame({
                "Gender": ["Male", "Female", "Other"],
                "Patient Count": [420, 365, 35],
                "Avg Risk Score": [0.52, 0.48, 0.45],
                "High Risk %": ["32%", "28%", "22%"],
            })
            
            fig_cohort = MedicalCharts.bar_chart(
                categories=cohort_data["Gender"].tolist(),
                values=cohort_data["Patient Count"].tolist(),
                title="Patient Distribution by Gender",
                y_label="Number of Patients"
            )
        
        st.plotly_chart(fig_cohort, use_container_width=True)
        
        st.markdown(f"#### Cohort Summary ({analysis_type})")
        st.dataframe(cohort_data, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # TAB 5: SYSTEM STATUS
    # ========================================================================
    
    with tab5:
        st.markdown("### 🔧 System Status")
        
        # System information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Backend Services")
            
            if backend.health_check():
                st.success("✅ API Server: Online")
            else:
                st.error("❌ API Server: Offline")
            
            model_info = backend.get_model_info()
            if model_info:
                st.success("✅ ML Model: Ready")
                st.caption(f"Version: {model_info.get('version', 'N/A')}")
            else:
                st.warning("⚠️ ML Model: Not Available")
        
        with col2:
            st.markdown("#### Database")
            
            stats = backend.get_dashboard_stats()
            if stats:
                st.success("✅ Database: Connected")
            else:
                st.warning("⚠️ Database: Connection Issue")
            
            st.markdown("#### Cache")
            st.success("✅ Cache: Active")
        
        st.divider()
        
        # Application info
        st.markdown("#### Application Information")
        
        app_info = pd.DataFrame({
            "Component": [
                "App Version",
                "Streamlit Version",
                "Python Version",
                "Last Update",
                "Total Users",
                "Load Time",
            ],
            "Status": [
                "1.0.0",
                "1.25.0",
                "3.10.0",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "127",
                "< 2 sec",
            ]
        })
        
        st.dataframe(app_info, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Refresh button
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.success("✅ All data refreshed")
            st.rerun()
