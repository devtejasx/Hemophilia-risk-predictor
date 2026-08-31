"""
Dashboard Page - Real-time statistics and overview
Shows key metrics, patient statistics, and recent activity
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# ============================================================================
# PATH SETUP & IMPORTS
# ============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(page_title="Dashboard", layout="wide")

from utils.session_state import init_session_state, get_session_var
from components.navbar import show_sidebar, show_page_header
from components.cards import (
    metric_card, stat_box, info_card, patient_card, 
    status_badge, divider_text, empty_state
)
from components.charts import (
    plot_risk_gauge, plot_feature_importance, plot_patient_metrics
)
from database.db import get_database
from utils.helpers import format_number, format_percentage, get_risk_level

# ============================================================================
# INITIALIZE
# ============================================================================
init_session_state()
show_sidebar()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_data(ttl=300)
def load_dashboard_data():
    """Load dashboard data from database"""
    try:
        db = get_database()
        patients = db.get_patients(limit=1000)
        return pd.DataFrame(patients) if patients else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def calculate_dashboard_stats(df: pd.DataFrame) -> dict:
    """Calculate dashboard statistics"""
    stats = {
        "total_patients": len(df),
        "high_risk": len(df[df.get("risk_score", 0) > 0.6]) if not df.empty else 0,
        "average_risk": df.get("risk_score", 0).mean() if not df.empty else 0,
        "severe_cases": len(df[df.get("severity", "") == "Severe"]) if not df.empty else 0,
    }
    return stats


# ============================================================================
# MAIN CONTENT
# ============================================================================
def main():
    show_page_header("📊 Dashboard", "Real-time Statistics & Overview")
    
    # Load data
    df = load_dashboard_data()
    
    # Handle empty state
    if df.empty:
        empty_state(
            icon="📭",
            title="No Patients Yet",
            message="Add patients to see dashboard statistics. Go to 'Add Patient' page."
        )
        return
    
    # Calculate stats
    stats = calculate_dashboard_stats(df)
    
    # ========================================================================
    # KEY METRICS ROW
    # ========================================================================
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stat_box(
            label="Total Patients",
            value=str(stats["total_patients"]),
            trend=f"↑ +{max(0, stats['total_patients']-10)}"
        )
    
    with col2:
        stat_box(
            label="High Risk Cases",
            value=str(stats["high_risk"]),
            trend="⚠️ Requires Attention"
        )
    
    with col3:
        avg_risk_pct = format_percentage(stats["average_risk"])
        stat_box(
            label="Average Risk",
            value=avg_risk_pct,
            trend="→ Stable"
        )
    
    with col4:
        stat_box(
            label="Severe Cases",
            value=str(stats["severe_cases"]),
            trend="📍 Monitor"
        )
    
    st.divider()
    
    # ========================================================================
    # RISK DISTRIBUTION
    # ========================================================================
    st.markdown("### 📊 Risk Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Risk categories
        low_risk = len(df[df.get("risk_score", 0) <= 0.30]) if not df.empty else 0
        med_risk = len(df[(df.get("risk_score", 0) > 0.30) & (df.get("risk_score", 0) <= 0.60)]) if not df.empty else 0
        high_risk = len(df[df.get("risk_score", 0) > 0.60]) if not df.empty else 0
        
        risk_data = {
            "🟢 Low": low_risk,
            "🟡 Moderate": med_risk,
            "🔴 High": high_risk
        }
        
        # Display as bars
        for risk_level, count in risk_data.items():
            st.write(f"{risk_level}: **{count}** patients")
            st.progress(count / max(stats["total_patients"], 1))
    
    with col2:
        st.markdown("### Severity Breakdown")
        severity_counts = df.get("severity", "").value_counts() if not df.empty else pd.Series()
        
        for severity, count in severity_counts.items():
            st.metric(severity, count)
    
    st.divider()
    
    # ========================================================================
    # RECENT PATIENTS
    # ========================================================================
    st.markdown("### 👥 Recent Patients")
    
    if not df.empty:
        # Show top 5 patients with highest risk
        top_patients = df.nlargest(5, "risk_score")[
            ["name", "age", "severity", "risk_score"]
        ] if "name" in df.columns else df.head(5)
        
        for idx, patient in top_patients.iterrows():
            try:
                patient_card(
                    name=patient.get("name", "Unknown"),
                    age=patient.get("age", 0),
                    severity=patient.get("severity", "Unknown"),
                    risk_score=patient.get("risk_score", 0),
                    patient_id=str(patient.get("_id", ""))
                )
            except Exception as e:
                st.warning(f"Error displaying patient: {e}")
    else:
        st.info("No recent patients to display")
    
    st.divider()
    
    # ========================================================================
    # ALERTS & NOTIFICATIONS
    # ========================================================================
    st.markdown("### 🚨 Active Alerts")
    
    alert_count = 0
    if stats["high_risk"] > 0:
        info_card(
            title="High Risk Cases",
            content=f"{stats['high_risk']} patients require immediate attention",
            icon="🔴",
            bg_color="danger"
        )
        alert_count += 1
    
    if stats["severe_cases"] > 5:
        info_card(
            title="Many Severe Cases",
            content=f"{stats['severe_cases']} severe hemophilia cases in system",
            icon="⚠️",
            bg_color="warning"
        )
        alert_count += 1
    
    if alert_count == 0:
        st.success("✅ No active alerts - system operating normally")
    
    st.divider()
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"📊 Patients in System: {stats['total_patients']}")
    
    with col2:
        st.caption(f"⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    with col3:
        st.caption(f"👤 User: {get_session_var('user_name', 'Guest')}")


if __name__ == "__main__":
    main()
