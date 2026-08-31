"""
Dashboard Page - Overview and statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.navbar import show_page_header
from components.cards import metric_card, stat_box, info_card, empty_state
from components.charts import plot_patient_metrics
from database.db import get_database
from utils.helpers import format_number, get_risk_level

# Page config
st.set_page_config(page_title="Dashboard", layout="wide")


def load_dashboard_data():
    """Load dashboard data from database"""
    try:
        db = get_database()
        patients = db.get_patients(limit=1000)
        return pd.DataFrame(patients) if patients else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def main():
    show_page_header("📊 Dashboard", "Overview and Key Metrics")
    
    # Load data
    df = load_dashboard_data()
    
    if df.empty:
        empty_state("📭", "No Data Available", "Add patients to see dashboard statistics")
        return
    
    # Key Metrics Row
    st.markdown("### 📈 Key Metrics")
    colu1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card(
            "Total Patients",
            str(len(df)),
            "all registered patients",
            icon="👥"
        )
    
    with col2:
        avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0
        metric_card(
            "Average Risk",
            f"{avg_risk:.1f}%",
            f"across all patients",
            icon="⚠️"
        )
    
    with col3:
        high_risk = len(df[df['risk_score'] > 70]) if 'risk_score' in df.columns else 0
        metric_card(
            "High Risk Patients",
            str(high_risk),
            f"{(high_risk/len(df)*100):.1f}% of total",
            icon="🔴"
        )
    
    with col4:
        recent_date = df['created_at'].max() if 'created_at' in df.columns else "N/A"
        metric_card(
            "Last Update",
            str(recent_date)[:10],
            "latest patient added",
            icon="⏰"
        )
    
    st.divider()
    
    # Statistics Section
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Severity Distribution")
        if 'severity' in df.columns:
            severity_counts = df['severity'].value_counts()
            st.bar_chart(severity_counts)
        else:
            st.info("No severity data available")
    
    with col2:
        st.markdown("#### Risk Score Distribution")
        if 'risk_score' in df.columns:
            st.histogram(df['risk_score'].dropna(), bins=20, title="Risk Scores")
        else:
            st.info("No risk score data available")
    
    st.divider()
    
    # Recent Patients
    st.markdown("### 👥 Recent Patients")
    if not df.empty:
        # Select columns to display
        display_cols = ['name', 'age', 'severity', 'mutation', 'risk_score']
        available_cols = [col for col in display_cols if col in df.columns]
        
        if available_cols:
            recent_df = df[available_cols].head(10)
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No patient data available")
    else:
        empty_state("📭", "No Patients", "Add patients to view recent records")
    
    # Summary Stats
    st.divider()
    st.markdown("### 📋 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 15px; background-color: #1a1f3a; border-radius: 8px;'>
            <h4 style='color: #888; margin: 0;'>Avg Age</h4>
            <h2 style='color: #00d4ff; margin: 5px 0;'>
                {:.1f}
            </h2>
        </div>
        """.format(df['age'].mean() if 'age' in df.columns else 0), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 15px; background-color: #1a1f3a; border-radius: 8px;'>
            <h4 style='color: #888; margin: 0;'>Total Records</h4>
            <h2 style='color: #00d4ff; margin: 5px 0;'>
                {}
            </h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col3:
        severe_count = len(df[df['severity'] == 'Severe']) if 'severity' in df.columns else 0
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: #1a1f3a; border-radius: 8px;'>
            <h4 style='color: #888; margin: 0;'>Severe Cases</h4>
            <h2 style='color: #ff1744; margin: 5px 0;'>
                {severe_count}
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        compliance_avg = df['treatment_adherence'].mean() if 'treatment_adherence' in df.columns else 0
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: #1a1f3a; border-radius: 8px;'>
            <h4 style='color: #888; margin: 0;'>Avg Compliance</h4>
            <h2 style='color: #00ff88; margin: 5px 0;'>
                {compliance_avg:.1f}%
            </h2>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
