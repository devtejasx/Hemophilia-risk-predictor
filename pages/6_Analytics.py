"""
Analytics Page - Advanced analytics and reporting
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.navbar import show_page_header
from components.cards import metric_card, empty_state
from components.charts import plot_patient_metrics, plot_correlation_heatmap
from database.db import get_database
from utils.helpers import convert_df_to_csv, format_number

st.set_page_config(page_title="Analytics", layout="wide")


def main():
    show_page_header("📈 Advanced Analytics", "Detailed data analysis and reporting")
    
    # Load data
    try:
        db = get_database()
        patients = db.get_patients()
        df = pd.DataFrame(patients) if patients else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    if df.empty:
        empty_state("📭", "No Data", "Add patients to see analytics")
        return
    
    st.markdown("### 🔍 Filters & Controls")
    
    # Filters section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=df['severity'].unique() if 'severity' in df.columns else [],
            default=df['severity'].unique() if 'severity' in df.columns else []
        )
    
    with col2:
        if 'risk_score' in df.columns:
            risk_range = st.slider(
                "Risk Score Range",
                0.0, 100.0,
                (0.0, 100.0),
                step=5.0
            )
        else:
            risk_range = (0, 100)
    
    with col3:
        if 'age' in df.columns:
            age_range = st.slider(
                "Age Range",
                int(df['age'].min()) if df['age'].dtype in [np.int64, np.float64] else 0,
                int(df['age'].max()) if df['age'].dtype in [np.int64, np.float64] else 100,
                (int(df['age'].min()) if df['age'].dtype in [np.int64, np.float64] else 0,
                 int(df['age'].max()) if df['age'].dtype in [np.int64, np.float64] else 100),
                step=5
            )
        else:
            age_range = (0, 100)
    
    # Apply filters
    filtered_df = df.copy()
    
    if severity_filter and 'severity' in df.columns:
        filtered_df = filtered_df[filtered_df['severity'].isin(severity_filter)]
    
    if 'risk_score' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['risk_score'] >= risk_range[0]) &
            (filtered_df['risk_score'] <= risk_range[1])
        ]
    
    if 'age' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['age'] >= age_range[0]) &
            (filtered_df['age'] <= age_range[1])
        ]
    
    st.divider()
    
    # Display filtered count
    st.markdown(f"### 📊 Results: {len(filtered_df)} patients")
    
    # Key metrics
    st.markdown("#### Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card(
            "Total Patients",
            str(len(filtered_df)),
            f"of {len(df)} total",
            icon="👥"
        )
    
    with col2:
        avg_age = filtered_df['age'].mean() if 'age' in filtered_df.columns else 0
        metric_card(
            "Average Age",
            f"{avg_age:.1f}",
            "years",
            icon="📅"
        )
    
    with col3:
        avg_risk = filtered_df['risk_score'].mean() if 'risk_score' in filtered_df.columns else 0
        metric_card(
            "Average Risk",
            f"{avg_risk:.1f}%",
            "all patients",
            icon="⚠️"
        )
    
    with col4:
        high_risk_count = len(filtered_df[filtered_df['risk_score'] > 70]) if 'risk_score' in filtered_df.columns else 0
        metric_card(
            "High Risk",
            str(high_risk_count),
            f"{safe_divide(high_risk_count, len(filtered_df)) * 100:.1f}%",
            icon="🔴"
        )
    
    st.divider()
    
    # Visualizations
    st.markdown("### 📊 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Severity Distribution")
        if 'severity' in filtered_df.columns:
            severity_counts = filtered_df['severity'].value_counts()
            st.bar_chart(severity_counts)
        else:
            st.info("No severity data available")
    
    with col2:
        st.markdown("#### Risk Distribution")
        if 'risk_score' in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(filtered_df['risk_score'].dropna(), bins=20, color='#00d4ff', alpha=0.7, edgecolor='#0099ff')
            ax.set_xlabel('Risk Score', color='#888')
            ax.set_ylabel('Number of Patients', color='#888')
            ax.set_title('Risk Score Distribution', color='#00d4ff', fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No risk data available")
    
    st.divider()
    
    # Data table
    st.markdown("### 📋 Patient Data")
    
    display_cols = ['name', 'age', 'severity', 'mutation', 'risk_score', 'treatment_adherence']
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    
    if available_cols:
        st.dataframe(
            filtered_df[available_cols],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No data available to display")
    
    st.divider()
    
    # Export options
    st.markdown("### 💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download CSV"):
            csv_data = convert_df_to_csv(filtered_df)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"hemophilia_analytics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Excel"):
            st.info("Excel export feature coming soon")
    
    with col3:
        if st.button("📄 Generate Report"):
            st.info("PDF report generation coming soon")


def safe_divide(num, denom, default=0):
    """Safe division"""
    return num / denom if denom != 0 else default


if __name__ == "__main__":
    main()
