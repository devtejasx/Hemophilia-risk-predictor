"""
Patient History Page - Historical data with filtering and analysis
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from streamlit_utils.state_manager import StateManager
from streamlit_utils.ui_components import (
    create_header, show_info, show_warning, create_tabs
)
from streamlit_utils.backend_client import get_backend_client
from streamlit_utils.plotly_charts import MedicalCharts


def render():
    """Render patient history page"""
    
    create_header("📋 Patient History", "View and analyze patient historical records")
    
    # Get clients
    state = StateManager()
    backend = get_backend_client()
    
    # Check if patient is loaded
    current_patient = state.get_current_patient()
    
    if not current_patient or not current_patient.get("patient_id"):
        st.warning("⚠️  Please load a patient from the Patient Form first")
        return
    
    # Display current patient
    st.markdown(f"### 👤 {current_patient.get('first_name', '')} {current_patient.get('last_name', '')} (ID: {current_patient.get('patient_id')})")
    st.divider()
    
    # ========================================================================
    # FILTERS
    # ========================================================================
    
    st.markdown("### 🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        filter_type = st.selectbox(
            "Record Type",
            ["All", "Predictions", "Vitals", "Labs", "Notes"],
            key="history_record_type"
        )
    
    with col2:
        date_range = st.selectbox(
            "Date Range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Last Year", "All Time"],
            key="history_date_range"
        )
    
    with col3:
        sort_order = st.selectbox(
            "Sort By",
            ["Most Recent", "Oldest First", "Risk Score (High to Low)", "Risk Score (Low to High)"],
            key="history_sort"
        )
    
    with col4:
        records_per_page = st.number_input(
            "Records Per Page",
            min_value=5,
            max_value=100,
            value=10,
            key="history_records_per_page"
        )
    
    st.divider()
    
    # Fetch history data
    with st.spinner("Loading patient history..."):
        history_data = backend.get_patient_history(
            current_patient.get("patient_id"),
            limit=200
        )
    
    if not history_data:
        st.warning("No historical data found for this patient")
        return
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = create_tabs(["Records Table", "Risk Timeline", "Analysis", "Export"])
    
    # ========================================================================
    # TAB 1: RECORDS TABLE
    # ========================================================================
    
    with tab1:
        st.markdown("### 📊 Historical Records")
        
        # Convert to DataFrame
        df = pd.DataFrame(history_data)
        
        # Add date column if not present
        if "date" not in df.columns and "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"])
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        # Apply filters
        if date_range != "All Time":
            days_map = {
                "Last 7 days": 7,
                "Last 30 days": 30,
                "Last 90 days": 90,
                "Last Year": 365,
            }
            days = days_map.get(date_range, 7)
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df["date"] >= cutoff_date]
        
        # Apply sorting
        if "risk_score" in df.columns:
            if sort_order == "Risk Score (High to Low)":
                df = df.sort_values("risk_score", ascending=False)
            elif sort_order == "Risk Score (Low to High)":
                df = df.sort_values("risk_score", ascending=True)
            elif sort_order == "Most Recent":
                df = df.sort_values("date", ascending=False)
            elif sort_order == "Oldest First":
                df = df.sort_values("date", ascending=True)
        
        # Display summary
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            if "risk_score" in df.columns:
                st.metric("Avg Risk", f"{df['risk_score'].mean():.1%}")
        with col3:
            if "risk_score" in df.columns:
                st.metric("Max Risk", f"{df['risk_score'].max():.1%}")
        with col4:
            if "risk_score" in df.columns:
                st.metric("Min Risk", f"{df['risk_score'].min():.1%}")
        with col5:
            if "date" in df.columns:
                days_span = (df["date"].max() - df["date"].min()).days
                st.metric("Date Span", f"{days_span} days")
        
        # Pagination
        total_pages = (len(df) + records_per_page - 1) // records_per_page
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=max(1, total_pages),
            value=1,
            key="history_page"
        )
        
        start_idx = (page - 1) * records_per_page
        end_idx = start_idx + records_per_page
        
        # Display table
        display_columns = [col for col in df.columns if col not in ["_id", "id", "patient_id"]]
        st.dataframe(
            df[display_columns].iloc[start_idx:end_idx],
            use_container_width=True,
            height=500
        )
        
        st.caption(f"Showing {start_idx + 1} to {min(end_idx, len(df))} of {len(df)} records (Page {page}/{total_pages})")
    
    # ========================================================================
    # TAB 2: RISK TIMELINE
    # ========================================================================
    
    with tab2:
        st.markdown("### 📈 Risk Episode Timeline")
        
        if "date" in df.columns and "risk_score" in df.columns:
            # Sort by date
            df_timeline = df.sort_values("date")
            
            # Create timeline chart
            fig_timeline = MedicalCharts.time_series(
                data={
                    "Risk Score": df_timeline["risk_score"].tolist() if "risk_score" in df_timeline.columns else [],
                    "Baseline": [0.5] * len(df_timeline),
                },
                dates=[d.strftime("%Y-%m-%d %H:%M") for d in df_timeline["date"]],
                title="Risk Score Over Time"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Key events
            st.markdown("#### Notable Events")
            
            # Find peaks
            if len(df_timeline) > 1:
                risk_scores = df_timeline["risk_score"].values
                max_idx = risk_scores.argmax()
                min_idx = risk_scores.argmin()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.warning(f"⬆️ **Highest Risk**: {risk_scores[max_idx]:.1%} on {df_timeline['date'].iloc[max_idx].strftime('%Y-%m-%d')}")
                
                with col2:
                    st.success(f"⬇️ **Lowest Risk**: {risk_scores[min_idx]:.1%} on {df_timeline['date'].iloc[min_idx].strftime('%Y-%m-%d')}")
        else:
            st.warning("Risk score data not available")
    
    # ========================================================================
    # TAB 3: ANALYSIS
    # ========================================================================
    
    with tab3:
        st.markdown("### 📊 Statistical Analysis")
        
        if "risk_score" in df.columns:
            # Risk distribution
            st.markdown("#### Risk Distribution")
            
            risk_counts = {
                "Low (0-40%)": len(df[df["risk_score"] < 0.4]),
                "Moderate (40-60%)": len(df[(df["risk_score"] >= 0.4) & (df["risk_score"] < 0.6)]),
                "High (60-75%)": len(df[(df["risk_score"] >= 0.6) & (df["risk_score"] < 0.75)]),
                "Critical (>75%)": len(df[df["risk_score"] >= 0.75]),
            }
            
            fig_dist = MedicalCharts.risk_distribution(risk_counts)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Statistics
            st.markdown("#### Descriptive Statistics")
            
            stats = {
                "Mean": f"{df['risk_score'].mean():.1%}",
                "Median": f"{df['risk_score'].median():.1%}",
                "Std Dev": f"{df['risk_score'].std():.1%}",
                "Min": f"{df['risk_score'].min():.1%}",
                "Max": f"{df['risk_score'].max():.1%}",
                "Q1 (25%)": f"{df['risk_score'].quantile(0.25):.1%}",
                "Q3 (75%)": f"{df['risk_score'].quantile(0.75):.1%}",
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                for key in list(stats.keys())[:4]:
                    st.metric(key, stats[key])
            
            with col2:
                for key in list(stats.keys())[4:]:
                    st.metric(key, stats[key])
            
            # Trend analysis
            st.markdown("#### Trend Indicators")
            
            if len(df) > 1:
                first_score = df_timeline["risk_score"].iloc[0]
                last_score = df_timeline["risk_score"].iloc[-1]
                trend = last_score - first_score
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("First Recording", f"{first_score:.1%}")
                
                with col2:
                    st.metric("Latest Recording", f"{last_score:.1%}")
                
                with col3:
                    st.metric("Change", f"{trend:+.1%}")
                
                # Interpretation
                if trend > 0.1:
                    st.warning("⬆️ Risk is significantly increasing - increased monitoring recommended")
                elif trend > 0.05:
                    st.warning("⬆️ Risk trending upward - consider intervention")
                elif trend < -0.1:
                    st.success("⬇️ Risk is significantly decreasing - treatment may be effective")
                elif trend < -0.05:
                    st.success("⬇️ Risk trending downward - continue current treatment")
                else:
                    st.info("→ Risk is stable - maintain current monitoring")
        else:
            st.warning("Risk score data not available for analysis")
    
    # ========================================================================
    # TAB 4: EXPORT
    # ========================================================================
    
    with tab4:
        st.markdown("### 💾 Export Data")
        
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"patient_{current_patient.get('patient_id')}_history.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # JSON export
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"patient_{current_patient.get('patient_id')}_history.json",
            mime="application/json",
            use_container_width=True
        )
        
        # Excel export (if openpyxl is available)
        try:
            excel_buffer = df.to_excel(index=False, engine="openpyxl")
            st.download_button(
                label="📥 Download as Excel",
                data=excel_buffer,
                file_name=f"patient_{current_patient.get('patient_id')}_history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except:
            pass
        
        st.divider()
        st.markdown("#### Export Summary")
        st.write(f"- Total Records: {len(df)}")
        st.write(f"- Date Range: {df['date'].min() if 'date' in df.columns else 'N/A'} to {df['date'].max() if 'date' in df.columns else 'N/A'}")
        st.write(f"- Columns: {', '.join(df.columns.tolist())}")
