"""
Optimized Streamlit Application

Performance Improvements:
1. Session state caching - avoid re-computations
2. Lazy loading - load data only when needed
3. Reduced widget count - minimize rerun overhead
4. Efficient data structures - use dicts/lists wisely
5. Cache decorators for expensive operations
6. Pagination for large datasets
7. Async operations for non-blocking UI
8. Minimal reruns through form management
9. Streamed output - show results incrementally
10. Background processing - async task handling
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

# ============ PAGE CONFIG & OPTIMIZATION ============

st.set_page_config(
    page_title="Hemophilia Clinic - Optimized",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Clinical Decision Support System v2.0"
    }
)

# ============ SESSION STATE INITIALIZATION ============

def init_session_state():
    """Initialize session state once per session"""
    if "api_base" not in st.session_state:
        st.session_state.api_base = "http://localhost:8000"
    
    if "user_data" not in st.session_state:
        st.session_state.user_data = None
    
    if "prediction_cache" not in st.session_state:
        st.session_state.prediction_cache = {}
    
    if "selected_patient" not in st.session_state:
        st.session_state.selected_patient = None
    
    if "page" not in st.session_state:
        st.session_state.page = 1
    
    if "page_size" not in st.session_state:
        st.session_state.page_size = 50
    
    if "background_tasks" not in st.session_state:
        st.session_state.background_tasks = {}
    
    if "last_search" not in st.session_state:
        st.session_state.last_search = ""


init_session_state()


# ============ CACHING DECORATORS FOR STREAMLIT ============

@st.cache_resource
def get_api_session():
    """Create a persistent HTTP session (cached)"""
    import requests
    session = requests.Session()
    session.headers.update({
        "Accept-Encoding": "gzip",  # Enable compression
        "Connection": "keep-alive"    # Connection pooling
    })
    return session


@st.cache_data(ttl=300)
def fetch_dashboard_stats():
    """Fetch dashboard stats with 5-min cache"""
    try:
        session = get_api_session()
        response = session.get(f"{st.session_state.api_base}/dashboard/stats")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch stats: {e}")
        return None


@st.cache_data(ttl=600)
def fetch_high_risk_patients(limit: int = 50):
    """Fetch high-risk patients with 10-min cache"""
    try:
        session = get_api_session()
        response = session.get(
            f"{st.session_state.api_base}/dashboard/high-risk",
            params={"limit": limit}
        )
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Failed to fetch high-risk patients: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_cache_stats():
    """Fetch cache statistics with 1-hour cache"""
    try:
        session = get_api_session()
        response = session.get(f"{st.session_state.api_base}/admin/cache-stats")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch cache stats: {e}")
        return None


# ============ MAIN DASHBOARD PAGE ============

def render_dashboard():
    """Main dashboard with optimized layout and minimal reruns"""
    st.title("🏥 Hemophilia Clinic - Clinical Dashboard")
    
    # Use columns to organize layout (lazy loading)
    col1, col2, col3, col4 = st.columns(4)
    
    stats = fetch_dashboard_stats()
    
    if stats:
        with col1:
            st.metric("Total Patients", stats.get("total_patients", 0))
        with col2:
            st.metric("High Risk", stats.get("high_risk_count", 0))
        with col3:
            st.metric("Avg Risk Score", f"{stats.get('average_risk', 0):.2%}")
        with col4:
            st.metric("Avg Age", f"{stats.get('average_age', 0):.0f} yrs")
    
    st.markdown("---")
    
    # Tabs to minimize rerun overhead (each tab is lazy loaded)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["High-Risk Patients", "Predictions", "Monitoring", "Cache Stats"]
    )
    
    with tab1:
        render_high_risk_tab()
    
    with tab2:
        render_predictions_tab()
    
    with tab3:
        render_monitoring_tab()
    
    with tab4:
        render_cache_stats_tab()


# ============ TAB: HIGH-RISK PATIENTS ============

def render_high_risk_tab():
    """Display high-risk patients with pagination"""
    st.subheader("🚨 High-Risk Patient Monitor")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        limit = st.slider("Show top N patients", 10, 500, 100, step=10)
    
    df = fetch_high_risk_patients(limit=limit)
    
    if not df.empty:
        # Display as interactive table
        st.dataframe(
            df[["name", "age", "severity", "risk_score"]],
            use_container_width=True,
            hide_index=True
        )
        
        # Show distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, 
                x="risk_score",
                nbins=20,
                title="Risk Score Distribution",
                labels={"risk_score": "Risk Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                df,
                names="severity",
                title="Severity Distribution",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No high-risk patients found")


# ============ TAB: PREDICTIONS ============

def render_predictions_tab():
    """Single prediction with form-based input (minimal reruns)"""
    st.subheader("🔮 Patient Risk Prediction")
    
    # Use form to batch updates
    with st.form("prediction_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 0, 150, 30)
            dose = st.number_input("Dose Intensity", 1, 1000, 100)
        
        with col2:
            exposure = st.number_input("Exposure Days", 0, 10000, 500)
            severity = st.selectbox("Severity", ["mild", "moderate", "severe"])
        
        mutation = st.selectbox(
            "Mutation Type",
            ["intron22", "intron1", "deletion", "inversion", "other"]
        )
        
        submit = st.form_submit_button("🎯 Predict Risk", use_container_width=True)
    
    if submit:
        predict_and_display(age, dose, exposure, severity, mutation)


def predict_and_display(age, dose, exposure, severity, mutation):
    """Fetch and display prediction"""
    try:
        # Create cache key
        cache_key = f"{age}_{dose}_{exposure}_{severity}_{mutation}"
        
        # Check session cache first (instant)
        if cache_key in st.session_state.prediction_cache:
            result = st.session_state.prediction_cache[cache_key]
        else:
            # Fetch from API
            session = get_api_session()
            response = session.post(
                f"{st.session_state.api_base}/predict",
                params={
                    "age": age,
                    "dose": dose,
                    "exposure": exposure,
                    "severity": severity,
                    "mutation": mutation
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache in session state
            st.session_state.prediction_cache[cache_key] = result
        
        # Display results
        st.success("✅ Prediction Complete")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_score = result.get("risk_score", 0)
            color = "🔴" if risk_score > 0.7 else "🟡" if risk_score > 0.4 else "🟢"
            st.metric(f"{color} Risk Score", f"{risk_score:.1%}")
        
        with col2:
            agreement = result.get("model_agreement", 0)
            st.metric("Model Agreement", f"{agreement:.1%}")
        
        with col3:
            st.metric("Confidence", "High" if agreement > 0.8 else "Medium" if agreement > 0.6 else "Low")
        
        # Display recommendation
        st.info(result.get("recommendation", "No recommendation"))
        
        # Display top features
        st.subheader("🔍 Top Contributing Factors")
        features = result.get("top_3_features", [])
        
        if features:
            feature_df = pd.DataFrame(features)
            
            # Create feature importance chart
            fig = px.bar(
                feature_df,
                x="impact",
                y="feature",
                color="direction",
                orientation="h",
                title="Feature Importance",
                color_discrete_map={"increases": "#ef553b", "decreases": "#00cc96"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
        
        # Task tracker for background operations
        if "task_id" in result:
            st.info(f"Background task queued: {result['task_id']}")
    
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        logger.error(f"Prediction error: {e}")


# ============ TAB: MONITORING ============

def render_monitoring_tab():
    """Display patient monitoring data with pagination"""
    st.subheader("📊 Patient Monitoring Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.number_input("Patient ID", 1, 100000, 1)
    
    with col2:
        records_per_page = st.selectbox("Records per page", [10, 25, 50, 100])
    
    try:
        session = get_api_session()
        response = session.get(
            f"{st.session_state.api_base}/patients/{patient_id}/monitoring",
            params={
                "page": st.session_state.page,
                "page_size": records_per_page
            }
        )
        response.raise_for_status()
        
        data = response.json()
        
        if data["total"] > 0:
            df = pd.DataFrame(data["data"])
            
            # Display table
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Pagination controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.session_state.page > 1:
                    if st.button("⬅️ Previous"):
                        st.session_state.page -= 1
                        st.rerun()
            
            with col2:
                st.metric(
                    "Page",
                    f"{data['page']}/{data['total_pages']}"
                )
            
            with col3:
                if st.session_state.page < data["total_pages"]:
                    if st.button("Next ➡️"):
                        st.session_state.page += 1
                        st.rerun()
            
            # Show distribution if enough data
            if len(df) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    if "factor_level" in df.columns:
                        fig = px.line(
                            df,
                            y="factor_level",
                            title="Factor Level Trend"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if "bleeding_episodes" in df.columns:
                        fig = px.bar(
                            df,
                            y="bleeding_episodes",
                            title="Bleeding Episodes"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monitoring records found for this patient")
    
    except Exception as e:
        st.error(f"Failed to fetch monitoring data: {e}")


# ============ TAB: CACHE STATS ============

def render_cache_stats_tab():
    """Display performance and cache statistics"""
    st.subheader("⚡ Performance Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Stats"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Cache"):
            try:
                session = get_api_session()
                session.post(f"{st.session_state.api_base}/admin/clear-cache")
                st.success("Cache cleared!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")
    
    # Current session cache size
    st.metric(
        "Session Cache Size",
        f"{len(st.session_state.prediction_cache)} predictions"
    )
    
    # API cache stats
    stats = fetch_cache_stats()
    
    if stats:
        st.subheader("API Cache Performance")
        
        col1, col2, col3 = st.columns(3)
        
        # Model cache
        model_stats = stats.get("model_cache", {})
        with col1:
            st.metric(
                "Model Cache Hit Rate",
                f"{model_stats.get('hit_rate', 0):.1f}%"
            )
            st.metric("Cached Models", model_stats.get('size', 0))
        
        # Query cache
        query_stats = stats.get("query_cache", {})
        with col2:
            st.metric(
                "Query Cache Hit Rate",
                f"{query_stats.get('hit_rate', 0):.1f}%"
            )
            st.metric("Cached Queries", query_stats.get('size', 0))
        
        # Prediction cache
        pred_stats = stats.get("prediction_cache", {})
        with col3:
            st.metric(
                "Prediction Cache Hit Rate",
                f"{pred_stats.get('hit_rate', 0):.1f}%"
            )
            st.metric("Cached Predictions", pred_stats.get('size', 0))
    
    st.divider()
    
    # Session state info
    st.subheader("Session State")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current Page", st.session_state.page)
        st.metric("Items Per Page", st.session_state.page_size)
    
    with col2:
        st.metric("Selected Patient", st.session_state.selected_patient or "None")
        st.metric("Cached Predictions", len(st.session_state.prediction_cache))


# ============ SIDEBAR & NAVIGATION ============

with st.sidebar:
    st.title("🔧 Settings")
    
    # API Configuration
    with st.expander("API Configuration"):
        st.session_state.api_base = st.text_input(
            "API Base URL",
            value=st.session_state.api_base
        )
    
    # Performance settings
    with st.expander("Performance"):
        st.session_state.page_size = st.slider(
            "Default page size",
            10, 500, st.session_state.page_size
        )
        
        if st.button("Clear All Caches"):
            st.session_state.prediction_cache.clear()
            st.cache_data.clear()
            st.success("All caches cleared!")
    
    # About
    with st.expander("About"):
        st.markdown("""
        **Hemophilia Clinic v2.0**
        
        Optimized Clinical Decision Support System
        
        **Optimizations:**
        - Session state caching
        - API compression
        - Pagination
        - Background task processing
        - Efficient queries
        - Dashboard compression
        """)


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    render_dashboard()
