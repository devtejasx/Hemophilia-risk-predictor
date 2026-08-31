"""
Sidebar navigation and branding component
"""

import streamlit as st
from utils.session_state import get_session_var, set_session_var


def show_sidebar() -> None:
    """Display sidebar with navigation and branding"""
    
    with st.sidebar:
        # Logo/Branding
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: #667eea; margin: 0; font-size: 2.5rem;'>🏥</h1>
            <p style='color: #667eea; font-weight: bold; margin: 5px 0; font-size: 1.2rem;'>Hemophilia AI</p>
            <p style='color: #888; font-size: 11px; margin: 0;'>Risk Assessment Platform v3.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # User Info (if authenticated)
        if get_session_var("authenticated"):
            st.markdown(f"👤 **{get_session_var('user_name')}**")
            st.markdown(f"Role: {get_session_var('user_role').title()}")
            st.divider()
        
        # Navigation
        st.markdown("### 📍 Navigation")
        current_page = st.radio(
            "Select Page",
            options=[
                "Dashboard",
                "Add Patient",
                "Predictions",
                "SHAP Explainability",
                "Chatbot",
                "Analytics"
            ],
            key="nav_radio",
            label_visibility="collapsed",
            format_func=lambda x: {
                "Dashboard": "📊 Dashboard",
                "Add Patient": "👤 Add Patient",
                "Predictions": "🔮 Predictions",
                "SHAP Explainability": "🧠 SHAP Analysis",
                "Chatbot": "🤖 AI Assistant",
                "Analytics": "📈 Analytics"
            }[x]
        )
        
        st.session_state.current_page = current_page
        
        st.divider()
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        theme = get_session_var("theme", "dark")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌙" if theme == "light" else "☀️", use_container_width=True):
                new_theme = "light" if theme == "dark" else "dark"
                set_session_var("theme", new_theme)
                st.rerun()
        
        with col2:
            if st.button("🔍", use_container_width=True, help="Advanced View"):
                set_session_var("show_advanced", not get_session_var("show_advanced"))
                st.rerun()
        
        st.divider()
        
        # Info Section
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            **Hemophilia AI Platform**
            
            An intelligent clinical system for:
            - 🩸 Patient risk assessment
            - 💊 Treatment planning
            - 🤖 AI-powered insights
            - 📊 Real-time monitoring
            
            **Key Features:**
            - Multi-page dashboard
            - Patient management
            - ML-based predictions
            - SHAP explainability
            - Clinical AI assistant
            - Advanced analytics
            """)
        
        # Footer
        st.divider()
        st.markdown(
            "<p style='text-align: center; color: #888; font-size: 11px;'>© 2024 Hemophilia AI Labs</p>",
            unsafe_allow_html=True
        )


def show_page_header(title: str, subtitle: str = "", icon: str = "📄") -> None:
    """Show consistent page header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<h1 style='text-align: center; color: #00d4ff;'>{title}</h1>", unsafe_allow_html=True)
        if subtitle:
            st.markdown(f"<p style='text-align: center; color: #888;'>{subtitle}</p>", unsafe_allow_html=True)
    st.divider()
