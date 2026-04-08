"""
Header component for application layout.
Renders top bar with title, user info, and theme toggle.
"""

import streamlit as st
from utils import is_logged_in, get_user_name, toggle_dark_mode, is_dark_mode
from constants import APP_NAME, APP_EMOJI
from colors import THEMES


def render_header():
    """Render application header with navigation and user controls."""
    
    # Header HTML/CSS
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    # Create header layout
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.markdown(
            f"""
            <style>
                .header-title {{
                    font-size: 24px;
                    font-weight: bold;
                    color: {theme_colors['primary']};
                    margin: 0;
                    padding: 0;
                }}
            </style>
            <div class="header-title">{APP_EMOJI} {APP_NAME}</div>
            """,
            unsafe_allow_html=True
        )
    
    # User info column
    with col2:
        if is_logged_in():
            user_name = get_user_name()
            st.markdown(
                f"""
                <div style="text-align: right; margin-top: 5px; font-size: 14px; color: {theme_colors['text']};'">
                    👤 {user_name}
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Theme toggle
    with col3:
        theme_button = "☀️" if is_dark_mode() else "🌙"
        if st.button(theme_button, key="theme_toggle", help="Toggle dark/light mode"):
            toggle_dark_mode()
            st.rerun()
    
    # Settings/Help
    with col4:
        if st.button("⚙️", key="settings_button", help="Settings"):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)
            st.rerun()
    
    # Divider line
    st.divider()


def render_header_with_tabs(*tabs):
    """Render header with tab navigation.
    
    Args:
        *tabs: Tab names to display
    
    Returns:
        Selected tab name
    """
    
    render_header()
    
    if tabs:
        selected_tab = st.radio(
            "Navigation",
            tabs,
            horizontal=True,
            label_visibility="collapsed"
        )
        return selected_tab
    
    return None


def render_status_bar(status: str, message: str = ""):
    """Render a status bar (success, error, warning, info).
    
    Args:
        status: 'success', 'error', 'warning', or 'info'
        message: Status message to display
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    status_colors = {
        "success": theme_colors["success"],
        "error": theme_colors["danger"],
        "warning": theme_colors["warning"],
        "info": theme_colors["primary"],
    }
    
    status_icons = {
        "success": "✓",
        "error": "✕",
        "warning": "⚠",
        "info": "ℹ",
    }
    
    color = status_colors.get(status, theme_colors["primary"])
    icon = status_icons.get(status, "•")
    
    st.markdown(
        f"""
        <style>
            .status-bar {{
                background-color: {color}22;
                border-left: 4px solid {color};
                padding: 12px;
                border-radius: 4px;
                margin: 10px 0;
            }}
            .status-text {{
                color: {color};
                font-weight: 600;
            }}
        </style>
        <div class="status-bar">
            <span class="status-text">{icon} {message}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_breadcrumb(*items):
    """Render breadcrumb navigation.
    
    Args:
        *items: Breadcrumb items to display
    """
    
    if items:
        breadcrumb_html = " / ".join(f"<span>{item}</span>" for item in items)
        st.markdown(
            f"<small>{breadcrumb_html}</small>",
            unsafe_allow_html=True
        )
