"""
Sidebar component for application navigation.
Reusable sidebar with menu options and user controls.
"""

import streamlit as st
from typing import List, Optional
from utils import is_logged_in, get_user_name, set_logged_out
from constants import APP_NAME, APP_EMOJI


def render_sidebar() -> Optional[str]:
    """Render main application sidebar with navigation menu.
    
    Returns:
        Selected menu item
    """
    
    with st.sidebar:
        # App title
        st.markdown(f"## {APP_EMOJI} {APP_NAME}")
        st.divider()
        
        # Navigation menu
        st.markdown("**Navigation**")
        
        menu = st.radio(
            "Select Page",
            ["Dashboard", "Patients", "Predictions", "Chat", "Analytics"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # User section
        if is_logged_in():
            st.markdown("**User**")
            user_name = get_user_name()
            st.markdown(f"👤 {user_name}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📝 Profile", use_container_width=True):
                    st.session_state.show_profile = True
            with col2:
                if st.button("🚪 Logout", use_container_width=True):
                    set_logged_out()
                    st.rerun()
            
            st.divider()
        
        # Help section
        st.markdown("**Help & Settings**")
        
        if st.button("❓ Help", use_container_width=True):
            st.session_state.show_help = True
        
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.show_settings = True
        
        st.divider()
        
        # Footer
        st.markdown(
            """
            <small style='text-align: center; display: block;'>
            v2.0 | Clinical Decision Support<br>
            © 2024 All Rights Reserved
            </small>
            """,
            unsafe_allow_html=True
        )
        
        return menu


def render_sidebar_menu(menu_items: List[str]) -> Optional[str]:
    """Render sidebar with custom menu items.
    
    Args:
        menu_items: List of menu item names
    
    Returns:
        Selected menu item
    """
    
    with st.sidebar:
        st.markdown(f"## {APP_EMOJI} {APP_NAME}")
        st.divider()
        
        selected = st.radio(
            "Menu",
            menu_items,
            label_visibility="collapsed"
        )
        
        return selected


def render_sidebar_info_box(title: str, content: str, icon: str = "ℹ️") -> None:
    """Render an info box in the sidebar.
    
    Args:
        title: Box title
        content: Box content
        icon: Emoji icon
    """
    
    with st.sidebar:
        st.markdown(
            f"""
            <style>
                .info-box {{
                    background-color: #e7f3ff;
                    border-left: 4px solid #2196F3;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }}
                .info-title {{
                    font-weight: bold;
                    color: #1976D2;
                }}
                .info-content {{
                    font-size: 12px;
                    color: #333;
                    margin-top: 5px;
                }}
            </style>
            <div class="info-box">
                <div class="info-title">{icon} {title}</div>
                <div class="info-content">{content}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_sidebar_with_tabs(*tabs) -> Optional[str]:
    """Render sidebar with tab selection.
    
    Args:
        *tabs: Tab names
    
    Returns:
        Selected tab
    """
    
    with st.sidebar:
        st.markdown(f"## {APP_EMOJI} {APP_NAME}")
        st.divider()
        
        selected_tab = st.radio(
            "Select Tab",
            tabs,
            label_visibility="collapsed"
        )
        
        return selected_tab


def render_sidebar_stats(**stats) -> None:
    """Render statistics display in sidebar.
    
    Args:
        **stats: Key-value pairs of stat_name -> stat_value
    """
    
    with st.sidebar:
        st.markdown("**Statistics**")
        st.divider()
        
        for stat_name, stat_value in stats.items():
            st.metric(stat_name, stat_value)
        
        st.divider()


def render_collapsible_sidebar_section(
    title: str,
    content_func,
    icon: str = "▶",
) -> None:
    """Render a collapsible section in sidebar.
    
    Args:
        title: Section title
        content_func: Function to call to render content
        icon: Section icon
    """
    
    with st.sidebar:
        with st.expander(f"{icon} {title}"):
            content_func()


def render_sidebar_quick_links(links: dict) -> None:
    """Render quick links section in sidebar.
    
    Args:
        links: Dictionary of link_text -> link_url or link_text -> action_function
    """
    
    with st.sidebar:
        st.markdown("**Quick Links**")
        
        for link_text, link_target in links.items():
            if callable(link_target):
                if st.button(link_text, use_container_width=True):
                    link_target()
            else:
                st.markdown(f"[{link_text}]({link_target})")


def render_sidebar_filter_section(filter_config: dict) -> dict:
    """Render filter controls in sidebar.
    
    Args:
        filter_config: Configuration for filter controls
        Example: {
            "search": {"type": "text", "label": "Search"},
            "filter": {"type": "select", "label": "Filter by", "options": []}
        }
    
    Returns:
        Dictionary with filter values
    """
    
    filter_values = {}
    
    with st.sidebar:
        st.markdown("**Filters**")
        st.divider()
        
        for filter_key, filter_spec in filter_config.items():
            filter_type = filter_spec.get("type", "text")
            label = filter_spec.get("label", filter_key)
            
            if filter_type == "text":
                filter_values[filter_key] = st.text_input(
                    label,
                    placeholder=filter_spec.get("placeholder", "")
                )
            elif filter_type == "select":
                filter_values[filter_key] = st.selectbox(
                    label,
                    filter_spec.get("options", [])
                )
            elif filter_type == "slider":
                filter_values[filter_key] = st.slider(
                    label,
                    min_value=filter_spec.get("min", 0),
                    max_value=filter_spec.get("max", 100),
                    value=filter_spec.get("value", 50)
                )
            elif filter_type == "checkbox":
                filter_values[filter_key] = st.checkbox(
                    label,
                    value=filter_spec.get("value", False)
                )
    
    return filter_values
