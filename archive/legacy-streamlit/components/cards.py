"""
Reusable card components for UI
"""

import streamlit as st
from typing import Any, Optional


def metric_card(label: str, value: str, delta: str = "", icon: str = "📊") -> None:
    """Display a metric card"""
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"<h2>{icon}</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<small style='color: #888;'>{label}</small>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin: 5px 0; color: #00d4ff;'>{value}</h3>", unsafe_allow_html=True)
        if delta:
            st.markdown(f"<small style='color: #00ff88;'>{delta}</small>", unsafe_allow_html=True)


def info_card(title: str, content: str, icon: str = "ℹ️", bg_color: str = "info") -> None:
    """Display an info card"""
    st.markdown(f"""
    <div class='{bg_color}-card' style='border-radius: 10px; padding: 20px; margin: 10px 0;'>
        <h3 style='margin: 0; color: #00d4ff;'>{icon} {title}</h3>
        <p style='margin: 10px 0 0 0; color: #bbb;'>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def patient_card(name: str, age: int, severity: str, risk_score: float, patient_id: str = "") -> None:
    """Display patient information card"""
    risk_emoji = "🔴" if risk_score > 70 else "🟠" if risk_score > 50 else "🟡" if risk_score > 25 else "🟢"
    
    st.markdown(f"""
    <div class='info-card' style='border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid #00d4ff;'>
        <h4 style='margin: 0; color: #00d4ff;'>{name}</h4>
        <p style='margin: 5px 0; color: #888; font-size: 12px;'>ID: {patient_id}</p>
        <div style='display: flex; gap: 15px; margin-top: 10px;'>
            <span>👤 Age: {age}</span>
            <span>⚕️ Severity: {severity}</span>
            <span>{risk_emoji} Risk: {risk_score:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_box(label: str, value: str, trend: str = "") -> None:
    """Display stat box"""
    trend_color = "color: #00ff88;" if "↑" in trend or "+" in trend else "color: #ff6b6b;" if "↓" in trend else ""
    
    st.markdown(f"""
    <div style='
        background-color: #1a1f3a;
        border: 1px solid #2a3f5f;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    '>
        <p style='margin: 0; color: #888; font-size: 13px;'>{label}</p>
        <h3 style='margin: 10px 0 5px 0; color: #00d4ff;'>{value}</h3>
        {f'<p style="margin: 0; {trend_color} font-size: 12px;">{trend}</p>' if trend else ''}
    </div>
    """, unsafe_allow_html=True)


def status_badge(status: str, color: str = "primary") -> None:
    """Display status badge"""
    color_map = {
        "primary": "#00d4ff",
        "success": "#00ff88",
        "warning": "#ffa500",
        "danger": "#ff1744",
    }
    
    badge_color = color_map.get(color, color_map["primary"])
    
    st.markdown(f"""
    <span style='
        background-color: {badge_color};
        color: #0a0e27;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    '>{status}</span>
    """, unsafe_allow_html=True)


def divider_text(text: str = "") -> None:
    """Display decorative divider with optional text"""
    if text:
        st.markdown(f"<p style='text-align: center; color: #888; margin: 20px 0;'>─ {text} ─</p>", unsafe_allow_html=True)
    else:
        st.divider()


def empty_state(icon: str = "📭", title: str = "No Data", message: str = "") -> None:
    """Display empty state"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 40px 0;'>
            <h1 style='font-size: 48px; margin: 0;'>{icon}</h1>
            <h3 style='color: #888; margin: 10px 0;'>{title}</h3>
            <p style='color: #666; margin: 0;'>{message}</p>
        </div>
        """, unsafe_allow_html=True)
