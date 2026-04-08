"""
Chart components for data visualization.
Plotly-based charts for dashboard analytics.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional
from colors import THEMES, get_risk_color
from utils import is_dark_mode


def render_risk_distribution(
    risk_scores: List[float],
    patient_names: Optional[List[str]] = None,
) -> None:
    """Render a bar chart of patient risk scores.
    
    Args:
        risk_scores: List of risk scores (0-1)
        patient_names: List of patient names (optional)
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    if not risk_scores:
        st.info("No risk data available")
        return
    
    # Create patient names if not provided
    if not patient_names:
        patient_names = [f"Patient {i+1}" for i in range(len(risk_scores))]
    
    # Assign colors based on risk level
    colors = [get_risk_color(score) for score in risk_scores]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=patient_names,
            y=risk_scores,
            marker=dict(color=colors),
            text=[f"{s:.1%}" for s in risk_scores],
            textposition="auto",
            hovertemplate="%{x}<br>Risk: %{y:.1%}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Risk Distribution",
        yaxis_title="Risk Score",
        xaxis_title="Patient",
        hovermode="x unified",
        plot_bgcolor=theme_colors["bg"],
        paper_bgcolor=theme_colors["bg"],
        font=dict(color=theme_colors["text"]),
        xaxis=dict(gridcolor=theme_colors["card_border"]),
        yaxis=dict(gridcolor=theme_colors["card_border"]),
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_trend_chart(
    values: List[float],
    labels: Optional[List[str]] = None,
    title: str = "Trend Analysis",
) -> None:
    """Render a line chart showing trend over time.
    
    Args:
        values: List of values to plot
        labels: X-axis labels (optional)
        title: Chart title
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    if not values:
        st.info("No trend data available")
        return
    
    # Create labels if not provided
    if not labels:
        labels = [f"Point {i+1}" for i in range(len(values))]
    
    # Create line chart
    fig = go.Figure(data=[
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            line=dict(color=theme_colors["primary"], width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor=f"{theme_colors['primary']}22",
            hovertemplate="%{x}<br>Value: %{y:.2f}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=title,
        yaxis_title="Value",
        xaxis_title="Time",
        hovermode="x unified",
        plot_bgcolor=theme_colors["bg"],
        paper_bgcolor=theme_colors["bg"],
        font=dict(color=theme_colors["text"]),
        xaxis=dict(gridcolor=theme_colors["card_border"]),
        yaxis=dict(gridcolor=theme_colors["card_border"]),
        height=350,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_metric_gauge(
    value: float,
    min_val: float = 0,
    max_val: float = 100,
    title: str = "Metric",
) -> None:
    """Render a gauge chart for metric visualization.
    
    Args:
        value: Current value
        min_val: Minimum value on gauge
        max_val: Maximum value on gauge
        title: Gauge title
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    # Determine color based on value position
    percentage = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
    if percentage < 0.4:
        color = theme_colors["success"]
    elif percentage < 0.7:
        color = theme_colors["warning"]
    else:
        color = theme_colors["danger"]
    
    fig = go.Figure(data=[
        go.Gauge(
            value=value,
            title={"text": title},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge=dict(
                axis=dict(range=[min_val, max_val], color=theme_colors["text"]),
                bar=dict(color=color),
                steps=[
                    {"range": [min_val, max_val * 0.4], "color": f"{theme_colors['success']}22"},
                    {"range": [max_val * 0.4, max_val * 0.7], "color": f"{theme_colors['warning']}22"},
                    {"range": [max_val * 0.7, max_val], "color": f"{theme_colors['danger']}22"},
                ],
                threshold=dict(
                    line=dict(color="white", width=2),
                    thickness=0.75,
                    value=max_val * 0.8,
                ),
            ),
        )
    ])
    
    fig.update_layout(
        height=350,
        plot_bgcolor=theme_colors["bg"],
        paper_bgcolor=theme_colors["bg"],
        font=dict(color=theme_colors["text"]),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_distribution_histogram(
    data: List[float],
    title: str = "Distribution",
    nbins: int = 20,
) -> None:
    """Render a histogram of data distribution.
    
    Args:
        data: List of values
        title: Histogram title
        nbins: Number of bins
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    if not data:
        st.info("No distribution data available")
        return
    
    fig = go.Figure(data=[
        go.Histogram(
            x=data,
            nbinsx=nbins,
            marker=dict(color=theme_colors["primary"]),
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Frequency",
        hovermode="x unified",
        plot_bgcolor=theme_colors["bg"],
        paper_bgcolor=theme_colors["bg"],
        font=dict(color=theme_colors["text"]),
        xaxis=dict(gridcolor=theme_colors["card_border"]),
        yaxis=dict(gridcolor=theme_colors["card_border"]),
        height=350,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_pie_chart(
    values: List[float],
    labels: List[str],
    title: str = "Distribution",
) -> None:
    """Render a pie chart.
    
    Args:
        values: List of values
        labels: List of labels
        title: Chart title
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    if not values or not labels:
        st.info("No pie chart data available")
        return
    
    # Create color palette based on theme
    colors = [
        theme_colors["primary"],
        theme_colors["success"],
        theme_colors["warning"],
        theme_colors["danger"],
    ]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors[:len(labels)]),
            hovertemplate="%{label}<br>%{value} (%{percent})<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=title,
        plot_bgcolor=theme_colors["bg"],
        paper_bgcolor=theme_colors["bg"],
        font=dict(color=theme_colors["text"]),
        height=350,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(
    data: List[List[float]],
    x_labels: List[str],
    y_labels: List[str],
    title: str = "Heatmap",
) -> None:
    """Render a heatmap visualization.
    
    Args:
        data: 2D list of values
        x_labels: X-axis labels
        y_labels: Y-axis labels
        title: Heatmap title
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale="RdYlGn_r",
        hovertemplate="%{y} - %{x}<br>Value: %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        plot_bgcolor=theme_colors["bg"],
        paper_bgcolor=theme_colors["bg"],
        font=dict(color=theme_colors["text"]),
        height=350,
    )
    
    st.plotly_chart(fig, use_container_width=True)
