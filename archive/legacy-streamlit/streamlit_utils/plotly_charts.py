"""
Plotly Charts - Medical visualization utilities
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class MedicalCharts:
    """Medical chart utilities for Plotly"""
    
    # Color scheme for medical visualizations
    COLORS = {
        "risk_critical": "#ff3333",
        "risk_high": "#ff9900",
        "risk_moderate": "#ffcc00",
        "risk_low": "#00cc33",
        "primary": "#00d4ff",
        "secondary": "#0099cc",
        "accent": "#ff6b6b",
    }
    
    THEME_CONFIG = {
        "template": "plotly_dark",
        "font": {"family": "Arial, sans-serif", "color": "#e0e6ff"},
        "plot_bgcolor": "#0a0e27",
        "paper_bgcolor": "#1a1f3a",
        "xaxis": {"gridcolor": "#2a2f4a", "showgrid": True},
        "yaxis": {"gridcolor": "#2a2f4a", "showgrid": True},
        "margin": {"l": 60, "r": 60, "t": 60, "b": 60},
    }
    
    @staticmethod
    def risk_gauge(risk_score: float, title: str = "Risk Assessment") -> go.Figure:
        """Create risk assessment gauge chart
        
        Args:
            risk_score: Risk score between 0 and 1
            title: Chart title
            
        Returns:
            Plotly gauge figure
        """
        # Determine color and risk level
        if risk_score > 0.75:
            color = MedicalCharts.COLORS["risk_critical"]
            risk_text = "CRITICAL"
        elif risk_score > 0.6:
            color = MedicalCharts.COLORS["risk_high"]
            risk_text = "HIGH"
        elif risk_score > 0.4:
            color = MedicalCharts.COLORS["risk_moderate"]
            risk_text = "MODERATE"
        else:
            color = MedicalCharts.COLORS["risk_low"]
            risk_text = "LOW"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title},
            delta={"reference": 50, "prefix": "vs Baseline"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 25], "color": MedicalCharts.COLORS["risk_low"]},
                    {"range": [25, 50], "color": MedicalCharts.COLORS["risk_moderate"]},
                    {"range": [50, 75], "color": MedicalCharts.COLORS["risk_high"]},
                    {"range": [75, 100], "color": MedicalCharts.COLORS["risk_critical"]},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
            number={"suffix": "%", "font": {"size": 24}},
        ))
        
        fig.update_layout(
            **MedicalCharts.THEME_CONFIG,
            height=400,
            font={"size": 14}
        )
        
        return fig
    
    @staticmethod
    def trend_line(
        dates: List[str],
        values: List[float],
        title: str = "Risk Trend",
        y_label: str = "Risk Score"
    ) -> go.Figure:
        """Create trend line chart
        
        Args:
            dates: List of date strings
            values: List of values
            title: Chart title
            y_label: Y-axis label
            
        Returns:
            Plotly line figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=y_label,
            line=dict(color=MedicalCharts.COLORS["primary"], width=3),
            marker=dict(size=8, color=MedicalCharts.COLORS["primary"]),
            fill='tozeroy',
            fillcolor=f'rgba(0, 212, 255, 0.1)',
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label,
            **MedicalCharts.THEME_CONFIG,
            height=400,
            hovermode='x unified',
        )
        
        return fig
    
    @staticmethod
    def time_series(
        data: Dict[str, List[float]],
        dates: List[str],
        title: str = "Time Series Data"
    ) -> go.Figure:
        """Create time series chart with multiple lines
        
        Args:
            data: Dictionary with series names as keys and value lists as values
            dates: List of date strings
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = [
            MedicalCharts.COLORS["primary"],
            MedicalCharts.COLORS["secondary"],
            MedicalCharts.COLORS["accent"],
        ]
        
        for idx, (name, values) in enumerate(data.items()):
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=name,
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=6),
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            **MedicalCharts.THEME_CONFIG,
            height=400,
            hovermode='x unified',
        )
        
        return fig
    
    @staticmethod
    def bar_chart(
        categories: List[str],
        values: List[float],
        title: str = "Bar Chart",
        y_label: str = "Value"
    ) -> go.Figure:
        """Create bar chart
        
        Args:
            categories: Category names
            values: Category values
            title: Chart title
            y_label: Y-axis label
            
        Returns:
            Plotly bar figure
        """
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=MedicalCharts.COLORS["primary"],
                marker_line_color=MedicalCharts.COLORS["secondary"],
                marker_line_width=2,
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Category",
            yaxis_title=y_label,
            **MedicalCharts.THEME_CONFIG,
            height=400,
        )
        
        return fig
    
    @staticmethod
    def feature_importance(
        features: List[str],
        importance: List[float],
        title: str = "Feature Importance"
    ) -> go.Figure:
        """Create feature importance bar chart
        
        Args:
            features: Feature names
            importance: Importance scores
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        features = [features[i] for i in sorted_idx]
        importance = [importance[i] for i in sorted_idx]
        
        fig = go.Figure(data=[
            go.Bar(
                y=features,
                x=importance,
                orientation='h',
                marker_color=MedicalCharts.COLORS["accent"],
                marker_line_color=MedicalCharts.COLORS["primary"],
                marker_line_width=2,
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            **MedicalCharts.THEME_CONFIG,
            height=400,
        )
        
        return fig
    
    @staticmethod
    def distribution_histogram(
        values: List[float],
        title: str = "Distribution",
        x_label: str = "Value"
    ) -> go.Figure:
        """Create distribution histogram
        
        Args:
            values: Data values
            title: Chart title
            x_label: X-axis label
            
        Returns:
            Plotly histogram figure
        """
        fig = go.Figure(data=[
            go.Histogram(
                x=values,
                nbinsx=30,
                marker_color=MedicalCharts.COLORS["primary"],
                marker_line_color=MedicalCharts.COLORS["secondary"],
                marker_line_width=1,
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title="Frequency",
            **MedicalCharts.THEME_CONFIG,
            height=400,
        )
        
        return fig
    
    @staticmethod
    def risk_distribution(risk_scores: Dict[str, int]) -> go.Figure:
        """Create risk distribution pie chart
        
        Args:
            risk_scores: Dictionary with risk levels and counts
                        e.g., {"Low": 45, "Moderate": 30, "High": 20, "Critical": 5}
            
        Returns:
            Plotly pie figure
        """
        colors_list = [
            MedicalCharts.COLORS["risk_low"],
            MedicalCharts.COLORS["risk_moderate"],
            MedicalCharts.COLORS["risk_high"],
            MedicalCharts.COLORS["risk_critical"],
        ]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(risk_scores.keys()),
                values=list(risk_scores.values()),
                marker=dict(colors=colors_list[:len(risk_scores)]),
                textposition='inside',
                textinfo='label+percent',
            )
        ])
        
        fig.update_layout(
            title="Risk Distribution",
            **MedicalCharts.THEME_CONFIG,
            height=400,
        )
        
        return fig
    
    @staticmethod
    def patient_vitals(vitals: Dict[str, float], ranges: Dict[str, tuple]) -> go.Figure:
        """Create patient vitals indicator chart
        
        Args:
            vitals: Dictionary of vital signs and values
                   e.g., {"BP": 120, "HR": 75, "O2": 98}
            ranges: Dictionary of normal ranges
                   e.g., {"BP": (90, 130), "HR": (60, 100), "O2": (95, 100)}
            
        Returns:
            Plotly figure
        """
        names = list(vitals.keys())
        values = list(vitals.values())
        
        # Create color coding based on ranges
        colors = []
        for name, value in zip(names, values):
            if name in ranges:
                min_val, max_val = ranges[name]
                if min_val <= value <= max_val:
                    colors.append(MedicalCharts.COLORS["risk_low"])
                elif value < min_val - 10 or value > max_val + 10:
                    colors.append(MedicalCharts.COLORS["risk_critical"])
                else:
                    colors.append(MedicalCharts.COLORS["risk_high"])
            else:
                colors.append(MedicalCharts.COLORS["primary"])
        
        fig = go.Figure(data=[
            go.Bar(
                x=names,
                y=values,
                marker_color=colors,
                marker_line_color=MedicalCharts.COLORS["secondary"],
                marker_line_width=2,
                text=values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Patient Vitals",
            yaxis_title="Value",
            **MedicalCharts.THEME_CONFIG,
            height=400,
        )
        
        return fig
    
    @staticmethod
    def correlation_heatmap(
        data: Dict[str, List[float]],
        title: str = "Feature Correlation"
    ) -> go.Figure:
        """Create correlation heatmap
        
        Args:
            data: Dictionary with feature names and value lists
            title: Chart title
            
        Returns:
            Plotly heatmap figure
        """
        import pandas as pd
        
        df = pd.DataFrame(data)
        corr = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title=title,
            **MedicalCharts.THEME_CONFIG,
            height=500,
        )
        
        return fig
    
    @staticmethod
    def shap_waterfall(
        feature_names: List[str],
        shap_values: List[float],
        base_value: float,
        prediction_value: float,
        title: str = "SHAP Waterfall Plot"
    ) -> go.Figure:
        """Create SHAP waterfall plot
        
        Args:
            feature_names: Feature names
            shap_values: SHAP values for each feature
            base_value: Model base value
            prediction_value: Final prediction
            title: Chart title
            
        Returns:
            Plotly waterfall figure
        """
        # Create cumulative values for waterfall
        cumulative = [base_value]
        measures = ['relative']
        
        for val in shap_values:
            cumulative.append(cumulative[-1] + val)
            measures.append('relative')
        
        # Add final prediction
        feature_names = ['Base'] + feature_names + ['Prediction']
        cumulative[0] = base_value
        measures[0] = 'absolute'
        
        # Determine connector colors based on positive/negative
        colors = ['gray']
        for val in shap_values:
            if val > 0:
                colors.append(MedicalCharts.COLORS["risk_high"])
            else:
                colors.append(MedicalCharts.COLORS["risk_low"])
        colors.append('blue')
        
        fig = go.Figure(go.Waterfall(
            x=feature_names,
            y=shap_values + [0],
            measure=measures + ['total'],
            text=shap_values + [0],
            textposition='auto',
            connector={"line": {"color": "rgba(0, 212, 255, 0.5)"}},
            increasing={"marker": {"color": MedicalCharts.COLORS["accent"]}},
            decreasing={"marker": {"color": MedicalCharts.COLORS["risk_low"]}},
            totals={"marker": {"color": MedicalCharts.COLORS["primary"]}},
        ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Impact on Prediction",
            **MedicalCharts.THEME_CONFIG,
            height=400,
        )
        
        return fig
    
    @staticmethod
    def comparison_chart(
        labels: List[str],
        data_sets: Dict[str, List[float]],
        title: str = "Comparison"
    ) -> go.Figure:
        """Create grouped bar chart for comparison
        
        Args:
            labels: X-axis labels
            data_sets: Dictionary with series names and values
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = [
            MedicalCharts.COLORS["primary"],
            MedicalCharts.COLORS["accent"],
            MedicalCharts.COLORS["secondary"],
        ]
        
        for idx, (name, values) in enumerate(data_sets.items()):
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                name=name,
                marker_color=colors[idx % len(colors)],
            ))
        
        fig.update_layout(
            title=title,
            barmode='group',
            **MedicalCharts.THEME_CONFIG,
            height=400,
            xaxis_title="Category",
            yaxis_title="Value",
        )
        
        return fig


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def create_empty_chart(message: str = "No data available") -> go.Figure:
    """Create empty placeholder chart"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20, color="#888")
    )
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        plot_bgcolor="#0a0e27",
        paper_bgcolor="#1a1f3a",
        height=400
    )
    return fig


def generate_sample_trend_data(days: int = 30) -> tuple:
    """Generate sample trend data for testing
    
    Args:
        days: Number of days of data
        
    Returns:
        Tuple of (dates, values)
    """
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    values = np.random.uniform(0.3, 0.8, days).tolist()
    return list(reversed(dates)), list(reversed(values))
