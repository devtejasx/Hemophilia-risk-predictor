"""
Chart and visualization components
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Use dark background for matplotlib
matplotlib.style.use('dark_background')


def plot_risk_gauge(risk_score: float) -> None:
    """Plot risk score as a gauge chart"""
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection='polar'))
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)
    
    # Color zones
    colors = []
    for t in theta:
        if t < np.pi * 0.25:
            colors.append('#00ff88')  # Green - Low
        elif t < np.pi * 0.5:
            colors.append('#ffa500')  # Orange - Medium
        elif t < np.pi * 0.75:
            colors.append('#ff8800')  # Dark Orange - High
        else:
            colors.append('#ff1744')  # Red - Critical
    
    # Plot gauge background
    for i in range(len(theta)-1):
        ax.plot(theta[i:i+2], r[i:i+2], color=colors[i], linewidth=20)
    
    # Plot needle
    needle_theta = (risk_score / 100) * np.pi
    ax.plot([needle_theta, needle_theta], [0, 1], 'w-', linewidth=3)
    ax.plot(needle_theta, 1, 'wo', markersize=15)
    
    # Styling
    ax.set_ylim(0, 1.3)
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, np.pi, 5))
    ax.set_xticklabels(['0', '25', '50', '75', '100'])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    # Title
    risk_level = "🔴 Critical" if risk_score >= 75 else "🟠 High" if risk_score >= 50 else "🟡 Medium" if risk_score >= 25 else "🟢 Low"
    fig.suptitle(f'Risk Score: {risk_score:.1f}% - {risk_level}', fontsize=16, color='#00d4ff', y=0.98)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_feature_importance(importance_dict: Dict[str, float], title: str = "Feature Importance") -> None:
    """Plot horizontal bar chart of feature importance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    # Sort by value
    sorted_pairs = sorted(zip(features, values), key=lambda x: x[1])
    features_sorted, values_sorted = zip(*sorted_pairs)
    
    # Create bars
    bars = ax.barh(features_sorted, values_sorted, color='#00d4ff', edgecolor='#0099ff', linewidth=1.5)
    
    # Highlight top bar
    bars[-1].set_color('#ffa500')
    
    # Styling
    ax.set_xlabel('Importance Score', color='#888', fontsize=11)
    ax.set_title(title, color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2a3f5f')
    ax.spines['bottom'].set_color('#2a3f5f')
    ax.tick_params(colors='#888')
    ax.grid(axis='x', alpha=0.2, color='#2a3f5f')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_patient_metrics(data: Dict[str, Any]) -> None:
    """Plot multiple patient metrics in subplots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Patient Overview', color='#00d4ff', fontsize=16, fontweight='bold')
    
    # Age distribution (placeholder)
    ax = axes[0, 0]
    ages = [10, 20, 30, 40, 50, 60, 70, 80]
    distribution = [5, 10, 15, 20, 18, 15, 10, 5]
    ax.bar(ages, distribution, color='#00d4ff', alpha=0.7, edgecolor='#0099ff')
    ax.set_title('Age Distribution', color='#888')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    
    # Severity breakdown
    ax = axes[0, 1]
    severities = ['Mild', 'Moderate', 'Severe', 'Critical']
    counts = [20, 35, 30, 15]
    colors_pie = ['#00ff88', '#ffa500', '#ff8800', '#ff1744']
    ax.pie(counts, labels=severities, colors=colors_pie, autopct='%1.1f%%')
    ax.set_title('Severity Distribution', color='#888')
    
    # Risk scores
    ax = axes[1, 0]
    risk_scores = np.random.rand(100) * 100
    ax.hist(risk_scores, bins=20, color='#00d4ff', alpha=0.7, edgecolor='#0099ff')
    ax.set_title('Risk Score Distribution', color='#888')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Number of Patients')
    
    # Trends
    ax = axes[1, 1]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    avg_risk = [55, 54, 52, 53, 51, 50]
    ax.plot(months, avg_risk, marker='o', color='#00d4ff', linewidth=2, markersize=8)
    ax.fill_between(range(len(months)), avg_risk, alpha=0.2, color='#00d4ff')
    ax.set_title('Average Risk Trend', color='#888')
    ax.set_ylabel('Average Risk Score')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    """Plot correlation heatmap"""
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns[:8]
    corr_matrix = data[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                linewidths=0.5, linecolor='#2a3f5f')
    
    ax.set_title('Feature Correlation Matrix', color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_shap_summary(shap_values: Dict[str, float]) -> None:
    """Plot SHAP summary bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(shap_values.keys())
    impacts = list(shap_values.values())
    
    colors = ['#ff1744' if x > 0 else '#00ff88' for x in impacts]
    ax.barh(features, impacts, color=colors, edgecolor='#0099ff', linewidth=1.5)
    
    ax.set_xlabel('SHAP Impact', color='#888')
    ax.set_title('SHAP Summary - Feature Impact on Predictions', color='#00d4ff', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
