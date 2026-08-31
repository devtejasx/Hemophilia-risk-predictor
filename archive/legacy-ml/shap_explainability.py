"""
SHAP Explainability Module for Machine Learning Models
Provides comprehensive model interpretability with TreeExplainer for Random Forest and XGBoost
Generates summary plots, waterfall plots, force plots with simple language explanations
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    Unified SHAP explainability interface for tree-based models
    Supports Random Forest, XGBoost, and other tree-based models
    """
    
    def __init__(self, model, feature_names: List[str], model_type: str = "random_forest"):
        """
        Initialize SHAP explainer for tree-based models
        
        Args:
            model: Trained tree-based model (Random Forest or XGBoost)
            feature_names: List of feature names
            model_type: Type of model ('random_forest', 'xgboost', or 'auto')
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.background_data = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP TreeExplainer for the model"""
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            st.warning(f"Failed to initialize TreeExplainer: {str(e)[:100]}")
            raise
    
    def explain_prediction(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate SHAP explanation for predictions
        
        Args:
            X: Input features (single row or multiple rows)
            
        Returns:
            Dictionary containing SHAP values and metadata
        """
        try:
            # Ensure X is DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names)
            
            # Generate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle binary classification (returns list of arrays)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Take positive class
            
            # Get base value (expected model output)
            base_value = self.explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
            
            # Get model predictions
            predictions = self.model.predict_proba(X)
            if isinstance(predictions, np.ndarray) and len(predictions.shape) > 1:
                predictions = predictions[:, 1]
            
            return {
                "shap_values": shap_values,
                "base_value": base_value,
                "predictions": predictions,
                "features": self.feature_names,
                "X": X
            }
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")
            return None
    
    def get_feature_importance(self, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values
        
        Args:
            X: Input features for background
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance ranking
        """
        try:
            shap_values = self.explainer.shap_values(X)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values
            importance = np.abs(shap_values).mean(axis=0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                "Feature": self.feature_names[:len(importance)],
                "Importance": importance
            }).sort_values("Importance", ascending=False).head(top_n)
            
            return importance_df
        except Exception as e:
            st.error(f"Error calculating feature importance: {str(e)}")
            return pd.DataFrame()


class SHAPVisualizer:
    """
    SHAP visualization functions for different plot types
    Includes summary plots, force plots, waterfall plots, and more
    """
    
    @staticmethod
    def plot_summary(explanation: Dict, top_features: int = 10, plot_type: str = "bar") -> Optional[plt.Figure]:
        """
        Generate SHAP summary plot
        
        Args:
            explanation: SHAP explanation dictionary from SHAPExplainer
            top_features: Number of top features to display
            plot_type: 'bar' (default) or 'violin'
            
        Returns:
            Matplotlib figure object
        """
        try:
            shap_values = explanation["shap_values"]
            X = explanation["X"]
            feature_names = explanation["features"]
            
            # Ensure 2D shape for plotting
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Create figure with dark theme
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if plot_type == "bar":
                # Mean absolute SHAP values
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                indices = np.argsort(mean_abs_shap)[-top_features:]
                
                colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, top_features))
                ax.barh(range(len(indices)), mean_abs_shap[indices], color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_names[i] for i in indices])
                ax.set_xlabel("Mean |SHAP Value|", fontweight='bold', fontsize=11)
                ax.set_title("🧠 SHAP Summary Plot - Feature Importance", fontweight='bold', fontsize=13, pad=15)
            
            elif plot_type == "violin":
                # Prepare data for violin plot
                shap_data = []
                labels = []
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                indices = np.argsort(mean_abs_shap)[-top_features:]
                
                for idx in indices:
                    shap_data.append(shap_values[:, idx])
                    labels.append(feature_names[idx])
                
                positions = range(len(indices))
                parts = ax.violinplot(shap_data, positions=positions, widths=0.7, showmeans=True)
                
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel("SHAP Value", fontweight='bold', fontsize=11)
                ax.set_title("🧠 SHAP Summary Plot - Value Distribution", fontweight='bold', fontsize=13, pad=15)
            
            # Dark theme styling
            ax.set_facecolor('#0a0e27')
            fig.patch.set_facecolor('#0a0e27')
            ax.tick_params(colors='#e0e6ff', labelsize=10)
            ax.spines['bottom'].set_color('#e0e6ff')
            ax.spines['left'].set_color('#e0e6ff')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            st.error(f"Error generating summary plot: {str(e)}")
            return None
    
    @staticmethod
    def plot_waterfall(explanation: Dict, instance_idx: int = 0, top_features: int = 10) -> Optional[plt.Figure]:
        """
        Generate SHAP waterfall plot for individual prediction
        Shows how each feature contribution builds up to final prediction
        
        Args:
            explanation: SHAP explanation dictionary
            instance_idx: Index of instance to explain (default 0)
            top_features: Number of top contributing features to display
            
        Returns:
            Matplotlib figure object
        """
        try:
            shap_values = explanation["shap_values"]
            base_value = explanation["base_value"]
            feature_names = explanation["features"]
            X = explanation["X"]
            prediction = explanation["predictions"][instance_idx]
            
            # Ensure correct shape
            if len(shap_values.shape) == 1:
                shap_vals = shap_values
            else:
                shap_vals = shap_values[instance_idx]
            
            # Get feature values for this instance
            feature_values = X.iloc[instance_idx].values if isinstance(X, pd.DataFrame) else X[instance_idx]
            
            # Create DataFrame with SHAP values
            shap_df = pd.DataFrame({
                "Feature": feature_names[:len(shap_vals)],
                "SHAP_Value": shap_vals,
                "Feature_Value": feature_values[:len(shap_vals)],
                "Abs_SHAP": np.abs(shap_vals)
            }).sort_values("Abs_SHAP", ascending=False).head(top_features)
            
            # Calculate cumulative values for waterfall
            cumulative = np.cumsum(shap_df["SHAP_Value"].values)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Colors: blue for positive impact, red for negative
            colors = ['#00d4ff' if x > 0 else '#ff6b6b' for x in shap_df["SHAP_Value"]]
            
            # Plot bars
            y_pos = np.arange(len(shap_df))
            ax.barh(y_pos, shap_df["SHAP_Value"], color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
            
            # Add reference line at 0
            ax.axvline(x=0, color='white', linestyle='-', linewidth=1)
            
            # Labels and formatting
            feature_labels = [f"{feat}\n({val:.3f})" for feat, val in zip(
                shap_df["Feature"], shap_df["Feature_Value"]
            )]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_labels, fontsize=10)
            ax.set_xlabel("SHAP Value (Impact on Prediction)", fontweight='bold', fontsize=11)
            ax.set_title(f"⛲ SHAP Waterfall Plot - Prediction Breakdown\nBase: {base_value:.3f} → Final: {prediction:.3f}",
                        fontweight='bold', fontsize=13, pad=15)
            
            # Dark theme styling
            ax.set_facecolor('#0a0e27')
            fig.patch.set_facecolor('#0a0e27')
            ax.tick_params(colors='#e0e6ff', labelsize=9)
            ax.spines['bottom'].set_color('#e0e6ff')
            ax.spines['left'].set_color('#e0e6ff')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            st.error(f"Error generating waterfall plot: {str(e)}")
            return None
    
    @staticmethod
    def plot_force(explanation: Dict, instance_idx: int = 0) -> Optional[plt.Figure]:
        """
        Generate SHAP force plot for individual prediction
        Shows how features push prediction above or below base value
        
        Args:
            explanation: SHAP explanation dictionary
            instance_idx: Index of instance to explain
            
        Returns:
            Matplotlib figure object
        """
        try:
            shap_values = explanation["shap_values"]
            base_value = explanation["base_value"]
            feature_names = explanation["features"]
            X = explanation["X"]
            prediction = explanation["predictions"][instance_idx]
            
            # Ensure correct shape
            if len(shap_values.shape) == 1:
                shap_vals = shap_values
            else:
                shap_vals = shap_values[instance_idx]
            
            # Get feature values
            feature_values = X.iloc[instance_idx].values if isinstance(X, pd.DataFrame) else X[instance_idx]
            
            # Create DataFrame
            force_df = pd.DataFrame({
                "Feature": feature_names[:len(shap_vals)],
                "SHAP_Value": shap_vals,
                "Feature_Value": feature_values[:len(shap_vals)],
                "Abs_SHAP": np.abs(shap_vals)
            }).sort_values("Abs_SHAP", ascending=False).head(8)
            
            # Split into positive and negative impacts
            positive = force_df[force_df["SHAP_Value"] > 0].sort_values("SHAP_Value", ascending=True)
            negative = force_df[force_df["SHAP_Value"] < 0].sort_values("SHAP_Value", ascending=False)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
            
            # Plot negative (risk increasing) features
            if len(negative) > 0:
                ax1.barh(negative["Feature"], negative["SHAP_Value"], color='#ff6b6b', alpha=0.8, edgecolor='white', linewidth=1.5)
                ax1.set_xlabel("SHAP Value (Risk ↑)", fontweight='bold', fontsize=10, color='#ff6b6b')
                ax1.set_title("⬇️ Risk-Increasing Factors", fontweight='bold', fontsize=11)
            
            # Plot positive (risk decreasing) features
            if len(positive) > 0:
                ax2.barh(positive["Feature"], positive["SHAP_Value"], color='#00d4ff', alpha=0.8, edgecolor='white', linewidth=1.5)
                ax2.set_xlabel("SHAP Value (Risk ↓)", fontweight='bold', fontsize=10, color='#00d4ff')
                ax2.set_title("⬆️ Risk-Decreasing Factors", fontweight='bold', fontsize=11)
            
            # Dark theme
            for ax in [ax1, ax2]:
                ax.set_facecolor('#0a0e27')
                ax.tick_params(colors='#e0e6ff')
                ax.spines['bottom'].set_color('#e0e6ff')
                ax.spines['left'].set_color('#e0e6ff')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            fig.patch.set_facecolor('#0a0e27')
            fig.suptitle(f"⚡ Force Plot - How Features Push Prediction\nBase: {base_value:.3f} → Final: {prediction:.3f}",
                        fontweight='bold', fontsize=12, y=1.02, color='#e0e6ff')
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            st.error(f"Error generating force plot: {str(e)}")
            return None
    
    @staticmethod
    def plot_dependence(explanation: Dict, feature_name: str) -> Optional[plt.Figure]:
        """
        Generate SHAP dependence plot for a specific feature
        Shows relationship between feature value and SHAP value
        
        Args:
            explanation: SHAP explanation dictionary
            feature_name: Name of feature to plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            shap_values = explanation["shap_values"]
            X = explanation["X"]
            features = explanation["features"]
            
            # Find feature index
            if feature_name not in features:
                st.error(f"Feature '{feature_name}' not found in model")
                return None
            
            feature_idx = features.index(feature_name)
            
            # Ensure 2D shape
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Get feature and SHAP values
            feature_vals = X.iloc[:, feature_idx].values if isinstance(X, pd.DataFrame) else X[:, feature_idx]
            shap_vals = shap_values[:, feature_idx]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot with color gradient
            scatter = ax.scatter(feature_vals, shap_vals, c=shap_vals, cmap='RdYlGn_r', 
                               s=100, alpha=0.6, edgecolors='white', linewidth=1)
            
            # Add reference line
            ax.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
            
            # Labels and formatting
            ax.set_xlabel(f"{feature_name} (Feature Value)", fontweight='bold', fontsize=11)
            ax.set_ylabel("SHAP Value", fontweight='bold', fontsize=11)
            ax.set_title(f"📊 SHAP Dependence Plot - {feature_name}", fontweight='bold', fontsize=13, pad=15)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("SHAP Value", fontweight='bold', fontsize=10)
            
            # Dark theme
            ax.set_facecolor('#0a0e27')
            fig.patch.set_facecolor('#0a0e27')
            ax.tick_params(colors='#e0e6ff')
            ax.spines['bottom'].set_color('#e0e6ff')
            ax.spines['left'].set_color('#e0e6ff')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            st.error(f"Error generating dependence plot: {str(e)}")
            return None


class SHAPInterpreter:
    """
    Convert SHAP values to simple language explanations
    Provides clinically relevant interpretations
    """
    
    @staticmethod
    def interpret_prediction(explanation: Dict, instance_idx: int = 0, 
                            risk_threshold: float = 0.5, context: str = "medical") -> Dict[str, str]:
        """
        Generate simple language interpretation of prediction
        
        Args:
            explanation: SHAP explanation dictionary
            instance_idx: Index of instance to interpret
            risk_threshold: Threshold for risk classification
            context: Context for interpretation ('medical', 'clinical', 'patient')
            
        Returns:
            Dictionary with interpretations
        """
        try:
            shap_values = explanation["shap_values"]
            base_value = explanation["base_value"]
            prediction = explanation["predictions"][instance_idx]
            feature_names = explanation["features"]
            X = explanation["X"]
            
            # Ensure correct shape
            if len(shap_values.shape) == 1:
                shap_vals = shap_values
            else:
                shap_vals = shap_values[instance_idx]
            
            # Get feature values
            feature_values = X.iloc[instance_idx].values if isinstance(X, pd.DataFrame) else X[instance_idx]
            
            # Get top contributing features
            top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]  # Top 3
            
            # Build interpretation
            interpretation = {
                "prediction": float(prediction),
                "risk_level": "HIGH" if prediction > risk_threshold else "LOW",
                "prediction_phrase": SHAPInterpreter._predict_phrase(prediction, risk_threshold, context),
                "key_factors": SHAPInterpreter._get_key_factors_explanation(
                    shap_vals[top_indices], 
                    feature_names[top_indices] if isinstance(feature_names, np.ndarray) else [feature_names[i] for i in top_indices],
                    feature_values[top_indices],
                    context
                ),
                "overall_assessment": SHAPInterpreter._get_overall_assessment(
                    prediction, 
                    shap_vals, 
                    feature_names, 
                    context
                )
            }
            
            return interpretation
        
        except Exception as e:
            st.error(f"Error interpreting prediction: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def _predict_phrase(prediction: float, threshold: float, context: str) -> str:
        """Generate simple phrase for prediction"""
        if prediction > threshold + 0.2:
            phrases = {
                "medical": f"Significantly elevated risk ({prediction:.1%})",
                "clinical": f"Strong risk indicators present ({prediction:.1%})",
                "patient": f"High likelihood of condition ({prediction:.1%})"
            }
        elif prediction > threshold:
            phrases = {
                "medical": f"Moderately elevated risk ({prediction:.1%})",
                "clinical": f"Notable risk factors detected ({prediction:.1%})",
                "patient": f"Moderate likelihood of condition ({prediction:.1%})"
            }
        else:
            phrases = {
                "medical": f"Low risk profile ({prediction:.1%})",
                "clinical": f"Minimal risk indicators ({prediction:.1%})",
                "patient": f"Low likelihood of condition ({prediction:.1%})"
            }
        
        return phrases.get(context, phrases["medical"])
    
    @staticmethod
    def _get_key_factors_explanation(shap_vals: np.ndarray, feature_names, 
                                    feature_values, context: str) -> List[str]:
        """Generate explanations for key contributing factors"""
        explanations = []
        
        for shap_val, feat_name, feat_val in zip(shap_vals, feature_names, feature_values):
            direction = "increases" if shap_val > 0 else "decreases"
            impact = "substantially" if abs(shap_val) > 0.1 else "moderately"
            
            if context == "patient":
                explanation = f"'{feat_name}' at {feat_val:.2f} {direction} risk {impact}"
            else:
                explanation = f"{feat_name}={feat_val:.3f} {direction} risk score {direction} by {abs(shap_val):.3f}"
            
            explanations.append(explanation)
        
        return explanations
    
    @staticmethod
    def _get_overall_assessment(prediction: float, shap_vals: np.ndarray, 
                               feature_names, context: str) -> str:
        """Generate overall clinical assessment"""
        total_contribution = np.sum(np.abs(shap_vals))
        dominant_factor_idx = np.argmax(np.abs(shap_vals))
        dominant_factor = feature_names[dominant_factor_idx]
        
        if context == "patient":
            return (f"Based on your clinical profile, the analysis suggests a {prediction:.0%} likelihood. "
                   f"The most influential factor is '{dominant_factor}'. "
                   f"Professional medical review is recommended.")
        else:
            return (f"Model prediction: {prediction:.1%}. "
                   f"Primary driver: {dominant_factor} (absolute contribution: {total_contribution:.3f}). "
                   f"Validate with clinical judgment and additional diagnostic information.")


# ============================================================================
# Streamlit Integration Functions
# ============================================================================

def display_shap_dashboard(explanation: Dict, model_type: str = "Random Forest"):
    """
    Display comprehensive SHAP dashboard in Streamlit
    
    Args:
        explanation: SHAP explanation dictionary
        model_type: Type of model for display
    """
    st.markdown("### 🧠 SHAP Model Explainability Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Base Value (Expected)", f"{explanation['base_value']:.3f}", delta=None)
        if len(explanation['predictions']) > 0:
            st.metric("Prediction", f"{explanation['predictions'][0]:.3f}")
    
    with col2:
        st.metric("Model Type", model_type)
        st.metric("Features Analyzed", len(explanation['features']))
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "⛲ Waterfall", "⚡ Force", "Interpretation"])
    
    with tab1:
        st.subheader("Feature Importance Summary")
        plot = SHAPVisualizer.plot_summary(explanation, top_features=10, plot_type="bar")
        if plot:
            st.pyplot(plot)
    
    with tab2:
        st.subheader("Prediction Waterfall")
        col_inst = st.columns(1)
        instance_idx = st.slider("Select Instance", 0, len(explanation['predictions'])-1, 0)
        plot = SHAPVisualizer.plot_waterfall(explanation, instance_idx=instance_idx, top_features=10)
        if plot:
            st.pyplot(plot)
    
    with tab3:
        st.subheader("Force Plot - Risk Drivers")
        plot = SHAPVisualizer.plot_force(explanation, instance_idx=instance_idx)
        if plot:
            st.pyplot(plot)
    
    with tab4:
        st.subheader("Simple Language Interpretation")
        interpreter = SHAPInterpreter()
        interpretation = interpreter.interpret_prediction(explanation, instance_idx=instance_idx, context="clinical")
        
        if "error" not in interpretation:
            st.markdown(f"**Risk Level:** `{interpretation['risk_level']}`")
            st.markdown(f"**Assessment:** {interpretation['prediction_phrase']}")
            st.markdown("**Key Contributing Factors:**")
            for factor in interpretation['key_factors']:
                st.markdown(f"- {factor}")
            st.markdown(f"**Overall Assessment:** {interpretation['overall_assessment']}")


def display_feature_importance(explanation: Dict, top_n: int = 10):
    """
    Display feature importance ranking
    
    Args:
        explanation: SHAP explanation dictionary
        top_n: Number of top features to display
    """
    shap_values = explanation["shap_values"]
    feature_names = explanation["features"]
    
    # Ensure 2D
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)
    
    # Calculate importance
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names[:len(importance)],
        "Importance": importance
    }).sort_values("Importance", ascending=False).head(top_n)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_xlabel("Mean |SHAP Value|", fontweight='bold')
        ax.set_title("Global Feature Importance", fontweight='bold', fontsize=12)
        ax.set_facecolor('#0a0e27')
        fig.patch.set_facecolor('#0a0e27')
        ax.tick_params(colors='#e0e6ff')
        ax.spines['bottom'].set_color('#e0e6ff')
        ax.spines['left'].set_color('#e0e6ff')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Feature Ranking**")
        for idx, row in importance_df.iterrows():
            st.metric(row["Feature"], f"{row['Importance']:.4f}")


def explain_individual_prediction(explanation: Dict, instance_idx: int = 0):
    """
    Detailed explanation of individual prediction
    
    Args:
        explanation: SHAP explanation dictionary
        instance_idx: Index of instance to explain
    """
    interpreter = SHAPInterpreter()
    interpretation = interpreter.interpret_prediction(explanation, instance_idx=instance_idx)
    
    if "error" in interpretation:
        st.error("Error generating interpretation")
        return
    
    # Display in clinical format
    st.markdown("### 📋 Prediction Explanation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction Score", f"{interpretation['prediction']:.1%}")
    with col2:
        st.metric("Risk Level", interpretation['risk_level'])
    with col3:
        st.metric("Assessment", interpretation['prediction_phrase'][:20] + "...")
    
    st.markdown("#### Key Contributing Factors:")
    for i, factor in enumerate(interpretation['key_factors'], 1):
        st.write(f"{i}. {factor}")
    
    st.markdown(f"**Clinical Assessment:** {interpretation['overall_assessment']}")
