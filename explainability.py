"""
Machine Learning Explainability Module - SHAP + LIME
======================================================

Implements comprehensive explainability using both SHAP (SHapley Additive exPlanations)
and LIME (Local Interpretable Model-agnostic Explanations).

Why SHAP?
- Game-theoretic approach: calculates each feature's contribution to prediction
- Globally consistent interpretations
- Shows feature importance across entire model
- Explains both individual predictions and model behavior
- SHAP values: "How much does this feature change the prediction from the base value?"

Why LIME?
- Local linear approximation around specific prediction
- Model-agnostic: works with any model type
- Easier to understand for clinicians: "This prediction is mainly because X"
- Provides local explanations for individual patients
- LIME perturbations: "What happens if we slightly change this patient's data?"

This implements the "Explainable AI" component of our PPT.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import warnings
from typing import Dict, Tuple, Optional, List
import joblib

warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAP-based model explainability for global and local interpretations.
    """
    
    def __init__(self, model, X_background: pd.DataFrame):
        """
        Initialize SHAP explainer.
        
        SHAP uses model background data to establish "baseline" predictions.
        What's the average prediction across all patients?
        How much does this particular patient deviate from average?
        
        Args:
            model: Trained ML model (sklearn, XGBoost, CatBoost, etc)
            X_background: Background dataset for establishing baseline (use sample/subset)
        """
        self.model = model
        self.X_background = X_background
        self.explainer = None
        self.shap_values = None
        
        # Initialize explainer based on model type
        self._init_explainer()
    
    def _init_explainer(self):
        """Initialize appropriate SHAP explainer for model type."""
        try:
            # Try TreeExplainer first (for tree-based models)
            try:
                self.explainer = shap.TreeExplainer(self.model)
                print("✅ Using TreeExplainer (optimal for tree models)")
                return
            except:
                pass
            
            # Fall back to KernelExplainer for any model
            print("ℹ️  Using KernelExplainer (works with any model, slower)")
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,  # Probability predictions
                self.X_background
            )
        except Exception as e:
            print(f"⚠️  Could not initialize SHAP explainer: {e}")
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for dataset.
        
        SHAP value for a feature: "How much does this feature's value
        contribute to moving the prediction away from the base value?"
        
        Args:
            X: Features to explain
            
        Returns:
            SHAP values array (same shape as X)
        """
        if self.explainer is None:
            print("❌ Explainer not initialized")
            return None
        
        try:
            print(f"Computing SHAP values for {len(X)} samples...")
            self.shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class output
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Use positive class
            
            print(f"✅ SHAP values computed: {self.shap_values.shape}")
            return self.shap_values
            
        except Exception as e:
            print(f"❌ Error computing SHAP values: {e}")
            return None
    
    def plot_waterfall(self, X: pd.DataFrame, sample_idx: int = 0,
                      save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot SHAP waterfall for individual prediction.
        
        Waterfall shows: Base value → Features push up/down → Final prediction
        
        Args:
            X: Features
            sample_idx: Which sample to explain
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        try:
            # Get base value (average model output)
            base_value = self.explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
            
            # Create explanation object
            explanation = shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=base_value,
                data=X.iloc[sample_idx].values,
                feature_names=X.columns.tolist()
            )
            
            fig = plt.figure(figsize=(14, 8))
            shap.plots.waterfall(explanation, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Waterfall plot saved → {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating waterfall: {e}")
            return None
    
    def plot_summary(self, X: pd.DataFrame, feature_count: int = 20,
                    save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot SHAP summary (feature importance) across dataset.
        
        Summary shows which features have biggest impact on predictions,
        and whether impact is positive (increases risk) or negative (decreases risk).
        
        Args:
            X: Features
            feature_count: Number of top features to display
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create explanation
            base_value = self.explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
            
            explanation = shap.Explanation(
                values=self.shap_values,
                base_values=base_value,
                data=X.values,
                feature_names=X.columns.tolist()
            )
            
            shap.summary_plot(explanation, plot_type="bar", show=False, max_display=feature_count)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Summary plot saved → {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating summary: {e}")
            return None
    
    def get_feature_importance(self, X: pd.DataFrame, 
                              feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get mean absolute SHAP value (feature importance) for all features.
        
        Args:
            X: Features
            feature_names: Feature names (default: X.columns)
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        if feature_names is None:
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Mean absolute SHAP value = importance
        importance = np.abs(self.shap_values).mean(axis=0)
        importance_dict = dict(zip(feature_names, importance))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0,
                          top_features: int = 5) -> Dict:
        """
        Generate textual explanation for a prediction.
        
        Args:
            X: Features
            sample_idx: Which sample to explain
            top_features: Number of top contributing features
            
        Returns:
            Dictionary with explanation details
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Get prediction
        prediction = self.model.predict(X.iloc[[sample_idx]])[0]
        probability = self.model.predict_proba(X.iloc[[sample_idx]])[0][1]
        
        # Get SHAP values for this sample
        sample_shap = self.shap_values[sample_idx]
        
        # Get feature names
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Find top contributing features (largest absolute SHAP values)
        top_idx = np.argsort(np.abs(sample_shap))[-top_features:][::-1]
        
        explanation = {
            'prediction': int(prediction),
            'probability': float(probability),
            'prediction_text': 'High Inhibitor Risk' if prediction == 1 else 'Low Inhibitor Risk',
            'top_features': []
        }
        
        # Format top features
        for idx in top_idx:
            feature_value = X.iloc[sample_idx, idx]
            shap_value = sample_shap[idx]
            contribution = 'increases' if shap_value > 0 else 'decreases'
            
            explanation['top_features'].append({
                'feature': feature_names[idx],
                'value': float(feature_value),
                'shap_value': float(shap_value),
                'contribution': contribution
            })
        
        return explanation


class LIMEExplainer:
    """
    LIME-based local model explanations for individual predictions.
    """
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        Initialize LIME explainer.
        
        LIME: "Pretend the model is linear around this specific prediction.
        Which features matter the most locally?"
        
        Args:
            model: Trained ML model
            X_train: Training data (for feature statistics)
            feature_names: Names of features
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        
        # LIME expects numpy arrays
        X_np = X_train.values if hasattr(X_train, 'values') else X_train
        
        # Initialize tabular explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_np,
            feature_names=self.feature_names,
            class_names=['No Inhibitor', 'Inhibitor'],
            mode='classification',
            random_state=42
        )
    
    def explain_instance(self, X: pd.DataFrame, sample_idx: int = 0,
                        num_features: int = 5) -> Dict:
        """
        Generate LIME explanation for individual sample.
        
        LIME creates a local linear approximation around the sample.
        It perturbs the input and sees how model's prediction changes.
        
        Args:
            X: Features
            sample_idx: Which sample to explain
            num_features: Number of features to highlight
            
        Returns:
            Dictionary with explanation
        """
        try:
            # Convert to numpy
            X_np = X.values if hasattr(X, 'values') else X
            
            # Get sample
            sample = X_np[sample_idx]
            
            # Get LIME explanation
            exp = self.explainer.explain_instance(
                sample,
                self.model.predict_proba,
                num_features=num_features,
                top_labels=1
            )
            
            # Extract explanation
            explanation = {
                'prediction': int(self.model.predict([sample])[0]),
                'probability': float(self.model.predict_proba([sample])[0][1]),
                'features': []
            }
            
            # Get feature contributions
            for feature_idx, weight in exp.as_list():
                explanation['features'].append({
                    'feature_description': feature_idx,
                    'weight': weight
                })
            
            return explanation
            
        except Exception as e:
            print(f"❌ Error in LIME explanation: {e}")
            return {}
    
    def plot_explanation(self, X: pd.DataFrame, sample_idx: int = 0,
                        num_features: int = 10,
                        save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot LIME explanation.
        
        Args:
            X: Features
            sample_idx: Which sample to explain
            num_features: Number of features to display
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        try:
            X_np = X.values if hasattr(X, 'values') else X
            sample = X_np[sample_idx]
            
            exp = self.explainer.explain_instance(
                sample,
                self.model.predict_proba,
                num_features=num_features,
                top_labels=1
            )
            
            fig = exp.show_in_notebook(show_table=True, show_all=False)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ LIME plot saved → {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Error plotting LIME explanation: {e}")
            return None


class ExplainabilityComparison:
    """
    Compare SHAP and LIME explanations for the same prediction.
    """
    
    def __init__(self, model, X_train: pd.DataFrame, X_background: Optional[pd.DataFrame] = None):
        """
        Initialize with both explainers.
        
        Args:
            model: Trained ML model
            X_train: Training data
            X_background: Background data for SHAP (default: use X_train sample)
        """
        self.model = model
        self.X_train = X_train
        
        # Use background data (smaller sample for SHAP efficiency)
        if X_background is None:
            X_background = X_train.sample(min(100, len(X_train)), random_state=42)
        
        self.shap_explainer = SHAPExplainer(model, X_background)
        self.lime_explainer = LIMEExplainer(model, X_train)
    
    def compare_explanations(self, X: pd.DataFrame, sample_idx: int = 0,
                            top_features: int = 5) -> Dict:
        """
        Compare SHAP and LIME explanations.
        
        Args:
            X: Features
            sample_idx: Which sample to explain
            top_features: Number of top features
            
        Returns:
            Dictionary with both explanations
        """
        print(f"\n🔍 Generating Explanations for Sample {sample_idx}...\n")
        
        # SHAP explanation
        print("📊 SHAP (Global + Local) Explanation:")
        shap_explanation = self.shap_explainer.explain_prediction(X, sample_idx, top_features)
        print(f"   Prediction: {shap_explanation['prediction_text']}")
        print(f"   Probability: {shap_explanation['probability']:.2%}")
        print("   Top Contributing Factors:")
        for feat in shap_explanation['top_features'][:3]:
            print(f"      - {feat['feature']}: {feat['contribution']} risk (SHAP: {feat['shap_value']:.4f})")
        
        # LIME explanation
        print("\n💬 LIME (Local Linear) Explanation:")
        lime_explanation = self.lime_explainer.explain_instance(X, sample_idx, top_features)
        print(f"   Prediction: {'High Inhibitor Risk' if lime_explanation['prediction'] == 1 else 'Low Inhibitor Risk'}")
        print(f"   Probability: {lime_explanation['probability']:.2%}")
        print("   Local Feature Importance:")
        for feat in lime_explanation['features'][:3]:
            print(f"      - {feat['feature_description']}")
        
        return {
            'shap': shap_explanation,
            'lime': lime_explanation
        }


if __name__ == "__main__":
    print("Explainability Module Ready")
    print("Supports SHAP (global/local) and LIME (local linear) explanations")
