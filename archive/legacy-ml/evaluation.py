"""
Machine Learning Model Evaluation Module
==========================================

Comprehensive evaluation of trained models with metrics, visualizations, and reporting.
Supports multiple models and generates detailed performance analysis.

Features:
- Calculate accuracy, precision, recall, F1-score, ROC-AUC
- Generate confusion matrix visualization
- Generate ROC curve with AUC scores
- Create detailed evaluation reports
- Compare multiple models
- Export results to CSV/JSON
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle
import gc

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """
    Comprehensive ML model evaluation and analysis.
    
    Handles evaluation of trained classification models with support for
    multiple models, metrics calculation, and visualization generation.
    """
    
    def __init__(self, data_path: str = "genomic.csv", clinical_path: str = "clinical.csv"):
        """
        Initialize evaluator with data paths.
        
        Args:
            data_path: Path to genomic data CSV
            clinical_path: Path to clinical data CSV
        """
        self.data_path = data_path
        self.clinical_path = clinical_path
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.train_columns = []
        
    def load_data(self, test_size: float = 0.2, random_state: int = 42) -> bool:
        """
        Load and prepare data for evaluation.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("📊 Loading data for evaluation...")
            
            # Check if files exist
            if not os.path.exists(self.data_path):
                print(f"❌ Error: {self.data_path} not found")
                return False
            if not os.path.exists(self.clinical_path):
                print(f"❌ Error: {self.clinical_path} not found")
                return False
            
            # Load data
            genomic = pd.read_csv(self.data_path, low_memory=False)
            clinical = pd.read_csv(self.clinical_path, low_memory=False)
            
            # Merge
            df = pd.merge(genomic, clinical, on="patient_id", how="inner")
            
            # Clean: keep only 0 and 1 targets
            df = df[df["target"].isin([0, 1])]
            
            # Prepare X and y
            y = df["target"]
            X = df.drop(["target", "patient_id"], axis=1, errors="ignore")
            
            # Encode categorical variables
            X = pd.get_dummies(X)
            self.train_columns = list(X.columns)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"✅ Data loaded successfully!")
            print(f"   Train: {len(self.X_train)} samples, Test: {len(self.X_test)} samples")
            print(f"   Features: {X.shape[1]}")
            print(f"   Class distribution - Train: {self.y_train.value_counts().to_dict()}")
            print(f"                       - Test:  {self.y_test.value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def load_models(self, model_paths: Dict[str, str]) -> bool:
        """
        Load pre-trained models.
        
        Args:
            model_paths: Dictionary mapping model names to file paths
                        Example: {"RandomForest": "rf.pkl", "XGBoost": "xgb.pkl"}
        
        Returns:
            bool: True if at least one model loaded successfully (or mock models available)
        """
        try:
            print("\n🤖 Loading models...")
            success = True
            models_found = False
            
            for model_name, path in model_paths.items():
                if not os.path.exists(path):
                    print(f"⚠️  Model file not found: {path} - Using mock model for testing")
                    # Create a mock model for testing
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        from xgboost import XGBClassifier
                        
                        if "RandomForest" in model_name:
                            self.models[model_name] = RandomForestClassifier(n_estimators=10, random_state=42)
                            print(f"✅ Created mock {model_name} for testing")
                            models_found = True
                        elif "XGBoost" in model_name or "xgb" in model_name.lower():
                            self.models[model_name] = XGBClassifier(n_estimators=10, random_state=42, use_label_encoder=False)
                            print(f"✅ Created mock {model_name} for testing")
                            models_found = True
                    except Exception as e:
                        print(f"⚠️  Could not create mock {model_name}: {e}")
                        success = False
                    continue
                
                try:
                    model = joblib.load(path)
                    self.models[model_name] = model
                    print(f"✅ Loaded: {model_name} from {path}")
                    models_found = True
                except Exception as e:
                    print(f"⚠️  Error loading {model_name}: {e} - Will use mock model")
                    # Fallback to mock model
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        from xgboost import XGBClassifier
                        
                        if "RandomForest" in model_name:
                            self.models[model_name] = RandomForestClassifier(n_estimators=10, random_state=42)
                            print(f"✅ Created mock {model_name}")
                            models_found = True
                        elif "XGBoost" in model_name or "xgb" in model_name.lower():
                            self.models[model_name] = XGBClassifier(n_estimators=10, random_state=42, use_label_encoder=False)
                            print(f"✅ Created mock {model_name}")
                            models_found = True
                    except Exception as mock_error:
                        print(f"❌ Could not load or mock {model_name}: {mock_error}")
                        success = False
            
            if not self.models:
                print("❌ No models loaded successfully and could not create mocks")
                return False
            
            if models_found:
                print(f"✅ Models loaded: {', '.join(self.models.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in load_models: {e}")
            return False
    
    def evaluate_model(self, model_name: str, model=None) -> Dict:
        """
        Evaluate a single model and calculate all metrics.
        
        Args:
            model_name: Name of the model
            model: Model object (if not in self.models)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        try:
            if model is None:
                if model_name not in self.models:
                    print(f"❌ Model '{model_name}' not found")
                    return {}
                model = self.models[model_name]
            
            print(f"\n📈 Evaluating {model_name}...")
            
            # Get predictions
            y_pred = model.predict(self.X_test)
            
            # Get probability predictions for ROC curve
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            else:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = {
                'model_name': model_name,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
                'classification_report': classification_report(
                    self.y_test, y_pred, output_dict=True
                )
            }
            
            # Store predictions for visualization
            metrics['y_pred'] = y_pred
            metrics['y_pred_proba'] = y_pred_proba
            
            self.results[model_name] = metrics
            
            # Print summary
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1_score']:.4f}")
            if metrics['roc_auc']:
                print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            return {}
    
    def evaluate_all_models(self) -> Dict:
        """
        Evaluate all loaded models.
        
        Returns:
            Dictionary containing results for all models
        """
        if not self.models:
            print("❌ No models loaded. Call load_models() first.")
            return {}
        
        for model_name in self.models:
            self.evaluate_model(model_name)
        
        return self.results
    
    def generate_confusion_matrix_plot(
        self, 
        model_name: str, 
        save_path: str = "confusion_matrix.png"
    ) -> str:
        """
        Generate and save confusion matrix visualization.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            if model_name not in self.results:
                print(f"❌ No results for {model_name}. Evaluate first.")
                return ""
            
            results = self.results[model_name]
            cm = np.array(results['confusion_matrix'])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot confusion matrix
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                cbar=True,
                ax=ax,
                xticklabels=['No Risk', 'Risk'],
                yticklabels=['No Risk', 'Risk']
            )
            
            ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Confusion matrix saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error generating confusion matrix: {e}")
            return ""
    
    def generate_roc_curve_plot(
        self, 
        model_names: Optional[List[str]] = None,
        save_path: str = "roc_curve.png"
    ) -> str:
        """
        Generate and save ROC curve visualization.
        
        Args:
            model_names: List of model names to include (None = all)
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            if model_names is None:
                model_names = list(self.results.keys())
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot ROC curve for each model
            for model_name in model_names:
                if model_name not in self.results:
                    print(f"⚠️  Skipping {model_name} - not evaluated")
                    continue
                
                results = self.results[model_name]
                
                if results['y_pred_proba'] is None:
                    print(f"⚠️  {model_name} doesn't support probability predictions")
                    continue
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
                roc_auc = results['roc_auc']
                
                # Plot
                ax.plot(
                    fpr, tpr, 
                    lw=2.5,
                    label=f'{model_name} (AUC = {roc_auc:.4f})'
                )
            
            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ ROC curve saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error generating ROC curve: {e}")
            return ""
    
    def generate_metrics_comparison_plot(
        self, 
        model_names: Optional[List[str]] = None,
        save_path: str = "metrics_comparison.png"
    ) -> str:
        """
        Generate bar plot comparing metrics across models.
        
        Args:
            model_names: List of model names to compare
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            if model_names is None:
                model_names = list(self.results.keys())
            
            # Prepare data
            metrics_data = {
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1-Score': [],
                'ROC-AUC': []
            }
            
            for model_name in model_names:
                if model_name not in self.results:
                    continue
                
                results = self.results[model_name]
                metrics_data['Accuracy'].append(results['accuracy'])
                metrics_data['Precision'].append(results['precision'])
                metrics_data['Recall'].append(results['recall'])
                metrics_data['F1-Score'].append(results['f1_score'])
                metrics_data['ROC-AUC'].append(results['roc_auc'] if results['roc_auc'] else 0)
            
            # Create dataframe
            df_metrics = pd.DataFrame(metrics_data, index=model_names)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            df_metrics.plot(kind='bar', ax=ax, width=0.8)
            
            ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylim([0, 1.05])
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Metrics comparison saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error generating metrics comparison: {e}")
            return ""
    
    def generate_report(self, save_path: str = "evaluation_report.json") -> str:
        """
        Generate comprehensive evaluation report as JSON.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'train_samples': len(self.X_train),
                    'test_samples': len(self.X_test),
                    'features': len(self.train_columns),
                    'class_distribution_train': self.y_train.value_counts().to_dict(),
                    'class_distribution_test': self.y_test.value_counts().to_dict()
                },
                'models': {}
            }
            
            for model_name, results in self.results.items():
                report['models'][model_name] = {
                    'accuracy': float(results['accuracy']),
                    'precision': float(results['precision']),
                    'recall': float(results['recall']),
                    'f1_score': float(results['f1_score']),
                    'roc_auc': float(results['roc_auc']) if results['roc_auc'] else None,
                    'confusion_matrix': results['confusion_matrix'],
                    'classification_report': results['classification_report']
                }
            
            # Save report
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"✅ Report saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return ""
    
    def export_metrics_csv(self, save_path: str = "model_metrics.csv") -> str:
        """
        Export metrics to CSV for easy comparison.
        
        Args:
            save_path: Path to save the CSV
            
        Returns:
            Path to saved CSV
        """
        try:
            rows = []
            
            for model_name, results in self.results.items():
                row = {
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score'],
                    'ROC-AUC': results['roc_auc'] if results['roc_auc'] else 'N/A'
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(save_path, index=False)
            
            print(f"✅ Metrics CSV saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error exporting metrics: {e}")
            return ""
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all evaluated models.
        
        Returns:
            DataFrame with summary statistics
        """
        rows = []
        
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'ROC-AUC': f"{results['roc_auc']:.4f}" if results['roc_auc'] else "N/A"
            }
            rows.append(row)
        
        return pd.DataFrame(rows)


def quick_evaluate() -> Dict:
    """
    Quick evaluation with default settings.
    
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Load data
        if not evaluator.load_data():
            return {}
        
        # Load models
        model_paths = {
            'Random Forest': 'rf.pkl',
            'XGBoost': 'xgb.pkl'
        }
        
        if not evaluator.load_models(model_paths):
            print("⚠️  Some models failed to load")
        
        # Evaluate all models
        evaluator.evaluate_all_models()
        
        # Generate visualizations
        evaluator.generate_confusion_matrix_plot('Random Forest')
        if 'XGBoost' in evaluator.results:
            evaluator.generate_confusion_matrix_plot('XGBoost', 'confusion_matrix_xgb.png')
        
        evaluator.generate_roc_curve_plot()
        evaluator.generate_metrics_comparison_plot()
        
        # Generate reports
        evaluator.generate_report()
        evaluator.export_metrics_csv()
        
        return evaluator.results
        
    except Exception as e:
        print(f"❌ Error in quick_evaluate: {e}")
        return {}


if __name__ == "__main__":
    """
    Example usage of the evaluation module
    """
    print("=" * 60)
    print("ML MODEL EVALUATION")
    print("=" * 60)
    
    results = quick_evaluate()
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)
