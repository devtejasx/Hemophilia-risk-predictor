"""
Class Imbalance Handling Module with SMOTE
============================================

Implements SMOTE (Synthetic Minority Over-sampling Technique) to handle
class imbalance in inhibitor risk prediction dataset.

Why SMOTE?
- Real-world inhibitor development is rare (class imbalance ~10-20% positive)
- Training models on imbalanced data leads to bias toward majority class
- SMOTE synthetically generates minority samples avoiding overfitting
- Creates "synthetic" inhibitor cases by interpolating between existing cases
- Improves minority class recall and F1-score significantly

This implements the "Class Imbalance Handling" component of our PPT.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# SMOTE and related techniques
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold


class ClassImbalanceHandler:
    """
    Comprehensive class imbalance handling using SMOTE and related techniques.
    
    Handles the common problem in medical datasets where positive cases
    (inhibitor development) are much rarer than negative cases.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the imbalance handler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.original_distributions = {}
        self.smoted_distributions = {}
        self.smote_pipeline = None
        
    def analyze_imbalance(self, y: pd.Series, dataset_name: str = "Dataset") -> Dict:
        """
        Analyze class imbalance in the dataset.
        
        Args:
            y: Target variable
            dataset_name: Name of dataset (for reporting)
            
        Returns:
            Dictionary with imbalance statistics
        """
        class_counts = y.value_counts().sort_index()
        class_dist = y.value_counts(normalize=True).sort_index()
        
        stats = {
            'class_counts': class_counts.to_dict(),
            'class_distribution': class_dist.to_dict(),
            'imbalance_ratio': class_dist[1] / class_dist[0] if len(class_dist) > 1 else 0,
            'minority_class': 1 if class_dist[1] < class_dist[0] else 0,
            'majority_class': 0 if class_dist[1] < class_dist[0] else 1,
        }
        
        print(f"\n📊 Class Imbalance Analysis - {dataset_name}")
        print(f"   Class 0 (No Inhibitor):  {class_counts[0]:5d} samples ({class_dist[0]*100:5.1f}%)")
        print(f"   Class 1 (Inhibitor):     {class_counts[1]:5d} samples ({class_dist[1]*100:5.1f}%)")
        print(f"   Imbalance Ratio:         1:{1/stats['imbalance_ratio']:.2f}")
        
        self.original_distributions[dataset_name] = stats
        
        return stats
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series,
                   sampling_strategy: float = 0.8,
                   k_neighbors: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance the dataset.
        
        Why k_neighbors = 5?
        SMOTE generates synthetic samples by:
        1. Finding k-nearest neighbors of each minority sample
        2. Randomly selecting one neighbor
        3. Interpolating between the sample and neighbor
        4. Creates more diverse synthetic samples than simple oversampling
        
        Args:
            X: Feature matrix
            y: Target variable
            sampling_strategy: Target ratio of minority to majority class (0.8 = 80%)
            k_neighbors: Number of nearest neighbors for SMOTE
            
        Returns:
            Tuple of (X_smoted, y_smoted) - balanced datasets
        """
        print("\n🔄 Applying SMOTE (Synthetic Minority Over-sampling)...")
        print(f"   Target sampling ratio: {sampling_strategy:.1f}")
        print(f"   k-neighbors: {k_neighbors}")
        
        try:
            # Create SMOTE sampler
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Apply SMOTE
            X_smoted, y_smoted = smote.fit_resample(X, y)
            
            # Analyze result
            X_smoted_df = pd.DataFrame(X_smoted, columns=X.columns)
            y_smoted_series = pd.Series(y_smoted)
            
            class_counts_after = y_smoted_series.value_counts().sort_index()
            class_dist_after = y_smoted_series.value_counts(normalize=True).sort_index()
            
            print(f"\n   ✅ SMOTE Applied Successfully!")
            print(f"   Before SMOTE:")
            print(f"      Class 0: {y.value_counts()[0]:6d} samples")
            print(f"      Class 1: {y.value_counts()[1]:6d} samples")
            print(f"   After SMOTE:")
            print(f"      Class 0: {class_counts_after[0]:6d} samples")
            print(f"      Class 1: {class_counts_after[1]:6d} samples")
            print(f"   New Imbalance Ratio: 1:{class_counts_after[0]/class_counts_after[1]:.2f}")
            
            self.smoted_distributions['after_smote'] = {
                'class_counts': class_counts_after.to_dict(),
                'class_distribution': class_dist_after.to_dict()
            }
            
            return X_smoted_df, y_smoted_series
            
        except Exception as e:
            print(f"❌ Error applying SMOTE: {e}")
            return X, y
    
    def apply_adasyn(self, X: pd.DataFrame, y: pd.Series,
                    sampling_strategy: float = 0.8) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply ADASYN (Adaptive Synthetic Sampling) - more sophisticated than SMOTE.
        
        Why ADASYN?
        - Adapts sampling density based on difficulty of each minority sample
        - Focuses on harder-to-learn cases
        - Often better than SMOTE for complex models
        
        Args:
            X: Feature matrix
            y: Target variable
            sampling_strategy: Target ratio of minority to majority class
            
        Returns:
            Tuple of (X_adasyn, y_adasyn) - adaptively balanced datasets
        """
        print("\n🔄 Applying ADASYN (Adaptive Synthetic Sampling)...")
        print(f"   Target sampling ratio: {sampling_strategy:.1f}")
        
        try:
            adasyn = ADASYN(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
            X_adasyn_df = pd.DataFrame(X_adasyn, columns=X.columns)
            y_adasyn_series = pd.Series(y_adasyn)
            
            class_counts_after = y_adasyn_series.value_counts().sort_index()
            
            print(f"\n   ✅ ADASYN Applied Successfully!")
            print(f"   After ADASYN:")
            print(f"      Class 0: {class_counts_after[0]:6d} samples")
            print(f"      Class 1: {class_counts_after[1]:6d} samples")
            
            return X_adasyn_df, y_adasyn_series
            
        except Exception as e:
            print(f"❌ Error applying ADASYN: {e}")
            return X, y
    
    def apply_combined_strategy(self, X: pd.DataFrame, y: pd.Series,
                               over_ratio: float = 0.8,
                               under_ratio: float = 1.0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply combined over-sampling (SMOTE) and under-sampling for balanced dataset.
        
        Why Combined Strategy?
        - SMOTE handles minority class
        - Random under-sampling reduces majority class
        - Combined approach is often best for very imbalanced data
        
        Args:
            X: Feature matrix
            y: Target variable
            over_ratio: SMOTE target ratio
            under_ratio: Under-sampling ratio for majority class
            
        Returns:
            Tuple of (X_balanced, y_balanced) - balanced datasets
        """
        print("\n🔄 Applying Combined SMOTE + Random Under-sampling Strategy...")
        print(f"   SMOTE sampling ratio: {over_ratio:.1f}")
        print(f"   Under-sampling ratio: {under_ratio:.1f}")
        
        try:
            pipeline = ImbPipeline([
                ('smote', SMOTE(sampling_strategy=over_ratio, random_state=self.random_state)),
                ('under', RandomUnderSampler(sampling_strategy=under_ratio, random_state=self.random_state))
            ])
            
            X_balanced, y_balanced = pipeline.fit_resample(X, y)
            X_balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced_series = pd.Series(y_balanced)
            
            class_counts_after = y_balanced_series.value_counts().sort_index()
            
            print(f"\n   ✅ Combined Strategy Applied Successfully!")
            print(f"   After SMOTE + Under-sampling:")
            print(f"      Class 0: {class_counts_after[0]:6d} samples")
            print(f"      Class 1: {class_counts_after[1]:6d} samples")
            print(f"   New Imbalance Ratio: 1:{class_counts_after[0]/class_counts_after[1]:.2f}")
            
            return X_balanced_df, y_balanced_series
            
        except Exception as e:
            print(f"❌ Error in combined strategy: {e}")
            return X, y
    
    def visualize_imbalance(self, y_before: pd.Series, y_after: pd.Series,
                           title: str = "Class Distribution: Before vs After SMOTE",
                           save_path: Optional[str] = None):
        """
        Visualize class imbalance before and after SMOTE.
        
        Args:
            y_before: Original target distribution
            y_after: Post-SMOTE target distribution
            title: Plot title
            save_path: Optional path to save figure
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Before SMOTE
            counts_before = y_before.value_counts().sort_index()
            axes[0].bar(['Class 0\n(No Inhibitor)', 'Class 1\n(Inhibitor)'], 
                       counts_before.values, color=['skyblue', 'salmon'])
            axes[0].set_title('Before SMOTE', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Sample Count', fontsize=11)
            for i, v in enumerate(counts_before.values):
                axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
            
            # After SMOTE
            counts_after = y_after.value_counts().sort_index()
            axes[1].bar(['Class 0\n(No Inhibitor)', 'Class 1\n(Inhibitor)'],
                       counts_after.values, color=['lightgreen', 'lightcoral'])
            axes[1].set_title('After SMOTE', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Sample Count', fontsize=11)
            for i, v in enumerate(counts_after.values):
                axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')
            
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Visualization saved → {save_path}")
            else:
                plt.show()
            
            return fig
            
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")
            return None
    
    def get_stratified_folds(self, X: pd.DataFrame, y: pd.Series,
                            n_splits: int = 5) -> list:
        """
        Generate stratified k-folds for cross-validation.
        
        Stratified folding ensures each fold has similar class distribution
        to the original dataset - important for imbalanced data.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of folds
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        folds = []
        
        for train_idx, test_idx in skf.split(X, y):
            folds.append((train_idx, test_idx))
        
        print(f"\n📋 Generated {n_splits}-Fold Stratified Cross-Validation")
        print(f"   Each fold maintains class distribution proportional to original data")
        
        return folds


class BalancedTrainingStrategy:
    """
    Advanced training strategy with built-in SMOTE and class balancing.
    """
    
    def __init__(self, handler: ClassImbalanceHandler):
        """
        Initialize with imbalance handler.
        
        Args:
            handler: ClassImbalanceHandler instance
        """
        self.handler = handler
        self.X_train_balanced = None
        self.y_train_balanced = None
        
    def prepare_balanced_dataset(self, X_train: pd.DataFrame, y_train: pd.Series,
                                strategy: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare balanced training dataset.
        
        Args:
            X_train: Training features
            y_train: Training target
            strategy: 'smote', 'adasyn', or 'combined'
            
        Returns:
            Tuple of (X_balanced, y_balanced)
        """
        self.handler.analyze_imbalance(y_train, "Training Data (Before Balancing)")
        
        if strategy.lower() == 'smote':
            X_bal, y_bal = self.handler.apply_smote(X_train, y_train)
        elif strategy.lower() == 'adasyn':
            X_bal, y_bal = self.handler.apply_adasyn(X_train, y_train)
        elif strategy.lower() == 'combined':
            X_bal, y_bal = self.handler.apply_combined_strategy(X_train, y_train)
        else:
            X_bal, y_bal = X_train, y_train
        
        self.X_train_balanced = X_bal
        self.y_train_balanced = y_bal
        
        return X_bal, y_bal


if __name__ == "__main__":
    print("Class Imbalance Handler - Ready to use with your dataset")
    print("Supports SMOTE, ADASYN, and combined balancing strategies")
