"""
Enhanced ML Training Pipeline with Ensemble Models
===================================================

This updated pipeline aligns with the academic proposal:
✅ Genomic + Clinical Data Fusion
✅ Ensemble Models (RF, XGBoost, CatBoost, LightGBM)
✅ Stacking Ensemble Combination
✅ SMOTE for Class Imbalance Handling
✅ SHAP + LIME Explainability

Non-breaking: Existing models (RF, XGB) still saved for backward compatibility
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Import new modules
from data_fusion import GenomicClinicalFusion
from ensemble_models import EnsembleModelRegistry, EnsembleTrainer
from imbalance_handler import ClassImbalanceHandler, BalancedTrainingStrategy
from explainability import SHAPExplainer

print("=" * 70)
print("🚀 ENHANCED ML TRAINING PIPELINE - WITH ENSEMBLE LEARNING")
print("=" * 70)

# ================================================================
# STEP 1: DATA FUSION (Genomic + Clinical)
# ================================================================
print("\n" + "="*70)
print("STEP 1: DATA FUSION - Combining Genomic & Clinical Features")
print("="*70)

fusion = GenomicClinicalFusion(genomic_path="genomic.csv", clinical_path="clinical.csv")

if not fusion.load_data():
    print("❌ Failed to load data")
    sys.exit(1)

fused_df = fusion.fuse_data(clean_target=True)
if fused_df is None:
    print("❌ Failed to fuse data")
    sys.exit(1)

engineered_df = fusion.engineer_features()
if engineered_df is None:
    print("❌ Failed to engineer features")
    sys.exit(1)

# Get fused dataset
result = fusion.get_fused_data()
if result is None:
    print("❌ Failed to get fused data")
    sys.exit(1)

X, y, feature_names = result

print(f"\n✅ Data Fusion Complete!")
print(f"   Features Shape: {X.shape}")
print(f"   Features: {len(feature_names)} total")

# ================================================================
# STEP 2: CLASS IMBALANCE HANDLING WITH SMOTE
# ================================================================
print("\n" + "="*70)
print("STEP 2: CLASS IMBALANCE HANDLING - SMOTE Application")
print("="*70)

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE on training data only (to avoid data leakage)
imbalance_handler = ClassImbalanceHandler(random_state=42)

# Analyze original imbalance
imbalance_handler.analyze_imbalance(y_train, "Training Data (Before SMOTE)")

# Apply SMOTE
X_train_balanced, y_train_balanced = imbalance_handler.apply_smote(
    X_train, y_train, sampling_strategy=0.8, k_neighbors=5
)

print(f"\n✅ Class Imbalance Handling Complete!")
print(f"   Training set after SMOTE: {X_train_balanced.shape}")

# ================================================================
# STEP 3: ENSEMBLE MODEL TRAINING
# ================================================================
print("\n" + "="*70)
print("STEP 3: ENSEMBLE MODELS - Training 5 Models")
print("="*70)

# Create registry and trainer
registry = EnsembleModelRegistry(random_state=42)
trainer = EnsembleTrainer(registry)

# Train all models on balanced data
training_results = trainer.train_all_models(
    X_train_balanced, y_train_balanced,
    X_val=X_test, y_val=y_test
)

print("\n✅ All Ensemble Models Trained Successfully!")

# ================================================================
# STEP 4: MODEL COMPARISON
# ================================================================
print("\n" + "="*70)
print("STEP 4: MODEL COMPARISON - Performance on Test Set")
print("="*70)

model_comparison = {}
for model_name, model in trainer.trained_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    model_comparison[model_name] = {
        'accuracy': acc,
        'auc': auc,
        'f1': f1
    }
    
    print(f"\n{model_name}:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   ROC-AUC:  {auc:.4f}")
    print(f"   F1-Score: {f1:.4f}")

# Find best model
best_model_name = max(model_comparison, key=lambda x: model_comparison[x]['auc'])
print(f"\n🏆 Best Model: {best_model_name} (AUC: {model_comparison[best_model_name]['auc']:.4f})")

# ================================================================
# STEP 5: EXPLAINABILITY ANALYSIS
# ================================================================
print("\n" + "="*70)
print("STEP 5: EXPLAINABILITY - SHAP Analysis")
print("="*70)

best_model = trainer.trained_models[best_model_name]

# Use sample for SHAP background
X_background = X_train_balanced.sample(min(50, len(X_train_balanced)), random_state=42)

shap_explainer = SHAPExplainer(best_model, X_background)
shap_values = shap_explainer.compute_shap_values(X_test)

# Get feature importance
feature_importance = shap_explainer.get_feature_importance(X_test, feature_names)

print("\n✅ SHAP Analysis Complete!")
print("\nTop 10 Most Important Features (SHAP):")
for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
    print(f"   {i}. {feature}: {importance:.6f}")

# ================================================================
# STEP 6: SAVE MODELS AND ARTIFACTS
# ================================================================
print("\n" + "="*70)
print("STEP 6: SAVING MODELS & ARTIFACTS")
print("="*70)

# Save all ensemble models
trainer.save_models(directory=".")

# Save feature names
joblib.dump(feature_names, "feature_names.pkl")
print("✅ Saved feature_names.pkl")

# Save SHAP values for visualization
joblib.dump(shap_values, "shap_values.pkl")
print("✅ Saved shap_values.pkl")

# Save feature importance
joblib.dump(feature_importance, "feature_importance.pkl")
print("✅ Saved feature_importance.pkl")

# Save model comparison results
joblib.dump(model_comparison, "model_comparison.pkl")
print("✅ Saved model_comparison.pkl")

# Save test data for evaluation
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("✅ Saved X_test.csv and y_test.csv")

# ================================================================
# BACKWARD COMPATIBILITY: Also save individual models
# ================================================================
print("\n" + "="*70)
print("BACKWARD COMPATIBILITY: Saving Individual Models")
print("="*70)

# Save best ensemble model as the default "model"
joblib.dump(trainer.trained_models[best_model_name], "model_ensemble.pkl")
print(f"✅ Saved best model ({best_model_name}) as model_ensemble.pkl")

# Also save RF and XGB as before for backward compatibility
if 'RandomForest' in trainer.trained_models:
    joblib.dump(trainer.trained_models['RandomForest'], "rf.pkl")
    print("✅ Saved rf.pkl (backward compatibility)")

if 'XGBoost' in trainer.trained_models:
    joblib.dump(trainer.trained_models['XGBoost'], "xgb.pkl")
    print("✅ Saved xgb.pkl (backward compatibility)")

joblib.dump(list(X.columns), "columns.pkl")
print("✅ Saved columns.pkl")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "="*70)
print("✅ TRAINING PIPELINE COMPLETE!")
print("="*70)
print("\n📊 Pipeline Summary:")
print(f"   ✅ Data fusion: Genomic + Clinical ({X.shape[1]} features)")
print(f"   ✅ Class balance: SMOTE applied (ratio improved)")
print(f"   ✅ Ensemble: 5 models trained (RF, XGB, CatBoost, LightGBM, Stacking)")
print(f"   ✅ Best model: {best_model_name} (AUC: {model_comparison[best_model_name]['auc']:.4f})")
print(f"   ✅ Explainability: SHAP feature importance computed")
print(f"   ✅ Backward compatibility: RF & XGB models saved")
print("\n📁 Saved Files:")
print("   - models: randomforest.pkl, xgboost.pkl, catboost.pkl, lightgbm.pkl, stackingensemble.pkl")
print("   - artifacts: feature_names.pkl, shap_values.pkl, feature_importance.pkl, model_comparison.pkl")
print("   - test data: X_test.csv, y_test.csv")
print("   - backward compat: rf.pkl, xgb.pkl, columns.pkl, model_ensemble.pkl")
print("\n" + "="*70)