#!/usr/bin/env python3
"""
Optimized model training with memory efficiency
Fixes issues with large pickle files
"""

import pandas as pd
import joblib
import sys
import gc
from pathlib import Path

print("🚀 Optimized Model Training")
print("=" * 50)

def load_and_prepare_data():
    """Load and prepare data efficiently"""
    print("\n📊 Loading data...")
    
    try:
        # Load CSV files
        print("  - Loading genomic.csv...")
        genomic = pd.read_csv("genomic.csv", low_memory=False)
        print(f"    Rows: {len(genomic)}, Columns: {len(genomic.columns)}")
        
        print("  - Loading clinical.csv...")
        clinical = pd.read_csv("clinical.csv", low_memory=False)
        print(f"    Rows: {len(clinical)}, Columns: {len(clinical.columns)}")
        
        # Merge data
        print("  - Merging data...")
        df = pd.merge(genomic, clinical, on="patient_id", how="inner")
        print(f"    Merged: {len(df)} rows × {len(df.columns)} columns")
        
        # Check target variable
        print("\n🎯 Checking target variable...")
        print(f"  Unique values: {df['target'].unique()}")
        print(f"  Value counts:\n{df['target'].value_counts()}")
        
        # Keep only 0 and 1
        print("\n🔍 Cleaning data...")
        df_clean = df[df["target"].isin([0, 1])].copy()
        print(f"  After cleaning: {len(df_clean)} rows")
        print(f"  Value counts:\n{df_clean['target'].value_counts()}")
        
        # Separate target and features
        y = df_clean["target"]
        X = df_clean.drop(["target", "patient_id"], axis=1, errors="ignore")
        
        print(f"\n✅ Features: {X.shape}")
        print(f"   Target: {y.shape}")
        
        return X, y, df_clean
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: CSV file not found!")
        print(f"   {e}")
        print("\nPlease ensure both files exist:")
        print("  - genomic.csv")
        print("  - clinical.csv")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        sys.exit(1)

def optimize_dataframe(X):
    """Optimize DataFrame to reduce memory usage"""
    print("\n💾 Optimizing DataFrame...")
    
    # Convert object columns to category where possible
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')
    
    # Convert float64 to float32 where possible
    for col in X.select_dtypes(include=['float64']).columns:
        X[col] = X[col].astype('float32')
    
    # Convert int64 to int32 where possible
    for col in X.select_dtypes(include=['int64']).columns:
        if X[col].max() < 2147483647:  # int32 max
            X[col] = X[col].astype('int32')
    
    print(f"  Memory before: ~{df_clean.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    print(f"  New estimate: ~{X.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    return X

def train_models(X, y):
    """Train models with memory-efficient approach"""
    print("\n🤖 Training models...")
    
    try:
        # Import models
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        
        print("  - Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        rf.fit(X, y)
        print(f"    ✅ Complete")
        
        print("  - Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            random_state=42
        )
        xgb.fit(X, y)
        print(f"    ✅ Complete")
        
        return rf, xgb
        
    except ImportError as e:
        print(f"\n❌ Missing required package: {e}")
        print("   Run: pip install scikit-learn xgboost")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        sys.exit(1)

def save_models(rf, xgb, X):
    """Save models and columns efficiently"""
    print("\n💾 Saving models...")
    
    try:
        # Get columns list (stays small)
        columns = list(X.columns)
        print(f"  - Saving {len(columns)} column names...")
        joblib.dump(columns, "columns.pkl", compress=3)
        print(f"    ✅ columns.pkl saved")
        
        # Save RF model
        print(f"  - Saving Random Forest model...")
        joblib.dump(rf, "rf.pkl", compress=3)
        size_rf = Path("rf.pkl").stat().st_size / (1024**2)
        print(f"    ✅ rf.pkl saved ({size_rf:.2f} MB)")
        
        # Save XGB model
        print(f"  - Saving XGBoost model...")
        joblib.dump(xgb, "xgb.pkl", compress=3)
        size_xgb = Path("xgb.pkl").stat().st_size / (1024**2)
        print(f"    ✅ xgb.pkl saved ({size_xgb:.2f} MB)")
        
        # Verify files
        print("\n🔍 Verifying files...")
        for filename in ["columns.pkl", "rf.pkl", "xgb.pkl"]:
            if Path(filename).exists():
                size = Path(filename).stat().st_size / (1024**2)
                print(f"  ✅ {filename:<15} {size:>8.2f} MB")
            else:
                print(f"  ❌ {filename:<15} NOT FOUND")
        
    except Exception as e:
        print(f"\n❌ Error saving models: {e}")
        sys.exit(1)

def main():
    """Main training pipeline"""
    try:
        # Load data
        X, y, df_clean = load_and_prepare_data()
        
        # Encode categorical variables
        print("\n🔤 Encoding categorical variables...")
        X = pd.get_dummies(X, drop_first=True)
        print(f"   Features after encoding: {X.shape[1]}")
        
        # Optimize DataFrame
        # X = optimize_dataframe(X)  # Optional: reduces memory but may lose precision
        
        # Train models
        rf, xgb = train_models(X, y)
        
        # Save models
        save_models(rf, xgb, X)
        
        # Summary
        print("\n" + "=" * 50)
        print("✨ TRAINING COMPLETE!")
        print("=" * 50)
        print("\n🎯 Models ready to use:")
        print("  ✅ rf.pkl - Random Forest")
        print("  ✅ xgb.pkl - XGBoost")
        print("  ✅ columns.pkl - Feature columns")
        
        print("\n🚀 Next step:")
        print("   streamlit run app.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        gc.collect()

if __name__ == "__main__":
    global df_clean
    main()
