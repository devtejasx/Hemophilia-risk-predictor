#!/usr/bin/env python3
"""
Repair corrupted pickle files by regenerating them from source data
Fixes MemoryError issues when loading trained models
"""

import os
import joblib
import pandas as pd
import pickle
from pathlib import Path

def repair_pickle_files():
    """
    Regenerate corrupted pickle files
    """
    print("🔧 Pickle File Repair Tool")
    print("=" * 50)
    
    # 1. Check if pickle files exist
    print("\n📁 Checking pickle files...")
    
    pickle_files = {
        "rf.pkl": "Random Forest model",
        "xgb.pkl": "XGBoost model",
        "columns.pkl": "Feature columns"
    }
    
    corrupted_files = []
    working_files = []
    
    for filename, description in pickle_files.items():
        if os.path.exists(filename):
            try:
                # Try to load the file
                data = joblib.load(filename, mmap_mode='r')
                print(f"✅ {filename:<15} ({description:<25}) - OK")
                working_files.append(filename)
            except (MemoryError, EOFError, pickle.UnpicklingError, OSError) as e:
                print(f"❌ {filename:<15} ({description:<25}) - CORRUPTED")
                print(f"   Error: {str(e)[:60]}")
                corrupted_files.append((filename, str(e)))
        else:
            print(f"⚠️  {filename:<15} ({description:<25}) - NOT FOUND")
    
    # 2. If files are corrupted, offer repair
    if corrupted_files:
        print("\n" + "=" * 50)
        print("🔨 Repair Options:")
        print("=" * 50)
        
        print("\n1. DELETE corrupted files (will be regenerated on next training)")
        print("2. RESTORE from backup (if available)")
        print("3. RETRAIN models (requires train data)")
        
        # Attempt automatic repair
        print("\n" + "=" * 50)
        print("🔄 Attempting automatic repair...")
        print("=" * 50)
        
        for filename, error in corrupted_files:
            print(f"\n🔧 Repairing {filename}...")
            
            try:
                # Delete corrupted file
                if os.path.exists(filename):
                    os.remove(filename)
                    print(f"  ✅ Deleted corrupted {filename}")
                
                # Regenerate if possible
                if filename == "columns.pkl":
                    # Check if we have training data
                    if os.path.exists("clinical.csv") and os.path.exists("genomic.csv"):
                        print(f"  📊 Loading training data...")
                        clinical = pd.read_csv("clinical.csv")
                        genomic = pd.read_csv("genomic.csv")
                        
                        # Merge data
                        df = pd.merge(clinical, genomic, on="patient_id")
                        df = pd.get_dummies(df, drop_first=True)
                        
                        # Save columns
                        columns = list(df.columns[:-1])  # Exclude target
                        joblib.dump(columns, filename)
                        print(f"  ✅ Regenerated {filename} with {len(columns)} columns")
                    else:
                        print(f"  ℹ️  {filename} - will be regenerated on next model training")
                
            except Exception as e:
                print(f"  ⚠️  Could not auto-repair: {str(e)[:60]}")
    
    # 3. Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    print(f"✅ Working files: {len(working_files)}")
    print(f"❌ Corrupted files: {len(corrupted_files)}")
    
    if len(corrupted_files) == 0:
        print("\n✨ All pickle files are healthy!")
        return True
    else:
        print("\n⚠️  Some files need attention.")
        print("\nNext steps:")
        print("1. Run `python train.py` to regenerate models and columns")
        print("2. Or increase available memory and restart Streamlit")
        return False

def clear_cache():
    """Clear Streamlit cache"""
    print("\n🗑️  Clearing Streamlit cache...")
    try:
        import shutil
        cache_dir = os.path.expanduser("~/.streamlit/logger")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("  ✅ Cache cleared")
    except Exception as e:
        print(f"  ⚠️  Could not clear cache: {e}")

def check_memory():
    """Check available system memory"""
    print("\n💾 System Memory Status:")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  Total: {memory.total / (1024**3):.1f} GB")
        print(f"  Used: {memory.used / (1024**3):.1f} GB")
        print(f"  Available: {memory.available / (1024**3):.1f} GB")
        print(f"  Percent: {memory.percent}%")
        
        if memory.percent > 80:
            print("  ⚠️  WARNING: System memory is running low!")
            print("     Try closing other applications")
    except ImportError:
        print("  ℹ️  Install psutil for detailed memory info: pip install psutil")

if __name__ == "__main__":
    # Check memory
    check_memory()
    
    # Repair pickle files
    success = repair_pickle_files()
    
    # Clear cache if needed
    if not success:
        clear_cache()
    
    print("\n" + "=" * 50)
    print("Done! 🎉")
    print("=" * 50)
