#!/usr/bin/env python3
"""
Quick diagnostic for pickle file issues
No external dependencies required
"""

import os
import sys
from pathlib import Path

def check_pickle_files():
    """Check pickle file integrity without joblib"""
    print("🔍 Pickle File Diagnostic")
    print("=" * 50)
    
    pickle_files = {
        "columns.pkl": "Feature columns",
        "rf.pkl": "Random Forest model",
        "xgb.pkl": "XGBoost model"
    }
    
    found_files = {}
    missing_files = {}
    
    for filename, desc in pickle_files.items():
        if os.path.exists(filename):
            size_bytes = os.path.getsize(filename)
            size_mb = size_bytes / (1024 * 1024)
            found_files[filename] = (desc, size_mb, size_bytes)
            status = "✅ OK"
            if filename == "columns.pkl" and size_mb > 10:
                status = "⚠️  LARGE (possibly corrupted)"
            elif size_bytes == 0:
                status = "❌ EMPTY (corrupted)"
            print(f"{status}  {filename:<15} ({desc:<25}) {size_mb:.2f} MB")
        else:
            missing_files[filename] = desc
            print(f"❌ NOT FOUND  {filename:<15} ({desc:<25})")
    
    print("\n" + "=" * 50)
    
    # Check file integrity
    print("\n🔬 Integrity Check:")
    print("=" * 50)
    
    for filename in found_files:
        print(f"\nTesting {filename}...")
        try:
            with open(filename, 'rb') as f:
                # Read first few bytes to check pickle magic
                magic = f.read(3)
                if magic == b'\x80':  # Pickle magic bytes
                    print(f"  ✅ Valid pickle file (protocol 3+)")
                elif magic[:2] == b'BZh':  # Bzip2 compressed
                    print(f"  ℹ️  Compressed pickle file")
                else:
                    print(f"  ❌ Not a valid pickle file!")
                    print(f"     Magic bytes: {magic}")
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("=" * 50)
    print(f"✅ Found: {len(found_files)}/3 pickle files")
    print(f"❌ Missing: {len(missing_files)}/3 pickle files")
    
    if len(missing_files) > 0:
        print("\n🔧 Action Required:")
        print("Run: python train.py")
        print("This will regenerate all missing pickle files")
    
    print("\n" + "=" * 50)
    print("💾 System Memory:")
    print("=" * 50)
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        print(f"Memory Limit: {hard / (1024**3):.1f} GB (unlimited on Windows)")
    except:
        pass
    
    print("\n🎯 Next Steps:")
    if len(missing_files) > 0:
        print("1. Run: python train.py")
    else:
        print("1. Verify files look good from the sizes above")
    print("2. If columns.pkl is > 10 MB, delete it and retrain")
    print("3. Close other applications to free up RAM")
    print("4. Run: streamlit run app.py")

if __name__ == "__main__":
    check_pickle_files()
