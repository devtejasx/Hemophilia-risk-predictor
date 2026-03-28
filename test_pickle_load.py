#!/usr/bin/env python3
"""
Test loading pickle files with proper memory handling
"""

import sys
import os

print("🔧 Testing Pickle File Loading")
print("=" * 50)

# Attempt to load with proper error handling
def test_load_pickle(filename):
    """Test loading a pickle file"""
    print(f"\nTesting: {filename}")
    
    try:
        # Try importing joblib
        import joblib
        print("  📦 joblib available")
        
        # Try loading with mmap (efficient memory usage)
        try:
            print("  - Attempting load with mmap_mode='r'...")
            data = joblib.load(filename, mmap_mode='r')
            print(f"    ✅ SUCCESS (mmap)")
            return True
        except (MemoryError, OSError) as e:
            print(f"    ⚠️  Failed with mmap: {str(e)[:50]}")
            
            # Try regular load
            print("  - Attempting regular load...")
            data = joblib.load(filename)
            print(f"    ✅ SUCCESS (regular)")
            return True
            
    except ImportError:
        print("  ❌ joblib not available")
        return False
    except MemoryError as e:
        print(f"  ❌ MEMORY ERROR: {e}")
        return False
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)[:100]}")
        return False

# Test all pickle files
print("\n" + "=" * 50)
print("SYSTEM STATUS:")
print("=" * 50)

try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM Available: {mem.available / (1024**3):.1f} GB")
    print(f"RAM Used: {mem.percent}%")
except ImportError:
    print("(memory info not available)")

print("\n" + "=" * 50)
print("LOADING TEST:")
print("=" * 50)

results = {}
for filename in ["columns.pkl", "rf.pkl", "xgb.pkl"]:
    if os.path.exists(filename):
        results[filename] = test_load_pickle(filename)
    else:
        print(f"\n{filename}: ❌ FILE NOT FOUND")
        results[filename] = False

# Summary
print("\n" + "=" * 50)
print("SUMMARY:")
print("=" * 50)

success_count = sum(1 for v in results.values() if v)
total_count = len(results)

for filename, success in results.items():
    status = "✅" if success else "❌"
    print(f"{status} {filename}")

print(f"\n{success_count}/{total_count} files loaded successfully")

if success_count == 3:
    print("\n✨ All pickle files are working!")
    print("🎯 Ready to run Streamlit:")
    print("   streamlit run app.py")
else:
    print("\n⚠️  Some files failed to load")
    print("\n🔧 FIX OPTIONS:")
    print("1. Close other applications to free RAM")
    print("2. Run: python train.py (regenerates files)")
    print("3. Restart Python/Streamlit")

print("\n" + "=" * 50)
