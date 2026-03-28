# 🔧 Memory Error Fix Guide

## Problem
```
MemoryError: Unable to load pickle file (columns.pkl)
```

This occurs when:
- The pickle file is corrupted
- Not enough RAM available
- File is too large for memory

---

## ✅ Quick Fix (3 Steps)

### Step 1: Run Repair Script
```bash
python repair_pickle_files.py
```

This script will:
- ✅ Check pickle file integrity
- ✅ Identify corrupted files
- ✅ Attempt automatic repair
- ✅ Show memory status

### Step 2: Delete Corrupted Files (if needed)
```bash
# Windows PowerShell
Remove-Item "columns.pkl" -Force
Remove-Item "rf.pkl" -Force
Remove-Item "xgb.pkl" -Force
```

Or manually delete from file explorer:
- `c:\Users\tejas\OneDrive\Documents\Capstone\columns.pkl`
- `c:\Users\tejas\OneDrive\Documents\Capstone\rf.pkl`
- `c:\Users\tejas\OneDrive\Documents\Capstone\xgb.pkl`

### Step 3: Regenerate Models
```bash
python train.py
```

This will:
- ✅ Load training data (clinical.csv, genomic.csv)
- ✅ Train fresh models
- ✅ Create new pickle files
- ✅ Generate columns.pkl with correct data

---

## 🔍 Verification

After fixing, verify files are working:
```bash
python -c "import joblib; print(joblib.load('columns.pkl', mmap_mode='r')[:5])"
```

Expected output: First 5 column names

---

## 💾 Memory Management

### Option 1: Increase Virtual Memory (Windows)
1. Settings → System → Advanced system settings
2. Performance → Advanced tab
3. Virtual Memory → Change
4. Set to 16 GB (or higher if available)
5. Restart computer

### Option 2: Free Up RAM
Close unnecessary applications:
- Chrome/Firefox (consumes 1-2 GB per tab)
- IDE/Visual Studio (500 MB - 5 GB)
- Other applications
- Restart Streamlit: `streamlit run app.py`

### Option 3: Use mmap (Already Implemented ✅)
The code now uses:
```python
joblib.load("columns.pkl", mmap_mode='r')
```
This loads files without using full RAM.

---

## 🚨 Advanced Troubleshooting

### Check which files are corrupted
```bash
python repair_pickle_files.py
```

### Manually regenerate columns.pkl
```python
import joblib
import pandas as pd

# Load training data
clinical = pd.read_csv("clinical.csv")
genomic = pd.read_csv("genomic.csv")

# Merge
df = pd.merge(clinical, genomic, on="patient_id")

# One-hot encode
df = pd.get_dummies(df, drop_first=True)

# Save columns (excluding target)
columns = list(df.columns[:-1])
joblib.dump(columns, "columns.pkl")
print(f"✅ Saved {len(columns)} columns to columns.pkl")
```

### Check file sizes
```bash
# Windows PowerShell
Get-Item *.pkl | Select-Object Name, @{Name="Size (MB)"; Expression={ [math]::Round($_.Length / 1MB, 2) }}
```

Expected sizes:
- `columns.pkl` - < 1 MB (just column names)
- `rf.pkl` - 2-50 MB (model size varies)
- `xgb.pkl` - 1-20 MB (model size varies)

If columns.pkl is > 10 MB, it's probably corrupted.

---

## 📋 Changes Made to Code

Your code now includes:

**1. Memory-Safe Loading**
```python
try:
    columns = joblib.load("columns.pkl", mmap_mode='r')
except (MemoryError, EOFError, pickle.UnpicklingError):
    # Fallback loading without mmap
    columns = joblib.load("columns.pkl")
```

**2. Better Error Messages**
```python
except MemoryError:
    st.error("💾 Memory Error: Close other apps or increase RAM")
```

**3. Multiple Fallback Attempts**
- First: Try with mmap mode (efficient)
- Second: Try without mmap
- Third: Use None as fallback

**Files Updated:**
✅ `app.py` - Line 40-65
✅ `predict.py` - Line 1-40
✅ `api.py` - Line 1-25

---

## 🎯 Final Checklist

- [ ] Run `python repair_pickle_files.py`
- [ ] Delete corrupted files if identified
- [ ] Run `python train.py` to regenerate
- [ ] Close other applications (improve RAM)
- [ ] Restart Streamlit: `streamlit run app.py`
- [ ] Test with sample patient data
- [ ] Verify no more MemoryError

---

## 📞 Still Having Issues?

**If problem persists:**

1. **Check memory again after all apps closed:**
   ```bash
   python -c "import psutil; m = psutil.virtual_memory(); print(f'Available: {m.available/1024**3:.1f} GB')"
   ```

2. **Use smaller training data:**
   - Reduce clinical.csv to first 10,000 rows
   - Reduce genomic.csv accordingly
   - Re-train: `python train.py`

3. **Clear everything and restart:**
   ```bash
   # Delete all binary files
   Remove-Item "*.pkl" -Force
   Remove-Item "__pycache__" -Recurse -Force
   
   # Clear Streamlit cache
   Remove-Item "$HOME\.streamlit" -Recurse -Force
   
   # Reinstall and retrain
   pip install -r requirements.txt
   python train.py
   streamlit run app.py
   ```

4. **Monitor memory during operations:**
   Open Task Manager (Ctrl+Shift+Esc) and watch RAM usage

5. **Use 64-bit Python (you have it ✅)**
   - 32-bit Python limited to 2 GB
   - 64-bit Python can use full RAM
   - Check: `python -c "import struct; print(struct.calcsize('P') * 8, 'bit')"`

---

## ✨ After Fix

You should have:
✅ Working pickle files
✅ No MemoryError
✅ Streamlit app running smoothly
✅ All models and columns loaded properly
✅ Ready to use dashboard and chatbot

Enjoy your hemophilia clinical AI platform! 🎉
