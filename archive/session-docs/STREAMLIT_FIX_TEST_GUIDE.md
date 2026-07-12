# 🧪 QUICK TEST GUIDE - Verify the Fix

Run these tests to confirm the Streamlit state management bug is fixed.

---

## ✅ Test 1: Basic Evaluation & Display

**Steps:**
1. Run the Streamlit app: `streamlit run app.py`
2. Navigate to "🧪 Evaluation" page
3. Click "🔄 Load Data & Evaluate Models" button
4. Wait for success message

**Expected Results:**
- ✅ See "✅ Evaluation complete!" message
- ✅ See metrics immediately (Accuracy, Precision, Recall, etc.)
- ✅ See "📋 Summary Table" with model results
- ✅ DO NOT see "Evaluate models first" message

**If BROKEN:**
- ❌ Shows "👈 Click button above to load data" after evaluation
- ❌ Metrics don't display
- ❌ Page shows placeholder message

---

## ✅ Test 2: Tab Persistence

**Steps:**
1. Evaluate models (see Test 1)
2. Click "📈 Visualizations" tab
3. Click "Generate Confusion Matrix"
4. Switch to "📋 Reports" tab
5. Switch to "🔍 Details" tab
6. Switch back to "📊 Metrics" tab

**Expected Results:**
- ✅ Results persist across all tabs
- ✅ Metrics still visible on Metrics tab
- ✅ No "Evaluate models first" message
- ✅ Visualizations don't disappear

---

## ✅ Test 3: Debug Mode

**Steps:**
1. Evaluate models (see Test 1)
2. Click "🔍 Debug" button
3. Expand "🐛 Debug Information" expander

**Expected Results:**
- ✅ Shows: "Evaluation results in session state: True"
- ✅ Shows: "Results keys: ['Random Forest', 'XGBoost']"
- ✅ Shows: Timestamp of evaluation

**If BROKEN:**
- ❌ Shows "False" for results in session state
- ❌ No keys displayed
- ❌ Empty debug info

---

## ✅ Test 4: Clear & Reset

**Steps:**
1. Evaluate models (see Test 1)
2. Verify metrics are showing ✅
3. Click "🗑️ Clear" button
4. Wait for page rerun

**Expected Results:**
- ✅ Shows "Cleared evaluation results" message
- ✅ Page reruns
- ✅ Dashboard shows "👈 Click button to evaluate" message
- ✅ Metrics disappear

---

## ✅ Test 5: Report Generation

**Steps:**
1. Evaluate models (see Test 1)
2. Go to "📋 Reports" tab
3. Click "📄 Generate JSON Report"

**Expected Results:**
- ✅ Shows spinner during generation
- ✅ Shows "⬇️ Download JSON Report" button
- ✅ Shows "✅ Report generated!" message
- ✅ Can download file successfully

**If BROKEN:**
- ❌ Nothing happens
- ❌ Shows "Evaluate models first"
- ❌ Error messages

---

## ✅ Test 6: Confusion Matrix Visualization

**Steps:**
1. Evaluate models (see Test 1)
2. Go to "📈 Visualizations" tab
3. Keep model selected as "Random Forest"
4. Click "Generate Confusion Matrix"
5. Wait for success

**Expected Results:**
- ✅ Shows spinner while generating
- ✅ Displays confusion matrix image
- ✅ Shows "⬇️ Download Confusion Matrix" button
- ✅ No errors

---

## ✅ Test 7: ROC Curves

**Steps:**
1. Evaluate models (see Test 1)
2. Go to "📈 Visualizations" tab
3. Click "Generate ROC Curves"

**Expected Results:**
- ✅ Shows spinner while generating
- ✅ Displays ROC curves plot
- ✅ Shows "⬇️ Download ROC Curves" button
- ✅ Compares all models in one chart

---

## ✅ Test 8: Metrics Comparison

**Steps:**
1. Evaluate models (see Test 1)
2. Go to "📈 Visualizations" tab
3. Click "Generate Metrics Comparison Chart"

**Expected Results:**
- ✅ Shows spinner while generating
- ✅ Displays comparison bar chart
- ✅ Shows all metrics for all models
- ✅ Can download PNG

---

## ✅ Test 9: Detailed Analysis

**Steps:**
1. Evaluate models (see Test 1)
2. Go to "🔍 Details" tab
3. Expand first model details

**Expected Results:**
- ✅ Shows expander with model name
- ✅ Displays performance metrics
- ✅ Shows confusion matrix values
- ✅ Shows classification report table

---

## ✅ Test 10: Multiple Model Comparison

**Steps:**
1. Evaluate models (see Test 1)
2. Go to "📊 Metrics" tab
3. Check metrics for both "Random Forest" and "XGBoost"
4. Go to "🔍 Details" tab
5. Expand both models

**Expected Results:**
- ✅ Two separate model sections
- ✅ Different metrics for each model
- ✅ Can compare performance
- ✅ Both models visible

---

## 🔴 Common Issues & Fixes

### Issue: "Evaluate models first" always shows

**Diagnosis:**
```python
# Check if files exist
import os
print(os.path.exists("rf.pkl"))     # Should be True
print(os.path.exists("xgb.pkl"))    # Should be True
print(os.path.exists("genomic.csv")) # Should be True
print(os.path.exists("clinical.csv")) # Should be True
```

**Fix:** Make sure all required files are in the working directory

### Issue: Metrics don't display after clicking button

**Diagnosis:** Old code might still be present

**Fix:** Verify these changes are in app.py:
```python
# Check line ~2896
if "evaluation_results" not in st.session_state:
    st.session_state["evaluation_results"] = None

# Check line ~2940 (should NOT have st.rerun())
st.session_state["evaluation_results"] = {...}
st.success("✅ Evaluation complete!")
# No st.rerun() here!
```

### Issue: Debug mode shows False but results exist

**Possible causes:**
- Session expired (refresh page, all data lost)
- Multiple Streamlit instances running
- Browser cache issues

**Fix:**
```python
# Clear browser cache
# Or use private/incognito mode
# Or restart Streamlit: Ctrl+C in terminal, then streamlit run app.py
```

---

## 📊 Expected Metrics Values

When evaluation runs successfully, you should see approximately:

```
Random Forest:
- Accuracy: 0.7000 - 0.9500 (70-95%)
- Precision: 0.6000 - 0.9000
- Recall: 0.6000 - 0.9000
- F1-Score: 0.6000 - 0.9000
- ROC-AUC: 0.7000 - 0.9500

XGBoost:
- Accuracy: 0.7000 - 0.9500
- Precision: 0.6000 - 0.9000
- Recall: 0.6000 - 0.9000
- F1-Score: 0.6000 - 0.9000
- ROC-AUC: 0.7000 - 0.9500
```

If metrics are all 0 or 1.0, something is wrong with data loading.

---

## 🎯 Quick Checklist

Run through these quickly:

- [ ] Click evaluation button
- [ ] See metrics immediately ✓
- [ ] Switch tabs (metrics persist) ✓
- [ ] Click "Debug" - shows True ✓
- [ ] Click "Clear" - results disappear ✓
- [ ] Generate confusion matrix ✓
- [ ] Generate ROC curves ✓
- [ ] Download reports ✓
- [ ] See detailed analysis ✓
- [ ] All tabs functional ✓

**If all checkmarks**: ✅ FIX IS WORKING!

---

## 📝 What to Do If Tests Fail

1. **Check Python/Package Versions**
   ```bash
   pip list | grep streamlit
   pip list | grep scikit
   python --version
   ```

2. **Clear Cache**
   ```bash
   # Delete .streamlit/cache
   rm -rf .streamlit/cache/  # Mac/Linux
   rmdir /s .streamlit\cache  # Windows
   ```

3. **Restart Everything**
   ```bash
   # Kill Streamlit
   Ctrl+C
   
   # Wait 5 seconds, then restart
   streamlit run app.py
   ```

4. **Check Logs**
   ```python
   # Enable Streamlit logging
   export STREAMLIT_LOGGER_LEVEL=debug  # Mac/Linux
   set STREAMLIT_LOGGER_LEVEL=debug     # Windows
   streamlit run app.py
   ```

5. **Verify Session State**
   ```python
   # Add this to debug section
   st.write(f"Type: {type(st.session_state.get('evaluation_results'))}")
   st.write(f"Keys in results: {list(st.session_state.get('evaluation_results', {}).get('results', {}).keys())}")
   ```

---

## ✅ Final Verification

After all tests pass:

1. Close the app (Ctrl+C)
2. Reopen: `streamlit run app.py`
3. Run evaluation again
4. Everything should work ✅

**You're done!** The fix is confirmed working.

---

**Test Duration**: ~15 minutes  
**Success Criteria**: All 10 tests pass  
**Status**: Ready for production use after confirmation
