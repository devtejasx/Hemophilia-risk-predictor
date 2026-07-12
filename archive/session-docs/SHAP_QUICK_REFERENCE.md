# SHAP Explainability - Quick Reference Card

## At a Glance

**What**: SHAP provides model transparency by showing how each feature contributes to predictions  
**Why**: Enables clinical validation and decision support confidence  
**Where**: Integrated into "Prediction" page in 5 visualization tabs  
**How**: Uses TreeExplainer for fast, accurate explainability  

---

## Quick Start (2 Minutes)

### Step 1: Make a Prediction
Navigate to **"Patient Form"** → Enter patient data → Get prediction

### Step 2: View SHAP Analysis  
Scroll to **"SHAP Model Explainability Analysis"** section → Choose tab

### Step 3: Interpret Results
- **📊 Summary**: What features matter most?
- **⛲ Waterfall**: How did we get this risk score?
- **⚡ Force**: What increases vs decreases risk?
- **📈 Importance**: Ranking of all features
- **📋 Interpretation**: Simple English explanation

---

## The 5 SHAP Tabs Explained

### 📊 Summary Tab
```
Shows: Top 10 most important features (bar chart)
Read: Longer bars = more important
Use: Quick understanding of key risk factors
Clinical: "Which features matter most in this population?"
```

### ⛲ Waterfall Tab  
```
Shows: Step-by-step how prediction was built
Read: Blue = increases risk, Red = decreases risk
Use: Explain why THIS patient has THIS risk score
Clinical: "Here's why we recommend [intervention]"
```

### ⚡ Force Tab
```
Shows: Risk-increasing (left) vs risk-decreasing (right)
Read: Split visualization showing opposing forces
Use: Quick visual balance of risks vs protections
Clinical: "Do protective factors outweigh risks?"
```

### 📈 Importance Tab
```
Shows: Global feature ranking table
Read: All features scored by average impact
Use: Population-level view of feature importance  
Clinical: "What features drive risk across all patients?"
```

### 📋 Interpretation Tab
```
Shows: Simple language explanation
Read: Risk level, key factors, clinical assessment
Use: Non-technical readout for stakeholders
Clinical: "What do the ML results actually mean?"
```

---

## Color Coding Guide

```
🔵 BLUE   = Increases Risk (concerning)
🔴 RED    = Decreases Risk (protective)
🟢 GREEN  = Low Risk
🟡 YELLOW = Moderate Risk
🟠 ORANGE = High Risk
🔴 RED    = Critical Risk
```

---

## Interpretation Cheat Sheet

### Reading Waterfall Plot
```
Base Value: 0.35 (starting point)
  +Feature A: +0.20  ↑ Risk UP (bad)
  +Feature B: +0.15  ↑ Risk UP (bad)
  -Feature C: -0.05  ↓ Risk DOWN (good)
= Final: 0.65 (65% risk) = HIGH RISK
```

**Translation**: Multiple risk factors accumulate to create HIGH risk

### Reading Force Plot
```
Left (Risk ↑):  0.50  |||||||||
Right (Risk ↓): 0.10  ||

Balance: Heavily skewed left = AGGRESSIVE MANAGEMENT
```

**Translation**: Risk factors far outweigh protective factors

### Reading Feature Importance
```
1. exposure_days          0.087  ████████░  (Most important)
2. age_first_treatment    0.065  ██████░
3. severity               0.058  █████░
```

**Translation**: Exposure history matters most, then age, then severity

---

## Clinical Decision Reference

### Risk Score Interpretation
```
< 0.3  = 🟢 LOW       → Standard care
0.3-0.4 = 🟡 MODERATE → Monitor closely
0.4-0.6 = 🟠 ELEVATED → Consider intervention
> 0.6  = 🔴 HIGH      → Active management
```

### Feature Impact Magnitude
```
|SHAP| > 0.20  = Major impact (deal-breaker)
|SHAP| 0.10-0.20 = Moderate impact (important)
|SHAP| < 0.10  = Minor impact (consider)
```

### Clinical Actions by SHAP Pattern
```
Pattern: All positive (blue) SHAP values
→ Action: Aggressive intervention
→ Reason: Multiple compounding risk factors

Pattern: Mixed (both blue and red)
→ Action: Standard management
→ Reason: Risk and protective factors balance

Pattern: Many negative (red) SHAP values
→ Action: Standard care + reassurance
→ Reason: Strong protective factors present
```

---

## Top 10 SHAP Tips for Clinicians

1. **Use SHAP, don't trust SHAP blindly**
   - Always validate against clinical judgment
   - SHAP explains the model, not necessarily reality

2. **Focus on top 3 features**
   - Summary plot shows them clearly
   - Often 80% of prediction variance

3. **Waterfall for individual patients**
   - Best for explaining risk to patients/families
   - Shows step-by-step reasoning

4. **Force plot reveals balance**
   - Imbalanced = aggressive management
   - Balanced = standard protocols

5. **Dependence plot shows relationships**
   - Understand how features affect predictions
   - Validate non-linear effects

6. **Compare across patients**
   - Is Patient A's risk profile similar to Patient B?
   - Use SHAP for patient stratification

7. **Document SHAP rationale**
   - Include in clinical notes
   - Supports decision-making audit trail

8. **Train your team on interpretation**
   - Not all clinicians are familiar with SHAP
   - Use simple language explanations

9. **Combine with EHR data**
   - SHAP explains model, EHR provides context
   - Together they build strong clinical case

10. **Challenge suspicious patterns**
    - If SHAP doesn't make clinical sense, investigate
    - May indicate data quality or model issues

---

## Common Questions Answered

**Q: Why does this patient have high risk when they look fine clinically?**
```
A: SHAP shows statistical associations. The model is trained on population data.
Individual outliers exist. Request clinical team review before acting on model.
```

**Q: Can I use SHAP to explain why patient developed inhibitors?**
```
A: No. SHAP explains predictive associations, not causation.
It shows which factors correlate with risk, not what caused inhibitor development.
```

**Q: What if SHAP conflicts with clinical judgment?**
```
A: Investigate. This could mean:
- Model has data quality issue
- Clinical situation is unusual (outlier)
- Additional clinical factors not in model
→ Always defer to clinical judgment
```

**Q: Is the model prediction correct?**
```
A: SHAP explains the prediction but doesn't validate it.
To validate:
- Compare predictions to actual patient outcomes
- Get clinical team validation
- Perform prospective assessment
```

**Q: Which features should we focus on?**
```
A: Top 3 features in summary plot typically drive 50-70% of prediction variation.
Start there, then add other features as resources allow.
```

---

## Workflow Integration

### In Clinical Decision-Making
```
1. Patient presentation
2. Enter data into prediction form
3. Get risk prediction
4. Review 5 SHAP tabs for explanation
5. Share interpretation with team
6. Clinical team validates/challenges SHAP findings
7. Make clinical decision incorporating SHAP insights
8. Document SHAP analysis in clinical notes
```

### In Multidisciplinary Team Rounds
```
"Let me show you the SHAP analysis for this patient...

Summary tab shows the top risk factors are [X], [Y], [Z].

Looking at the waterfall, here's how we got to the 68% risk score...

The force plot shows the risk factors outweigh protective factors,
so I recommend we [clinical action].

Let's discuss if this aligns with your clinical assessment..."
```

### In Patient/Family Counseling
```
"Based on our AI analysis, here's your risk:

The model identified these key factors increasing your risk:
- [Factor 1]: [explanation]
- [Factor 2]: [explanation]

These factors help protect you:
- [Factor 1]: [explanation]

This is why we recommend [intervention]."
```

---

## Troubleshooting

**SHAP not showing?**
```
❌ Problem: Visualizations don't appear
✅ Fix: 
   - Reload browser
   - Clear Streamlit cache: streamlit cache clear
   - Verify models are loaded
```

**Results don't make sense?**
```
❌ Problem: SHAP values are all same/too high
✅ Fix:
   - Verify feature scaling
   - Check model training data
   - Compare against validation set
```

**Slow performance?**
```
❌ Problem: Visualization takes >10 seconds
✅ Fix:
   - Reduce top_features parameter
   - Use subset of data
   - Check system resources
```

---

## For Developers: Integration Code

### Minimal Integration (Copy-Paste Ready)
```python
from shap_explainability import SHAPExplainer, SHAPVisualizer

# Load model
explainer = SHAPExplainer(model, feature_names)

# Get explanation
explanation = explainer.explain_prediction(X)

# Visualize
fig = SHAPVisualizer.plot_summary(explanation)
st.pyplot(fig)
```

### Full Dashboard
```python
from shap_explainability import display_shap_dashboard

display_shap_dashboard(explanation, model_type="Random Forest")
```

---

## Key Statistics

- **TreeExplainer Speed**: O(n_features) per prediction = FAST
- **Memory Overhead**: ~2-3x model size for SHAP values
- **Accuracy**: Optimal for tree-based models (RF, XGBoost)
- **Computation**: Cached to prevent recalculation

---

## Resources

📚 **Documentation Files**:
- `SHAP_INTEGRATION_COMPLETE.md` - Full technical guide
- `SHAP_VISUALIZATION_CLINICAL_GUIDE.md` - Detailed interpretation guide
- `shap_explainability.py` - Source code with docstrings

🔗 **External**:
- SHAP Library: https://shap.readthedocs.io/
- TreeExplainer: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html

---

## Version Info
- **Created**: April 7, 2026
- **Status**: Production Ready ✅
- **Updated**: Last comprehensive revision
- **Tested**: All 5 visualization types verified