# SHAP Visualizations - Clinical Interpretation Guide

## Complete Reference for SHAP Plots

This guide explains each SHAP visualization type, how to interpret it, and what clinical insights it provides.

---

## 1. Summary Plot 📊

### What It Is
A bar chart showing the average "importance" of each feature across all predictions.

### How to Read It
```
Feature Name        Importance Score
────────────────────────────────────
exposure_days            0.087      ████████░
age_first_treatment      0.065      ██████░
severity_severe          0.058      █████░
mutation_type_nonsense   0.045      ████░
baseline_factor_level    0.038      ███░
```

### What It Shows
- **Longer bar** = Feature impacts predictions more consistently
- **Order** = Top to bottom ranking by importance
- **Scale** = Mean absolute SHAP value (0-max)

### Clinical Interpretation

**Example: Exposure Days (0.087)**
- Patients with different exposure days (cumulative treatment days) show larger differences in inhibitor risk
- This is the **single most important factor** in model predictions
- Implication: Exposure history is crucial for risk assessment

**Example: Severity (0.058)**
- Hemophilia severity affects risk predictions moderately
- But less than exposure days
- Implication: Exposure accumulation matters more than baseline severity

### Clinical Use Cases

1. **Risk Factor Prioritization**
   - Focus monitoring on top 3 features
   - Allocate resources to high-impact factors

2. **Patient Stratification**
   - Group patients by presence of top features
   - Design targeted interventions

3. **Model Validation**
   - Confirm rankings match clinical knowledge
   - Identify surprising feature importances

### Optimal Interpretation Range
```
✅ Makes Clinical Sense:
   - Exposure accumulated → High importance
   - Age at exposure → Moderate importance
   - Mutation type → Variable importance

⚠️ Suspicious Results:
   - Age/date fields → High importance (overfitting?)
   - Random features → Non-zero importance (noise?)
   - Missing clinical features → Lower importance
```

---

## 2. Waterfall Plot (Force-Oriented) ⛲

### What It Is
A step-by-step breakdown of how each feature pushes the prediction up or down from the baseline.

### How to Read It
```
Baseline: 0.35
  ↓
exposure_days=150     +0.18  → 0.53 (INCREASES RISK)
  ↓
severity_severe       +0.12  → 0.65 (INCREASES RISK)
  ↓
baseline_factor=40%   -0.08  → 0.57 (DECREASES RISK)
  ↓
Final Prediction: 0.65 (65% risk)
```

### What It Shows
- **Blue bars** (positive): Features increasing risk (push right)
- **Red bars** (negative): Features decreasing risk (push left)
- **Height** of each bar: Magnitude of impact
- **Cumulative effect**: How features combine to final prediction

### Color Scheme
```
🔵 Blue   = Risk INCREASES   (concerning)
🔴 Red    = Risk DECREASES   (protective)
```

### Clinical Interpretation

**For High-Risk Patient (0.75 prediction):**
```
Baseline: 0.35
├─ exposure_days=200    +0.25  (biggest risk)
├─ severity_severe      +0.12
├─ family_history_yes   +0.08
├─ mutation_nonsense    +0.08
└─ baseline_factor=15%  -0.03  (minimal help)
Final: 0.75 = HIGH RISK
```

**Clinical Assessment:**
- Extensive exposure history is PRIMARY concern
- Severe form adds moderate risk
- Genetic factors contributing
- Low baseline factor level makes it worse
- Overall: Multiple risk factors compound

**For Low-Risk Patient (0.25 prediction):**
```
Baseline: 0.35
├─ baseline_factor=75%  -0.08  (protective)
├─ age_at_exposure=10   -0.05  (later start)
└─ exposure_days=30     -0.02  (minimal exposure)
Final: 0.25 = LOW RISK
```

**Clinical Assessment:**
- Good baseline factor level is protective
- Later age at first exposure reduces risk
- Minimal cumulative exposure
- Overall: Favorable risk profile

### Clinical Use Cases

#### 1. Patient Counseling
Show waterfall to patients:
- "Your risk score is 0.65"
- "Main driver: More exposure days with treatment"
- "Protective factor: Good baseline factor level"

#### 2. Treatment Decision Making
```
Question: Should we switch to prophylaxis?
Waterfall shows:
├─ exposure_days is the biggest driver (+0.25)
├─ Severe form contributes (+0.12)
└─ Current baseline factor helps (-0.03)

Answer: YES - reducing exposure accumulation through prophylaxis 
could significantly lower risk
```

#### 3. Monitoring Strategy
```
Waterfall analysis identifies:
├─ exposure_days → Start inhibitor screening
├─ family_history → More frequent testing
└─ baseline_factor → Monitor replacement therapy
```

### How to Use Waterfall in Clinical Notes
```
CLINICAL ASSESSMENT:
Risk Assessment: 0.68 (MODERATE-HIGH)
Top Drivers:
1. Cumulative exposure (200 days) - Primary concern
2. Severe phenotype - Compounds risk
3. Family history - Genetic predisposition

Protective Factors:
- Baseline factor 45% - Moderate protection
- Recent immune function - Stable

Clinical Plan:
- Increase inhibitor screening frequency
- Consider prophylaxis trial
- Monitor immune status

SHAP Interpretation: Waterfall plot shows exposure accumulation
as primary risk driver, supporting prophylaxis consideration.
```

---

## 3. Force Plot ⚡

### What It Is
A two-panel visualization splitting features into:
- **Left panel**: Risk-INCREASING factors (push probability up)
- **Right panel**: Risk-DECREASING factors (push probability down)

### How to Read It

**Left Side (Red - Risk Increasing):**
```
Risk-Increasing Factors ⬇️
─────────────────────────────
exposure_days=150   0.18
severity_severe     0.12
family_history      0.06
────────────────────────────
Total upward pressure: 0.36
```

**Right Side (Blue - Risk Decreasing):**
```
Risk-Decreasing Factors ⬆️
─────────────────────────────
baseline_factor=45% 0.08
age_at_exposure=8   0.03
────────────────────────────
Total downward pressure: 0.11
```

### Clinical Interpretation

#### Scenario 1: Imbalanced Risk Profile
```
Force Plot shows:
Left (Risk ↑):  0.45
Right (Risk ↓): 0.08

Analysis: Heavily skewed toward risk increases
Clinical Action: Aggressive intervention needed
- Risk factors outweigh protective factors
- Multiple pathways to increased risk
- Limited protective mechanisms
```

#### Scenario 2: Balanced Risk Profile
```
Force Plot shows:
Left (Risk ↑):  0.20
Right (Risk ↓): 0.15

Analysis: Risk factors slightly outweigh protection
Clinical Action: Standard protocols sufficient
- Some protective factors present
- Can manage with monitoring
- May respond to intervention
```

#### Scenario 3: Protected Profile
```
Force Plot shows:
Left (Risk ↑):  0.08
Right (Risk ↓): 0.35

Analysis: Heavily skewed toward protection
Clinical Action: Standard care appropriate
- Strong protective factors
- Multiple risk mitigators
- Low immediate concern
```

### Quick Decision Framework

```
Force Plot Balance
─────────────────────────────────────
Left >> Right (2:1+)    → AGGRESSIVE MANAGEMENT
                           - Increases prophylaxis
                           - Close monitoring
                           - Inhibitor screening early

Left > Right (1.5:1)    → STANDARD MANAGEMENT
                           - Regular monitoring
                           - Consider prophylaxis
                           - Standard inhibitor screening

Left ≈ Right (1:1)      → OBSERVATION + SUPPORT
                           - Monitor trends
                           - Symptom-driven intervention
                           - Standard protocols

Left < Right (1:1.5+)   → REASSURANCE + STANDARD CARE
                           - Protective factors present
                           - May benefit from surveillance
                           - Routine follow-up
```

### Clinical Use Cases

#### Multidisciplinary Team Discussion
```
Presenter: "This patient's force plot shows..."
Left panel: Exposure, severity, family history
Right panel: Good factor levels, young at exposure start

Team consensus: "Risk factors present but manageable. 
Can proceed with current regimen if well-tolerated."
```

#### Insurance/Authorization Discussion
```
Justification using Force Plot:
"Risk factors (left panel) include extensive 
exposure history and severe phenotype. 
These outweigh protective factors (right panel).
Prophylaxis is clinically justified."
```

#### Patient/Family Education
```
Simple explanation:
"On the left, we see things pushing your risk up 
(more exposure time, severe type). 
On the right, things helping protect you 
(good factor levels, good immune system).
Right now, the risk factors are stronger."
```

---

## 4. Dependence Plot 📊

### What It Is
A scatter plot showing the relationship between a specific feature's value and its SHAP value.

### How to Read It
```
Feature: baseline_factor_level
Y-axis: SHAP Value (impact on prediction)
X-axis: Baseline Factor % (0-100%)

Points scatter in pattern:
Low Factor (20%)  → High SHAP values (+0.3) → Increases risk
Mid Factor (50%)  → Medium SHAP values (0.1) → Variable
High Factor (80%) → Low/negative SHAP (-0.1) → Decreases risk
```

### What It Shows
- **Upward trend**: Higher feature value → Higher risk
- **Downward trend**: Higher feature value → Lower risk
- **Horizontal**: Feature doesn't strongly affect predictions
- **Scatter**: Variable effects depending on other features

### Clinical Interpretation

#### Example 1: Exposure Days (Upward Linear Trend)
```
SHAP Value
    ▲
    │     ●
    │   ●   ●
    │ ●       ●
    │●         ●
    └─────────────→ Exposure Days
    
Pattern: Clear upward trend
Clinical insight: More exposure = Consistently higher risk
Implication: Exposure is primary risk driver
Action: Every additional exposure day increases inhibitor risk
```

**Clinical Translation:**
- Patient at 50 exposure days: ~0.12 risk increase
- Patient at 150 exposure days: ~0.35 risk increase
- Risk scales linearly with cumulative exposure

#### Example 2: Baseline Factor Level (Downward Trend)
```
SHAP Value
    ▲
    │●
    │ ●
    │  ●
    │   ● ●
    │     ●●●
    └─────────────→ Baseline Factor %
    
Pattern: Clear downward trend
Clinical insight: Higher factor = Lower risk
Implication: Better baseline hemostasis is protective
Action: Optimize factor replacement to raise baseline
```

**Clinical Translation:**
- Patient with 30% baseline: +0.15 risk
- Patient with 60% baseline: -0.05 risk
- 30% improvement in baseline reduces risk by ~0.20

#### Example 3: Age at First Exposure (Weak Trend)
```
SHAP Value
    ▲
    │ ●  ●
    │●  ● ●  ●
    │ ●●  ●●
    │  ●
    └─────────────→ Age at First Exposure
    
Pattern: Scattered, no clear trend
Clinical insight: Age weakly affects risk
Implication: Other factors are more important
Action: Don't over-emphasize age; focus on exposure management
```

---

## Feature-Specific Interpretation Library

### Key Features in Hemophilia Context

#### 1. **Exposure Days** 📈
```
Typical Pattern: Strong upward slope
Clinical Meaning: Cumulative burden of factor development
Interpretation Guide:
├─ 0-50 days    → Low risk from exposure alone
├─ 50-150 days  → Moderate risk accumulation
├─ 150-300 days → High risk from extensive exposure
└─ >300 days    → Very high risk

Action: Consider prophylaxis if trajectory steep
```

#### 2. **Severity** 🔴
```
Typical Pattern: Categorical steps (mild < moderate < severe)
Clinical Meaning: Bleeding phenotype correlates with inhibitor risk
Interpretation Guide:
├─ Mild        → ~-0.05 SHAP (protective)
├─ Moderate    → ~0.00 SHAP (neutral)
└─ Severe      → ~+0.12 SHAP (increases risk)

Action: Severe patients need enhanced monitoring
```

#### 3. **Mutation Type** 🧬
```
Typical Pattern: Variable by mutation (missense < nonsense/intron22)
Clinical Meaning: Genetic factors predict immune response
Interpretation Guide:
├─ Missense        → Lower risk
├─ Nonsense        → Higher risk
└─ Large deletion  → Variable risk

Action: Genetics inform baseline risk assessment
```

#### 4. **Age at First Exposure** 👶
```
Typical Pattern: Weak protective for older start
Clinical Meaning: Earlier exposure window may increase immunogenicity
Interpretation Guide:
├─ Infant (0-2)     → May increase risk (SHAP ~0.05)
├─ Young child (2-6) → Moderate risk
└─ Older (>6)       → Slightly lower risk (SHAP ~-0.02)

Action: Very early treatment children warrant careful monitoring
```

#### 5. **Baseline Factor Level** 📊
```
Typical Pattern: Strong downward slope
Clinical Meaning: Higher baseline = Better protection
Interpretation Guide:
├─ <20%    → Severe deficiency (SHAP +0.15)
├─ 20-40%  → Moderate deficiency (SHAP +0.08)
├─ 40-60%  → Moderate preservation (SHAP 0.00)
└─ >60%    → Good preservation (SHAP -0.10)

Action: Optimize replacement therapy to raise baseline
```

---

## Clinical Decision Tree Using SHAP

```
START: Review SHAP Waterfall

ANY risk factor with SHAP > +0.20?
├─ YES: Review Feature
│   ├─ Exposure days > 150?
│   │   └─ YES: Consider prophylaxis trial
│   ├─ Severity = Severe?
│   │   └─ YES: Increase monitoring frequency
│   └─ Family history positive?
│       └─ YES: Earlier inhibitor screening
│
└─ NO: Continue to protective factors

ANY protective factor with SHAP < -0.15?
├─ YES: Reinforce Intervention
│   ├─ High baseline factor?
│   │   └─ YES: Maintain current replacement therapy
│   └─ Optimal immune function?
│       └─ YES: Lower inhibitor risk assessment
│
└─ NO: Continue standard monitoring

Final Risk = Σ Risk Factors + Σ Protective Factors
├─ Net Risk > +0.15  → CLOSE MONITORING
├─ Net Risk ≈ 0      → STANDARD PROTOCOLS
└─ Net Risk < -0.10  → REASSURANCE
```

---

## Comparing SHAP Plots in Clinical Context

### When to Use Each Plot

| Situation | Best Plot | Why | Example |
|-----------|-----------|-----|---------|
| **Educating patient/family** | Force Plot | Visual, intuitive split | "These things push risk UP, these push DOWN" |
| **Team case discussion** | Waterfall Plot | Step-by-step logic | "Here's how we got to 65% risk" |
| **Risk factor prioritization** | Summary Plot | Quick ranking | "These 3 features matter most" |
| **Understanding feature effects** | Dependence Plot | Relationship visualization | "How does baseline factor affect risk?" |
| **Regulatory documentation** | Waterfall Plot | Audit trail clarity | Document clinical decision-making |
| **Research/validation** | All plots | Comprehensive view | Validate model against outcomes |

---

## Common Interpretation Mistakes ❌

### Mistake 1: Confusing Correlation with Causation
```
❌ WRONG: "High SHAP value for exposure days CAUSES inhibitor risk"
✅ RIGHT: "High SHAP value indicates exposure days are ASSOCIATED WITH 
           increased inhibitor risk; causation requires clinical validation"
```

### Mistake 2: Ignoring Feature Interactions
```
❌ WRONG: Looking at waterfall factors in isolation
✅ RIGHT: Considering how factors combine
         "Severe + High exposure + Family history = triple risk"
```

### Mistake 3: Over-interpreting for Individual Patients
```
❌ WRONG: "This patient will develop inhibitors because SHAP shows risk"
✅ RIGHT: "This patient has risk factors for inhibitor development;
           regular monitoring and preventive interventions justified"
```

### Mistake 4: Assuming All High SHAP Values Are Bad
```
❌ WRONG: All large SHAP contributions require action
✅ RIGHT: Positive SHAP = Increases risk (concerning)
          Negative SHAP = Decreases risk (protective)
```

### Mistake 5: Neglecting Clinical Context
```
❌ WRONG: Following SHAP recommendations rigidly
✅ RIGHT: Using SHAP to inform clinical judgment
         "SHAP suggests intervention; clinical assessment confirms appropriateness"
```

---

## Documentation Templates

### For Clinical Notes
```
SHAP ANALYSIS SUMMARY:
Risk Score: 0.68 (68%)

Waterfall Analysis:
├─ Primary drivers: [Feature1: +0.18], [Feature2: +0.12]
├─ Protective factors: [Feature3: -0.08]
└─ Net risk contribution: +0.22

Clinical Integration:
The model's primary risk factors ([Feature1], [Feature2]) align with 
clinical assessment. Protective factor [Feature3] is stable.

Recommendation basis:
SHAP analysis identified [specific factor] as primary driver.
This supports clinical plan for [specific intervention].
```

### For Team Communication
```
SHAP INTERPRETATION:
Patient: [Name]
Prediction: [X%] risk

Key Findings:
1. [Feature] = [Value] → [SHAP] impact
2. [Feature] = [Value] → [SHAP] impact  
3. [Feature] = [Value] → [SHAP] impact

Model Explanation:
The ensemble model identified [top feature] as the primary
risk factor, with [secondary feature] having secondary influence.

Clinical Significance:
This aligns with/differs from clinical assessment in [ways].
Recommendation: [specific clinical action]
```

---

## Quality Assurance Checklist

Before trusting SHAP interpretations:

- [ ] Features make clinical sense
- [ ] Top features match clinical knowledge
- [ ] Unexpected features are questionable (investigate)
- [ ] SHAP values align with prediction direction
- [ ] Magnitude of effects are reasonable
- [ ] Waterfall sums correctly to prediction
- [ ] Force plot is balanced (not all push one direction)
- [ ] Results replicate across similar patients
- [ ] Clinical team validates interpretation

---

**Version**: 1.0  
**Last Updated**: April 7, 2026  
**Status**: Clinical Ready ✅
