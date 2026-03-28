import joblib
import pandas as pd
import pickle

# Load models
rf = joblib.load("rf.pkl")
xgb = joblib.load("xgb.pkl")

# NEW PATIENT DATA (you can change values)
new_data = {
    "mutation_type": "intron22",
    "exon": 22,
    "severity": "severe",
    "age_first_treatment": 2,
    "dose_intensity": 80,
    "exposure_days": 20
}

# Convert to DataFrame
df = pd.DataFrame([new_data])

# Encode (same as training)
df = pd.get_dummies(df)

# IMPORTANT: Align columns (missing columns = 0)
# Load columns with memory-safe approach
try:
    train_columns = joblib.load("columns.pkl", mmap_mode='r')
except (MemoryError, EOFError, pickle.UnpicklingError):
    try:
        # Try without mmap if mmap fails
        train_columns = joblib.load("columns.pkl")
    except Exception as e:
        print(f"Error loading columns: {e}. Using fallback.")
        train_columns = None

for col in train_columns:
    if col not in df:
        df[col] = 0

df = df[train_columns]

# Predict
p1 = rf.predict_proba(df)[0][1]
p2 = xgb.predict_proba(df)[0][1]

risk = (p1 + p2) / 2

print("🔮 Risk Score:", risk)

if risk > 0.6:
    print("⚠️ HIGH RISK")
else:
    print("✅ LOW RISK")