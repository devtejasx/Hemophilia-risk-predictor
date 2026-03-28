import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Load data
data = pd.read_csv("genomic.csv")
clinical = pd.read_csv("clinical.csv")

# Merge
df = pd.merge(data, clinical, on="patient_id")

# PRINT to check problem
print("Before cleaning:", df["target"].unique())

# KEEP ONLY 0 and 1
df = df[df["target"].isin([0, 1])]

# Split
y = df["target"]
X = df.drop(["target", "patient_id"], axis=1)

# Encode
X = pd.get_dummies(X)

print("After cleaning:", y.unique())

# Train
rf = RandomForestClassifier()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

rf.fit(X, y)
xgb.fit(X, y)

# Save
joblib.dump(rf, "rf.pkl")
joblib.dump(xgb, "xgb.pkl")

joblib.dump(list(X.columns), "columns.pkl")

print("✅ Model trained successfully!")