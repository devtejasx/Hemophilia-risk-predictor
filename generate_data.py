import pandas as pd
import random

data = []

for i in range(100):
    data.append({
        "patient_id": i,
        "age_first_treatment": random.randint(1, 10),
        "dose_intensity": random.randint(10, 100),
        "exposure_days": random.randint(1, 50),
        "mutation_type": random.choice(["intron22", "missense", "nonsense", "frameshift"]),
        "exon": random.randint(1, 25),
        "severity": random.choice(["mild", "moderate", "severe"]),
        "target": random.choice([0, 1])
    })

df = pd.DataFrame(data)

clinical = df[["patient_id", "age_first_treatment", "dose_intensity", "exposure_days", "target"]]
genomic = df[["patient_id", "mutation_type", "exon", "severity"]]

clinical.to_csv("clinical.csv", index=False)
genomic.to_csv("genomic.csv", index=False)

print("Data generated!")