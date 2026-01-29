import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 20

FEATURES = [
    "v1","v2","v3","v4","v5","v6","v7","v8","v9","v10",
    "v11","v12","v13","v14","v15","v16","v17","v18","v19","v20",
    "v21","v22","v23","v24","v25","v26","v27","v28",
    "amount_scaled",
    "hour",
    "dayofweek",
    "is_weekend"
]

data = []

for i in range(N_SAMPLES):
    is_fraud_like = i % 6 == 0

    row = {}

    for v in FEATURES:
        if v.startswith("v"):
            row[v] = np.random.normal(3, 2) if is_fraud_like else np.random.normal(0, 1)
        elif v == "amount_scaled":
            row[v] = np.random.uniform(3, 6) if is_fraud_like else np.random.uniform(-1, 1)
        elif v == "hour":
            row[v] = np.random.randint(0, 24)
        elif v == "dayofweek":
            row[v] = np.random.randint(0, 7)
        elif v == "is_weekend":
            row[v] = np.random.choice([0, 1])

    data.append(row)

df = pd.DataFrame(data, columns=FEATURES)

df.to_csv("creditcard_demo.csv", index=False)

print("✅ Credit Card demo CSV generated: creditcard_demo.csv")
