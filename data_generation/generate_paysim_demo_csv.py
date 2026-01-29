import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 20  # small demo-friendly CSV

# ⚠️ Must match PaySim Stage-2 feature order
FEATURES = [
    "amount",
    "hour",
    "dayofweek",
    "is_weekend",
    "rolling_mean_amount",
    "rolling_std_amount",
    "cumulative_amount",
    "balance_mismatch_rate",
    "amt_vs_recent_avg",
    "txn_velocity_5",
    "errorbalanceorig",
    "errorbalancedest",
    "upi_type_upi_payment",
    "upi_type_upi_transfer"
]

data = []

for i in range(N_SAMPLES):
    is_fraud_like = i % 5 == 0  # every 5th transaction is suspicious

    row = {
        "amount": np.random.uniform(50000, 150000) if is_fraud_like else np.random.uniform(100, 5000),
        "hour": np.random.randint(0, 24),
        "dayofweek": np.random.randint(0, 7),
        "is_weekend": np.random.choice([0, 1]),

        "rolling_mean_amount": np.random.uniform(10000, 50000),
        "rolling_std_amount": np.random.uniform(1000, 20000),
        "cumulative_amount": np.random.uniform(1e5, 1e7),

        "balance_mismatch_rate": 1.0 if is_fraud_like else np.random.uniform(0, 0.1),

        "amt_vs_recent_avg": np.random.uniform(4, 8) if is_fraud_like else np.random.uniform(0.5, 1.5),
        "txn_velocity_5": np.random.randint(1, 10),

        "errorbalanceorig": np.random.uniform(1e4, 1e6) if is_fraud_like else 0,
        "errorbalancedest": np.random.uniform(1e4, 1e6) if is_fraud_like else 0,

        "upi_type_upi_payment": 1 if not is_fraud_like else 0,
        "upi_type_upi_transfer": 1 if is_fraud_like else 0
    }

    data.append(row)

df = pd.DataFrame(data, columns=FEATURES)

df.to_csv("/data_generation/paysim_demo.csv", index=False)

print("✅ PaySim demo CSV generated: paysim_demo.csv")
