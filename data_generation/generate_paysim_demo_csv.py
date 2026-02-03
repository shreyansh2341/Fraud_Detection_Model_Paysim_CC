import numpy as np
import pandas as pd
import joblib
import os

# =====================================================
# FIX BASE DIRECTORY (MOVE UP FROM data_generation/)
# =====================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PREPROC_PATH = os.path.join(
    BASE_DIR, "artifacts", "paysim_preproc.joblib"
)

# =====================================================
# LOAD TRAINED FEATURE ORDER
# =====================================================
paysim_preproc = joblib.load(PREPROC_PATH)
FEATURES = list(paysim_preproc.feature_names_in_)

# =====================================================
# CONFIG
# =====================================================
N_ROWS = 50
FRAUD_RATIO = 0.2   # 2% fraud

rng = np.random.default_rng(42)

# =====================================================
# DATA GENERATION
# =====================================================
rows = []

for _ in range(N_ROWS):
    is_fraud = rng.random() < FRAUD_RATIO

    amount = rng.uniform(2000, 120000) if is_fraud else rng.uniform(10, 5000)

    oldbalanceOrg = rng.uniform(0, 200000)
    newbalanceOrig = (
        oldbalanceOrg if is_fraud else max(oldbalanceOrg - amount, 0)
    )

    oldbalanceDest = rng.uniform(0, 100000)
    newbalanceDest = (
        oldbalanceDest if is_fraud else oldbalanceDest + amount
    )

    errorBalanceOrig = oldbalanceOrg - newbalanceOrig - amount
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest

    row = {
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "errorBalanceOrig": errorBalanceOrig,
        "errorBalanceDest": errorBalanceDest,
        "dayofweek": rng.integers(0, 7),
        "hour": rng.integers(0, 24),
        "is_weekend": rng.integers(0, 2),
        "has_balance_mismatch": int(is_fraud)
    }

    rows.append([row.get(f, 0) for f in FEATURES])

# =====================================================
# SAVE CSV
# =====================================================
df = pd.DataFrame(rows, columns=FEATURES)
OUTPUT_PATH = os.path.join(BASE_DIR, "data_generation/paysim_demo_final.csv")
df.to_csv(OUTPUT_PATH, index=False)

print("✅ PaySim demo CSV generated")
print("📄 Path:", OUTPUT_PATH)
print("Rows:", len(df))
print("Approx fraud %:", int(FRAUD_RATIO * 100))
