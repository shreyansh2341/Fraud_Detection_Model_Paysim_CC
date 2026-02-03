import numpy as np
import pandas as pd
import joblib
import os

# =====================================================
# FIX BASE DIRECTORY
# =====================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

FEATURE_PATH = os.path.join(BASE_DIR, "models", "cc_features.pkl")
cc_features = joblib.load(FEATURE_PATH)

# =====================================================
# CONFIG
# =====================================================
N_ROWS = 50
FRAUD_RATIO = 0.15

rng = np.random.default_rng(123)

# =====================================================
# DATA GENERATION
# =====================================================
data = []

for _ in range(N_ROWS):
    is_fraud = rng.random() < FRAUD_RATIO

    row = {
        f: rng.normal(5, 2) if is_fraud else rng.normal(0, 1)
        for f in cc_features
    }

    data.append(row)

# =====================================================
# SAVE CSV
# =====================================================
df = pd.DataFrame(data)[cc_features]
OUTPUT_PATH = os.path.join(BASE_DIR, "data_generation/creditcard_demo_final.csv")
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Credit Card demo CSV generated")
print("📄 Path:", OUTPUT_PATH)
print("Rows:", len(df))
print("Approx fraud %:", int(FRAUD_RATIO * 100))
