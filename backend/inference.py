import sys
import os

# -------------------------------------------------
# FIX PYTHON PATH (CRITICAL FOR UVICORN)
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# NOW SAFE TO IMPORT PROJECT MODULES
# -------------------------------------------------
import numpy as np
import pandas as pd

from src.final_ensemble_inference import ensemble_predict
from src.utils.preprocessor import clean_and_engineer_upi



def run_inference(payload):
    raw_values = payload.tabular_features

    # Raw PaySim schema
    column_names = [
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud"
    ]

    raw_df = pd.DataFrame([raw_values], columns=column_names)

    # -------- Feature Engineering --------
    if payload.transaction_type == "paysim":
        engineered_df = clean_and_engineer_upi(raw_df)

    elif payload.transaction_type == "creditcard":
        engineered_df = raw_df.copy()

    else:
        raise ValueError("Invalid transaction_type")

    # -------- LSTM (Optional) --------
    lstm_sequence = None
    if payload.transaction_type == "paysim" and payload.lstm_sequence is not None:
        lstm_sequence = np.asarray(payload.lstm_sequence, dtype=np.float32)

    # -------- Final Prediction --------
    result = ensemble_predict(
        transaction_type=payload.transaction_type,
        raw_df=engineered_df,
        lstm_sequence=lstm_sequence
    )

    return result["decision"], result["explanation"]
