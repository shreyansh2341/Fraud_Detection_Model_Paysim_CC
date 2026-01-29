# import sys
# import os
# import numpy as np

# # ======================================================
# # ADD src/ DIRECTORY TO PYTHON PATH
# # ======================================================
# CURRENT_DIR = os.path.dirname(__file__)
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# if SRC_DIR not in sys.path:
#     sys.path.insert(0, SRC_DIR)

# # ======================================================
# # IMPORT ENSEMBLE
# # ======================================================
# from final_ensemble_inference import ensemble_predict


# # ======================================================
# # FASTAPI → ENSEMBLE BRIDGE
# # ======================================================
# def run_inference(payload):
#     """
#     Converts FastAPI request payload into
#     proper numpy inputs for ensemble_predict()
#     """

#     # -------------------------------
#     # TABULAR FEATURES
#     # -------------------------------
#     tabular_features = np.asarray(
#         payload.tabular_features,
#         dtype=np.float32
#     ).reshape(1, -1)

#     # -------------------------------
#     # LSTM SEQUENCE (PaySim only)
#     # -------------------------------
#     lstm_sequence = None

#     if payload.transaction_type == "paysim":
#         if payload.lstm_sequence is None:
#             raise ValueError("lstm_sequence is required for PaySim transactions")

#         lstm_sequence = np.asarray(
#             payload.lstm_sequence,
#             dtype=np.float32
#         )

#         if lstm_sequence.ndim == 2:
#             lstm_sequence = lstm_sequence.reshape(
#                 1,
#                 lstm_sequence.shape[0],
#                 lstm_sequence.shape[1]
#             )

#     # -------------------------------
#     # CALL ENSEMBLE
#     # -------------------------------
#     decision, explanation = ensemble_predict(
#         transaction_type=payload.transaction_type,
#         tabular_features=tabular_features,
#         lstm_sequence=lstm_sequence
#     )

#     return decision, explanation


import sys
import os
import numpy as np
import pandas as pd

# ======================================================
# ADD src/ DIRECTORY TO PYTHON PATH
# ======================================================
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ======================================================
# IMPORT ENSEMBLE & PREPROCESSOR
# ======================================================
from final_ensemble_inference import ensemble_predict
from utils.preprocessor import clean_and_engineer_upi

def run_inference(payload):
    """
    Bridge between FastAPI and the ML Ensemble.
    Converts raw frontend data into engineered features for the models.
    """

    # 1. Map raw list back to DataFrame for Feature Engineering
    # This is critical to ensure clean_and_engineer_upi can find column names
    raw_values = payload.tabular_features
    
    # Standard PaySim columns as expected from your demo generation script
    column_names = [
        "step", "type", "amount", "nameorig", "oldbalanceorg", "newbalanceorig",
        "namedest", "oldbalancedest", "newbalancedest", "isfraud", "isflaggedfraud"
    ]
    
    # Create a temporary DataFrame (1 row)
    try:
        raw_df = pd.DataFrame([raw_values], columns=column_names)
    except ValueError:
        # Fallback if the CSV has different columns (e.g., if it's already engineered)
        # This makes the model more robust to different CSV formats
        return -1, "Feature mismatch: CSV columns do not match expected PaySim format."

    # 2. Handle LSTM Sequence (Temporal Expert)
    lstm_sequence = None
    if payload.transaction_type == "paysim":
        if payload.lstm_sequence is None:
            raise ValueError("lstm_sequence is required for PaySim transactions")

        # Convert list to (1, 5, 12) shape
        lstm_sequence = np.asarray(payload.lstm_sequence, dtype=np.float32)
        if lstm_sequence.ndim == 2:
            lstm_sequence = lstm_sequence.reshape(1, lstm_sequence.shape[0], lstm_sequence.shape[1])

    # 3. Call Parallel Ensemble
    # We pass the raw_df, NOT a pre-scaled array, because ensemble_predict
    # now handles its own cleaning using the unified preprocessor.
    decision, explanation = ensemble_predict(
        transaction_type=payload.transaction_type,
        raw_df=raw_df,
        lstm_sequence=lstm_sequence
    )

    return decision, explanation