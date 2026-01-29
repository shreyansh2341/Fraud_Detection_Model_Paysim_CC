import pandas as pd
import numpy as np

def clean_and_engineer_upi(df):
    """
    Standardizes raw CSV data into model-ready features for UPI fraud detection.
    Based on Pay-Sim Dataset.ipynb cleaning steps.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # 1. Feature Engineering (The "UPI Signals")
    df['errorbalanceorig'] = df['oldbalanceorg'] - df['amount'] - df['newbalanceorig']
    df['errorbalancedest'] = df['oldbalancedest'] + df['amount'] - df['newbalancedest']
    df['has_balance_mismatch'] = ((df['errorbalanceorig'].abs() > 1e-6) | 
                                  (df['errorbalancedest'].abs() > 1e-6)).astype(int)

    # 2. Temporal Features
    df['hour'] = df['step'] % 24
    df['is_weekend'] = ((df['step'] // 24) % 7 >= 5).astype(int)

    # 3. Categorical Alignment
    # Force columns to exist to prevent "missing column" errors in XGBoost
    df['upi_type_upi_payment'] = (df['type'].str.upper() == 'PAYMENT').astype(int)
    df['upi_type_upi_transfer'] = (df['type'].str.upper() == 'TRANSFER').astype(int)

    # 4. Feature Selection (Ordering must match paysim_stage2_features.pkl)
    FEATURES = [
        'amount', 'oldbalanceorg', 'newbalanceorig', 'oldbalancedest', 'newbalancedest',
        'errorbalanceorig', 'errorbalancedest', 'hour', 'is_weekend', 
        'has_balance_mismatch', 'upi_type_upi_payment', 'upi_type_upi_transfer'
    ]
    
    return df[FEATURES]