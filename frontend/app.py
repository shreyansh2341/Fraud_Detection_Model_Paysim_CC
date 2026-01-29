# import streamlit as st
# import pandas as pd
# import requests
# import numpy as np

# # ======================================================
# # CONFIG
# # ======================================================
# API_URL = "http://127.0.0.1:8000/predict"

# st.set_page_config(
#     page_title="Fraud Detection Dashboard",
#     layout="wide"
# )

# st.title("📊 CSV-Based Fraud Detection Dashboard")

# # ======================================================
# # SIDEBAR
# # ======================================================
# st.sidebar.header("⚙️ Settings")

# transaction_type = st.sidebar.selectbox(
#     "Transaction Type",
#     ["paysim", "creditcard"]
# )

# st.sidebar.info(
#     "Upload a CSV file with transaction features.\n\n"
#     "The system will detect fraud using the trained ensemble model:\n"
#     "- PaySim → LSTM + AE + XGBoost\n"
#     "- Credit Card → XGBoost"
# )

# # ======================================================
# # CSV UPLOAD
# # ======================================================
# uploaded_file = st.file_uploader(
#     "📂 Upload CSV file",
#     type=["csv"]
# )

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     st.subheader("📄 Uploaded Data Preview")
#     st.dataframe(df.head())

#     st.write(f"**Total transactions:** {len(df)}")

#     # ==================================================
#     # RUN DETECTION
#     # ==================================================
#     if st.button("🚨 Run Fraud Detection"):
#         results = []
#         progress = st.progress(0.0)

#         for i, row in df.iterrows():
#             tabular_features = row.values.tolist()

#             payload = {
#                 "transaction_type": transaction_type,
#                 "tabular_features": tabular_features
#             }

#             # ----------------------------------------------
#             # PaySim → Dummy LSTM sequence (demo-safe)
#             # ----------------------------------------------
#             if transaction_type == "paysim":
#                 seq_len = 5
#                 n_features = len(tabular_features)

#                 dummy_sequence = np.tile(
#                     tabular_features,
#                     (seq_len, 1)
#                 ).tolist()

#                 payload["lstm_sequence"] = dummy_sequence

#             # ----------------------------------------------
#             # API CALL
#             # ----------------------------------------------
#             try:
#                 response = requests.post(API_URL, json=payload, timeout=10)

#                 if response.status_code != 200:
#                     raise RuntimeError(response.text)

#                 response_json = response.json()

#                 decision = response_json["decision"]
#                 explanation = response_json["explanation"]

#             except Exception as e:
#                 decision = -1
#                 explanation = f"ERROR: {str(e)}"

#             results.append({
#                 "Prediction": decision,                     # 0 / 1
#                 "Label": "FRAUD" if decision == 1 else "LEGIT",
#                 "Explanation": explanation
#             })

#             progress.progress((i + 1) / len(df))

#         # ==================================================
#         # RESULTS TABLE
#         # ==================================================
#         result_df = pd.concat(
#             [df.reset_index(drop=True), pd.DataFrame(results)],
#             axis=1
#         )

#         st.subheader("✅ Detection Results")
#         st.dataframe(result_df)

#         # ==================================================
#         # SUMMARY METRICS (FIXED)
#         # ==================================================
#         fraud_count = (result_df["Prediction"] == 1).sum()
#         legit_count = (result_df["Prediction"] == 0).sum()
#         error_count = (result_df["Prediction"] == -1).sum()

#         col1, col2, col3, col4 = st.columns(4)

#         col1.metric("Total Transactions", len(result_df))
#         col2.metric("Fraud Detected", fraud_count)
#         col3.metric("Legitimate", legit_count)
#         col4.metric("Errors", error_count)

#         # ==================================================
#         # DOWNLOAD RESULTS
#         # ==================================================
#         st.subheader("⬇️ Download Results")

#         csv_out = result_df.to_csv(index=False).encode("utf-8")

#         st.download_button(
#             label="Download Fraud Detection Results",
#             data=csv_out,
#             file_name="fraud_detection_results.csv",
#             mime="text/csv"
#         )

import streamlit as st
import pandas as pd
import requests
import numpy as np
import time

# ======================================================
# CONFIG
# ======================================================
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="UPI Fraud Intelligence Dashboard",
    page_icon="🚨",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🚨 UPI Fraud Intelligence Dashboard")
st.markdown("---")

# ======================================================
# SIDEBAR & SETTINGS
# ======================================================
st.sidebar.header("⚙️ Model Configuration")

transaction_type = st.sidebar.selectbox(
    "Select Dataset Logic",
    ["paysim", "creditcard"],
    help="Determines which preprocessing and model stack to use."
)

st.sidebar.divider()
st.sidebar.info(
    "**Ensemble Expert Stack:**\n"
    "- **XGBoost**: Tabular Patterns\n"
    "- **LSTM**: Temporal Sequences\n"
    "- **Autoencoder**: Anomaly Detection\n"
    "- **Isolation Forest**: Outlier Analysis"
)

# ======================================================
# DATA UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "📂 Upload Transaction CSV (Raw Data)",
    type=["csv"],
    help="Upload raw transaction logs. The system will automatically engineer UPI-specific features."
)

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)
    
    # Display Preview
    st.subheader("📄 Raw Transaction Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.write(f"**Total rows to analyze:** {len(df)}")

    # ==================================================
    # EXECUTION ENGINE
    # ==================================================
    if st.button("🚀 Start Fraud Analysis"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()

        # Iterate through rows
        for i, row in df.iterrows():
            status_text.text(f"Processing transaction {i+1} of {len(df)}...")
            
            # Prepare Payload
            # We send the raw row as a dictionary so the backend preprocessor 
            # can calculate errorBalanceOrig, upi_type, etc.
            raw_data = row.to_dict()
            
            payload = {
                "transaction_type": transaction_type,
                "tabular_features": list(raw_data.values()), 
                "lstm_sequence": None
            }

            # Generate real-time sequence if PaySim
            if transaction_type == "paysim":
                # For high generalization, we simulate a sequence from the raw row
                # In production, this would be pulled from a Redis cache of past user txns
                row_values = list(raw_data.values())
                payload["lstm_sequence"] = np.tile(row_values, (5, 1)).tolist()

            # API Communication
            try:
                response = requests.post(API_URL, json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    # Capture "FRAUD" or "LEGIT" from backend
                    res_label = data["decision"] 
                    res_reason = data["explanation"]
                else:
                    res_label = "ERROR"
                    res_reason = f"Server Error: {response.status_code}"

            except Exception as e:
                res_label = "ERROR"
                res_reason = str(e)

            results.append({
                "Detection": res_label,
                "Insight": res_reason
            })
            
            # Update Progress
            progress_bar.progress((i + 1) / len(df))

        end_time = time.time()
        status_text.success(f"Analysis Complete! Processed {len(df)} rows in {end_time - start_time:.2f}s")

        # ==================================================
        # RESULTS & ANALYTICS
        # ==================================================
        # Merge results back to original dataframe
        final_df = pd.concat([df, pd.DataFrame(results)], axis=1)

        # Metrics Row
        # FIXED: Using string matching to fix the "Zero result" bug
        fraud_total = (final_df["Detection"] == "FRAUD").sum()
        legit_total = (final_df["Detection"] == "LEGIT").sum()
        err_total = (final_df["Detection"] == "ERROR").sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Scanned", len(final_df))
        m2.metric("Fraud Flagged", fraud_total, delta=f"{fraud_total/len(final_df):.1%}", delta_color="inverse")
        m3.metric("Legitimate", legit_total)
        m4.metric("Pipeline Errors", err_total)

        # Highlight Results
        st.subheader("📋 Detailed Analysis Report")
        
        def highlight_fraud(val):
            color = '#ff4b4b' if val == "FRAUD" else '#90ee90' if val == "LEGIT" else '#ffcc00'
            return f'background-color: {color}; color: black; font-weight: bold'

        st.dataframe(
            final_df.style.applymap(highlight_fraud, subset=['Detection']),
            use_container_width=True
        )

        # ==================================================
        # EXPORT
        # ==================================================
        st.divider()
        csv_download = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Audit Report (CSV)",
            data=csv_download,
            file_name=f"fraud_audit_{int(time.time())}.csv",
            mime="text/csv"
        )