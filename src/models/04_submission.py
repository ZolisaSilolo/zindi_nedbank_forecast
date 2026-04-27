"""
Phase 5 | Google Colab (T4 GPU)
Loads trained hurdle models and generates the Zindi submission CSV.
Run after 03_hurdle_train.py has completed successfully.
"""
import os
import joblib
import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv

load_dotenv()

FEATURE_STORE = "data/processed/customer_features_with_timesfm.parquet"
TEST_LABELS   = "data/raw/Test.csv"
MODEL_DIR     = "data/processed"
OUT_PATH      = "submissions/final_submission.csv"


def generate_submission():
    df = pl.read_parquet(FEATURE_STORE)
    test = pl.read_csv(TEST_LABELS)
    df_test = test.join(df, on="UniqueID", how="left").to_pandas()

    clf      = joblib.load(f"{MODEL_DIR}/clf.joblib")
    reg      = joblib.load(f"{MODEL_DIR}/reg.joblib")
    features = joblib.load(f"{MODEL_DIR}/features.joblib")

    X_test = df_test[features]
    prob_active = clf.predict_proba(X_test)[:, 1]
    log_count   = reg.predict(X_test)

    # Zindi evaluates RMSLE on raw counts — submit raw counts, not log1p
    preds = np.clip(prob_active * np.expm1(log_count), 0, None)

    os.makedirs("submissions", exist_ok=True)
    submission = pd.DataFrame({"UniqueID": df_test["UniqueID"], "next_3m_txn_count": preds})
    submission.to_csv(OUT_PATH, index=False)
    print(f"Submission written → {OUT_PATH}  ({len(submission)} rows)")


if __name__ == "__main__":
    generate_submission()
