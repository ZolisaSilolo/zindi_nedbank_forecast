"""
Phase 5 | Google Colab (T4 GPU)
Loads trained hurdle models and generates the Zindi submission CSV.
Run after 03_hurdle_train.py has completed successfully.
"""
import pickle
import numpy as np
import pandas as pd
import polars as pl

FEATURE_STORE = "data/processed/customer_features_with_timesfm.parquet"
TEST_LABELS   = "data/raw/Test.csv"
MODEL_DIR     = "data/processed"
OUT_PATH      = "submissions/final_submission.csv"


def generate_submission():
    df = pl.read_parquet(FEATURE_STORE)
    test = pl.read_csv(TEST_LABELS)
    df_test = test.join(df, on="UniqueID", how="left").to_pandas()

    with open(f"{MODEL_DIR}/clf.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(f"{MODEL_DIR}/reg.pkl", "rb") as f:
        reg = pickle.load(f)
    with open(f"{MODEL_DIR}/features.pkl", "rb") as f:
        features = pickle.load(f)

    X_test = df_test[features]
    prob_active = clf.predict_proba(X_test)[:, 1]
    log_count   = reg.predict(X_test)

    # Zindi expects log1p-transformed predictions
    preds = np.log1p(prob_active * np.expm1(log_count))

    submission = pd.DataFrame({"UniqueID": df_test["UniqueID"], "next_3m_txn_count": preds})
    submission.to_csv(OUT_PATH, index=False)
    print(f"Submission written → {OUT_PATH}  ({len(submission)} rows)")


if __name__ == "__main__":
    generate_submission()
