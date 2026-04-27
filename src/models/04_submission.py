"""
Phase 5 | Google Colab (T4 GPU)
Generates Zindi submission from blended predictions.
Run after 03_hurdle_train.py.
"""
import os
import joblib
import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = "data/processed"
OUT_PATH  = "submissions/final_submission.csv"


def generate_submission():
    preds = joblib.load(f"{MODEL_DIR}/predictions.joblib")
    test  = pl.read_csv("data/raw/Test.csv").to_pandas()

    # Zindi evaluates RMSLE on raw counts — submit raw counts
    submission = pd.DataFrame({
        'UniqueID': test['UniqueID'],
        'next_3m_txn_count': np.clip(preds['test_blend'], 0, None),
    })

    os.makedirs("submissions", exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)
    print(f"Submission written → {OUT_PATH}  ({len(submission)} rows)")
    print(f"Prediction stats: min={submission['next_3m_txn_count'].min():.2f}, "
          f"max={submission['next_3m_txn_count'].max():.2f}, "
          f"mean={submission['next_3m_txn_count'].mean():.2f}, "
          f"zeros={( submission['next_3m_txn_count'] < 0.5).sum()}")


if __name__ == "__main__":
    generate_submission()
