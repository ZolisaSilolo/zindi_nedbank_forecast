"""
Phase 4 | Google Colab (T4 GPU)
Two-Stage Hurdle Model: LightGBM classifier → XGBoost regressor.
Prints validation RMSLE. Saves trained models for submission script.
"""
import os
import joblib
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

load_dotenv()

FEATURE_STORE = "data/processed/customer_features_with_timesfm.parquet"
TRAIN_LABELS  = "data/raw/Train.csv"
MODEL_DIR     = "data/processed"

EXCLUDE_COLS = {"UniqueID", "next_3m_txn_count", "target_is_active", "target_log1p"}


def rmsle(y_true_counts: np.ndarray, y_pred_counts: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(np.log1p(y_true_counts), np.log1p(y_pred_counts)))


def train():
    df = pl.read_parquet(FEATURE_STORE)
    labels = pl.read_csv(TRAIN_LABELS)

    df_train = labels.join(df, on="UniqueID", how="left").to_pandas()
    df_train["target_is_active"] = (df_train["next_3m_txn_count"] > 0).astype(int)
    df_train["target_log1p"]     = np.log1p(df_train["next_3m_txn_count"])

    features = [c for c in df_train.columns if c not in EXCLUDE_COLS]
    X      = df_train[features]
    y_cls  = df_train["target_is_active"]
    y_reg  = df_train["target_log1p"]

    X_tr, X_val, yc_tr, yc_val, yr_tr, yr_val = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42
    )

    print("Training Stage 1 (LightGBM classifier)...")
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, random_state=42)
    clf.fit(X_tr, yc_tr)
    prob_val = clf.predict_proba(X_val)[:, 1]

    print("Training Stage 2 (XGBoost regressor)...")
    active = yc_tr == 1
    reg = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=5, random_state=42
    )
    reg.fit(X_tr[active], yr_tr[active])
    log_pred_val = reg.predict(X_val)

    final_val = prob_val * np.clip(np.expm1(log_pred_val), 0, None)
    true_val_counts = np.expm1(yr_val.values)
    print(f"Validation RMSLE: {rmsle(true_val_counts, final_val):.5f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf,      f"{MODEL_DIR}/clf.joblib")
    joblib.dump(reg,      f"{MODEL_DIR}/reg.joblib")
    joblib.dump(features, f"{MODEL_DIR}/features.joblib")
    print("Models saved to data/processed/")
    return rmsle(true_val_counts, final_val)


if __name__ == "__main__":
    train()
