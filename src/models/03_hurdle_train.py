"""
Phase 4 | Google Colab (T4 GPU)

Evidence-based model stack from literature:
- arxiv 2307.07771 (So, Scandinavian Actuarial Journal 2024):
    CatBoost > XGBoost > LightGBM for zero-inflated insurance count data.
    CatBoost with Poisson loss outperforms two-stage hurdle on similar data.
- arxiv 2406.16206: Zero-Inflated Tweedie + CatBoost best for aggregate claims.
- M5 competition: LightGBM log1p target + lag features as strong baseline.

Strategy:
  Model A: CatBoost with Poisson loss + 5-fold CV (primary, research-backed)
  Model B: LightGBM with log1p target + 5-fold CV (M5-style baseline)
  Final:   Rank-average blend of A and B (reduces variance, standard Zindi trick)

CatBoost handles cat features natively — no encoding needed.
"""
import os
import joblib
import numpy as np
import pandas as pd
import polars as pl
import catboost as cb
import lightgbm as lgb
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

load_dotenv()

FEATURE_STORE = "data/processed/customer_features_with_timesfm.parquet"
TRAIN_LABELS  = "data/raw/Train.csv"
MODEL_DIR     = "data/processed"
N_FOLDS       = 5

CAT_FEATURES = [
    'Gender', 'IncomeCategory', 'CustomerStatus', 'ClientType',
    'MaritalStatus', 'OccupationCategory', 'IndustryCategory',
    'CustomerBankingType', 'CustomerOnboardingChannel',
    'ResidentialCityName', 'CountryCodeNationality',
    'CertificationTypeDescription', 'ContactPreference',
]

EXCLUDE_COLS = {'UniqueID', 'next_3m_txn_count', 'BirthDate'}


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))))


def rank_avg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rank-average blend: reduces variance better than simple average."""
    from scipy.stats import rankdata
    return (rankdata(a) + rankdata(b)) / 2


def train():
    df = pl.read_parquet(FEATURE_STORE)
    labels = pl.read_csv(TRAIN_LABELS)

    df_train = labels.join(df, on="UniqueID", how="left").to_pandas()
    df_test  = pl.read_csv("data/raw/Test.csv").join(df, on="UniqueID", how="left").to_pandas()

    # Fill cat features — CatBoost handles them natively as strings
    df_train[CAT_FEATURES] = df_train[CAT_FEATURES].fillna('Missing').astype(str)
    df_test[CAT_FEATURES]  = df_test[CAT_FEATURES].fillna('Missing').astype(str)

    y = df_train['next_3m_txn_count'].values
    features = [c for c in df_train.columns if c not in EXCLUDE_COLS and c != 'next_3m_txn_count']
    X      = df_train[features]
    X_test = df_test[features]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_cat = np.zeros(len(df_train))
    oof_lgb = np.zeros(len(df_train))
    test_cat = np.zeros(len(df_test))
    test_lgb = np.zeros(len(df_test))

    print(f"Training on {len(features)} features, {N_FOLDS}-fold CV\n")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"── Fold {fold+1}/{N_FOLDS} ──")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # ── Model A: CatBoost Poisson ──────────────────────────────────────
        # Research (arxiv 2307.07771): CatBoost best for zero-inflated counts.
        # Poisson loss directly models count distribution — no hurdle needed.
        cat_model = cb.CatBoostRegressor(
            loss_function='Poisson',
            iterations=2000,
            learning_rate=0.02,
            depth=6,
            cat_features=CAT_FEATURES,
            eval_metric='RMSE',
            early_stopping_rounds=100,
            random_seed=42,
            verbose=0,
        )
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )
        oof_cat[val_idx] = np.clip(cat_model.predict(X_val), 0, None)
        test_cat += np.clip(cat_model.predict(X_test), 0, None) / N_FOLDS
        print(f"  CatBoost RMSLE: {rmsle(y_val, oof_cat[val_idx]):.5f}")

        # ── Model B: LightGBM log1p (M5-style) ────────────────────────────
        # M5 winning approach: log1p target with RMSE loss ≈ optimises RMSLE.
        X_tr_lgb, X_val_lgb = X_tr.copy(), X_val.copy()
        X_tr_lgb[CAT_FEATURES]  = X_tr_lgb[CAT_FEATURES].astype('category')
        X_val_lgb[CAT_FEATURES] = X_val_lgb[CAT_FEATURES].astype('category')

        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=127,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr_lgb, np.log1p(y_tr),
            eval_set=[(X_val_lgb, np.log1p(y_val))],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb[val_idx] = np.clip(np.expm1(lgb_model.predict(X_val_lgb)), 0, None)
        X_test_lgb = X_test.copy()
        X_test_lgb[CAT_FEATURES] = X_test_lgb[CAT_FEATURES].astype('category')
        test_lgb += np.clip(np.expm1(lgb_model.predict(X_test_lgb)), 0, None) / N_FOLDS
        print(f"  LightGBM RMSLE: {rmsle(y_val, oof_lgb[val_idx]):.5f}")

    print(f"\n{'='*40}")
    print(f"OOF CatBoost RMSLE : {rmsle(y, oof_cat):.5f}")
    print(f"OOF LightGBM RMSLE : {rmsle(y, oof_lgb):.5f}")

    # Rank-average blend
    oof_blend  = rank_avg(oof_cat, oof_lgb)
    # Convert ranks back to count scale via percentile mapping on OOF
    # Simple approach: weighted average in count space (0.6 CatBoost, 0.4 LightGBM)
    oof_blend  = 0.6 * oof_cat + 0.4 * oof_lgb
    test_blend = 0.6 * test_cat + 0.4 * test_lgb
    print(f"OOF Blend RMSLE    : {rmsle(y, oof_blend):.5f}")
    print(f"{'='*40}\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'test_blend': test_blend, 'test_cat': test_cat, 'test_lgb': test_lgb,
                 'oof_blend': oof_blend, 'features': features}, f"{MODEL_DIR}/predictions.joblib")
    print(f"Predictions saved → {MODEL_DIR}/predictions.joblib")
    return rmsle(y, oof_blend)


if __name__ == "__main__":
    train()
