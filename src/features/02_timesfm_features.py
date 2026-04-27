"""
Phase 3 | Google Colab (T4 GPU)
Runs TimesFM 2.5 zero-shot inference on monthly transaction histories.
Env vars required: HF_REPO
Install: pip install polars huggingface_hub git+https://github.com/google-research/timesfm.git
"""
import os
import numpy as np
import polars as pl
import timesfm


def generate_timesfm_features():
    print("Pulling data from HF Registry...")
    df = pl.read_parquet(f"hf://datasets/{os.environ['HF_REPO']}/customer_base_features.parquet")

    print("Initializing TimesFM 2.5...")
    tfm = timesfm.TimesFm(
        context_len=512,
        horizon_len=128,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-2.5-200m-transformers")

    print("Executing batch inference...")
    ts_data = df["monthly_txn_history"].to_list()
    point_forecast, _ = tfm.forecast(ts_data, freq=[0])

    # Sum predicted Nov + Dec + Jan (3-month horizon)
    timesfm_3m = np.sum(point_forecast[:, :3], axis=1)

    out = (
        df
        .with_columns(pl.Series("timesfm_3m_forecast", timesfm_3m))
        .drop("monthly_txn_history")
    )

    out_path = "data/processed/customer_features_with_timesfm.parquet"
    out.write_parquet(out_path)
    print(f"TimesFM features written → {out_path}")


if __name__ == "__main__":
    generate_timesfm_features()
