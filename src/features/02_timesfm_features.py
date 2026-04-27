"""
Phase 3 | Google Colab (T4 GPU)
Runs TimesFM 2.5 zero-shot inference on monthly transaction histories.
Env vars required: HF_REPO, HF_TOKEN (for private dataset read)
Install: pip install polars huggingface_hub python-dotenv
         pip install -e ".[torch]"  (from cloned timesfm repo)
"""
import os
import torch
import numpy as np
import polars as pl
import timesfm
from dotenv import load_dotenv

load_dotenv()

torch.set_float32_matmul_precision("high")


def generate_timesfm_features():
    hf_repo  = os.environ["HF_REPO"]
    hf_token = os.environ.get("HF_TOKEN")

    print("Pulling data from HF Registry...")
    df = pl.read_parquet(
        f"hf://datasets/{hf_repo}/customer_base_features.parquet",
        storage_options={"token": hf_token},
    )

    print("Initializing TimesFM 2.5...")
    # TimesFM 2.5 API: from_pretrained class method, no hparams constructor
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    # compile() takes a ForecastConfig — freq parameter removed in 2.5
    model.compile(
        timesfm.ForecastConfig(
            max_context=512,
            max_horizon=3,
            normalize_inputs=True,
            infer_is_positive=True,  # transaction counts are non-negative
        )
    )

    print("Executing batch inference...")
    ts_data = [np.array(x, dtype=np.float32) for x in df["monthly_txn_history"].to_list()]

    # forecast() in 2.5: no freq argument
    point_forecast, _ = model.forecast(horizon=3, inputs=ts_data)

    # Sum predicted Nov + Dec + Jan (3-month horizon)
    timesfm_3m = np.sum(point_forecast[:, :3], axis=1)

    out = (
        df
        .with_columns(pl.Series("timesfm_3m_forecast", timesfm_3m))
        .drop("monthly_txn_history")
    )

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/customer_features_with_timesfm.parquet"
    out.write_parquet(out_path)
    print(f"TimesFM 2.5 features written → {out_path}")


if __name__ == "__main__":
    generate_timesfm_features()
