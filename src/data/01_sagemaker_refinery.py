"""
Phase 2 | SageMaker Studio Lab (CPU)
Aggregates 18M transaction rows into a customer-level parquet and uploads to HF.
Env vars required: HF_TOKEN, HF_REPO, DATA_DIR (optional, defaults to data/raw)
"""
import os
import polars as pl
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def process_and_upload():
    data_dir = os.environ.get("DATA_DIR", "data/raw")
    print(f"Initiating 18M row aggregation from {data_dir}...")

    txns = pl.scan_parquet(f"{data_dir}/transactions_features.parquet").with_columns(
        pl.col("TransactionDate").dt.truncate("1mo").cast(pl.Date).alias("Month_Start")
    )

    months = pl.DataFrame({
        "Month_Start": pl.date_range(
            datetime(2012, 12, 1), datetime(2015, 10, 1), "1mo", eager=True
        ).cast(pl.Date)
    }).lazy()

    dense_grid = txns.select("UniqueID").unique().join(months, how="cross")

    actual_monthly = txns.group_by(["UniqueID", "Month_Start"]).agg([
        pl.len().alias("txn_count"),
        pl.col("TransactionAmount").sum().alias("monthly_spend"),
    ])

    full_history = (
        dense_grid
        .join(actual_monthly, on=["UniqueID", "Month_Start"], how="left")
        .with_columns([
            pl.col("txn_count").fill_null(0),
            pl.col("monthly_spend").fill_null(0.0),
        ])
        .sort(["UniqueID", "Month_Start"])
    )

    customer_features = full_history.group_by("UniqueID").agg([
        pl.col("txn_count").alias("monthly_txn_history"),
        pl.col("txn_count").sum().alias("total_historical_txns"),
        pl.col("txn_count").last().alias("last_month_txns"),
        # Holiday anchor: Nov/Dec/Jan transactions across all history years
        pl.col("txn_count")
          .filter(pl.col("Month_Start").dt.month().is_in([11, 12, 1]))
          .sum().alias("historical_holiday_txns"),
    ])

    # README: 567 customers have no financials — left join + fill_null preserves them
    fin_features = pl.scan_parquet(f"{data_dir}/financials_features.parquet").group_by("UniqueID").agg([
        pl.col("NetInterestIncome").mean().alias("avg_net_interest_income"),
        pl.col("NetInterestIncome").std().alias("volatility_interest_income"),
        pl.col("NetInterestRevenue").mean().alias("avg_net_interest_revenue"),
    ])

    final_df = (
        customer_features
        .join(fin_features, on="UniqueID", how="left")
        .fill_null(0.0)
        .collect()
    )

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/customer_base_features.parquet"
    final_df.write_parquet(out_path)
    print(f"Wrote {len(final_df)} rows → {out_path}")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj=out_path,
        path_in_repo="customer_base_features.parquet",
        repo_id=os.environ["HF_REPO"],
        repo_type="dataset",
    )
    print("Refinery complete. Parquet shipped to Hugging Face.")


if __name__ == "__main__":
    process_and_upload()
