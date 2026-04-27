"""
Phase 2 | SageMaker Studio Lab (CPU)
Aggregates 18M transaction rows into a customer-level parquet and uploads to HF.
Env vars required: HF_TOKEN, HF_REPO
"""
import os
import polars as pl
from datetime import datetime
from huggingface_hub import HfApi


def process_and_upload():
    print("Initiating 18M row aggregation...")

    txns = pl.scan_csv("data/raw/transactions_features.csv").with_columns(
        pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
          .dt.truncate("1mo").alias("Month_Start")
    )

    months = pl.DataFrame({
        "Month_Start": pl.date_range(
            datetime(2012, 12, 1), datetime(2015, 10, 1), "1mo", eager=True
        )
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
        pl.col("txn_count")
          .filter(pl.col("Month_Start").dt.month().is_in([11, 12, 1]))
          .sum().alias("historical_holiday_txns"),
    ])

    fin_features = pl.scan_csv("data/raw/financials_features.csv").group_by("UniqueID").agg([
        pl.col("NetInterestIncome").mean().alias("avg_net_interest_income"),
        pl.col("NetInterestIncome").var().alias("volatility_interest_income"),
        (pl.col("TransactionalRevenue").sum() / (pl.col("InvestmentsRevenue").sum() + 1.0))
          .alias("liquidity_preference_ratio"),
    ])

    final_df = customer_features.join(fin_features, on="UniqueID", how="left").collect()

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
