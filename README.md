# Zindi Nedbank Transaction Volume Forecasting
## Two-Stage Hurdle Model + TimesFM 2.5

### Architecture
```
Raw CSVs (18M rows)
    │
    ▼ Phase 2 — SageMaker Studio Lab (CPU)
01_sagemaker_refinery.py   →  customer_base_features.parquet  →  HuggingFace Dataset
    │
    ▼ Phase 3 — Google Colab (T4 GPU)
02_timesfm_features.py     →  customer_features_with_timesfm.parquet
    │
    ▼ Phase 4 — Google Colab (T4 GPU)
03_hurdle_train.py         →  clf.pkl + reg.pkl  (prints validation RMSLE)
    │
    ▼ Phase 5 — Google Colab (T4 GPU)
04_submission.py           →  submissions/final_submission.csv
```

### Environment Setup (local / Codespace)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Required Secrets
| Variable | Where to set |
|---|---|
| `HF_TOKEN` | HuggingFace → Settings → Access Tokens |
| `HF_REPO`  | Your HF dataset repo, e.g. `username/nedbank-features` |

Set them in SageMaker and Colab via:
```python
import os
os.environ["HF_TOKEN"] = "hf_..."
os.environ["HF_REPO"]  = "username/nedbank-features"
```

### Execution Order

**Phase 2 — SageMaker Studio Lab (CPU, free tier)**
```bash
# 1. Clone repo, activate venv, install requirements
# 2. Download competition data
zindi download -c nedbank-transaction-volume-forecasting-challenge
mv *.csv data/raw/
unzip transactions_features.zip -d data/raw/
unzip financials_features.zip -d data/raw/

# 3. Run refinery (aggregates 18M rows → parquet → HF upload)
python src/data/01_sagemaker_refinery.py

# 4. Shut down instance immediately to preserve free hours
```

**Phase 3 & 4 — Google Colab (T4 GPU)**
```python
# Install extra deps
!pip install polars huggingface_hub git+https://github.com/google-research/timesfm.git

# Run in order
!python src/features/02_timesfm_features.py
!python src/models/03_hurdle_train.py   # target: RMSLE < 0.35
!python src/models/04_submission.py
```

**Phase 5 — Submit**
1. Download `submissions/final_submission.csv` from Colab
2. Upload to [Zindi leaderboard](https://zindi.africa/competitions/nedbank-transaction-volume-forecasting-challenge)
3. Push all scripts back to GitHub for reproducibility review

### Model Details
- **Stage 1**: LightGBM classifier — predicts P(customer makes ≥1 transaction in next 3 months)
- **Stage 2**: XGBoost regressor — predicts log1p(volume) conditioned on active customers
- **Ensemble**: `prediction = P(active) × expm1(log_volume)`
- **TimesFM 2.5**: Zero-shot 3-month horizon forecast used as a meta-feature
- **Submission format**: Zindi expects `log1p(prediction)`
