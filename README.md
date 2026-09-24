# CG-DBNet

**Context-Guided Dual-Branch Network for Sea Surface Temperature Prediction and Quality Control**

## Overview

CG-DBNet is a deep learning framework for **ocean time-series prediction** and **data quality control (QC)**. It integrates the following core components:

- **Multi-Scale Atrous Convolution (MSAC)**: parallel convolutional branches with kernel sizes 3/5/7 to capture local multi-scale temporal patterns;
- **Dual-Attention Global Average Pooling (DAGAP)**: channel and temporal attention to model long-range dependencies and feature importance;
- **Cyclic Positional Encoding**: daily and annual periodic components to capture seasonal rhythms of ocean variables;
- **Reversible Instance Normalization (RevIN)**: instance-level mean/variance normalization with inverse transform at inference to mitigate distribution shift;
- **Adaptive Affine Fusion**: learnable affine combination (weight * branch + bias) of the two branches; degenerates to simple addition in ablation.

The model supports two tasks:

1. **SST / ocean variable prediction** (single-value regression, output mean mu);
2. **Quality control (QC)**: anomaly detection and repair, output Gaussian distribution (mu, sigma), trained with Gaussian NLL loss, with sequential QC using mu +/- k*sigma prediction intervals.

Baseline models (LSTM / GRU / TCN / Transformer / XGBoost) and data processing scripts (PCA, correlation analysis, synthetic anomaly injection) are included.

## Environment

- Python environment: conda base (Python 3.x), with `torch 2.5.1`, `numpy 1.26.4`, `pandas 2.2.3` pre-installed.
- Install remaining dependencies:

```bash
pip install -r requirements.txt
```

> Note: XGBoost is **optional**. If `xgboost` is not installed, `--model xgboost` automatically falls back to `sklearn.ensemble.GradientBoostingRegressor`.

## Directory Structure

```
CG-DBNet/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── model.py              # CGDBNetBackbone, PredictionNet, QCNet, RevIN,
│   │                         #   CyclicPositionalEncoding, AdaptiveAffineFusion, GaussianHead
│   ├── data_utils.py         # DataProcessor, MultiFileTimeSeriesDataset,
│   │                         #   lat/lon filename parsing, feature engineering, normalization
│   └── metrics.py            # MAE/RMSE/R2, Gaussian NLL, dynamic k-sigma calibration, classification metrics
├── experiments/
│   ├── train_prediction.py   # Prediction experiment entry (multi-file grid-point data)
│   ├── train_qc.py           # QC experiment entry (single Excel with multiple sheets)
│   └── baselines/            # GRU, LSTM, TCN, Transformer, XGBoost
└── data_processing/
    ├── pca_analysis.py       # PCA principal component analysis
    ├── correlation_analysis.py  # Pearson correlation heatmap
    └── synthetic_anomaly.py  # Synthetic anomaly injection for QC test data
```

## Data Preparation

### 1. Prediction Experiment (Multi-File)

`experiments/train_prediction.py` reads a **directory** containing multiple Excel (`.xlsx`) or CSV (`.csv`) files, each corresponding to one grid point / station. Requirements:

- Each file must contain:
  - **Time column** (default name `time`, modifiable via `--time_col`);
  - **Target column** (default name `sst`, modifiable via `--target_col`);
  - **Optional external meteorological feature columns** (e.g., air temperature, sea level pressure; default `--feature_cols air_temperature sea_level_pressure`, pass empty to disable).
- The program parses **latitude/longitude from filenames** and automatically generates geographic features (lat/lon values and cyclic encodings). Filenames should contain recognizable lat/lon information.
- External test data can be specified via `--test_path` (default `./data/test.xlsx`).

### 2. QC Experiment (Single File, Multiple Sheets)

`experiments/train_qc.py` reads a **single Excel file** specified via `--data_file` (required), containing three sheets:

| Sheet (default name) | Content |
| --- | --- |
| `train` | Normal observations for training |
| `val` | Normal observations for hyperparameter selection / calibration |
| `test` | Contains anomaly label column (0/1) for evaluation |

Column requirements:

- **Date column** (default `date`, modifiable via `--time_col`);
- **Target column** (default `sst`, modifiable via `--target_col`);
- Optional feature columns (`--feature_cols`, default empty);
- **Label column**: binary anomaly labels in the test sheet.

### 3. Data Processing Scripts

The `data_processing/` directory provides three utility scripts:

- `pca_analysis.py`: PCA on station features, outputs scree plots, loadings, and scores;
- `correlation_analysis.py`: computes Pearson correlation matrix and heatmap;
- `synthetic_anomaly.py`: injects multiple anomaly types (out-of-range, gradient shift, spike, Gaussian noise) into clean sequences to generate labeled QC datasets.

## Usage

> All commands are executed from the **package root directory** (where this README resides). All paths are passed via command-line arguments; no hardcoded absolute paths.

### Prediction Experiment

```bash
# Basic run (full CG-DBNet model)
python experiments/train_prediction.py --data_dir ./data --output_dir ./outputs/prediction

# Specify feature columns and hyperparameters
python experiments/train_prediction.py --data_dir ./data --feature_cols air_temperature sea_level_pressure --seq_len 48 --epochs 50 --batch_size 64

# Ablation studies (disable components)
python experiments/train_prediction.py --data_dir ./data --no_msac        # remove MSAC branch
python experiments/train_prediction.py --data_dir ./data --no_gap         # remove DAGAP branch
python experiments/train_prediction.py --data_dir ./data --no_pos         # remove cyclic positional encoding
python experiments/train_prediction.py --data_dir ./data --no_affine      # affine fusion -> addition
python experiments/train_prediction.py --data_dir ./data --serial_gru     # parallel -> serial GRU

# Baseline models
python experiments/train_prediction.py --data_dir ./data --model lstm
python experiments/train_prediction.py --data_dir ./data --model gru
python experiments/train_prediction.py --data_dir ./data --model tcn
python experiments/train_prediction.py --data_dir ./data --model transformer
python experiments/train_prediction.py --data_dir ./data --model xgboost
```

Common optional parameters: `--target_col` (default `sst`), `--time_col` (default `time`), `--test_path` (default `./data/test.xlsx`), `--lr` (default 1e-3), `--hidden_dim` (default 64), `--dropout` (default 0.15), `--patience` (default 15). Additional `--no_conv3` / `--no_conv5` / `--no_conv7` flags can individually disable corresponding MSAC kernel sizes.

### QC Experiment

```bash
# Basic run (--data_file is required)
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --output_dir ./outputs/qc

# Specify sheets and hyperparameters
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --train_sheet train --eval_sheet val --test_sheet test --seq_len 48 --epochs 200 --target_coverage 0.95

# Ablation variants
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_no_msac
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_no_dagap
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_no_pos
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_affine_add
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_serial_gru
```

Common optional parameters: `--time_col` (default `date`), `--target_col` (default `sst`), `--feature_cols` (default empty), `--eval_val_ratio` (default 0.5), `--lr` (default 1e-3), `--hidden_dim` (default 64), `--dropout` (default 0.15), `--patience` (default 30). Baseline comparison uses `--model lstm|gru|tcn|transformer|xgboost`.

### Data Processing

```bash
python data_processing/pca_analysis.py --input ./data --output ./outputs/pca
python data_processing/correlation_analysis.py --input ./data --output ./outputs/corr
python data_processing/synthetic_anomaly.py --input ./data/raw.xlsx --output ./data/qc_data.xlsx
```

`synthetic_anomaly.py` supports `--sheet` (default `test`), `--seed` (default 42), and various anomaly ratio/magnitude parameters (`--ratio-out-of-range`, `--ratio-gradient`, `--ratio-spike`, `--ratio-gaussian`, etc.). See `--help` for details.

## Outputs

### Prediction Experiment (`--output_dir`, default `./outputs/prediction`)

- Model checkpoint (`.pth`);
- Training/validation loss curves (`loss_history.json` and `.png`);
- Evaluation metrics (`metrics.json`): **MAE / RMSE / R2**;
- Per-timestep prediction CSV (predicted vs. actual).

### QC Experiment (`--output_dir`, default `./outputs/qc`)

- Model checkpoint;
- Gaussian NLL loss curve;
- **k-sigma calibration** results (determined by `--target_coverage` (default 0.95));
- **Sequential QC results** on test set (per-sample anomaly flag, repaired values);
- Anomaly classification metrics: **F1 / Precision / Recall / Accuracy**;
- QC-processed (repaired/flagged) data file.

## Model Architecture

```
Input sequence
    │
    ▼
RevIN (instance-level mean/variance normalization, inverse at inference)
    │
    ▼
CyclicPositionalEncoding (daily/annual periodic injection)
    │
    ├──► Branch A: MSAC (kernels 3/5/7 parallel -> local multi-scale features)
    │
    └──► Branch B: DAGAP (channel/temporal attention -> global aggregated features)
    │
    ▼
AdaptiveAffineFusion (learnable weight/bias; addition in ablation)
    │
    ▼
Task heads
     ├─ Prediction head: single-point mean mu (regression)
     └─ QC head: Gaussian (mu, sigma), NLL training,
                 mu +/- k*sigma intervals for sequential QC
```

- `CGDBNetBackbone` in `src/model.py` is the shared encoder; `PredictionNet` and `QCNet` are task-specific wrappers;
- `build_pred_model(...)` / `build_qc_model(...)` construct models with ablation flags;
- The QC head outputs a heteroscedastic Gaussian distribution, enabling both point prediction and uncertainty estimation for anomaly detection and repair.

## Notes

- **All paths are parameterized**: input/output paths are specified via command-line arguments; no hardcoded absolute paths in the code.
- **XGBoost is optional**: falls back to sklearn GradientBoostingRegressor if not installed.
- **Reproducibility**: data processing scripts support `--seed` (default 42) for synthetic anomaly generation.
