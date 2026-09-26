# CG-DBNet

**Context-Guided Dual-Branch Network for Sea Surface Temperature Prediction and Quality Control**

A PyTorch framework for **ocean time-series prediction** and **data quality control (QC)** on
sea surface temperature (SST) observations, together with baseline models and data-processing
utilities. Every input and output path is supplied through command-line arguments.

## Overview

The shared encoder combines the following components:

- **Multi-Scale Atrous Convolution (MSAC)**: parallel atrous convolutions with kernel sizes 3/5/7
  (dilation 2) whose outputs are concatenated, normalized and projected back to `hidden_dim` by a
  1x1 convolution, to capture local multi-scale temporal patterns.
- **Dual-Attention Global Average Pooling (DAGAP)**: two sigmoid gates are computed from the pooled
  input embedding - a feature-dimension gate (mean over time) and a temporal-dimension gate
  (mean over features) - and the aggregated representation is the element-wise product
  `a_global = w_feat * w_time * x_emb`.
- **Weight Generation**: two small MLPs read the fused representation and emit a temporal weight per
  sequence step and a feature weight per hidden channel; these gate the input and the output of the
  recurrent layer.
- **Cyclic Positional Encoding**: a sine/cosine temporal prior over the sequence positions, added to
  the projected features through a learnable per-channel weight. The daily and annual cyclic inputs
  (`hour_sin`/`hour_cos` from the hour of day, `day_sin`/`day_cos` from the day of year) are part of
  the input feature engineering in `src/data_utils.py`, not part of this positional encoding.
- **Reversible Instance Normalization (RevIN)**: per-window mean/std normalization of the temporal
  channels only; geographic (spatial) channels are concatenated unchanged, and the predicted mean is
  rescaled by the instance standard deviation of the target channel. The QC network instead receives
  a `StandardScaler` fitted once on the training split.
- **Adaptive Affine Fusion**: the additive combination (`local + global`) and the multiplicative
  combination (`local * global`) are computed together and merged into a normalized weighted sum,
  where the two weights come from two learnable scalars passed through sigmoid. Disabling the module
  (`--no_affine` for prediction, `--model abl_affine_add` for QC) keeps only the additive path.

Two tasks are supported:

1. **SST prediction** - a Gaussian head emits `mu` and `sigma`; the training objective is MSE on
   `mu`, where `mu` is the one-step difference between the next target value and the last observed
   value in the window. Results are reported for the in-grid test split and, when external test
   data is supplied, for the external test split.
2. **Quality control (QC)** - the same head emits a heteroscedastic Gaussian (`mu`, `sigma`) trained
   with the Gaussian negative log-likelihood. A sequential QC loop flags every observation outside
   `mu +/- k * sigma` as anomalous, where `k` is calibrated on the validation split, and repairs
   flagged values with `mu` before recomputing the features of subsequent steps.

Baselines: **LSTM / GRU / TCN / Transformer / XGBoost**, selectable through the two experiment entry
scripts and also provided as standalone runnable prediction scripts under `experiments/baselines/`.

## Environment

- Tested environment: conda base (Python 3.12, PyTorch 2.5.1, CUDA 12.1).

```bash
pip install -r requirements.txt
```

`requirements.txt` pins every dependency to a tested range (`torch>=2.2,<3`, `numpy>=1.24,<2`,
`pandas>=2.0,<3`, `openpyxl>=3.0,<4`, `scikit-learn>=1.0,<2`, `scipy>=1.10,<2`,
`matplotlib>=3.5,<4`, `seaborn>=0.12,<1`, `tqdm>=4.60,<5`, `xgboost>=2.0,<3`).

> **XGBoost is a required dependency.** The `--model xgboost` path of both experiment entry points
> imports `xgboost` directly and raises a `RuntimeError` when it cannot be imported, so those runs
> require an environment installed from `requirements.txt`.

Figures produced by the two experiment entry points use the `DejaVu Sans` font, and all in-figure
text is English. The utilities in `data_processing/` apply their own journal style (serif face,
Times New Roman) and likewise label their figures in English.

## Configuration checklist

Every item below has a working default, so the package runs out of the box on data that uses the
default naming and column scheme. Each one still has to be reviewed when the inputs are different.

| What | Default | How to change it |
| --- | --- | --- |
| Series column names | prediction `--time_col time`, `--target_col sst`, `--feature_cols air_temperature sea_level_pressure`; QC `--time_col date`, `--target_col sst`, `--feature_cols` empty | pass the flags, or rename the columns in your files |
| Grid-point file names | built-in Latin rule `(lat|latitude)...(lon|longitude)`, which also supplies the coordinates | `--file_pattern` with **exactly two capture groups**, the first the latitude and the second the longitude |
| Latitude / longitude source | column aliases `lat`/`latitude` and `lon`/`longitude` (case-insensitive), otherwise the two capture groups above | provide either one; a station with neither is reported and skipped |
| Non-Latin file names or column headers | not recognised by default | supply `--file_pattern` and the matching `--time_col` / `--target_col` / `--feature_cols` values |
| QC workbook layout | sheets `train`, `val`, `test`; anomaly column `label` | `--train_sheet`, `--eval_sheet`, `--test_sheet`, `--label_col` |
| External evaluation data | `--test_path ./data/test.xlsx`; when it resolves to nothing, the DatasetB outputs are simply not written | point it at a single file or a directory of stations |
| Study-region bounds | `LAT_MIN/LAT_MAX = 28.0/32.0`, `LON_MIN/LON_MAX = 122.0/126.0`, seven `geo_*` features | class attributes in the entry scripts (no flag); edit them for another basin |
| Split proportions | prediction `TRAIN_RATIO 0.7` / `VAL_RATIO 0.15`; QC validation share `--eval_val_ratio 0.5` | class attributes / the QC flag |
| Random seed | `SEED = 42` in both entry scripts | edit the constant; the standalone baselines expose `--seed` instead |
| Optimiser internals | gradient clipping 1.0, `SIGMA_MIN 1e-4`, QC `MAX_CONSECUTIVE_REPLACE 2`, `ReduceLROnPlateau(patience=5, factor=0.5)` | class attributes only, no flags |
| Compute device | CUDA when available, otherwise CPU | no `--device` flag; control it with `CUDA_VISIBLE_DEVICES` |
| k-sigma threshold | `k` is the `--target_coverage` quantile of the validation standardized residuals, default 0.95 | `--target_coverage`; there is intentionally no offset parameter |
| Feature-engineering cache | `_pred_data_cache_<hash>.pt` inside the output directory, keyed by the feature-column list | delete it, or choose a different `--output_dir`, to force a rebuild |
| In-figure units | ASCII spellings such as `degC` rather than the degree sign, so figures stay pure ASCII | edit the label maps in `data_processing/` if you need other glyphs |

The shell matters for non-ASCII values: under some terminals (for example Git Bash on Windows)
arguments containing non-ASCII characters are re-encoded and arrive corrupted. If your column names
or file names are not Latin, prefer a native terminal (cmd or PowerShell) or a small launcher script
that keeps those values inside the Python process.

## Directory Structure

```
CG-DBNet/
|-- README.md
|-- LICENSE                     # MIT license
|-- requirements.txt
|-- src/
|   |-- __init__.py
|   |-- model.py                # CGDBNetBackbone, PredictionNet, QCNet, RevIN,
|   |                           #   CyclicPositionalEncoding, AdaptiveAffineFusion, GaussianHead,
|   |                           #   LSTMModel, GRUModel, TCNModel, TransformerModel, RevINBaseline,
|   |                           #   build_pred_model, build_qc_model
|   |-- data_utils.py           # DataProcessor, MultiFileTimeSeriesDataset, grid-file naming,
|   |                           #   feature engineering, spatial features, QC sheet pipeline,
|   |                           #   SeqDataset, build_feature_matrix
|   |-- metrics.py              # MAE/MSE/RMSE/MAPE/SMAPE/R2/MASE, classification metrics,
|   |                           #   Gaussian NLL, dynamic k-sigma calibration
|-- experiments/
|   |-- train_prediction.py     # prediction entry point (many grid-point files)
|   |-- train_qc.py             # QC entry point (one Excel workbook, train/val/test sheets)
|   `-- baselines/              # standalone runnable baseline scripts
|       |-- gru.py              #   python experiments/baselines/gru.py --data_dir ... --output_dir ...
|       |-- lstm.py
|       |-- tcn.py
|       |-- transformer.py
|       `-- xgboost.py
`-- data_processing/
    |-- pca_analysis.py         # PCA of multi-station features
    |-- correlation_analysis.py # Pearson correlation matrix and heatmaps
    `-- synthetic_anomaly.py    # labeled anomaly injection for QC test data
```

## Data Preparation

### 1. Prediction experiment (many files, one per grid point)

`experiments/train_prediction.py` reads a **directory** of Excel (`.xlsx`, `.xls`) or CSV (`.csv`)
files, one file per grid point or station. Each file must provide:

| Content | Default column name | Command-line override |
| --- | --- | --- |
| Timestamp column | `time` | `--time_col` |
| Target column | `sst` | `--target_col` |
| Optional external meteorological columns | `air_temperature`, `sea_level_pressure` | `--feature_cols` (pass none to disable) |

Additional requirements:

- **Latitude/longitude** are taken from the data columns when present (`lat` / `latitude` and
  `lon` / `longitude`, matched case-insensitively), and otherwise parsed from the file name. Seven
  geographic features are generated automatically: `geo_lat_sin`, `geo_lat_cos`, `geo_lon_sin`,
  `geo_lon_cos`, `geo_lat_norm`, `geo_lon_norm`, `geo_dist_center`.
- **Grid-file naming**: only files whose name matches a configurable regular expression are loaded.
  The default expression recognizes Latin-letter naming with **two capture groups, latitude first and
  longitude second**, for example `lat29.0_lon123.5.xlsx`, `lat_29.0_lon_123.5.csv`,
  `Lat=29.0_Lon=123.5.csv` or `latitude29.0longitude123.5.xlsx`. If your files name latitude and
  longitude with other words - for instance local-language abbreviations - pass a custom expression
  through `--file_pattern`; the two capture groups must still be latitude first and longitude
  second:

  ```bash
  # files named like "p29.0_q123.5.xlsx"
  python experiments/train_prediction.py --data_dir ./data --file_pattern 'p(\-?\d+\.?\d*)_q(\-?\d+\.?\d*)'
  ```

- Each file is split chronologically into train / validation / test with the `TRAIN_RATIO` and
  `VAL_RATIO` constants of `PredConfig` (0.7 and 0.15, that is 70/15/15); files that are too short to
  fill all three segments after feature engineering are skipped with a message.
- The region bounds used to normalize `geo_lat_norm`, `geo_lon_norm` and `geo_dist_center` are the
  `LAT_MIN` / `LAT_MAX` / `LON_MIN` / `LON_MAX` constants of `PredConfig` in
  `experiments/train_prediction.py` (28.0-32.0 N, 122.0-126.0 E); edit them in the file if your study
  region differs.
- Derived input features are generated automatically: target `lag1`, `lag24`, `diff1`, `diff2`,
  rolling means over 6 and 24 steps; for every external feature column `lag1`-`lag3`, `diff1`,
  `diff2`; and the cyclic time features `hour_sin`, `hour_cos`, `day_sin`, `day_cos`.
- **External test data** is read from `--test_path` (default `./data/test.xlsx`), which accepts a
  single file or a directory of files. When it is absent, only the in-grid test set is evaluated.

### 2. QC experiment (single workbook, three sheets)

`experiments/train_qc.py` reads a **single Excel workbook** given by `--data_file` (required):

| Sheet | Default name | Command-line override | Content |
| --- | --- | --- | --- |
| Training | `train` | `--train_sheet` | Normal observations |
| Evaluation pool | `val` | `--eval_sheet` | Normal observations; its first `--eval_val_ratio` fraction (default 0.5) forms the validation split used for early stopping and k-sigma calibration, while the QC test itself always comes from `--test_sheet` |
| QC test | `test` | `--test_sheet` | The sequence the sequential QC loop runs on, with the anomaly label column |

| Content | Default column name | Command-line override |
| --- | --- | --- |
| Timestamp column | `date` | `--time_col` |
| Target column | `sst` | `--target_col` |
| Optional feature columns | none | `--feature_cols` |
| Binary anomaly labels (test sheet) | `label` | `--label_col` |

Any non-zero label value is treated as an anomaly, so multi-code labels produced by
`data_processing/synthetic_anomaly.py` can be used directly. The workbook produced by that script
(sheets `train`, `val`, `test`) is a valid `--data_file`. Derived features are generated as in the
prediction pipeline (cyclic hour/day features, target `lag1`/`lag24`/rolling means, feature lags and
first differences) and the whole matrix is standardized with `StandardScaler`.

### 3. Data-processing scripts

- `pca_analysis.py`: PCA over stations, time-aligned and regionally averaged; writes scree plot,
  loadings and scores figures plus CSV tables.
- `correlation_analysis.py`: Pearson correlation and p-value matrices with heatmap figures.
- `synthetic_anomaly.py`: injects four anomaly types into a clean series - out-of-range, gradient
  shift, spike, Gaussian - and writes a three-sheet workbook with a `label` column
  (0 = normal, 1 = out-of-range, 2 = gradient, 3 = spike, 4 = Gaussian).

## Usage

> All commands are executed from the **package root directory** (where this README resides).

### Prediction experiment

```bash
# Full CG-DBNet model
python experiments/train_prediction.py --data_dir ./data --output_dir ./outputs/prediction

# Feature columns and hyper-parameters
python experiments/train_prediction.py --data_dir ./data --feature_cols air_temperature sea_level_pressure \
    --seq_len 48 --epochs 50 --batch_size 64 --lr 1e-3

# Component ablations
python experiments/train_prediction.py --data_dir ./data --no_msac     # bypass the atrous conv stack
python experiments/train_prediction.py --data_dir ./data --no_gap      # bypass the GRU weighting
python experiments/train_prediction.py --data_dir ./data --no_pos      # bypass the positional prior
python experiments/train_prediction.py --data_dir ./data --no_affine   # fusion -> additive only
python experiments/train_prediction.py --data_dir ./data --serial_gru  # GRU directly on the projection

# MSAC kernel-size subsets (combine freely)
python experiments/train_prediction.py --data_dir ./data --no_conv3
python experiments/train_prediction.py --data_dir ./data --no_conv5
python experiments/train_prediction.py --data_dir ./data --no_conv7

# Baselines through the shared entry point
python experiments/train_prediction.py --data_dir ./data --model lstm
python experiments/train_prediction.py --data_dir ./data --model gru
python experiments/train_prediction.py --data_dir ./data --model tcn
python experiments/train_prediction.py --data_dir ./data --model transformer
python experiments/train_prediction.py --data_dir ./data --model xgboost
```

Arguments of `experiments/train_prediction.py` and their defaults: `--data_dir ./data`,
`--test_path ./data/test.xlsx`, `--file_pattern` (defaults to the Latin-letter naming rule above),
`--output_dir ./outputs/prediction`, `--model proposed` (`proposed|lstm|gru|tcn|transformer|xgboost`),
`--target_col sst`, `--time_col time`, `--feature_cols air_temperature sea_level_pressure`,
`--seq_len 48`, `--batch_size 64`, `--epochs 50`, `--lr 0.001`, `--hidden_dim 64`,
`--dropout 0.15`, `--patience 15`, and the store-true ablation flags `--no_pos`, `--no_msac`,
`--no_gap`, `--no_affine`, `--serial_gru`, `--no_conv3`, `--no_conv5`, `--no_conv7`.

### QC experiment

```bash
# Full CG-DBNet model (--data_file is required)
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --output_dir ./outputs/qc

# Sheets, features and hyper-parameters
python experiments/train_qc.py --data_file ./data/qc_data.xlsx \
    --train_sheet train --eval_sheet val --test_sheet test --label_col label \
    --seq_len 48 --epochs 200 --target_coverage 0.95

# Component ablations
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_no_msac
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_no_dagap
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_no_pos
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_affine_add
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_serial_gru

# Baseline comparison
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model gru
```

Arguments of `experiments/train_qc.py`: `--data_file` (required), `--output_dir ./outputs/qc`,
`--train_sheet train`, `--eval_sheet val`, `--test_sheet test`, `--time_col date`,
`--target_col sst`, `--feature_cols` (default empty), `--label_col label`, `--eval_val_ratio 0.5`,
`--seq_len 48`, `--batch_size 64`, `--epochs 200`, `--lr 0.001`, `--hidden_dim 64`,
`--dropout 0.15`, `--patience 30`, `--target_coverage 0.95`. `--model` defaults to `proposed` and
accepts `lstm`, `gru`, `tcn`, `transformer`, `xgboost` and the ablation variants listed below.
`--patience` and `--target_coverage` fall back to the values of `QcConfig` (30 and 0.95) when they
are not passed.

**k-sigma calibration.** Let `z = |actual - mu| / sigma` on the validation split. The threshold `k`
is exactly the `--target_coverage` quantile of `z` - no additional offset is added - and the
empirical coverage reached on validation is stored next to it. For `--model xgboost`, which emits no
`sigma`, the corresponding threshold is the `--target_coverage` quantile of the validation absolute
residuals in physical units.

**Sequential QC loop.** Starting at index `seq_len`, the model predicts the next step, the
observation is flagged when it falls outside `mu +/- k * sigma`, and flagged values are replaced by
`mu` while rebuilding the features of the following steps. At most two consecutive replacements are
applied (`MAX_CONSECUTIVE_REPLACE` in `QcConfig`).

### Ablation switches

| Prediction flag | QC `--model` value | Effect on the network |
| --- | --- | --- |
| `--no_msac` | `abl_no_msac` | the atrous convolution stack is bypassed; the position-encoded projection feeds the local gating and the fusion directly |
| `--no_gap` | `abl_no_dagap` | the temporal/feature weighting of the recurrent layer is disabled: the GRU receives the projected embedding and its output is used unweighted |
| `--no_pos` | `abl_no_pos` | the sine/cosine temporal prior is not added |
| `--no_affine` | `abl_affine_add` | fusion keeps only the additive combination of the local and global representations |
| `--serial_gru` | `abl_serial_gru` | the context-guided branch is skipped entirely and the GRU runs on the projected input |
| `--no_conv3`, `--no_conv5`, `--no_conv7` | not exposed on the QC command line | drop the corresponding kernel from the MSAC concatenation; the normalization and the 1x1 projection adapt to the number of remaining branches |

The prediction flags are independent and can be combined.

### Standalone baseline scripts

Each baseline under `experiments/baselines/` is also a self-contained entry point, so a single
baseline can be run without touching the CG-DBNet configuration:

```bash
python experiments/baselines/gru.py --data_dir ./data --output_dir ./outputs/baselines/gru --epochs 50
python experiments/baselines/lstm.py --data_dir ./data --output_dir ./outputs/baselines/lstm
python experiments/baselines/tcn.py --data_dir ./data --output_dir ./outputs/baselines/tcn
python experiments/baselines/transformer.py --data_dir ./data --output_dir ./outputs/baselines/transformer
python experiments/baselines/xgboost.py --data_dir ./data --output_dir ./outputs/baselines/xgboost
```

Every script reads the same multi-file grid data through `DataProcessor` and accepts the shared data
arguments `--data_dir ./data`, `--test_path ./data/test.xlsx`, `--file_pattern`,
`--target_col sst`, `--time_col time`, `--feature_cols air_temperature sea_level_pressure`,
`--seq_len 48`, plus `--output_dir` (default `./outputs/baselines/<name>`). On top of that:

| Script | Additional arguments |
| --- | --- |
| `gru.py`, `lstm.py`, `tcn.py`, `transformer.py` | `--epochs 50`, `--batch_size 64`, `--lr 0.001`, `--hidden_dim 64`, `--dropout 0.15`, `--patience 15`, `--seed 42`, `--use_revin` |
| `xgboost.py` | `--lr 0.05`, `--n_estimators 300`, `--max_depth 6`, `--seed 42` |

Each writes its results into its own `--output_dir`: `training_config.json`,
`predictions_DatasetA_Test.csv` / `metrics_DatasetA_Test.txt` / `prediction_DatasetA_Test.png`, the
matching `DatasetB_External` trio when `--test_path` provides data, and `loss_history.txt`; the
neural scripts additionally save `best_model.pth`, and `xgboost.py` saves `xgb_model.pkl` and a
`Validation` set of prediction/metric/plot files.

How the two routes relate:

- A standalone script builds the model class defined in its own file (for example `GRUModel` in
  `experiments/baselines/gru.py`), whereas `experiments/train_prediction.py --model gru` builds the
  `GRUModel` defined in `src/model.py`. Both see the same windows, the same differenced target and
  the same training loop structure.
- The prediction entry point always wraps its non-tree baselines in `RevINBaseline`. In the
  standalone scripts that wrapper is selected explicitly with `--use_revin`, which is off by
  default; pass it to reproduce the entry-point setup.
- The tree baseline has no minibatch, hidden-size, dropout or early-stopping options, so `xgboost.py`
  exposes tree hyper-parameters instead.
- `experiments/train_qc.py --model lstm|gru|tcn|transformer|xgboost` uses the classes defined in
  `src/model.py`, since its inputs are already standardized; the standalone scripts cover the
  prediction task only.

### Data processing

```bash
python data_processing/pca_analysis.py --input ./data --output ./outputs/pca
python data_processing/correlation_analysis.py --input ./data --output ./outputs/corr
python data_processing/synthetic_anomaly.py --input ./data/raw.xlsx --output ./data/qc_data.xlsx
```

`pca_analysis.py` and `correlation_analysis.py` take `--input` (directory of station files),
`--output` (directory) and `--pattern` (file glob, default `lat*_lon*.xlsx`).
`synthetic_anomaly.py` takes `--input` (Excel workbook with at least two sheets carrying `date`
and `sst`), `--output` (default: the input name suffixed with `_test_with_anomalies.xlsx`),
`--mode full|plot`, `--sheet` (default `test`, used by `--mode plot`), `--seed` (default 42),
the per-type ratios `--ratio-out-of-range` (0.2), `--ratio-gradient` (0.3), `--ratio-spike` (0.3),
`--ratio-gaussian` (0.2), the bounds `--normal-lower` / `--normal-upper` / `--extreme-lower` /
`--extreme-upper`, the magnitude ranges `--gradient-delta-min` / `--gradient-delta-max` /
`--spike-score-min` / `--spike-score-max` / `--gaussian-mean-scale` / `--gaussian-std-scale`,
`--anomaly-denominator` (default 34), `--anomaly-start-index` (default 49), and the figure options
`--zoom-days` (default 5) and `--zoom-time-range`. Run `--help` for the complete list.

## Outputs

Both experiment scripts write into `--output_dir`, which is created automatically. The
data-processing utilities write into their own `--output` directory.

### Prediction experiment (default `./outputs/prediction`)

| File | Content |
| --- | --- |
| `training_config.json` | Resolved configuration of the run |
| `_pred_data_cache_<hash>.pt` | Processed arrays cache, keyed by the feature-column hash; re-used on the next run |
| `best_model.pth` | Checkpoint at the best validation loss (neural models) |
| `xgb_model.json` | Saved XGBoost model (`--model xgboost`) |
| `loss_history.txt` | Per-epoch train/validation MSE and validation MAE/RMSE/R2/MAPE/SMAPE/MASE, plus the best epoch |
| `predictions_DatasetA_Test.csv` | Actual vs. predicted values on the in-grid test split |
| `metrics_DatasetA_Test.txt` | Regression metrics on the in-grid test split |
| `prediction_DatasetA_Test.png` | Actual vs. predicted series for the in-grid test split |
| `predictions_DatasetB_External.csv` | Actual vs. predicted values on the external test data |
| `metrics_DatasetB_External.txt` | Regression metrics on the external test data |
| `prediction_DatasetB_External.png` | Actual vs. predicted series for the external test data |

The three `DatasetB_External` files are written only when `--test_path` resolves to usable data.

### QC experiment (default `./outputs/qc`)

| File | Content |
| --- | --- |
| `config.json` | Model kind, input dimension and column order, hyper-parameters, column settings |
| `scaler.pkl`, `target_scaler.pkl` | Fitted input and target scalers, for reuse on new data |
| `model.pth` | Checkpoint at the best validation NLL (neural models) |
| `xgb_model.json` | Saved XGBoost model (`--model xgboost`) |
| `loss_history.txt` | Per-epoch train/validation Gaussian NLL and the best epoch |
| `train_log.json` | Loss history, best validation NLL, number of executed epochs |
| `k_sigma_config.json` | `target_coverage`, calibrated `k_sigma` (or the residual threshold and its `residual_quantile` method for XGBoost), validation coverage and sigma statistics |
| `qc_result.csv` | Per-step `Actual`, `Mu`, `Sigma`, `Lower`, `Upper`, `k_i`, `IsAnomaly`, `ReplacedForNextStep`, `CorrectedInputValue` and `TrueLabel` (when a label column is available) |
| `metrics.txt` | `ModelName`, then the calibrated threshold under the key `k_sigma` for the neural-network models and under the key `residual_threshold` for `--model xgboost`, detected/total point counts, replaced count and the classification metrics |
| `qc_plot.png` | Observed series, predicted mean, acceptance band, detected anomalies and true labels |

### Data processing (default `--output .`)

- `pca_analysis.py`: `pca_scree_plot.png/.pdf`, `pca_loadings_bars.png/.pdf`, `pca_scores.png/.pdf`,
  `pca_loadings.csv`, `pca_explained_variance.csv`, `pca_scores.csv`,
  `regional_mean_timeseries.csv`.
- `correlation_analysis.py`: `pearson_correlation_heatmap.png/.pdf`,
  `pearson_correlation_heatmap_compact.png/.pdf`, `pearson_correlation_matrix.csv`,
  `pearson_pvalue_matrix.csv`, `regional_mean_timeseries.csv`.
- `synthetic_anomaly.py`: the output workbook with sheets `train`, `val`, `test`, and
  `anomaly_injection_zoom.png/.pdf` beside it.

## Model Architecture

```
Input window: (batch, seq_len, input_dim)
    |
    v
[1] Normalization
    |   Prediction: RevIN - instance mean/std over the temporal channels, geographic
    |               channels concatenated unchanged, target std kept for the output
    |   QC:         StandardScaler fitted once on the training split
    v
[2] Linear input projection (input_dim -> hidden_dim)  =  x_emb
    |
    v
[3] Cyclic positional encoding
    |   sine/cosine temporal prior, added to x_emb through a learnable per-channel weight;
    |   the prior is applied to all but the last four projected channels, which are
    |   treated as the time-encoding block and passed through unchanged
    v
[4] Context-guided branch
    |   MSAC    : atrous conv 3 / 5 / 7 in parallel -> concat -> LayerNorm -> 1x1 conv
    |             -> gated local features (tanh * sigmoid, with a skip connection)
    |   DAGAP   : w_feat  = sigmoid(mean over time of x_emb)
    |             w_time  = sigmoid(mean over features of x_emb)
    |             a_global = w_feat * w_time * x_emb
    |   Fusion  : Adaptive Affine Fusion of the local and global representations
    |             (normalized sigmoid weights over the additive and multiplicative
    |              paths, LayerNorm, plus a skip connection); additive only when disabled
    |   Weights : time-attention MLP    -> weights over the seq_len dimension
    |             feature-attention MLP -> weights over the hidden_dim dimension
    v
[5] Temporal modeling branch: 2-layer GRU (hidden_dim)
    |   GRU input  = x_emb * temporal weights   (which time steps matter)
    |   GRU output = GRU output * feature weights (which channels matter)
    |   residual path with a learnable sigmoid-scaled weight
    v
[6] Last time step -> Gaussian head
        mu    = Linear(hidden_dim, 1)
        sigma = Softplus(Linear(hidden_dim, 1)) + sigma_min   (sigma_min = 1e-4)
    |
    +--> Prediction: mu is the one-step target difference, trained with MSE
    +--> QC        : (mu, sigma) trained with Gaussian NLL; mu +/- k*sigma is the
                     acceptance interval used by the sequential QC loop
```

About the name: the **two branches** of CG-DBNet are the context-guided branch (step 4, which turns
the auxiliary context - external meteorological columns, geographic features and cyclic time
features - into gating weights) and the temporal modeling branch (step 5, the GRU that produces the
forecast representation). MSAC, DAGAP, Adaptive Affine Fusion and Weight Generation are **modules
inside the context-guided branch**, not separate branches.

Implementation notes:

- `CGDBNetBackbone` in `src/model.py` is the encoder shared by both tasks; `PredictionNet` adds
  RevIN and rescales `mu` by the instance target standard deviation, `QCNet` keeps `mu` and `sigma`
  in the standardized space.
- `build_pred_model(...)` and `build_qc_model(...)` assemble the model for the requested `--model`
  value and the ablation flags.
- `PredictionNet` and `QCNet` always expose the Gaussian head, so point prediction and uncertainty
  estimation come from one forward pass; the QC task uses both outputs, the prediction task optimizes
  and reports `mu`.

## Metrics

Regression (written to `metrics_*.txt` and to `loss_history.txt`):

- **MAE** - mean absolute error; **MSE** - mean squared error; **RMSE** - square root of MSE.
- **MAPE** - mean absolute percentage error over non-zero actuals, in percent.
- **SMAPE** - symmetric MAPE, in percent.
- **R2** - coefficient of determination.
- **MASE** - MAE divided by the mean absolute step-to-step change of the actual series.

QC classification (written to `metrics.txt`, only when the test sheet carries a label column):

- **Accuracy / Precision / Recall / F1 / Specificity / FPR / FNR**, plus the confusion-matrix counts
  **TP / TN / FP / FN**.
- **ROC_AUC**, computed from the per-step standardized deviation `|actual - mu| / sigma` as the
  score; recorded as `null` when the labeled test slice contains only one class.

Losses: MSE on `mu` (prediction) and Gaussian negative log-likelihood
`0.5 * log(sigma^2) + 0.5 * (y - mu)^2 / sigma^2` (QC).

## Notes

- **All paths are parameterized**: input and output locations come from command-line arguments; no
  local absolute paths are hardcoded.
- **Device**: CUDA is used when `torch.cuda.is_available()` is true, otherwise CPU; there is no
  device flag.
- **Randomness**: both experiment scripts keep the seed at the module-level constant `SEED = 42`;
  `data_processing/synthetic_anomaly.py` exposes `--seed` (default 42).
- **Invalid rows**: non-finite values and rows missing required features are dropped per file, and a
  file is skipped with a printed reason when it is missing required columns, coordinates or length.
- **Caching**: the processed prediction arrays are cached in `--output_dir` keyed by the feature
  column set; delete `_pred_data_cache_*.pt` to force reprocessing.

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

```
Copyright (c) 2026 SUZUKAZEA08
```
