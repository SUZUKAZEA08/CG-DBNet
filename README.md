# CG-DBNet

**Context-Guided Dual-Branch Network for Sea Surface Temperature Prediction and Quality Control**

A PyTorch framework for **ocean time-series prediction** and **data quality control (QC)** on
sea surface temperature (SST) observations, together with data-processing utilities. Every input
and output path is supplied through command-line arguments.

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

## Environment

- Tested environment: conda base (Python 3.12, PyTorch 2.5.1, CUDA 12.1).

```bash
pip install -r requirements.txt
```

`requirements.txt` pins every dependency to a tested range (`torch>=2.2,<3`, `numpy>=1.24,<2`,
`pandas>=2.0,<3`, `openpyxl>=3.0,<4`, `scikit-learn>=1.0,<2`, `scipy>=1.10,<2`,
`matplotlib>=3.5,<4`, `seaborn>=0.12,<1`, `tqdm>=4.60,<5`).

Figures produced by the two experiment entry points use the `DejaVu Sans` font, and all in-figure
text is English. The utilities in `data_processing/` apply their own journal style (serif face,
Times New Roman) and likewise label their figures in English.

## Configuration checklist

Every item below has a working default, so the package runs out of the box on data that uses the
default naming and column scheme. Each one still has to be reviewed when the inputs are different.

| What | Default | How to change it |
| --- | --- | --- |
| Series column names | prediction `--time_col time`, `--target_col sst`, `--feature_cols air_temperature sea_level_pressure`; QC `--time_col date`, `--target_col sst`, `--feature_cols` empty | pass the flags, or rename the columns in your files |
| Grid-point file names | built-in Latin rule `(lat\|latitude)...(lon\|longitude)`, which also supplies the coordinates | `--file_pattern` with **exactly two capture groups**, the first the latitude and the second the longitude |
| Latitude / longitude source | column aliases `lat`/`latitude` and `lon`/`longitude` (case-insensitive), otherwise the two capture groups above | provide either one; a station with neither is reported and skipped |
| Non-Latin file names or column headers | not recognised by default | supply `--file_pattern` and the matching `--time_col` / `--target_col` / `--feature_cols` values |
| QC workbook layout | sheets `train`, `val`, `test`; anomaly column `label` | `--train_sheet`, `--eval_sheet`, `--test_sheet`, `--label_col` |
| External evaluation data | `--test_path ./data/test.xlsx`; when it resolves to nothing, the DatasetB outputs are simply not written | point it at a single file or a directory of stations |
| Study-region bounds | 29.0-31.5 N and 123.0-125.5 E, seven `geo_*` features | `REGION_LAT_MIN` / `REGION_LAT_MAX` and `REGION_LON_MIN` / `REGION_LON_MAX` in `src/data_utils.py` are the only place the numbers are written (no flag); `PredConfig` in `experiments/train_prediction.py` copies them, the QC entry point has no region bounds |
| Split proportions | prediction `TRAIN_RATIO 0.7` / `VAL_RATIO 0.15`; QC validation share `--eval_val_ratio 0.5` | class attributes / the QC flag |
| Random seed | `SEED = 42` in both entry scripts | edit the constant |
| Optimiser internals | gradient clipping 1.0, `SIGMA_MIN 1e-4`, QC `MAX_CONSECUTIVE_REPLACE 2`, `ReduceLROnPlateau(patience=5, factor=0.5)` | class attributes only, no flags |
| Compute device | CUDA when available, otherwise CPU | no `--device` flag; control it with `CUDA_VISIBLE_DEVICES` |
| k-sigma threshold | `k` is the `--target_coverage` quantile of the validation standardized residuals, default 0.95 | `--target_coverage`; there is intentionally no offset parameter |
| Feature-engineering cache | `_pred_data_cache_<signature>.pt` inside the output directory, keyed by a 16-character hexadecimal fingerprint of the resolved data directory, the matched files (name, size, modification time), the external test source, the file-name rule, the column settings, `SEQ_LEN` and the two split ratios | delete the file to rebuild; the key itself changes when the inputs or any of those settings change, so a new data directory rebuilds without any manual step. Files left by the earlier 8-character naming scheme are ignored and reported, and one output directory can therefore hold several `.pt` caches |
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
|   |                           #   build_pred_model, build_qc_model
|   |-- data_utils.py           # DataProcessor, MultiFileTimeSeriesDataset, grid-file naming,
|   |                           #   feature engineering, spatial features, QC sheet pipeline,
|   |                           #   SeqDataset, build_feature_matrix
|   |-- metrics.py              # MAE/MSE/RMSE/MAPE/SMAPE/R2/MASE, classification metrics,
|   |                           #   Gaussian NLL, dynamic k-sigma calibration
|-- experiments/
|   |-- train_prediction.py     # prediction entry point (many grid-point files)
|   `-- train_qc.py             # QC entry point (one Excel workbook, train/val/test sheets)
`-- data_processing/
    |-- pca_analysis.py         # PCA of multi-station features
    |-- correlation_analysis.py # Pearson correlation matrix and heatmaps
    `-- synthetic_anomaly.py    # labeled anomaly injection for QC test data
```

## Data Preparation

### 1. Prediction experiment (many files, one per grid point)

`experiments/train_prediction.py` reads a **directory** of Excel (`.xlsx`, `.xls`) or CSV (`.csv`)
files, one file per grid point or station. Legacy `.xls` passes the file-name filter, but `pandas`
requires the optional `xlrd` package to open it; `xlrd` is not listed in `requirements.txt`, which
covers the default `.xlsx` and `.csv` inputs, so install it separately when `.xls` files have to be
read. Each file must provide:

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
- The region bounds used to normalize `geo_lat_norm`, `geo_lon_norm` and `geo_dist_center` are
  written once in `src/data_utils.py` as `REGION_LAT_MIN` / `REGION_LAT_MAX` and
  `REGION_LON_MIN` / `REGION_LON_MAX` (29.0-31.5 N, 123.0-125.5 E). `PredConfig` in
  `experiments/train_prediction.py` copies those constants, so another basin means editing the two
  lines in `src/data_utils.py`. The QC pipeline works on a single station sheet and has no region
  bounds.
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
(sheets `train`, `val`, `test`) is a valid `--data_file`, but see the note below on which column a QC
run has to target. Derived features are generated as in the prediction pipeline (cyclic hour/day
features, target `lag1`/`lag24`/rolling means, feature lags and first differences) and the whole
matrix is standardized with `StandardScaler`.

> **Which target column carries the injected anomalies.** `synthetic_anomaly.py` writes the injected
> values into a separate `sst_anomaly` column and never modifies `sst`, so the three sheets are not
> symmetric: `test` provides `date`, `sst` (clean series), `sst_anomaly` (same series with the four
> injected anomaly types) and `label`, while `train` and `val` provide only `date` and `sst`. A QC
> evaluation on this workbook therefore has to target the column that holds the anomalies, that is
> `--target_col sst_anomaly`. With the default `--target_col sst` the sequential loop reads the clean
> series while `label` marks points that are normal in it, so the classification block of
> `metrics.txt` does not measure anomaly detection. Note that `--target_col` applies to every sheet
> the run reads: `load_sheet` in `src/data_utils.py` requires the time and target columns in each of
> them and otherwise raises `ValueError: Sheet train missing required columns: ['sst_anomaly']`. As
> this script does not write `sst_anomaly` into `train` or `val`, supplying a workbook whose three
> sheets all provide the requested target column is left to the user.

### 3. Data-processing scripts

- `pca_analysis.py`: PCA over stations, time-aligned and regionally averaged; writes scree plot,
  loadings and scores figures plus CSV tables.
- `correlation_analysis.py`: Pearson correlation and p-value matrices with heatmap figures.
- `synthetic_anomaly.py`: injects four anomaly types into a clean series - out-of-range, gradient
  shift, spike, Gaussian - and writes a three-sheet workbook with a `label` column
  (0 = normal, 1 = out-of-range, 2 = gradient, 3 = spike, 4 = Gaussian).

`pca_analysis.py` and `correlation_analysis.py` address a station file set with a wider variable list
than the prediction experiment uses. Both require every matched file to expose the eight columns
`u10`, `v10`, `d2m`, `air_temperature`, `sea_level_pressure`, `swh`, `sst` and `time` (the
`REQUIRED_COLS` list at the top of each script), and a file missing any of them aborts the run with
`KeyError: File ... is missing columns: [...]`. Coordinates are not part of that list: the two
scripts average the stations found in the directory and never read a latitude or a longitude, so
neither a `lat`/`lon` data column nor a coordinate parsed from the file name is used; `--pattern`
still decides which files enter the run. The wind components (`u10`, `v10`), the dewpoint column
(`d2m`) and the significant wave height (`swh`) have no counterpart in the default prediction column
scheme (`time`, `sst`, `air_temperature`, `sea_level_pressure`), so a table built for the prediction
experiment does not satisfy this requirement as it stands.

## Usage

> All commands are executed from the **package root directory** (where this README resides).

> **The package contains no sample data.** The distributed tree holds code only - `data/` and the
> `*.csv` / `*.xlsx` inputs are excluded from the repository - so right after a clone the default
> `--data_dir ./data`, `--test_path ./data/test.xlsx` and `--data_file ./data/qc_data.xlsx` resolve
> to nothing and a prediction run stops with `No valid grid-point data file found under ./data`.
> Prepare your own inputs as described in Data Preparation, or generate a labeled QC workbook with
> `data_processing/synthetic_anomaly.py`, before running the commands below.

### Prediction experiment

```bash
# Full CG-DBNet model
python experiments/train_prediction.py --data_dir ./data --output_dir ./outputs/prediction

# Feature columns and hyper-parameters
python experiments/train_prediction.py --data_dir ./data --feature_cols air_temperature sea_level_pressure \
    --seq_len 48 --epochs 100 --batch_size 64 --lr 1e-3

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
```

Arguments of `experiments/train_prediction.py` and their defaults: `--data_dir ./data`,
`--test_path ./data/test.xlsx`, `--file_pattern` (defaults to the Latin-letter naming rule above),
`--output_dir ./outputs/prediction`, `--model proposed` (`proposed` is the only accepted value),
`--target_col sst`, `--time_col time`, `--feature_cols air_temperature sea_level_pressure`,
`--seq_len 48`, `--batch_size 64`, `--epochs 100`, `--lr 0.001`, `--hidden_dim 64`,
`--dropout 0.15`, `--patience 15`, and the store-true ablation flags `--no_pos`, `--no_msac`,
`--no_gap`, `--no_affine`, `--serial_gru`, `--no_conv3`, `--no_conv5`, `--no_conv7`.

### QC experiment

```bash
# Full CG-DBNet model (--data_file is required)
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --output_dir ./outputs/qc

# Sheets, features and hyper-parameters
python experiments/train_qc.py --data_file ./data/qc_data.xlsx \
    --train_sheet train --eval_sheet val --test_sheet test --label_col label \
    --seq_len 48 --epochs 100 --target_coverage 0.95

# Component ablations
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_no_msac
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_no_dagap
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_no_pos
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_affine_add
python experiments/train_qc.py --data_file ./data/qc_data.xlsx --model abl_serial_gru
```

Arguments of `experiments/train_qc.py`: `--data_file` (required), `--output_dir ./outputs/qc`,
`--train_sheet train`, `--eval_sheet val`, `--test_sheet test`, `--time_col date`,
`--target_col sst`, `--feature_cols` (default empty), `--label_col label`, `--eval_val_ratio 0.5`,
`--seq_len 48`, `--batch_size 64`, `--epochs 100`, `--lr 0.001`, `--hidden_dim 64`,
`--dropout 0.15`, `--patience 30`, `--target_coverage 0.95`. `--model` accepts `proposed` (the
default) and the five ablation values `abl_no_msac`, `abl_no_dagap`, `abl_no_pos`, `abl_affine_add`,
`abl_serial_gru`, which are described in the table below.
`--patience` and `--target_coverage` fall back to the values of `QcConfig` (30 and 0.95) when they
are not passed.

**k-sigma calibration.** Let `z = |actual - mu| / sigma` on the validation split. The threshold `k`
is exactly the `--target_coverage` quantile of `z` - no additional offset is added - and the
empirical coverage reached on validation is stored next to it.

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

The workbook written by `synthetic_anomaly.py` is the intended `--data_file` of
`experiments/train_qc.py`, and its columns are split across the sheets: `test` carries `date`, `sst`
(the clean series), `sst_anomaly` (the series with the injected anomalies) and `label`, whereas
`train` and `val` carry `date` and `sst` only. The injected values are never written into `sst`, so
evaluating QC against the labels means passing `--target_col sst_anomaly`, and because `--target_col`
is read from all three sheets the requested column has to exist in each of them - see the note in the
QC data requirements above.

## Outputs

Both experiment scripts write into `--output_dir`, which is created automatically. The
data-processing utilities write into their own `--output` directory.

### Prediction experiment (default `./outputs/prediction`)

| File | Content |
| --- | --- |
| `training_config.json` | Resolved configuration of the run: every `PredConfig` entry (paths, column settings, region bounds, split ratios, hyper-parameters), plus `model_kind` (the `--model` value) and `ablation_toggles`, the eight ablation switches as they took effect in this run |
| `_pred_data_cache_<signature>.pt` | Processed arrays cache, keyed by a 16-character hexadecimal fingerprint of the input data and of the file-name rule, column settings, `SEQ_LEN` and split ratios; re-used on the next run when the fingerprint matches, so one directory can hold one file per input set |
| `best_model.pth` | Checkpoint at the best validation loss |
| `loss_history.txt` | Per-epoch train/validation MSE and validation MAE/RMSE/R2/MAPE/SMAPE/MASE, plus the best epoch |
| `predictions_DatasetA_Test.csv` | Actual vs. predicted values on the in-grid test split |
| `metrics_DatasetA_Test.txt` | Regression metrics on the in-grid test split |
| `prediction_DatasetA_Test.png` | Actual vs. predicted series for the in-grid test split |
| `predictions_DatasetB_External.csv` | Actual vs. predicted values on the external test data |
| `metrics_DatasetB_External.txt` | Regression metrics on the external test data |
| `prediction_DatasetB_External.png` | Actual vs. predicted series for the external test data |

The three `DatasetB_External` files are written only when `--test_path` resolves to usable data.
Because `training_config.json` records `model_kind` together with the eight ablation switches, two
runs that share every hyper-parameter but differ in an ablation flag now leave distinguishable
configuration files behind. The QC entry point stores `model_kind` in its own
`training_config.json`, which is where its `abl_*` variants are recorded.

### QC experiment (default `./outputs/qc`)

| File | Content |
| --- | --- |
| `training_config.json` | Every `QcConfig` entry as resolved for the run (window, batch, epochs, learning rate, hidden size, dropout, patience, gradient clip, `sigma_min`, target coverage, replacement limit, device) plus `model_kind` |
| `config.json` | Model kind, input dimension and column order, hyper-parameters, column settings |
| `scaler.pkl`, `target_scaler.pkl` | Fitted input and target scalers, for reuse on new data |
| `model.pth` | Checkpoint at the best validation NLL |
| `loss_history.txt` | Per-epoch train/validation Gaussian NLL and the best epoch |
| `train_log.json` | Loss history, best validation NLL, number of executed epochs |
| `k_sigma_config.json` | `target_coverage`, calibrated `k_sigma`, validation coverage and sigma statistics |
| `qc_result.csv` | Per-step `Actual`, `Mu`, `Sigma`, `Lower`, `Upper`, `k_i`, `IsAnomaly`, `ReplacedForNextStep`, `CorrectedInputValue` and `TrueLabel` (when a label column is available) |
| `metrics.txt` | `ModelName`, then the calibrated threshold under the key `k_sigma`, detected/total point counts, replaced count and the classification metrics |
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
- **Caching**: the processed prediction arrays are cached in `--output_dir` as
  `_pred_data_cache_<signature>.pt`, where the 16-character hexadecimal signature fingerprints the
  resolved training directory and the files it contributes (name, size, modification time), the
  external test source, the file-name rule, the column settings, `SEQ_LEN` and the split ratios. A
  different key rebuilds the arrays on its own, so pointing the run at other data is enough; delete
  `_pred_data_cache_*.pt` to force reprocessing of the current data. A cache file named with the
  earlier 8-character key is never read: the run reports it and writes a fresh file.

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

```
Copyright (c) 2026 SUZUKAZEA08
```
