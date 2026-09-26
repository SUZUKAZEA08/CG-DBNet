"""
Data loading and feature engineering
====================================
Two data pipelines are combined in this module:

  Prediction experiment (multiple files, one grid point per file):
    - extract_lat_lon / generate_spatial_features: read latitude and longitude from
      columns or parse them from the file name, then build the 7 geospatial features
    - DataProcessor: multi-file loading, three-way split along the time axis,
      feature engineering (lags, differences, rolling windows, cyclic time
      encoding), concatenation of the spatial features, optional caching
    - MultiFileTimeSeriesDataset: differenced target (y = t_curr - t_prev) plus
      Gaussian noise augmentation during training

  Quality-control experiment (one file, several sheets):
    - DataConfig / load_sheet / feature_engineering / build_input_cols / prepare_datasets:
      load the train/val sheets of one file, normalize with StandardScaler
    - SeqDataset: y is the normalized target value directly (for Gaussian NLL)
    - build_feature_matrix: rebuild the feature matrix on the fly during sequential
      quality control (recompute the rolling features after an anomaly is replaced)

Every path is supplied by the caller (the Config / argparse of the experiment
scripts); this module hard-codes no local paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# File-name rule for grid-point data files. Group 1 is the latitude, group 2 the longitude.
# Override it through cfg.GRID_FILE_PATTERN (the experiment scripts expose --file_pattern).
GRID_FILE_PATTERN = r'(?i)(?:lat|latitude)[\s_=]*(-?\d+(?:\.\d+)?)[\s_,;+-]*(?:lon|longitude)[\s_=]*(-?\d+(?:\.\d+)?)'

LAT_COLUMN_ALIASES = ('lat', 'latitude')
LON_COLUMN_ALIASES = ('lon', 'longitude')

# Default study region, in degrees north and degrees east. It is the single source of truth for
# the bounds that normalize geo_lat_norm, geo_lon_norm and geo_dist_center; every experiment
# script copies them into its own configuration object, where they can still be edited.
REGION_LAT_MIN, REGION_LAT_MAX = 29.0, 31.5
REGION_LON_MIN, REGION_LON_MAX = 123.0, 125.5


# ===================== Latitude / longitude + 7 spatial features =====================
def _first_matching_column(df: pd.DataFrame, aliases):
    """First column whose name equals one of aliases (case and surrounding blanks ignored)."""
    for alias in aliases:
        for col in df.columns:
            if str(col).strip().lower() == alias:
                return col
    return None


def extract_lat_lon(df: pd.DataFrame, filepath: str, pattern: str | None = None):
    """Take latitude and longitude from the DataFrame columns, otherwise parse the file
    name. Returns (lat, lon) or (None, None). pattern overrides GRID_FILE_PATTERN for
    the file-name branch."""
    lat, lon = None, None
    lat_col = _first_matching_column(df, LAT_COLUMN_ALIASES)
    if lat_col is not None:
        lat = float(df[lat_col].iloc[0])
    lon_col = _first_matching_column(df, LON_COLUMN_ALIASES)
    if lon_col is not None:
        lon = float(df[lon_col].iloc[0])
    if lat is not None and lon is not None:
        return lat, lon
    match = re.search(pattern or GRID_FILE_PATTERN, os.path.basename(filepath), re.IGNORECASE)
    if match is not None and len(match.groups()) >= 2:
        return float(match.group(1)), float(match.group(2))
    return None, None


def generate_spatial_features(df: pd.DataFrame, lat: float, lon: float, config) -> pd.DataFrame:
    """Build the 7 geospatial features: sine/cosine of latitude and longitude, the
    normalized coordinates, and the normalized distance to the region center."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    df['geo_lat_sin'] = np.sin(lat_rad)
    df['geo_lat_cos'] = np.cos(lat_rad)
    df['geo_lon_sin'] = np.sin(lon_rad)
    df['geo_lon_cos'] = np.cos(lon_rad)
    lat_range = config.LAT_MAX - config.LAT_MIN
    lon_range = config.LON_MAX - config.LON_MIN
    df['geo_lat_norm'] = (lat - config.LAT_MIN) / (lat_range if lat_range > 0 else 1.0)
    df['geo_lon_norm'] = (lon - config.LON_MIN) / (lon_range if lon_range > 0 else 1.0)
    center_lat = (config.LAT_MIN + config.LAT_MAX) / 2.0
    center_lon = (config.LON_MIN + config.LON_MAX) / 2.0
    dlat = lat - center_lat
    dlon = (lon - center_lon) * np.cos(lat_rad)
    dist = np.sqrt(dlat ** 2 + dlon ** 2)
    max_dist = np.sqrt((lat_range / 2) ** 2 + ((lon_range / 2) * np.cos(np.deg2rad(center_lat))) ** 2)
    df['geo_dist_center'] = dist / (max_dist if max_dist > 0 else 1.0)
    return df


# ===================== Prediction experiment: multi-file data processor =====================
CACHE_FILE_PREFIX = '_pred_data_cache_'
# Length of the fingerprint part of a cache file name (hexadecimal characters).
CACHE_SIG_CHARS = 16


def _file_stamp(path) -> list:
    """Name, size and modification time of one data file; -1 placeholders when it cannot be
    read. Used to detect that the input data behind a cache changed."""
    name = os.path.basename(str(path))
    try:
        st = os.stat(path)
        return [name, int(st.st_size), int(st.st_mtime_ns)]
    except OSError:
        return [name, -1, -1]


class DataProcessor:
    """Loading and feature engineering for a grid of stations held in one file per point.
    config must provide:
    TRAIN_DIR, TEST_PATH, TRAIN_MAX_FILES, TRAIN_FILE_START_INDEX, TIME_COL, TARGET_COL,
    FEATURE_COLS, LAT_MIN/MAX, LON_MIN/MAX, SPATIAL_FEATURE_NAMES, SEQ_LEN,
    TRAIN_RATIO, VAL_RATIO. GRID_FILE_PATTERN is optional and falls back to the
    module-level constant of the same name."""

    def __init__(self, config, cache_dir: str | None = None):
        self.cfg = config
        self.cache_dir = cache_dir  # None disables reading and writing the cache

    def load_and_process(self):
        train_files = self._list_data_files(self.cfg.TRAIN_DIR,
                                            getattr(self.cfg, 'TRAIN_MAX_FILES', None),
                                            getattr(self.cfg, 'TRAIN_FILE_START_INDEX', 0))
        cache_path = None
        cache_sig = None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_sig = self._cache_signature(train_files)
            cache_path = os.path.join(self.cache_dir,
                                      f'{CACHE_FILE_PREFIX}{cache_sig}.pt')
            if os.path.exists(cache_path):
                c = torch.load(cache_path, map_location='cpu', weights_only=False)
                if c.get('cache_sig') != cache_sig or 'internal_test_arrays' not in c:
                    os.remove(cache_path)
                    print(f"[CACHE] discarded {os.path.basename(cache_path)}: the stored "
                          "arrays were built from different inputs")
                else:
                    print(f"[CACHE] hit ({os.path.basename(cache_path)}) train stations={len(c['train_arrays'])} "
                          f"input dims={len(c['input_cols'])}")
                    return (c['train_arrays'], c['val_arrays'], c['internal_test_arrays'],
                            c['external_test_arrays'], c['input_cols'], c['target_idx'],
                            c['n_spatial'], c['augment_indices'], c['file_info_list'])
        self._report_legacy_caches()

        print(f"\n{'='*60}\n[PREDICTION] train dir: {self.cfg.TRAIN_DIR}\n[PREDICTION] test path: {self.cfg.TEST_PATH}\n"
              f"[PREDICTION] external features: {self.cfg.FEATURE_COLS}\n{'='*60}")
        if not train_files:
            raise RuntimeError(
                f"No valid grid-point data file found under {self.cfg.TRAIN_DIR}. "
                "File names are matched against GRID_FILE_PATTERN; pass --file_pattern to use your own rule.")
        print(f"Found {len(train_files)} training data files")

        train_arrays, val_arrays, internal_test_arrays = [], [], []
        input_cols, n_spatial, file_info_list = None, 0, []
        for filepath in train_files:
            try:
                df = self._load_file(filepath)
                required = [self.cfg.TIME_COL, self.cfg.TARGET_COL] + self.cfg.FEATURE_COLS
                missing = [c for c in required if c not in df.columns]
                if missing:
                    print(f"  [SKIP] {os.path.basename(filepath)}: missing columns {missing}")
                    continue
                lat, lon = extract_lat_lon(df, filepath, pattern=self._grid_pattern())
                if lat is None or lon is None:
                    print(f"  [SKIP] {os.path.basename(filepath)}: latitude/longitude not found in the "
                          "columns and not parseable from the file name (see --file_pattern)")
                    continue
                n = len(df)
                train_idx = int(n * self.cfg.TRAIN_RATIO)
                val_idx = int(n * (self.cfg.TRAIN_RATIO + self.cfg.VAL_RATIO))
                if (train_idx < self.cfg.SEQ_LEN + 25
                        or (val_idx - train_idx) < self.cfg.SEQ_LEN + 1
                        or (n - val_idx) < self.cfg.SEQ_LEN + 1):
                    print(f"  [SKIP] {os.path.basename(filepath)}: too few rows for the three-way split")
                    continue
                train_df = df.iloc[:train_idx].copy()
                val_df = df.iloc[train_idx:val_idx].copy()
                test_df = df.iloc[val_idx:].copy()
                train_df = self._feature_engineering(train_df, lat, lon, is_test=False)
                history_steps = 24
                val_start_time = val_df[self.cfg.TIME_COL].iloc[0]
                val_with_context = pd.concat(
                    [df.iloc[max(0, train_idx - history_steps):train_idx], val_df], axis=0, ignore_index=True)
                val_feat = self._feature_engineering(val_with_context, lat, lon, is_test=True)
                val_df = val_feat[val_feat[self.cfg.TIME_COL] >= val_start_time].reset_index(drop=True)
                test_start_time = test_df[self.cfg.TIME_COL].iloc[0]
                test_with_context = pd.concat(
                    [df.iloc[max(0, val_idx - history_steps):val_idx], test_df], axis=0, ignore_index=True)
                test_feat = self._feature_engineering(test_with_context, lat, lon, is_test=True)
                test_df = test_feat[test_feat[self.cfg.TIME_COL] >= test_start_time].reset_index(drop=True)
                if len(train_df) < self.cfg.SEQ_LEN + 1 or len(val_df) < self.cfg.SEQ_LEN + 1 \
                        or len(test_df) < self.cfg.SEQ_LEN + 1:
                    print(f"  [SKIP] {os.path.basename(filepath)}: too few rows after feature engineering")
                    continue
                if input_cols is None:
                    input_cols, n_spatial = self._determine_column_order(train_df)
                    print(f"\n  [COLUMNS] input dims: {len(input_cols)} "
                          f"(temporal:{len(input_cols)-n_spatial}, spatial:{n_spatial})")
                cols_missing = [c for c in input_cols if c not in train_df.columns]
                if cols_missing:
                    print(f"  [SKIP] {os.path.basename(filepath)}: inconsistent columns {cols_missing}")
                    continue
                train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=input_cols).reset_index(drop=True)
                val_df = val_df.replace([np.inf, -np.inf], np.nan).dropna(subset=input_cols).reset_index(drop=True)
                test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=input_cols).reset_index(drop=True)
                if len(train_df) < self.cfg.SEQ_LEN + 1 or len(val_df) < self.cfg.SEQ_LEN + 1 \
                        or len(test_df) < self.cfg.SEQ_LEN + 1:
                    print(f"  [SKIP] {os.path.basename(filepath)}: too few rows after the second cleaning pass")
                    continue
                train_arrays.append(train_df[input_cols].to_numpy(dtype=np.float32))
                val_arrays.append(val_df[input_cols].to_numpy(dtype=np.float32))
                internal_test_arrays.append(test_df[input_cols].to_numpy(dtype=np.float32))
                file_info_list.append({
                    'file': os.path.basename(filepath), 'lat': lat, 'lon': lon,
                    'train_len': len(train_df), 'val_len': len(val_df), 'test_len': len(test_df)})
                print(f"  [OK] {os.path.basename(filepath)} | (lat={lat:.1f} N, lon={lon:.1f} E) | "
                      f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
            except Exception as e:
                print(f"  [ERROR] {os.path.basename(filepath)}: {e}")
                continue
        if not train_arrays:
            raise RuntimeError(
                "No usable training data left after loading every grid-point file. "
                "File names are matched against GRID_FILE_PATTERN; pass --file_pattern to use your own rule.")

        target_idx = input_cols.index(self.cfg.TARGET_COL)
        external_test_arrays = self._load_test_data(input_cols)
        augment_indices = [i for i, c in enumerate(input_cols)
                           if not ('_diff' in str(c) or '_sin' in str(c) or '_cos' in str(c)
                                   or str(c).startswith('geo_'))]
        total_train = sum(len(a) for a in train_arrays)
        total_val = sum(len(a) for a in val_arrays)
        total_internal_test = sum(len(a) for a in internal_test_arrays)
        total_external_test = sum(len(a) for a in external_test_arrays) if external_test_arrays else 0
        print(f"\n{'='*60}\n[PREDICTION] summary: stations={len(train_arrays)} | train={total_train} | "
              f"val={total_val} | grid test={total_internal_test} | external buoys={total_external_test}\n"
              f"  target '{self.cfg.TARGET_COL}' (idx={target_idx}) | spatial dims={n_spatial} | "
              f"augmentable dims={len(augment_indices)}\n{'='*60}\n")

        if cache_path is not None:
            torch.save({'cache_sig': cache_sig,
                        'train_arrays': train_arrays, 'val_arrays': val_arrays,
                        'internal_test_arrays': internal_test_arrays, 'external_test_arrays': external_test_arrays,
                        'input_cols': input_cols, 'target_idx': target_idx, 'n_spatial': n_spatial,
                        'augment_indices': augment_indices, 'file_info_list': file_info_list}, cache_path)
            print(f"[CACHE] saved: {os.path.basename(cache_path)}")
        return train_arrays, val_arrays, internal_test_arrays, external_test_arrays, \
            input_cols, target_idx, n_spatial, augment_indices, file_info_list

    def _cache_signature(self, train_files) -> str:
        """Fingerprint of every input that shapes the cached arrays: the resolved training
        directory and the files it contributes, the external test source, the file-name rule,
        the column settings and the window/split geometry."""
        test_path = os.path.abspath(str(self.cfg.TEST_PATH))
        if os.path.isdir(test_path):
            test_source = ['directory', test_path,
                           [_file_stamp(p) for p in self._list_data_files(test_path)]]
        elif os.path.isfile(test_path):
            test_source = ['file', test_path, _file_stamp(test_path)]
        else:
            test_source = ['absent', test_path]
        payload = {
            'train_dir': os.path.abspath(str(self.cfg.TRAIN_DIR)),
            'train_files': [_file_stamp(p) for p in train_files],
            'test_source': test_source,
            'grid_pattern': str(self._grid_pattern()),
            'time_col': str(self.cfg.TIME_COL),
            'target_col': str(self.cfg.TARGET_COL),
            'feature_cols': [str(c) for c in self.cfg.FEATURE_COLS],
            'spatial_feature_names': [str(c) for c in self.cfg.SPATIAL_FEATURE_NAMES],
            'seq_len': int(self.cfg.SEQ_LEN),
            'train_ratio': float(self.cfg.TRAIN_RATIO),
            'val_ratio': float(self.cfg.VAL_RATIO),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.md5(blob.encode()).hexdigest()[:CACHE_SIG_CHARS]

    def _report_legacy_caches(self):
        """Point out cache files whose name carries the 8-character fingerprint of an earlier
        naming scheme; such a file is never read, because the current key is longer and covers
        the data sources as well."""
        if self.cache_dir is None or not os.path.isdir(self.cache_dir):
            return
        legacy_len = len(CACHE_FILE_PREFIX) + 8 + len('.pt')
        legacy = [n for n in sorted(os.listdir(self.cache_dir))
                  if n.startswith(CACHE_FILE_PREFIX) and n.endswith('.pt') and len(n) == legacy_len]
        if legacy:
            print(f"[CACHE] ignoring cache file(s) from an earlier naming scheme: {legacy}")

    def _grid_pattern(self):
        """File-name rule: cfg.GRID_FILE_PATTERN when set, else the module default."""
        return getattr(self.cfg, 'GRID_FILE_PATTERN', GRID_FILE_PATTERN)

    def _list_data_files(self, directory, max_files=None, start_index=0):
        if not os.path.isdir(directory):
            return []
        pattern = self._grid_pattern()
        file_re = re.compile(pattern, re.IGNORECASE)
        files = []
        for f in sorted(os.listdir(directory)):
            if f.startswith('~$'):
                continue
            if not f.endswith(('.xlsx', '.xls', '.csv')):
                continue
            if not file_re.search(f):
                continue
            files.append(os.path.join(directory, f))
        start_index = max(0, int(start_index) if start_index is not None else 0)
        files = files[start_index:]
        if max_files is not None:
            files = files[:max(0, int(max_files))]
        return files

    def _load_file(self, filepath):
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        if self.cfg.TIME_COL not in df.columns:
            raise ValueError(f"Missing time column '{self.cfg.TIME_COL}'")
        df[self.cfg.TIME_COL] = pd.to_datetime(df[self.cfg.TIME_COL])
        return df.sort_values(self.cfg.TIME_COL).reset_index(drop=True)

    def _feature_engineering(self, df, lat, lon, is_test=False):
        hour = df[self.cfg.TIME_COL].dt.hour
        day = df[self.cfg.TIME_COL].dt.dayofyear
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * day / 365)
        df['day_cos'] = np.cos(2 * np.pi * day / 365)
        t = self.cfg.TARGET_COL
        df[f'{t}_lag1'] = df[t].shift(1)
        df[f'{t}_lag24'] = df[t].shift(24)
        df[f'{t}_diff1'] = df[t].diff()
        df[f'{t}_diff2'] = df[f'{t}_diff1'].diff()
        for win, p in {6: {'window': 6, 'min_periods': 1 if is_test else 6},
                       24: {'window': 24, 'min_periods': 1 if is_test else 24}}.items():
            df[f'{t}_roll_mean_{win}'] = df[t].rolling(**p).mean()
        for col in self.cfg.FEATURE_COLS:
            if col in df.columns:
                for lag in range(1, 4):
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
                df[f'{col}_diff1'] = df[col].diff()
                df[f'{col}_diff2'] = df[f'{col}_diff1'].diff()
        df = generate_spatial_features(df, lat, lon, self.cfg)
        req = [self.cfg.TARGET_COL, self.cfg.TIME_COL] + self.cfg.FEATURE_COLS
        req += [f'{t}_lag1', f'{t}_lag24', f'{t}_diff1', f'{t}_diff2',
                f'{t}_roll_mean_6', f'{t}_roll_mean_24', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        for col in self.cfg.FEATURE_COLS:
            if col in df.columns:
                req += [f'{col}_lag1', f'{col}_lag2', f'{col}_lag3', f'{col}_diff1', f'{col}_diff2']
        req += [c for c in self.cfg.SPATIAL_FEATURE_NAMES if c in df.columns]
        req = [c for c in req if c in df.columns]
        return df.dropna(subset=req).reset_index(drop=True)

    def _determine_column_order(self, sample_df):
        t = self.cfg.TARGET_COL
        other = [f'{t}_lag1', f'{t}_lag24', f'{t}_diff1', f'{t}_diff2',
                 f'{t}_roll_mean_6', f'{t}_roll_mean_24']
        for col in self.cfg.FEATURE_COLS:
            other += [col, f'{col}_lag1', f'{col}_lag2', f'{col}_lag3', f'{col}_diff1', f'{col}_diff2']
        time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        spatial_cols = [c for c in self.cfg.SPATIAL_FEATURE_NAMES if c in sample_df.columns]
        n_spatial = len(spatial_cols)
        other = [c for c in other if c in sample_df.columns
                 and pd.api.types.is_numeric_dtype(sample_df[c].dtype)]
        return other + [c for c in time_cols if c in sample_df.columns] + [self.cfg.TARGET_COL] + spatial_cols, n_spatial

    def _load_test_data(self, input_cols):
        out = []
        if not os.path.exists(self.cfg.TEST_PATH):
            return out
        if os.path.isdir(self.cfg.TEST_PATH):
            for fp in self._list_data_files(self.cfg.TEST_PATH):
                a = self._process_single_test_file(fp, input_cols)
                if a is not None:
                    out.append(a)
        else:
            a = self._process_single_test_file(self.cfg.TEST_PATH, input_cols)
            if a is not None:
                out.append(a)
        return out

    def _process_single_test_file(self, filepath, input_cols):
        try:
            df = self._load_file(filepath)
            required = [self.cfg.TIME_COL, self.cfg.TARGET_COL] + self.cfg.FEATURE_COLS
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"  [TEST-SKIP] {os.path.basename(filepath)}: missing columns {missing}")
                return None
            lat, lon = extract_lat_lon(df, filepath, pattern=self._grid_pattern())
            if lat is None or lon is None:
                print(f"  [TEST-SKIP] {os.path.basename(filepath)}: latitude/longitude not found in the "
                      "columns and not parseable from the file name (see --file_pattern)")
                return None
            df = self._feature_engineering(df, lat, lon, is_test=True)
            if len(df) < self.cfg.SEQ_LEN + 1:
                print(f"  [TEST-SKIP] {os.path.basename(filepath)}: too few rows")
                return None
            missing = [c for c in input_cols if c not in df.columns]
            if missing:
                print(f"  [TEST-SKIP] {os.path.basename(filepath)}: missing columns {missing}")
                return None
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=input_cols).reset_index(drop=True)
            if len(df) < self.cfg.SEQ_LEN + 1:
                print(f"  [TEST-SKIP] {os.path.basename(filepath)}: too few rows after the second cleaning pass")
                return None
            print(f"  [TEST-OK] {os.path.basename(filepath)} | (lat={lat:.5f} N, lon={lon:.5f} E) | samples={len(df)}")
            return df[input_cols].to_numpy(dtype=np.float32)
        except Exception as e:
            print(f"  [TEST-ERROR] {os.path.basename(filepath)}: {e}")
            return None


# ===================== Prediction experiment: dataset with a differenced target =====================
class MultiFileTimeSeriesDataset(Dataset):
    """Multi-file time series dataset whose target is the first difference
    y = t_curr - t_prev. In train mode Gaussian noise is added to the dimensions that are
    not differences, not time encodings and not spatial features."""

    def __init__(self, data_arrays, seq_len, target_idx, mode='test', augment_indices=None):
        self.seq_len = seq_len
        self.target_idx = target_idx
        self.mode = mode
        self.augment_indices = augment_indices
        self.index_map = []
        self.tensors = []
        for i, arr in enumerate(data_arrays):
            t = torch.FloatTensor(arr)
            self.tensors.append(t)
            for j in range(max(0, len(arr) - seq_len)):
                self.index_map.append((i, j))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        i, j = self.index_map[idx]
        data = self.tensors[i]
        x = data[j: j + self.seq_len].clone()
        t_curr = data[j + self.seq_len, self.target_idx]
        t_prev = data[j + self.seq_len - 1, self.target_idx]
        y = t_curr - t_prev
        if self.mode == 'train' and self.augment_indices:
            offset = torch.randn(1, len(self.augment_indices)) * 0.5
            x[:, self.augment_indices] = x[:, self.augment_indices] + offset
        return x, y


# ===================== Quality-control experiment: single-file, multi-sheet pipeline =====================
@dataclass
class DataConfig:
    data_path: str
    train_sheet: str
    eval_sheet: str
    eval_val_ratio: float
    time_col: str
    target_col: str
    feature_cols: list = field(default_factory=list)


def load_sheet(df_path: str, sheet_name: str, time_col: str, target_col: str,
               feature_cols: list) -> pd.DataFrame:
    df = pd.read_excel(df_path, sheet_name=sheet_name)
    required = [time_col, target_col] + list(feature_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet {sheet_name} missing required columns: {missing}")
    df = df[required].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[time_col, target_col]).sort_values(time_col).reset_index(drop=True)
    return df


def feature_engineering(df: pd.DataFrame, time_col: str, target_col: str,
                        feature_cols: list, is_test: bool) -> pd.DataFrame:
    out = df.copy()
    hour = out[time_col].dt.hour
    dayofyear = out[time_col].dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_sin"] = np.sin(2 * np.pi * dayofyear / 365)
    out["day_cos"] = np.cos(2 * np.pi * dayofyear / 365)
    out[f"{target_col}_lag1"] = out[target_col].shift(1)
    out[f"{target_col}_lag24"] = out[target_col].shift(24)
    win6 = 1 if is_test else 6
    win24 = 1 if is_test else 24
    out[f"{target_col}_roll_mean_6"] = out[target_col].rolling(window=6, min_periods=win6).mean()
    out[f"{target_col}_roll_mean_24"] = out[target_col].rolling(window=24, min_periods=win24).mean()
    for col in feature_cols:
        if col in out.columns:
            for lag in (1, 2, 3):
                out[f"{col}_lag{lag}"] = out[col].shift(lag)
            out[f"{col}_diff"] = out[col].diff()
    return out.dropna().reset_index(drop=True)


def build_input_cols(df: pd.DataFrame, time_col: str, target_col: str) -> list:
    excluded = [time_col, target_col]
    rest = [c for c in df.columns if c not in excluded]
    time_cols = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
    ordered = [c for c in rest if c not in time_cols] + [c for c in time_cols if c in rest]
    return ordered + [target_col]


def split_train_val_test(cfg: DataConfig):
    train_df = load_sheet(cfg.data_path, cfg.train_sheet, cfg.time_col, cfg.target_col, cfg.feature_cols)
    eval_df = load_sheet(cfg.data_path, cfg.eval_sheet, cfg.time_col, cfg.target_col, cfg.feature_cols)
    split_idx = int(len(eval_df) * cfg.eval_val_ratio)
    val_df = eval_df.iloc[:split_idx].copy()
    test_df = eval_df.iloc[split_idx:].copy()
    return train_df, val_df, test_df


def prepare_datasets(cfg: DataConfig):
    """Load the train/eval sheets, run feature engineering and StandardScaler normalization.
    Returns a dict with X_train/X_val (normalized, for SeqDataset), target_idx, input_cols,
    scaler and target_scaler."""
    train_df, val_df, _test_df = split_train_val_test(cfg)
    train_df = feature_engineering(train_df, cfg.time_col, cfg.target_col, cfg.feature_cols, is_test=False)
    val_df = feature_engineering(val_df, cfg.time_col, cfg.target_col, cfg.feature_cols, is_test=True)

    input_cols = build_input_cols(train_df, cfg.time_col, cfg.target_col)
    X_train = train_df[input_cols].values
    X_val = val_df[input_cols].values

    scaler = StandardScaler()
    target_scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    target_scaler.fit(train_df[[cfg.target_col]].values)

    target_idx = input_cols.index(cfg.target_col)
    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "target_idx": target_idx,
        "input_cols": input_cols,
        "scaler": scaler,
        "target_scaler": target_scaler,
    }


class SeqDataset(Dataset):
    """Quality-control dataset: x is the normalized window of length seq_len, y the normalized
    target value at the next time step (for Gaussian NLL)."""

    def __init__(self, data: np.ndarray, seq_len: int, target_idx: int):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.target_idx = target_idx

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len, self.target_idx]
        return x, y


def build_feature_matrix(df_base: pd.DataFrame, target_series: pd.Series, target_col: str,
                         feature_cols: list, input_cols: list, time_col: str = "date") -> np.ndarray:
    """During sequential quality control, rebuild the feature matrix from the target series with
    the anomalies already replaced, so that the input features of the next point are up to date."""
    keep_cols = [time_col] + [c for c in feature_cols if c in df_base.columns]
    out = df_base[keep_cols].copy() if keep_cols else df_base[[time_col]].copy()
    out[target_col] = target_series.values

    time_data = pd.to_datetime(out[time_col])
    hour = time_data.dt.hour
    day = time_data.dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_sin"] = np.sin(2 * np.pi * day / 365)
    out["day_cos"] = np.cos(2 * np.pi * day / 365)

    t = target_col
    out[f"{t}_lag1"] = out[t].shift(1)
    out[f"{t}_lag24"] = out[t].shift(24)
    out[f"{t}_roll_mean_6"] = out[t].rolling(window=6, min_periods=1).mean()
    out[f"{t}_roll_mean_24"] = out[t].rolling(window=24, min_periods=1).mean()

    for col in feature_cols:
        if col not in out.columns:
            continue
        for lag in (1, 2, 3):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
        out[f"{col}_diff"] = out[col].diff()

    out = out[input_cols].copy()
    out = out.ffill().bfill()
    return out.values


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
