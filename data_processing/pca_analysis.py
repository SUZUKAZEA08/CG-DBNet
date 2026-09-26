#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Principal component analysis (PCA) of multi-station ocean meteorological data
=============================================================================
- Reads every .xlsx station file in the input directory (file glob configurable)
- Time alignment + spatial averaging
- PCA dimensionality-reduction analysis
- Writes three journal-standard figures (PNG + PDF) and the data tables

Usage:
    python pca_analysis.py --input ./data --output ./output
    python pca_analysis.py --input ./data --pattern "lat*_lon*.xlsx"
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# Global plotting style, aligned with journal requirements
# ============================================================================
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.5,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'stix',
})

# Column names as recorded in the station files (must match exactly)
COL_U10 = 'u10'
COL_V10 = 'v10'
COL_D2M = 'd2m'
COL_AIR_TEMP = 'air_temperature'
COL_MSLP = 'sea_level_pressure'
COL_SWH = 'swh'
COL_SST = 'sst'
COL_TIME = 'time'

# Latitude and longitude are not analyzed here (the averaging is purely spatial over the
# stations found in the directory), so they are not required columns of the input files.
REQUIRED_COLS = [COL_U10, COL_V10, COL_D2M, COL_AIR_TEMP, COL_MSLP,
                 COL_SWH, COL_SST, COL_TIME]

ALL_VAR_NAMES = ['d2m', 'air_temp', 'mslp', 'swh', 'sst', 'wind_speed']
ENGLISH_LABELS = {
    'd2m': 'Dewpoint Temp (degC)',
    'air_temp': 'Air Temp (degC)',
    'mslp': 'MSLP (Pa)',
    'swh': 'Significant Wave Height (m)',
    'sst': 'SST (degC)',
    'wind_speed': 'Wind Speed (m/s)',
}


def load_station_data(data_dir: str, pattern: str):
    """Read every matching station Excel file in the directory."""
    xlsx_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not xlsx_files:
        raise FileNotFoundError(f"No file found in {data_dir} matching the pattern {pattern}.")

    station_data = {}
    for fpath in tqdm(xlsx_files, desc='Reading Excel files', unit='files'):
        fname = os.path.basename(fpath)
        df = pd.read_excel(fpath)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"File {fname} is missing columns: {missing}")

        station_df = pd.DataFrame({
            'time': pd.to_datetime(df[COL_TIME]),
            'd2m': df[COL_D2M].astype(float),
            'air_temp': df[COL_AIR_TEMP].astype(float),
            'mslp': df[COL_MSLP].astype(float),
            'swh': df[COL_SWH].astype(float),
            'sst': df[COL_SST].astype(float),
            'wind_speed': np.sqrt(df[COL_U10].astype(float) ** 2 +
                                  df[COL_V10].astype(float) ** 2),
        })
        station_data[fname] = station_df
    return station_data


def align_and_average(station_data: dict) -> tuple[pd.DataFrame, list]:
    """Time alignment + spatial averaging, returning the regionally averaged time series and
    the names of the variables that survived the alignment."""
    indexed_dfs = []
    for fname, sdf in station_data.items():
        idx_df = sdf.set_index('time')[ALL_VAR_NAMES]
        idx_df.columns = pd.MultiIndex.from_product([[fname], ALL_VAR_NAMES])
        indexed_dfs.append(idx_df)

    all_stations = pd.concat(indexed_dfs, axis=1, join='inner')

    merged = pd.DataFrame({
        var: all_stations.xs(var, axis=1, level=1).mean(axis=1)
        for var in ALL_VAR_NAMES
    })
    merged.index.name = 'time'
    merged = merged.reset_index()

    # Detect and drop variables that are entirely NaN
    valid_vars = []
    for var in ALL_VAR_NAMES:
        if not merged[var].isna().all():
            valid_vars.append(var)
    merged = merged.dropna(subset=valid_vars).reset_index(drop=True)

    # Kelvin -> Celsius
    for col in ['d2m', 'air_temp']:
        if col in valid_vars and merged[col].mean() > 200:
            merged[col] -= 273.15
    if 'sst' in valid_vars and merged['sst'].mean() > 200:
        merged['sst'] -= 273.15

    return merged, valid_vars


def run_pca(merged: pd.DataFrame, valid_vars: list):
    """Run Z-score standardization followed by PCA dimensionality reduction."""
    X = merged[valid_vars].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca, X_scaled


def plot_scree(explained_var_ratio, n_vars, output_dir):
    """Scree plot: variance ratio explained by each principal component."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    x = np.arange(1, n_vars + 1)
    bars = ax.bar(x, explained_var_ratio * 100, color='#4472C4',
                  edgecolor='white', linewidth=0.5, width=0.65, zorder=3)
    for bar, val in zip(bars, explained_var_ratio):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{val * 100:.1f}%', ha='center', va='bottom',
                fontsize=8, fontfamily='serif', fontweight='bold', color='#333')
    ax.set_xlabel('Principal Component', fontsize=10, fontfamily='serif')
    ax.set_ylabel('Explained Variance Ratio (%)', fontsize=10, fontfamily='serif')
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i}' for i in x], fontsize=9, fontfamily='serif')
    ax.set_ylim(0, max(explained_var_ratio * 100) * 1.25)
    ax.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.6, zorder=0)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_scree_plot.png'), dpi=300,
                facecolor='white', edgecolor='none')
    fig.savefig(os.path.join(output_dir, 'pca_scree_plot.pdf'),
                facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_loadings_bars(loadings, explained_var_ratio, english_short, output_dir):
    """Bar charts of the variable loadings on PC1 and PC2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    pos_color, neg_color = '#C00000', '#4472C4'

    for ax, pc_load, pc_idx, title_pct in [
        (ax1, loadings[:, 0], 0, explained_var_ratio[0] * 100),
        (ax2, loadings[:, 1], 1, explained_var_ratio[1] * 100),
    ]:
        sort_idx = np.argsort(np.abs(pc_load))[::-1]
        for k, idx in enumerate(sort_idx):
            val = pc_load[idx]
            color = pos_color if val >= 0 else neg_color
            ax.barh(k, val, color=color, edgecolor='white',
                    linewidth=0.5, height=0.55, alpha=0.88)
            x_text = val + 0.03 * np.sign(val) if abs(val) > 0.05 else 0.03
            ha = 'left' if val >= 0 else 'right'
            ax.text(x_text, k, f'{val:+.3f}', va='center', ha=ha,
                    fontsize=7.8, fontfamily='serif', fontweight='bold',
                    color='#333')
        ax.set_yticks(np.arange(len(sort_idx)))
        ax.set_yticklabels([english_short[i] for i in sort_idx],
                           fontsize=8.5, fontfamily='serif')
        ax.axvline(x=0, color='#333', linewidth=0.6)
        ax.set_xlim(-1.15, 1.15)
        ax.set_xlabel('Loading Coefficient', fontsize=9.5, fontfamily='serif')
        ax.set_title(f'PC{pc_idx + 1} Loadings ({title_pct:.1f}%)',
                     fontsize=10.5, fontfamily='serif', fontweight='bold', pad=10)
        ax.grid(axis='x', linestyle='--', linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle('PCA Variable Loadings: Contribution to Each Principal Component',
                 fontsize=11, fontfamily='serif', fontweight='bold', y=1.04)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_loadings_bars.png'), dpi=300,
                facecolor='white', edgecolor='none')
    fig.savefig(os.path.join(output_dir, 'pca_loadings_bars.pdf'),
                facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_scores(X_pca, merged, explained_var_ratio, output_dir):
    """Scatter plot of the PC1 and PC2 scores, colored by month."""
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    months = merged['time'].dt.month.values
    years = merged['time'].dt.year.values
    year_list = sorted(set(years))

    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=months, cmap='Spectral',
                    s=3, alpha=0.5, edgecolors='none', rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, aspect=25)
    cbar.set_label('Month', fontsize=9, fontfamily='serif')
    cbar.set_ticks([1, 3, 5, 7, 9, 11])
    cbar.set_ticklabels(['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])

    ax.axhline(y=0, color='#999', linewidth=0.4, linestyle='--')
    ax.axvline(x=0, color='#999', linewidth=0.4, linestyle='--')
    ax.set_xlabel(f'PC1 ({explained_var_ratio[0] * 100:.1f}%)',
                  fontsize=10, fontfamily='serif')
    ax.set_ylabel(f'PC2 ({explained_var_ratio[1] * 100:.1f}%)',
                  fontsize=10, fontfamily='serif')
    ax.set_title('PCA Score Scatter: PC1 vs PC2 (Colored by Month)',
                 fontsize=11, fontfamily='serif', fontweight='bold', pad=12)
    ax.set_aspect('equal')
    ax.text(0.98, 0.02, f'Years: {year_list[0]}-{year_list[-1]}',
            transform=ax.transAxes, fontsize=7, fontfamily='serif',
            ha='right', va='bottom', style='italic', color='#666')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_scores.png'), dpi=300,
                facecolor='white', edgecolor='none')
    fig.savefig(os.path.join(output_dir, 'pca_scores.pdf'),
                facecolor='white', edgecolor='none')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='PCA analysis for multi-station ocean meteorological data')
    parser.add_argument('--input', default='.', help='Directory containing station Excel files')
    parser.add_argument('--output', default='.', help='Output directory for charts and tables')
    parser.add_argument('--pattern', default='lat*_lon*.xlsx',
                        help='Station file glob pattern (default lat*_lon*.xlsx)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Read the station files
    station_data = load_station_data(args.input, args.pattern)
    print(f"Read {len(station_data)} station files")

    # 2. Time alignment + spatial averaging
    merged, valid_vars = align_and_average(station_data)
    print(f"Valid time steps: {len(merged):,}, valid variables: {valid_vars}")

    n_vars = len(valid_vars)
    english_short = [ENGLISH_LABELS[v] for v in valid_vars]

    # 3. PCA
    pca, X_pca, _ = run_pca(merged, valid_vars)
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained_var_ratio)
    loadings = pca.components_.T
    print(f"PC1 explained variance: {explained_var_ratio[0] * 100:.2f}%, "
          f"PC2: {explained_var_ratio[1] * 100:.2f}%, "
          f"cumulative: {cumsum[1] * 100:.2f}%")

    # 4. Draw the figures
    plot_scree(explained_var_ratio, n_vars, args.output)
    plot_loadings_bars(loadings, explained_var_ratio, english_short, args.output)
    plot_scores(X_pca, merged, explained_var_ratio, args.output)

    # 5. Save the data tables
    pd.DataFrame(loadings, index=english_short,
                 columns=[f'PC{i + 1}' for i in range(n_vars)]
                 ).to_csv(os.path.join(args.output, 'pca_loadings.csv'),
                          float_format='%.6f')

    pd.DataFrame({
        'PC': [f'PC{i + 1}' for i in range(n_vars)],
        'Eigenvalue': pca.explained_variance_,
        'Explained_Variance_Ratio': explained_var_ratio,
        'Cumulative_Variance_Ratio': cumsum,
    }).to_csv(os.path.join(args.output, 'pca_explained_variance.csv'),
              index=False, float_format='%.6f')

    scores_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_vars)])
    scores_df.insert(0, 'time', merged['time'])
    scores_df.to_csv(os.path.join(args.output, 'pca_scores.csv'),
                     index=False, float_format='%.6f')

    merged.to_csv(os.path.join(args.output, 'regional_mean_timeseries.csv'),
                  index=False, float_format='%.6f')

    print("PCA analysis finished, outputs saved to:", args.output)


if __name__ == '__main__':
    main()
