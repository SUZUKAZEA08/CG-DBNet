#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pearson correlation analysis of multi-station ocean meteorological data
=======================================================================
- Reads every .xlsx station file in the input directory
- Time alignment + spatial averaging
- Computes the Pearson correlation coefficient matrix (with significance testing)
- Writes journal-standard heatmap figures (PNG + PDF)

Usage:
    python correlation_analysis.py --input ./data --output ./output
    python correlation_analysis.py --input ./data --pattern "lat*_lon*.xlsx"
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# Global plotting style, aligned with journal requirements
# ============================================================================
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
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

        station_data[fname] = pd.DataFrame({
            'time': pd.to_datetime(df[COL_TIME]),
            'd2m': df[COL_D2M].astype(float),
            'air_temp': df[COL_AIR_TEMP].astype(float),
            'mslp': df[COL_MSLP].astype(float),
            'swh': df[COL_SWH].astype(float),
            'sst': df[COL_SST].astype(float),
            'wind_speed': np.sqrt(df[COL_U10].astype(float) ** 2 +
                                   df[COL_V10].astype(float) ** 2),
        })
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


def compute_correlation(merged: pd.DataFrame, valid_vars: list):
    """Compute the Pearson correlation coefficient matrix and the p-value matrix."""
    n_vars = len(valid_vars)
    corr_matrix = np.zeros((n_vars, n_vars))
    pval_matrix = np.zeros((n_vars, n_vars))

    for i in tqdm(range(n_vars), desc='Computing correlations', unit='pairs'):
        for j in range(n_vars):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                vi = merged[valid_vars[i]].values
                vj = merged[valid_vars[j]].values
                mask = ~np.isnan(vi) & ~np.isnan(vj)
                r, p = stats.pearsonr(vi[mask], vj[mask])
                corr_matrix[i, j] = r
                pval_matrix[i, j] = p

    return corr_matrix, pval_matrix


def create_heatmap(corr_matrix, display_labels, n_vars,
                   figsize, annot_fontsize, filename, output_dir):
    """Draw the journal-standard heatmap of the Pearson correlation matrix."""
    annot = np.empty((n_vars, n_vars), dtype=object)
    for i in range(n_vars):
        for j in range(n_vars):
            annot[i, j] = '1.00' if i == j else f'{corr_matrix[i, j]:.2f}'

    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    h = sns.heatmap(
        corr_matrix, annot=annot, fmt='', cmap=cmap,
        vmin=-1.0, vmax=1.0, center=0.0,
        linewidths=0.8, linecolor='white',
        cbar_kws={
            'label': "Pearson Correlation Coefficient (r)",
            'shrink': 0.82, 'aspect': 25,
            'ticks': [-1.0, -0.5, 0.0, 0.5, 1.0],
        },
        xticklabels=display_labels, yticklabels=display_labels,
        ax=ax, square=True,
        annot_kws={'fontsize': annot_fontsize, 'fontfamily': 'serif',
                   'color': 'black'},
    )

    cbar = h.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8, width=0.5, length=2.5)
    cbar.ax.set_ylabel("Pearson Correlation Coefficient (r)",
                       fontsize=9, fontfamily='serif', labelpad=6)
    cbar.outline.set_linewidth(0.5)

    # Diagonal cells: grey background
    for i in range(n_vars):
        rect = plt.Rectangle((i, i), 1, 1, fill=True, facecolor='#E8E8E8',
                             edgecolor='white', linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(i + 0.5, i + 0.5, '1.00', ha='center', va='center',
                fontsize=annot_fontsize, fontfamily='serif',
                fontstyle='italic', color='#555', fontweight='bold')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right',
                       fontsize=8, fontfamily='serif')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=8, fontfamily='serif')
    ax.tick_params(length=0)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300,
                facecolor='white', edgecolor='none')
    fig.savefig(os.path.join(output_dir, f'{filename}.pdf'),
                facecolor='white', edgecolor='none')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Pearson correlation analysis for multi-station ocean meteorological data')
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
    print(f"Valid time steps: {len(merged):,}, valid variables: {len(valid_vars)}")

    display_labels = [ENGLISH_LABELS[v] for v in valid_vars]
    n_vars = len(valid_vars)

    # 3. Compute the correlation coefficients
    corr_matrix, pval_matrix = compute_correlation(merged, valid_vars)

    corr_df = pd.DataFrame(corr_matrix, index=display_labels, columns=display_labels)
    pval_df = pd.DataFrame(pval_matrix, index=display_labels, columns=display_labels)

    # 4. Draw the heatmaps (standard version + compact version)
    create_heatmap(corr_matrix, display_labels, n_vars,
                   figsize=(8.5, 7.0), annot_fontsize=8.5,
                   filename='pearson_correlation_heatmap', output_dir=args.output)
    create_heatmap(corr_matrix, display_labels, n_vars,
                   figsize=(6.5, 5.5), annot_fontsize=7.5,
                   filename='pearson_correlation_heatmap_compact',
                   output_dir=args.output)

    # 5. Save the data tables
    corr_df.to_csv(os.path.join(args.output, 'pearson_correlation_matrix.csv'),
                   float_format='%.6f')
    pval_df.to_csv(os.path.join(args.output, 'pearson_pvalue_matrix.csv'),
                   float_format='%.6e')
    merged.to_csv(os.path.join(args.output, 'regional_mean_timeseries.csv'),
                  index=False, float_format='%.6f')

    print("Correlation analysis finished, output files saved to:", args.output)


if __name__ == '__main__':
    main()
