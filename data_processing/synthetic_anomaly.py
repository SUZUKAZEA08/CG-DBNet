#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synthetic anomaly injection
===========================
Injects four types of synthetic anomalies (out-of-range, gradient, spike, Gaussian)
into a time series, for robustness testing of anomaly detection models.

Usage:
    # Full mode: inject anomalies + save the workbook + draw the figure
    python synthetic_anomaly.py --input data.xlsx --output output.xlsx

    # Plot-only mode: read an existing anomaly-labelled workbook and draw the figure
    python synthetic_anomaly.py --mode plot --input output.xlsx --sheet test
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# Anomaly type annotations
ANOMALY_NAMES = {
    1: 'Out-of-Range',
    2: 'Gradient',
    3: 'Spike',
    4: 'Gaussian',
}
ANOMALY_COLORS = {
    1: '#C00000',   # out-of-range - red
    2: '#ED7D31',   # gradient - orange
    3: '#4472C4',   # spike - blue
    4: '#70AD47',   # Gaussian - green
}


# ============================================================================
# Core algorithm functions
# ============================================================================
def safe_read_two_sheets(file_path: Path) -> pd.DataFrame:
    """Read the first two sheets of the workbook and stack their date and sst columns."""
    xls = pd.ExcelFile(file_path)
    if len(xls.sheet_names) < 2:
        raise ValueError("The Excel workbook must contain at least two sheets.")

    first_name, second_name = xls.sheet_names[0], xls.sheet_names[1]
    df1 = pd.read_excel(file_path, sheet_name=first_name)
    df2 = pd.read_excel(file_path, sheet_name=second_name)

    required_cols = {"date", "sst"}
    for idx, df in enumerate([df1, df2], start=1):
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Sheet {idx} is missing the required columns: date, sst")

    merged = pd.concat(
        [df1[["date", "sst"]], df2[["date", "sst"]]],
        ignore_index=True,
    )
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["sst"] = pd.to_numeric(merged["sst"], errors="coerce")
    merged = merged.dropna(subset=["date", "sst"]).reset_index(drop=True)
    return merged


def split_6_2_2(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the series into train / validation / test in the ratio 6:2:2."""
    n = len(df)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train: n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    return train_df, val_df, test_df


def sample_out_of_range_value(normal_lower, normal_upper,
                             extreme_lower, extreme_upper) -> float:
    """Draw a value from one of the two out-of-range bands."""
    if random.random() < 0.5:
        return random.uniform(extreme_lower, normal_lower)
    return random.uniform(normal_upper + 0.1, extreme_upper)


def sample_gaussian_anomaly(mean_anomaly: float, std_anomaly: float,
                            extreme_lower, extreme_upper) -> float:
    """Draw a Gaussian value and clip it into the extreme range."""
    val = float(np.random.normal(mean_anomaly, std_anomaly))
    return max(extreme_lower, min(extreme_upper, val))


def allocate_counts(target: int, ratio_out_of_range: float,
                    ratio_gradient: float, ratio_spike: float,
                    ratio_gaussian: float) -> dict:
    """Allocate the target count across the four types; the remainder goes to the largest fractions."""
    raw = {
        "out_of_range": target * ratio_out_of_range,
        "gradient": target * ratio_gradient,
        "spike": target * ratio_spike,
        "gaussian": target * ratio_gaussian,
    }
    counts = {k: int(v) for k, v in raw.items()}
    used = sum(counts.values())
    remain = target - used
    if remain > 0:
        frac_sorted = sorted(raw.items(),
                             key=lambda x: (x[1] - int(x[1])), reverse=True)
        for i in range(remain):
            counts[frac_sorted[i % len(frac_sorted)][0]] += 1
    return counts


def inject_anomalies(test_df: pd.DataFrame, train_df: pd.DataFrame,
                     cfg) -> pd.DataFrame:
    """
    Inject four anomaly types into the test split.
    The original sst column is left unchanged; the new sst_anomaly column holds the
    values after injection.
    label column: 0 = normal, 1 = out-of-range, 2 = gradient, 3 = spike, 4 = Gaussian
    """
    out = test_df.copy().reset_index(drop=True)
    out["sst_anomaly"] = out["sst"].copy()
    out["label"] = 0

    n = len(out)
    start_idx = cfg.anomaly_start_index
    if n <= start_idx:
        return out

    eligible = list(range(start_idx, n))
    target = max(1, int(round(n / cfg.anomaly_total_denominator)))
    target = min(target, len(eligible))
    if target <= 0:
        return out

    counts = allocate_counts(target, cfg.ratio_out_of_range,
                             cfg.ratio_gradient, cfg.ratio_spike,
                             cfg.ratio_gaussian)
    count_range = counts["out_of_range"]
    count_gradient = counts["gradient"]
    count_spike = counts["spike"]
    count_gaussian = counts["gaussian"]

    mean_normal = float(train_df["sst"].mean())
    std_normal = float(train_df["sst"].std())
    if not np.isfinite(std_normal) or std_normal <= 0:
        std_normal = 1.0
    mean_anomaly = mean_normal * cfg.gaussian_mean_scale
    std_anomaly = std_normal * cfg.gaussian_std_scale

    remaining = set(eligible)

    def pick(pool, k):
        if k <= 0 or not pool:
            return []
        k = min(k, len(pool))
        return random.sample(sorted(pool), k)

    # 1) Spikes first (they need both neighbouring values)
    spike_pool = {i for i in remaining if 1 <= i <= n - 2}
    idx_spike = pick(spike_pool, count_spike)
    remaining -= set(idx_spike)

    # 2) Gradient shifts (they need the previous value)
    grad_pool = {i for i in remaining if i >= 1}
    idx_gradient = pick(grad_pool, count_gradient)
    remaining -= set(idx_gradient)

    # 3) Out-of-range values
    idx_range = pick(remaining, count_range)
    remaining -= set(idx_range)

    # 4) Gaussian values
    idx_gaussian = pick(remaining, count_gaussian)
    remaining -= set(idx_gaussian)

    assigned = len(idx_range) + len(idx_spike) + len(idx_gradient) + len(idx_gaussian)
    if assigned < target:
        needed = target - assigned
        fallback = sorted(remaining)
        idx_gaussian.extend(fallback[:needed])

    # Write the injected values into the sst_anomaly column
    for i in idx_range:
        out.at[i, "sst_anomaly"] = sample_out_of_range_value(
            cfg.normal_lower, cfg.normal_upper,
            cfg.extreme_lower, cfg.extreme_upper)
        out.at[i, "label"] = 1

    for i in idx_gradient:
        if i < 1:
            continue
        prev_val = float(out.at[i - 1, "sst_anomaly"])
        delta = random.uniform(cfg.gradient_delta_min, cfg.gradient_delta_max)
        sign = 1.0 if random.random() < 0.5 else -1.0
        out.at[i, "sst_anomaly"] = max(
            cfg.extreme_lower,
            min(cfg.extreme_upper, prev_val + sign * delta))
        out.at[i, "label"] = 2

    for i in idx_spike:
        if i <= 0 or i >= n - 1:
            continue
        prev_val = float(out.at[i - 1, "sst_anomaly"])
        next_val = float(out.at[i + 1, "sst_anomaly"])
        mid = (prev_val + next_val) / 2.0
        half_diff = abs(next_val - prev_val) / 2.0
        desired_spike_score = random.uniform(cfg.spike_score_min, cfg.spike_score_max)
        amplitude = half_diff + desired_spike_score
        sign = 1.0 if random.random() < 0.5 else -1.0
        out.at[i, "sst_anomaly"] = max(
            cfg.extreme_lower,
            min(cfg.extreme_upper, mid + sign * amplitude))
        out.at[i, "label"] = 3

    for i in idx_gaussian:
        out.at[i, "sst_anomaly"] = sample_gaussian_anomaly(
            mean_anomaly, std_anomaly,
            cfg.extreme_lower, cfg.extreme_upper)
        out.at[i, "label"] = 4

    return out


def plot_anomaly_zoom(test_df: pd.DataFrame, output_dir: Path, cfg):
    """Zoomed figure over a local window: clean reference vs anomalous series, colored by type."""
    df = test_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    anomaly_mask = df["label"] != 0
    anomaly_times = df.loc[anomaly_mask, "date"].values
    if len(anomaly_times) == 0:
        print("No anomaly in the test split, skipping the figure.")
        return

    # Determine the zoomed time window
    if cfg.zoom_time_range is not None:
        t_start = pd.Timestamp(cfg.zoom_time_range[0])
        t_end = pd.Timestamp(cfg.zoom_time_range[1])
    else:
        t_start = pd.Timestamp(anomaly_times.min()) - pd.Timedelta(hours=12)
        window = pd.Timedelta(days=cfg.zoom_days)
        if len(anomaly_times) >= 3:
            best_start = t_start
            best_count = 0
            candidates = pd.date_range(anomaly_times.min(), anomaly_times.max(), freq='6h')
            for c in candidates:
                cnt = ((anomaly_times >= c) & (anomaly_times <= c + window)).sum()
                if cnt > best_count:
                    best_count = cnt
                    best_start = c
            t_start = best_start
        t_end = t_start + window

    zoom_mask = (df["date"] >= t_start) & (df["date"] <= t_end)
    seg = df.loc[zoom_mask].copy()
    if len(seg) == 0:
        print("No data inside the selected window, skipping the figure.")
        return

    seg_anomaly = seg[seg["label"] != 0]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    # Bottom layer: light grey dashed line, the clean reference values
    ax.plot(seg["date"], seg["sst"],
            color='#AAAAAA', linestyle='--', linewidth=0.8,
            alpha=0.7, label='Original (clean)', zorder=1)
    # Upper layer: dark blue solid line, the series with injected anomalies
    ax.plot(seg["date"], seg["sst_anomaly"],
            color='#1F4E79', linestyle='-', linewidth=0.8,
            alpha=0.9, label='With anomalies', zorder=2)

    for label_val in [1, 2, 3, 4]:
        pts = seg_anomaly[seg_anomaly["label"] == label_val]
        if len(pts) == 0:
            continue
        color = ANOMALY_COLORS[label_val]
        name = ANOMALY_NAMES[label_val]
        ax.scatter(pts["date"], pts["sst_anomaly"],
                   s=60, facecolors='none', edgecolors=color,
                   linewidths=1.5, zorder=4, label=f'{name} (n={len(pts)})')
        first = pts.iloc[0]
        ax.annotate(name, (first["date"], first["sst_anomaly"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=6.5, fontfamily='serif', color=color,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=0.6, connectionstyle='arc3,rad=0.2'),
                    zorder=5)

    ax.set_xlabel('Time', fontsize=10, fontfamily='serif')
    ax.set_ylabel('Sea surface temperature (SST, degC)', fontsize=10, fontfamily='serif')
    ax.set_title(f'Anomaly Injection Detail: {t_start.date()} to {t_end.date()}',
                 fontsize=11, fontfamily='serif', fontweight='bold', pad=10)

    time_span = (t_end - t_start).days
    if time_span <= 3:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    elif time_span <= 14:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

    ax.legend(fontsize=7, loc='upper right', framealpha=0.9,
              edgecolor='#CCCCCC', fancybox=False, ncol=2)
    ax.grid(linestyle='--', linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / 'anomaly_injection_zoom.png', dpi=300,
                facecolor='white', edgecolor='none')
    fig.savefig(output_dir / 'anomaly_injection_zoom.pdf',
                facecolor='white', edgecolor='none')
    plt.close(fig)


# ============================================================================
# Main workflow
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Synthetic anomaly injection')
    parser.add_argument('--mode', choices=['full', 'plot'], default='full',
                        help="'full' = inject anomalies + save + plot; 'plot' = plot an existing Excel file only")
    parser.add_argument('--input', type=str, required=True,
                        help='Input Excel file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output Excel file path (default: input filename with suffix)')
    parser.add_argument('--sheet', type=str, default='test',
                        help='Sheet name read in plot mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Anomaly ratio parameters
    parser.add_argument('--anomaly-denominator', type=int, default=34,
                        help='Anomaly ratio denominator (anomaly:normal ~ 1:denominator)')
    parser.add_argument('--anomaly-start-index', type=int, default=49,
                        help='First N points are exempt from anomaly injection')
    parser.add_argument('--ratio-out-of-range', type=float, default=0.2)
    parser.add_argument('--ratio-gradient', type=float, default=0.3)
    parser.add_argument('--ratio-spike', type=float, default=0.3)
    parser.add_argument('--ratio-gaussian', type=float, default=0.2)

    # Range parameters
    parser.add_argument('--normal-lower', type=float, default=-2.6)
    parser.add_argument('--normal-upper', type=float, default=40.0)
    parser.add_argument('--extreme-lower', type=float, default=-20.0)
    parser.add_argument('--extreme-upper', type=float, default=50.0)
    parser.add_argument('--gradient-delta-min', type=float, default=1.0)
    parser.add_argument('--gradient-delta-max', type=float, default=3.0)
    parser.add_argument('--spike-score-min', type=float, default=1.0)
    parser.add_argument('--spike-score-max', type=float, default=3.0)
    parser.add_argument('--gaussian-mean-scale', type=float, default=0.8)
    parser.add_argument('--gaussian-std-scale', type=float, default=0.7071)

    # Figure parameters
    parser.add_argument('--zoom-days', type=int, default=5,
                        help='Auto-select days with highest anomaly density')
    parser.add_argument('--zoom-time-range', type=str, nargs=2, default=None,
                        help='Manual time range, e.g.: 2022-07-01 2022-07-06')

    args = parser.parse_args()

    # Bundle the parsed arguments into a configuration object
    class Cfg:
        pass
    cfg = Cfg()
    cfg.anomaly_total_denominator = args.anomaly_denominator
    cfg.anomaly_start_index = args.anomaly_start_index
    cfg.ratio_out_of_range = args.ratio_out_of_range
    cfg.ratio_gradient = args.ratio_gradient
    cfg.ratio_spike = args.ratio_spike
    cfg.ratio_gaussian = args.ratio_gaussian
    cfg.normal_lower = args.normal_lower
    cfg.normal_upper = args.normal_upper
    cfg.extreme_lower = args.extreme_lower
    cfg.extreme_upper = args.extreme_upper
    cfg.gradient_delta_min = args.gradient_delta_min
    cfg.gradient_delta_max = args.gradient_delta_max
    cfg.spike_score_min = args.spike_score_min
    cfg.spike_score_max = args.spike_score_max
    cfg.gaussian_mean_scale = args.gaussian_mean_scale
    cfg.gaussian_std_scale = args.gaussian_std_scale
    cfg.zoom_days = args.zoom_days
    cfg.zoom_time_range = args.zoom_time_range

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_path = Path(args.input)

    if args.mode == 'full':
        # Full mode: read -> inject anomalies -> save -> plot
        merged_df = safe_read_two_sheets(input_path)
        train_df, val_df, test_df = split_6_2_2(merged_df)
        test_with_anomaly = inject_anomalies(test_df, train_df, cfg)

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_name(
                f"{input_path.stem}_test_with_anomalies.xlsx")

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            train_df.to_excel(writer, sheet_name="train", index=False)
            val_df.to_excel(writer, sheet_name="val", index=False)
            test_with_anomaly.to_excel(writer, sheet_name="test", index=False)

        print(f"Output file: {output_path}")
        print(f"train/val/test rows: {len(train_df)}/{len(val_df)}/{len(test_with_anomaly)}")
        anomaly_cnt = int((test_with_anomaly["label"] != 0).sum())
        print(f"Anomalies in the test split: {anomaly_cnt}")
        for label_val in [1, 2, 3, 4]:
            cnt = int((test_with_anomaly["label"] == label_val).sum())
            print(f"  - {ANOMALY_NAMES[label_val]}: {cnt}")

        plot_anomaly_zoom(test_with_anomaly, output_path.parent, cfg)

    elif args.mode == 'plot':
        # Plot-only mode
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        df = pd.read_excel(input_path, sheet_name=args.sheet)
        required_cols = {"date", "sst", "sst_anomaly", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"File is missing the required columns: {missing}")

        print(f"Read file: {input_path}, sheet: {args.sheet}")
        print(f"Total rows: {len(df)}, anomalies: {int((df['label'] != 0).sum())}")
        plot_anomaly_zoom(df, input_path.parent, cfg)


if __name__ == '__main__':
    main()
