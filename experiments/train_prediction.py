"""
Prediction experiment entry point (CG-DBNet)
============================================
Usage (run from the package root):
    python experiments/train_prediction.py --data_dir ./data/train --test_path ./data/test.xlsx \
        --output_dir ./outputs/prediction --model proposed --epochs 100

Standalone argparse entry point. Implements:
  - RevIN + multi-scale MSAC CNN + dual-attention GAP + adaptive affine fusion (shared backbone)
  - Differenced target (y = t_curr - t_prev), MSE on mu
  - A/B evaluation (in-grid test set / external buoy station)
  - Ablation switches (--no_msac / --no_gap / --no_pos / --no_affine / --serial_gru / --no_conv3 ...)
  - Loss history and prediction plots
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make `from src.xxx import ...` work both from experiments/ and from the package root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from src.model import build_pred_model  # noqa: E402
from src.data_utils import (  # noqa: E402
    DataProcessor, MultiFileTimeSeriesDataset,
    REGION_LAT_MAX, REGION_LAT_MIN, REGION_LON_MAX, REGION_LON_MIN,
)
from src.metrics import calculate_metrics  # noqa: E402

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================== Training (neural networks, MSE on mu) =====================
def save_loss_history(history, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 100 + "\n")
        f.write(f"{'Epoch':>6} | {'Train_Loss':>12} | {'Val_Loss':>12} | {'MAE':>10} | "
                f"{'RMSE':>10} | {'R2':>10} | {'MAPE(%)':>10} | {'SMAPE(%)':>10} | {'MASE':>10}\n"
                + "-" * 100 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:>6} | {history['train_loss'][i]:>12.6f} | {history['val_loss'][i]:>12.6f} | "
                    f"{history['val_mae'][i]:>10.6f} | {history['val_rmse'][i]:>10.6f} | "
                    f"{history['val_r2'][i]:>10.6f} | {history['val_mape'][i]:>10.4f} | "
                    f"{history['val_smape'][i]:>10.4f} | {history['val_mase'][i]:>10.6f}\n")
        f.write("=" * 100 + "\n")
        if history['val_loss']:
            f.write(f"Best epoch: {int(np.argmin(history['val_loss']))+1} "
                    f"(Val Loss: {min(history['val_loss']):.6f})\n")


def save_config_to_file(config, filepath, extra: dict | None = None):
    """Write the non-callable attributes of a configuration object to JSON.
    extra adds run-level entries that are not attributes of the configuration object."""
    d = {}
    for k in sorted(dir(config)):
        if k.startswith('__') or callable(getattr(config, k)):
            continue
        v = getattr(config, k)
        d[k] = str(v) if not isinstance(v, (int, float, bool, str, list)) else v
    for k, v in (extra or {}).items():
        d[k] = str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=4, ensure_ascii=False)


def train_model(name, model, train_loader, val_loader, target_idx, output_dir, cfg, device):
    print(f"\n{'='*60}\n[Prediction] Training {name} | parameters "
          f"{sum(p.numel() for p in model.parameters()):,}\n{'='*60}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [],
               'val_r2': [], 'val_mape': [], 'val_smape': [], 'val_mase': []}
    best_loss = float('inf')
    patience_counter = 0
    ckpt = os.path.join(output_dir, 'best_model.pth')
    for epoch in range(cfg.EPOCHS):
        model.train()
        tl = 0
        for x, y in tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1}/{cfg.EPOCHS}",
                         disable=not sys.stdout.isatty()):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mu, _ = model(x)
            loss = criterion(mu.squeeze(1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()
            tl += loss.item()
        avg_train = tl / len(train_loader)

        model.eval()
        vl = 0
        mu_list, act_list = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mu, _ = model(x)
                vl += criterion(mu.squeeze(1), y).item()
                last_val = x[:, -1, target_idx]
                mu_list.append((last_val + mu.squeeze(1)).cpu().numpy())
                act_list.append((last_val + y).cpu().numpy())
        avg_val = vl / len(val_loader)
        vm = calculate_metrics(np.concatenate(act_list).flatten(), np.concatenate(mu_list).flatten())
        for k, v in [('train_loss', avg_train), ('val_loss', avg_val), ('val_mae', vm['MAE']),
                     ('val_rmse', vm['RMSE']), ('val_r2', vm['R2']), ('val_mape', vm['MAPE']),
                     ('val_smape', vm['SMAPE']), ('val_mase', vm['MASE'])]:
            history[k].append(v)
        print(f"[{name}] Epoch {epoch+1} | Train {avg_train:.4f} | Val {avg_val:.4f} | "
              f"MAE {vm['MAE']:.4f} | RMSE {vm['RMSE']:.4f} | R2 {vm['R2']:.4f} | MASE {vm['MASE']:.4f}")
        scheduler.step(avg_val)
        if avg_val < best_loss:
            best_loss = avg_val
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict()}, ckpt)
        else:
            patience_counter += 1
        if patience_counter >= cfg.PATIENCE:
            print(f"[{name}] Early stopping (epoch {epoch+1})")
            break
    model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
    save_loss_history(history, os.path.join(output_dir, 'loss_history.txt'))
    print(f"[{name}] Training finished | best val loss: {best_loss:.4f}")
    return model, history, float(best_loss)


def evaluate_loader(model, loader, target_idx, device):
    model.eval()
    mu_list, act_list = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Predicting", disable=not sys.stdout.isatty()):
            x = x.to(device)
            mu, _ = model(x)
            last_val = x[:, -1, target_idx]
            mu_list.append((last_val + mu.squeeze(1)).cpu().numpy())
            act_list.append((last_val.cpu() + y).numpy())
    actuals = np.concatenate(act_list).flatten()
    preds = np.concatenate(mu_list).flatten()
    return actuals, preds, calculate_metrics(actuals, preds)


def save_eval_outputs(actuals, preds, metrics, output_dir, dataset_name, target_col):
    pd.DataFrame({'Actual': actuals, 'Predicted': preds}).to_csv(
        os.path.join(output_dir, f'predictions_{dataset_name}.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(output_dir, f'metrics_{dataset_name}.txt'), 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            if isinstance(v, float) and np.isnan(v):
                continue
            f.write(f"{k}: {v:.6f}\n")
    plot_len = min(3000, len(actuals))
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.plot(range(plot_len), actuals[:plot_len], label='Actual', color='black', alpha=0.7, lw=1)
    ax.plot(range(plot_len), preds[:plot_len], label='Predicted', color='#FF7F0E', lw=1.2)
    ax.set_title(f"{dataset_name} | MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
                 f"R2={metrics['R2']:.4f}")
    ax.set_xlabel("Time step")
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'prediction_{dataset_name}.png'), dpi=120)
    plt.close(fig)


# ===================== Configuration =====================
class PredConfig:
    """Experiment configuration (paths and hyper-parameters are overridden by argparse)."""
    TRAIN_DIR = './data'
    TEST_PATH = './data/test.xlsx'
    TRAIN_MAX_FILES = None
    TRAIN_FILE_START_INDEX = 0
    GRID_FILE_PATTERN = r'(?i)(?:lat|latitude)[\s_=]*(-?\d+(?:\.\d+)?)[\s_,;+-]*(?:lon|longitude)[\s_=]*(-?\d+(?:\.\d+)?)'

    TIME_COL = 'time'
    TARGET_COL = 'sst'
    FEATURE_COLS = ['air_temperature', 'sea_level_pressure']

    # Study region used by the geospatial features; the values live in src.data_utils.
    LAT_MIN, LAT_MAX = REGION_LAT_MIN, REGION_LAT_MAX
    LON_MIN, LON_MAX = REGION_LON_MIN, REGION_LON_MAX
    SPATIAL_FEATURE_NAMES = [
        'geo_lat_sin', 'geo_lat_cos', 'geo_lon_sin', 'geo_lon_cos',
        'geo_lat_norm', 'geo_lon_norm', 'geo_dist_center',
    ]

    SEQ_LEN = 48
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    HIDDEN_DIM = 64
    DROPOUT = 0.15
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    PATIENCE = 15
    GRAD_CLIP_NORM = 1.0
    SIGMA_MIN = 1e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="CG-DBNet prediction experiment")
    p.add_argument('--data_dir', type=str, default='./data', help='Training grid-point data directory')
    p.add_argument('--test_path', type=str, default='./data/test.xlsx', help='External test file/directory')
    p.add_argument('--file_pattern', type=str, default=None,
                   help='Regular expression a grid-point filename must contain to be loaded '
                        '(default: the PredConfig.GRID_FILE_PATTERN value). The expression must '
                        'contain two capturing groups: the first captures the latitude, the second '
                        'captures the longitude. If your filenames use non-Latin characters (for '
                        'example local-language words for latitude and longitude), pass a custom '
                        'regular expression here.')
    p.add_argument('--output_dir', type=str, default='./outputs/prediction', help='Output directory')
    p.add_argument('--model', type=str, default='proposed', choices=['proposed'])
    p.add_argument('--target_col', type=str, default='sst')
    p.add_argument('--time_col', type=str, default='time')
    p.add_argument('--feature_cols', type=str, nargs='*', default=['air_temperature', 'sea_level_pressure'],
                   help='External meteorological feature columns (empty list to disable)')
    # hyper-parameters
    p.add_argument('--seq_len', type=int, default=48)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.15)
    p.add_argument('--patience', type=int, default=15)
    # ablation switches
    p.add_argument('--no_pos', action='store_true', help='Disable cyclic positional encoding')
    p.add_argument('--no_msac', action='store_true', help='Disable multi-scale MSAC CNN')
    p.add_argument('--no_gap', action='store_true', help='Disable dual-attention GAP (DAGAP)')
    p.add_argument('--no_affine', action='store_true', help='Disable adaptive affine fusion (use addition)')
    p.add_argument('--serial_gru', action='store_true', help='Direct GRU (skip MSAC/DAGAP)')
    p.add_argument('--no_conv3', action='store_true')
    p.add_argument('--no_conv5', action='store_true')
    p.add_argument('--no_conv7', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    cfg = PredConfig()
    cfg.TRAIN_DIR = args.data_dir
    cfg.TEST_PATH = args.test_path
    if args.file_pattern:
        cfg.GRID_FILE_PATTERN = args.file_pattern
    cfg.TARGET_COL = args.target_col
    cfg.TIME_COL = args.time_col
    cfg.FEATURE_COLS = list(args.feature_cols) if args.feature_cols else []
    cfg.SEQ_LEN = args.seq_len
    cfg.BATCH_SIZE = args.batch_size
    cfg.EPOCHS = args.epochs
    cfg.LEARNING_RATE = args.lr
    cfg.HIDDEN_DIM = args.hidden_dim
    cfg.DROPOUT = args.dropout
    cfg.PATIENCE = args.patience
    device = cfg.DEVICE

    # Ablation switches
    toggles = {
        "use_pos": not args.no_pos,
        "use_msac": not args.no_msac,
        "use_dagap": not args.no_gap,
        "use_affine_fusion": not args.no_affine,
        "context_direct_gru": args.serial_gru,
        "use_conv3": not args.no_conv3,
        "use_conv5": not args.no_conv5,
        "use_conv7": not args.no_conv7,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_config_to_file(cfg, os.path.join(args.output_dir, 'training_config.json'),
                        extra={"model_kind": args.model, "ablation_toggles": toggles})

    # Data (the cache file is written into the output directory)
    processor = DataProcessor(cfg, cache_dir=args.output_dir)
    (train_arrays, val_arrays, internal_test_arrays, external_test_arrays,
     input_cols, target_idx, n_spatial, augment_indices, file_info_list) = processor.load_and_process()

    # Loaders for the A/B evaluation
    internal_test_ds = MultiFileTimeSeriesDataset(internal_test_arrays, cfg.SEQ_LEN, target_idx,
                                                 mode='test', augment_indices=augment_indices)
    internal_test_loader = DataLoader(internal_test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                     pin_memory=True, num_workers=4)
    external_test_loader = None
    if external_test_arrays:
        external_test_ds = MultiFileTimeSeriesDataset(external_test_arrays, cfg.SEQ_LEN, target_idx,
                                                     mode='test', augment_indices=augment_indices)
        external_test_loader = DataLoader(external_test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                         pin_memory=True, num_workers=4)

    train_ds = MultiFileTimeSeriesDataset(train_arrays, cfg.SEQ_LEN, target_idx,
                                         mode='train', augment_indices=augment_indices)
    val_ds = MultiFileTimeSeriesDataset(val_arrays, cfg.SEQ_LEN, target_idx,
                                       mode='val', augment_indices=augment_indices)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, pin_memory=True,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=4)
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]
    model = build_pred_model(args.model, input_dim, n_spatial, cfg.SEQ_LEN, cfg.HIDDEN_DIM,
                             cfg.DROPOUT, cfg.SIGMA_MIN, toggles).to(device)
    model, history, best_val = train_model(args.model, model, train_loader, val_loader,
                                          target_idx, args.output_dir, cfg, device)
    print(f"\n>>> [DatasetA] Evaluating the in-grid test set")
    a_act, a_pred, a_metrics = evaluate_loader(model, internal_test_loader, target_idx, device)
    save_eval_outputs(a_act, a_pred, a_metrics, args.output_dir, "DatasetA_Test", cfg.TARGET_COL)
    b_metrics = None
    if external_test_loader is not None:
        print(">>> [DatasetB] Evaluating the external buoy station")
        b_act, b_pred, b_metrics = evaluate_loader(model, external_test_loader, target_idx, device)
        save_eval_outputs(b_act, b_pred, b_metrics, args.output_dir, "DatasetB_External", cfg.TARGET_COL)

    b_rmse = b_metrics['RMSE'] if b_metrics else float('nan')
    b_r2 = b_metrics['R2'] if b_metrics else float('nan')
    print(f"\n=== {args.model} | A RMSE={a_metrics['RMSE']:.4f} R2={a_metrics['R2']:.4f} | "
          f"B RMSE={b_rmse:.4f} R2={b_r2:.4f} ===")
    print(f"[Done] Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
