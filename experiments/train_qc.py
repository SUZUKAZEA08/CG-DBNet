"""
Quality-control experiment entry point (CG-DBNet)
=================================================
Usage (run from the package root):
    python experiments/train_qc.py --data_file ./data/qc.xlsx --output_dir ./outputs/qc \
        --model proposed --epochs 100

Standalone argparse entry point. Implements:
  - Gaussian NLL training (mu, sigma in the StandardScaler-normalized space)
  - k-sigma calibration (k is the TARGET_COVERAGE quantile of the validation
    standardized residuals |actual - mu| / sigma, default 0.95)
  - Sequential QC loop (predict one step ahead -> flag outside +/- k*sigma ->
    replace a flagged value with the prediction, at most 2 consecutive replacements)
  - Classification metrics (F1 / Precision / Recall / Accuracy / Specificity / ROC_AUC)
  - Ablation switches (--model abl_no_msac / abl_no_dagap / abl_no_pos / abl_affine_add /
    abl_serial_gru)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from src.model import build_qc_model  # noqa: E402
from src.data_utils import (  # noqa: E402
    DataConfig, SeqDataset, build_feature_matrix, prepare_datasets, save_json,
)
from src.metrics import (  # noqa: E402
    compute_dynamic_k_by_coverage, evaluate_cls, gaussian_nll_loss,
)

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


# ===================== Neural-network inference and training (Gaussian NLL + k calibration) =====================
def _run_inf(model, loader, target_scaler, device):
    model.eval()
    mu_list, sigma_list, actual_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, sigma = model(x)
            mu_list.append(mu.cpu().numpy())
            sigma_list.append(sigma.cpu().numpy())
            actual_list.append(y.numpy())
    mu_scaled = np.concatenate(mu_list).flatten()
    sigma_scaled = np.concatenate(sigma_list).flatten()
    actual_scaled = np.concatenate(actual_list).flatten()
    mu_inv = target_scaler.inverse_transform(mu_scaled.reshape(-1, 1)).flatten()
    actual_inv = target_scaler.inverse_transform(actual_scaled.reshape(-1, 1)).flatten()
    sigma_inv = np.clip(sigma_scaled * float(target_scaler.scale_[0]), 1e-8, None)
    return actual_inv, mu_inv, sigma_inv


def train_qc_nn(name, model, train_loader, val_loader, target_scaler, output_dir, cfg, device):
    print(f"\n{'='*60}\n[QC] Training {name} | parameters "
          f"{sum(p.numel() for p in model.parameters()):,} | loss: Gaussian NLL\n{'='*60}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val = float('inf')
    patience = 0
    history = {"train_nll": [], "val_nll": []}
    ckpt = os.path.join(output_dir, 'model.pth')
    for epoch in range(cfg.EPOCHS):
        model.train()
        tl = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mu, sigma = model(x)
            loss = gaussian_nll_loss(mu, sigma, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()
            tl += loss.item()
        avg_train = tl / max(len(train_loader), 1)

        model.eval()
        vl = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mu, sigma = model(x)
                vl += gaussian_nll_loss(mu, sigma, y).item()
        avg_val = vl / max(len(val_loader), 1)
        history["train_nll"].append(avg_train)
        history["val_nll"].append(avg_val)
        print(f"[{name}] Epoch {epoch+1} | Train NLL: {avg_train:.4f} | Val NLL: {avg_val:.4f}")
        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            patience = 0
            torch.save({"model_state_dict": model.state_dict()}, ckpt)
        else:
            patience += 1
        if patience >= cfg.PATIENCE:
            print(f"[{name}] Early stopping (epoch {epoch+1})")
            break
    model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])

    # k calibration: k is the target_coverage quantile of the validation standardized
    # residuals, no additional offset is applied
    val_actual, val_mu, val_sigma = _run_inf(model, val_loader, target_scaler, device)
    k_sigma, emp_cov, _ = compute_dynamic_k_by_coverage(val_actual, val_mu, val_sigma,
                                                        cfg.TARGET_COVERAGE)
    save_json(os.path.join(output_dir, 'k_sigma_config.json'), {
        "target_coverage": cfg.TARGET_COVERAGE, "k_sigma": k_sigma,
        "empirical_coverage_validation": emp_cov,
        "sigma_stats_validation": {
            "mean": float(np.mean(val_sigma)), "std": float(np.std(val_sigma)),
            "min": float(np.min(val_sigma)), "max": float(np.max(val_sigma))},
    })
    save_json(os.path.join(output_dir, 'train_log.json'),
              {"history": history, "best_val": best_val, "epochs_run": len(history["train_nll"])})
    with open(os.path.join(output_dir, 'loss_history.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 60 + "\n")
        f.write(f"{'Epoch':>6} | {'Train_NLL':>12} | {'Val_NLL':>12}\n" + "-" * 60 + "\n")
        for i in range(len(history['train_nll'])):
            f.write(f"{i+1:>6} | {history['train_nll'][i]:>12.6f} | {history['val_nll'][i]:>12.6f}\n")
        f.write("=" * 60 + f"\nBest epoch: {int(np.argmin(history['val_nll']))+1} "
                f"(Val NLL: {min(history['val_nll']):.6f})\n"
                f"k_sigma ({cfg.TARGET_COVERAGE:.0%} quantile of the validation "
                f"standardized residuals): {k_sigma:.6f}\n")
    print(f"[{name}] Training finished | best val NLL: {best_val:.4f} | "
          f"k_sigma={k_sigma:.4f} ({cfg.TARGET_COVERAGE:.0%} quantile)")
    return model, k_sigma, history, float(best_val)


# ===================== Sequential QC loop (neural network) =====================
def sequential_qc(model, df_raw, input_cols, scaler, target_scaler, k_sigma,
                   feature_cols, seq_len, time_col, target_col, device,
                   max_consecutive=2):
    """Predict one step ahead at every point, flag values outside +/- k*sigma and feed the
    prediction back as the next input, at most max_consecutive times in a row."""
    df = df_raw.copy().reset_index(drop=True)
    corrected_target = df[target_col].copy()
    scale_factor = float(target_scaler.scale_[0])

    actual_list, mu_list, sigma_list = [], [], []
    lower_list, upper_list = [], []
    anomaly_list, replaced_list, corrected_list, k_list = [], [], [], []
    consecutive = 0

    for t in range(seq_len, len(df)):
        features_all = build_feature_matrix(df, corrected_target, target_col,
                                            feature_cols, input_cols, time_col=time_col)
        features_scaled = scaler.transform(features_all)
        x_window = torch.FloatTensor(features_scaled[t - seq_len: t]).unsqueeze(0).to(device)
        with torch.no_grad():
            mu_t, sigma_t = model(x_window)
        mu_scaled = float(mu_t.squeeze().cpu().item())
        sigma_scaled = float(sigma_t.squeeze().cpu().item())
        actual_scaled = float(features_scaled[t, -1])

        mu_inv = float(target_scaler.inverse_transform([[mu_scaled]]).item())
        actual_inv = float(target_scaler.inverse_transform([[actual_scaled]]).item())
        sigma_inv = max(sigma_scaled * scale_factor, 1e-8)
        lower = mu_inv - k_sigma * sigma_inv
        upper = mu_inv + k_sigma * sigma_inv

        is_anomaly = int((actual_inv < lower) or (actual_inv > upper))
        if is_anomaly:
            if consecutive < max_consecutive:
                corrected_target.iloc[t] = mu_inv
                replaced = 1
                consecutive += 1
            else:
                corrected_target.iloc[t] = actual_inv
                replaced = 0
                consecutive = 0
        else:
            corrected_target.iloc[t] = actual_inv
            replaced = 0
            consecutive = 0

        k_value = abs(actual_inv - mu_inv) / sigma_inv
        actual_list.append(actual_inv)
        mu_list.append(mu_inv)
        sigma_list.append(sigma_inv)
        lower_list.append(lower)
        upper_list.append(upper)
        anomaly_list.append(is_anomaly)
        replaced_list.append(replaced)
        corrected_list.append(corrected_target.iloc[t])
        k_list.append(k_value)

    return {
        "actual": np.asarray(actual_list), "mu": np.asarray(mu_list),
        "sigma": np.asarray(sigma_list), "lower": np.asarray(lower_list),
        "upper": np.asarray(upper_list), "is_anomaly": np.asarray(anomaly_list).astype(int),
        "replaced": np.asarray(replaced_list).astype(int), "corrected": np.asarray(corrected_list),
        "k": np.asarray(k_list),
    }


# ===================== Output saving =====================
def _save_qc_outputs(qc, cls_metrics, threshold_value, model_name, output_dir, target_col,
                     y_true=None):
    """Write qc_result.csv / metrics.txt / qc_plot.png.

    threshold_value is the calibrated k_sigma, so the acceptance band is
    mu +/- k_sigma * sigma and is drawn in that unit on the plot.
    """
    result_df = pd.DataFrame({
        "Actual": qc["actual"], "Mu": qc["mu"], "Sigma": qc["sigma"],
        "Lower": qc["lower"], "Upper": qc["upper"], "k_i": qc["k"],
        "IsAnomaly": qc["is_anomaly"], "ReplacedForNextStep": qc["replaced"],
        "CorrectedInputValue": qc["corrected"],
    })
    if y_true is not None:
        result_df["TrueLabel"] = np.asarray(y_true).astype(int)
    result_df.to_csv(os.path.join(output_dir, 'qc_result.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(output_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"ModelName: {model_name}\n")
        f.write(f"k_sigma: {threshold_value:.6f}\n")
        f.write(f"DetectedAnomalies: {int(np.sum(qc['is_anomaly']))}\n")
        f.write(f"TotalPoints: {len(qc['is_anomaly'])}\n")
        f.write(f"ReplacedCount: {int(np.sum(qc['replaced']))}\n")
        if cls_metrics:
            for k in ["Accuracy", "Precision", "Recall", "F1", "Specificity", "FPR", "FNR",
                      "TP", "TN", "FP", "FN"]:
                if k in cls_metrics:
                    f.write(f"{k}: {cls_metrics[k]}\n")
            if cls_metrics.get("ROC_AUC") is not None:
                f.write(f"ROC_AUC: {cls_metrics['ROC_AUC']}\n")
    band_label = f'+/-{threshold_value:.3f} sigma'
    plot_len = min(2000, len(qc["actual"]))
    x = np.arange(plot_len)
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.plot(x, qc["actual"][:plot_len], 'k', lw=1.1, label='Actual')
    ax.plot(x, qc["mu"][:plot_len], '#ff7f0e', lw=1.2, label='Predicted')
    ax.fill_between(x, qc["lower"][:plot_len], qc["upper"][:plot_len], color='#1f77b4',
                    alpha=0.16, label=band_label)
    anom_idx = np.where(qc["is_anomaly"][:plot_len] == 1)[0]
    if len(anom_idx):
        ax.scatter(anom_idx, qc["actual"][:plot_len][anom_idx], c='red', s=22, label='Detected')
    if y_true is not None and len(y_true) >= plot_len:
        ta = np.where(np.asarray(y_true[:plot_len]).astype(int) == 1)[0]
        if len(ta):
            ax.scatter(ta, qc["actual"][:plot_len][ta], c='#1f77b4', marker='x', s=34, label='True')
    if cls_metrics:
        ax.set_title(f"QC ({model_name}) | F1={cls_metrics['F1']:.4f} "
                     f"P={cls_metrics['Precision']:.4f} R={cls_metrics['Recall']:.4f}")
    else:
        ax.set_title(f"QC ({model_name})")
    ax.set_xlabel("Time step")
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'qc_plot.png'), dpi=120)
    plt.close(fig)


def _load_test_with_label(data_path, test_sheet, time_col, target_col, label_col_name,
                          feature_cols=()):
    """Load the test sheet.

    Feature columns that exist in the sheet are kept so that the sequential QC loop can
    rebuild the same input matrix that was used in training. The label column is used only
    when it is present; otherwise no classification metrics are computed.
    """
    df_raw = pd.read_excel(data_path, sheet_name=test_sheet)
    label_col = label_col_name if label_col_name in df_raw.columns else None
    if label_col:
        df_raw[label_col] = pd.to_numeric(df_raw[label_col], errors="coerce").fillna(0)
        df_raw[label_col] = (df_raw[label_col] > 0).astype(int)
    df_raw[time_col] = pd.to_datetime(df_raw[time_col], errors="coerce")
    df_raw[target_col] = pd.to_numeric(df_raw[target_col], errors="coerce")
    for c in feature_cols:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
    df_raw = df_raw.dropna(subset=[time_col, target_col]).sort_values(time_col).reset_index(drop=True)
    base_cols = [time_col, target_col]
    base_cols += [c for c in feature_cols if c in df_raw.columns]
    if label_col:
        base_cols.append(label_col)
    base_cols = list(dict.fromkeys(base_cols))
    return df_raw[base_cols].copy(), label_col


# ===================== Configuration =====================
class QcConfig:
    SEQ_LEN = 48
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    HIDDEN_DIM = 64
    DROPOUT = 0.15
    PATIENCE = 30
    GRAD_CLIP_NORM = 1.0
    SIGMA_MIN = 1e-4
    TARGET_COVERAGE = 0.95
    MAX_CONSECUTIVE_REPLACE = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="CG-DBNet quality-control experiment")
    p.add_argument('--data_file', type=str, required=True, help='Path to Excel file with multiple sheets')
    p.add_argument('--output_dir', type=str, default='./outputs/qc', help='Output directory')
    p.add_argument('--model', type=str, default='proposed',
                   choices=['proposed', 'abl_no_msac', 'abl_no_dagap', 'abl_no_pos',
                            'abl_affine_add', 'abl_serial_gru'])
    p.add_argument('--train_sheet', type=str, default='train')
    p.add_argument('--eval_sheet', type=str, default='val')
    p.add_argument('--test_sheet', type=str, default='test')
    p.add_argument('--time_col', type=str, default='date')
    p.add_argument('--target_col', type=str, default='sst')
    p.add_argument('--feature_cols', type=str, nargs='*', default=[])
    p.add_argument('--label_col', type=str, default='label',
                   help='Column of the test sheet that holds the 0/1 anomaly label. '
                        'If the column is absent the classification metrics are skipped.')
    p.add_argument('--eval_val_ratio', type=float, default=0.5)
    # hyper-parameters
    p.add_argument('--seq_len', type=int, default=48)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.15)
    p.add_argument('--patience', type=int, default=None,
                   help='Early-stopping patience (default: the QcConfig value, 30)')
    p.add_argument('--target_coverage', type=float, default=None,
                   help='Quantile of the validation standardized residuals used as k '
                        '(default: the QcConfig value, 0.95)')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    cfg = QcConfig()
    cfg.SEQ_LEN = args.seq_len
    cfg.BATCH_SIZE = args.batch_size
    cfg.EPOCHS = args.epochs
    cfg.LEARNING_RATE = args.lr
    cfg.HIDDEN_DIM = args.hidden_dim
    cfg.DROPOUT = args.dropout
    cfg.PATIENCE = args.patience if args.patience is not None else cfg.PATIENCE
    cfg.TARGET_COVERAGE = (args.target_coverage if args.target_coverage is not None
                           else cfg.TARGET_COVERAGE)
    device = cfg.DEVICE

    os.makedirs(args.output_dir, exist_ok=True)
    save_config_to_file(cfg, os.path.join(args.output_dir, 'training_config.json'),
                        extra={"model_kind": args.model})

    data_cfg = DataConfig(
        data_path=args.data_file, train_sheet=args.train_sheet, eval_sheet=args.eval_sheet,
        eval_val_ratio=args.eval_val_ratio, time_col=args.time_col, target_col=args.target_col,
        feature_cols=list(args.feature_cols),
    )
    data = prepare_datasets(data_cfg)
    target_idx = data["target_idx"]
    input_cols = data["input_cols"]
    target_scaler = data["target_scaler"]
    scaler = data["scaler"]

    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({"model_kind": args.model, "input_dim": data["X_train"].shape[1],
                   "input_cols": input_cols, "hidden_dim": cfg.HIDDEN_DIM,
                   "seq_len": cfg.SEQ_LEN, "dropout": cfg.DROPOUT, "sigma_min": cfg.SIGMA_MIN,
                   "feature_cols": list(args.feature_cols), "time_col": args.time_col,
                   "target_col": args.target_col}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.output_dir, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_scaler, f)

    train_ds = SeqDataset(data["X_train"], cfg.SEQ_LEN, target_idx)
    val_ds = SeqDataset(data["X_val"], cfg.SEQ_LEN, target_idx)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    df_test_base, label_col = _load_test_with_label(
        args.data_file, args.test_sheet, args.time_col, args.target_col, args.label_col,
        list(args.feature_cols))
    print(f"[QC] Test sheet '{args.test_sheet}': {len(df_test_base)} points | "
          f"label column: {label_col}")

    model = build_qc_model(args.model, data["X_train"].shape[1], cfg.SEQ_LEN, cfg.HIDDEN_DIM,
                           cfg.DROPOUT, cfg.SIGMA_MIN).to(device)
    model, k_sigma, history, best_val = train_qc_nn(
        args.model, model, train_loader, val_loader, target_scaler, args.output_dir, cfg, device)
    qc = sequential_qc(model, df_test_base, input_cols, scaler, target_scaler, k_sigma,
                       list(args.feature_cols), cfg.SEQ_LEN, args.time_col, args.target_col,
                       device, cfg.MAX_CONSECUTIVE_REPLACE)
    threshold_value = k_sigma

    cls_metrics = None
    y_true = None
    if label_col:
        y_true = df_test_base[label_col].values[cfg.SEQ_LEN: cfg.SEQ_LEN + len(qc["is_anomaly"])]
        if len(y_true) == len(qc["is_anomaly"]):
            cls_metrics = evaluate_cls(y_true, qc["is_anomaly"], y_score=qc["k"])
            print(f"\n=== {args.model} QC classification metrics ===")
            print(f"  TP={cls_metrics['TP']} FP={cls_metrics['FP']} FN={cls_metrics['FN']} "
                  f"TN={cls_metrics['TN']}")
            print(f"  F1={cls_metrics['F1']:.4f} P={cls_metrics['Precision']:.4f} "
                  f"R={cls_metrics['Recall']:.4f} Acc={cls_metrics['Accuracy']:.4f} "
                  f"Spec={cls_metrics['Specificity']:.4f}")
            print(f"  FPR={cls_metrics['FPR']:.4f} FNR={cls_metrics['FNR']:.4f} "
                  f"ROC_AUC={cls_metrics.get('ROC_AUC')}")
        else:
            print(f"[QC] Label length mismatch ({len(y_true)} vs {len(qc['is_anomaly'])}); "
                  f"skipping classification metrics")

    _save_qc_outputs(qc, cls_metrics, threshold_value, args.model, args.output_dir,
                     args.target_col, y_true=y_true)
    print(f"[Done] Output directory: {args.output_dir}\n")


if __name__ == '__main__':
    main()
