"""
Evaluation metrics
==================
  - Regression metrics for prediction: MAE / MSE / RMSE / MAPE / SMAPE / R2 / MASE
  - Classification metrics for quality control: Accuracy / Precision / Recall / F1 /
    Specificity / FPR / FNR / confusion matrix TP/TN/FP/FN / ROC_AUC
  - Probabilistic training loss: Gaussian NLL (negative log-likelihood)
  - k-sigma calibration: quantile threshold k for a given empirical coverage
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# ===================== Regression metrics for prediction =====================
def calculate_metrics(actual, pred) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = {}
    m['MAE'] = mean_absolute_error(actual, pred)
    m['MSE'] = mean_squared_error(actual, pred)
    m['RMSE'] = np.sqrt(m['MSE'])
    nz = actual != 0
    m['MAPE'] = (np.mean(np.abs((actual[nz] - pred[nz]) / actual[nz])) * 100
                 if np.sum(nz) > 0 else np.nan)
    m['SMAPE'] = np.mean(2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-8)) * 100
    m['R2'] = r2_score(actual, pred)
    m['MASE'] = (m['MAE'] / (np.mean(np.abs(actual[1:] - actual[:-1])) + 1e-8)) \
        if len(actual) > 1 else np.nan
    return m


# ===================== Classification metrics for quality control =====================
def evaluate_cls(y_true, y_pred, y_score=None) -> dict:
    """Binary classification evaluation. Returns Accuracy/Precision/Recall/F1/Specificity/FPR/
    FNR/TP/TN/FP/FN, plus ROC_AUC when y_score is given and both classes are present."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["ROC_AUC"] = roc_auc_score(y_true, y_score)
    else:
        out["ROC_AUC"] = None

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    out["TP"] = int(TP)
    out["TN"] = int(TN)
    out["FP"] = int(FP)
    out["FN"] = int(FN)
    out["Specificity"] = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    out["FPR"] = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    out["FNR"] = FN / (FN + TP) if (FN + TP) > 0 else 0.0  # 1 - Recall
    return out


# ===================== Gaussian NLL loss =====================
def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood: 0.5*log(sigma**2) + 0.5*((target - mu)**2 / sigma**2),
    averaged over the batch."""
    mu = mu.squeeze(1)
    sigma = sigma.squeeze(1)
    var = sigma.pow(2).clamp_min(1e-8)
    nll = 0.5 * torch.log(var) + 0.5 * ((target - mu).pow(2) / var)
    return nll.mean()


# ===================== k-sigma calibration =====================
def compute_dynamic_k_by_coverage(actual: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                                 target_coverage: float):
    """Find k on the validation set such that the empirical coverage of
    |actual - mu| / sigma <= k equals target_coverage. Returns (k, empirical_coverage, z)."""
    sigma_safe = np.clip(sigma.astype(float), 1e-8, None)
    z = np.abs(actual.astype(float) - mu.astype(float)) / sigma_safe
    k = float(np.quantile(z, target_coverage))
    empirical_coverage = float(np.mean(z <= k))
    return k, empirical_coverage, z
