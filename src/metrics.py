from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

@dataclass
class ClassificationMetrics:
    n: int
    accuracy: float
    brier: float
    logloss: Optional[float]
    confusion: Dict[str, int]     # TP, TN, FP, FN for class "1"
    calibration: List[Dict[str, float]]  # bins with prob_range, count, avg_pred, emp_rate

@dataclass
class RegressionMetrics:
    n: int
    mae: float
    rmse: float
    mean_err: float

def _safe_log(p: float) -> float:
    p = min(max(p, 1e-15), 1 - 1e-15)
    return math.log(p)

def classification_metrics(y_true: List[int], y_prob: List[float], threshold: float=0.5, bins: int=10) -> ClassificationMetrics:
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_hat = (y_prob >= threshold).astype(int)

    n = int(y_true.shape[0])
    if n == 0:
        return ClassificationMetrics(0, 0.0, 0.0, None, {"TP":0,"TN":0,"FP":0,"FN":0}, [])

    acc = float((y_true == y_hat).mean())
    brier = float(np.mean((y_prob - y_true) ** 2))

    try:
        ll = -float(np.mean(y_true * np.log(np.clip(y_prob, 1e-15, 1)) + (1 - y_true) * np.log(np.clip(1 - y_prob, 1e-15, 1))))
    except Exception:
        ll = None

    TP = int(((y_hat == 1) & (y_true == 1)).sum())
    TN = int(((y_hat == 0) & (y_true == 0)).sum())
    FP = int(((y_hat == 1) & (y_true == 0)).sum())
    FN = int(((y_hat == 0) & (y_true == 1)).sum())

    edges = np.linspace(0, 1, bins + 1)
    cal_rows: List[Dict[str, float]] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        m = (y_prob >= lo) & (y_prob < hi if i < bins-1 else y_prob <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            cal_rows.append({"lo":float(lo), "hi":float(hi), "count":0, "avg_pred":float("nan"), "emp_rate":float("nan")})
        else:
            avg_pred = float(y_prob[m].mean())
            emp_rate = float(y_true[m].mean())
            cal_rows.append({"lo":float(lo), "hi":float(hi), "count":cnt, "avg_pred":avg_pred, "emp_rate":emp_rate})

    return ClassificationMetrics(n, acc, brier, ll, {"TP":TP,"TN":TN,"FP":FP,"FN":FN}, cal_rows)

def regression_metrics(y_true: List[float], y_pred: List[float]) -> RegressionMetrics:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = int(y_true.shape[0])
    if n == 0:
        return RegressionMetrics(0, 0.0, 0.0, 0.0)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mean_err = float(np.mean(err))
    return RegressionMetrics(n, mae, rmse, mean_err)
