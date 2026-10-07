"""Small shared validation, metrics and artifact helpers for supplementary analyses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=True, float_format="%.17g").encode("utf-8")).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g", na_rep="NA")


def safe_rate(num, denom):
    return float(num / denom) if denom else float("nan")


def validate_times(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if not len(idx) or idx.hasnans or not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError("Times must be nonempty, unique, increasing and nonmissing")
    return idx


def confusion_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    if y.ndim != 1 or y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("Confusion labels must be equal-length finite binary arrays")
    tp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    fp, tn = int(((y == 0) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
    return {"n_hours": len(y), "P": tp + fn, "N": fp + tn, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "miss_rate": safe_rate(fn, tp + fn), "false_alarm_rate": safe_rate(fp, fp + tn),
            "precision": safe_rate(tp, tp + fp), "recall": safe_rate(tp, tp + fn),
            "f1": safe_rate(2 * tp, 2 * tp + fp + fn)}


def prediction_metrics(frame: pd.DataFrame) -> dict:
    required = ["y_avg", "y_peak", "pred_avg", "pred_peak", "y_cls", "peak_pred_label"]
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all() or not len(frame):
        raise ValueError("Metrics require nonempty finite truth and predictions")
    out = confusion_metrics(frame.y_cls, frame.peak_pred_label)
    err = np.abs(frame.y_avg.to_numpy() - frame.pred_avg.to_numpy())
    peak_err = np.abs(frame.y_peak.to_numpy() - frame.pred_peak.to_numpy())
    peak = frame.y_cls.to_numpy() == 1
    out.update({"mae": float(err.mean()), "peak_mae": float(err[peak].mean()) if peak.any() else float("nan"),
                "peak15_mae": float(peak_err[peak].mean()) if peak.any() else float("nan"),
                "peak15_mae_all": float(peak_err.mean())})
    return out


def final_rows(ns: dict, key: str, model_name: str) -> pd.DataFrame:
    rows = ns[key].copy()
    rows = rows[(rows.model == model_name) & (rows.data_condition == "D1")].copy()
    if "eligible" in rows:
        flag = rows["eligible"]
        if flag.dtype == object:
            flag = flag.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if flag.isna().any():
            raise ValueError("Invalid snapshot eligibility flag")
        rows = rows[flag.astype(bool)].copy()
    rows["datetime"] = pd.to_datetime(rows["datetime"])
    validate_times(rows["datetime"])
    return rows.reset_index(drop=True)
