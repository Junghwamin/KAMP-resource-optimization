"""E1: denominator-aware descriptive OOF diagnostics with date-cluster CIs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import final_rows, prediction_metrics, validate_times, write_csv, write_json

RATE_NAMES = ("miss_rate", "false_alarm_rate", "precision", "recall", "f1", "mae")


def cluster_rate_intervals(frame: pd.DataFrame, mask, *, n_boot=2000, seed=42) -> dict:
    """Resample all OOF dates; within each sampled date retain every condition row."""
    idx = validate_times(frame.datetime)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(frame),) or n_boot < 1:
        raise ValueError("Invalid condition mask or bootstrap count")
    dates = idx.normalize()
    unique = dates.unique()
    y, p = frame.y_cls.to_numpy(), frame.peak_pred_label.to_numpy()
    err = np.abs(frame.y_avg.to_numpy() - frame.pred_avg.to_numpy())
    # Sufficient statistics per whole date: TP,FN,FP,TN,error sum,n.
    day_stats = np.zeros((len(unique), 6))
    for i, d in enumerate(unique):
        take = (dates == d) & mask
        yt, pt = y[take], p[take]
        day_stats[i] = [np.sum((yt == 1) & (pt == 1)), np.sum((yt == 1) & (pt == 0)),
                        np.sum((yt == 0) & (pt == 1)), np.sum((yt == 0) & (pt == 0)), err[take].sum(), take.sum()]
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    tp, fn, fp, tn, esum, n = day_stats[chosen].sum(axis=1).T
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = [fn / (tp + fn), fp / (fp + tn), tp / (tp + fp), tp / (tp + fn),
                 2 * tp / (2 * tp + fp + fn), esum / n]
    result = {"ci_n_boot": n_boot, "ci_cluster_dates": len(unique), "ci_condition_dates": int(np.sum(day_stats[:, 5] > 0))}
    for name, values in zip(RATE_NAMES, draws):
        valid = values[np.isfinite(values)]
        lo, hi = np.percentile(valid, [2.5, 97.5]) if len(valid) else [np.nan, np.nan]
        result.update({name + "_ci_low": float(lo), name + "_ci_high": float(hi),
                       name + "_valid_replicates": len(valid), name + "_na_fraction": 1 - len(valid) / n_boot})
    return result


def condition_masks(frame: pd.DataFrame, threshold: float) -> dict:
    hour = pd.DatetimeIndex(frame.datetime).hour
    return {"all": np.ones(len(frame), dtype=bool), "production_zero": frame["prod"] == 0,
            "production_positive": frame["prod"] > 0, "production_high": frame["prod"] > threshold,
            "hour_08": hour == 8, "hour_13": hour == 13, "hour_17_18": np.isin(hour, [17, 18]),
            "daytime_09_17": np.isin(hour, range(9, 18)), "nighttime": ~np.isin(hour, range(9, 18)),
            "shutdown": frame.is_shutdown == 1, "operating": frame.is_shutdown == 0,
            "first_day_after_shutdown": frame.is_first_day_back == 1}


def cell_summary(frame: pd.DataFrame, *, sparse_n=20, sparse_days=5) -> dict:
    result = prediction_metrics(frame)
    result.update({"n_days": int(pd.DatetimeIndex(frame.datetime).normalize().nunique()),
                   "peak_rate": float(frame.y_cls.mean()), "mean_peak_kw": float(frame.y_peak.mean())})
    result["sparse_cell"] = result["n_hours"] < sparse_n or result["n_days"] < sparse_days
    return result


def build_diagnostics(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    frame = frame.copy()
    validate_times(frame.datetime)
    prediction_metrics(frame)  # Validate labels and finite observations before subset calculations.
    need = ["prod", "temp", "is_shutdown", "is_first_day_back"]
    if not np.isfinite(frame[need].to_numpy(float)).all():
        raise ValueError("Diagnostic condition columns contain missing or infinite values")
    cfg = config.get("diagnostics", {})
    threshold = float(frame["prod"].quantile(float(cfg.get("high_production_quantile", .75))))
    q50 = float(frame["prod"].quantile(.5))
    n_boot = int(cfg.get("n_boot", 2000))
    seed = int(config.get("seed", 42))
    rows = []
    for condition, mask in condition_masks(frame, threshold).items():
        subset = frame.loc[mask]
        if len(subset):
            row = cell_summary(subset, sparse_n=cfg.get("sparse_hours", 20), sparse_days=cfg.get("sparse_days", 5))
        else:
            from .common import confusion_metrics
            row = confusion_metrics([], [])
            row.update({"n_days": 0, "mae": np.nan, "peak_mae": np.nan, "peak15_mae": np.nan,
                        "peak_rate": np.nan, "mean_peak_kw": np.nan, "sparse_cell": True})
        row.update(cluster_rate_intervals(frame, mask, n_boot=n_boot, seed=seed))
        rows.append({"condition": condition, "threshold_scope": "all_selected_OOF_descriptive_only",
                     "high_production_threshold": threshold, "evaluation_role": "internal_descriptive", **row})
    frame["hour"] = pd.DatetimeIndex(frame.datetime).hour
    frame["production_band"] = np.select([frame["prod"] == 0, frame["prod"] <= q50, frame["prod"] <= threshold],
                                          ["zero", "positive_to_q50", "q50_to_q75"], default="above_q75")
    production = []
    for (band, hour), part in frame.groupby(["production_band", "hour"], observed=True, sort=True):
        production.append({"production_band": band, "hour": hour, **cell_summary(part),
                           "q50": q50, "q75": threshold, "evaluation_role": "internal_descriptive"})
    tlo, thi = frame["temp"].quantile([.25, .75]).to_numpy()
    frame["temperature_band"] = np.select([frame["temp"] <= tlo, frame["temp"] <= thi], ["low", "middle"], default="high")
    temperatures = []
    for (hour, shut, band), part in frame.groupby(["hour", "is_shutdown", "temperature_band"], observed=True, sort=True):
        temperatures.append({"hour": hour, "is_shutdown": shut, "temperature_band": band,
                             **cell_summary(part), "temperature_q25": tlo, "temperature_q75": thi,
                             "evaluation_role": "internal_descriptive"})
    meta = {"high_production_threshold": threshold, "production_q50": q50, "temperature_q25": tlo, "temperature_q75": thi,
            "threshold_scope": "all_selected_OOF_descriptive_only", "overlapping_conditions": True,
            "ci": "95% percentile bootstrap, sampling whole dates including dates with zero condition rows",
            "undefined_rates": "NA when denominator is zero; finite replicate count and NA fraction reported",
            "interpretation": "Descriptive conditional associations; hour-defined startup is not an independent causal effect."}
    return pd.DataFrame(rows), pd.DataFrame(production), pd.DataFrame(temperatures), meta


def run(root: Path, snapshot: Path, output: Path, config: dict) -> dict:
    from tools.analysis_snapshot import load_snapshot
    ns = load_snapshot(snapshot)
    manifest = json.loads((Path(root) / "outputs/models/full/eval/manifest.json").read_text(encoding="utf-8"))
    rows = final_rows(ns, "oof_long", manifest["model_name"])
    feats = ns["features"] if "features" in ns else ns["feat"]
    if not feats.index.is_unique:
        raise ValueError("Snapshot feature index must be unique")
    for c in ["prod", "temp", "is_shutdown", "is_first_day_back"]:
        rows[c] = feats.reindex(pd.DatetimeIndex(rows.datetime))[c].to_numpy()
    tables = build_diagnostics(rows, config)
    for name, table in zip(["condition_confusion_rates.csv", "production_hour_cells.csv", "temperature_stratified_cells.csv"], tables[:3]):
        write_csv(output / name, table)
    write_csv(output / "diagnostics_selected_oof.csv", rows)
    summary = {"experiment": "diagnostics", "status": "complete", "model": manifest["model_name"],
               "data_condition": "D1", "n_oof": len(rows), **tables[3]}
    write_json(output / "summary.json", summary)
    return summary
