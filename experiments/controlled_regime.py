"""E4: fixed-parameter single LightGBM versus three-regime comparison."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.model_adapter import (
    LGB_BASE, OUTER_STARTS, atomic_json, calibrate, json_bytes, jsonable, load_data,
    metrics, partition, prediction_rows, predict_job, prepare_split, profile_audit, protocol_guard, sha, split_audit,
)

DEFAULTS = {"profile": "full", "seed": 42, "threads": 4, "n_estimators": 800,
            "outer_starts": list(OUTER_STARTS), "outer_days": 14, "calibration_days": 14,
            "gap_hours": 24, "candidate_models": ["lgb", "regime3"], "lgb_base": LGB_BASE,
            "n_boot": 2000, "same_regression_params": True, "regression_objective": "l1",
            "calibration_min_positive": 5, "calibration_min_negative": 20,
            "isolation": "serial_subprocess"}


def paired_intervals(rows, repetitions=2000, seed=42):
    """Paired daily clusters, positive difference favours the regime model."""
    outputs = []
    for outer, subset in rows.groupby("outer"):
        left = subset[subset.model == "lgb"].set_index("datetime")
        right = subset[subset.model == "regime3"].set_index("datetime").reindex(left.index)
        if right.pred_avg.isna().any() or not np.array_equal(left.y_avg, right.y_avg):
            raise ValueError("Controlled comparison lost paired row identity")
        dates = pd.to_datetime(left.index).normalize()
        blocks = [np.flatnonzero(dates == day) for day in dates.unique()]
        for target, prediction in (("y_avg", "pred_avg"), ("y_peak", "pred_peak")):
            difference = np.abs(left[target].to_numpy()-left[prediction].to_numpy()) - np.abs(right[target].to_numpy()-right[prediction].to_numpy())
            rng = np.random.default_rng(seed)
            draws = [difference[np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])].mean()
                     for _ in range(repetitions)]
            lo, hi = np.percentile(draws, [2.5, 97.5])
            outputs.append({"outer": outer, "target": target, "n": len(left), "days": len(blocks),
                            "MAE_single_minus_regime": float(difference.mean()),
                            "ci_low": float(lo), "ci_high": float(hi), "n_boot": repetitions,
                            "method": "paired calendar-day cluster percentile bootstrap"})
    return pd.DataFrame(outputs)


def run(root: Path, snapshot: Path, output: Path, config: dict) -> dict:
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = {**DEFAULTS, **config.get("controlled_regime", {})}
    frame, features, metadata = load_data(snapshot)
    guard = protocol_guard(cfg, DEFAULTS, features, diagnostic_keys=("design_note",))
    full = guard["matches_registered_full"]
    if not metadata.get("fingerprint"):
        raise ValueError("Controlled comparison requires a verified snapshot fingerprint")
    protocol = {"experiment": "controlled_regime", "config": cfg, "features": features,
                "comparison": "Same core/calibration/outer rows, features, regression objective and 800-tree head settings",
                "difference_from_original": "Core reserves a calibration tail; original full-development OOF is a separate reference",
                "capacity_limit": "Regime gate, multiple heads and fallback increase total capacity; capacity is not equalized",
                "evaluation_role": "retrospective_controlled_comparison", "original_outputs_modified": False,
                "snapshot_fingerprint": metadata["fingerprint"], "full_protocol_guard": guard,
                "plan_availability_assumption": "Calendar/ERP quality flags are retrospectively diagnosed; assumed prior plan confirmation, not timestamp-verified origin availability"}
    protocol["fingerprint"] = hashlib.sha256(json_bytes(protocol)).hexdigest()
    atomic_json(output / "controlled_protocol.json", protocol)
    atomic_json(output / "manifest.json", {"status": "running", "protocol": protocol})
    predictions, metric_rows, audit_rows, resource_rows, duplicate_rows, failures = [], [], [], [], [], []
    started = time.perf_counter()
    for i, start in enumerate(cfg["outer_starts"], 2):
        outer = f"O{i}"
        spec = partition(start, eval_days=cfg["outer_days"], calibration_days=cfg["calibration_days"], gap_hours=cfg["gap_hours"])
        try:
            rebuilt, masks, theta = prepare_split(frame, features, spec)
            current_audit = split_audit(rebuilt, masks, theta, spec, features)
            pd.DataFrame(current_audit).to_csv(output / f"{outer}_split_audit.csv", index=False)
            tr, cal, ev = (rebuilt.loc[masks[k]] for k in ("core", "calibration", "evaluation"))
            inference = pd.concat([cal, ev])
            fold_parts = []
            for model in cfg["candidate_models"]:
                pred, fit = predict_job(root, root / ".cache/additional_fits", model, tr[features],
                                       tr[["y_avg", "y_peak", "y_cls"]], inference[features], cfg)
                calibration, _ = calibrate(cal.y_peak, pred["pred_peak"][:len(cal)], theta,
                                           min_positive=cfg["calibration_min_positive"], min_negative=cfg["calibration_min_negative"])
                outer_pred = {k: v[len(cal):] for k, v in pred.items()}
                rows = prediction_rows(rebuilt, masks["evaluation"], outer_pred, theta, calibration["tau"],
                                       spec, model=model, outer=outer)
                rows["evaluation_role"] = "retrospective_controlled_comparison"
                fold_parts.append(rows)
                metric_rows.append({"outer": outer, "model": model, "theta": theta, **calibration,
                                    **metrics(ev.y_avg, ev.y_peak, outer_pred["pred_avg"], outer_pred["pred_peak"], theta, calibration["tau"])})
                resource_rows.append({"outer": outer, "model": model, "n_core": len(tr), "n_calibration": len(cal),
                                      "n_evaluation": len(ev), "fit_seconds": fit["fit_seconds"], "cache_hit": fit["cache_hit"],
                                      "job_fingerprint": fit["fingerprint"], "regressor_parameters": json_bytes(cfg["lgb_base"]).decode(),
                                      "trees_per_regressor": cfg["n_estimators"], "threads": cfg["threads"],
                                      "extra_gate": model == "regime3", "n_possible_heads_per_target": 3 if model == "regime3" else 1,
                                      "fallback_regressor_per_target": model == "regime3", "fallback": fit.get("fallback")})
            fold = pd.concat(fold_parts, ignore_index=True)
            fold.to_csv(output / f"{outer}_predictions.csv", index=False)
            predictions.append(fold)
            audit_rows.extend({"outer": outer, **a} for a in current_audit)
            duplicate_rows.extend({"outer": outer, **a} for a in profile_audit(frame, spec))
            atomic_json(output / f"{outer}_checkpoint.json", {"status": "complete", "protocol_fingerprint": protocol["fingerprint"],
                        "output_sha256": sha(output / f"{outer}_predictions.csv"), "spec": spec})
        except Exception as exc:
            failures.append({"outer": outer, "error": f"{type(exc).__name__}: {exc}"})
            atomic_json(output / f"{outer}_checkpoint.json", {"status": "failed", "error": str(exc)})
    paths = []
    if predictions:
        combined = pd.concat(predictions, ignore_index=True)
        combined.to_csv(output / "controlled_regime_predictions.csv", index=False)
        paired_intervals(combined, int(cfg.get("n_boot", 2000)), cfg["seed"]).to_csv(output / "controlled_regime_paired_ci.csv", index=False)
        paths.extend(["controlled_regime_predictions.csv", "controlled_regime_paired_ci.csv"])
    for name, rows in (("controlled_regime_metrics.csv", metric_rows), ("controlled_split_audit.csv", audit_rows),
                       ("controlled_resources.csv", resource_rows), ("controlled_profile_novelty.csv", duplicate_rows)):
        pd.DataFrame(rows).to_csv(output / name, index=False)
        paths.append(name)
    manifest = {"status": "complete" if full and not failures else "partial", "experiment": "controlled_regime",
                "failures": failures, "completed_outer_count": len(predictions), "expected_outer_count": 4,
                "seconds": time.perf_counter()-started, "protocol": jsonable(protocol),
                "files": {p: sha(output / p) for p in paths}}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "summary.json", manifest)
    return manifest
