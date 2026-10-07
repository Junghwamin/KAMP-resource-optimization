"""E3: frozen evaluation bundle under preregistered raw-plan input errors.

No fit or calibration occurs here. Each day is replayed from unchanged historical
observations; scenario errors are applied only to that day's raw plan inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import final_rows, frame_hash, prediction_metrics, sha256_file, write_csv, write_json


def default_scenarios() -> list[dict]:
    scenarios = [{"id": "C0", "kind": "baseline"}]
    scenarios += [{"id": f"P{int(f * 100)}", "kind": "production_scale", "production_factor": f} for f in [.8, .9, 1.1, 1.2]]
    scenarios += [{"id": f"H{int(f * 100)}", "kind": "headcount_scale", "headcount_factor": f} for f in [.9, 1.1]]
    scenarios += [{"id": f"T{d:+d}", "kind": "temperature_shift", "temperature_delta": d} for d in [-2, -1, 1, 2]]
    scenarios += [{"id": f"P{int(f * 100)}T{d:+d}", "kind": "combined", "production_factor": f, "temperature_delta": d}
                  for f in [.8, 1.2] for d in [-2, 2]]
    scenarios += [{"id": f"S{d:+d}", "kind": "production_time_shift", "shift_hours": d, "supplementary": True} for d in [-1, 1]]
    return scenarios


def perturb_plan(plan: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    from serving.contract import normalize_plan
    out = normalize_plan(plan).copy(deep=True)
    for key, column in [("production_factor", "생산량"), ("headcount_factor", "공장인원")]:
        factor = float(scenario.get(key, 1))
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Scaling factors must be finite and positive")
        out[column] = out[column] * factor
    delta = float(scenario.get("temperature_delta", 0))
    if not np.isfinite(delta):
        raise ValueError("Temperature offset must be finite")
    out["기온"] += delta
    shift = int(scenario.get("shift_hours", 0))
    if shift not in [-1, 0, 1]:
        raise ValueError("Only preregistered +/-1-hour shifts are supported")
    if shift:
        original = out["생산량"].to_numpy(float)
        moved = np.zeros(24)
        # Boundary mass remains at the nearest edge hour of the same day.
        # It never wraps to another day and no production volume is discarded.
        np.add.at(moved, np.clip(np.arange(24) + shift, 0, 23), original)
        out["생산량"] = moved
    if (out[["생산량", "공장인원"]] < 0).any().any():
        raise ValueError("Production and headcount cannot be negative")
    return out


def validate_scenarios(scenarios: list[dict]) -> None:
    expected = {s["id"]: s for s in default_scenarios()}
    if len(scenarios) != 17 or len({s["id"] for s in scenarios}) != 17:
        raise ValueError("E3 requires every one of the 15 basic and 2 supplementary scenarios")
    if scenarios[0]["id"] != "C0":
        raise ValueError("C0 must be the first preregistered scenario")
    for scenario in scenarios:
        if scenario != expected.get(scenario["id"]):
            raise ValueError("Scenario differs from the preregistered raw-plan protocol")


def assert_eval_bundle(bundle) -> None:
    if bundle.role != "eval" or bundle.manifest["training"]["cutoff_exclusive"] != "2021-09-01 00:00:00":
        raise ValueError("Sensitivity requires frozen September eval bundle; deploy bundle is forbidden")


def assert_baseline(reference: pd.DataFrame, candidate: pd.DataFrame, atol=1e-8) -> dict:
    a, b = reference.set_index("datetime"), candidate.set_index("datetime")
    if not a.index.equals(b.index):
        raise ValueError("C0 baseline timestamps do not match frozen reference")
    differences = {}
    for c in ["pred_avg", "pred_peak", "prob_raw"]:
        if c not in a or not np.isfinite(a[c].to_numpy(float)).all():
            if c == "prob_raw":
                continue
            raise ValueError(f"Missing frozen baseline column: {c}")
        differences[c] = float(np.max(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))))
        if not np.allclose(a[c], b[c], atol=atol, rtol=0):
            raise ValueError(f"C0 frozen prediction mismatch: {c}, max_abs={differences[c]}")
    if not np.array_equal(a.peak_pred_label, b.peak_pred_label):
        raise ValueError("C0 frozen alert labels mismatch")
    return differences


def run(root: Path, snapshot: Path, output: Path, config: dict) -> dict:
    from serving import _core
    from serving.bundle import load_bundle
    from serving.contract import HISTORY_COLUMNS, PLAN_COLUMNS
    from serving.pipeline import assemble_output, build_day_ahead_features
    from tools.analysis_snapshot import load_snapshot

    root, output = Path(root), Path(output)
    cfg = config.get("sensitivity", {})
    scenarios = cfg.get("scenarios", default_scenarios())
    validate_scenarios(scenarios)
    bundle_dir = root / "outputs/models/full/eval"
    bundle = load_bundle(bundle_dir, check_versions=True, selftest=True)
    assert_eval_bundle(bundle)
    frozen_manifest = sha256_file(bundle_dir / "manifest.json")
    ns = load_snapshot(snapshot)
    reference = final_rows(ns, "test_long", bundle.manifest["model_name"])
    if len(reference) != 336:
        raise ValueError("Frozen eval comparison requires the original 336 test hours")
    truth = reference[["datetime", "y_avg", "y_peak", "y_cls"]].copy()
    truth_hash = frame_hash(truth)
    data_path = root / bundle.manifest["training"]["data_file"]
    if sha256_file(data_path) != bundle.manifest["training"]["data_sha256"]:
        raise ValueError("Evaluation raw data hash differs from frozen bundle")
    raw = _core.restore_hours(pd.read_csv(data_path, encoding="utf-8-sig"))
    raw = _core.build_datetime_index(raw)
    raw_hash = frame_hash(raw)
    dates = pd.DatetimeIndex(reference.datetime).normalize().unique()
    if len(dates) != 14:
        raise ValueError("Expected 14 frozen evaluation dates")
    baseline_features = {}
    baseline_predictions = None
    all_predictions, metric_rows, audit_rows = [], [], []
    write_json(output / "scenario_protocol.json", {"scenarios": scenarios, "interpretation": "assumed input-error stress; not measured forecast-error distribution or causal production intervention",
                                                   "shift_boundary": "clamp within same day and sum colliding production volumes; no circular wrap",
                                                   "history": "actual pre-origin history reset for each day; immutable across scenarios"})
    for scenario in scenarios:
        day_predictions = []
        for day in dates:
            hist = raw.loc[raw.index < day, HISTORY_COLUMNS].copy(deep=True)
            plan = raw.loc[raw.index.normalize() == day, PLAN_COLUMNS].copy(deep=True).reset_index(drop=True)
            hist_hash = frame_hash(hist)
            altered = perturb_plan(plan, scenario)
            X, _ = build_day_ahead_features(hist, altered, columns=bundle.columns, theta=bundle.theta,
                                            holidays=bundle.holidays_for(day), min_days=bundle.history_days["min"], target_date=day)
            if scenario["id"] == "C0":
                baseline_features[day] = X.copy(deep=True)
            base = baseline_features[day]
            fixed = [c for c in X if c.startswith("y_") or c.startswith("roll")]
            if not np.array_equal(X[fixed].to_numpy(), base[fixed].to_numpy()):
                raise ValueError("Scenario changed historical power features")
            if frame_hash(hist) != hist_hash or frame_hash(truth) != truth_hash:
                raise ValueError("Scenario changed historical observations or evaluation truth")
            prod = altered["생산량"].to_numpy(float)
            temp = altered["기온"].to_numpy(float)
            expected = {"prod": prod, "prod_cum_day": np.cumsum(prod), "temp": temp,
                        "cdd": np.maximum(temp - 24, 0), "hdd": np.maximum(18 - temp, 0)}
            yesterday = raw.loc[(raw.index >= day - pd.Timedelta(days=1)) & (raw.index < day), "생산량"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (prod - yesterday) / yesterday
            expected["prod_chg_ratio"] = np.where(np.isfinite(ratio), ratio, 0)
            # Headcount is nullable: pipeline fills forward only using available history/plan.
            hc = pd.concat([hist["공장인원"], altered["공장인원"]], ignore_index=True).ffill().iloc[-24:].to_numpy(float)
            expected["headcount"] = hc
            for c, values in expected.items():
                if not np.allclose(X[c].to_numpy(float), values, atol=1e-8, rtol=1e-12):
                    raise ValueError(f"Raw-plan derived feature mismatch: {c}")
            scale_case = not scenario.get("shift_hours")
            zero_ok = np.array_equal(plan["생산량"].to_numpy() == 0, prod == 0) if scale_case else None
            head_zero_ok = np.array_equal(plan["공장인원"].to_numpy() == 0, altered["공장인원"].to_numpy() == 0)
            sum_ok = np.isclose(prod.sum(), plan["생산량"].sum() * float(scenario.get("production_factor", 1)), rtol=1e-12, atol=1e-8)
            shutdown_ok = bool(np.array_equal(X["is_shutdown"], base["is_shutdown"]))
            if zero_ok is False or not head_zero_ok or not sum_ok or not shutdown_ok:
                raise ValueError("Input scenario failed zero, shutdown or production total invariance")
            pred = bundle.predict_features(X)
            out = assemble_output(X, pred, tau=bundle.tau, tau_cls=bundle.tau_cls,
                                  calibrator=bundle.calibrator, halfwidths=bundle.halfwidths)
            frame = pd.DataFrame({"datetime": pd.to_datetime(out.ts), "pred_avg": out.y_avg_pred, "pred_peak": out.y_peak_pred,
                                  "regime": out.regime, "peak_pred_label": out.peak_label, "prob_raw": out.peak_prob,
                                  "prob_cal": out.peak_prob_cal, "theta": bundle.theta, "tau": bundle.tau})
            day_predictions.append(frame)
            audit_rows.append({"scenario": scenario["id"], "date": day, "history_sha256": hist_hash, "truth_sha256": truth_hash,
                               "bundle_manifest_sha256": frozen_manifest, "n_hours": len(X), "historical_power_features_unchanged": True,
                               "production_zero_support_unchanged": zero_ok, "headcount_zero_support_unchanged": head_zero_ok,
                               "production_daily_total_rule_passed": bool(sum_ok), "shutdown_unchanged": shutdown_ok,
                               "derived_features_checked": ",".join(expected), "input_plan_sha256": frame_hash(altered),
                               "max_feature_change": float(np.max(np.abs(X.to_numpy() - base.to_numpy())))})
        frame = pd.concat(day_predictions, ignore_index=True)
        frame = frame.merge(truth, on="datetime", how="left", validate="one_to_one")
        if scenario["id"] == "C0":
            c0_diff = assert_baseline(reference, frame, atol=float(cfg.get("atol", 1e-8)))
            baseline_predictions = frame.copy(deep=True)
        if baseline_predictions is None:
            raise ValueError("C0 must run before perturbation scenarios")
        metrics = prediction_metrics(frame)
        metrics.update({"scenario": scenario["id"], "supplementary": bool(scenario.get("supplementary", False)),
                        "gate_changed_hours": int(np.sum(frame.regime.to_numpy() != baseline_predictions.regime.to_numpy())),
                        "gate_changed_rate": float(np.mean(frame.regime.to_numpy() != baseline_predictions.regime.to_numpy())),
                        "mae_delta_c0": metrics["mae"] - prediction_metrics(baseline_predictions)["mae"],
                        "evaluation_role": "frozen_reference_input_stress"})
        metric_rows.append(metrics)
        frame["scenario"] = scenario["id"]
        frame["evaluation_role"] = "frozen_reference_input_stress"
        all_predictions.append(frame)
    if frame_hash(raw) != raw_hash or sha256_file(bundle_dir / "manifest.json") != frozen_manifest:
        raise ValueError("Frozen raw inputs or bundle changed during E3")
    write_csv(output / "input_sensitivity_predictions.csv", pd.concat(all_predictions, ignore_index=True))
    write_csv(output / "input_sensitivity_metrics.csv", pd.DataFrame(metric_rows))
    write_csv(output / "feature_consistency_audit.csv", pd.DataFrame(audit_rows))
    summary = {"experiment": "sensitivity", "status": "complete", "n_scenarios": len(scenarios), "basic_scenarios": 15,
               "supplementary_scenarios": 2, "bundle_id": bundle.bundle_id, "bundle_manifest_sha256": frozen_manifest,
               "c0_max_absolute_difference": c0_diff, "truth_sha256": truth_hash, "raw_sha256": raw_hash,
               "evaluation_role": "frozen_reference_input_stress", "scenario_ids": [s["id"] for s in scenarios],
               "limitations": ["Assumed input-error ranges, not observed operational forecast errors", "Historical observations reset daily; no causal production intervention", "Existing no-plan ablation is separate; zeroing production here is not a no-plan model"]}
    write_json(output / "summary.json", summary)
    return summary
