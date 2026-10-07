"""E6: fixed-prediction, open-loop policy simulation with separated action/scoring.

The action selector accepts only forecasts and declared plans. Observed peak is
used in the response fit (past core only) and in the *subsequent* evaluator.
Neither the response slope nor assumed alpha is an identified causal effect.
"""
from __future__ import annotations

import itertools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import frame_hash, sha256_file, validate_times, write_csv, write_json


POLICIES = ("none", "fixed_L1", "alert_L1", "alert_L2")
DEFAULTS = {
    "alpha": .20,
    "alpha_grid": [.10, .20, .30],
    "shift_fraction": .20,
    "receiving_hours": [10, 11, 15, 16],
    "start_hours": [8, 13],
    "receiving_capacity_quantile": .95,
    "receiving_capacity_multiplier": 1.20,
    "demand_rate_krw_per_kw_month": 8320.0,
    "demand_rate_multipliers": [.7, 1., 1.3],
    "action_cost_krw": [0., 1000., 5000.],
    "shift_unit_cost_krw": [0., .01, .10],
}
ACTION_COLUMNS = (
    "datetime", "origin", "pred_peak", "tau", "planned_production",
    "plan_known", "is_operating",
)


def _bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.copy()
    parsed = values.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        raise ValueError("Boolean flags must be explicit; missing plan flags cannot mean idle")
    return parsed.astype(bool)


def _settings(config: dict) -> dict:
    cfg = {**DEFAULTS, **config}
    for name in ("alpha", "shift_fraction", "receiving_capacity_quantile"):
        if not 0 <= float(cfg[name]) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not np.isfinite(float(cfg["receiving_capacity_multiplier"])) or float(cfg["receiving_capacity_multiplier"]) < 1:
        raise ValueError("Receiving capacity multiplier must be >= 1")
    for name in ("start_hours", "receiving_hours"):
        if any(int(h) != h or h < 0 or h > 23 for h in cfg[name]):
            raise ValueError("Policy hours must be integers in [0, 23]")
    for name in ("alpha_grid", "demand_rate_multipliers", "action_cost_krw", "shift_unit_cost_krw"):
        arr = np.asarray(cfg[name], dtype=float)
        if not arr.size or not np.isfinite(arr).all() or (arr < 0).any():
            raise ValueError(f"Invalid policy sensitivity values: {name}")
    rate = float(cfg["demand_rate_krw_per_kw_month"])
    if max(cfg["alpha_grid"]) > 1 or not np.isfinite(rate) or rate < 0:
        raise ValueError("Invalid alpha or assumed demand unit rate")
    return cfg


def fit_response(training: pd.DataFrame, core_train_end, evaluation_start, config: dict) -> dict:
    """OLS peak ~ production + hour effects; exactly the past-core rows only.

    Within-hour centering estimates the same production slope as an intercept
    plus hour dummies without importing or executing the notebook stages.
    """
    cfg = _settings(config)
    data = training.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    validate_times(data["datetime"])
    cutoff, start = pd.Timestamp(core_train_end), pd.Timestamp(evaluation_start)
    if cutoff >= start:
        raise ValueError("Response fit cutoff must precede evaluation")
    core = data.loc[data.datetime <= cutoff].copy()
    required = ["planned_production", "y_peak"]
    valid = _bool(core["plan_known"]) & np.isfinite(core[required].to_numpy(float)).all(axis=1)
    valid &= (core.planned_production > 0) & (core.y_peak >= 0)
    quality_ok = pd.Series(True, index=core.index)
    for name in ("is_outage", "is_erp_missing", "is_warmup"):
        if name in core:
            quality_ok &= ~_bool(core[name])
    n_core, n_quality_excluded = len(core), int((~quality_ok).sum())
    valid &= quality_ok
    core = core.loc[valid].copy()
    if core.empty:
        raise ValueError("No finite known positive-production core data for response fit")
    core["hour"] = core.datetime.dt.hour
    means = core.groupby("hour")[required].transform("mean")
    xr = core.planned_production.to_numpy(float) - means.planned_production.to_numpy(float)
    yr = core.y_peak.to_numpy(float) - means.y_peak.to_numpy(float)
    denominator = float(xr @ xr)
    identifiable = denominator > np.finfo(float).eps * max(1., float(np.square(core.planned_production).sum()))
    raw_beta = float(xr @ yr / denominator) if identifiable else 0.
    beta = max(0., raw_beta)
    pred = means.y_peak.to_numpy(float) + raw_beta * xr
    total_variance = float(np.square(core.y_peak - core.y_peak.mean()).sum())
    r2 = 1. - float(np.square(core.y_peak - pred).sum()) / total_variance if total_variance else None
    cap = float(core.planned_production.quantile(float(cfg["receiving_capacity_quantile"])))
    cap *= float(cfg["receiving_capacity_multiplier"])
    return {
        "core_train_end": cutoff.isoformat(), "fit_max_datetime": core.datetime.max().isoformat(),
        "evaluation_start": start.isoformat(), "n_fit": len(core),
        "n_core_rows": n_core, "n_core_quality_excluded": n_quality_excluded,
        "beta_raw_kw_per_unit": raw_beta, "beta_kw_per_unit": beta, "ols_r2": r2,
        "beta_status": "estimated" if identifiable else "unidentified_zero_fallback",
        "negative_slope_clipped": raw_beta < 0, "receiving_capacity_units": cap,
        "fit_rows_sha256": frame_hash(core[["datetime", "planned_production", "y_peak"]]),
        "response_interpretation": "Past-core observational hour-adjusted association; causal response unverified",
        "capacity_interpretation": "Assumed core production quantile times fixed multiplier, not verified equipment capacity",
    }


def select_actions(forecasts_and_plans: pd.DataFrame, policy: str, capacity: float, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose actions without actual power, actual labels, or evaluator access.

    An unconfirmed/invalid plan anywhere in a day yields manual confirmation and
    no actions for that whole day. L2 conserves each day's planned production;
    requested but unallocated units stay at the donor (never disappear).
    """
    if set(forecasts_and_plans.columns) != set(ACTION_COLUMNS):
        raise ValueError("Action input must contain exactly forecast/plan contract columns; observed outcomes forbidden")
    if policy not in POLICIES or not np.isfinite(capacity) or capacity < 0:
        raise ValueError("Unknown policy or invalid assumed capacity")
    cfg = _settings(config)
    data = forecasts_and_plans.copy().reset_index(drop=True)
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["origin"] = pd.to_datetime(data["origin"])
    validate_times(data["datetime"])
    data["plan_known"] = _bool(data.plan_known)
    data["is_operating"] = _bool(data.is_operating)
    finite = np.isfinite(data[["pred_peak", "tau", "planned_production"]].to_numpy(float)).all(axis=1)
    valid = finite & (data.planned_production >= 0) & data.plan_known
    valid &= data.origin.notna() & (data.origin <= data.datetime)
    data["valid_plan_day"] = pd.Series(valid).groupby(data.datetime.dt.normalize()).transform("all")
    data["alert"] = finite & (data.pred_peak >= data.tau)
    data["policy"] = policy
    data["l1_action"] = False
    for name in ("shift_requested", "shift_out", "shift_in", "unallocated_units"):
        data[name] = 0.
    data["decision_status"] = np.where(data.valid_plan_day, "no_action", "manual_confirmation_no_action")
    transfers = []
    eligible = data.valid_plan_day & data.is_operating & (data.datetime > data.origin)
    if policy in ("fixed_L1", "alert_L1"):
        target = eligible & data.datetime.dt.hour.isin(cfg["start_hours"])
        if policy == "alert_L1":
            target &= data.alert
        data.loc[target, "l1_action"] = True
        data.loc[target, "decision_status"] = "assumed_startup_spreading"
    elif policy == "alert_L2":
        donors = data.index[eligible & data.alert & (data.planned_production > 0)]
        for source in donors:
            row = data.loc[source]
            requested = float(row.planned_production) * float(cfg["shift_fraction"])
            remaining = requested
            data.loc[source, "shift_requested"] = requested
            recipients = data.index[
                eligible & ~data.alert
                & data.datetime.dt.hour.isin(cfg["receiving_hours"])
                & (data.datetime.dt.normalize() == row.datetime.normalize())
                & (data.datetime > row.datetime) & (data.datetime > row.origin)
            ]
            for target in recipients:
                available = max(0., capacity - float(data.loc[target, "planned_production"] + data.loc[target, "shift_in"]))
                moved = min(remaining, available)
                if moved <= 0:
                    continue
                data.loc[source, "shift_out"] += moved
                data.loc[target, "shift_in"] += moved
                remaining -= moved
                transfers.append({"policy": policy, "origin": row.origin,
                                  "source_datetime": row.datetime, "recipient_datetime": data.loc[target, "datetime"],
                                  "shifted_units": moved, "assumed_capacity_units": capacity})
                if remaining <= 1e-10:
                    break
            data.loc[source, "unallocated_units"] = max(0., remaining)
            data.loc[source, "decision_status"] = "shifted" if data.loc[source, "shift_out"] > 0 else "infeasible_no_action"
            if remaining > 1e-10 and data.loc[source, "shift_out"] > 0:
                data.loc[source, "decision_status"] = "partially_shifted_remainder_stays"
    data["production_delta"] = data.shift_in - data.shift_out
    data["production_after"] = data.planned_production + data.production_delta
    data["l2_action"] = data.shift_out > 0
    columns = ["policy", "origin", "source_datetime", "recipient_datetime", "shifted_units", "assumed_capacity_units"]
    return data, pd.DataFrame(transfers, columns=columns)


def audit_constraints(actions: pd.DataFrame, transfers: pd.DataFrame, capacity: float) -> pd.DataFrame:
    rows = []
    for day, group in actions.groupby(actions.datetime.dt.normalize(), sort=True):
        moved = transfers.loc[pd.to_datetime(transfers.source_datetime).dt.normalize() == day]
        finite_plan = (np.isfinite(group.planned_production.to_numpy(float)).all()
                       and bool((group.planned_production >= 0).all()))
        conservation = abs(float(group.production_delta.sum())) <= 1e-8
        receiver = group.shift_in > 0
        capacity_ok = bool((group.loc[receiver, "production_after"] <= capacity + 1e-8).all())
        future_ok = bool(((pd.to_datetime(moved.recipient_datetime) > pd.to_datetime(moved.source_datetime))
                          & (pd.to_datetime(moved.recipient_datetime) > pd.to_datetime(moved.origin))).all())
        no_invalid_action = bool((~(group.l1_action | group.l2_action) | group.valid_plan_day).all())
        nonnegative = bool((group.loc[group.valid_plan_day, "production_after"] >= -1e-8).all())
        rows.append({"date": day, "policy": group.policy.iloc[0], "n_hours": len(group),
                     "plan_total_before": float(group.planned_production.sum()) if finite_plan else None,
                     "plan_total_after": float(group.production_after.sum()) if finite_plan else None,
                     "plan_totals_available": finite_plan, "production_delta_sum": float(group.production_delta.sum()),
                     "production_conserved": conservation, "receiver_capacity_ok": capacity_ok,
                     "future_receivers_only": future_ok, "nonnegative_eligible_production": nonnegative,
                     "invalid_plan_no_action": no_invalid_action,
                     "manual_confirmation_hours": int((~group.valid_plan_day).sum()),
                     "infeasible_requested_units_stay_at_source": float(group.unallocated_units.sum()),
                     "passed": conservation and capacity_ok and future_ok and no_invalid_action and nonnegative})
    result = pd.DataFrame(rows)
    if not result.passed.all():
        raise ValueError("Policy feasibility audit failed")
    return result


def evaluate_policy(actions: pd.DataFrame, observed_peak, theta, response: dict, alpha: float,
                    *, evaluation_usable=None) -> tuple[pd.DataFrame, dict]:
    """Score valid observations only after fixing full-calendar decisions.

    Actions and their assumed execution costs cover the whole planned calendar.
    Outcomes, confusion counts and attributed false-action costs use only E5's
    valid target rows. Unknown-target actions stay explicitly unclassified.
    """
    truth = np.asarray(observed_peak, dtype=float)
    thresholds = np.broadcast_to(np.asarray(theta, dtype=float), truth.shape)
    if truth.shape != (len(actions),):
        raise ValueError("Observed peak must align with full-calendar actions")
    if not np.isfinite(thresholds).all() or not 0 <= alpha <= 1:
        raise ValueError("Invalid thresholds or alpha")
    usable = (np.ones(len(actions), dtype=bool) if evaluation_usable is None
              else _bool(pd.Series(evaluation_usable)).to_numpy())
    if usable.shape != truth.shape:
        raise ValueError("Evaluation quality flags must align with actions")
    valid = usable & np.isfinite(truth) & (truth >= 0)
    beta = float(response["beta_kw_per_unit"])
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("Response slope must be finite and nonnegative")
    observed = np.where(valid, truth, np.nan)
    raw = observed * (1. - float(alpha) * (2. / 3.) * actions.l1_action.to_numpy(float))
    raw += beta * actions.production_delta.to_numpy(float)
    # Flooring is an explicit response-model constraint, never an observed-range clip.
    simulated = np.maximum(0., raw)
    actual_event = valid & (truth >= thresholds)
    actual_normal = valid & ~actual_event
    alert = actions.alert.to_numpy(bool)
    l1 = actions.l1_action.to_numpy(bool)
    l2 = actions.l2_action.to_numpy(bool)
    acted = l1 | l2
    idx = pd.DatetimeIndex(actions.datetime)
    days = idx.normalize().unique()
    exposure = float(sum(1. / d.days_in_month for d in days))
    observed_exposure = float(sum(1. / (24. * d.days_in_month) for d in idx[valid]))
    out = actions[["datetime", "policy", "planned_production", "production_after", "l1_action", "shift_out", "shift_in"]].copy()
    out["alpha"] = alpha
    out["evaluation_usable"] = valid
    out["observed_peak_kw"] = observed
    out["assumed_peak_kw"] = simulated
    out["theta"] = thresholds
    out["new_peak_event"] = actual_normal & (simulated >= thresholds)
    original_max = float(truth[valid].max()) if valid.any() else None
    simulated_max = float(simulated[valid].max()) if valid.any() else None
    metrics = {
        "policy": actions.policy.iloc[0], "alpha": alpha, "n_hours": len(actions), "n_days": len(days),
        "n_eval": int(valid.sum()), "n_excluded": int((~valid).sum()),
        "observation_coverage": float(valid.mean()), "partial_observation": bool((~valid).any()),
        "n_flag_excluded": int((~usable).sum()), "n_invalid_target": int((~np.isfinite(truth) | (truth < 0)).sum()),
        "observed_peak_kw": original_max, "assumed_peak_kw": simulated_max,
        "assumed_peak_reduction_kw": original_max - simulated_max if valid.any() else None,
        "fractional_observed_months": exposure,
        "fractional_calendar_months": exposure,
        "fractional_valid_observation_months": observed_exposure,
        "peak_metric_scope": "quality-valid observed subset; incomplete maximum when partial_observation",
        "TP": int((actual_event & alert).sum()), "FN": int((actual_event & ~alert).sum()),
        "FP": int((actual_normal & alert).sum()), "TN": int((actual_normal & ~alert).sum()),
        "action_events": int(l1.sum() + l2.sum()), "l1_action_hours": int(l1.sum()),
        "l2_donor_hours": int(l2.sum()), "shifted_units": float(actions.shift_out.sum()),
        "evaluated_action_events": int((l1 & valid).sum() + (l2 & valid).sum()),
        "unknown_target_action_events": int((l1 & ~valid).sum() + (l2 & ~valid).sum()),
        "unknown_target_shifted_units": float(actions.loc[~valid, "shift_out"].sum()),
        "unknown_target_receiver_hours": int(((actions.shift_in > 0).to_numpy() & ~valid).sum()),
        "false_positive_action_events": int((l1 & actual_normal).sum() + (l2 & actual_normal).sum()),
        "false_positive_shifted_units": float(actions.loc[actual_normal, "shift_out"].sum()),
        "missed_peak_hours_no_action": int((actual_event & ~alert & ~acted).sum()),
        "actual_peak_hours_no_action": int((actual_event & ~acted).sum()),
        "new_peak_event_hours": int(out.new_peak_event.sum()),
        "hours_exceeding_original_observed_max": int((simulated > original_max + 1e-8).sum()) if valid.any() else None,
        "response_negative_floor_hours": int((raw < 0).sum()),
        "unallocated_units_stay_at_source": float(actions.unallocated_units.sum()),
        "manual_confirmation_hours": int((~actions.valid_plan_day).sum()),
    }
    return out, metrics


def cost_scenarios(config: dict) -> list[dict]:
    cfg = _settings(config)
    values = {(float(a), float(r), 0., 0.) for a, r in itertools.product(cfg["alpha_grid"], cfg["demand_rate_multipliers"])}
    values |= {(float(cfg["alpha"]), 1., float(c), float(u))
               for c, u in itertools.product(cfg["action_cost_krw"], cfg["shift_unit_cost_krw"])}
    return [{"scenario": f"A{a:g}_R{r:g}_C{c:g}_U{u:g}", "alpha": a, "rate_multiplier": r,
             "assumed_rate_krw_per_kw_month": float(cfg["demand_rate_krw_per_kw_month"]) * r,
             "action_cost_krw": c, "shift_unit_cost_krw": u}
            for a, r, c, u in sorted(values)]


def evaluate_cost(metrics: dict, scenario: dict) -> dict:
    reduction = metrics["assumed_peak_reduction_kw"]
    gross = (reduction * scenario["assumed_rate_krw_per_kw_month"] * metrics["fractional_observed_months"]
             if reduction is not None else None)
    execution = metrics["action_events"] * scenario["action_cost_krw"] + metrics["shifted_units"] * scenario["shift_unit_cost_krw"]
    false_cost = metrics["false_positive_action_events"] * scenario["action_cost_krw"] + metrics["false_positive_shifted_units"] * scenario["shift_unit_cost_krw"]
    unknown_cost = metrics["unknown_target_action_events"] * scenario["action_cost_krw"] + metrics["unknown_target_shifted_units"] * scenario["shift_unit_cost_krw"]
    complete = not metrics["partial_observation"]
    return {**metrics, **scenario, "assumed_observed_period_gross_difference_krw": gross if complete else None,
            "available_observation_gross_proxy_krw": gross,
            "assumed_execution_cost_krw": execution, "false_positive_action_cost_krw": false_cost,
            "unknown_target_action_cost_krw": unknown_cost,
            "assumed_observed_period_net_difference_krw": gross - execution if complete and gross is not None else None,
            "available_observation_net_proxy_krw": gross - execution if gross is not None else None,
            "cost_completeness": "all_target_hours_observed" if complete else "partial_observation_unconfirmed",
            "execution_cost_scope": "all planned actions, including hours with unobserved/invalid targets",
            "false_action_cost_scope": "valid observed target hours only; unknown-target action costs classified separately",
            "cost_role": "hypothetical observed-period demand-cost comparison; not an actual bill or energy saving"}


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    aliases = {"production": "planned_production", "생산량": "planned_production", "공장인원": "headcount"}
    for old, new in aliases.items():
        if new not in data and old in data:
            data[new] = data[old]
    if "datetime" not in data:
        raise ValueError("Snapshot data requires explicit datetime")
    data["datetime"] = pd.to_datetime(data.datetime)
    if "plan_known" not in data:
        if "is_erp_missing" not in data:
            raise ValueError("Missing plan-known provenance; cannot infer missing ERP as idle")
        data["plan_known"] = ~_bool(data.is_erp_missing)
    if "is_operating" not in data:
        data["is_operating"] = data.planned_production > 0
        if "headcount" in data:
            data["is_operating"] |= data.headcount > 0
    return data


def _load_training(snapshot: Path) -> pd.DataFrame:
    # The exporter owns validation and field loading; no notebook imports occur.
    from tools.analysis_snapshot import load_snapshot
    loaded = load_snapshot(Path(snapshot))
    # Warmup is introduced during feature construction and is absent in df.
    # Prefer the fully annotated feature frame; never silently omit a mask.
    for key in ("feat", "cleaned", "df", "data"):
        if key in loaded and isinstance(loaded[key], pd.DataFrame):
            frame = loaded[key]
            if "datetime" not in frame and isinstance(frame.index, pd.DatetimeIndex):
                frame = frame.reset_index(names="datetime")
            masks_present = all(name in frame for name in ("is_outage", "is_erp_missing", "is_warmup"))
            if masks_present and "y_peak" in frame and ("production" in frame or "생산량" in frame or "planned_production" in frame):
                return _canonical(frame)
    raise ValueError("Snapshot lacks production/peak rows with all three required quality masks for core-only fit")


def _verify_temporal_source(source: Path, selected: pd.DataFrame, snapshot: Path, config: dict) -> dict:
    """Never upgrade an unfinished, stale, or reduced E5 run to full E6."""
    from experiments.model_adapter import json_bytes
    from experiments.temporal_selection import DEFAULTS as TEMPORAL_DEFAULTS
    manifest_path = source.parent / "manifest.json"
    protocol_path = source.parent / "temporal_protocol.json"
    if not manifest_path.is_file() or not protocol_path.is_file():
        raise ValueError("E6 requires E5 manifest and protocol, not a standalone prediction CSV")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "temporal_selection" or manifest.get("protocol") != protocol:
        raise ValueError("E5 manifest/protocol linkage mismatch")
    fingerprint = protocol.get("fingerprint")
    unsigned = {k: v for k, v in protocol.items() if k != "fingerprint"}
    if not fingerprint or hashlib.sha256(json_bytes(unsigned)).hexdigest() != fingerprint:
        raise ValueError("E5 protocol fingerprint mismatch")
    expected_config = {**TEMPORAL_DEFAULTS, **config.get("temporal_selection", {})}
    if json_bytes(protocol.get("config")) != json_bytes(expected_config):
        raise ValueError("E5 protocol config differs from requested run")
    snapshot_meta = json.loads((Path(snapshot) / "metadata.json").read_text(encoding="utf-8"))
    if not protocol.get("snapshot_fingerprint") or protocol["snapshot_fingerprint"] != snapshot_meta.get("fingerprint"):
        raise ValueError("E5 protocol belongs to a different or unidentified snapshot")
    files = manifest.get("files", {})
    if files.get(source.name) != sha256_file(source):
        raise ValueError("E5 prediction CSV hash mismatch or absent hash")
    for name, expected in files.items():
        path = (source.parent / name).resolve()
        if source.parent.resolve() not in path.parents or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"E5 artifact missing or hash mismatch: {name}")
    observed = set(selected.outer.astype(str))
    full = (manifest.get("status") == "complete" and not manifest.get("failures")
            and manifest.get("completed_outer_count") == 4 and manifest.get("expected_outer_count") == 4
            and observed == {"O2", "O3", "O4", "O5"})
    allow_partial = config.get("policy", {}).get("allow_partial_temporal") is True
    if manifest.get("status") not in ("complete", "partial") or (not full and not allow_partial):
        raise ValueError("E5 is not verified complete with O2-O5; reduced development needs allow_partial_temporal=true")
    if full:
        for number, start in enumerate(expected_config["outer_starts"], 2):
            times = pd.DatetimeIndex(pd.to_datetime(selected.loc[selected.outer == f"O{number}", "datetime"]))
            expected = pd.date_range(start, periods=14 * 24, freq="h")
            if not times.equals(expected):
                raise ValueError(f"E5 O{number} must contain the complete 336-hour selected clock")
    return {"status": "complete" if full else "partial", "protocol_fingerprint": fingerprint,
            "snapshot_fingerprint": protocol["snapshot_fingerprint"], "manifest_sha256": sha256_file(manifest_path),
            "verified_outer": sorted(observed), "allow_partial_temporal": allow_partial}


def run(root: Path, snapshot: Path, output: Path, config: dict) -> dict:
    cfg = _settings(config.get("policy", {}))
    temporal_dir = Path(config.get("_run_context", {}).get("temporal_dir", Path(output).parent / "temporal"))
    source = Path(cfg.get("predictions_path", temporal_dir / "temporal_outer_predictions.csv"))
    if not source.is_absolute():
        source = Path(root) / source
    if not source.is_file():
        raise FileNotFoundError(f"E6 requires completed E5 outer predictions: {source}")
    raw = pd.read_csv(source)
    if "selected" not in raw:
        raise ValueError("E6 needs E5's explicit selected-model flag")
    selected = raw.loc[_bool(raw.selected)].copy()
    if selected.empty:
        raise ValueError("No selected outer predictions")
    if "evaluation_usable" not in selected:
        raise ValueError("E6 requires E5's explicit target-quality evaluation_usable flag")
    selected = selected.sort_values(["outer", "datetime"]).reset_index(drop=True)
    dependency = _verify_temporal_source(source, selected, Path(snapshot), config)
    training = _load_training(Path(snapshot))
    selected = _canonical(selected)
    selected = selected.sort_values(["outer", "datetime"]).reset_index(drop=True)
    scenarios = cost_scenarios(cfg)
    all_actions, all_transfers, all_audits, all_outcomes, all_costs, responses = [], [], [], [], [], []
    for outer, group in selected.groupby("outer", sort=True):
        group = group.sort_values("datetime").reset_index(drop=True)
        validate_times(group.datetime)
        if group.core_train_end.nunique() != 1 or group.theta.nunique() != 1 or group.tau.nunique() != 1:
            raise ValueError("Selected outer predictor must have frozen cutoff/theta/tau")
        response = fit_response(training, group.core_train_end.iloc[0], group.datetime.min(), cfg)
        response["outer"] = outer
        responses.append(response)
        decision_input = group.loc[:, list(ACTION_COLUMNS)].copy()
        prediction_hash = frame_hash(group[["datetime", "origin", "pred_peak", "theta", "tau"]])
        for policy in POLICIES:
            actions, transfers = select_actions(decision_input, policy, response["receiving_capacity_units"], cfg)
            audit = audit_constraints(actions, transfers, response["receiving_capacity_units"])
            for frame in (actions, transfers, audit):
                frame["outer"] = outer
                frame["prediction_sha256"] = prediction_hash
            all_actions.append(actions)
            all_transfers.append(transfers)
            all_audits.append(audit)
            metrics_by_alpha = {}
            for alpha in sorted({s["alpha"] for s in scenarios}):
                outcomes, metrics = evaluate_policy(actions, group.y_peak, group.theta, response, alpha,
                                                    evaluation_usable=group.evaluation_usable)
                outcomes["outer"] = outer
                outcomes["prediction_sha256"] = prediction_hash
                all_outcomes.append(outcomes)
                metrics.update({"outer": outer, "prediction_sha256": prediction_hash,
                                "evaluation_role": "retrospective_outer_fixed_prediction_open_loop"})
                metrics_by_alpha[alpha] = metrics
            all_costs.extend(evaluate_cost(metrics_by_alpha[s["alpha"]], s) for s in scenarios)
    artifacts = {
        "policy_actions.csv": pd.concat(all_actions, ignore_index=True),
        "policy_transfers.csv": pd.concat(all_transfers, ignore_index=True),
        "policy_constraint_audit.csv": pd.concat(all_audits, ignore_index=True),
        "policy_assumed_outcomes.csv": pd.concat(all_outcomes, ignore_index=True),
        "policy_cost_sensitivity.csv": pd.DataFrame(all_costs),
        "policy_response_fits.csv": pd.DataFrame(responses),
    }
    for name, frame in artifacts.items():
        write_csv(Path(output) / name, frame)
    contract = {
        "action_input_columns": list(ACTION_COLUMNS), "policies": list(POLICIES),
        "configuration": cfg, "cost_scenarios": scenarios,
        "prediction_feedback": False, "simulation_mode": "fixed predictions, open loop, post-hoc assumed response",
        "L1": "Spread assumed startup component within hour; alpha times 2/3 reduction, unchanged production",
        "L2": "Shift alert-hour plan production only to later, non-alert, known operating hours in same day",
        "missing_plan": "Any unconfirmed or invalid hourly plan makes whole day manual-confirmation/no-action",
        "plan_confirmation_provenance": "E5 plan_known derives from retrospectively diagnosed ERP-missing flags; availability at the forecast origin is assumed as a plan-confirmation proxy, not demonstrated",
        "response_fit": "OLS production + hour effects using each outer core only; negative slope clipped to 0",
        "response_quality": "Exclude outage, ERP-missing and warmup rows from both beta and capacity fits",
        "evaluation_quality": "Full-calendar actions; target metrics only where E5 evaluation_usable and observed peak is finite/nonnegative",
        "execution_cost_scope": "All actual simulated action events; unknown-target costs remain separate from valid-target false-action costs",
        "partial_observation": "Missing targets leave complete-period gross/net blank; available-observation proxies remain explicitly unconfirmed",
        "cost_formula": "valid-observation max reduction * assumed unit rate * nominal calendar-window fractional months - all-calendar action and moved-unit costs; incomplete-target calculation is an unconfirmed available-observation proxy only",
        "forbidden_claims": ["actual causal peak reduction", "kWh/energy-charge saving", "actual monthly/annual bill saving", "closed-loop operation validation"],
        "limitations": ["ERP production is a plan proxy; physical-time correspondence unverified",
                        "Receiver capacity and operating feasibility are assumptions, not machine constraints",
                        "Observed regression slope is not an intervention response model",
                        "No altered-load lag feedback, labor rescheduling, deadlines, or field validation",
                        "Cost exposures and maxima are reported separately for each outer window; no annualization"],
    }
    write_json(Path(output) / "policy_contract.json", contract)
    summary = {"experiment": "policy", "status": dependency["status"], "evaluation_role": "retrospective_outer_fixed_prediction_open_loop",
               "temporal_dependency": dependency,
               "source_sha256": sha256_file(source), "source_file": source.name,
               "outer_count": len(responses), "policies": list(POLICIES), "scenario_count": len(scenarios),
               "prediction_feedback": False, "constraint_audit_passed": True,
               "observation_coverage": {str(row["outer"]): row["observation_coverage"] for row in all_costs},
               "partial_observation": any(row["partial_observation"] for row in all_costs),
               "target_quality_by_outer": {str(row["outer"]): {name: row[name] for name in
                                           ("n_hours", "n_eval", "n_excluded", "observation_coverage", "partial_observation")}
                                           for row in all_costs},
               "response_fits": responses, "artifacts": {name: len(frame) for name, frame in artifacts.items()},
               "interpretation": "All scenarios retained. Hypothetical policy comparison, not measured operational or billing savings."}
    write_json(Path(output) / "summary.json", summary)
    return summary
