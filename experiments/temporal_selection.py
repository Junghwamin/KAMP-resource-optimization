"""E5: retrospective temporal selection with a separate calibration tail.

The fixed 44-feature, five-candidate experiment is narrower than replaying the
original full feature/LOFO search. Outer targets enter scoring only. Calibration
targets determine tau/isotonic, never model choice or final fitting.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.model_adapter import (
    CANDIDATES, LGB_BASE, OUTER_STARTS, atomic_json, calibrate, frame_hash, json_bytes,
    jsonable, load_data, metrics, partition, prediction_rows, predict_job, prepare_split,
    profile_audit, protocol_guard, sha, split_audit, valid_rows,
)

DEFAULTS = {
    "profile": "full", "seed": 42, "threads": 4, "n_estimators": 800,
    "trials_reg": 50, "trials_clf": 50, "gap_hours": 24, "calibration_days": 14,
    "inner_days": 14, "max_inner_folds": 3, "min_core_days": 14,
    "outer_starts": list(OUTER_STARTS), "outer_days": 14, "candidates": list(CANDIDATES),
    "weights": {"mae": .30, "peak_mae": .15, "recall": .25, "fold_std": .20, "feasible": .10},
    "calibration_min_positive": 5, "calibration_min_negative": 20,
    "calibration_fallback": "tau=theta; isotonic disabled", "isolation": "serial_subprocess",
    "feasibility": "prespecified_all_candidates_eligible_resource_time_not_scored",
    "resource_warning_seconds": 60, "tie_break": "prespecified_candidate_order",
    "undefined_metric_score": 0.0, "lgb_base": LGB_BASE,
    "search_space": {"num_leaves": [15, 127], "learning_rate": [.01, .2],
                     "max_depth": [3, 12], "min_child_samples": [5, 50],
                     "feature_fraction": [.6, 1.], "bagging_fraction": [.6, 1.]},
}


def inner_partitions(frame, outer_spec, config):
    """Most recent non-overlapping inner windows wholly inside outer core."""
    specs = []
    first = frame.index.min().normalize()
    for offset in reversed(range(config["max_inner_folds"])):
        end = outer_spec["core_end_exclusive"] - pd.Timedelta(days=offset*config["inner_days"])
        start = end - pd.Timedelta(days=config["inner_days"])
        spec = partition(start, eval_days=config["inner_days"],
                         calibration_days=config["calibration_days"], gap_hours=config["gap_hours"])
        if spec["core_end_exclusive"]-first < pd.Timedelta(days=config["min_core_days"]):
            continue
        specs.append(spec)
    return specs


def score_candidates(rows, config):
    """Deterministic prespecified score; wall-clock timing cannot select a model."""
    table = pd.DataFrame(rows)
    summary = []
    for model in config["candidates"]:
        part = table[table.model == model]
        if part.empty:
            raise ValueError(f"Missing prespecified candidate {model}")
        # Hour-weighted mean errors, pooled recall and fold-wise variation.
        total = float(part.n.sum())
        p_total = float((part.TP+part.FN).sum())
        peak_weight = part.TP + part.FN
        summary.append({"model": model, "mae": float((part.MAE*part.n).sum()/total),
                        "peak_mae": float((part["Peak-MAE"].fillna(0)*peak_weight).sum()/p_total) if p_total else np.nan,
                        "recall": float(part.TP.sum()/p_total) if p_total else np.nan,
                        "fold_std": float(part.MAE.std(ddof=1)) if len(part)>1 else 0.,
                        "feasible": 1., "inner_folds": len(part), "n": int(total)})
    result = pd.DataFrame(summary)
    score = np.zeros(len(result))
    for field, weight in config["weights"].items():
        values = result[field].to_numpy(dtype=float)
        good = np.isfinite(values)
        normalized = np.full(len(values), float(config["undefined_metric_score"]))
        if field == "feasible":
            normalized = values
        elif good.any():
            spread = values[good].max()-values[good].min()
            normalized[good] = 1. if spread == 0 else (values[good]-values[good].min())/spread
            if field in ("mae", "peak_mae", "fold_std") and spread != 0:
                normalized[good] = 1-normalized[good]
        result[field+"_normalized"] = normalized
        score += weight*normalized
    result["score"] = score
    # np.argmax keeps the first registered candidate on exact ties.
    winner = str(result.iloc[int(np.argmax(score))].model)
    result["selected"] = result.model == winner
    return winner, result


def _search_params(trial, config):
    s = config["search_space"]
    return {"num_leaves": trial.suggest_int("num_leaves", *s["num_leaves"]),
            "learning_rate": trial.suggest_float("learning_rate", *s["learning_rate"], log=True),
            "max_depth": trial.suggest_int("max_depth", *s["max_depth"]),
            "min_child_samples": trial.suggest_int("min_child_samples", *s["min_child_samples"]),
            "feature_fraction": trial.suggest_float("feature_fraction", *s["feature_fraction"]),
            "bagging_fraction": trial.suggest_float("bagging_fraction", *s["bagging_fraction"]),
            "bagging_freq": 1}


def _inner_prediction(root, cache, model, item, features, config, params=None):
    frame, masks, _, _ = item
    tr = frame.loc[masks["core"]]
    prediction_frame = pd.concat([frame.loc[masks["calibration"]], frame.loc[masks["evaluation"]]])
    return predict_job(root, cache, model, tr[features], tr[["y_avg", "y_peak", "y_cls"]],
                       prediction_frame[features], config, params=params)


def optimize(root, cache, inner, features, config, output, kind):
    """Replayable TPE search; each completed trial and numeric job is checkpointed."""
    import optuna
    from sklearn.metrics import average_precision_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    history = []
    def objective(trial):
        params = _search_params(trial, config)
        values, fit_seconds, fallbacks = [], 0., []
        for i, item in enumerate(inner):
            frame, masks, theta, _ = item
            cal_n = int(masks["calibration"].sum())
            target = frame.loc[masks["evaluation"]]
            pred, fit = _inner_prediction(root, cache, "lgb" if kind == "reg" else "clf",
                                          item, features, config, params)
            fit_seconds += fit["fit_seconds"]
            if kind == "reg":
                value = float(np.abs(target.y_avg.to_numpy()-pred["pred_avg"][cal_n:]).mean())
            else:
                labels = (target.y_peak >= theta).astype(int)
                value = float(average_precision_score(labels, pred["prob"][cal_n:])) if labels.any() else 0.
                if labels.nunique() < 2:
                    fallbacks.append({"inner": i+1, "reason": "single_class_validation_AP_fixed_0_if_no_positives_else_1"})
            if fit.get("fallback"):
                fallbacks.append({"inner": i+1, "reason": fit["fallback"]})
            values.append(value)
        value = float(np.mean(values))
        history.append({"trial": trial.number, "kind": kind, "value": value, "params": json_bytes(params).decode(),
                        "fold_values": json_bytes(values).decode(), "fit_seconds": fit_seconds,
                        "fallbacks": json_bytes(fallbacks).decode(), "state": "complete"})
        pd.DataFrame(history).to_csv(output / f"trials_{kind}.csv", index=False)
        return value
    study = optuna.create_study(direction="minimize" if kind == "reg" else "maximize",
                               sampler=optuna.samplers.TPESampler(seed=config["seed"]))
    study.optimize(objective, n_trials=config["trials_"+kind], show_progress_bar=False)
    return {**study.best_params, "bagging_freq": 1}, history


def select_on_core(root, cache, frame, features, spec, config, output):
    """Selection receives core only: cannot observe outer/calibration targets."""
    if len(frame) and frame.index.max() >= spec["core_end_exclusive"]:
        raise ValueError("Selection input includes data outside core")
    inner = []
    audit = []
    rejected_inner = []
    for i, inner_spec in enumerate(inner_partitions(frame, spec, config), 1):
        try:
            rebuilt, masks, theta = prepare_split(frame, features, inner_spec)
        except ValueError as exc:
            rejected_inner.append({"inner": i, "reason": str(exc), "spec": jsonable(inner_spec)})
            continue
        inner.append((rebuilt, masks, theta, inner_spec))
        audit.extend({"inner": i, **row} for row in split_audit(rebuilt, masks, theta, inner_spec, features))
    pd.DataFrame(audit).to_csv(output / "inner_split_audit.csv", index=False)
    if not inner:
        return {"selected_model": config["candidates"][0], "reg_params": {}, "clf_params": {},
                "selection_fallback": "no_usable_inner_windows_use_first_preregistered_candidate",
                "rejected_inner": rejected_inner, "scorecard": [], "trials_completed": {"reg": 0, "clf": 0}}
    reg_params, reg_history = optimize(root, cache, inner, features, config, output, "reg")
    clf_params, clf_history = optimize(root, cache, inner, features, config, output, "clf")
    all_rows, resources = [], []
    for model in config["candidates"]:
        for i, item in enumerate(inner, 1):
            d, masks, theta, _ = item
            cal, val = d.loc[masks["calibration"]], d.loc[masks["evaluation"]]
            prediction, fit = _inner_prediction(root, cache, model, item, features, config,
                                                reg_params if model == "lgb" else None)
            cal_info, _ = calibrate(cal.y_peak, prediction["pred_peak"][:len(cal)], theta,
                                    min_positive=config["calibration_min_positive"], min_negative=config["calibration_min_negative"])
            row = {"inner": i, "model": model, "theta": theta, "tau": cal_info["tau"],
                   "tau_source": "inner_calibration_tail", "fallback": cal_info["fallback"],
                   **metrics(val.y_avg, val.y_peak, prediction["pred_avg"][len(cal):],
                             prediction["pred_peak"][len(cal):], theta, cal_info["tau"])}
            all_rows.append(row)
            resources.append({"inner": i, "model": model, "fit_seconds": fit["fit_seconds"],
                              "cache_hit": fit["cache_hit"], "job_fingerprint": fit["fingerprint"],
                              "over_60s_warning": fit["fit_seconds"] > config["resource_warning_seconds"]})
    selected, scorecard = score_candidates(all_rows, config)
    scorecard.to_csv(output / "selection_scorecard.csv", index=False)
    pd.DataFrame(all_rows).to_csv(output / "inner_candidate_metrics.csv", index=False)
    pd.DataFrame(resources).to_csv(output / "inner_resources.csv", index=False)
    # Do not include measured time or cache-hit flags in the selection manifest.
    return {"selected_model": selected, "reg_params": reg_params, "clf_params": clf_params,
            "scorecard": jsonable(scorecard.to_dict("records")), "selection_fallback": None,
            "rejected_inner": rejected_inner, "inner_count": len(inner),
            "trials_completed": {"reg": len(reg_history), "clf": len(clf_history)}}


def evaluate_outer(root, cache, frame, features, spec, selection, config, outer):
    rebuilt, masks, theta = prepare_split(frame, features, spec)
    all_eval = ((rebuilt.index >= spec["eval_start"]) & (rebuilt.index < spec["eval_end_exclusive"]))
    core, cal = (rebuilt.loc[masks[k]] for k in ("core", "calibration"))
    ev = rebuilt.loc[all_eval]
    score_mask = masks["evaluation"][all_eval].to_numpy()
    if not np.isfinite(ev[features]).all().all():
        raise ValueError("Full-calendar policy predictions require finite repaired features; do not silently delete unknown-plan hours")
    inference = pd.concat([cal, ev])
    clf_pred, clf_fit = predict_job(root, cache, "clf", core[features], core[["y_avg", "y_peak", "y_cls"]],
                                     inference[features], config, selection["clf_params"])
    predictions, metric_rows, resources, calibration_rows = [], [], [], []
    resources.append({"outer": outer, "model": "clf", "fit_seconds": clf_fit["fit_seconds"],
                      "cache_hit": clf_fit["cache_hit"], "job_fingerprint": clf_fit["fingerprint"]})
    for model in config["candidates"]:
        pred, fit = predict_job(root, cache, model, core[features], core[["y_avg", "y_peak", "y_cls"]],
                               inference[features], config, selection["reg_params"] if model == "lgb" else None)
        cal_info, iso = calibrate(cal.y_peak, pred["pred_peak"][:len(cal)], theta, clf_pred["prob"][:len(cal)],
                                  min_positive=config["calibration_min_positive"], min_negative=config["calibration_min_negative"])
        outer_pred = {k: v[len(cal):] for k, v in pred.items()}
        part = prediction_rows(rebuilt, all_eval, outer_pred, theta, cal_info["tau"], spec,
                               model=model, outer=outer, selected=model == selection["selected_model"])
        part["prob_raw"] = clf_pred["prob"][len(cal):]
        part["prob_cal"] = iso.predict(part.prob_raw) if iso is not None else part.prob_raw
        part["probability_calibration"] = cal_info["probability_calibration"]
        part["evaluation_usable"] = score_mask
        part["high_production_boundary"] = float(core["prod"].quantile(.75))
        predictions.append(part)
        metric_rows.append({"outer": outer, "model": model, "selected": model == selection["selected_model"],
                            "theta": theta, "tau": cal_info["tau"], "n_core": len(core), "n_calibration": len(cal),
                            **metrics(ev.y_avg.to_numpy()[score_mask], ev.y_peak.to_numpy()[score_mask],
                                      outer_pred["pred_avg"][score_mask], outer_pred["pred_peak"][score_mask], theta, cal_info["tau"])})
        calibration_rows.append({"outer": outer, "model": model, **cal_info,
                                 "core_train_end": spec["core_train_end"], "refit_on_calibration": False,
                                 "calibration_target_hash": frame_hash(cal[["y_cls"]])})
        resources.append({"outer": outer, "model": model, "fit_seconds": fit["fit_seconds"], "cache_hit": fit["cache_hit"],
                          "job_fingerprint": fit["fingerprint"], "over_60s_warning": fit["fit_seconds"] > config["resource_warning_seconds"],
                          "fallback": fit.get("fallback"), "peak_memory_bytes": fit.get("peak_memory_bytes")})
    return pd.concat(predictions, ignore_index=True), metric_rows, resources, calibration_rows, split_audit(rebuilt, masks, theta, spec, features)


def classifier_metrics(rows):
    """Auxiliary classifier probabilities, scored only after frozen selection."""
    from sklearn.metrics import average_precision_score
    outputs = []
    for outer, part in rows[rows.selected & rows.evaluation_usable].groupby("outer"):
        truth = part.y_cls.to_numpy(dtype=int)
        for field in ("prob_raw", "prob_cal"):
            probability = part[field].to_numpy(dtype=float)
            alarm = probability >= .5
            positive = truth == 1
            tp, fp = int((positive & alarm).sum()), int((~positive & alarm).sum())
            fn, tn = int((positive & ~alarm).sum()), int((~positive & ~alarm).sum())
            outputs.append({"outer": outer, "probability": field, "n": len(part),
                            "threshold": .5, "threshold_source": "prespecified_not_tuned_on_outer",
                            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                            "Recall": tp/(tp+fn) if tp+fn else np.nan,
                            "Precision": tp/(tp+fp) if tp+fp else np.nan,
                            "F1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                            "PR-AUC": float(average_precision_score(truth, probability)) if len(np.unique(truth))>1 else np.nan,
                            "Brier": float(np.square(truth-probability).mean()),
                            "probability_calibration": part.probability_calibration.iloc[0]})
    return pd.DataFrame(outputs)


def run(root: Path, snapshot: Path, output: Path, config: dict) -> dict:
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = {**DEFAULTS, **config.get("temporal_selection", {})}
    if cfg["candidates"] != list(CANDIDATES):
        raise ValueError("Changing candidates requires a separately versioned protocol implementation")
    if cfg["feasibility"] != DEFAULTS["feasibility"] or cfg["tie_break"] != DEFAULTS["tie_break"]:
        raise ValueError("Only the preregistered deterministic selection rule is implemented")
    frame, features, metadata = load_data(snapshot)
    guard = protocol_guard(cfg, DEFAULTS, features, diagnostic_keys=("resource_warning_seconds",))
    full = guard["matches_registered_full"]
    if not metadata.get("fingerprint"):
        raise ValueError("Temporal selection requires a verified snapshot fingerprint")
    protocol = {"experiment": "temporal_selection", "config": cfg, "features": features,
                "evaluation_role": "retrospective_temporal_candidate_selection_not_fresh_independent_test",
                "scope": "44 fixed features and five candidates; original 62 transformations/LOFO search not replayed",
                "selection_feasibility": "All registered candidates eligible a priori; measured timing reported only as resources",
                "calibration_rule": "Core-only model and q95; tail-only tau/isotonic; no refit including calibration",
                "snapshot_fingerprint": metadata["fingerprint"], "full_protocol_guard": guard,
                "no_inner_theta_label_pooling": True, "data_condition": "D1_repeated_profiles_retained",
                "quality_rule": "exclude warmup, outage, ERP missing, nonfinite features/targets equally from fit/cal/score",
                "novelty_audit_role": "post_scoring_only_no_target_profiles_in_selection",
                "plan_availability_assumption": "plan_known/calendar derive from retrospective ERP missingness diagnosis, including realized daily power; treated as a prior-plan-confirmation proxy, not verified prospective information",
                "future_history": "Observed power through previous day 23h available at daily origin 00h; gap data may supply lags, never fit rows"}
    protocol["fingerprint"] = hashlib.sha256(json_bytes(protocol)).hexdigest()
    atomic_json(output / "temporal_protocol.json", protocol)
    atomic_json(output / "manifest.json", {"status": "running", "protocol": protocol})
    combined, metric_rows, resources, calibrations, audits, novelty, failures, selections = [], [], [], [], [], [], [], []
    started = time.perf_counter()
    for start in cfg["outer_starts"]:
        outer = f"O{list(OUTER_STARTS).index(start)+2}" if start in OUTER_STARTS else f"O{len(selections)+2}"
        folder = output / outer
        folder.mkdir(exist_ok=True)
        spec = partition(start, eval_days=cfg["outer_days"], calibration_days=cfg["calibration_days"], gap_hours=cfg["gap_hours"])
        atomic_json(folder / "checkpoint.json", {"status": "running", "spec": spec, "protocol_fingerprint": protocol["fingerprint"]})
        try:
            core_frame = frame.loc[frame.index < spec["core_end_exclusive"]].copy(deep=True)
            choice = select_on_core(root, root / ".cache/additional_fits", core_frame, features, spec, cfg, folder)
            selection_manifest = {"outer": outer, "spec": spec, "core_input_hash": frame_hash(core_frame),
                                  "theta_core_q95": float(core_frame.loc[valid_rows(core_frame, features), "y_peak"].quantile(.95)),
                                  "protocol_fingerprint": protocol["fingerprint"], **choice}
            atomic_json(folder / "selection_manifest.json", selection_manifest)
            # This boundary is after all HPO and model selection have been frozen.
            rows, m, r, c, a = evaluate_outer(root, root / ".cache/additional_fits", frame, features, spec, choice, cfg, outer)
            rows.to_csv(folder / "outer_predictions.csv", index=False)
            atomic_json(folder / "calibration_manifest.json", c)
            combined.append(rows)
            metric_rows.extend(m)
            resources.extend(r)
            calibrations.extend(c)
            audits.extend({"outer": outer, **row} for row in a)
            novelty.extend({"outer": outer, **row} for row in profile_audit(frame, spec))
            selections.append({"outer": outer, **choice})
            atomic_json(folder / "checkpoint.json", {"status": "complete", "spec": spec,
                        "protocol_fingerprint": protocol["fingerprint"], "selection_sha256": sha(folder / "selection_manifest.json"),
                        "predictions_sha256": sha(folder / "outer_predictions.csv")})
        except Exception as exc:
            failure = {"outer": outer, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            atomic_json(folder / "checkpoint.json", {"status": "failed", **failure})
    filenames = []
    if combined:
        all_predictions = pd.concat(combined, ignore_index=True)
        all_predictions.to_csv(output / "temporal_outer_predictions.csv", index=False)
        classifier_metrics(all_predictions).to_csv(output / "temporal_peak_classifier_metrics.csv", index=False)
        filenames.extend(["temporal_outer_predictions.csv", "temporal_peak_classifier_metrics.csv"])
    for name, values in (("temporal_outer_metrics.csv", metric_rows), ("temporal_resources.csv", resources),
                         ("temporal_split_audit.csv", audits), ("temporal_profile_novelty.csv", novelty)):
        pd.DataFrame(values).to_csv(output / name, index=False)
        filenames.append(name)
    atomic_json(output / "temporal_calibration.json", calibrations)
    atomic_json(output / "temporal_selection_log.json", selections)
    filenames.extend(["temporal_calibration.json", "temporal_selection_log.json"])
    manifest = {"status": "complete" if full and not failures and len(combined) == 4 else "partial",
                "experiment": "temporal_selection", "failures": failures,
                "completed_outer_count": len(combined), "expected_outer_count": 4,
                "seconds": time.perf_counter()-started, "protocol": jsonable(protocol),
                "files": {f: sha(output / f) for f in filenames}}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "summary.json", manifest)
    return manifest
