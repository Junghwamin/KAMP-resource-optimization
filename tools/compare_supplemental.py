"""Compare two completed E1-E6 runs without fitting models or importing stages.

This verifies the supplied records; it does not establish that either directory
was produced by a fresh execution on KAMP. Protocol version 1 is supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ATOL = RTOL = 1e-6
EXPERIMENTS = ("bootstrap", "diagnostics", "sensitivity", "controlled", "temporal", "policy")
OUTERS = ("O2", "O3", "O4", "O5")
SUMMARY_NAMES = {"controlled": "controlled_regime", "temporal": "temporal_selection"}

# Every CSV is aligned on its declared scientific key, never physical row order.
CSV_KEYS = {
    "bootstrap/bootstrap_block_sensitivity.csv": ("block_days",),
    "bootstrap/paired_daily_errors.csv": ("date",),
    "diagnostics/condition_confusion_rates.csv": ("condition",),
    "diagnostics/diagnostics_selected_oof.csv": ("datetime", "model", "data_condition"),
    "diagnostics/production_hour_cells.csv": ("production_band", "hour"),
    "diagnostics/temperature_stratified_cells.csv": ("hour", "is_shutdown", "temperature_band"),
    "sensitivity/feature_consistency_audit.csv": ("scenario", "date"),
    "sensitivity/input_sensitivity_metrics.csv": ("scenario",),
    "sensitivity/input_sensitivity_predictions.csv": ("scenario", "datetime"),
    "controlled/controlled_profile_novelty.csv": ("outer", "date"),
    "controlled/controlled_regime_metrics.csv": ("outer", "model"),
    "controlled/controlled_regime_paired_ci.csv": ("outer", "target"),
    "controlled/controlled_regime_predictions.csv": ("outer", "model", "datetime"),
    "controlled/controlled_resources.csv": ("outer", "model"),
    "controlled/controlled_split_audit.csv": ("outer", "role"),
    "temporal/temporal_outer_predictions.csv": ("outer", "model", "datetime"),
    "temporal/temporal_outer_metrics.csv": ("outer", "model"),
    "temporal/temporal_peak_classifier_metrics.csv": ("outer", "probability"),
    "temporal/temporal_resources.csv": ("outer", "model"),
    "temporal/temporal_split_audit.csv": ("outer", "role"),
    "temporal/temporal_profile_novelty.csv": ("outer", "date"),
    "policy/policy_actions.csv": ("outer", "policy", "datetime"),
    "policy/policy_transfers.csv": ("outer", "policy", "source_datetime", "recipient_datetime"),
    "policy/policy_constraint_audit.csv": ("outer", "policy", "date"),
    "policy/policy_assumed_outcomes.csv": ("outer", "policy", "alpha", "datetime"),
    "policy/policy_cost_sensitivity.csv": ("outer", "policy", "scenario"),
    "policy/policy_response_fits.csv": ("outer",),
}
JSON_FILES = {f"{name}/summary.json" for name in EXPERIMENTS} | {
    "sensitivity/scenario_protocol.json", "controlled/controlled_protocol.json", "controlled/manifest.json",
    "temporal/temporal_protocol.json", "temporal/manifest.json", "temporal/temporal_calibration.json",
    "temporal/temporal_selection_log.json", "policy/policy_contract.json",
}
for _outer in OUTERS:
    CSV_KEYS.update({
        f"controlled/{_outer}_predictions.csv": ("model", "datetime"),
        f"controlled/{_outer}_split_audit.csv": ("role",),
        f"temporal/{_outer}/inner_split_audit.csv": ("inner", "role"),
        f"temporal/{_outer}/trials_reg.csv": ("trial",),
        f"temporal/{_outer}/trials_clf.csv": ("trial",),
        f"temporal/{_outer}/selection_scorecard.csv": ("model",),
        f"temporal/{_outer}/inner_candidate_metrics.csv": ("inner", "model"),
        f"temporal/{_outer}/inner_resources.csv": ("inner", "model"),
        f"temporal/{_outer}/outer_predictions.csv": ("model", "datetime"),
    })
    JSON_FILES.update({f"controlled/{_outer}_checkpoint.json", f"temporal/{_outer}/checkpoint.json",
                       f"temporal/{_outer}/selection_manifest.json", f"temporal/{_outer}/calibration_manifest.json"})

VOLATILE = {"fit_seconds", "elapsed_seconds", "seconds", "started_epoch", "created_at", "completed_at",
            "recorded_at", "timestamp", "cache_hit", "peak_memory_bytes", "over_60s_warning", "log", "log_path",
            "runtime", "platform", "os", "python", "packages", "source", "files", "files_sha256", "bundle_id"}
EXACT_JSON_BRANCHES = {"config", "configuration", "spec", "features", "candidates", "candidate_models", "params",
                       "reg_params", "clf_params", "search_space", "scenarios", "cost_scenarios", "weights"}
EXACT = set("""datetime date origin history_end train_end core_train_end core_end_exclusive cal_start cal_end
cal_end_exclusive eval_start eval_end_exclusive first last evaluation_start fit_max_datetime
outer fold inner model target role kind trial state condition data_condition scenario production_band
temperature_band hour selected selected_model theta theta_core_q95 y_avg y_peak y_cls y_cls_reference_187
peak_pred_label regime is_warmup is_outage is_erp_missing eligible evaluation_usable plan_known plan_known_source
is_operating is_shutdown is_first_day_back prod production planned_production headcount plan_headcount temp
threshold_scope threshold_role evaluation_role probability probability_calibration fallback tau_source method
sampling p_method ci_includes_zero supplementary sparse_cell mask_counts_overlap seen_in_core extra_gate
fallback_regressor_per_target regressor_parameters policy valid_plan_day alert l1_action l2_action decision_status
plan_totals_available production_conserved receiver_capacity_ok future_receivers_only
nonnegative_eligible_production invalid_plan_no_action passed new_peak_event beta_status negative_slope_clipped
response_interpretation capacity_interpretation partial_observation cost_completeness cost_role peak_metric_scope
execution_cost_scope false_action_cost_scope source_datetime recipient_datetime threshold_source
historical_power_features_unchanged production_zero_support_unchanged headcount_zero_support_unchanged
production_daily_total_rule_passed shutdown_unchanged derived_features_checked high_production_threshold
high_production_boundary production_q50 q50 q75 temperature_q25 temperature_q75 alpha rate_multiplier
assumed_rate_krw_per_kw_month action_cost_krw shift_unit_cost_krw threshold n P N TP FP FN TN days seed block_days
n_boot positives negatives inner_folds trees_per_regressor threads core_unique_profiles core_duplicate_days_retained_D1
ci_n_boot ci_cluster_dates ci_condition_dates missed_peak_hours_no_action actual_peak_hours_no_action
hours_exceeding_original_observed_max params fallbacks""".split())
FLOATS = set("""pred_avg pred_peak prob_raw prob_cal tau regression_tau rf_mae final_mae diff ci_low ci_high
ci90_low ci90_high p_value_approx miss_rate false_alarm_rate precision recall f1 mae peak_mae peak15_mae
peak15_mae_all peak_rate mean_peak_kw max_feature_change gate_changed_rate mae_delta_c0 MAE Peak-MAE Peak15-MAE
Recall Precision F1 false_positive_rate MAE_single_minus_regime PR-AUC Brier fold_std feasible score value
shift_requested shift_out shift_in unallocated_units production_delta production_after shifted_units
assumed_capacity_units plan_total_before plan_total_after production_delta_sum infeasible_requested_units_stay_at_source
observed_peak_kw assumed_peak_kw assumed_peak_reduction_kw fractional_observed_months fractional_calendar_months
fractional_valid_observation_months observation_coverage unknown_target_shifted_units false_positive_shifted_units
unallocated_units_stay_at_source beta_raw_kw_per_unit beta_kw_per_unit ols_r2 receiving_capacity_units
assumed_observed_period_gross_difference_krw available_observation_gross_proxy_krw assumed_execution_cost_krw
false_positive_action_cost_krw unknown_target_action_cost_krw assumed_observed_period_net_difference_krw
available_observation_net_proxy_krw""".split())
JSON_CELLS = {"regressor_parameters": True, "params": True, "fallbacks": True, "fold_values": False}
PREDICTION_REQUIRED = {"datetime", "y_avg", "y_peak", "y_cls", "pred_avg", "pred_peak", "theta", "tau", "peak_pred_label"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Nonstandard JSON {token}")))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact_path(folder, relative):
    require(isinstance(relative, str) and "\\" not in relative, "Unsupported artifact path separator")
    value = (Path(folder) / relative).resolve()
    require(value.is_relative_to(Path(folder).resolve()) and value != Path(folder).resolve(), f"Artifact escapes folder: {relative}")
    return value


def volatile(name):
    return (name in VOLATILE or name == "fingerprint" or name.endswith(("_sha256", "_fingerprint", "_hash")))


def column_role(name):
    if volatile(name):
        return "provenance_only"
    if name in JSON_CELLS:
        return "json_exact" if JSON_CELLS[name] else "json_tolerant"
    if exact_field(name):
        return "exact"
    if name in FLOATS or name.endswith(("_ci_low", "_ci_high", "_na_fraction", "_normalized")):
        return "numeric_tolerance"
    raise ValueError(f"Unsupported CSV column: {name}")


def exact_field(name):
    return name in EXACT or name.startswith("n_") or name.endswith(("_count", "_hours", "_days", "_events", "_valid_replicates"))


def minimum_columns(relative):
    columns = set(CSV_KEYS[relative])
    if relative.endswith("_predictions.csv") or relative.endswith("diagnostics_selected_oof.csv"):
        columns |= PREDICTION_REQUIRED
    if relative in ("temporal/temporal_outer_predictions.csv",) or re.fullmatch(r"temporal/O[2-5]/outer_predictions.csv", relative):
        columns |= {"selected", "evaluation_usable", "is_warmup", "is_outage", "is_erp_missing", "plan_known"}
    return columns


def read_table(path, relative):
    # Read as text first: this preserves exact labels, keys and missing patterns.
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    require(not table.columns.duplicated().any(), f"Duplicate CSV column: {relative}")
    require(minimum_columns(relative) <= set(table), f"Missing required CSV columns: {relative}")
    roles = {name: column_role(name) for name in table}
    keys = list(CSV_KEYS[relative])
    require(not table[keys].isin(["", "NA", "NaN", "nan"]).any().any(), f"Missing row key: {relative}")
    require(not table.duplicated(keys).any(), f"Duplicate row key: {relative}")
    return table.sort_values(keys, kind="stable").reset_index(drop=True), roles


def verify_nested_hashes(folder, obj, label):
    if isinstance(obj, dict):
        if isinstance(obj.get("files"), dict):
            for name, value in obj["files"].items():
                require(artifact_path(folder, name).is_file() and digest(artifact_path(folder, name)) == value,
                        f"Nested manifest hash mismatch: {label}/{name}")
        protocol = obj.get("protocol")
        if isinstance(protocol, dict) and "fingerprint" in protocol:
            unsigned = {k: v for k, v in protocol.items() if k != "fingerprint"}
            require(hashlib.sha256(canonical(unsigned).encode()).hexdigest() == protocol["fingerprint"], f"Protocol fingerprint mismatch: {label}")


def check_json_schema(label, obj):
    list_root = label in {"temporal/temporal_calibration.json", "temporal/temporal_selection_log.json"} or label.endswith("/calibration_manifest.json")
    require(isinstance(obj, list if list_root else dict), f"Unsupported JSON root schema: {label}")
    if list_root:
        require(all(isinstance(row, dict) for row in obj), f"Unsupported JSON record schema: {label}")
        return
    required = set()
    if label.endswith("summary.json"):
        required = {"status", "experiment"}
    elif label.endswith("selection_manifest.json"):
        required = {"selected_model", "reg_params", "theta_core_q95"}
    elif label.endswith("checkpoint.json") or label.endswith("manifest.json"):
        required = {"status"}
    elif label in {"controlled/controlled_protocol.json", "temporal/temporal_protocol.json"}:
        required = {"config", "features"}
    elif label == "sensitivity/scenario_protocol.json":
        required = {"scenarios", "history", "shift_boundary"}
    elif label == "policy/policy_contract.json":
        required = {"configuration", "policies", "action_input_columns"}
    require(required <= set(obj), f"Missing required JSON fields: {label}: {sorted(required-set(obj))}")


def validate_run(directory):
    """Verify all six producers and schemas before any between-run comparison."""
    root = Path(directory).resolve()
    config = read_json(root / "protocol.json")
    require(config.get("schema_version") == 1 and config.get("protocol") == "additional-experiments-2026-10-06",
            "Unsupported supplemental protocol/schema version")
    require({"seed", "bootstrap", "diagnostics", "sensitivity", "controlled_regime", "temporal_selection", "policy"} <= set(config),
            "Unsupported/incomplete scientific configuration schema")
    status = read_json(root / "run_status.json")
    require(status.get("status") == "complete" and status.get("all_six_verified") is True,
            "Aggregate run is incomplete or missing six-experiment verification")
    require(set(status.get("experiments", {})) == set(EXPERIMENTS)
            and all(status["experiments"][name] == "complete" for name in EXPERIMENTS), "Six experiment statuses are not complete")
    documents, tables, provenance = {}, {}, {}
    for name in EXPERIMENTS:
        folder = root / name
        manifest = read_json(folder / "executor_manifest.json")
        require(manifest.get("status") == "complete" and manifest.get("experiment") == name,
                f"Incomplete or wrong experiment manifest: {name}")
        inputs = manifest.get("inputs", {})
        require(inputs.get("experiment") == name and inputs.get("config") == config, f"Executor protocol/config mismatch: {name}")
        require(bool(inputs.get("source")) and bool(inputs.get("runtime")), f"Missing source/runtime provenance: {name}")
        require(hashlib.sha256(canonical(inputs).encode()).hexdigest() == manifest.get("fingerprint"), f"Executor fingerprint mismatch: {name}")
        hashes = manifest.get("files_sha256", {})
        require(isinstance(hashes, dict) and "summary.json" in hashes, f"Missing artifact hash inventory: {name}")
        expected = {p.split("/", 1)[1] for p in set(CSV_KEYS) | JSON_FILES if p.startswith(name+"/")}
        require(set(hashes) == expected, f"Unsupported/incomplete artifact inventory {name}: missing={sorted(expected-set(hashes))}, extra={sorted(set(hashes)-expected)}")
        for relative, value in hashes.items():
            path = artifact_path(folder, relative)
            require(path.is_file() and digest(path) == value, f"Artifact missing or hash mismatch: {name}/{relative}")
            label = f"{name}/{relative}"
            if label in CSV_KEYS:
                tables[label] = read_table(path, label)
            else:
                obj = read_json(path)
                check_json_schema(label, obj)
                if isinstance(obj, dict) and "status" in obj:
                    require(obj["status"] == "complete", f"Incomplete nested artifact: {label}")
                verify_nested_hashes(folder, obj, label)
                if label.endswith("_protocol.json") and isinstance(obj, dict) and "fingerprint" in obj:
                    require(hashlib.sha256(canonical({k:v for k,v in obj.items() if k != "fingerprint"}).encode()).hexdigest()
                            == obj["fingerprint"], f"Protocol self-hash mismatch: {label}")
                documents[label] = obj
        summary = documents[f"{name}/summary.json"]
        require(summary.get("status") == "complete" and summary.get("experiment") == SUMMARY_NAMES.get(name, name), f"Summary incomplete/unsupported: {name}")
        require(summary == manifest.get("summary"), f"Executor and stored summary disagree: {name}")
        if name in ("controlled", "temporal"):
            require(summary.get("completed_outer_count") == 4 and summary.get("expected_outer_count") == 4
                    and summary.get("failures") == [], f"Outer coverage incomplete: {name}")
            require(summary.get("protocol", {}).get("full_protocol_guard", {}).get("matches_registered_full") is True,
                    f"Altered scientific protocol is not full: {name}")
        provenance[name] = {"executor_manifest_sha256": digest(folder / "executor_manifest.json"),
                            "executor_manifest": manifest}
    return {"config": config, "status": status, "documents": documents, "tables": tables,
            "provenance": provenance, "root": root}


def normalized_scalar(value):
    """Numeric formatting differences are harmless, while numeric values are exact."""
    if value in ("", "NA", "NaN", "nan", None):
        return ("missing", None)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return ("boolean", value.lower() == "true")
    if isinstance(value, str):
        try:
            number = float(value)
            if np.isfinite(number):
                return ("number", number)
        except ValueError:
            pass
    return ("literal", value)


def compare_json(left, right, path, differences, *, exact=False, ignored=None):
    if isinstance(left, dict) and isinstance(right, dict):
        keys = (set(left) | set(right)) if exact else {k for k in set(left) | set(right) if not volatile(k)}
        if ignored is not None and not exact:
            ignored.extend(path+"/"+k for k in set(left) | set(right) if volatile(k))
        for key in sorted(keys):
            if key not in left or key not in right:
                differences.append({"path": path+"/"+key, "reason": "JSON field missing"})
            else:
                compare_json(left[key], right[key], path+"/"+key, differences,
                             exact=exact or key in EXACT_JSON_BRANCHES or exact_field(key),
                             ignored=ignored)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append({"path": path, "reason": "JSON list length differs"})
            return
        if not exact and left and all(isinstance(v, dict) for v in left+right):
            for keys in (("outer", "model"), ("model",), ("outer",), ("block_days",), ("scenario",)):
                if all(all(k in v for k in keys) for v in left+right):
                    def key(row):
                        return tuple(str(row[k]) for k in keys)
                    require(len({key(v) for v in left}) == len(left) and len({key(v) for v in right}) == len(right), f"Duplicate JSON row identity: {path}")
                    left, right = sorted(left, key=key), sorted(right, key=key)
                    break
        for i, (a, b) in enumerate(zip(left, right)):
            compare_json(a, b, f"{path}/{i}", differences, exact=exact, ignored=ignored)
        return
    numeric = isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool)
    if numeric and not exact and (isinstance(left, float) or isinstance(right, float)):
        match = bool(np.isclose(left, right, atol=ATOL, rtol=RTOL, equal_nan=False))
    else:
        match = type(left) is type(right) and left == right
        if exact and numeric:
            match = left == right
    if not match:
        differences.append({"path": path, "reason": "exact value differs" if exact else "JSON value/tolerance differs",
                            "reference": left, "current": right})


def compare_table(left, right, label, roles):
    differences, ignored = [], []
    if set(left) != set(right):
        return [{"path": label, "reason": "CSV column schema differs"}], {"artifact": label, "status": "failed"}
    keys = list(CSV_KEYS[label])
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        return [{"path": label, "reason": "Scientific row identities/count differ", "key_columns": keys}], {"artifact": label, "status": "failed"}
    max_abs = 0.
    for col in left:
        role = roles[col]
        if role == "provenance_only":
            ignored.append(col)
            continue
        a, b = left[col], right[col]
        missing_a, missing_b = a.isin(["", "NA", "NaN", "nan"]), b.isin(["", "NA", "NaN", "nan"])
        if not missing_a.equals(missing_b):
            differences.append({"path": f"{label}/{col}", "reason": "Missing/NaN pattern differs"})
            continue
        if role == "numeric_tolerance":
            av, bv = pd.to_numeric(a[~missing_a], errors="raise").to_numpy(float), pd.to_numeric(b[~missing_b], errors="raise").to_numpy(float)
            require(np.isfinite(av).all() and np.isfinite(bv).all(), f"Infinite numeric value: {label}/{col}")
            bad = ~np.isclose(av, bv, atol=ATOL, rtol=RTOL)
            if len(av):
                max_abs = max(max_abs, float(np.max(np.abs(av-bv))))
            if bad.any():
                positions = np.flatnonzero(~missing_a)[np.flatnonzero(bad)[:5]]
                differences.append({"path": f"{label}/{col}", "reason": "Numeric tolerance exceeded", "count": int(bad.sum()),
                                    "examples": [{"key": left.loc[int(i), keys].to_dict(), "reference": a.iloc[i], "current": b.iloc[i]} for i in positions]})
        elif role.startswith("json_"):
            for i in np.flatnonzero(~missing_a):
                def decode(value):
                    return json.loads(value, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Nonstandard JSON cell {token}")))
                compare_json(decode(a.iloc[i]), decode(b.iloc[i]), f"{label}/{col}/{i}", differences, exact=role == "json_exact")
        else:
            bad = np.asarray([normalized_scalar(x) != normalized_scalar(y) for x, y in zip(a, b)])
            if bad.any():
                positions = np.flatnonzero(bad)[:5]
                differences.append({"path": f"{label}/{col}", "reason": "Exact column differs", "count": int(bad.sum()),
                                    "examples": [{"key": left.loc[int(i), keys].to_dict(), "reference": a.iloc[i], "current": b.iloc[i]} for i in positions]})
    return differences, {"artifact": label, "status": "passed" if not differences else "failed", "rows": len(left),
                         "key_columns": keys, "column_rules": roles, "provenance_only_columns": ignored,
                         "maximum_absolute_numeric_difference": max_abs}


def compare_runs(reference, current):
    report = {"schema_version": 1, "comparison": "supplemental_E1_E6_cross_run", "status": "failed",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "atol": ATOL, "rtol": RTOL,
              "scope": "All six completed experiment artifacts, scientific configurations, selection/calibration records and keyed row outcomes",
              "excluded_equality": "Runtime/OS, wall time, memory, cache hits, provenance hash values and timestamps; each run's own hashes are verified",
              "execution_claim": "Comparison of supplied directories only; does not prove a fresh KAMP execution",
              "reference_directory_label": Path(reference).name, "current_directory_label": Path(current).name,
              "input_validation": {}, "differences": [], "artifacts": [], "provenance": {}}
    runs = {}
    for side, directory in (("reference", reference), ("current", current)):
        try:
            runs[side] = validate_run(directory)
            report["input_validation"][side] = {"status": "passed", "experiments": list(EXPERIMENTS)}
            report["provenance"][side] = {"experiments": runs[side]["provenance"], "aggregate_status": runs[side]["status"]}
        except (ValueError, TypeError, OSError, KeyError, pd.errors.ParserError) as exc:
            report["input_validation"][side] = {"status": "failed", "reason": str(exc)}
    if len(runs) != 2:
        return report
    left, right = runs["reference"], runs["current"]
    compare_json(left["config"], right["config"], "protocol.json", report["differences"], exact=True)
    try:
        for label in sorted(CSV_KEYS):
            differences, result = compare_table(left["tables"][label][0], right["tables"][label][0], label, left["tables"][label][1])
            report["differences"].extend(differences)
            report["artifacts"].append(result)
        for label in sorted(JSON_FILES):
            before, ignored = len(report["differences"]), []
            compare_json(left["documents"][label], right["documents"][label], label, report["differences"], ignored=ignored)
            report["artifacts"].append({"artifact": label, "status": "passed" if before == len(report["differences"]) else "failed",
                                        "provenance_only_paths": sorted(set(ignored))})
    except (ValueError, TypeError, KeyError) as exc:
        report["differences"].append({"path": "comparison", "reason": f"Unsupported/invalid value schema: {exc}"})
    report["status"] = "passed" if not report["differences"] else "failed"
    report["artifact_count"] = len(report["artifacts"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_runs(args.reference, args.current)
    destination = args.output.resolve()
    for directory in (args.reference.resolve(), args.current.resolve()):
        if any(destination.is_relative_to(directory / name) for name in EXPERIMENTS):
            parser.error("Comparison output must be outside the six experiment artifact folders")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"SUPPLEMENTAL_COMPARE_{report['status'].upper()} {destination}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
