"""Small artifact fixtures use the actual writer's names; no fitting or stage import."""
from __future__ import annotations

import copy
import json
import shutil

import pandas as pd
import pytest

from tools import compare_supplemental as c


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def resign(root, experiment):
    """Keep fixtures internally intact after a scientific or harmless change."""
    folder = root / experiment
    files = {p.relative_to(folder).as_posix(): c.digest(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name != "executor_manifest.json"}
    old = c.read_json(folder / "executor_manifest.json") if (folder / "executor_manifest.json").exists() else {}
    inputs = old.get("inputs", {"experiment": experiment, "config": c.read_json(root / "protocol.json"),
                                "source": {"fixture-source.py": "fixture-source-hash"},
                                "runtime": {"python": "3.11.9", "platform": "fixture-Windows"}})
    manifest = {"status": "complete", "experiment": experiment, "inputs": inputs,
                "fingerprint": __import__("hashlib").sha256(c.canonical(inputs).encode()).hexdigest(),
                "elapsed_seconds": old.get("elapsed_seconds", 1.),
                "summary": c.read_json(folder / "summary.json"), "files_sha256": files}
    write_json(folder / "executor_manifest.json", manifest)


def value(column, index):
    if column in ("date", "datetime", "source_datetime", "recipient_datetime"):
        return f"2021-07-{7+index:02d} 08:00:00"
    if column == "outer":
        return "O2"
    if column == "model":
        return "lgb"
    if column == "data_condition":
        return "D1"
    if column in ("role", "condition", "production_band", "temperature_band", "scenario", "policy", "target", "probability"):
        return f"fixture_{index}"
    if column in ("is_warmup", "is_outage", "is_erp_missing"):
        return False
    if column in ("selected", "evaluation_usable", "plan_known"):
        return True
    if column in ("y_cls", "peak_pred_label"):
        return index
    if column in ("y_avg", "y_peak"):
        return 100.+index
    if column in ("pred_avg", "pred_peak"):
        return 101.25+index
    if column == "theta":
        return 187.
    if column == "tau":
        return 174.25
    if column in ("fold", "inner", "trial", "hour", "block_days"):
        return index+1
    return index


@pytest.fixture
def runs(tmp_path, project_root):
    reference = tmp_path / "reference"
    reference.mkdir()
    config = c.read_json(project_root / "experiments/config.json")
    write_json(reference / "protocol.json", config)
    write_json(reference / "run_status.json", {"status": "complete", "all_six_verified": True,
               "experiments": {name: "complete" for name in c.EXPERIMENTS}, "runtime": {"platform": "fixture-Windows"}})
    for relative in c.CSV_KEYS:
        columns = sorted(c.minimum_columns(relative))
        table = pd.DataFrame([{col: value(col, i) for col in columns} for i in range(2)])
        # Include actual writer output fields exercised by the tests.
        if relative.endswith("resources.csv"):
            table["fit_seconds"] = [1., 2.]
            table["cache_hit"] = False
            table["job_fingerprint"] = "fixture-job"
        if relative.endswith("selection_scorecard.csv"):
            table["score"] = [.4, .5]
            # This artifact's real key is model; ensure distinct keys.
            table["model"] = ["naive", "lgb"]
            table["selected"] = [False, True]
        if table.duplicated(list(c.CSV_KEYS[relative])).any():
            last = c.CSV_KEYS[relative][-1]
            table[last] = [f"fixture_{last}_0", f"fixture_{last}_1"]
        path = reference / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    for relative in c.JSON_FILES:
        experiment = relative.split("/")[0]
        if relative.endswith("summary.json"):
            obj = {"experiment": c.SUMMARY_NAMES.get(experiment, experiment), "status": "complete"}
            if experiment in ("controlled", "temporal"):
                obj.update(completed_outer_count=4, expected_outer_count=4, failures=[],
                           protocol={"full_protocol_guard": {"matches_registered_full": True}})
        elif relative.endswith("selection_manifest.json"):
            obj = {"selected_model": "naive", "reg_params": {"num_leaves": 31}, "theta_core_q95": 187.}
        elif relative.endswith("calibration_manifest.json"):
            obj = [{"outer": "O2", "model": "lgb", "theta": 187., "tau": 174.25}]
        elif relative.endswith("checkpoint.json") or relative.endswith("manifest.json"):
            obj = {"status": "complete"}
        elif relative == "sensitivity/scenario_protocol.json":
            obj = {"scenarios": config["sensitivity"]["scenarios"], "history": "fixture history policy", "shift_boundary": "fixture within-day policy"}
        elif relative == "policy/policy_contract.json":
            obj = {"configuration": config["policy"], "policies": ["none", "fixed_L1", "alert_L1", "alert_L2"],
                   "action_input_columns": ["datetime", "origin", "pred_peak", "tau", "planned_production", "plan_known", "is_operating"]}
        elif relative.endswith("protocol.json") or relative.endswith("contract.json"):
            obj = {"config": copy.deepcopy(config), "features": ["prod", "headcount"], "quality_rule": "fixture-quality-rule"}
        else:
            obj = []
        write_json(reference / relative, obj)
    for experiment in c.EXPERIMENTS:
        resign(reference, experiment)
    current = tmp_path / "current"
    shutil.copytree(reference, current)
    return reference, current


def edit_csv(root, relative, change):
    path = root / relative
    frame = pd.read_csv(path)
    frame = change(frame)
    frame.to_csv(path, index=False)
    resign(root, relative.split("/")[0])


def test_identical_complete_artifacts_pass_and_preserve_provenance(runs):
    reference, current = runs
    result = c.compare_runs(reference, current)
    assert result["status"] == "passed", result.get("input_validation")
    assert result["artifact_count"] == len(c.CSV_KEYS)+len(c.JSON_FILES)
    assert set(result["provenance"]) == {"reference", "current"}
    assert "does not prove" in result["execution_claim"]


def test_different_runtime_timing_and_cache_pass(runs):
    reference, current = runs
    edit_csv(current, "controlled/controlled_resources.csv", lambda f: f.assign(fit_seconds=[900., 1500.], cache_hit=True, job_fingerprint="Linux-job"))
    manifest_path = current / "controlled/executor_manifest.json"
    manifest = c.read_json(manifest_path)
    manifest["inputs"]["runtime"] = {"python": "3.11.9", "platform": "fixture-Linux"}
    manifest["elapsed_seconds"] = 99999.
    write_json(manifest_path, manifest)
    resign(current, "controlled")
    result = c.compare_runs(reference, current)
    assert result["status"] == "passed", result
    recorded = result["provenance"]["current"]["experiments"]["controlled"]["executor_manifest"]
    assert recorded["inputs"]["runtime"]["platform"] == "fixture-Linux"


def test_changed_prediction_fails_beyond_fixed_tolerance(runs):
    reference, current = runs
    edit_csv(current, "temporal/temporal_outer_predictions.csv", lambda f: f.assign(pred_avg=f.pred_avg+.01))
    result = c.compare_runs(reference, current)
    assert result["status"] == "failed"
    assert any("pred_avg" in d["path"] and "tolerance" in d["reason"] for d in result["differences"])


def test_tiny_prediction_difference_passes_but_truth_difference_is_exact(runs):
    reference, current = runs
    edit_csv(current, "temporal/temporal_outer_predictions.csv", lambda f: f.assign(pred_avg=f.pred_avg+1e-7))
    assert c.compare_runs(reference, current)["status"] == "passed"
    edit_csv(current, "temporal/temporal_outer_predictions.csv", lambda f: f.assign(y_avg=f.y_avg+1e-7))
    result = c.compare_runs(reference, current)
    assert result["status"] == "failed"
    assert any("y_avg" in d["path"] and "Exact" in d["reason"] for d in result["differences"])


def test_selected_model_change_fails(runs):
    reference, current = runs
    path = current / "temporal/O2/selection_manifest.json"
    obj = c.read_json(path)
    obj["selected_model"] = "rf"
    write_json(path, obj)
    resign(current, "temporal")
    result = c.compare_runs(reference, current)
    assert result["status"] == "failed"
    assert any(d["path"].endswith("selected_model") for d in result["differences"])


def test_row_order_is_harmless(runs):
    reference, current = runs
    for label in c.CSV_KEYS:
        edit_csv(current, label, lambda f: f.iloc[::-1].reset_index(drop=True))
    assert c.compare_runs(reference, current)["status"] == "passed"


def test_duplicate_key_is_rejected_before_comparison(runs):
    reference, current = runs
    edit_csv(current, "policy/policy_actions.csv", lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True))
    result = c.compare_runs(reference, current)
    assert result["status"] == "failed"
    assert "Duplicate row key" in result["input_validation"]["current"]["reason"]
    assert not result["artifacts"]


def test_partial_nested_artifact_is_rejected_even_when_resigned(runs):
    reference, current = runs
    write_json(current / "temporal/O2/checkpoint.json", {"status": "running"})
    resign(current, "temporal")
    result = c.compare_runs(reference, current)
    assert "Incomplete nested artifact" in result["input_validation"]["current"]["reason"]


def test_missing_and_corrupt_artifacts_are_rejected(runs):
    reference, current = runs
    (current / "sensitivity/scenario_protocol.json").unlink()
    result = c.compare_runs(reference, current)
    assert "hash mismatch" in result["input_validation"]["current"]["reason"]


def test_changed_nan_pattern_fails(runs):
    reference, current = runs
    edit_csv(current, "temporal/temporal_outer_predictions.csv", lambda f: f.assign(pred_avg=[float("nan"), 102.25]))
    result = c.compare_runs(reference, current)
    assert any(d["reason"] == "Missing/NaN pattern differs" for d in result["differences"])


def test_unknown_column_and_protocol_schema_rejected(runs):
    reference, current = runs
    edit_csv(current, "policy/policy_actions.csv", lambda f: f.assign(unregistered_metric=1.))
    result = c.compare_runs(reference, current)
    assert "Unsupported CSV column" in result["input_validation"]["current"]["reason"]
    config = c.read_json(reference / "protocol.json")
    config["schema_version"] = 2
    write_json(reference / "protocol.json", config)
    assert "Unsupported supplemental" in c.compare_runs(reference, current)["input_validation"]["reference"]["reason"]


def test_changed_label_and_quality_mask_are_exact(runs):
    reference, current = runs
    edit_csv(current, "temporal/temporal_outer_predictions.csv", lambda f: f.assign(evaluation_usable=False, peak_pred_label=1))
    result = c.compare_runs(reference, current)
    paths = [d["path"] for d in result["differences"]]
    assert any(p.endswith("evaluation_usable") for p in paths)
    assert any(p.endswith("peak_pred_label") for p in paths)


def test_cli_reports_nonzero_and_writes_failure_json(runs, tmp_path):
    reference, current = runs
    status = c.read_json(current / "run_status.json")
    status["status"] = "partial"
    write_json(current / "run_status.json", status)
    output = tmp_path / "comparison.json"
    assert c.main(["--reference", str(reference), "--current", str(current), "--output", str(output)]) == 1
    assert c.read_json(output)["status"] == "failed"


def test_actual_completed_e1_e4_schema_read_only(project_root):
    """Use current stored real columns when available; never synthesize E5 training."""
    folder = project_root / "outputs/additional/validated_run"
    present = [p for p in c.CSV_KEYS if p.split("/")[0] in {"bootstrap", "diagnostics", "sensitivity", "controlled"}
               and (folder / p).is_file()]
    if not present:
        pytest.skip("Optional existing E1-E4 outputs are absent in this checkout")
    for label in present:
        table, roles = c.read_table(folder / label, label)
        diff, result = c.compare_table(table, table.copy(), label, roles)
        assert not diff and result["status"] == "passed"


def test_actual_e6_pure_writer_columns_are_supported():
    import numpy as np
    from experiments import policy as p
    index = pd.date_range("2021-07-07", periods=48, freq="h")
    plans = pd.DataFrame({"datetime": index, "origin": index.normalize(), "pred_peak": 170., "tau": 160.,
                          "planned_production": 100., "plan_known": True, "is_operating": True})
    actions, transfers = p.select_actions(plans, "alert_L2", 150., p.DEFAULTS)
    audits = p.audit_constraints(actions, transfers, 150.)
    outcomes, metrics = p.evaluate_policy(actions, np.full(48, 175.), 180., {"beta_kw_per_unit": .1}, .2)
    costs = pd.DataFrame([p.evaluate_cost(metrics, p.cost_scenarios(p.DEFAULTS)[0])])
    for frame in (actions, transfers, audits, outcomes, costs):
        frame["outer"] = "O2"
        frame["prediction_sha256"] = "fixture-hash"
        for column in frame:
            c.column_role(column)


def test_actual_e5_writer_columns_supported_with_mocked_fit(tmp_path, monkeypatch):
    import numpy as np
    from experiments import model_adapter as a, temporal_selection as temporal
    index = pd.date_range("2021-06-01", "2021-07-20 23:00", freq="h")
    frame = pd.DataFrame(0., index=index, columns=a.REGISTERED_FEATURES)
    frame["y_avg"], frame["y_peak"], frame["prod"], frame["headcount"] = 100., 150., 10., 1.
    for mask in a.MASKS:
        frame[mask] = False
    def fake_job(root, cache, model, Xtr, ytr, Xpred, config, params=None):
        prediction = {"prob": np.full(len(Xpred), .2)} if model == "clf" else {
            "pred_avg": np.full(len(Xpred), 101.), "pred_peak": np.full(len(Xpred), 149.)}
        return prediction, {"fit_seconds": 1., "cache_hit": False, "fingerprint": "fixture-job"}
    monkeypatch.setattr(temporal, "predict_job", fake_job)
    prediction, metrics, resources, _, audits = temporal.evaluate_outer(
        tmp_path, tmp_path, frame, a.REGISTERED_FEATURES, a.partition("2021-07-07"),
        {"selected_model": "naive", "reg_params": {}, "clf_params": {}}, temporal.DEFAULTS, "O2")
    classifier = temporal.classifier_metrics(prediction)
    for table in (prediction, pd.DataFrame(metrics), pd.DataFrame(resources), pd.DataFrame(audits), classifier):
        for column in table:
            c.column_role(column)
