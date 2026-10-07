"""Small, deterministic E6 fixtures; no model training or stage imports."""
from __future__ import annotations

import inspect
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments import policy as module
from experiments.policy import (ACTION_COLUMNS, POLICIES, audit_constraints,
                                cost_scenarios, evaluate_cost, evaluate_policy,
                                fit_response, select_actions)
from experiments.model_adapter import json_bytes
from experiments.temporal_selection import DEFAULTS as TEMPORAL_DEFAULTS


def plan():
    times = pd.date_range("2021-07-07 08:00", periods=9, freq="h")
    return pd.DataFrame({"datetime": times, "origin": pd.Timestamp("2021-07-07"),
                         "pred_peak": [220., 80., 80., 80., 80., 210., 80., 80., 80.],
                         "tau": 190., "planned_production": [100., 50., 50., 50., 50., 100., 50., 50., 50.],
                         "plan_known": True, "is_operating": True})


def training():
    times = pd.date_range("2021-06-01", periods=24 * 25, freq="h")
    production = np.arange(len(times), dtype=float) % 37 + 1.
    peak = production * .3 + times.hour.to_numpy() * 4 + 20.
    return pd.DataFrame({"datetime": times, "planned_production": production,
                         "y_peak": peak, "plan_known": True})


def write_e5_fixture(tmp_path, predictions, *, status="partial"):
    folder = tmp_path / "temporal"
    folder.mkdir(exist_ok=True)
    source = folder / "temporal_outer_predictions.csv"
    predictions.to_csv(source, index=False)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(exist_ok=True)
    (snapshot / "metadata.json").write_text(json.dumps({"fingerprint": "fixture"}), encoding="utf-8")
    protocol = {"experiment": "temporal_selection", "config": TEMPORAL_DEFAULTS,
                "snapshot_fingerprint": "fixture"}
    protocol["fingerprint"] = hashlib.sha256(json_bytes(protocol)).hexdigest()
    (folder / "temporal_protocol.json").write_bytes(json_bytes(protocol))
    manifest = {"experiment": "temporal_selection", "status": status, "protocol": protocol,
                "files": {source.name: module.sha256_file(source)}, "failures": [],
                "completed_outer_count": predictions.outer.nunique(), "expected_outer_count": 4}
    (folder / "manifest.json").write_bytes(json_bytes(manifest))
    return source, snapshot


def test_selector_contract_excludes_actual_outcomes():
    p = plan()
    p["y_peak"] = 9999.
    with pytest.raises(ValueError, match="outcomes forbidden"):
        select_actions(p, "alert_L1", 100., {})
    assert "observed" not in inspect.signature(select_actions).parameters
    assert set(ACTION_COLUMNS) == set(plan().columns)


def test_response_fit_uses_only_past_core_and_matches_hour_ols():
    df = training()
    a = fit_response(df, "2021-06-20 23:00", "2021-07-07", {})
    df.loc[df.datetime > "2021-06-20 23:00", "y_peak"] = 1e9
    df.loc[df.datetime > "2021-06-20 23:00", "planned_production"] = 1e8
    b = fit_response(df, "2021-06-20 23:00", "2021-07-07", {})
    assert a == b
    assert a["n_fit"] == 480
    assert a["beta_kw_per_unit"] == pytest.approx(.3)
    assert pd.Timestamp(a["fit_max_datetime"]) < pd.Timestamp(a["evaluation_start"])
    with pytest.raises(ValueError, match="precede"):
        fit_response(df, "2021-07-07", "2021-07-07", {})


def test_response_fit_excludes_all_three_quality_masks():
    df = training()
    for i, name in enumerate(("is_outage", "is_erp_missing", "is_warmup")):
        df[name] = False
        df.loc[i, name] = True
        df.loc[i, "y_peak"] = 1e8
        df.loc[i, "planned_production"] = 1e8
    masked = fit_response(df, "2021-06-20 23:00", "2021-07-07", {})
    removed = fit_response(df.iloc[3:].copy(), "2021-06-20 23:00", "2021-07-07", {})
    assert masked["n_fit"] == 477
    assert masked["n_core_quality_excluded"] == 3
    assert masked["beta_kw_per_unit"] == pytest.approx(.3)
    assert masked["beta_kw_per_unit"] == removed["beta_kw_per_unit"]
    assert masked["receiving_capacity_units"] == removed["receiving_capacity_units"]


def test_snapshot_loader_uses_feature_warmup_mask_not_unannotated_df(tmp_path, monkeypatch):
    from tools import analysis_snapshot
    df = training().set_index("datetime")
    df["is_outage"] = False
    df["is_erp_missing"] = False
    features = df.copy()
    features["is_warmup"] = features.index < "2021-06-08"
    monkeypatch.setattr(analysis_snapshot, "load_snapshot", lambda path: {"df": df, "feat": features})
    loaded = module._load_training(tmp_path)
    fitted = fit_response(loaded, "2021-06-20 23:00", "2021-07-07", {})
    assert fitted["n_core_quality_excluded"] == 168
    assert fitted["n_fit"] == 480 - 168
    monkeypatch.setattr(analysis_snapshot, "load_snapshot", lambda path: {"df": df})
    with pytest.raises(ValueError, match="three required quality masks"):
        module._load_training(tmp_path)


def test_negative_beta_and_unidentified_slope_are_explicit():
    df = training()
    df["y_peak"] = 1000 - df.planned_production
    result = fit_response(df, "2021-06-20 23:00", "2021-07-07", {})
    assert result["negative_slope_clipped"]
    assert result["beta_kw_per_unit"] == 0
    df["planned_production"] = 100.
    result = fit_response(df, "2021-06-20 23:00", "2021-07-07", {})
    assert result["beta_status"] == "unidentified_zero_fallback"


@pytest.mark.parametrize("name", POLICIES)
def test_policies_conserve_production_and_obey_future_capacity(name):
    data = plan()
    original = data.copy(deep=True)
    actions, transfers = select_actions(data, name, 55., {})
    pd.testing.assert_frame_equal(data, original)
    assert actions.planned_production.sum() == actions.production_after.sum()
    if not transfers.empty:
        assert (transfers.recipient_datetime > transfers.source_datetime).all()
        assert (transfers.recipient_datetime > transfers.origin).all()
        assert (transfers.recipient_datetime.dt.normalize() == transfers.source_datetime.dt.normalize()).all()
    assert audit_constraints(actions, transfers, 55.).passed.all()
    if name == "alert_L2":
        assert actions.shift_requested.sum() == 40
        assert actions.shift_out.sum() == 20
        assert actions.unallocated_units.sum() == 20
        assert actions.loc[actions.shift_in > 0, "production_after"].max() <= 55


def test_none_preserves_observed_peak_and_production_exactly():
    actions, _ = select_actions(plan(), "none", 55., {})
    actual = np.arange(len(actions)) * 10. + 150.
    result, metrics = evaluate_policy(actions, actual, 187., {"beta_kw_per_unit": .3}, .2)
    np.testing.assert_array_equal(result.assumed_peak_kw, actual)
    np.testing.assert_array_equal(result.production_after, result.planned_production)
    assert metrics["assumed_peak_reduction_kw"] == 0
    assert metrics["action_events"] == 0


@pytest.mark.parametrize("bad_field,value", [("plan_known", False), ("planned_production", np.nan),
                                            ("planned_production", -1.), ("pred_peak", np.nan)])
def test_invalid_hour_makes_whole_day_manual_noop(bad_field, value):
    data = plan()
    data.loc[2, bad_field] = value
    actions, transfers = select_actions(data, "alert_L2", 100., {})
    assert not actions.valid_plan_day.any()
    assert not actions.l2_action.any()
    assert actions.shift_out.sum() == 0
    assert len(transfers) == 0
    audit = audit_constraints(actions, transfers, 100.)
    assert audit.passed.all()
    assert audit.manual_confirmation_hours.sum() == len(data)


def test_missing_erp_cannot_be_assumed_idle():
    data = training().drop(columns=["plan_known"])
    with pytest.raises(ValueError, match="plan-known"):
        module._canonical(data)
    data["is_erp_missing"] = True
    result = module._canonical(data)
    assert not result.plan_known.any()


def test_donor_cannot_move_to_past_or_alerted_hours():
    data = plan()
    data["pred_peak"] = 80.
    data.loc[data.datetime.dt.hour == 16, "pred_peak"] = 220.
    actions, transfers = select_actions(data, "alert_L2", 100., {})
    assert len(transfers) == 0
    assert actions.unallocated_units.sum() == 10
    assert actions.shift_out.sum() == 0


def test_actual_truth_changes_metrics_not_decisions():
    data = plan()
    a, _ = select_actions(data, "alert_L1", 100., {})
    _, low = evaluate_policy(a, np.full(len(a), 100.), 187., {"beta_kw_per_unit": .3}, .2)
    _, high = evaluate_policy(a, np.full(len(a), 230.), 187., {"beta_kw_per_unit": .3}, .2)
    b, _ = select_actions(data, "alert_L1", 100., {})
    pd.testing.assert_frame_equal(a, b)
    assert low["false_positive_action_events"] == 2
    assert high["false_positive_action_events"] == 0
    assert high["missed_peak_hours_no_action"] == 7


def test_l2_new_receiver_peak_and_false_action_cost_are_counted():
    actions, _ = select_actions(plan(), "alert_L2", 100., {})
    actual = np.full(len(actions), 180.)
    outcomes, metrics = evaluate_policy(actions, actual, 187., {"beta_kw_per_unit": 1.}, .2)
    assert outcomes.new_peak_event.sum() == 2
    assert metrics["false_positive_action_events"] == 2
    assert metrics["false_positive_shifted_units"] == 40
    s = {"alpha": .2, "assumed_rate_krw_per_kw_month": 8320., "action_cost_krw": 1000., "shift_unit_cost_krw": .1}
    costs = evaluate_cost(metrics, s)
    assert costs["false_positive_action_cost_krw"] == 2004
    assert costs["assumed_execution_cost_krw"] == 2004
    assert costs["assumed_observed_period_net_difference_krw"] < 0
    assert not any("kwh" in c.lower() or "annual" in c.lower() for c in costs)


def test_invalid_targets_do_not_enter_peaks_confusion_or_false_action_costs():
    actions, _ = select_actions(plan(), "alert_L1", 100., {})
    truth = np.full(len(actions), 100.)
    truth[0] = 1e9  # interpolated outage value must not become observed maximum
    truth[1] = np.nan
    truth[2] = -10.
    usable = np.ones(len(actions), dtype=bool)
    usable[0] = False
    outcomes, metrics = evaluate_policy(actions, truth, 187., {"beta_kw_per_unit": .3}, .2,
                                       evaluation_usable=usable)
    assert metrics["n_eval"] == 6
    assert metrics["n_excluded"] == 3
    assert metrics["observation_coverage"] == pytest.approx(6 / 9)
    assert metrics["partial_observation"]
    assert sum(metrics[c] for c in ("TP", "FP", "FN", "TN")) == 6
    assert metrics["observed_peak_kw"] == 100.
    assert metrics["action_events"] == 2
    assert metrics["unknown_target_action_events"] == 1
    assert metrics["false_positive_action_events"] == 1
    assert outcomes.loc[:2, "assumed_peak_kw"].isna().all()
    scenario = {"alpha": .2, "assumed_rate_krw_per_kw_month": 8320., "action_cost_krw": 1000., "shift_unit_cost_krw": .1}
    costs = evaluate_cost(metrics, scenario)
    assert costs["assumed_execution_cost_krw"] == 2000.
    assert costs["false_positive_action_cost_krw"] == 1000.
    assert costs["unknown_target_action_cost_krw"] == 1000.
    assert costs["assumed_observed_period_net_difference_krw"] is None
    assert costs["available_observation_net_proxy_krw"] is not None
    assert costs["cost_completeness"] == "partial_observation_unconfirmed"


def test_all_unobserved_targets_keep_actions_but_no_outcome_claim():
    actions, _ = select_actions(plan(), "alert_L1", 100., {})
    outcomes, metrics = evaluate_policy(actions, np.full(len(actions), np.nan), 187.,
                                       {"beta_kw_per_unit": .3}, .2,
                                       evaluation_usable=np.zeros(len(actions), dtype=bool))
    assert metrics["n_eval"] == 0
    assert metrics["observed_peak_kw"] is None
    assert metrics["action_events"] == 2
    assert outcomes.assumed_peak_kw.isna().all()
    cost = evaluate_cost(metrics, cost_scenarios({})[0])
    assert cost["available_observation_net_proxy_krw"] is None


def test_all_preregistered_cost_conditions_retained():
    scenarios = cost_scenarios({})
    assert len(scenarios) == 17  # 9 alpha/rate + 9 execution-cost cells - common origin
    assert {s["alpha"] for s in scenarios} == {.1, .2, .3}
    assert {s["action_cost_krw"] for s in scenarios} == {0., 1000., 5000.}
    assert len({s["scenario"] for s in scenarios}) == 17


def test_run_artifacts_and_prediction_hash_are_consistent(tmp_path, monkeypatch):
    p = plan()
    p["outer"] = "O2"
    p["core_train_end"] = "2021-06-20 23:00"
    p["theta"] = 187.
    p["y_peak"] = 180.
    p["selected"] = True
    p["evaluation_usable"] = True
    _, snapshot = write_e5_fixture(tmp_path, p)
    monkeypatch.setattr(module, "_load_training", lambda path: training())
    output = tmp_path / "policy"
    result = module.run(tmp_path, snapshot, output, {"policy": {"allow_partial_temporal": True}})
    assert result["status"] == "partial"
    assert result["scenario_count"] == 17
    assert result["outer_count"] == 1
    assert result["constraint_audit_passed"]
    actions = pd.read_csv(output / "policy_actions.csv")
    assert actions.prediction_sha256.nunique() == 1
    assert set(actions.policy) == set(POLICIES)
    contract = json.loads((output / "policy_contract.json").read_text(encoding="utf-8"))
    assert contract["prediction_feedback"] is False
    assert "actual monthly/annual bill saving" in contract["forbidden_claims"]
    costs = pd.read_csv(output / "policy_cost_sensitivity.csv")
    assert len(costs) == 4 * 17
    assert costs.loc[costs.policy == "none", "assumed_observed_period_net_difference_krw"].eq(0).all()


def test_incomplete_e5_fails_instead_of_fabricated_policy(tmp_path):
    with pytest.raises(FileNotFoundError, match="completed E5"):
        module.run(tmp_path, tmp_path / "snapshot", tmp_path / "policy", {})


@pytest.mark.parametrize("problem", ["missing_manifest", "partial", "failed", "stale_csv", "wrong_snapshot", "wrong_protocol", "fake_complete"])
def test_temporal_dependency_rejects_missing_failed_stale_and_partial(tmp_path, problem):
    p = plan()
    p["outer"] = "O2"
    p["selected"] = True
    p["evaluation_usable"] = True
    source, snapshot = write_e5_fixture(tmp_path, p, status="complete" if problem == "fake_complete" else "partial")
    manifest_path = source.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if problem == "missing_manifest":
        manifest_path.unlink()
    elif problem == "failed":
        manifest["status"] = "failed"
        manifest_path.write_bytes(json_bytes(manifest))
    elif problem == "stale_csv":
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif problem == "wrong_snapshot":
        (snapshot / "metadata.json").write_text(json.dumps({"fingerprint": "other"}), encoding="utf-8")
    elif problem == "wrong_protocol":
        protocol_path = source.parent / "temporal_protocol.json"
        content = json.loads(protocol_path.read_text(encoding="utf-8"))
        content["config"]["seed"] = 999
        protocol_path.write_bytes(json_bytes(content))
    with pytest.raises(ValueError):
        module._verify_temporal_source(source, p, snapshot, {})


def test_full_temporal_dependency_requires_all_four_complete_clocks(tmp_path):
    frames = []
    for i, start in enumerate(TEMPORAL_DEFAULTS["outer_starts"], 2):
        frames.append(pd.DataFrame({"datetime": pd.date_range(start, periods=336, freq="h"),
                                    "outer": f"O{i}", "selected": True}))
    selected = pd.concat(frames, ignore_index=True)
    source, snapshot = write_e5_fixture(tmp_path, selected, status="complete")
    assert module._verify_temporal_source(source, selected, snapshot, {})["status"] == "complete"
    with pytest.raises(ValueError, match="336-hour"):
        module._verify_temporal_source(source, selected.iloc[1:], snapshot, {})
