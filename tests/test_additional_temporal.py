"""Boundary tests use small numeric fixtures and fake fit jobs, never full training."""
from __future__ import annotations

import copy
import sys

import numpy as np
import pandas as pd
import pytest

from experiments import model_adapter as a
from experiments import temporal_selection as t


@pytest.fixture
def hourly():
    idx = pd.date_range("2021-01-01", "2021-08-31 23:00", freq="h")
    values = 100 + 75 * np.sin(np.arange(len(idx)) / 9)
    frame = pd.DataFrame({"y_avg": values, "y_peak": values + 20., "prod": values * 3,
                          "headcount": 10., "y_avg_lag168": 90., "y_peak_lag168": 110.,
                          "roll24_peak_cnt": 0., "roll168_peak_cnt": 0.,
                          "is_warmup": False, "is_outage": False, "is_erp_missing": False}, index=idx)
    frame.loc[idx[:168], "is_warmup"] = True
    return frame


FEATURES = ["prod", "headcount", "y_avg_lag168", "y_peak_lag168", "roll24_peak_cnt", "roll168_peak_cnt"]


def test_o2_partition_exact_boundary_and_all_outer_gaps():
    spec = a.partition("2021-07-07")
    assert spec["core_train_end"] == pd.Timestamp("2021-06-20 23:00")
    assert spec["cal_start"] == pd.Timestamp("2021-06-22")
    assert spec["cal_end"] == pd.Timestamp("2021-07-05 23:00")
    for start in a.OUTER_STARTS:
        s = a.partition(start)
        assert s["cal_start"] - s["core_end_exclusive"] == pd.Timedelta(hours=24)
        assert s["eval_start"] - s["cal_end_exclusive"] == pd.Timedelta(hours=24)
        assert s["eval_end_exclusive"] - s["eval_start"] == pd.Timedelta(days=14)


def test_theta_depends_only_on_core_and_future_power_blind(hourly):
    spec = a.partition("2021-07-07")
    original, masks, theta = a.prepare_split(hourly, FEATURES, spec)
    modified = hourly.copy()
    modified.loc[modified.index >= spec["core_end_exclusive"], "y_peak"] = 99999.
    changed, _, new_theta = a.prepare_split(modified, FEATURES, spec)
    assert theta == new_theta
    pd.testing.assert_frame_equal(original.loc[masks["core"], FEATURES], changed.loc[masks["core"], FEATURES])
    target_day = pd.Timestamp("2021-07-10")
    modified = hourly.copy()
    modified.loc[modified.index >= target_day, "y_peak"] = 88888.
    changed, _, _ = a.prepare_split(modified, FEATURES, spec)
    pd.testing.assert_frame_equal(original.loc[target_day:target_day+pd.Timedelta(hours=23), FEATURES],
                                  changed.loc[target_day:target_day+pd.Timedelta(hours=23), FEATURES])


def test_masks_exclude_bad_rows_but_gap_history_is_retained(hourly):
    spec = a.partition("2021-07-07")
    hourly.loc["2021-06-20 20:00", "is_erp_missing"] = True
    rebuilt, masks, _ = a.prepare_split(hourly, FEATURES, spec)
    assert not masks["core"].loc["2021-06-20 20:00"]
    for role in masks:
        assert not masks[role].loc["2021-06-21"].any()
        assert not masks[role].loc["2021-07-06"].any()
    assert len(rebuilt) == len(hourly)


def test_inner_partitions_ending_before_outer_core(hourly):
    outer = a.partition("2021-07-07")
    core = hourly.loc[hourly.index < outer["core_end_exclusive"]]
    splits = t.inner_partitions(core, outer, t.DEFAULTS)
    assert len(splits) == 3
    assert max(s["eval_end_exclusive"] for s in splits) == outer["core_end_exclusive"]
    assert all(s["core_train_end"] < s["cal_start"] < s["cal_end_exclusive"] < s["eval_start"] for s in splits)


def test_calibration_sparse_classes_use_declared_fallback():
    meta, iso = a.calibrate([100]*30+[200]*4, np.arange(34), 150, np.linspace(0, 1, 34))
    assert meta["tau"] == 150 and iso is None
    assert meta["fallback"] == "insufficient_calibration_classes"
    assert meta["probability_calibration"] == "raw_fallback"


def test_calibration_uses_fixed_theta_and_deterministic_threshold():
    y = np.array([100]*25+[200]*8)
    pred = np.array([90]*25+[180]*8)
    meta, iso = a.calibrate(y, pred, 150, np.linspace(0, 1, len(y)))
    assert iso is not None and meta["fallback"] is None
    assert meta["theta"] == 150
    assert meta["positives"] == 8 and meta["negatives"] == 25
    assert (pred >= meta["tau"]).sum() == 8


def test_score_is_tied_by_registered_order_and_not_runtime():
    rows = [{"model": model, "n": 24, "MAE": 10, "Peak-MAE": 12,
             "TP": 2, "FN": 1, "fit_seconds": i*100} for i, model in enumerate(a.CANDIDATES)]
    chosen, scores = t.score_candidates(rows, t.DEFAULTS)
    assert chosen == "naive"
    assert scores.score.nunique() == 1
    assert scores.feasible.eq(1).all()
    for row in rows:
        row["fit_seconds"] = 99999.
    new_choice, new_scores = t.score_candidates(rows, t.DEFAULTS)
    assert new_choice == chosen
    pd.testing.assert_frame_equal(scores, new_scores)


def test_no_peaks_score_remains_finite():
    rows = [{"model": model, "n": 24, "MAE": 10+i, "Peak-MAE": np.nan,
             "TP": 0, "FN": 0} for i, model in enumerate(a.CANDIDATES)]
    _, scores = t.score_candidates(rows, t.DEFAULTS)
    assert np.isfinite(scores.score).all()
    assert scores.peak_mae_normalized.eq(0).all()


def test_selection_rejects_noncore_rows(hourly, tmp_path):
    with pytest.raises(ValueError, match="outside core"):
        t.select_on_core(tmp_path, tmp_path, hourly, FEATURES, a.partition("2021-07-07"), t.DEFAULTS, tmp_path)


def test_outer_targets_and_calibration_targets_do_not_enter_selection(hourly, tmp_path, monkeypatch):
    def fake_optimization(root, cache, inner, features, config, output, kind):
        return {"num_leaves": 31, "bagging_freq": 1}, [{"trial": 0}]
    def fake_prediction(root, cache, model, item, features, config, params=None):
        d, masks, _, _ = item
        infer = pd.concat([d.loc[masks["calibration"]], d.loc[masks["evaluation"]]])
        assert infer.index.max() < pd.Timestamp("2021-06-21")
        return {"pred_avg": np.repeat(100., len(infer)), "pred_peak": np.repeat(150., len(infer))}, {
            "fit_seconds": 1, "cache_hit": False, "fingerprint": "fake"}
    monkeypatch.setattr(t, "optimize", fake_optimization)
    monkeypatch.setattr(t, "_inner_prediction", fake_prediction)
    spec = a.partition("2021-07-07")
    first = hourly.loc[hourly.index < spec["core_end_exclusive"]].copy()
    result = t.select_on_core(tmp_path, tmp_path, first, FEATURES, spec, t.DEFAULTS, tmp_path)
    changed = hourly.copy()
    changed.loc[changed.index >= spec["core_end_exclusive"], ["y_avg", "y_peak"]] = -999.
    second = changed.loc[changed.index < spec["core_end_exclusive"]].copy()
    result2 = t.select_on_core(tmp_path, tmp_path, second, FEATURES, spec, t.DEFAULTS, tmp_path)
    assert a.json_bytes(result) == a.json_bytes(result2)
    assert "prob" not in result


def test_outer_full_clock_keeps_unknown_erp_and_never_fits_calibration(hourly, tmp_path, monkeypatch):
    calls = []
    def fake_fit(root, cache, model, Xtr, ytr, Xpred, config, params=None):
        calls.append((Xtr.copy(), ytr.copy()))
        assert Xtr.index.max() <= pd.Timestamp("2021-06-20 23:00")
        assert list(ytr.columns) == ["y_avg", "y_peak", "y_cls"]
        pred = {"prob": np.full(len(Xpred), .2)} if model == "clf" else {
            "pred_avg": np.full(len(Xpred), 100.), "pred_peak": np.full(len(Xpred), 150.)}
        return pred, {"fit_seconds": 1., "cache_hit": False, "fingerprint": "fake"}
    monkeypatch.setattr(t, "predict_job", fake_fit)
    hourly.loc["2021-07-08", "is_erp_missing"] = True
    choice = {"selected_model": "naive", "reg_params": {}, "clf_params": {}}
    result = t.evaluate_outer(tmp_path, tmp_path, hourly, FEATURES, a.partition("2021-07-07"), choice, t.DEFAULTS, "O2")
    rows = result[0]
    selected = rows[rows.selected]
    assert len(selected) == 336
    assert selected.plan_known.sum() == 312
    assert selected.evaluation_usable.sum() == 312
    assert all(row["n"] == 312 for row in result[1])
    assert len(calls) == 6
    reference = calls[0]
    for training in calls[1:]:
        pd.testing.assert_frame_equal(reference[0], training[0])
        pd.testing.assert_frame_equal(reference[1], training[1])


def test_factory_extraction_does_not_execute_source_top_level(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    definitions = "\n".join(f"def {name}(*args, **kwargs):\n    return None\n" for name in
                            ("make_lgb_regressor", "make_regime_model", "make_peak_classifier", "rf_fit", "naive_fit"))
    (src / "s05_models.py").write_text("raise RuntimeError('top level must not run')\n" + definitions, encoding="utf-8")
    before = set(sys.modules)
    namespace = a.factory_namespace(tmp_path, {"seed": 42})
    assert callable(namespace["make_regime_model"])
    assert not any(n.startswith("s0") for n in set(sys.modules)-before)


def test_checkpoint_rejects_partial_corrupt_and_wrong_hash(tmp_path):
    np.savez_compressed(tmp_path / "result.npz", pred_avg=np.array([1., 2.]))
    meta = {"status": "complete", "fingerprint": "right", "payload_sha256": a.sha(tmp_path / "result.npz")}
    a.atomic_json(tmp_path / "result.json", meta)
    assert a._cache_read(tmp_path, "right", 2, ["pred_avg"]) is not None
    assert a._cache_read(tmp_path, "wrong", 2, ["pred_avg"]) is None
    assert a._cache_read(tmp_path, "right", 3, ["pred_avg"]) is None
    meta["status"] = "running"
    a.atomic_json(tmp_path / "result.json", meta)
    assert a._cache_read(tmp_path, "right", 2, ["pred_avg"]) is None
    meta["status"] = "complete"
    a.atomic_json(tmp_path / "result.json", meta)
    (tmp_path / "result.npz").write_bytes(b"not a zip")
    assert a._cache_read(tmp_path, "right", 2, ["pred_avg"]) is None


def test_profile_audit_is_post_scoring_only(hourly):
    rows = a.profile_audit(hourly, a.partition("2021-07-07"))
    assert len(rows) == 14
    assert all(r["role"] == "post_scoring_novelty_audit_only" for r in rows)


def test_controlled_paired_ci_identity_is_exact_zero():
    from experiments.controlled_regime import paired_intervals
    dates = pd.date_range("2021-07-07", periods=48, freq="h")
    base = pd.DataFrame({"datetime": dates, "outer": "O2", "y_avg": np.arange(48), "y_peak": np.arange(48)+5,
                         "pred_avg": np.arange(48)+1, "pred_peak": np.arange(48)+6})
    rows = pd.concat([base.assign(model="lgb"), base.assign(model="regime3")], ignore_index=True)
    result = paired_intervals(rows, repetitions=20)
    assert result[["MAE_single_minus_regime", "ci_low", "ci_high"]].eq(0).all().all()


def test_small_real_isolated_job_then_verified_cache(tmp_path, project_root):
    idx = pd.date_range("2021-01-01", periods=96, freq="h")
    x = pd.DataFrame({"prod": np.arange(96, dtype=float), "hour": idx.hour.astype(float)}, index=idx)
    y = pd.DataFrame({"y_avg": x["prod"]+1, "y_peak": x["prod"]+5,
                      "y_cls": (x["prod"] > 70).astype(int)}, index=idx)
    config = {"profile": "unit_fixture", "n_estimators": 8, "threads": 1, "seed": 42, "retries": 0}
    first, info = a.predict_job(project_root, tmp_path, "lgb", x.iloc[:72], y.iloc[:72], x.iloc[72:], config)
    assert info["status"] == "complete" and not info["cache_hit"]
    assert info["peak_memory_bytes"] is None or info["peak_memory_bytes"] > 0
    second, cached = a.predict_job(project_root, tmp_path, "lgb", x.iloc[:72], y.iloc[:72], x.iloc[72:], config)
    assert cached["cache_hit"] and info["fingerprint"] == cached["fingerprint"]
    np.testing.assert_array_equal(first["pred_avg"], second["pred_avg"])
    np.testing.assert_array_equal(first["pred_peak"], second["pred_peak"])


@pytest.mark.parametrize("experiment,key,value", [
    ("controlled", "seed", 7), ("controlled", "threads", 1),
    ("controlled", "calibration_days", 7), ("controlled", "gap_hours", 0),
    ("controlled", "outer_days", 7), ("controlled", "n_boot", 20),
    ("controlled", "n_estimators", 8), ("controlled", "calibration_min_positive", 1),
    ("temporal", "trials_reg", 1), ("temporal", "trials_clf", 1),
    ("temporal", "max_inner_folds", 1), ("temporal", "min_core_days", 7),
    ("temporal", "calibration_min_negative", 5), ("temporal", "calibration_min_positive", 1),
    ("temporal", "search_space", {"num_leaves": [15, 15]}),
    ("temporal", "outer_starts", ["2021-07-07"]),
    ("temporal", "weights", {"mae": 1.}),
    ("temporal", "lgb_base", {"objective": "l2"}),
])
def test_full_protocol_guard_rejects_scientific_overrides(experiment, key, value):
    from experiments.controlled_regime import DEFAULTS as E4_DEFAULTS
    registered = E4_DEFAULTS if experiment == "controlled" else t.DEFAULTS
    normal = a.protocol_guard(registered, registered, a.REGISTERED_FEATURES)
    assert normal["matches_registered_full"]
    changed = copy.deepcopy(registered)
    changed[key] = value
    guard = a.protocol_guard(changed, registered, a.REGISTERED_FEATURES)
    assert not guard["matches_registered_full"]
    assert key in [item["field"] for item in guard["deviations"]]
    assert guard["registered_scientific_sha256"] == normal["registered_scientific_sha256"]


def test_full_guard_checks_feature_identity_and_order_and_unknown_fields():
    features = list(a.REGISTERED_FEATURES)
    features[0], features[1] = features[1], features[0]
    assert not a.protocol_guard(t.DEFAULTS, t.DEFAULTS, features)["matches_registered_full"]
    assert not a.protocol_guard({**t.DEFAULTS, "skip_inner": True}, t.DEFAULTS, a.REGISTERED_FEATURES)["matches_registered_full"]
    assert a.protocol_guard({**t.DEFAULTS, "retries": 0, "timeout_seconds": 1}, t.DEFAULTS, a.REGISTERED_FEATURES)["matches_registered_full"]


def test_changed_configuration_cannot_reuse_verified_fit_cache(tmp_path, project_root, monkeypatch):
    idx = pd.date_range("2021-01-01", periods=40, freq="h")
    x = pd.DataFrame({"prod": np.arange(40, dtype=float)}, index=idx)
    y = pd.DataFrame({"y_avg": x["prod"]+1, "y_peak": x["prod"]+5, "y_cls": 0}, index=idx)
    cfg = {"profile": "unit_fixture", "n_estimators": 4, "threads": 1, "seed": 42, "retries": 0}
    _, old = a.predict_job(project_root, tmp_path, "lgb", x.iloc[:30], y.iloc[:30], x.iloc[30:], cfg)
    def refuse_spawn(*args, **kwargs):
        raise RuntimeError("changed configuration requested a fresh worker")
    monkeypatch.setattr(a.subprocess, "Popen", refuse_spawn)
    with pytest.raises(RuntimeError, match="fresh worker"):
        a.predict_job(project_root, tmp_path, "lgb", x.iloc[:30], y.iloc[:30], x.iloc[30:], {**cfg, "seed": 43})
    assert a._cache_read(tmp_path / old["fingerprint"], old["fingerprint"], 10, ("pred_avg", "pred_peak")) is not None


@pytest.mark.parametrize("experiment", ["controlled", "temporal"])
def test_standalone_reduced_run_stays_partial_and_binds_snapshot(hourly, tmp_path, monkeypatch, experiment):
    from experiments import controlled_regime as e4
    module = e4 if experiment == "controlled" else t
    frame = hourly.copy()
    for column in a.REGISTERED_FEATURES:
        if column not in frame:
            frame[column] = 0.
    monkeypatch.setattr(module, "load_data", lambda path: (frame, a.REGISTERED_FEATURES, {"fingerprint": "verified-fixture-snapshot"}))
    def fake_fit(root, cache, model, Xtr, ytr, Xpred, config, params=None):
        prediction = {"prob": np.full(len(Xpred), .2)} if model == "clf" else {
            "pred_avg": np.full(len(Xpred), 100.), "pred_peak": np.full(len(Xpred), 150.)}
        return prediction, {"fit_seconds": 1., "cache_hit": False, "fingerprint": "fake"}
    monkeypatch.setattr(module, "predict_job", fake_fit)
    if experiment == "controlled":
        cfg = {"controlled_regime": {"n_boot": 2}}
    else:
        monkeypatch.setattr(t, "select_on_core", lambda *args: {"selected_model": "naive", "reg_params": {}, "clf_params": {}})
        cfg = {"temporal_selection": {"trials_reg": 2}}
    result = module.run(tmp_path, tmp_path / "snapshot", tmp_path / "out", cfg)
    assert not result["failures"]
    assert result["completed_outer_count"] == 4
    assert result["status"] == "partial"
    protocol = result["protocol"]
    assert protocol["snapshot_fingerprint"] == "verified-fixture-snapshot"
    assert not protocol["full_protocol_guard"]["matches_registered_full"]
