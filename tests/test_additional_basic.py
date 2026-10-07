"""Small independent fixtures for E1/E2/E3; no training or source-stage imports."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.bootstrap import moving_day_indices, paired_block_bootstrap
from experiments.common import confusion_metrics, prediction_metrics, validate_times
from experiments.diagnostics import build_diagnostics, cluster_rate_intervals
from experiments.sensitivity import assert_baseline, assert_eval_bundle, default_scenarios, perturb_plan, validate_scenarios


def diagnostic_fixture():
    return pd.DataFrame({"datetime": pd.date_range("2021-07-01", periods=8, freq="12h"),
                         "y_cls": [1, 1, 0, 0, 1, 0, 0, 0], "peak_pred_label": [1, 0, 1, 0, 1, 0, 0, 0],
                         "y_avg": [1.] * 8, "pred_avg": [2.] * 8, "y_peak": [3.] * 8, "pred_peak": [4.] * 8,
                         "prod": [0, 10, 20, 0, 10, 20, 0, 30], "temp": [20, 21, 22, 23, 24, 25, 26, 27],
                         "is_shutdown": [0] * 8, "is_first_day_back": [0] * 8})


def raw_plan():
    return pd.DataFrame({"날짜": [20210901] * 24, "시간": np.arange(24), "생산량": np.arange(24, dtype=float),
                         "공장인원": np.r_[0., np.full(23, .7)], "기온": np.full(24, 23.), "풍속": np.ones(24),
                         "습도": np.full(24, 60.), "강수량": np.zeros(24)})


def test_confusion_denominators_and_undefined_rates():
    got = confusion_metrics([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert (got["TP"], got["FN"], got["FP"], got["TN"]) == (1, 1, 1, 2)
    assert got["miss_rate"] == .5
    assert got["false_alarm_rate"] == 1 / 3  # Not 1 - precision.
    assert np.isnan(confusion_metrics([0, 0], [0, 0])["recall"])
    assert np.isnan(confusion_metrics([1, 1], [0, 0])["false_alarm_rate"])
    assert np.isnan(confusion_metrics([0, 0], [0, 0])["precision"])
    with pytest.raises(ValueError):
        confusion_metrics([1, np.nan], [0, 1])


@pytest.mark.parametrize("values", [["2021-01-02", "2021-01-01"], ["2021-01-01", "2021-01-01"], [None]])
def test_reject_corrupted_timestamp_order(values):
    with pytest.raises(ValueError):
        validate_times(values)


def test_cluster_bootstrap_keeps_whole_dates_and_reports_na():
    frame = diagnostic_fixture()
    # A single two-row date has one hit and one miss. Every defined replicate
    # must therefore have recall=.5; row-wise resampling would create 0 and 1.
    result = cluster_rate_intervals(frame, np.arange(8) < 2, n_boot=300, seed=42)
    assert result["recall_ci_low"] == result["recall_ci_high"] == .5
    assert 0 < result["recall_na_fraction"] < 1
    empty = cluster_rate_intervals(frame, np.zeros(8, bool), n_boot=20)
    assert empty["recall_valid_replicates"] == 0
    assert empty["recall_na_fraction"] == 1


def test_diagnostic_conditions_overlap_and_reconcile():
    frame = diagnostic_fixture()
    conditions, production, temperature, meta = build_diagnostics(frame, {"diagnostics": {"n_boot": 20}})
    assert meta["overlapping_conditions"] is True
    assert (conditions.TP + conditions.FN == conditions.P).all()
    assert (conditions.FP + conditions.TN == conditions.N).all()
    assert (conditions.P + conditions.N == conditions.n_hours).all()
    assert conditions.n_hours.sum() > len(frame)
    assert production.n_hours.sum() == temperature.n_hours.sum() == len(frame)
    assert np.isnan(conditions.set_index("condition").loc["shutdown", "miss_rate"])


def test_moving_blocks_non_circular_with_exact_length():
    days = pd.date_range("2021-09-01", periods=14, freq="D")
    for size in [1, 2, 3]:
        ids = moving_day_indices(days, size, np.random.default_rng(42))
        assert len(ids) == 14 and ids.min() >= 0 and ids.max() < 14
        for start in range(0, 14, size):
            assert (np.diff(ids[start:start + size]) == 1).all()
    with pytest.raises(ValueError):
        moving_day_indices(pd.DatetimeIndex(["2021-01-01", "2021-01-03"]), 2, np.random.default_rng(42))


def test_paired_block_bootstrap_direction_and_degenerate_identity():
    idx = pd.date_range("2021-09-01", periods=96, freq="h")
    truth = np.zeros(96)
    same = paired_block_bootstrap(idx, truth, truth, truth, n_boot=30)
    assert same["diff"] == same["ci_low"] == same["ci_high"] == 0
    better = paired_block_bootstrap(idx, truth, np.full(96, 3.), np.ones(96), block_days=3, n_boot=30)
    assert better["diff"] == better["ci_low"] == better["ci_high"] == 2


def test_one_day_matches_original_sampling_algorithm():
    idx = pd.date_range("2021-09-01", periods=120, freq="h")
    y = np.arange(120) % 7
    a, b = y + np.arange(120) / 100, y - np.arange(120) / 300
    got = paired_block_bootstrap(idx, y, a, b, n_boot=100, seed=42)
    rng = np.random.default_rng(42)
    blocks = [np.where(idx.normalize() == d)[0] for d in idx.normalize().unique()]
    stats = []
    for _ in range(100):
        select = np.concatenate([blocks[j] for j in rng.integers(0, 5, 5)])
        stats.append(np.abs(y[select] - a[select]).mean() - np.abs(y[select] - b[select]).mean())
    np.testing.assert_array_equal([got["ci_low"], got["ci_high"]], np.percentile(stats, [2.5, 97.5]))


def test_every_preregistered_scenario_and_zero_preservation():
    plan = raw_plan()
    original = plan.copy(deep=True)
    scenarios = default_scenarios()
    validate_scenarios(scenarios)
    assert sum(not s.get("supplementary", False) for s in scenarios) == 15
    for scenario in scenarios:
        changed = perturb_plan(plan, scenario)
        assert len(changed) == 24
        np.testing.assert_array_equal(changed["시간"], plan["시간"])
        assert changed["날짜"].nunique() == 1
        assert changed["생산량"].min() >= 0
        if not scenario.get("shift_hours"):
            np.testing.assert_array_equal(changed["생산량"] == 0, plan["생산량"] == 0)
        np.testing.assert_array_equal(changed["공장인원"] == 0, plan["공장인원"] == 0)
        assert np.isclose(changed["생산량"].sum(), plan["생산량"].sum() * scenario.get("production_factor", 1))
    pd.testing.assert_frame_equal(plan, original)
    with pytest.raises(ValueError):
        validate_scenarios(scenarios[:-1])


def test_shift_boundaries_do_not_wrap_or_discard():
    plan = raw_plan()
    plan["생산량"] = 0.
    plan.loc[0, "생산량"] = 2.
    plan.loc[23, "생산량"] = 3.
    earlier = perturb_plan(plan, {"shift_hours": -1})
    later = perturb_plan(plan, {"shift_hours": 1})
    assert earlier.loc[0, "생산량"] == 2 and earlier.loc[22, "생산량"] == 3
    assert later.loc[1, "생산량"] == 2 and later.loc[23, "생산량"] == 3
    assert earlier["생산량"].sum() == later["생산량"].sum() == 5


def test_eval_rejects_deploy_and_c0_rejects_drift():
    with pytest.raises(ValueError):
        assert_eval_bundle(SimpleNamespace(role="deploy"))
    frame = diagnostic_fixture()
    assert_baseline(frame, frame)
    changed = frame.copy()
    changed.loc[0, "pred_avg"] += .01
    with pytest.raises(ValueError, match="C0"):
        assert_baseline(frame, changed)


def test_metrics_peak_error_mask_uses_actual_peak():
    frame = diagnostic_fixture()
    frame.loc[frame.y_cls == 0, "pred_avg"] = 100.
    frame.loc[frame.y_cls == 0, "pred_peak"] = 103.
    result = prediction_metrics(frame)
    assert result["peak_mae"] == 1.
    assert result["mae"] > 1.
    assert result["peak15_mae"] == 1.
    assert result["peak15_mae_all"] == (3 * 1. + 5 * 100.) / 8


def test_metrics_without_actual_peaks_report_na_for_peak_only_metrics():
    frame = diagnostic_fixture()
    frame["y_cls"] = 0
    result = prediction_metrics(frame)
    assert np.isnan(result["peak_mae"])
    assert np.isnan(result["peak15_mae"])
    assert result["peak15_mae_all"] == 1.


def test_raw_plan_recomputes_features_without_changing_power_history():
    from serving.pipeline import build_day_ahead_features
    plan = raw_plan()
    history_days = []
    for day in pd.date_range("2021-08-20", "2021-08-31"):
        part = raw_plan()
        part["날짜"] = int(day.strftime("%Y%m%d"))
        for power in ["15분", "30분", "45분", "60분", "평균"]:
            part[power] = 50. + np.arange(24)
        part["인건비"] = 1.
        history_days.append(part)
    hist = pd.concat(history_days, ignore_index=True)
    original_history = hist.copy(deep=True)
    cols = ["y_avg_lag24", "roll24_avg_mean", "prod", "prod_cum_day", "prod_chg_ratio",
            "headcount", "temp", "cdd", "hdd", "is_shutdown"]
    base, _ = build_day_ahead_features(hist, plan, columns=cols, theta=187, holidays=[], min_days=8)
    for scenario in default_scenarios():
        changed = perturb_plan(plan, scenario)
        x, _ = build_day_ahead_features(hist, changed, columns=cols, theta=187, holidays=[], min_days=8)
        np.testing.assert_array_equal(x["y_avg_lag24"], base["y_avg_lag24"])
        np.testing.assert_array_equal(x["roll24_avg_mean"], base["roll24_avg_mean"])
        np.testing.assert_allclose(x["prod_cum_day"], np.cumsum(changed["생산량"]))
        np.testing.assert_allclose(x["cdd"], np.maximum(changed["기온"] - 24, 0))
        np.testing.assert_allclose(x["hdd"], np.maximum(18 - changed["기온"], 0))
        np.testing.assert_allclose(x["headcount"], changed["공장인원"])
        assert len(x) == 24
    pd.testing.assert_frame_equal(hist, original_history)
