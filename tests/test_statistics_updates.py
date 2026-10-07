"""지표·유의성·평가 계약 회귀검증. 단계 import/전체 모델학습은 실행하지 않는다.

단계 파일의 순수 함수만 AST로 읽어 실제 구현을 실행한다. 산출물과 모델을 쓰지 않는다.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]


def load_functions(filename, names, **extra):
    path = ROOT / "src" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in tree.body} == set(names)
    namespace = {"np": np, "pd": pd, "_st": stats,
                 "average_precision_score": average_precision_score,
                 "precision_recall_fscore_support": precision_recall_fscore_support, **extra}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def metrics():
    return load_functions("s03_split.py", {
        "mae", "rmse", "smape", "peak_mae", "regression_metrics", "paired_bootstrap_mae",
        "clopper_pearson", "classification_metrics", "tune_tau",
    })


def test_peak_metrics_use_same_actual_peak_hours_for_both_targets(metrics):
    result = metrics["regression_metrics"](
        [10, 200, 100], [9, 180, 120], 187,
        y_peak=[190, 220, 186], pred_peak=[180, 215, 200],
    )
    assert result["Peak-MAE"] == 10.5  # includes low mean power with a short peak
    assert result["Peak15-MAE"] == 7.5
    assert result["Peak-n"] == 2
    assert result["High-load-MAE"] == 20
    assert result["High-load-n"] == 1
    assert result["Peak-mask-source"] == "y_peak"


def test_legacy_peak_call_is_explicit_and_empty_peak_set_is_nan(metrics):
    old = metrics["regression_metrics"]([10, 200], [9, 180], 187)
    assert old["Peak-MAE"] == old["High-load-MAE"] == 20
    assert old["Peak-mask-source"] == "y_avg (legacy)"
    assert np.isnan(old["Peak15-MAE"])
    empty = metrics["regression_metrics"]([10, 20], [11, 22], 187,
                                           y_peak=[20, 30], pred_peak=[22, 33])
    assert empty["Peak-n"] == 0
    assert np.isnan(empty["Peak-MAE"]) and np.isnan(empty["Peak15-MAE"])
    with pytest.raises(ValueError):
        metrics["peak_mae"]([10, 20], [9, 18], 187, y_peak=[200])


def test_90pct_exploration_does_not_replace_95pct_primary_analysis(metrics):
    # Fourteen day blocks: evidence crosses 10%, but not the primary 5% threshold.
    differences = np.array([
        -.4817135902, .7750258523, .6133601328, .7561640354, 1.6700002891,
        .7630337683, 1.4004789883, .5446106973, .8547671582, 1.3338035162,
        -1.5900181478, .0024602971, -.2085217160, -.4444289875,
    ])
    result = metrics["paired_bootstrap_mae"](
        np.zeros(14), 5 + differences, np.full(14, 5),
        index=pd.date_range("2021-01-01", periods=14),
    )
    assert result["ci_low"] <= 0 < result["ci90_low"]
    assert not result["유의"] and not result["개선_5pct"]
    assert result["개선_10pct"]
    assert result["p_value"] == pytest.approx(.073)
    assert "근사" in result["p_method"]
    assert result["ci_low"] <= result["ci90_low"] <= result["ci90_high"] <= result["ci_high"]


def test_bootstrap_preserves_day_blocks_and_direction(metrics):
    bootstrap = metrics["paired_bootstrap_mae"]
    days = pd.date_range("2021-01-01", periods=8)
    error_a = np.arange(1.0, 9.0)
    base = bootstrap(np.zeros(8), error_a, np.zeros(8), days, n_boot=200)
    hourly = bootstrap(np.zeros(24 * 8), np.repeat(error_a, 24), np.zeros(24 * 8),
                       pd.date_range("2021-01-01", periods=24 * 8, freq="h"), n_boot=200)
    for key in ("diff", "ci_low", "ci_high", "ci90_low", "ci90_high", "p_value"):
        assert base[key] == pytest.approx(hourly[key])
    worse = bootstrap(np.zeros(8), np.zeros(8), error_a, days, n_boot=200)
    assert worse["유의_5pct"] and not worse["개선_5pct"]
    assert worse["ci_high"] == pytest.approx(-base["ci_low"])
    with pytest.raises(ValueError):
        bootstrap([1, 2], [1, 2], [1, 2], days)
    with pytest.raises(ValueError):
        bootstrap([], [], [])


def test_reference_is_fixed_rf_even_when_another_baseline_is_better(metrics):
    reference = "Random Forest (보정)"
    results = {
        reference: {"y_avg": np.zeros(8), "pred_avg": np.full(8, 2),
                    "index": pd.date_range("2021-01-01", periods=8)},
        "Seasonal Naive": {"y_avg": np.zeros(8), "pred_avg": np.ones(8),
                           "index": pd.date_range("2021-01-01", periods=8)},
    }
    env = load_functions("s06_eval.py", {"significance_vs_baseline"},
                         REFERENCE_BASELINE_NAME=reference, REGRESSION_MODELS=list(results),
                         test_results=results, **metrics)
    table, baseline, _ = env["significance_vs_baseline"]()
    assert baseline == reference
    naive = table.set_index("모델").loc["Seasonal Naive"]
    assert naive["개선율(%)"] == 50
    assert naive["개선_5pct(주분석)"] and naive["개선_10pct(사후 탐색)"]
    assert not table.set_index("모델").loc[reference, "유의"]


def test_generated_wording_keeps_primary_failure_and_exploratory_result_separate():
    significance = pd.DataFrame([{
        "모델": "model", "유의": False, "개선_10pct(사후 탐색)": True,
        "차이 95% CI": "[-0.04, 0.86]", "차이 90% CI(사후 탐색)": "[0.03, 0.80]",
        "p-value": .073,
    }])
    env = load_functions("s06_eval.py", {"interpret_results"},
                         significance_tbl=significance, FINAL_MODEL_NAME="model",
                         IMPROVE_MAE_PCT=10.0, OOF_EVALUATION_NOTE="internal",
                         FINAL_PEAK_N=28, FINAL_PEAK_MAE=8.0, FINAL_PEAK15_MAE=10.0, THETA=187,
                         ablation_tbl=pd.DataFrame({"제거 피처군": []}),
                         rank_preservation=pd.DataFrame({"값": [.9, 1]}))
    text = env["interpret_results"]().set_index("항목").loc["MAE 개선의 유의성", "보고서 서술 교정"]
    assert "5% 주분석에서는 개선을 확인하지 못했다" in text
    assert "10% 사후 탐색" in text and "개선 방향의 차이를 보였다" in text
    assert "사후 탐색 결과로 5% 주분석 판정을 대체하지 않는다" in text
    assert "근사 p=0.0730" in text


def test_inference_timing_reuses_fitted_predictors(metrics):
    calls = {"fit": 0, "avg": 0, "peak": 0}

    def fit(*args):
        calls["fit"] += 1

        def predict_avg(frame):
            calls["avg"] += 1
            return np.zeros(len(frame))

        def predict_peak(frame):
            calls["peak"] += 1
            return np.zeros(len(frame))

        return {"_predict": predict_avg, "_predict_peak": predict_peak}

    ticks = iter(np.arange(20, dtype=float))
    env = load_functions("s06_eval.py", {"measure_inference_time"},
                         MODEL_REGISTRY={"model": fit}, time=SimpleNamespace(perf_counter=lambda: next(ticks)),
                         get_test_data=lambda cond: (None, None, pd.DataFrame({"x": [1, 2]}), None, None))
    assert env["measure_inference_time"]("model", n_rep=10) == 1
    assert calls == {"fit": 1, "avg": 10, "peak": 10}


def oof_example():
    frame = pd.DataFrame({
        "fold": [2, 2, 3, 3, 4, 4], "y_avg": [50, 170, 51, 171, 52, 172],
        "y_peak": [180, 190, 181, 191, 182, 192], "y_cls": [0, 1, 0, 1, 0, 1],
        "pred_avg": [55, 165, 56, 166, 57, 167], "pred_peak": [175, 190, 180, 189, 180, 190],
        "prob": [np.nan] * 6,
    }, index=pd.date_range("2021-07-01", periods=6, freq="D"))
    return {"oof": frame, "tau": 185}


def test_temporal_thresholds_never_use_current_or_later_fold_labels(metrics):
    seen = []

    def tune(y_peak, score, theta):
        seen.append(list(y_peak))
        return 185

    env = load_functions("s06_eval.py", {"build_peak_temporal_threshold_table"},
                         cv_results={"model": oof_example()}, THETA=187,
                         classification_metrics=metrics["classification_metrics"], tune_tau=tune)
    table = env["build_peak_temporal_threshold_table"]()
    assert seen == [[180, 190], [180, 190, 181, 191]]
    assert table["n"].tolist() == [2, 2, 4]
    folds = table[table["평가구간"].str.startswith("fold")]
    assert (pd.to_datetime(folds["보정 종료"]) < pd.to_datetime(folds["평가 시작"])).all()
    assert table["평가 성격"].str.contains("HPO").all()


def test_plan_ablation_removes_all_target_day_production_calendar_proxies(metrics):
    direct = ["prod", "headcount", "prod_chg_ratio", "prod_cum_day"]
    proxies = ["is_shutdown", "shutdown_nth", "is_first_day_back", "is_state_switch"]
    columns = direct + proxies + ["prev_day_shutdown", "hour", "y_avg_lag24"]
    seen = {}

    def run_cv(name, fit, cond, features):
        seen["features"] = features
        return oof_example()

    def make_regime_model(n_classes, gate_features):
        seen["gate"] = gate_features
        return None

    env = load_functions("s06_eval.py", {"run_plan_information_ablation"},
                         FEATURE_GROUPS={"생산정보": direct}, FEATURE_COLS=columns,
                         MODEL_REGISTRY={"2단계 레짐(3분류)": None},
                         cv_results={"2단계 레짐(3분류)": oof_example()},
                         run_cv=run_cv, make_regime_model=make_regime_model, THETA=187,
                         OOF_EVALUATION_NOTE="internal", **metrics)
    table = env["run_plan_information_ablation"]("2단계 레짐(3분류)")
    assert seen["features"] == seen["gate"] == ["prev_day_shutdown", "hour", "y_avg_lag24"]
    assert len(table) == 2 and (table["Peak-n"] == 3).all()


def test_all_evaluation_regression_metrics_calls_supply_actual_peak_truth():
    tree = ast.parse((ROOT / "src" / "s06_eval.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "regression_metrics"]
    assert calls
    assert all({"y_peak", "pred_peak"} <= {keyword.arg for keyword in node.keywords} for node in calls)
