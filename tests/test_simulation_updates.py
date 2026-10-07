"""Run reporting/simulation regressions without importing the training pipeline."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"


def load_functions(filename, names, namespace):
    tree = ast.parse((SRC / filename).read_text(encoding="utf-8-sig"))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def simulation_namespace():
    index = pd.date_range("2021-01-01", "2021-02-14 23:00", freq="h")
    peak = pd.Series(100.0, index=index)
    production = pd.Series(1000.0, index=index)
    namespace = dict(np=np, pd=pd, BASE_RATE_KRW_PER_KW_MONTH=100.0,
                     BILLING_MONTHS=12, ALPHA_SIMULTANEOUS=0.2,
                     PEAK15=peak, PROD=production, BASE_BILLING_PEAK=100.0,
                     BASE_BASIC_CHARGE=120000.0, BETA=0.01, THETA=90.0,
                     OBS_PEAK_MIN=0.0, OBS_PEAK_MAX=1000.0,
                     TOP_PEAK_TIMES=peak.nlargest(20).index,
                     NIGHT_LABOR_MULTIPLIER=1.5, WEEKEND_PREMIUM_PARAM=1.5,
                     LABOR_COST_KRW_PER_UNIT_HOUR=25000)
    return load_functions("s08_simulation.py", {
        "billing_units", "annual_billing_peak", "basic_charge",
        "observed_billing_months", "clip_to_observed", "lever_L1", "lever_L2",
        "lever_L3", "lever_L4", "evaluate_lever",
    }, namespace)


def test_peak_reduction_does_not_create_energy_savings():
    ns = simulation_namespace()
    reduced = ns["PEAK15"] - 10
    row = ns["evaluate_lever"]("test", reduced, ns["PROD"], extra_labor=100)
    assert row["전력량요금 차액(원)"] == 0
    assert "산정 제외" in row["전력량요금 산정"]
    assert row["순절감액(원)"] == row["기본요금 절감(원)"] - 100


def test_receiver_load_can_create_a_peak_above_the_observed_maximum():
    ns = simulation_namespace()
    ns["OBS_PEAK_MAX"] = 100.0
    shifted, production = ns["lever_L2"](ns["PEAK15"], ns["PROD"])
    assert shifted.max() == pytest.approx(100.5)
    row = ns["evaluate_lever"]("receiver peak", shifted, production)
    assert row["Δ최대수요(kW)"] == -0.5
    assert row["순절감액(원)"] < 0
    assert row["관측최대 초과 여부"] is True
    assert row["관측최대 초과 시간수"] == 45 * 4
    assert ns["clip_to_observed"](pd.Series([-1.0, 0.0, 120.0])).tolist() == [0.0, 0.0, 120.0]


def test_cost_comparison_uses_observed_months_and_separate_annual_reference():
    ns = simulation_namespace()
    assert ns["observed_billing_months"](ns["PEAK15"]) == pytest.approx(1.5)
    row = ns["evaluate_lever"]("test", ns["PEAK15"] - 10, ns["PROD"], 100)
    assert row["비교기간(개월)"] == 1.5
    assert row["기본요금 절감(원)"] == 1500
    assert row["연간 기본요금 절감 환산(원)"] == 12000
    assert row["순절감액(원)"] == 1400


def test_cost_comparison_rejects_different_observation_period():
    ns = simulation_namespace()
    with pytest.raises(ValueError, match="비교기간"):
        ns["evaluate_lever"]("test", ns["PEAK15"].iloc[:-1], ns["PROD"])


def test_production_movements_preserve_the_claimed_period():
    ns = simulation_namespace()
    peak, prod = ns["PEAK15"], ns["PROD"]
    for name in ("lever_L2", "lever_L3"):
        result = ns[name](peak, prod)
        pd.testing.assert_series_equal(result[1].resample("D").sum(), prod.resample("D").sum())
    _, moved, _ = ns["lever_L4"](peak, prod, "weekend")
    np.testing.assert_allclose(moved.resample("W-SUN").sum(), prod.resample("W-SUN").sum())


def test_l3_keeps_end_of_day_work_instead_of_dropping_it():
    ns = simulation_namespace()
    idx = pd.date_range("2021-01-01", periods=24, freq="h")
    prod = pd.Series([10.0] * 23 + [1000.0], index=idx)
    peak = pd.Series([20.0] * 23 + [100.0], index=idx)
    _, moved, unresolved = ns["lever_L3"](peak, prod)
    assert moved.sum() == pytest.approx(prod.sum())
    assert unresolved == 0


def test_failure_description_follows_observed_direction_and_shutdown():
    idx = pd.to_datetime(["2021-08-02 08:00", "2021-08-03 08:00"])
    observations = pd.DataFrame({
        "y_avg": [20.0, 180.0], "pred_avg": [180.0, 20.0],
        "abs_err": [160.0, 160.0], "생산량": [0.0, 100.0], "기온": [25.0, 25.0],
        "is_shutdown": [1, 0], "prev_day_shutdown": [1, 1],
        "prod_chg_ratio": [0.0, 1.0], "is_startup_08": [1, 1], "is_restart_13": [0, 0],
    }, index=idx)
    history = pd.DataFrame({"y_avg": [150.0, 150.0]}, index=idx - pd.Timedelta(hours=168))
    ns = load_functions("s07_analysis.py", {"pick_failure_cases"}, dict(pd=pd, df=history))
    rows = ns["pick_failure_cases"](observations)
    assert "과대예측" in rows.iloc[0]["오류 원인"]
    assert "휴무" in rows.iloc[0]["오류 원인"]
    assert "과소예측" in rows.iloc[1]["오류 원인"]
    assert "돌입전류" not in " ".join(rows["오류 원인"])


def test_condition_uncertainty_keeps_same_day_errors_together():
    idx = pd.date_range("2021-07-01", periods=96, freq="h")
    observations = pd.DataFrame({
        "abs_err": np.repeat([1.0, 1.0, 10.0, 10.0], 24),
        "is_shutdown": 0, "생산량": 100.0, "is_startup_08": 0,
        "is_restart_13": 0, "is_daytime": 1, "prev_day_shutdown": 0,
        "prod_chg_ratio": 0.0,
    }, index=idx)
    ns = load_functions("s07_analysis.py", {"condition_error_table"}, dict(
        np=np, pd=pd, SEED=42, FINAL_MODEL_NAME="test",
        test_results={"test": {"y_avg": np.zeros(24), "pred_avg": np.zeros(24)}},
    ))
    table = ns["condition_error_table"](observations)
    row = table[table["구분"] == "가동상태"].iloc[0]
    assert row["일 블록 수"] == 4
    assert row["95% CI(조건 충족)"] == "[1.00, 10.00]"


def test_shap_uses_the_named_model_instead_of_a_fixed_lightgbm(monkeypatch):
    idx = pd.date_range("2021-07-01", periods=2, freq="h")
    features = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)
    expected_model = object()
    seen = []

    def tree_explainer(model):
        seen.append(model)
        return SimpleNamespace(shap_values=lambda frame: np.zeros(frame.shape))

    monkeypatch.setitem(__import__("sys").modules, "shap", SimpleNamespace(TreeExplainer=tree_explainer))
    ns = load_functions("s07_analysis.py", {"compute_shap"}, dict(
        get_test_data=lambda condition: (features, features, features, None, None),
        MODEL_REGISTRY={"Random Forest(보정)": lambda *args: {"_models": {"y_avg": expected_model}}},
        feat=features, OOF=features, FEATURE_COLS=["x"], SEED=42,
    ))
    values, explained = ns["compute_shap"]("Random Forest(보정)")
    assert seen == [expected_model]
    assert values.shape == (2, 1)
    pd.testing.assert_frame_equal(explained, features)


def test_protocol_separates_service_manifest_from_research_model_and_peak_from_q90():
    ns = load_functions("s08_simulation.py", {"build_protocol"}, dict(pd=pd, FINAL_MODEL_NAME="연구모델"))
    protocol = ns["build_protocol"]()
    prediction = protocol[protocol["조치"] == "예측 실행 (원점)"].iloc[0]
    assert "저장된 서비스 번들" in prediction["내용"]
    assert "manifest" in prediction["비고"]
    assert "연구모델" not in prediction["내용"]
    monitoring = protocol[protocol["조치"] == "실시간 감시"].iloc[0]
    assert "평균전력" in monitoring["비고"]
    assert "분리" in monitoring["비고"]


def test_simulation_draft_does_not_claim_automatic_scheduling_or_guaranteed_effect():
    source = (SRC / "s08_simulation.py").read_text(encoding="utf-8-sig")
    assert "관측기간" in source
    assert "사후" in source
    assert "순절감액 **{best['순절감액(원)']:,.0f}원**을 확보" not in source
    creativity = (SRC / "s09_creativity.py").read_text(encoding="utf-8-sig")
    assert "'확률 0.7 = 실제 70% 발생'이 성립한다" not in creativity
