"""학습 파이프라인을 실행하지 않는 HPO·시간순 보정·구간 회귀 테스트."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression


SOURCE = Path(__file__).resolve().parents[1] / "src" / "s05_models.py"


def load_functions(*names, **extra):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"np": np, "pd": pd, "IsotonicRegression": IsotonicRegression, **extra}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("function,factory", [
    ("optimize_lgb", "make_lgb_regressor"),
    ("optimize_peak_clf", "make_peak_classifier"),
])
@pytest.mark.parametrize("budget,fast", [(800, False), (120, True)])
def test_hpo_returns_fixed_sampling_and_uses_final_tree_budget(function, factory, budget, fast):
    """실제 objective를 가벼운 학습 대역으로 실행해 탐색·최종 조건을 비교한다."""
    class Trial:
        def __init__(self):
            self.params = {}

        def suggest_int(self, name, low, high):
            self.params[name] = low
            return low

        def suggest_float(self, name, low, high, **kwargs):
            self.params[name] = low
            return low

    class Study:
        def optimize(self, objective, **kwargs):
            trial = Trial()
            self.value = objective(trial)
            self.best_params = trial.params

        def trials_dataframe(self):
            return pd.DataFrame({"number": [0], "value": [self.value]})

    calls = []

    def fit_factory(params, n_estimators=None):
        n_estimators = budget if n_estimators is None else n_estimators
        calls.append((dict(params), n_estimators))
        return lambda *args: {"pred_avg": np.array([1., 2.]), "prob": np.array([0.1, 0.9])}

    y = pd.DataFrame({"y_avg": [1., 2.], "y_cls": [0, 1]})
    namespace = load_functions(
        function, **{factory: fit_factory}, N_ESTIMATORS=budget, FAST=fast,
        N_TRIALS=1, SEED=42, ACTIVE_FOLDS=[2],
        mae=lambda a, b: np.mean(np.abs(np.asarray(a) - b)),
        get_fold_data=lambda *args: (y, y, y, y, y.index),
        optuna=SimpleNamespace(create_study=lambda **kwargs: Study(),
                               samplers=SimpleNamespace(TPESampler=lambda **kwargs: None)),
    )
    params, history = namespace[function]()
    assert params["bagging_freq"] == 1
    assert calls == [(params, budget)]
    assert history["n_estimators"].eq(budget).all()
    assert history["bagging_freq"].eq(1).all()


def make_oof():
    return pd.DataFrame(
        {"fold": np.repeat([2, 3, 4], 6),
         "y_cls": [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         "prob": np.tile([0., .1, .2, .5, .8, 1.], 3)},
        index=pd.date_range("2021-01-01", periods=18, freq="h"),
    )


def test_calibration_scores_future_folds_without_refitting_on_their_labels():
    class UntouchableTestResults(dict):
        def __getitem__(self, key):
            raise AssertionError("테스트 정답을 보정 설정에 쓰면 안 됩니다.")

    namespace = load_functions("calibrate_probabilities", "expected_calibration_error")
    source = make_oof()
    table, result = namespace["calibrate_probabilities"](
        {"피크 직접분류": {"oof": source}}, UntouchableTestResults(),
    )
    assert len(result["y"]) == 12  # 최초 fold는 보정 적합에만 사용
    expected = IsotonicRegression(out_of_bounds="clip").fit(
        source.iloc[:6]["prob"], source.iloc[:6]["y_cls"],
    ).predict(source.iloc[6:12]["prob"])
    np.testing.assert_allclose(result["p_cal"][:6], expected)
    folds = result["fold_table"]
    assert (pd.to_datetime(folds["보정 종료"]) < pd.to_datetime(folds["평가 시작"])).all()
    assert table["평가 n"].eq(12).all()
    assert folds["보정 n"].tolist() == [6, 12]
    final = IsotonicRegression(out_of_bounds="clip").fit(source["prob"], source["y_cls"])
    np.testing.assert_allclose(result["iso"].predict(source["prob"]), final.predict(source["prob"]))

    changed = source.copy()
    changed.loc[changed["fold"] == 3, "y_cls"] = 1 - changed.loc[changed["fold"] == 3, "y_cls"]
    _, changed_result = namespace["calibrate_probabilities"]({"피크 직접분류": {"oof": changed}}, {})
    np.testing.assert_array_equal(result["p_cal"][:6], changed_result["p_cal"][:6])


def test_calibration_requires_separate_chronological_folds():
    namespace = load_functions("calibrate_probabilities", "expected_calibration_error")
    source = make_oof()
    source["fold"] = 2
    with pytest.raises(ValueError, match="fold"):
        namespace["calibrate_probabilities"]({"피크 직접분류": {"oof": source}}, {})

    source = make_oof()
    source.iloc[0, source.columns.get_loc("fold")] = 3
    with pytest.raises(ValueError, match="fold"):
        namespace["calibrate_probabilities"]({"피크 직접분류": {"oof": source}}, {})


def test_ece_includes_zero_probability_observations():
    fn = load_functions("expected_calibration_error")["expected_calibration_error"]
    assert fn([1, 0], [0., 1.]) == pytest.approx(1.)


def test_conformal_quantile_uses_finite_sample_order_statistic():
    fn = load_functions("conformal_quantile")["conformal_quantile"]
    assert fn(np.arange(1, 11), alpha=.10) == 10.
    assert fn(np.arange(1, 10), alpha=.10) == 9.
    assert np.isinf(fn([1., 2., 3.], alpha=.10))
    with pytest.raises(ValueError):
        fn([], alpha=.10)
    with pytest.raises(ValueError):
        fn([1., np.nan], alpha=.10)
    with pytest.raises(ValueError):
        fn([1., 2.], alpha=1.)


def test_interval_fit_applies_finite_sample_quantile_to_each_calibration_subset():
    class Regressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    namespace = load_functions(
        "fit_interval_models", "conformal_quantile", N_ESTIMATORS=800, LGB_BASE={},
        lgb=SimpleNamespace(LGBMRegressor=Regressor),
    )
    idx = pd.date_range("2021-01-01", periods=100, freq="D")
    X = pd.DataFrame({"feature": np.arange(100)}, index=idx)
    y = pd.DataFrame({"y_avg": np.arange(100)}, index=idx)
    result = namespace["fit_interval_models"](X, y, idx[-10:])
    assert result["qhat"]["전체(휴무 포함)"] == 98.
    assert result["qhat"]["운영일만(휴무 제외)"] == 89.
    assert result["n_cal_by_group"] == {"전체(휴무 포함)": 20, "운영일만(휴무 제외)": 10}


def test_operating_day_conformal_is_evaluated_only_on_operating_days():
    idx = pd.date_range("2021-01-01", periods=4, freq="D")
    X = pd.DataFrame({"x": np.zeros(4)}, index=idx)
    y = pd.DataFrame({"y_avg": [0., 0., 100., 100.]}, index=idx)
    predictor = SimpleNamespace(predict=lambda X: np.zeros(len(X)))
    fitted = {
        "qmodels": {q: predictor for q in [.1, .5, .9]}, "point": predictor,
        "qhat": {"전체(휴무 포함)": 100., "운영일만(휴무 제외)": 1.},
        "n_cal_by_group": {"전체(휴무 포함)": 20, "운영일만(휴무 제외)": 10},
    }
    namespace = load_functions(
        "quantile_intervals", get_test_data=lambda cond: (X, y, X, y, idx),
        fit_interval_models=lambda *args: fitted,
        operating_calendar=pd.DataFrame({"is_shutdown": [False, False, True, True]}, index=idx),
    )
    table, _ = namespace["quantile_intervals"]()
    operating = table[table["보정집합"] == "운영일만(휴무 제외)"].iloc[0]
    assert operating["평가 n"] == 2
    assert operating["실제 피복률"] == 1.
    assert "운영일 한정" in operating["해석"]


def test_cv_passes_true_and_predicted_peak_targets_to_metrics():
    idx = pd.date_range("2021-01-01", periods=2, freq="h")
    y = pd.DataFrame({"y_avg": [2., 3.], "y_peak": [187., 100.], "y_cls": [1, 0]}, index=idx)
    calls = []

    def metrics(y_avg, pred_avg, theta, y_peak=None, pred_peak=None):
        calls.append((y_peak, pred_peak))
        return {"MAE": 0.}

    namespace = load_functions(
        "run_cv", ACTIVE_FOLDS=[2], THETA=187., _TIMING=[],
        time=SimpleNamespace(perf_counter=lambda: 0.), regression_metrics=metrics,
        get_fold_data=lambda *args: (y, y, y, y, idx), tune_tau=lambda *args: 180.,
    )
    namespace["run_cv"]("example", lambda *args: {"pred_avg": [2., 3.], "pred_peak": [185., 102.]})
    pd.testing.assert_series_equal(calls[0][0], y["y_peak"])
    np.testing.assert_array_equal(calls[0][1], [185., 102.])


def test_dnn_prediction_callbacks_reuse_fitted_scalers_and_models():
    class Network:
        fits = 0

        def __init__(self, layers):
            self.offset = Network.fits

        def compile(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            Network.fits += 1

        def predict(self, X, verbose=0):
            return (np.asarray(X)[:, 0] * 1.5 + self.offset).reshape(-1, 1)

    def layer(*args, **kwargs):
        return None

    keras = SimpleNamespace(
        Sequential=Network, utils=SimpleNamespace(set_random_seed=layer),
        layers=SimpleNamespace(Input=layer, Dense=layer, BatchNormalization=layer, Dropout=layer),
        optimizers=SimpleNamespace(Adam=layer), callbacks=SimpleNamespace(EarlyStopping=layer),
    )
    ns = load_functions("dnn_fit", HAS_TF=True, SEED=42, DNN_EPOCHS=1, keras=keras)
    X = pd.DataFrame({"a": np.arange(50.), "b": np.arange(50.) * 2})
    y = pd.DataFrame({"y_avg": np.arange(50.) + 100, "y_peak": np.arange(50.) * 2 + 200})
    out = ns["dnn_fit"](X, y, X.iloc[-3:])
    assert Network.fits == 2
    for callback, key, target in (("_predict", "pred_avg", "y_avg"), ("_predict_peak", "pred_peak", "y_peak")):
        np.testing.assert_array_equal(out[callback](X.iloc[-3:]), out[key])
        changed = X.iloc[:4] + 7
        state = out["_persist"]
        expected = state["yscalers"][target].inverse_transform(
            state["models"][target].predict(state["xsc"].transform(changed)).reshape(-1, 1)
        ).ravel()
        np.testing.assert_array_equal(out[callback](changed), expected)
    assert Network.fits == 2


def test_ensemble_prediction_callbacks_reuse_members_for_new_inputs():
    from tools.model_persistence import predict_state, state_from_output

    class Regressor:
        def __init__(self, scale):
            self.scale = scale

        def predict(self, X):
            return X["x"].to_numpy() * self.scale

    fits, callbacks = [], []

    def member_a(Xtr, ytr, Xva):
        fits.append("a")

        def predict(X, scale):
            callbacks.append(scale)
            return X["x"].to_numpy() * scale

        return {"pred_avg": Xva["x"].to_numpy() * 2, "pred_peak": Xva["x"].to_numpy() * 3,
                "_predict": lambda X: predict(X, 2), "_predict_peak": lambda X: predict(X, 3),
                "_models": {"y_avg": Regressor(2), "y_peak": Regressor(3)}}

    def member_b(Xtr, ytr, Xva):
        fits.append("b")
        return {"pred_avg": Xva["x"].to_numpy() * 5, "pred_peak": Xva["x"].to_numpy() * 7,
                "_models": {"y_avg": Regressor(5), "y_peak": Regressor(7)}}

    ns = load_functions("ensemble_fit", _n1="a", _n2="b", _w=.25,
                        MODEL_REGISTRY={"a": member_a, "b": member_b},
                        predict_state=predict_state, state_from_output=state_from_output)
    X = pd.DataFrame({"x": [1., 2., 3.]})
    out = ns["ensemble_fit"](X, None, X)
    for callback, key, scale in (("_predict", "pred_avg", 4.25), ("_predict_peak", "pred_peak", 6.)):
        np.testing.assert_array_equal(out[callback](X), out[key])
        changed = pd.DataFrame({"x": [10., 20.]})
        np.testing.assert_array_equal(out[callback](changed), changed["x"].to_numpy() * scale)
    assert fits == ["a", "b"]
    assert callbacks == [2, 2, 3, 3]
