"""Process-isolated regime predictions, authenticated caches, and CV lifetime checks."""
from __future__ import annotations

import ast
import gc
import inspect
import json
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from tools import isolated_regime_cv as isolated


ROOT = Path(__file__).resolve().parents[1]


def functions(*names, **globals_):
    source = ROOT / "src" / "s05_models.py"
    nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    ns = {"np": np, "pd": pd, "lgb": lgb, "time": time, **globals_}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(source), "exec"), ns)
    return ns


@pytest.fixture
def small_folds():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-01", periods=168, freq="h")
    X = pd.DataFrame(rng.normal(size=(168, 4)), index=idx, columns=list("abcd"))
    X["flag"] = (X["a"] > 0).astype(int)
    avg = np.where(X["a"] < -.4, 20., np.where(X["a"] < .5, 50., 130.))
    y = pd.DataFrame({"y_avg": avg, "y_peak": avg + 30., "y_cls": (avg >= 120).astype(int)}, index=idx)
    folds = [(i + 2, X.iloc[:n].copy(), y.iloc[:n].copy(), X.iloc[n:n + 12].copy())
             for i, n in enumerate((96, 108, 120, 132))]
    config = {"FAST": True, "N_ESTIMATORS": 8, "REGIME_CUT": 70.,
              "LGB_BASE": {"objective": "l1", "verbosity": -1, "deterministic": True,
                           "force_row_wise": True, "num_threads": 4, "seed": 42}}
    return folds, config, X


@pytest.mark.parametrize("n_classes", [2, 3])
def test_isolated_predictions_match_in_process_exactly_and_verified_cache_reuses(
    small_folds, n_classes, tmp_path, monkeypatch,
):
    folds, config, frame = small_folds
    monkeypatch.setenv("KAMP_WORK_DIR", str(tmp_path))
    fit = functions("make_regime_model", **config)["make_regime_model"](n_classes, ["a", "flag"])
    expected = []
    for _, Xtr, ytr, Xva in folds:
        out = fit(Xtr, ytr, Xva)
        expected.append({key: out[key].copy() for key in ("pred_avg", "pred_peak")})
        del out
        gc.collect()
    result = isolated.isolated_regime_predictions(
        "tiny", folds, config, n_classes=n_classes, gate_features=["a", "flag"], frame=frame,
    )
    assert not result["cache_hit"]
    assert len(result["fit_seconds"]) == 4 and min(result["fit_seconds"]) > 0
    for actual, reference in zip(result["predictions"], expected):
        for key in reference:
            np.testing.assert_array_equal(actual[key], reference[key])

    def cannot_train(*args, **kwargs):
        raise AssertionError("a verified cache hit must not start a subprocess")

    monkeypatch.setattr(isolated.subprocess, "run", cannot_train)
    cached = isolated.isolated_regime_predictions(
        "tiny", folds, config, n_classes=n_classes, gate_features=["a", "flag"], frame=frame,
    )
    assert cached["cache_hit"] and cached["fit_seconds"] == result["fit_seconds"]


def test_cache_invalidates_and_corrupt_or_incomplete_payload_is_regenerated(small_folds, tmp_path, monkeypatch):
    folds, config, frame = small_folds
    monkeypatch.setenv("KAMP_WORK_DIR", str(tmp_path))
    run = isolated.subprocess.run
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return run(*args, **kwargs)

    monkeypatch.setattr(isolated.subprocess, "run", counted)

    def calculate(chosen=frame):
        return isolated.isolated_regime_predictions("tiny", folds, config, n_classes=2, frame=chosen)

    first = calculate()
    cache = tmp_path / "regime_cv"
    payload = cache / (first["fingerprint"] + ".npz")
    manifest = cache / (first["fingerprint"] + ".json")
    payload.write_bytes(b"corrupted")
    assert not calculate()["cache_hit"]
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    meta["status"] = "incomplete"
    manifest.write_text(json.dumps(meta), encoding="utf-8")
    assert not calculate()["cache_hit"]
    changed = frame.copy()
    changed.iloc[0, 0] += 1
    assert calculate(changed)["fingerprint"] != first["fingerprint"]
    assert len(calls) == 4


def test_native_failure_retries_identical_request_and_never_commits_cache(small_folds, tmp_path, monkeypatch):
    folds, config, frame = small_folds
    monkeypatch.setenv("KAMP_WORK_DIR", str(tmp_path))
    requests = []

    def crash(command, **kwargs):
        folder = Path(command[-1])
        requests.append(((folder / "input.json").read_bytes(), (folder / "input.npz").read_bytes()))
        return SimpleNamespace(returncode=3221225477, stderr="access violation", stdout="")

    monkeypatch.setattr(isolated.subprocess, "run", crash)
    with pytest.raises(RuntimeError, match="variant native/D1.*2 attempts; exit=3221225477"):
        isolated.isolated_regime_predictions("native", folds, config, n_classes=2, frame=frame)
    assert requests[0] == requests[1]
    assert not list((tmp_path / "regime_cv").glob("*.json"))


def test_cv_releases_previous_fit_before_starting_next_fold():
    class FittedModel:
        pass

    idx = pd.date_range("2021-01-01", periods=2, freq="h")
    y = pd.DataFrame({"y_avg": [2., 3.], "y_peak": [187., 100.], "y_cls": [1, 0]}, index=idx)
    refs = []

    def fit(*args):
        assert all(ref() is None for ref in refs), "previous fold is still alive during next fit"
        model = FittedModel()
        refs.append(weakref.ref(model))
        return {"pred_avg": [2., 3.], "pred_peak": [185., 102.], "_model": model}

    ns = functions("run_cv", ACTIVE_FOLDS=[2, 3], THETA=187., _TIMING=[],
                   regression_metrics=lambda *args, **kwargs: {"MAE": 0.},
                   get_fold_data=lambda *args: (y, y, y, y, idx), tune_tau=lambda *args: 180.)
    ns["run_cv"]("release", fit)
    assert len(refs) == 2 and all(ref() is None for ref in refs)


def test_lag_variant_replays_isolated_predictions_and_records_child_fit_time(small_folds):
    folds, config, frame = small_folds
    timing = []
    fit = functions("make_regime_model", **config)["make_regime_model"](3)
    by_fold = {fold: (Xtr, ytr, Xva, pd.DataFrame(
        {"y_avg": 0., "y_peak": 0., "y_cls": 0}, index=Xva.index,
    ), Xva.index) for fold, Xtr, ytr, Xva in folds}

    def predictions(variant, received, configuration, **kwargs):
        assert configuration == config and len(received) == 4
        assert kwargs["n_classes"] == 3
        return {"predictions": [{"pred_avg": np.zeros(len(Xva)), "pred_peak": np.zeros(len(Xva))}
                                for _, _, _, Xva in received],
                "fit_seconds": [1., 2., 3., 4.], "cache_hit": True}

    ns = functions("run_cv", ACTIVE_FOLDS=[x[0] for x in folds], THETA=187., _TIMING=timing,
                   regression_metrics=lambda *args, **kwargs: {"MAE": 0.},
                   get_fold_data=lambda fold, *args: by_fold[fold], tune_tau=lambda *args: 180.)
    ns.update(config, inspect=inspect, isolated_regime_predictions=predictions, feat=frame,
              MODEL_REGISTRY={"레짐(3분류)": fit}, FINAL_MODEL_NAME="레짐(3분류)", FEATURE_COLS=list(frame.columns))
    path = ROOT / "src" / "s06_eval.py"
    node = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name == "run_lag_variant")
    exec(compile(ast.Module([node], type_ignores=[]), str(path), "exec"), ns)
    result = ns["run_lag_variant"]("unit", "example")
    assert result["fit_sec"] == 2.5 and timing[-1]["초"] == 10.
    assert result["cache_hit"] and result["fit_sec_source"] == "checkpoint"
    assert len(result["oof"]) == 48 and result["oof"]["pred_avg"].eq(0).all()
