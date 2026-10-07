"""Small fixtures only: no training-stage import and no fitted-model rerun."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import run_all
from tools.analysis_snapshot import export_snapshot, load_snapshot


@pytest.fixture
def namespace():
    idx = pd.date_range("2021-07-07", periods=48, freq="h", name="시각")
    df = pd.DataFrame({"y_avg": np.linspace(1.125, 101.3456789012345, 48),
                       "y_peak": np.linspace(3.5, 130.9876543210123, 48),
                       "생산량": np.arange(48, dtype=float), "공장인원": np.arange(48, dtype=float) / 2}, index=idx)
    df["y_cls"] = (df["y_peak"] >= 100).astype(int)
    feat = df.copy()
    for col in ("is_warmup", "is_outage", "is_erp_missing"):
        feat[col] = False
    cv = df.iloc[:24][["y_avg", "y_peak", "y_cls"]].copy()
    cv["pred_avg"] = cv["y_avg"] + 0.123456789012345
    cv["pred_peak"] = cv["y_peak"] - 0.9876543210987
    cv["prob"], cv["fold"] = np.nan, 2
    test = df.iloc[24:]
    result = {"index": test.index, "name": "모델", "cond": "D1",
              **{k: test[k].to_numpy().copy() for k in ("y_avg", "y_peak", "y_cls")},
              "pred_avg": test["y_avg"].to_numpy() + 0.125,
              "pred_peak": test["y_peak"].to_numpy() - 0.625, "prob": None}
    cal = pd.DataFrame({"is_shutdown": [False, True]}, index=pd.date_range("2021-07-07", periods=2))
    condition = pd.DataFrame({"구분": ["전체", "NA", ""], "mixed": [None, ("tuple", 1), np.nan]})
    condition.attrs["overall"] = 1.12345678901234
    return {"df": df, "feat": feat, "operating_calendar": cal,
            "cv_results": {"모델": {"name": "모델", "cond": "D1", "oof": cv, "tau": 99.123456789,
                                   "fold_metrics": pd.DataFrame({"fold": [2], "MAE": [.123456789012345]})}},
            "test_results": {"모델": result}, "FOLD_SPEC": [(2, "2021-07-05", "2021-07-07", "2021-07-07")],
            "THETA": 100., "TEST_START": pd.Timestamp("2021-07-08"),
            "TEST_END": pd.Timestamp("2021-07-08 23:00"), "FINAL_MODEL_NAME": "모델",
            "FEATURE_COLS": ["생산량", "공장인원"], "SEED": 42, "N_ESTIMATORS": 800,
            "shap_values": np.array([[.12345678901234, -3.141592653589793]]),
            "shap_X": feat.iloc[:1][["생산량", "공장인원"]].copy(),
            "condition_tbl": condition, "_THETA_PREVIEW": 99.5, "_clf_name": "classifier",
            "_unc": {"coverage": .8}, "unfitted_callable": lambda: (_ for _ in ()).throw(AssertionError("fit"))}


def test_roundtrip_precision_schema_and_no_mutation(tmp_path, namespace):
    before = copy.deepcopy({k: namespace[k] for k in ("df", "feat", "cv_results", "test_results", "shap_values")})
    path = tmp_path / "한글 공백 경로" / "snapshot"
    meta = export_snapshot(namespace, path)
    restored = load_snapshot(path)
    pd.testing.assert_frame_equal(restored["df"], namespace["df"], check_exact=True)
    pd.testing.assert_frame_equal(restored["feat"], namespace["feat"], check_exact=True)
    pd.testing.assert_frame_equal(restored["cv_results"]["모델"]["oof"], namespace["cv_results"]["모델"]["oof"], check_exact=True)
    np.testing.assert_array_equal(restored["shap_values"], namespace["shap_values"])
    pd.testing.assert_frame_equal(namespace["df"], before["df"], check_exact=True)
    pd.testing.assert_frame_equal(namespace["feat"], before["feat"], check_exact=True)
    np.testing.assert_array_equal(namespace["test_results"]["모델"]["pred_avg"], before["test_results"]["모델"]["pred_avg"])
    assert restored["condition_tbl"].attrs == namespace["condition_tbl"].attrs
    assert restored["condition_tbl"]["구분"].tolist() == ["전체", "NA", ""]
    assert restored["condition_tbl"]["mixed"].iloc[1] == ("tuple", 1)
    assert restored["_THETA_PREVIEW"] == 99.5 and restored["_unc"]["coverage"] == .8
    assert restored["processed"] is restored["df"] and restored["features"] is restored["feat"]
    assert meta["complete"] and meta["constants"]["N_ESTIMATORS"] == 800
    assert meta["coverage"]["oof_rows_by_model"] == {"모델": 24}
    assert "unfitted_callable" in meta["skipped_objects"]
    assert all(v != "pickle" for v in (p.suffix for p in path.rglob("*")))


def test_long_form_boundaries_masks_and_original_threshold(tmp_path, namespace):
    export_snapshot(namespace, tmp_path / "snapshot")
    loaded = load_snapshot(tmp_path / "snapshot")
    oof, test = loaded["oof_long"], loaded["test_long"]
    assert len(oof) == 24 and len(test) == 24
    assert (oof["train_end"] < oof["datetime"]).all()
    for rows in (oof, test):
        assert (rows["origin"] == rows["datetime"].dt.normalize()).all()
        assert (rows["history_end"] == rows["origin"] - pd.Timedelta(hours=1)).all()
        assert (rows["history_end"] + pd.Timedelta(hours=1) <= rows["origin"]).all()
    assert oof["eligible"].all() and oof["tau"].eq(99.123456789).all()
    np.testing.assert_array_equal(test["peak_pred_label"], (test["pred_peak"] >= test["tau"]).astype(int))
    assert test["evaluation_role"].eq("frozen_reference").all()


@pytest.mark.parametrize("invalid", ["duplicate", "out_of_fold", "quality", "object_array", "missing", "stale_truth", "test_gap"])
def test_bad_inputs_fail_closed(tmp_path, namespace, invalid):
    if invalid == "duplicate":
        o = namespace["cv_results"]["모델"]["oof"]
        namespace["cv_results"]["모델"]["oof"] = pd.concat([o, o.iloc[:1]])
    elif invalid == "out_of_fold":
        namespace["FOLD_SPEC"] = [(2, "2021-07-05", "2021-07-08", "2021-07-09")]
    elif invalid == "quality":
        namespace["feat"].iloc[0, namespace["feat"].columns.get_loc("is_outage")] = True
    elif invalid == "object_array":
        namespace["bad"] = np.array([object()], dtype=object)
    elif invalid == "stale_truth":
        namespace["test_results"]["모델"]["y_avg"][0] += 1
    elif invalid == "test_gap":
        namespace["TEST_END"] += pd.Timedelta(hours=1)
    else:
        namespace.pop("feat")
    path = tmp_path / "failed"
    with pytest.raises(ValueError):
        export_snapshot(namespace, path)
    assert json.loads((path / "metadata.json").read_text(encoding="utf-8"))["complete"] is False
    with pytest.raises(ValueError, match="incomplete"):
        load_snapshot(path)


def test_modified_payload_and_nonempty_destination_rejected(tmp_path, namespace):
    path = tmp_path / "snapshot"
    meta = export_snapshot(namespace, path)
    first = path / next(iter(meta["files_hashes"]))
    first.write_bytes(first.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(path)
    with pytest.raises(FileExistsError):
        export_snapshot(namespace, path)


def test_modified_scientific_metadata_rejected(tmp_path, namespace):
    path = tmp_path / "snapshot"
    meta = export_snapshot(namespace, path)
    meta["namespace"]["THETA"]["value"] = 999
    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_snapshot(path)


def test_tree_export_preserves_weighted_and_actual_counts_without_fit(tmp_path, namespace):
    fit = Mock(side_effect=AssertionError("Must not fit"))
    namespace["OOF"] = namespace["df"].copy()
    namespace["rule_cols"] = ["생산량"]
    namespace["peak_tree"] = SimpleNamespace(fit=fit, classes_=np.array([0, 1]), tree_=SimpleNamespace(
        node_count=3, max_depth=1, children_left=np.array([1, -1, -1]), children_right=np.array([2, -1, -1]),
        feature=np.array([0, -2, -2]), threshold=np.array([23.5, -2., -2.]),
        n_node_samples=np.array([48, 24, 24]), weighted_n_node_samples=np.array([48., 20., 28.]),
        value=np.array([[[.5, .5]], [[.9, .1]], [[.2, .8]]])))
    export_snapshot(namespace, tmp_path / "snapshot")
    restored = load_snapshot(tmp_path / "snapshot")
    nodes = restored["peak_tree"]["nodes"]
    assert nodes[0]["actual_n"] == 48
    assert nodes[1]["actual_n"] == nodes[2]["actual_n"] == 24
    assert nodes[0]["actual_peak"] == int(namespace["df"]["y_cls"].sum())
    assert nodes[1]["weighted_n_node_samples"] == 20.
    assert nodes[1]["value"] == [[.9, .1]]
    fit.assert_not_called()


def test_classifier_unknown_exact_tau_is_not_fabricated(tmp_path, namespace):
    namespace["cv_results"]["모델"]["oof"]["prob"] = .2
    namespace["test_results"]["모델"]["prob"] = np.full(24, .3)
    export_snapshot(namespace, tmp_path / "snapshot")
    restored = load_snapshot(tmp_path / "snapshot")
    assert restored["test_long"]["tau"].isna().all()
    assert restored["test_long"]["peak_pred_label"].isna().all()


def test_export_hook_failure_is_reported(tmp_path, monkeypatch):
    runner = run_all.Runner(run_all.parse_args(["--export-analysis", str(tmp_path / "export")]))
    runner.index = {}
    runner.steps = []
    runner.ns = {"FINAL_MODEL_NAME": "dummy"}
    runner.write_index = Mock()
    runner.screen = Mock()
    monkeypatch.setattr("tools.analysis_snapshot.export_snapshot", Mock(side_effect=ValueError("fixture")))
    assert runner._export_analysis() is False
    assert runner.index["analysis_export"]["status"] == "failed"


def test_pipeline_propagates_export_failure(monkeypatch, tmp_path):
    runner = run_all.Runner(run_all.parse_args(["--export-analysis", str(tmp_path / "export")]))
    runner.index = {"checks": {"hard": [], "warn": [], "info": []}}
    runner._header = Mock()
    runner._summary = Mock()
    runner.write_index = Mock()
    runner._export_analysis = Mock(return_value=False)
    monkeypatch.setattr(run_all, "LOCK_PATH", tmp_path / "unused_lock")
    assert runner._run() == 1
    assert runner.index["status"] == "export_failed"
    runner._export_analysis.assert_called_once()


def test_cli_check_only_rejects_export_and_plan_stays_62():
    with pytest.raises(SystemExit) as exc:
        run_all.parse_args(["--check-only", "--export-analysis", "anything"])
    assert exc.value.code == 2
    steps, _ = run_all.build_plan()
    assert len(steps) == 62
    assert sum(s["cells"] for s in steps[:run_all.N_PLAN_STEPS]) == 68
