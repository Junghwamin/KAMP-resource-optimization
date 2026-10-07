"""Fixture-only checks of completion, provenance and orchestration boundaries."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import run_all
import run_experiments
import run_submission as pipeline
from tools import verify_submission as verify
from tools import verify_code_state as code_state
from tools import publish_figures as publisher


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def core(tmp_path, monkeypatch):
    root = tmp_path / "project"
    source = root / "data/source.csv"
    source.parent.mkdir(parents=True)
    source.write_text("x\n1\n", encoding="utf-8")
    (root / "run_all.py").write_text("# core fixture", encoding="utf-8")
    snapshot = root / "snapshot"
    steps = []
    ids = [s.split(":")[0] for s in run_all.EXPECTED_STEPS] + [s[0] for s in run_all.EXTRA_STEPS]
    for i, sid in enumerate(ids):
        log = root / f"outputs/steps/{i}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("completed", encoding="utf-8")
        steps.append({"id": sid, "status": "ok", "log": log.relative_to(root).as_posix()})
    index = {"status": "ok", "mode": "FULL", "started": "fixture", "flags": {}, "steps": steps,
             "serving": {key: {"status": "ok"} for key in ("S.1", "S.2", "S.3", "S.4")},
             "checks": {"hard": [{"ok": True}]}, "analysis_export": {"status": "complete", "fingerprint": "abc"}}
    meta = {"fingerprint": "abc", "run_metadata": {"mode": "FULL", "run_started": "fixture"},
            "source_hashes": {"data/source.csv": verify.digest(source)},
            "runtime": {"python": "fixture", "versions": {"numpy": "fixture"}}}
    original = {name: verify.digest(root / name) for name in ("run_all.py", "data/source.csv")}
    text = "".join(sorted(f"{value}  {name}\n" for name, value in original.items()))
    index["provenance"] = {"files": original, "code_sha256": hashlib.sha256(text.encode()).hexdigest()}
    put(root / "outputs/steps/index.json", index)
    put(snapshot / "metadata.json", meta)
    monkeypatch.setattr(verify, "_load_snapshot", lambda path: {"metadata": meta})
    return root, snapshot, index, meta


def test_complete_core_checks_all_steps_and_source(core):
    root, snapshot, _, _ = core
    assert verify.verify_core(root, snapshot, compare_runtime=False)["step_count"] == 62
    (root / "data/source.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify.verify_core(root, snapshot, compare_runtime=False)


@pytest.mark.parametrize("problem", ["test_skip", "step_missing", "export_wrong", "run_wrong", "hard_failure", "no_tests"])
def test_core_rejects_partial_or_unlinked_evidence(core, problem):
    root, snapshot, index, meta = core
    if problem == "test_skip":
        index["serving"]["S.4"]["status"] = "skipped"
    elif problem == "step_missing":
        index["steps"].pop()
    elif problem == "export_wrong":
        index["analysis_export"]["fingerprint"] = "different"
    elif problem == "run_wrong":
        meta["run_metadata"]["run_started"] = "older"
    elif problem == "hard_failure":
        index["checks"]["hard"][0]["ok"] = False
    else:
        index["flags"]["no_tests"] = True
    put(root / "outputs/steps/index.json", index)
    with pytest.raises(ValueError):
        verify.verify_core(root, snapshot, compare_runtime=False)


def test_reuse_does_not_accept_runtime_mismatch(core, monkeypatch):
    root, snapshot, _, _ = core
    monkeypatch.setattr(verify.platform, "python_version", lambda: "other")
    with pytest.raises(ValueError, match="Python version"):
        verify.verify_core(root, snapshot)


def experiment_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(run_experiments, "ROOT", tmp_path)
    monkeypatch.setattr(run_experiments, "fingerprint", lambda name, cfg, snap: (name + "-key", {}))
    cfg = {"seed": 42}
    for name in verify.EXPERIMENTS:
        folder = tmp_path / name
        put(folder / "summary.json", {"status": "complete"})
        put(folder / "executor_manifest.json", {"experiment": name, "status": "complete", "fingerprint": name + "-key",
            "inputs": {"config": cfg}, "summary": {"status": "complete"},
            "files_sha256": {"summary.json": verify.digest(folder / "summary.json")}})
    put(tmp_path / "run_status.json", {"status": "complete", "all_six_verified": True})
    return cfg


def test_experiments_reject_empty_or_stale_hashes(tmp_path, monkeypatch):
    cfg = experiment_fixture(tmp_path, monkeypatch)
    assert verify.verify_experiments(tmp_path, tmp_path / "snapshot", tmp_path, cfg)["status"] == "passed"
    path = tmp_path / "policy/executor_manifest.json"
    manifest = verify.read_json(path)
    manifest["files_sha256"] = {}
    put(path, manifest)
    with pytest.raises(ValueError, match="summary hash"):
        verify.verify_experiments(tmp_path, tmp_path / "snapshot", tmp_path, cfg)


def test_experiment_source_config_fingerprint_must_match(tmp_path, monkeypatch):
    cfg = experiment_fixture(tmp_path, monkeypatch)
    path = tmp_path / "temporal/executor_manifest.json"
    manifest = verify.read_json(path)
    manifest["fingerprint"] = "old source"
    put(path, manifest)
    with pytest.raises(ValueError, match="fingerprint"):
        verify.verify_experiments(tmp_path, tmp_path / "snapshot", tmp_path, cfg)


def figures_fixture(tmp_path):
    figs, run = tmp_path / "figures", tmp_path / "results"
    figs.mkdir()
    stage_source = tmp_path / "src/s00_env.py"
    stage_source.parent.mkdir()
    stage_source.write_text("# renderer stage fixture", encoding="utf-8")
    render_hashes = {}
    for relative in ("tools/render_figures.py", "tools/figure_layout.py"):
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("# renderer fixture", encoding="utf-8")
        render_hashes[relative] = verify.digest(path)
    put(tmp_path / "snapshot/metadata.json", {"fingerprint": "fixture"})
    rows = []
    for fid in verify.FIGURE_IDS:
        image = figs / f"{fid}.png"
        image.write_bytes(b"fixture image " + fid.encode())
        (figs / f"{fid}_src.csv").write_text("x\n1\n", encoding="utf-8")
        row = {"fid": fid, "png": image.name, "sha256": verify.digest(image), "glyph_warnings": [],
               "overlap_candidates": [{"a": "candidate", "b": "needs inspection"}], "minimum_font_pt": 8.}
        if fid.startswith("E"):
            source = run / fid / "source.csv"
            source.parent.mkdir(parents=True)
            source = source.with_name(fid + ".csv")
            source.write_text("x\n1\n", encoding="utf-8")
            row.update(source_table=source.name, source_sha256=verify.digest(source))
        put(figs / f"{fid}.qa.json", row)
        rows.append(row)
    put(figs / "render_manifest.json", {"status": "rendered_pending_visual_review", "rendered": 39, "failures": [],
        "snapshot_fingerprint": "fixture", "snapshot_metadata_sha256": verify.digest(tmp_path / "snapshot/metadata.json"),
        "source_sha256": {"s00_env.py": verify.digest(stage_source)}, "renderer_sha256": render_hashes,
        "snapshot_files_unchanged": True, "raw_arrays_unchanged": True, "figures": rows[:39]})
    put(figs / "experiment_render_manifest.json", {"status": "rendered_pending_visual_review", "rendered": 6,
        "renderer_sha256": render_hashes,
        "failures": [], "figures": rows[39:]})
    return figs, run


def test_all45_integrity_is_separate_from_visual_pass(tmp_path):
    figures, run = figures_fixture(tmp_path)
    result = verify.verify_figures(figures, run, snapshot=tmp_path / "snapshot", root=tmp_path)
    assert result["figure_count"] == 45
    assert verify.verify_visual(None, result)["status"] == "pending"
    (figures / "E06.png").write_bytes(b"modified")
    with pytest.raises(ValueError, match="PNG hash"):
        verify.verify_figures(figures, run, snapshot=tmp_path / "snapshot", root=tmp_path)


def test_glyph_warning_is_failure_even_with_png(tmp_path):
    figures, run = figures_fixture(tmp_path)
    path = figures / "F01.qa.json"
    data = verify.read_json(path)
    data["glyph_warnings"] = ["Missing Glyph"]
    put(path, data)
    with pytest.raises(ValueError, match="glyph"):
        verify.verify_figures(figures, run, snapshot=tmp_path / "snapshot", root=tmp_path)


def test_manual_visual_record_requires_each_actual_hash_and_both_scales(tmp_path):
    figures, run = figures_fixture(tmp_path)
    result = verify.verify_figures(figures, run, snapshot=tmp_path / "snapshot", root=tmp_path)
    review = {"schema": "kamp.visual_review.v1", "review_method": "direct_image_inspection",
              "reviewer": "fixture-only", "reviewed_at": "fixture", "figures": [
                  {**r, "verdict": "pass", "original_resolution_checked": True, "pdf_insertion_scale_checked": True,
                   "unresolved_defects": [], "notes": "Synthetic test attestation, not an actual inspection"} for r in result["figures"]]}
    path = tmp_path / "visual.json"
    put(path, review)
    assert verify.verify_visual(path, result)["reviewed_images"] == 45
    review["figures"][0]["png_sha256"] = "stale"
    put(path, review)
    with pytest.raises(ValueError, match="old image"):
        verify.verify_visual(path, result)


def test_computational_complete_does_not_imply_full_ok(tmp_path, monkeypatch):
    for name in ("verify_core", "verify_experiments", "verify_quality", "verify_figures"):
        monkeypatch.setattr(verify, name, lambda *a, **k: {"status": "passed"})
    monkeypatch.setattr(code_state, "verify_code_tests", lambda *a, **k: {"status": "passed"})
    monkeypatch.setattr(publisher, "verify_published", lambda *a, **k: {"status": "passed"})
    config = tmp_path / "config.json"
    put(config, {})
    result = verify.verify_submission(tmp_path, tmp_path, tmp_path, tmp_path, config)
    assert result["status"] == "compute_complete_visual_pending"
    assert result["computation_complete"] and not result["full_ok"]
    monkeypatch.setattr(verify, "verify_quality", Mock(side_effect=ValueError("bad mask")))
    result = verify.verify_submission(tmp_path, tmp_path, tmp_path, tmp_path, config)
    assert result["status"] == "failed" and not result["computation_complete"]


def test_pipeline_reuse_failure_prevents_children(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "verify_core", Mock(side_effect=ValueError("stale snapshot")))
    child = Mock()
    monkeypatch.setattr(pipeline, "run_step", child)
    assert pipeline.main(["--reuse-core", "--run-dir", str(tmp_path / "run"), "--snapshot", str(tmp_path / "snap")]) == 1
    child.assert_not_called()


@pytest.mark.parametrize("failure_stage,expected_calls", [("experiments", ["experiments"]), ("figures", ["experiments", "figures"])])
def test_pipeline_stops_on_failed_dependency(tmp_path, monkeypatch, failure_stage, expected_calls):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "verify_core", lambda *a: {"status": "passed"})
    calls = []
    def child(name, command, run_dir, **kwargs):
        calls.append(name)
        return {"step": name, "returncode": 1 if name == failure_stage else 0}
    monkeypatch.setattr(pipeline, "run_step", child)
    assert pipeline.main(["--reuse-core", "--run-dir", str(tmp_path / "run")]) == 1
    assert calls == expected_calls


def test_pipeline_runs_experiments_and_render_even_when_core_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "verify_core", lambda *a: {"status": "passed"})
    calls = []
    def child(name, command, run_dir, **kwargs):
        calls.append((name, command))
        if name == "verification":
            put(Path(run_dir) / "verification_summary.json", {"status": "compute_complete_visual_pending", "computation_complete": True, "full_ok": False})
        return {"step": name, "returncode": 3 if name == "verification" else 0}
    monkeypatch.setattr(pipeline, "run_step", child)
    run = tmp_path / "한글 공백"
    assert pipeline.main(["--profile", "full", "--cpu", "--reuse-core", "--run-dir", str(run)]) == 3
    assert [c[0] for c in calls] == ["experiments", "figures", "publish_figures", "current_code_tests", "verification"]
    assert "--all" in calls[1][1] and "--experiments" in calls[1][1]
    assert verify.read_json(run / "submission_pipeline.json")["full_ok"] is False


def test_paths_must_remain_within_artifact_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        verify.confined(tmp_path, "../outside")


def test_core_source_metadata_cannot_be_mutated_through_checker(core):
    root, snapshot, _, meta = core
    before = copy.deepcopy(meta)
    verify.verify_core(root, snapshot, compare_runtime=False)
    assert meta == before


def quality_fixture(tmp_path):
    import pandas as pd
    config = {"sensitivity": {"atol": 1e-8, "scenarios": [{"id": "C0"}]}}
    summaries = {name: {"status": "complete"} for name in verify.EXPERIMENTS}
    summaries["bootstrap"]["one_day_reference_verified"] = True
    for name in ("controlled", "temporal"):
        summaries[name].update(completed_outer_count=4, failures=[],
                               protocol={"full_protocol_guard": {"matches_registered_full": True}})
    summaries["sensitivity"].update(scenario_ids=["C0"], c0_max_absolute_difference={"pred_avg": 0., "pred_peak": 0.})
    summaries["policy"].update(constraint_audit_passed=True, outer_count=4, partial_observation=True)
    for name, summary in summaries.items():
        put(tmp_path / name / "summary.json", summary)
    audit = {key: [True] for key in ("historical_power_features_unchanged", "headcount_zero_support_unchanged",
                                    "production_daily_total_rule_passed", "shutdown_unchanged")}
    pd.DataFrame(audit).to_csv(tmp_path / "sensitivity/feature_consistency_audit.csv", index=False)
    pd.DataFrame({"passed": [True]}).to_csv(tmp_path / "policy/policy_constraint_audit.csv", index=False)
    pd.DataFrame({"TP": [2], "FN": [1], "FP": [4], "TN": [5], "P": [3], "N": [9], "n_hours": [12]}).to_csv(
        tmp_path / "diagnostics/condition_confusion_rates.csv", index=False)
    pd.DataFrame({"outer": ["O2"] * 3, "model": ["lgb"] * 3, "evaluation_usable": [True, True, False],
                  "y_avg": [1., 2., 999.], "pred_avg": [0., 1., -999.]}).to_csv(
        tmp_path / "temporal/temporal_outer_predictions.csv", index=False)
    pd.DataFrame({"outer": ["O2"], "model": ["lgb"], "n": [2], "TP": [0], "FN": [0], "FP": [1], "TN": [1],
                  "MAE": [1.]}).to_csv(tmp_path / "temporal/temporal_outer_metrics.csv", index=False)
    return config


def test_quality_check_scores_only_usable_rows_and_accepts_partial_observation_disclosure(tmp_path):
    config = quality_fixture(tmp_path)
    result = verify.verify_quality(tmp_path, config)
    assert result["status"] == "passed" and result["policy_partial_observation"] is True


@pytest.mark.parametrize("problem", ["mask", "confusion", "c0", "scientific_protocol"])
def test_quality_flags_cannot_be_hidden_by_complete_summaries(tmp_path, problem):
    import pandas as pd
    config = quality_fixture(tmp_path)
    if problem == "mask":
        path = tmp_path / "temporal/temporal_outer_predictions.csv"
        data = pd.read_csv(path)
        data["evaluation_usable"] = True
        data.to_csv(path, index=False)
    elif problem == "confusion":
        path = tmp_path / "diagnostics/condition_confusion_rates.csv"
        data = pd.read_csv(path)
        data["FN"] = 999
        data.to_csv(path, index=False)
    elif problem == "c0":
        path = tmp_path / "sensitivity/summary.json"
        data = verify.read_json(path)
        data["c0_max_absolute_difference"]["pred_avg"] = .1
        put(path, data)
    else:
        path = tmp_path / "temporal/summary.json"
        data = verify.read_json(path)
        data["protocol"]["full_protocol_guard"]["matches_registered_full"] = False
        put(path, data)
    with pytest.raises(ValueError):
        verify.verify_quality(tmp_path, config)


def test_reference_supports_absent_classifier_tau_and_labels_without_fabrication(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    rows = pd.DataFrame({"model": ["classifier"], "data_condition": ["D1"], "datetime": [pd.Timestamp("2021-09-01")],
                         "y_avg": [1.], "y_peak": [2.], "y_cls": [0], "fold": [2], "theta": [187.],
                         "peak_pred_label": pd.Series([pd.NA], dtype="Int64"), "pred_avg": [np.nan],
                         "pred_peak": [.3], "prob_raw": [.3], "prob_cal": [.2], "tau": [np.nan]})
    data = {"oof_long": rows, "test_long": rows}
    monkeypatch.setattr(verify, "_load_snapshot", lambda _: data)
    put(tmp_path / "metadata.json", {})
    result = verify.verify_reference(tmp_path, tmp_path)
    assert result["status"] == "passed"
    assert result["comparisons"][0]["max_absolute_differences"]["tau"] is None


@pytest.mark.parametrize("change", ["snapshot", "renderer", "stage"])
def test_stale_baseline_figures_cannot_pass_with_current_outputs(tmp_path, change):
    figures, run = figures_fixture(tmp_path)
    path = {"snapshot": tmp_path / "snapshot/metadata.json", "renderer": tmp_path / "tools/figure_layout.py",
            "stage": tmp_path / "src/s00_env.py"}[change]
    if change == "snapshot":
        put(path, {"fingerprint": "new-run"})
    else:
        path.write_text("changed since rendering", encoding="utf-8")
    with pytest.raises(ValueError):
        verify.verify_figures(figures, run, snapshot=tmp_path / "snapshot", root=tmp_path)


def test_core_start_provenance_catches_runner_change_even_if_snapshot_sources_match(core):
    root, snapshot, _, _ = core
    (root / "run_all.py").write_text("# new unverified training path", encoding="utf-8")
    with pytest.raises(ValueError, match="run_all.py"):
        verify.verify_core(root, snapshot, compare_runtime=False)


def test_added_numeric_source_cannot_hide_outside_old_provenance(core):
    root, snapshot, _, _ = core
    extra = root / "src/new_stage.py"
    extra.parent.mkdir()
    extra.write_text("# new numeric stage", encoding="utf-8")
    with pytest.raises(ValueError, match="added"):
        verify.verify_core(root, snapshot, compare_runtime=False)


def test_snapshot_readme_not_in_historical_numeric_manifest_is_current_test_scope(core):
    root, snapshot, _, meta = core
    readme = root / "src/README.md"
    readme.parent.mkdir()
    readme.write_text("Documentation recorded at export", encoding="utf-8")
    meta["source_hashes"]["src/README.md"] = verify.digest(readme)
    # The original run_all provenance never claimed to capture source README.
    assert verify.verify_core(root, snapshot, compare_runtime=False)["status"] == "passed"
    readme.write_text("Later documentation; covered by current-code tests", encoding="utf-8")
    assert verify.verify_core(root, snapshot, compare_runtime=False)["status"] == "passed"
    assert "src/README.md" in code_state.code_hashes(root)


@pytest.mark.parametrize("rc,log,changed", [(1, "4 passed", False), (0, "no tests ran", False),
                                         (0, "4 passed, 1 skipped", False), (0, "4 passed", True)])
def test_recorded_tests_reject_failed_skipped_empty_or_changed_suite(rc, log, changed):
    result = code_state.evaluate_evidence({"a": "old"}, {"a": "new" if changed else "old"}, {"python": "x"}, {"python": "x"}, rc, log)
    assert result["status"] == "failed"


def code_evidence_fixture(tmp_path, monkeypatch):
    (tmp_path / "run_submission.py").write_text("# fixture", encoding="utf-8")
    tests = tmp_path / "tests/test_one.py"
    tests.parent.mkdir()
    tests.write_text("def test_one(): pass", encoding="utf-8")
    log = tmp_path / "outputs/verification/current.pytest.log"
    log.parent.mkdir(parents=True)
    log.write_text("5 passed in 0.01s", encoding="utf-8")
    state = {"python": "fixture"}
    monkeypatch.setattr(code_state, "runtime_state", lambda: state)
    hashes = code_state.code_hashes(tmp_path)
    data = {"schema": "kamp.current_code_tests.v1", "status": "complete", "command": ["python", *code_state.SUITE_ARGS],
            "scope": "synthetic fixture only", "before_code_sha256": hashes, "after_code_sha256": hashes,
            "runtime_before": state, "runtime_after": state, "code_unchanged": True, "runtime_unchanged": True,
            "returncode": 0, "log_file": log.name, "log_sha256": code_state.digest(log)}
    manifest = log.with_name("current.json")
    put(manifest, data)
    return manifest, data


@pytest.mark.parametrize("change", ["new_test", "modified_helper", "test_subset", "log"])
def test_current_code_evidence_cannot_reuse_old_success(tmp_path, monkeypatch, change):
    manifest, data = code_evidence_fixture(tmp_path, monkeypatch)
    assert code_state.verify_code_tests(tmp_path, manifest)["passed_count"] == 5
    if change == "new_test":
        (tmp_path / "tests/test_new.py").write_text("new test", encoding="utf-8")
    elif change == "modified_helper":
        (tmp_path / "run_submission.py").write_text("changed helper", encoding="utf-8")
    elif change == "test_subset":
        data["command"].extend(["-k", "one"])
        put(manifest, data)
    else:
        manifest.with_name(data["log_file"]).write_text("6 passed forged", encoding="utf-8")
    with pytest.raises(ValueError):
        code_state.verify_code_tests(tmp_path, manifest)


def test_recorded_command_and_log_do_not_include_host_account(tmp_path, monkeypatch):
    secret_name = "fictional-private-account"
    fake_executable = "C:/Users/" + secret_name + "/Python/python.exe"
    monkeypatch.setattr(code_state.sys, "executable", fake_executable)
    monkeypatch.setattr(code_state, "runtime_state", lambda: {"python": "fixture"})
    class FakeChild:
        def __init__(self, command, **kwargs):
            assert command[0] == fake_executable
            kwargs["stdout"].write((fake_executable + ": fixture warning\n5 passed in 0.1s\n").encode())
        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(code_state.subprocess, "Popen", FakeChild)
    manifest = tmp_path / "outputs/verification/current.json"
    result = code_state.record_code_tests(tmp_path, manifest)
    assert result["status"] == "complete" and result["command"] == ["python", *code_state.SUITE_ARGS]
    assert result["executable_basename"] == "python.exe"
    assert secret_name not in manifest.read_text(encoding="utf-8")
    assert secret_name not in manifest.with_suffix(".pytest.log").read_text(encoding="utf-8")


def test_only_reproduction_constraints_join_verification_code_hashes(tmp_path):
    directory = tmp_path / "verification"
    directory.mkdir()
    (directory / "reproduction_constraints.txt").write_text("numpy==1.26.4", encoding="utf-8")
    (directory / "environment.json").write_text("generated", encoding="utf-8")
    assert set(code_state.code_hashes(tmp_path)) == {"verification/reproduction_constraints.txt"}


def test_noninserted_figures_are_not_forced_to_claim_pdf_scale_review(tmp_path):
    figures = {"figures": [{"fid": fid, "png_sha256": fid} for fid in verify.FIGURE_IDS]}
    rows = [{"fid": fid, "png_sha256": fid, "verdict": "pass", "original_resolution_checked": True,
             "pdf_insertion_applicability": "not_inserted", "pdf_insertion_scale_checked": False,
             "non_insertion_reason": "Included as standalone PNG only", "unresolved_defects": [], "notes": "Fixture only"}
            for fid in verify.FIGURE_IDS]
    path = tmp_path / "manual.json"
    put(path, {"schema": "kamp.visual_review.v1", "review_method": "direct_image_inspection",
               "reviewer": "fixture", "reviewed_at": "fixture", "figures": rows})
    assert verify.verify_visual(path, figures)["status"] == "passed"
    rows[0]["non_insertion_reason"] = ""
    data = verify.read_json(path)
    data["figures"] = rows
    put(path, data)
    with pytest.raises(ValueError, match="Non-insertion"):
        verify.verify_visual(path, figures)


def test_actual_report_variant_hash_must_have_direct_review(tmp_path):
    figures = {"figures": [{"fid": fid, "png_sha256": fid, "report_variant": {"sha256": fid + "-inserted"}}
                           for fid in verify.FIGURE_IDS]}
    rows = [{"fid": fid, "png_sha256": fid, "verdict": "pass", "original_resolution_checked": True,
             "pdf_insertion_scale_checked": True, "unresolved_defects": [], "notes": "Fixture only"}
            for fid in verify.FIGURE_IDS]
    path = tmp_path / "manual.json"
    data = {"schema": "kamp.visual_review.v1", "review_method": "direct_image_inspection",
            "reviewer": "fixture", "reviewed_at": "fixture", "figures": rows}
    put(path, data)
    with pytest.raises(ValueError, match="actual inserted"):
        verify.verify_visual(path, figures)
    for row in rows:
        row["report_variant_png_sha256"] = row["fid"] + "-inserted"
    put(path, data)
    assert verify.verify_visual(path, figures)["status"] == "passed"
