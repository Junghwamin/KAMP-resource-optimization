"""Experiment-runner integrity and dependency boundaries, without model fits."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_experiments as runner


def verified_cache(folder: Path, key="current", status="complete"):
    folder.mkdir(parents=True, exist_ok=True)
    runner.write_json(folder / "summary.json", {"status": status, "fixture": True})
    (folder / "values.csv").write_text("hour,value\n0,3.5\n", encoding="utf-8")
    manifest = {"status": status, "fingerprint": key,
                "files_sha256": {name: runner.digest(folder / name) for name in ("summary.json", "values.csv")}}
    runner.write_json(folder / "executor_manifest.json", manifest)
    return manifest


@pytest.mark.parametrize("status", ["running", "partial", "failed", "interrupted"])
def test_cache_rejects_noncomplete_status(tmp_path, status):
    verified_cache(tmp_path, status=status)
    assert runner.complete_manifest(tmp_path, "current") is False


@pytest.mark.parametrize("payload", [None, [], "complete", 42,
    {"status": "complete", "files_sha256": {}},
    {"status": "complete", "files_sha256": []},
    {"status": "complete", "files_sha256": ["summary.json"]},
    {"status": "complete", "files_sha256": "summary.json"}])
def test_empty_or_malformed_cache_is_a_miss_not_a_crash(tmp_path, payload):
    runner.write_json(tmp_path / "executor_manifest.json", payload)
    assert runner.complete_manifest(tmp_path) is False


def test_absent_or_broken_manifest_is_cache_miss(tmp_path):
    assert runner.complete_manifest(tmp_path) is False
    (tmp_path / "executor_manifest.json").write_text("{broken", encoding="utf-8")
    assert runner.complete_manifest(tmp_path) is False


def test_cache_checks_fingerprint_each_artifact_and_summary_presence(tmp_path):
    manifest = verified_cache(tmp_path)
    assert runner.complete_manifest(tmp_path, "current") is True
    assert runner.complete_manifest(tmp_path, "old-config") is False
    (tmp_path / "values.csv").write_text("hour,value\n0,999\n", encoding="utf-8")
    assert runner.complete_manifest(tmp_path, "current") is False
    manifest = verified_cache(tmp_path)
    (tmp_path / "values.csv").unlink()
    assert runner.complete_manifest(tmp_path, "current") is False
    manifest = verified_cache(tmp_path)
    del manifest["files_sha256"]["summary.json"]
    runner.write_json(tmp_path / "executor_manifest.json", manifest)
    assert runner.complete_manifest(tmp_path, "current") is False


@pytest.mark.parametrize("path_kind", ["parent", "absolute"])
def test_cache_rejects_outside_artifacts_even_when_hash_matches(tmp_path, path_kind):
    folder = tmp_path / "experiment"
    outside = tmp_path / "outside.csv"
    outside.write_text("not an experiment artifact", encoding="utf-8")
    manifest = verified_cache(folder)
    name = "../outside.csv" if path_kind == "parent" else str(outside.resolve())
    manifest["files_sha256"][name] = runner.digest(outside)
    runner.write_json(folder / "executor_manifest.json", manifest)
    assert runner.complete_manifest(folder, "current") is False


def test_fingerprint_changes_for_config_code_data_and_runtime(tmp_path, monkeypatch):
    root = tmp_path / "project"
    for name, text in {"experiments/bootstrap.py": "pass\n", "run_experiments.py": "# runner\n",
                       "data/okm_augumented_2021.csv": "hour,power\n0,1\n",
                       "outputs/tables/F18_src.csv": "hour,error\n0,.1\n"}.items():
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "runtime_versions", lambda: {"python": "3.fixture", "numpy": "1.fixture"})
    first, evidence = runner.fingerprint("bootstrap", {"seed": 42}, root / "no_snapshot_required")
    repeated, _ = runner.fingerprint("bootstrap", {"seed": 42}, root / "no_snapshot_required")
    changed_config, _ = runner.fingerprint("bootstrap", {"seed": 43}, root / "no_snapshot_required")
    assert first == repeated
    assert changed_config != first
    assert "data/okm_augumented_2021.csv" in evidence["source"] or "data\\okm_augumented_2021.csv" in evidence["source"]
    (root / "experiments/bootstrap.py").write_text("answer = 1\n", encoding="utf-8")
    changed_code, _ = runner.fingerprint("bootstrap", {"seed": 42}, root / "no_snapshot_required")
    assert changed_code != first
    (root / "data/okm_augumented_2021.csv").write_text("hour,power\n0,2\n", encoding="utf-8")
    changed_data, _ = runner.fingerprint("bootstrap", {"seed": 42}, root / "no_snapshot_required")
    assert changed_data != changed_code
    monkeypatch.setattr(runner, "runtime_versions", lambda: {"python": "3.fixture", "numpy": "2.fixture"})
    changed_runtime, _ = runner.fingerprint("bootstrap", {"seed": 42}, root / "no_snapshot_required")
    assert changed_runtime != changed_data


def fake_fingerprint(name, config, snapshot):
    inputs = {"experiment": name, "config": config, "snapshot": str(snapshot)}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(), inputs


@pytest.mark.parametrize("dependency", ["absent", "partial", "wrong_config", "modified_artifact"])
def test_policy_dependency_failure_is_persisted_before_module_import(tmp_path, monkeypatch, dependency):
    config = {"seed": 42, "temporal_selection": {"trials_reg": 50}}
    snapshot = tmp_path / "snapshot"
    key, _ = fake_fingerprint("temporal", config, snapshot)
    if dependency != "absent":
        if dependency == "wrong_config":
            key, _ = fake_fingerprint("temporal", {**config, "seed": 999}, snapshot)
        verified_cache(tmp_path / "temporal", key, "partial" if dependency == "partial" else "complete")
    if dependency == "modified_artifact":
        (tmp_path / "temporal/values.csv").write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(runner, "fingerprint", fake_fingerprint)
    imports = []
    monkeypatch.setattr(runner, "importlib", SimpleNamespace(import_module=lambda name: imports.append(name)))
    rc = runner.worker("policy", snapshot, tmp_path, config)
    manifest = json.loads((tmp_path / "policy/executor_manifest.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "ValueError"
    assert "current" in manifest["error"]
    assert imports == []


def test_worker_validates_current_dependency_injects_context_and_hashes_outputs(tmp_path, monkeypatch):
    config = {"seed": 42, "temporal_selection": {"trials_reg": 50}}
    original = copy.deepcopy(config)
    snapshot = tmp_path / "snapshot"
    key, _ = fake_fingerprint("temporal", config, snapshot)
    verified_cache(tmp_path / "temporal", key)
    monkeypatch.setattr(runner, "fingerprint", fake_fingerprint)
    received = {}

    def run(root, supplied_snapshot, output, supplied_config):
        received.update(root=root, snapshot=supplied_snapshot, config=supplied_config)
        summary = {"status": "complete", "experiment": "policy", "checked": True}
        runner.write_json(output / "summary.json", summary)
        (output / "policy.csv").write_text("policy,count\nnone,0\n", encoding="utf-8")
        (output / "ignore.tmp").write_text("incomplete temporary bytes", encoding="utf-8")
        return summary

    monkeypatch.setattr(runner, "importlib", SimpleNamespace(import_module=lambda name: SimpleNamespace(run=run)))
    assert runner.worker("policy", snapshot, tmp_path, config) == 0
    assert config == original
    assert received["snapshot"] == snapshot
    assert received["config"]["_run_context"]["temporal_dir"] == str(tmp_path / "temporal")
    policy_key, _ = fake_fingerprint("policy", config, snapshot)
    assert runner.complete_manifest(tmp_path / "policy", policy_key)
    manifest = json.loads((tmp_path / "policy/executor_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files_sha256"]) == {"summary.json", "policy.csv"}


def test_worker_cannot_report_complete_without_required_summary_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "fingerprint", fake_fingerprint)
    monkeypatch.setattr(runner, "importlib", SimpleNamespace(import_module=lambda name: SimpleNamespace(run=lambda *args: {"status": "complete"})))
    rc = runner.worker("bootstrap", tmp_path / "snapshot", tmp_path, {})
    manifest = json.loads((tmp_path / "bootstrap/executor_manifest.json").read_text(encoding="utf-8"))
    assert rc != 0
    assert manifest["status"] != "complete"
    assert not runner.complete_manifest(tmp_path / "bootstrap")


@pytest.mark.parametrize("persisted_summary", [{"status": "partial"}, []])
def test_worker_rejects_returned_complete_when_persisted_summary_disagrees(tmp_path, monkeypatch, persisted_summary):
    monkeypatch.setattr(runner, "fingerprint", fake_fingerprint)

    def run(root, snapshot, output, config):
        runner.write_json(output / "summary.json", persisted_summary)
        return {"status": "complete"}

    monkeypatch.setattr(runner, "importlib", SimpleNamespace(import_module=lambda name: SimpleNamespace(run=run)))
    assert runner.worker("bootstrap", tmp_path / "snapshot", tmp_path, {}) == 1
    manifest = json.loads((tmp_path / "bootstrap/executor_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not runner.complete_manifest(tmp_path / "bootstrap")


@pytest.mark.parametrize("returned_status", ["partial", "failed"])
def test_worker_partial_or_failed_module_is_never_completed(tmp_path, monkeypatch, returned_status):
    monkeypatch.setattr(runner, "fingerprint", fake_fingerprint)

    def run(root, snapshot, output, config):
        summary = {"status": returned_status}
        runner.write_json(output / "summary.json", summary)
        return summary

    monkeypatch.setattr(runner, "importlib", SimpleNamespace(import_module=lambda name: SimpleNamespace(run=run)))
    assert runner.worker("bootstrap", tmp_path / "snapshot", tmp_path, {}) != 0
    assert not runner.complete_manifest(tmp_path / "bootstrap")


@pytest.mark.parametrize("dependency_complete", [False, True])
def test_policy_resume_always_reenters_worker_dependency_gate(tmp_path, monkeypatch, dependency_complete):
    config = {"seed": 42}
    snapshot = tmp_path / "snapshot"
    policy_key, _ = fake_fingerprint("policy", config, snapshot)
    temporal_key, _ = fake_fingerprint("temporal", config, snapshot)
    verified_cache(tmp_path / "policy", policy_key)
    verified_cache(tmp_path / "temporal", temporal_key, "complete" if dependency_complete else "partial")
    args = SimpleNamespace(resume=True, snapshot=snapshot, config=tmp_path / "config.json")
    monkeypatch.setattr(runner, "fingerprint", fake_fingerprint)
    launches = []

    def popen(command, **kwargs):
        launches.append((command, kwargs))
        rc = 0 if dependency_complete else 1
        return SimpleNamespace(returncode=rc, poll=lambda: rc)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    assert runner.run_child("policy", args, tmp_path, config) == (0 if dependency_complete else 1)
    assert len(launches) == 1
    command, kwargs = launches[0]
    assert command[command.index("--_worker") + 1] == "policy"
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kwargs["env"]["MPLBACKEND"] == "Agg"


def test_verified_bootstrap_resume_avoids_subprocess(tmp_path, monkeypatch):
    config = {"seed": 42}
    snapshot = tmp_path / "snapshot"
    key, _ = fake_fingerprint("bootstrap", config, snapshot)
    verified_cache(tmp_path / "bootstrap", key)
    args = SimpleNamespace(resume=True, snapshot=snapshot, config=tmp_path / "config.json")
    monkeypatch.setattr(runner, "fingerprint", fake_fingerprint)

    def no_process(*args, **kwargs):
        pytest.fail("A verified matching bootstrap cache should be used")

    monkeypatch.setattr(runner.subprocess, "Popen", no_process)
    assert runner.run_child("bootstrap", args, tmp_path, config) == 0


def test_main_bootstrap_without_snapshot_finishes_requested_stage_and_reports_partial(tmp_path, monkeypatch, capsys):
    """A successful E2 must not fingerprint snapshot-dependent, unrun stages."""
    root = tmp_path / "project"
    for name, content in {"experiments/bootstrap.py": "pass\n", "run_experiments.py": "# runner\n",
                          "data/okm_augumented_2021.csv": "hour,power\n0,1\n",
                          "outputs/tables/F18_src.csv": "hour,error\n0,.1\n"}.items():
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
    config_path = root / "experiments/config.json"
    runner.write_json(config_path, {"seed": 42})
    snapshot = root / "outputs/not_yet_exported_snapshot"
    run_dir = root / "outputs/additional/fixture"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "runtime_versions", lambda: {"python": "3.fixture"})
    requested = []

    def completed_stage(name, args, output, config):
        requested.append(name)
        # Use the real fingerprint: accidentally fingerprinting diagnostics
        # would attempt to validate the nonexistent snapshot and fail.
        key, _ = runner.fingerprint(name, config, args.snapshot)
        verified_cache(output / name, key)
        return 0

    monkeypatch.setattr(runner, "run_child", completed_stage)
    rc = runner.main(["--experiment", "bootstrap", "--config", str(config_path),
                      "--snapshot", str(snapshot), "--run-dir", str(run_dir)])
    assert rc == 0
    assert requested == ["bootstrap"]
    assert not snapshot.exists()
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "partial"
    assert status["all_six_verified"] is False
    assert status["experiments"] == {"bootstrap": "complete"}
    assert "ADDITIONAL_PARTIAL" in capsys.readouterr().out
