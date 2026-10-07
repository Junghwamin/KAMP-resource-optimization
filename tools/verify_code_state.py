"""Run the complete current pytest suite and bind evidence to unchanged code.

This tool cannot merely assert a pass or choose a test subset. Its manifest is
separate from the immutable historical core-run provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "outputs/verification/current_code_test_manifest.json"
SUITE_ARGS = ["-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"]
SUFFIXES = {".py", ".sh", ".ps1", ".txt", ".ini", ".json", ".md", ".toml", ".yaml", ".yml"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes(root):
    root = Path(root)
    files = [p for p in root.iterdir() if p.is_file() and p.suffix in SUFFIXES]
    for folder in ("src", "tools", "tests", "experiments", "serving"):
        files.extend(p for p in (root / folder).rglob("*") if p.is_file() and p.suffix in SUFFIXES
                     and "__pycache__" not in p.parts and ".cache" not in p.parts)
    constraints = root / "verification/reproduction_constraints.txt"
    if constraints.is_file():
        files.append(constraints)
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(set(files))}


def runtime_state():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "optuna", "matplotlib",
                 "shap", "tensorflow", "keras", "pytest"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {"python": platform.python_version(), "platform": platform.system(), "packages": packages}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def evaluate_evidence(before, after, runtime_before, runtime_after, returncode, log):
    passed = re.search(r"\b([1-9]\d*) passed\b", log)
    skipped = re.search(r"\b([1-9]\d*) skipped\b", log)
    success = returncode == 0 and passed is not None and skipped is None and before == after and runtime_before == runtime_after
    return {"status": "complete" if success else "failed", "returncode": returncode,
            "passed_count": int(passed.group(1)) if passed else 0, "skipped_count": int(skipped.group(1)) if skipped else 0,
            "code_unchanged": before == after, "runtime_unchanged": runtime_before == runtime_after}


def sanitize_log(text):
    from run_all import redact
    return redact(text)


def record_code_tests(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".pytest.log")
    before, runtime_before = code_hashes(root), runtime_state()
    command = [sys.executable, *SUITE_ARGS]
    state = {"schema": "kamp.current_code_tests.v1", "status": "running",
             "started_utc": datetime.now(timezone.utc).isoformat(), "command": ["python", *SUITE_ARGS],
             "executable_basename": Path(sys.executable).name,
             "scope": "all tests including supplemental experiments and delivery tools",
             "before_code_sha256": before, "runtime_before": runtime_before, "log_file": log_path.name}
    save(output, state)
    env = dict(os.environ, PYTHONHASHSEED="42", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               MPLBACKEND="Agg", TF_ENABLE_ONEDNN_OPTS="0", TF_DETERMINISTIC_OPS="1")
    started = time.monotonic()
    try:
        with log_path.open("wb") as log:
            proc = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                while True:
                    try:
                        rc = proc.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        print(f"CODE_TEST_HEARTBEAT seconds={time.monotonic()-started:.0f}", flush=True)
            except BaseException:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
        text = sanitize_log(log_path.read_text(encoding="utf-8", errors="replace"))
        log_path.write_text(text, encoding="utf-8")
        after, runtime_after = code_hashes(root), runtime_state()
        state.update(evaluate_evidence(before, after, runtime_before, runtime_after, rc, text))
        state.update(after_code_sha256=after, runtime_after=runtime_after, log_sha256=digest(log_path),
                     finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started)
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error={"type": type(exc).__name__, "message": sanitize_log(str(exc).replace(str(root), "<project>"))})
        if log_path.is_file():
            log_path.write_text(sanitize_log(log_path.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
            state["log_sha256"] = digest(log_path)
    save(output, state)
    return state


def verify_code_tests(root, manifest_path):
    root, manifest_path = Path(root).resolve(), Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema") != "kamp.current_code_tests.v1" or data.get("status") != "complete":
        raise ValueError("Current-code full-suite test evidence is missing, incomplete, or failed")
    if data.get("command", [])[1:] != SUITE_ARGS:
        raise ValueError("Test evidence did not execute the required complete suite command")
    current = code_hashes(root)
    if data.get("before_code_sha256") != current or data.get("after_code_sha256") != current:
        raise ValueError("Code/tests/docs changed since the full-suite execution")
    if data.get("runtime_before") != runtime_state() or data.get("runtime_after") != runtime_state():
        raise ValueError("Runtime changed since the full-suite execution")
    log_path = (manifest_path.parent / data["log_file"]).resolve()
    if log_path.parent != manifest_path.parent.resolve() or not log_path.is_file() or digest(log_path) != data.get("log_sha256"):
        raise ValueError("Current-code pytest log is missing or changed")
    result = evaluate_evidence(current, current, data["runtime_before"], data["runtime_after"],
                               data.get("returncode"), log_path.read_text(encoding="utf-8", errors="replace"))
    if result["status"] != "complete" or not data.get("code_unchanged") or not data.get("runtime_unchanged"):
        raise ValueError("Full-suite execution did not pass without skipped tests or changed code")
    return {"status": "passed", "passed_count": result["passed_count"], "file_count": len(current),
            "manifest_sha256": digest(manifest_path), "scope": data["scope"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-code-tests", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = record_code_tests(ROOT, args.output)
    print("CURRENT_CODE_TESTS_" + result["status"].upper())
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
