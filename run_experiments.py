"""Run preregistered supplemental experiments in isolated sequential processes."""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "bootstrap": "bootstrap",
    "diagnostics": "diagnostics",
    "sensitivity": "sensitivity",
    "controlled": "controlled_regime",
    "temporal": "temporal_selection",
    "policy": "policy",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def runtime_versions():
    result = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "optuna"):
        result[name] = metadata.version(name)
    return result


def fingerprint(name, config, snapshot):
    files = list((ROOT / "experiments").glob("*.py")) + [ROOT / "run_experiments.py"]
    files += [ROOT / "data/okm_augumented_2021.csv", ROOT / "outputs/tables/F18_src.csv"]
    if name != "bootstrap":
        from tools.analysis_snapshot import load_snapshot
        load_snapshot(snapshot, verify=True)
        files += [snapshot / "metadata.json"]
        files += list((ROOT / "src").glob("*.py"))
        files += [ROOT / "tools/analysis_snapshot.py"]
    if name == "sensitivity":
        files += [p for p in (ROOT / "outputs/models/full/eval").rglob("*") if p.is_file()]
        files += list((ROOT / "serving").rglob("*.py"))
    source = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: digest(p) for p in files}
    payload = {"experiment": name, "config": config, "source": source, "runtime": runtime_versions()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(), payload


def complete_manifest(folder, expected_fingerprint=None):
    folder = Path(folder).resolve()
    try:
        data = json.loads((folder / "executor_manifest.json").read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        if data.get("status") != "complete" or (expected_fingerprint and data.get("fingerprint") != expected_fingerprint):
            return False
        hashes = data.get("files_sha256", {})
        if not isinstance(hashes, dict) or not hashes or "summary.json" not in hashes:
            return False
        return all((folder / name).resolve().is_relative_to(folder)
                   and (folder / name).is_file() and digest(folder / name) == value
                   for name, value in hashes.items())
    except (OSError, ValueError, TypeError):
        return False


def child_environment():
    env = os.environ.copy()
    env.update(PYTHONHASHSEED="42", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               MPLBACKEND="Agg", TF_ENABLE_ONEDNN_OPTS="0", TF_DETERMINISTIC_OPS="1")
    return env


def worker(name, snapshot, run_dir, config):
    from experiments.common import _json_safe
    output = run_dir / name
    output.mkdir(parents=True, exist_ok=True)
    key, inputs = fingerprint(name, config, snapshot)
    started = time.time()
    manifest_path = output / "executor_manifest.json"
    write_json(manifest_path, {"status": "running", "experiment": name, "fingerprint": key, "started_epoch": started})
    try:
        if name == "policy":
            temporal_key, _ = fingerprint("temporal", config, snapshot)
            if not complete_manifest(run_dir / "temporal", temporal_key):
                raise ValueError("Policy requires complete, current, hash-verified temporal experiment outputs")
        config = dict(config)
        config["_run_context"] = {"run_dir": str(run_dir), "temporal_dir": str(run_dir / "temporal")}
        module = importlib.import_module("experiments." + EXPERIMENTS[name])
        summary = _json_safe(module.run(ROOT, snapshot, output, config))
        status = "complete" if summary.get("status") == "complete" else "partial"
        files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p != manifest_path and p.suffix not in {".tmp", ".pyc", ".log"}
                 and ".cache" not in p.parts and "__pycache__" not in p.parts}
        if status == "complete":
            if "summary.json" not in files:
                raise ValueError("Complete experiment did not persist summary.json")
            persisted = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            if not isinstance(persisted, dict) or persisted.get("status") != "complete":
                raise ValueError("Persisted experiment summary does not confirm completion")
        write_json(manifest_path, {"status": status, "experiment": name, "fingerprint": key,
                                  "elapsed_seconds": time.time() - started, "inputs": inputs,
                                  "summary": summary, "files_sha256": files})
        print(f"EXPERIMENT_{status.upper()} {name} seconds={time.time()-started:.2f}", flush=True)
        return 0 if status == "complete" else 3
    except BaseException as exc:
        write_json(manifest_path, {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                  "experiment": name, "fingerprint": key, "elapsed_seconds": time.time()-started,
                                  "error_type": type(exc).__name__, "error": str(exc).replace(str(ROOT), "<project>")})
        traceback.print_exc()
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


def run_child(name, args, run_dir, config):
    output = run_dir / name
    key, _ = fingerprint(name, config, args.snapshot)
    # Policy is cheap and depends on freshly produced temporal evidence. Always
    # rerun its strict producer checks instead of reusing a detached policy cache.
    if name != "policy" and args.resume and complete_manifest(output, key):
        print(f"VERIFIED_CACHE_HIT {name}", flush=True)
        return 0
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()), "--_worker", name,
               "--config", str(args.config), "--snapshot", str(args.snapshot), "--run-dir", str(run_dir)]
    started = time.monotonic()
    with (log_dir / f"{name}.log").open("wb") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while proc.poll() is None:
                time.sleep(5)
                if int(time.monotonic()-started) % 30 < 5:
                    print(f"EXPERIMENT_HEARTBEAT {name} elapsed={int(time.monotonic()-started)}s", flush=True)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait(timeout=30)
            raise
    print(f"EXPERIMENT_EXIT {name} rc={proc.returncode} elapsed={time.monotonic()-started:.1f}s", flush=True)
    return proc.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/config.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/analysis_snapshot")
    parser.add_argument("--experiment", choices=["all", *EXPERIMENTS], default="all")
    parser.add_argument("--run-dir", "--output-dir", dest="run_dir", type=Path, default=ROOT / "outputs/additional/validated_run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--_worker", choices=EXPERIMENTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.config, args.snapshot, args.run_dir = args.config.resolve(), args.snapshot.resolve(), args.run_dir.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args._worker:
        return worker(args._worker, args.snapshot, args.run_dir, config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "protocol.json", config)
    selected = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    states = {}
    for name in selected:
        rc = run_child(name, args, args.run_dir, config)
        states[name] = "complete" if rc == 0 else "failed_or_partial"
        write_json(args.run_dir / "run_status.json", {"status": "running", "requested": selected, "experiments": states})
        if rc != 0:
            write_json(args.run_dir / "run_status.json", {"status": "failed_or_partial", "requested": selected, "experiments": states})
            return rc
    all_verified = all(complete_manifest(args.run_dir / name)
                       and complete_manifest(args.run_dir / name, fingerprint(name, config, args.snapshot)[0])
                       for name in EXPERIMENTS)
    status = "complete" if all_verified else "partial"
    write_json(args.run_dir / "run_status.json", {"status": status, "requested": selected, "experiments": states,
                                                  "all_six_verified": all_verified, "runtime": runtime_versions()})
    print(f"ADDITIONAL_{status.upper()}", flush=True)
    return 0 if all_verified or args.experiment != "all" else 3


if __name__ == "__main__":
    raise SystemExit(main())
