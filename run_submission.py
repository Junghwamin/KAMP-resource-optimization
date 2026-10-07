"""Portable full workflow, with fail-closed computational and visual gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.verify_submission import read_json, verify_core, write_json

ROOT = Path(__file__).resolve().parent


def run_step(name, command, run_dir, *, cpu=False):
    log_dir = Path(run_dir) / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED="42", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", TF_ENABLE_ONEDNN_OPTS="0", TF_DETERMINISTIC_OPS="1")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    with (log_dir / f"{name}.log").open("wb") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while True:
                try:
                    code = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f"SUBMISSION_HEARTBEAT {name} seconds={time.monotonic()-started:.0f}", flush=True)
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    return {"step": name, "returncode": code, "seconds": time.monotonic()-started,
            "log": f"pipeline_logs/{name}.log"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["full"], default="full")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reuse-core", action="store_true", help="Reuse only a completed, hash/source/runtime-verified core snapshot")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/analysis_snapshot")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/additional/validated_run")
    parser.add_argument("--figures", type=Path, default=ROOT / "outputs/figures_rerendered")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/config.json")
    parser.add_argument("--visual-review", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    for key in ("snapshot", "run_dir", "figures", "config", "visual_review", "reference"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "submission_pipeline.json"
    state = {"schema": "kamp.submission_pipeline.v1", "status": "running", "profile": "full",
             "started_utc": datetime.now(timezone.utc).isoformat(), "reused_core": args.reuse_core,
             "scope": "local_code_results_and_figures", "steps": [], "full_ok": False}
    write_json(status_path, state)
    python = [sys.executable, "-u", "-X", "utf8"]
    try:
        if args.reuse_core:
            state["core_reuse_validation"] = verify_core(ROOT, args.snapshot)
            state["steps"].append({"step": "core", "status": "verified_reused", "returncode": 0})
            write_json(status_path, state)
        else:
            require_new = not args.snapshot.exists() or (args.snapshot.is_dir() and not any(args.snapshot.iterdir()))
            if not require_new:
                raise ValueError("A new core export needs a new/empty snapshot directory; use --reuse-core only for verified existing exports")
            command = python + [str(ROOT / "run_all.py"), "--force", "--export-analysis", str(args.snapshot)]
            if args.cpu:
                command.append("--cpu")
            result = run_step("core", command, args.run_dir, cpu=args.cpu)
            state["steps"].append(result)
            write_json(status_path, state)
            if result["returncode"] != 0:
                raise RuntimeError(f"Core failed with exit {result['returncode']}")
            state["core_validation"] = verify_core(ROOT, args.snapshot)
        commands = [
            ("experiments", python + [str(ROOT / "run_experiments.py"), "--experiment", "all", "--resume",
                 "--snapshot", str(args.snapshot), "--run-dir", str(args.run_dir), "--config", str(args.config)]),
            ("figures", python + [str(ROOT / "tools/render_figures.py"), "--all", "--snapshot", str(args.snapshot),
                 "--experiments", str(args.run_dir), "--output", str(args.figures)]),
            ("publish_figures", python + [str(ROOT / "tools/publish_figures.py"), "--rendered", str(args.figures),
                 "--snapshot", str(args.snapshot)]),
            ("current_code_tests", python + [str(ROOT / "tools/verify_code_state.py"), "--record-code-tests"]),
            ("verification", python + [str(ROOT / "tools/verify_submission.py"), "--mode", "strict",
                 "--snapshot", str(args.snapshot), "--run-dir", str(args.run_dir), "--figures", str(args.figures),
                 "--config", str(args.config)]),
        ]
        if args.visual_review:
            commands[-1][1].extend(["--visual-review", str(args.visual_review)])
        if args.reference:
            commands[-1][1].extend(["--reference", str(args.reference)])
        for name, command in commands:
            result = run_step(name, command, args.run_dir, cpu=args.cpu)
            state["steps"].append(result)
            write_json(status_path, state)
            if result["returncode"] != 0:
                if name == "verification" and result["returncode"] == 3:
                    review = read_json(args.run_dir / "verification_summary.json")
                    if review.get("status") == "compute_complete_visual_pending" and review.get("computation_complete") is True:
                        state["status"] = "compute_complete_visual_pending"
                        write_json(status_path, state)
                        print("COMPUTE_OK_VISUAL_PENDING scope=local_code_results_and_figures")
                        return 3
                raise RuntimeError(f"{name} failed with exit {result['returncode']}")
        review = read_json(args.run_dir / "verification_summary.json")
        if review.get("status") != "complete" or review.get("full_ok") is not True:
            raise ValueError("Verifier did not attest completed local checks")
        state.update(status="complete", full_ok=True)
        write_json(status_path, state)
        print("FULL_OK scope=local_code_results_and_figures")
        return 0
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error={"type": type(exc).__name__, "message": str(exc).replace(str(ROOT), "<project>")})
        write_json(status_path, state)
        print(f"SUBMISSION_{state['status'].upper()}: {state['error']['message']}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
