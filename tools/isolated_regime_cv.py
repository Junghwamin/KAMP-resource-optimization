"""Run one regime ablation in a fresh process; cache only verified numeric predictions.

No stage module is imported, no pickle is read, and no trained model crosses the
process boundary. KAMP_WORK_DIR overrides the default package .cache directory.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GLOBALS = {"LGB_BASE", "N_ESTIMATORS", "REGIME_CUT", "FAST"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json.tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _frame_hash(frame):
    digest = hashlib.sha256(_json_bytes([list(frame.columns), [str(x) for x in frame.dtypes]]))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _pack_folds(folds):
    arrays, metadata = {}, []
    for i, (fold, Xtr, ytr, Xva) in enumerate(folds):
        entry = {"fold": int(fold), "frames": {}}
        for name, frame in (("Xtr", Xtr), ("ytr", ytr), ("Xva", Xva)):
            key = f"{i}_{name}"
            arrays[key] = frame.to_numpy()
            if arrays[key].dtype.hasobject:
                raise TypeError("Isolated regime CV accepts numeric frames only")
            arrays[key + "_index"] = frame.index.asi8
            entry["frames"][name] = {"columns": list(frame.columns), "dtypes": [str(x) for x in frame.dtypes]}
        metadata.append(entry)
    return arrays, metadata


def _load_result(payload, manifest, key, folds):
    """Incomplete, stale, corrupt or shape-incompatible checkpoints are cache misses."""
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("status") != "complete" or meta.get("fingerprint") != key or meta["payload_sha256"] != _sha(payload):
            return None
        results = []
        with np.load(payload, allow_pickle=False) as data:
            for i, (_, _, _, Xva) in enumerate(folds):
                result = {name: data[f"{i}_{name}"].copy() for name in ("pred_avg", "pred_peak")}
                if any(value.shape != (len(Xva),) or not np.isfinite(value).all() for value in result.values()):
                    return None
                results.append(result)
            seconds = data["fit_seconds"].astype(float).tolist()
        if len(seconds) != len(folds) or not np.isfinite(seconds).all() or min(seconds) < 0:
            return None
        return results, seconds
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return None


def isolated_regime_predictions(variant, folds, config, *, n_classes, gate_features=None,
                                frame=None, condition="D1", root=ROOT, retries=1):
    """Return per-fold predictions and original fit seconds; retry identical inputs once.

    The cache key covers all source files, this worker, raw data, runtime/binary,
    exact fold arrays and dtypes, selected features, the feature frame and parameters.
    Cache hits reuse historical fit time and explicitly report ``cache_hit=True``.
    """
    import lightgbm.libpath

    if set(config) != GLOBALS or n_classes not in (2, 3) or not folds:
        raise ValueError("Invalid isolated regime configuration or empty folds")
    root = Path(root).resolve()
    work = Path(os.environ.get("KAMP_WORK_DIR", root / ".cache")).expanduser().resolve() / "regime_cv"
    work.mkdir(parents=True, exist_ok=True)
    arrays, fold_meta = _pack_folds(folds)
    identity = {
        "schema": 1, "variant": variant, "condition": condition, "config": config,
        "n_classes": n_classes, "gate_features": gate_features, "folds": fold_meta,
        "frame": _frame_hash(frame) if frame is not None else None,
        "source": {p.relative_to(root).as_posix(): _sha(p) for p in sorted((root / "src").glob("*.py"))},
        "worker": _sha(__file__),
        "data": {p.name: _sha(p) for p in sorted((root / "data").glob("*.csv"))},
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm")},
                    "lightgbm_binary": [_sha(p) for p in lightgbm.libpath.find_lib_path()]},
    }
    digest = hashlib.sha256(_json_bytes(identity))
    for name, value in sorted(arrays.items()):
        digest.update(_json_bytes([name, str(value.dtype), list(value.shape)]))
        digest.update(np.ascontiguousarray(value).tobytes())
    key = digest.hexdigest()
    payload, manifest = work / f"{key}.npz", work / f"{key}.json"
    cached = _load_result(payload, manifest, key, folds)
    if cached is not None:
        print(f"  [6.6 {variant}/{condition}] verified checkpoint", flush=True)
        return {"predictions": cached[0], "fit_seconds": cached[1], "cache_hit": True, "fingerprint": key}
    with tempfile.TemporaryDirectory(prefix="regime-", dir=work) as temporary:
        folder = Path(temporary)
        np.savez_compressed(folder / "input.npz", **arrays)
        request = {"config": config, "n_classes": n_classes, "gate_features": gate_features,
                   "folds": fold_meta, "source": str(root / "src" / "s05_models.py"), "fingerprint": key,
                   "source_sha256": identity["source"]["src/s05_models.py"], "input_sha256": _sha(folder / "input.npz")}
        _atomic_json(folder / "input.json", request)
        for attempt in range(retries + 1):
            print(f"  [6.6 {variant}/{condition}] isolated fit {attempt + 1}/{retries + 1}", flush=True)
            (folder / "result.json").unlink(missing_ok=True)
            env = dict(os.environ, PYTHONHASHSEED=str(config["LGB_BASE"].get("seed", 42)))
            completed = subprocess.run(
                [sys.executable, "-B", "-X", "utf8", "-X", "faulthandler", str(Path(__file__).resolve()), "--worker", str(folder)],
                cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            fresh = _load_result(folder / "result.npz", folder / "result.json", key, folds) if completed.returncode == 0 else None
            if fresh is not None:
                os.replace(folder / "result.npz", payload)
                _atomic_json(manifest, {"status": "complete", "fingerprint": key,
                                       "payload_sha256": _sha(payload), "identity": identity})
                return {"predictions": fresh[0], "fit_seconds": fresh[1], "cache_hit": False, "fingerprint": key}
            detail = (completed.stderr or completed.stdout)[-2000:]
            print(f"  [6.6 {variant}/{condition}] child failed: exit={completed.returncode}", flush=True)
        raise RuntimeError(f"6.6 variant {variant}/{condition}: isolated LightGBM failed after {retries + 1} attempts; "
                           f"exit={completed.returncode}\n{detail}")


def _worker(folder):
    import lightgbm as lgb

    request = json.loads((folder / "input.json").read_text(encoding="utf-8"))
    if _sha(request["source"]) != request["source_sha256"] or _sha(folder / "input.npz") != request["input_sha256"]:
        raise ValueError("Regime worker source/input changed after fingerprinting")
    tree = ast.parse(Path(request["source"]).read_text(encoding="utf-8"))
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_regime_model")
    namespace = {"np": np, "pd": pd, "lgb": lgb, **request["config"]}
    exec(compile(ast.Module([factory], type_ignores=[]), request["source"], "exec"), namespace)
    fit = namespace["make_regime_model"](request["n_classes"], request["gate_features"])
    predictions, seconds = {}, []
    with np.load(folder / "input.npz", allow_pickle=False) as data:
        for i, entry in enumerate(request["folds"]):
            frames = {}
            for name, meta in entry["frames"].items():
                key = f"{i}_{name}"
                frames[name] = pd.DataFrame(data[key], index=pd.DatetimeIndex(data[key + "_index"]),
                                            columns=meta["columns"]).astype(dict(zip(meta["columns"], meta["dtypes"])))
            started = time.perf_counter()
            out = fit(frames["Xtr"], frames["ytr"], frames["Xva"])
            seconds.append(time.perf_counter() - started)
            for name in ("pred_avg", "pred_peak"):
                predictions[f"{i}_{name}"] = np.asarray(out[name], dtype=float).copy()
            del out, frames
            gc.collect()
    predictions["fit_seconds"] = np.asarray(seconds)
    np.savez_compressed(folder / "result.npz", **predictions)
    _atomic_json(folder / "result.json", {"status": "complete", "fingerprint": request["fingerprint"],
                                          "payload_sha256": _sha(folder / "result.npz")})


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Internal worker: --worker <job directory>")
    _worker(Path(sys.argv[2]))
