"""Pure boundaries and isolated numeric prediction jobs for additional experiments.

Only explicitly named function definitions are compiled from the notebook-style
source. Importing this module never imports a stage or fits a model.
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
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

OUTER_STARTS = ("2021-07-07", "2021-07-21", "2021-08-04", "2021-08-18")
CANDIDATES = ("naive", "rf", "lgb", "regime2", "regime3")
MASKS = ("is_warmup", "is_outage", "is_erp_missing")
REGISTERED_FEATURES = (
    [f"{target}_lag{hours}" for target in ("y_avg", "y_peak") for hours in (24, 48, 168)]
    + [f"roll{hours}_{stat}" for hours in (24, 168) for stat in
       ("avg_mean", "avg_max", "avg_std", "peak_max", "peak_cnt")]
    + ["hour", "dow", "month", "is_sat", "is_sun", "is_holiday", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    + ["is_daytime", "is_startup_08", "is_restart_13", "is_shutdown", "shutdown_nth",
       "is_first_day_back", "prev_day_shutdown", "is_state_switch"]
    + ["prod", "headcount", "prod_chg_ratio", "prod_cum_day", "temp", "humid", "wind", "rain", "cdd", "hdd"]
)
LGB_BASE = dict(objective="l1", verbosity=-1, deterministic=True,
                force_row_wise=True, num_threads=4, seed=42,
                bagging_seed=42, feature_fraction_seed=42, data_random_seed=42)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (Path, pd.Timestamp)):
        return str(obj)
    return obj


def json_bytes(value):
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True,
                      allow_nan=False).encode("utf-8")


def protocol_guard(config, registered, features, *, diagnostic_keys=()):
    """Only the preregistered scientific configuration can produce FULL status.

    Runtime retry/timeout/logging changes may be diagnostic. Unknown overrides
    are conservatively custom rather than silently being labelled full.
    """
    diagnostic = set(diagnostic_keys) | {"retries", "timeout_seconds"}
    deviations = []
    for key in sorted(set(config) | set(registered)):
        if key in diagnostic:
            continue
        if key not in registered or key not in config or json_bytes(config[key]) != json_bytes(registered[key]):
            deviations.append({"field": key, "registered": registered.get(key), "actual": config.get(key),
                               "reason": "unregistered_override" if key not in registered else "registered_value_changed"})
    if list(features) != REGISTERED_FEATURES:
        deviations.append({"field": "feature_columns", "registered": REGISTERED_FEATURES, "actual": list(features),
                           "reason": "fixed_44_feature_identity_or_order_changed"})
    scientific = {k: v for k, v in registered.items() if k not in diagnostic}
    return {"matches_registered_full": not deviations, "deviations": deviations,
            "registered_scientific_sha256": hashlib.sha256(json_bytes({"config": scientific, "features": REGISTERED_FEATURES})).hexdigest()}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        name = f.name
        f.write(json_bytes(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(name, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(frame):
    h = hashlib.sha256(json_bytes([list(frame.columns), list(map(str, frame.dtypes))]))
    h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return h.hexdigest()


@lru_cache(maxsize=1)
def runtime_identity():
    import lightgbm.libpath
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "optuna")},
            "lightgbm_binary_sha256": [sha(p) for p in lightgbm.libpath.find_lib_path()]}


def peak_memory_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_data(snapshot):
    from tools.analysis_snapshot import load_snapshot
    data = load_snapshot(Path(snapshot), verify=True)
    frame = data["feat"].copy(deep=True)
    metadata = data["metadata"]
    features = list(data.get("FEATURE_COLS", metadata.get("FEATURE_COLS", metadata.get("constants", {}).get("FEATURE_COLS", []))))
    if len(features) != 44 or len(set(features)) != 44:
        raise ValueError("The prespecified experiment requires exactly 44 unique features")
    required = set(features) | set(MASKS) | {"y_avg", "y_peak"}
    if required - set(frame):
        raise ValueError(f"Snapshot is missing columns: {sorted(required - set(frame))}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError("Snapshot needs a unique DatetimeIndex")
    frame = frame.sort_index()
    if frame.index.hasnans or not np.all(np.diff(frame.index.asi8) == pd.Timedelta(hours=1).value):
        raise ValueError("Snapshot must preserve the continuous hourly calendar including masked rows")
    for column in MASKS:
        if not frame[column].isin([True, False, 0, 1]).all():
            raise ValueError(f"Invalid quality mask {column}")
        frame[column] = frame[column].astype(bool)
    return frame, features, metadata


def partition(eval_start, *, eval_days=14, calibration_days=14, gap_hours=24):
    """Half-open intervals; O2 core ends June20 23h, cal June22-July5."""
    start = pd.Timestamp(eval_start)
    cal_end = start - pd.Timedelta(hours=gap_hours)
    cal_start = cal_end - pd.Timedelta(days=calibration_days)
    core_end = cal_start - pd.Timedelta(hours=gap_hours)
    return {"core_end_exclusive": core_end, "core_train_end": core_end - pd.Timedelta(hours=1),
            "cal_start": cal_start, "cal_end_exclusive": cal_end,
            "cal_end": cal_end - pd.Timedelta(hours=1), "eval_start": start,
            "eval_end_exclusive": start + pd.Timedelta(days=eval_days)}


def valid_rows(frame, features):
    return (~frame[list(MASKS)].any(axis=1)
            & np.isfinite(frame[features + ["y_avg", "y_peak"]]).all(axis=1))


def prepare_split(frame, features, spec, *, min_train_rows=48):
    """Fit theta solely on usable core; recompute all theta-dependent inputs."""
    base_ok = valid_rows(frame, features)
    tr = (frame.index < spec["core_end_exclusive"]) & base_ok
    if int(tr.sum()) < min_train_rows:
        raise ValueError(f"Insufficient core rows: {int(tr.sum())} < {min_train_rows}")
    theta = float(frame.loc[tr, "y_peak"].quantile(.95))
    rebuilt = frame.copy(deep=True)
    peak = (frame["y_peak"] >= theta).astype(float)
    key = frame.index.normalize() - pd.Timedelta(days=1)
    for hours in (24, 168):
        counts = peak.rolling(hours, min_periods=hours).sum()
        daily = counts[counts.index.hour == 23].copy()
        daily.index = daily.index.normalize()
        rebuilt[f"roll{hours}_peak_cnt"] = daily.reindex(key).to_numpy()
    rebuilt["y_cls"] = (rebuilt["y_peak"] >= theta).astype(int)
    ok = valid_rows(rebuilt, features)
    masks = {
        "core": (rebuilt.index < spec["core_end_exclusive"]) & ok,
        "calibration": ((rebuilt.index >= spec["cal_start"]) &
                        (rebuilt.index < spec["cal_end_exclusive"]) & ok),
        "evaluation": ((rebuilt.index >= spec["eval_start"]) &
                       (rebuilt.index < spec["eval_end_exclusive"]) & ok),
    }
    if any(int(mask.sum()) == 0 for mask in masks.values()):
        raise ValueError("Empty core/calibration/evaluation after prespecified quality masks")
    return rebuilt, masks, theta


def split_audit(frame, masks, theta, spec, features):
    rows = []
    raw_masks = {
        "core": frame.index < spec["core_end_exclusive"],
        "calibration": (frame.index >= spec["cal_start"]) & (frame.index < spec["cal_end_exclusive"]),
        "evaluation": (frame.index >= spec["eval_start"]) & (frame.index < spec["eval_end_exclusive"]),
    }
    for role, mask in masks.items():
        part = frame.loc[mask]
        raw = frame.loc[raw_masks[role]]
        rows.append({"role": role, "first": str(part.index.min()), "last": str(part.index.max()),
                     "n": len(part), "days": part.index.normalize().nunique(),
                     "n_raw": len(raw), "n_excluded": len(raw)-len(part),
                     **{name+"_count": int(raw[name].sum()) for name in MASKS},
                     "nonfinite_features_count": int((~np.isfinite(raw[features]).all(axis=1)).sum()),
                     "mask_counts_overlap": True,
                     "positives": int((part.y_peak >= theta).sum()), "theta": theta,
                     "row_hash": frame_hash(part[features]), **jsonable(spec)})
    return rows


def profile_audit(frame, spec):
    """Target-containing novelty audit; never returned to model selection."""
    daily = {}
    for day, part in frame.groupby(frame.index.normalize()):
        if len(part) == 24 and np.isfinite(part[["y_avg", "y_peak"]]).all().all():
            daily[day] = hashlib.sha256(part[["y_avg", "y_peak"]].to_numpy().tobytes()).hexdigest()
    train_list = [h for d, h in daily.items() if d < spec["core_end_exclusive"]]
    train = set(train_list)
    return [{"date": str(d.date()), "profile_sha256": h, "seen_in_core": h in train,
             "core_profile_days": len(train_list), "core_unique_profiles": len(train),
             "core_duplicate_days_retained_D1": len(train_list)-len(train),
             "role": "post_scoring_novelty_audit_only"}
            for d, h in daily.items() if spec["eval_start"] <= d < spec["eval_end_exclusive"]]


def factory_namespace(root, config):
    """Reuse original pure factories without executing source globals or imports."""
    import lightgbm as lgb
    source = Path(root) / "src/s05_models.py"
    allowed = {"make_lgb_regressor", "make_regime_model", "make_peak_classifier", "rf_fit", "naive_fit"}
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in allowed]
    if {n.name for n in nodes} != allowed:
        raise ValueError("Model source no longer matches the audited adapter contract")
    seed = int(config.get("seed", 42))
    base = {**LGB_BASE, **config.get("lgb_base", {}), "num_threads": int(config.get("threads", 4)),
            "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed, "data_random_seed": seed}
    namespace = {"np": np, "pd": pd, "lgb": lgb, "SEED": seed, "FAST": False,
                 "N_ESTIMATORS": int(config.get("n_estimators", 800)), "LGB_BASE": base, "REGIME_CUT": 70.0}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def direct_predict(root, model, Xtr, ytr, Xpred, config, params=None):
    """Worker-only fit; labels from calibration/evaluation are not accepted."""
    ns = factory_namespace(root, config)
    params = params or {}
    fallback = None
    if model == "clf" and ytr["y_cls"].nunique() < 2:
        pred = {"prob": np.full(len(Xpred), float(ytr.y_cls.mean()))}
        return pred, {"fallback": "single_class_training_probability", "fit_seconds": 0.0}
    if model.startswith("regime"):
        labels = np.digitize(ytr.y_avg, [30., 70.]) if model == "regime3" else (ytr.y_avg >= 70).astype(int)
        if len(np.unique(labels)) < 2:
            fallback = "single_regime_training_uses_single_lgb"
            fit = ns["make_lgb_regressor"]()
        else:
            fit = ns["make_regime_model"](int(model[-1]))
    elif model == "lgb":
        fit = ns["make_lgb_regressor"](params)
    elif model == "clf":
        fit = ns["make_peak_classifier"](params)
    elif model in ("rf", "naive"):
        fit = ns[model + "_fit"]
    else:
        raise ValueError(f"Unknown model {model}")
    started = time.perf_counter()
    out = fit(Xtr, ytr, Xpred)
    seconds = time.perf_counter() - started
    names = ("prob",) if model == "clf" else ("pred_avg", "pred_peak")
    numeric = {name: np.asarray(out[name], dtype=float) for name in names}
    if any(v.shape != (len(Xpred),) or not np.isfinite(v).all() for v in numeric.values()):
        raise ValueError("Non-finite or invalid prediction shape")
    return numeric, {"fit_seconds": seconds, "fallback": fallback}


def _cache_read(folder, fingerprint, n, names):
    try:
        manifest = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        payload = folder / "result.npz"
        if manifest.get("status") != "complete" or manifest.get("fingerprint") != fingerprint:
            return None
        if manifest["payload_sha256"] != sha(payload):
            return None
        with np.load(payload, allow_pickle=False) as data:
            pred = {k: data[k].copy() for k in names}
        if any(v.shape != (n,) or not np.isfinite(v).all() for v in pred.values()):
            return None
        return pred, manifest
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def predict_job(root, cache, model, Xtr, ytr, Xpred, config, params=None):
    """Sequential spawn-safe fit with hash-verified numeric-only checkpoint."""
    root, cache = Path(root).resolve(), Path(cache)
    identity = {"schema": 1, "model": model, "params": params or {}, "config": config,
                "Xtr": frame_hash(Xtr), "ytr": frame_hash(ytr), "Xpred": frame_hash(Xpred),
                "adapter_sha256": sha(__file__), "source_sha256": sha(root / "src/s05_models.py"),
                "runtime": runtime_identity()}
    fingerprint = hashlib.sha256(json_bytes(identity)).hexdigest()
    folder = cache / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    names = ("prob",) if model == "clf" else ("pred_avg", "pred_peak")
    hit = _cache_read(folder, fingerprint, len(Xpred), names)
    if hit is not None:
        return hit[0], {**hit[1], "cache_hit": True}
    if list(Xtr.columns) != list(Xpred.columns) or Xtr.index.has_duplicates or Xpred.index.has_duplicates:
        raise ValueError("Training and inference feature order or row identity mismatch")
    frames = {"Xtr": Xtr, "ytr": ytr, "Xpred": Xpred}
    arrays = {}
    for key, frame in frames.items():
        array = frame.to_numpy()
        if array.dtype.hasobject:
            raise TypeError("Prediction jobs accept numeric frames only")
        arrays[key] = array
        arrays[key + "_index"] = frame.index.asi8
    np.savez_compressed(folder / "input.npz", **arrays)
    request = {"root": str(root), "model": model, "params": params or {}, "config": config,
               "fingerprint": fingerprint, "identity": identity,
               "input_sha256": sha(folder / "input.npz"),
               "frames": {k: {"columns": list(f.columns), "dtypes": list(map(str, f.dtypes))}
                          for k, f in frames.items()}}
    atomic_json(folder / "input.json", request)
    atomic_json(folder / "result.json", {"status": "running", "fingerprint": fingerprint})
    for attempt in range(int(config.get("retries", 1)) + 1):
        print(f"[additional fit] {model} {len(Xtr)} train / {len(Xpred)} predict {fingerprint[:10]}", flush=True)
        log_path = folder / f"worker_{attempt}.log"
        env = dict(os.environ, PYTHONHASHSEED=str(config.get("seed", 42)),
                   OMP_NUM_THREADS=str(config.get("threads", 4)))
        try:
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen([sys.executable, "-B", "-X", "utf8", "-X", "faulthandler",
                                         str(Path(__file__).resolve()), "--worker", str(folder.resolve())],
                                        cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                started = time.monotonic()
                while True:
                    try:
                        code = proc.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        atomic_json(folder / "heartbeat.json", {"pid": proc.pid, "model": model,
                                    "elapsed_seconds": time.monotonic() - started, "status": "running"})
                        print(f"[additional fit heartbeat] {model} {time.monotonic()-started:.0f}s", flush=True)
                        if time.monotonic() - started > config.get("timeout_seconds", 3600):
                            proc.kill()
                            proc.wait()
                            raise TimeoutError("Isolated additional fit exceeded its resource ceiling")
                    except BaseException:
                        proc.kill()
                        proc.wait()
                        raise
            hit = _cache_read(folder, fingerprint, len(Xpred), names) if code == 0 else None
            if hit is not None:
                return hit[0], {**hit[1], "cache_hit": False, "log": str(log_path)}
            reason = f"worker exit={code}; log={log_path}"
        except (TimeoutError, OSError) as exc:
            reason = str(exc)
        atomic_json(folder / "result.json", {"status": "failed", "fingerprint": fingerprint,
                    "reason": reason, "attempt": attempt})
    raise RuntimeError(f"Additional fit failed: {reason}")


def calibrate(y_peak, pred_peak, theta, probability=None, *, min_positive=5, min_negative=20):
    """Use the fixed core predictor's calibration tail only."""
    y = (np.asarray(y_peak) >= theta).astype(int)
    p = int(y.sum())
    n = int(len(y) - p)
    meta = {"theta": float(theta), "positives": p, "negatives": n,
            "tau": float(theta), "probability_calibration": "not_requested", "fallback": None}
    if p < min_positive or n < min_negative:
        meta.update(fallback="insufficient_calibration_classes", probability_calibration="raw_fallback")
        return meta, None
    pred = np.asarray(pred_peak, float)
    candidates = np.unique(np.round(np.quantile(pred, np.linspace(.5, .999, 120)), 2))
    best, best_f1 = theta, -1.0
    for threshold in candidates:
        label = pred >= threshold
        tp = int(((y == 1) & label).sum())
        fp = int(((y == 0) & label).sum())
        fn = p - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if f1 > best_f1:
            best, best_f1 = float(threshold), f1
    meta["tau"] = float(best)
    iso = None
    if probability is not None:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(probability), y)
        meta["probability_calibration"] = "isotonic_calibration_tail_fixed_theta"
        meta["isotonic_x"] = iso.X_thresholds_.tolist()
        meta["isotonic_y"] = iso.y_thresholds_.tolist()
    return meta, iso


def metrics(y_avg, y_peak, pred_avg, pred_peak, theta, tau):
    y, peak, pa, pp = map(lambda a: np.asarray(a, float), (y_avg, y_peak, pred_avg, pred_peak))
    positive, alarm = peak >= theta, pp >= tau
    tp = int((positive & alarm).sum())
    fn = int((positive & ~alarm).sum())
    fp = int((~positive & alarm).sum())
    tn = int((~positive & ~alarm).sum())
    return {"n": len(y), "MAE": float(np.abs(y-pa).mean()), "Peak-MAE":
            float(np.abs(y[positive]-pa[positive]).mean()) if positive.any() else np.nan,
            "Peak15-MAE": float(np.abs(peak[positive]-pp[positive]).mean()) if positive.any() else np.nan,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "Recall": tp/(tp+fn) if tp+fn else np.nan,
            "Precision": tp/(tp+fp) if tp+fp else np.nan,
            "F1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
            "false_positive_rate": fp/(fp+tn) if fp+tn else np.nan}


def prediction_rows(frame, eval_mask, pred, theta, tau, spec, *, model, outer, selected=False):
    part = frame.loc[eval_mask]
    out = pd.DataFrame({"datetime": part.index, "origin": part.index.normalize(),
                        "history_end": part.index.normalize()-pd.Timedelta(hours=1),
                        "outer": outer, "fold": int(str(outer).lstrip("O")), "model": model,
                        "selected": selected, "y_avg": part.y_avg.to_numpy(),
                        "y_peak": part.y_peak.to_numpy(), "pred_avg": pred["pred_avg"],
                        "pred_peak": pred["pred_peak"], "theta": theta, "tau": tau,
                        "peak_pred_label": (pred["pred_peak"] >= tau).astype(int),
                        "y_cls": (part.y_peak.to_numpy() >= theta).astype(int),
                        "y_cls_reference_187": (part.y_peak.to_numpy() >= 187.).astype(int),
                        "core_train_end": spec["core_train_end"], "cal_start": spec["cal_start"],
                        "cal_end": spec["cal_end"], "evaluation_role": "retrospective_temporal_selection"})
    for target, column in (("production", "prod"), ("planned_production", "prod"),
                           ("headcount", "headcount"), ("plan_headcount", "headcount")):
        if column in part:
            out[target] = part[column].to_numpy()
    out["plan_known"] = (~part.is_erp_missing).to_numpy()
    out["plan_known_source"] = "retrospective_ERP_missingness_proxy_assumed_available_at_origin"
    out["is_operating"] = (part["headcount"] > 0).to_numpy()
    for mask in MASKS:
        out[mask] = part[mask].to_numpy()
    return out


def _worker(folder):
    folder = Path(folder)
    request = json.loads((folder / "input.json").read_text(encoding="utf-8"))
    if sha(folder / "input.npz") != request["input_sha256"]:
        raise ValueError("Worker input changed")
    root = Path(request["root"])
    if sha(root / "src/s05_models.py") != request["identity"]["source_sha256"]:
        raise ValueError("Worker model source changed")
    if sha(__file__) != request["identity"]["adapter_sha256"]:
        raise ValueError("Worker adapter changed")
    frames = {}
    with np.load(folder / "input.npz", allow_pickle=False) as data:
        for k, meta in request["frames"].items():
            frames[k] = pd.DataFrame(data[k], index=pd.DatetimeIndex(data[k+"_index"]),
                                     columns=meta["columns"]).astype(dict(zip(meta["columns"], meta["dtypes"])))
    pred, info = direct_predict(root, request["model"], frames["Xtr"], frames["ytr"], frames["Xpred"],
                                request["config"], request["params"])
    np.savez_compressed(folder / "result.npz", **pred)
    atomic_json(folder / "result.json", {"status": "complete", "fingerprint": request["fingerprint"],
                "payload_sha256": sha(folder / "result.npz"), "identity": request["identity"],
                "peak_memory_bytes": peak_memory_bytes(), **info})
    gc.collect()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Internal additional-experiment worker: --worker <folder>")
    _worker(sys.argv[2])
