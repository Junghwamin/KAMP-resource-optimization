"""Read-only export of a completed notebook namespace; never imports training stages.

DataFrames use full-precision CSV with explicit index/dtype schemas, numeric arrays
use NPZ (allow_pickle=False), and object metadata uses JSON. Model objects and
functions are never serialized. ``load_snapshot`` returns a namespace suitable for
diagnostics and figure-only rendering, not a fitted-model checkpoint.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import platform
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
REQUIRED = ("df", "feat", "operating_calendar", "cv_results", "test_results",
            "FOLD_SPEC", "THETA", "TEST_START", "TEST_END", "FINAL_MODEL_NAME")
QUALITY = ("is_warmup", "is_outage", "is_erp_missing")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _scalar(value):
    if value is pd.NA:
        return {"$na": True}
    if value is pd.NaT:
        return {"$nat": True}
    if isinstance(value, np.generic):
        if isinstance(value, np.datetime64):
            return {"$timestamp": str(pd.Timestamp(value))}
        value = value.item()
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return {"$timestamp": value.isoformat()}
    if isinstance(value, (pd.Timedelta, dt.timedelta)):
        return {"$timedelta_ns": pd.Timedelta(value).value}
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return {"$tuple": [_scalar(v) for v in value]}
    raise TypeError(type(value).__name__)


def _unscalar(value):
    if not isinstance(value, dict):
        return value
    if "$na" in value:
        return pd.NA
    if "$nat" in value:
        return pd.NaT
    if "$timestamp" in value:
        return pd.Timestamp(value["$timestamp"])
    if "$timedelta_ns" in value:
        return pd.Timedelta(value["$timedelta_ns"], unit="ns")
    if "$float" in value:
        return float(value["$float"])
    if "$tuple" in value:
        return tuple(_unscalar(v) for v in value["$tuple"])
    raise ValueError("Unknown scalar schema")


def _safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Snapshot path escapes its directory")
    return path


class _Writer:
    def __init__(self, root):
        self.root, self.manifest, self.skipped = root, {}, {}
        self.count = 0

    def file(self, suffix):
        self.count += 1
        relative = f"payload/{self.count:05d}.{suffix}"
        return relative, self.root / relative

    def frame(self, value, label):
        original = value.to_frame() if isinstance(value, pd.Series) else value
        index = original.index
        vectors = [index.get_level_values(i) for i in range(index.nlevels)]
        vectors.extend(original.iloc[:, i] for i in range(original.shape[1]))
        csv = pd.DataFrame(index=range(len(original)))
        schemas = []
        for i, vector in enumerate(vectors):
            vector = pd.Series(vector).reset_index(drop=True)
            dtype = str(vector.dtype)
            # JSON cells preserve mixed labels, empty strings and literal "NA".
            cell_json = dtype == "object" or isinstance(vector.dtype, pd.CategoricalDtype)
            csv[f"v{i}"] = (vector.map(lambda v: json.dumps(_scalar(v), ensure_ascii=False, allow_nan=False))
                             if cell_json else vector)
            schemas.append({"dtype": dtype, "json_cells": cell_json})
        relative, path = self.file("csv")
        csv.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")
        node = {"kind": "series" if isinstance(value, pd.Series) else "frame", "path": relative,
                "rows": len(original), "n_index": index.nlevels,
                "index_names": [_scalar(n) for n in index.names],
                "columns": [_scalar(c) for c in original.columns],
                "column_names": [_scalar(n) for n in original.columns.names],
                "columns_multi": isinstance(original.columns, pd.MultiIndex),
                "range_index": isinstance(index, pd.RangeIndex), "vectors": schemas}
        if isinstance(index, pd.DatetimeIndex):
            node["index_freq"] = index.freqstr
        node["attrs"] = self.encode(dict(value.attrs), f"{label}.attrs")
        if isinstance(value, pd.Series):
            node["series_name"] = _scalar(value.name)
        self.manifest[relative] = {"logical_name": label, **node}
        return node

    def encode(self, value, label, stack=()):
        if id(value) in stack:
            raise TypeError("recursive object")
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return self.frame(value, label)
        if isinstance(value, pd.Index):
            return {"kind": "index", "data": self.frame(pd.Series(value), label),
                    "name": _scalar(value.name)}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError(f"Object ndarray is forbidden: {label}")
            relative, path = self.file("npz")
            np.savez_compressed(path, values=value)
            node = {"kind": "array", "path": relative, "shape": list(value.shape), "dtype": str(value.dtype)}
            self.manifest[relative] = {"logical_name": label, **node}
            return node
        if isinstance(value, (dict, list, tuple, set)):
            stack = (*stack, id(value))
            if isinstance(value, dict):
                pairs = []
                for key, val in value.items():
                    if str(key).startswith("__"):
                        continue
                    try:
                        pair = [_scalar(key), self.encode(val, f"{label}.{key}", stack)]
                    except TypeError as exc:
                        self.skipped[f"{label}.{key}"] = str(exc)
                    else:
                        pairs.append(pair)
                return {"kind": "dict", "items": pairs}
            values = sorted(value, key=str) if isinstance(value, set) else value
            return {"kind": type(value).__name__, "items": [self.encode(v, f"{label}[{i}]", stack)
                                                            for i, v in enumerate(values)]}
        return {"kind": "scalar", "value": _scalar(value)}


def _read_frame(root, node):
    raw = pd.read_csv(_safe_path(root, node["path"]), dtype=str, keep_default_na=False,
                      encoding="utf-8", float_precision="round_trip")
    vectors = []
    for i, spec in enumerate(node["vectors"]):
        v = raw[f"v{i}"]
        dtype = spec["dtype"]
        if spec["json_cells"]:
            v = v.map(lambda text: _unscalar(json.loads(text)))
            v = v.astype("category" if dtype == "category" else "object")
        elif dtype.startswith("datetime64"):
            v = pd.to_datetime(v)
        elif dtype.startswith("timedelta64"):
            v = pd.to_timedelta(v)
        elif dtype in ("bool", "boolean"):
            v = v.map({"True": True, "False": False, "": pd.NA}).astype(dtype)
        else:
            v = v.where(v.ne(""), np.nan).astype(dtype)
        vectors.append(v)
    n = node["n_index"]
    names = [_unscalar(v) for v in node["index_names"]]
    if n > 1:
        index = pd.MultiIndex.from_arrays(vectors[:n], names=names)
    elif node["range_index"] and list(vectors[0]) == list(range(len(raw))):
        index = pd.RangeIndex(len(raw), name=names[0])
    else:
        index = pd.Index(vectors[0], name=names[0])
        if isinstance(index, pd.DatetimeIndex) and node.get("index_freq"):
            index.freq = node["index_freq"]
    data = pd.concat(vectors[n:], axis=1) if len(vectors) > n else pd.DataFrame(index=range(len(raw)))
    cols = [_unscalar(c) for c in node["columns"]]
    cnames = [_unscalar(c) for c in node["column_names"]]
    data.columns = (pd.MultiIndex.from_tuples(cols, names=cnames) if node["columns_multi"]
                    else pd.Index(cols, name=cnames[0]))
    data.index = index
    data.attrs = _decode(root, node["attrs"])
    if len(data) != node["rows"]:
        raise ValueError("Snapshot row count mismatch")
    if node["kind"] == "series":
        result = data.iloc[:, 0]
        result.name = _unscalar(node["series_name"])
        return result
    return data


def _decode(root, node):
    kind = node["kind"]
    if kind == "scalar":
        return _unscalar(node["value"])
    if kind in ("frame", "series"):
        return _read_frame(root, node)
    if kind == "index":
        return pd.Index(_read_frame(root, node["data"]), name=_unscalar(node["name"]))
    if kind == "array":
        with np.load(_safe_path(root, node["path"]), allow_pickle=False) as archive:
            value = archive["values"]
        if value.dtype.hasobject or list(value.shape) != node["shape"] or str(value.dtype) != node["dtype"]:
            raise ValueError("Snapshot array schema mismatch")
        return value
    if kind == "dict":
        return {_unscalar(k): _decode(root, v) for k, v in node["items"]}
    values = [_decode(root, v) for v in node["items"]]
    return {"list": list, "tuple": tuple, "set": set}[kind](values)


def _prediction_rows(ns, *, test=False):
    """Expose existing estimates; thresholds below retain their internal-CV role."""
    feat, theta = ns["feat"], float(ns["THETA"])
    specs = {int(f): (pd.Timestamp(tr) + pd.Timedelta(hours=23), pd.Timestamp(vs),
                      pd.Timestamp(ve) + pd.Timedelta(hours=23)) for f, tr, vs, ve in ns["FOLD_SPEC"]}
    rows = []
    for name, result in ns["test_results" if test else "cv_results"].items():
        condition = result.get("cond", "D1")
        if test:
            index = pd.DatetimeIndex(result["index"])
            o = pd.DataFrame({k: result[k] for k in ("y_avg", "y_peak", "y_cls", "pred_avg", "pred_peak")}, index=index)
            o["fold"] = -1
            o["prob"] = result.get("prob") if result.get("prob") is not None else np.nan
        else:
            o = result["oof"].copy(deep=True)
            index = pd.DatetimeIndex(o.index)
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unordered prediction times: {name}")
        if not index.isin(feat.index).all():
            raise ValueError(f"Prediction dates missing from features: {name}")
        for truth in ("y_avg", "y_peak", "y_cls"):
            if not np.array_equal(o[truth].to_numpy(), feat.loc[index, truth].to_numpy(), equal_nan=True):
                raise ValueError(f"Prediction truth/feature run mismatch: {name}/{truth}")
        if test and not index.equals(pd.date_range(ns["TEST_START"], ns["TEST_END"], freq="h")):
            raise ValueError(f"Test predictions do not cover the declared hourly test interval: {name}")
        o["datetime"] = index
        o["origin"] = index.normalize()
        o["history_end"] = o["origin"] - pd.Timedelta(hours=1)
        if test:
            o["train_end"] = pd.Timestamp(ns["TEST_START"]) - pd.Timedelta(hours=1)
        else:
            if not set(o["fold"]).issubset(specs):
                raise ValueError(f"Unknown fold: {name}")
            for fold, g in o.groupby("fold"):
                tr, vs, ve = specs[int(fold)]
                if tr >= vs or g.index.min() < vs or g.index.max() > ve:
                    raise ValueError(f"Invalid fold boundary: {name}/{fold}")
            o["train_end"] = o["fold"].map(lambda f: specs[int(f)][0])
        for quality in QUALITY:
            if quality in feat:
                o[quality] = feat.loc[index, quality].to_numpy(dtype=bool)
        o["eligible"] = True
        for quality in QUALITY:
            if quality in o:
                o["eligible"] &= ~o[quality]
        if not test and not o["eligible"].all():
            raise ValueError(f"Ineligible row present in OOF: {name}")
        tau = float(ns["cv_results"].get(name, {}).get("tau", np.nan))
        o["theta"], o["tau"] = theta, tau
        o["model"], o["data_condition"] = name, condition
        o["prob_raw"] = o.get("prob", np.nan)
        o["prob_cal"] = np.nan
        if name == "피크 직접분류" and not test:
            calibrated = ns.get("_calib", {}).get("evaluation_predictions")
            if isinstance(calibrated, pd.DataFrame):
                o["prob_cal"] = calibrated["prob_cal"].reindex(index)
        o["peak_pred_label"] = (o["pred_peak"] >= tau).astype(int)
        o["evaluation_role"] = "frozen_reference" if test else "internal_selection_oof"
        o["threshold_role"] = "median_internal_fold_tau"
        if o["prob_raw"].notna().any():
            # The exact pooled classifier tau is a discarded local value in the
            # original reporting function. Never substitute the regression tau.
            o["regression_tau"] = tau
            o["tau"] = np.nan
            o["peak_pred_label"] = pd.Series(pd.NA, index=o.index, dtype="Int64")
            o["threshold_role"] = "classifier_pooled_tau_not_retained; see original peak table"
        rows.append(o.reset_index(drop=True).drop(columns=["prob"], errors="ignore"))
    if not rows:
        raise ValueError("No prediction rows to export")
    long = pd.concat(rows, ignore_index=True)
    if long.duplicated(["datetime", "model", "data_condition"]).any():
        raise ValueError("Duplicate prediction keys")
    return long


def _tree_data(ns):
    tree, columns = ns.get("peak_tree"), ns.get("rule_cols")
    if tree is None or columns is None:
        return None
    t, o = tree.tree_, ns.get("OOF")
    samples = {0: np.ones(len(o), dtype=bool)} if isinstance(o, pd.DataFrame) else {}
    nodes = []
    for i in range(t.node_count):
        node = {"id": i, "left": int(t.children_left[i]), "right": int(t.children_right[i]),
                "feature": int(t.feature[i]), "threshold": float(t.threshold[i]),
                "n_node_samples": int(t.n_node_samples[i]),
                "weighted_n_node_samples": float(t.weighted_n_node_samples[i]),
                "value": t.value[i].tolist()}
        if i in samples:
            mask = samples[i]
            node.update(actual_n=int(mask.sum()), actual_peak=int(o.loc[mask, "y_cls"].sum()))
            if node["left"] >= 0:
                left = o[columns[node["feature"]]].to_numpy() <= node["threshold"]
                samples[node["left"]], samples[node["right"]] = mask & left, mask & ~left
        nodes.append(node)
    return {"nodes": nodes, "feature_names": list(columns), "classes": tree.classes_.tolist(),
            "max_depth": int(t.max_depth), "node_count": int(t.node_count)}


def _fingerprint(meta):
    numeric_names = {"df", "feat", "operating_calendar", "cv_results", "test_results", "oof_long", "test_long"}
    numeric = {meta["manifest"][k]["logical_name"]: v for k, v in meta["files_hashes"].items()
               if meta["manifest"][k]["logical_name"].split(".")[0] in numeric_names}
    configuration = {k: meta["namespace"][k] for k in
                     ("THETA", "FEATURE_COLS", "FOLD_SPEC", "LGB_BASE", "lgb_best_params", "SEED",
                      "N_ESTIMATORS", "REGIME_CUT", "FINAL_MODEL_NAME", "TEST_START", "TEST_END")
                     if k in meta["namespace"]}
    scientific = {"numeric": numeric, "config": configuration, "runtime": meta["runtime"]}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def export_snapshot(ns, path, *, root=None, run_metadata=None):
    """Export existing objects without fitting or mutating them; fail closed.

``path`` must be empty or new. Failed exports remain incomplete for inspection and
are rejected by ``load_snapshot``. All paths inside metadata are relative.
    """
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Snapshot destination must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "payload").mkdir()
    meta = {"schema_version": SCHEMA_VERSION, "complete": False, "status": "writing",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest": {}, "files_hashes": {}, "namespace": {}}
    _dump(path / "metadata.json", meta)
    writer = _Writer(path)
    try:
        missing = [key for key in REQUIRED if key not in ns]
        if missing:
            raise ValueError(f"Snapshot namespace missing required objects: {missing}")
        prepared = {key: value for key, value in ns.items() if not key.startswith("_")}
        for key in ("_calib", "_unc", "_THETA_PREVIEW", "_clf_name"):
            if key in ns:
                prepared[key] = ns[key]
        prepared["oof_long"], prepared["test_long"] = _prediction_rows(ns), _prediction_rows(ns, test=True)
        prepared["peak_tree_data"] = _tree_data(ns)
        prepared.pop("peak_tree", None)
        for key, value in prepared.items():
            try:
                meta["namespace"][key] = writer.encode(value, key)
            except TypeError as exc:
                writer.skipped[key] = str(exc)
        meta["manifest"] = writer.manifest
        meta["files_hashes"] = {relative: _sha(path / relative) for relative in writer.manifest}
        meta["skipped_objects"] = writer.skipped
        meta["coverage"] = {
            "oof_models": list(ns["cv_results"]), "test_models": list(ns["test_results"]),
            "final_model": str(ns["FINAL_MODEL_NAME"]),
            "oof_rows_by_model": prepared["oof_long"].groupby("model").size().to_dict(),
            "test_rows_by_model": prepared["test_long"].groupby("model").size().to_dict(),
            "limitations": ["D2 rerun OOF local variables are not retained by the original pipeline; D2 summary tables only.",
                            "Discarded local fits/ablation OOF cannot be recovered without another fit.",
                            "OOF thresholds and global theta retain internal-selection status; export does not make them independent."]}
        if (run_metadata or {}).get("mode") == "FULL":
            final = str(ns["FINAL_MODEL_NAME"])
            if (meta["coverage"]["oof_rows_by_model"].get(final),
                    meta["coverage"]["test_rows_by_model"].get(final)) != (1279, 336):
                raise ValueError("FULL reference must retain 1,279 final-model OOF rows and 336 test rows")
        versions = {}
        for package in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "matplotlib", "shap"):
            try:
                versions[package] = package_metadata.version(package)
            except package_metadata.PackageNotFoundError:
                versions[package] = "unavailable"
        meta["runtime"] = {"python": platform.python_version(), "platform": platform.system(), "versions": versions}
        meta["run_metadata"] = run_metadata or {}
        constant_names = ("THETA", "FEATURE_COLS", "FEATURE_GROUPS", "FEATURE_LABELS", "FOLD_SPEC",
                          "ACTIVE_FOLDS", "LGB_BASE", "N_ESTIMATORS", "REGIME_CUT", "SEED", "FAST",
                          "FINAL_MODEL_NAME", "lgb_best_params", "clf_best_params", "TEST_START", "TEST_END")
        meta["constants"] = {}
        for key in constant_names:
            if key in ns:
                # Explicit model configuration remains convenient JSON; dates
                # are ISO text and numpy scalars are converted to Python types.
                meta["constants"][key] = json.loads(json.dumps(ns[key], ensure_ascii=False,
                    default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
        provenance = {}
        if root is not None:
            root = Path(root)
            for relative in ("src", "data", "experiments/config.json"):
                target = root / relative
                candidates = sorted(target.rglob("*")) if target.is_dir() else [target]
                for file in candidates:
                    if file.is_file() and "__pycache__" not in file.parts:
                        provenance[file.relative_to(root).as_posix()] = _sha(file)
        meta["source_hashes"] = provenance
        # Numeric/config evidence has its own digest: figure style changes alone
        # cannot invalidate an otherwise identical scientific data snapshot.
        meta["fingerprint"] = _fingerprint(meta)
        meta["render_fingerprint"] = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        meta.update(complete=True, status="complete")
        _dump(path / "metadata.json", meta)
        load_snapshot(path)  # Validate persisted schemas and checksums before success.
        return meta
    except BaseException as exc:
        meta.update(complete=False, status="incomplete", error={"type": type(exc).__name__, "message": str(exc)},
                    manifest=writer.manifest)
        meta["files_hashes"] = {relative: _sha(path / relative) for relative in writer.manifest}
        _dump(path / "metadata.json", meta)
        raise


def load_snapshot(path, verify=True):
    """Return restored namespace and standardized aliases; reject incomplete data."""
    root = Path(path).resolve()
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("complete") is not True or meta.get("status") != "complete":
        raise ValueError("Snapshot is incomplete or schema is unsupported")
    if verify:
        if set(meta["manifest"]) != set(meta["files_hashes"]):
            raise ValueError("Snapshot manifest/hash coverage mismatch")
        if meta.get("fingerprint") != _fingerprint(meta):
            raise ValueError("Snapshot scientific fingerprint mismatch")
        for relative, expected in meta["files_hashes"].items():
            file = _safe_path(root, relative)
            if not file.is_file() or _sha(file) != expected:
                raise ValueError(f"Snapshot hash mismatch: {relative}")
    result = {name: _decode(root, node) for name, node in meta["namespace"].items()}
    result.update(metadata=meta, processed=result["df"], features=result["feat"], calendar=result["operating_calendar"])
    result["peak_tree"] = result.get("peak_tree_data")
    return result
