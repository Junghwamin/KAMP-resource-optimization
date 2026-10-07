"""Trusted-local model archives with preprocessing and prediction round-trip checks."""
from pathlib import Path
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version

import joblib
import numpy as np


def state_from_output(out):
    if "_persist" in out:
        return out["_persist"]
    if out.get("_artifacts") is not None:
        return out["_artifacts"]
    if "_models" in out:
        return {"kind": "regressors", "models": out["_models"]}
    raise ValueError("Fitted model did not expose persistence state")


def predict_state(state, X):
    kind = state["kind"]
    if kind == "naive":
        return {"pred_avg": np.asarray(X["y_avg_lag168"]),
                "pred_peak": np.asarray(X["y_peak_lag168"])}
    if kind == "ensemble":
        a, b = [predict_state(s, X) for s in state["members"]]
        w = state["weight"]
        return {k: w * a[k] + (1 - w) * b[k] for k in ("pred_avg", "pred_peak")}
    if kind == "binary_clf":
        p = state["clf"].predict_proba(X)[:, 1]
        return {"pred_avg": np.full(len(X), np.nan), "pred_peak": p, "prob": p}
    if kind == "regime":
        rr = state["gate"].predict(X[state["gcols"]])
        result = {}
        for tgt, key in (("y_avg", "pred_avg"), ("y_peak", "pred_peak")):
            pred = np.empty(len(X))
            assigned = np.zeros(len(X), bool)
            for r, model in state["regs"][tgt].items():
                mask = rr == r
                if mask.any():
                    pred[mask] = model.predict(X[mask])
                    assigned |= mask
            if (~assigned).any():
                pred[~assigned] = state["fallbacks"][tgt].predict(X[~assigned])
            result[key] = pred
        return result
    if kind == "regressors":
        return {"pred_" + tgt[2:]: m.predict(X) for tgt, m in state["models"].items()}
    if kind == "keras":
        values = state["xsc"].transform(X)
        if state.get("reshape"):
            values = values.reshape((len(values), *state["reshape"]))
        return {"pred_" + tgt[2:]: state["yscalers"][tgt].inverse_transform(
            m.predict(values, verbose=0).reshape(-1, 1)).ravel()
            for tgt, m in state["models"].items()}
    if kind == "rnn_original":
        # Input is the guidebook lag matrix, excluding its target column.
        sc = state["scaler"]
        values = np.asarray(X) * sc.scale_[1:] + sc.min_[1:]
        pred = state["model"].predict(values.reshape(-1, 24, 7), verbose=0).ravel()
        return {"pred_avg": pred * sc.data_range_[0] + sc.data_min_[0]}
    raise ValueError(f"Unknown model kind: {kind}")


def save_model_archive(root, name, state, features, X, expected, metadata=None):
    """Save final fit only; fail immediately if loaded predictions differ."""
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    keras_files = []

    def encode(value):
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [encode(v) for v in value]
        if hasattr(value, "save") and type(value).__module__.startswith(("keras", "tensorflow")):
            filename = f"network_{len(keras_files)}.keras"
            value.save(folder / filename)
            keras_files.append(filename)
            return {"__keras_file__": filename}
        return value

    joblib.dump({"state": encode(state), "features": list(features)}, folder / "model.joblib", compress=3)
    loaded = load_model_archive(folder)
    actual = predict_state(loaded["state"], X)
    errors = {}
    for key, values in expected.items():
        np.testing.assert_allclose(actual[key], values, rtol=1e-5, atol=1e-5, equal_nan=True)
        delta = np.abs(np.asarray(actual[key]) - np.asarray(values))
        errors[key] = float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0
    files = ["model.joblib", *keras_files]
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "tensorflow", "keras"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    manifest = {"name": name, "kind": state["kind"], "features": list(features),
                "python": platform.python_version(), "packages": packages, "metadata": metadata or {},
                "roundtrip_max_abs_error": errors, "verified_rows": len(X),
                "sha256": {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in files}}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


def load_model_archive(folder):
    """Load only trusted local archives (joblib is a pickle-based format)."""
    folder = Path(folder)

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"__keras_file__"}:
                from tensorflow import keras
                return keras.models.load_model(folder / value["__keras_file__"], compile=False)
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return decode(joblib.load(folder / "model.joblib"))
