"""E2: paired, noncircular consecutive-day moving-block sensitivity."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import sha256_file, validate_times, write_csv, write_json


def moving_day_indices(days: pd.DatetimeIndex, block_days: int, rng) -> np.ndarray:
    """Sample valid consecutive blocks, truncate excess days, never wrap boundaries."""
    if block_days < 1 or block_days > len(days):
        raise ValueError("Block length must be between 1 and number of dates")
    valid = [i for i in range(len(days) - block_days + 1)
             if (days[i + block_days - 1] - days[i]).days == block_days - 1]
    if not valid:
        raise ValueError("No calendar-contiguous block of the requested length")
    # For length 1 this is exactly the original rng.integers(0, nb, nb) algorithm.
    starts = np.asarray(valid)[rng.integers(0, len(valid), int(np.ceil(len(days) / block_days)))]
    return np.concatenate([np.arange(s, s + block_days) for s in starts])[:len(days)]


def paired_block_bootstrap(index, y, pred_rf, pred_final, *, block_days=1, n_boot=2000, seed=42) -> dict:
    idx = validate_times(index)
    y, pa, pb = [np.asarray(a, dtype=float) for a in (y, pred_rf, pred_final)]
    if y.ndim != 1 or len(y) != len(idx) or pa.shape != y.shape or pb.shape != y.shape:
        raise ValueError("Truth and predictions must match timestamps")
    if not all(np.isfinite(a).all() for a in (y, pa, pb)) or n_boot < 1:
        raise ValueError("Finite values and positive bootstrap count required")
    ea, eb = np.abs(y - pa), np.abs(y - pb)
    norm = idx.normalize()
    days = norm.unique()
    blocks = [np.flatnonzero(norm == d) for d in days]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        drawn_days = moving_day_indices(days, block_days, rng)
        take = np.concatenate([blocks[j] for j in drawn_days])
        stats[i] = ea[take].mean() - eb[take].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    lo90, hi90 = np.percentile(stats, [5, 95])
    return {"block_days": block_days, "n_boot": n_boot, "seed": seed, "n_hours": len(idx),
            "n_days": len(days), "rf_mae": float(ea.mean()), "final_mae": float(eb.mean()),
            "diff": float(ea.mean() - eb.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "ci90_low": float(lo90), "ci90_high": float(hi90),
            "p_value_approx": float(min(1, 2 * min((stats >= 0).mean(), (stats <= 0).mean()))),
            "ci_includes_zero": bool(lo <= 0 <= hi), "p_method": "uncentered two-sided bootstrap tail proportion (approximate)",
            "evaluation_role": "frozen_reference_sensitivity", "sampling": "noncircular moving calendar-day blocks; truncate excess days"}


def run(root: Path, snapshot: Path, output: Path, config: dict) -> dict:
    source = Path(root) / "outputs/tables/F18_src.csv"
    data = pd.read_csv(source, encoding="utf-8-sig")
    idx = validate_times(data["datetime"])
    y, pa, pb = (data[c].to_numpy(float) for c in ["실측", "Random Forest (보정)", "2단계 레짐(3분류)"])
    cfg = config.get("bootstrap", {})
    rows = [paired_block_bootstrap(idx, y, pa, pb, block_days=int(length), n_boot=int(cfg.get("n_boot", 2000)),
                                   seed=int(config.get("seed", 42))) for length in cfg.get("lengths", [1, 2, 3])]
    baseline = next((r for r in rows if r["block_days"] == 1), None)
    reference_verified = False
    if baseline and baseline["n_boot"] == 2000 and baseline["seed"] == 42:
        expected = {"final_mae": 5.3098695238, "rf_mae": 6.3293097664, "ci_low": 0.0084867001,
                    "ci_high": 2.5268885063, "p_value_approx": .046}
        for key, value in expected.items():
            if not np.isclose(baseline[key], value, atol=1e-8, rtol=0):
                raise ValueError(f"Frozen F18 one-day reference mismatch: {key}")
        reference_verified = True
    daily = pd.DataFrame({"date": idx.normalize(), "rf_abs_error": np.abs(y - pa), "final_abs_error": np.abs(y - pb)})
    daily = daily.groupby("date", as_index=False).agg(n_hours=("rf_abs_error", "size"), rf_mae=("rf_abs_error", "mean"), final_mae=("final_abs_error", "mean"))
    daily["diff"] = daily.rf_mae - daily.final_mae
    write_csv(output / "paired_daily_errors.csv", daily)
    write_csv(output / "bootstrap_block_sensitivity.csv", pd.DataFrame(rows))
    summary = {"experiment": "bootstrap", "status": "complete", "source_sha256": sha256_file(source),
               "one_day_reference_verified": reference_verified, "results": rows,
               "interpretation": "All preregistered block lengths retained; CI crossing zero is a valid result, not a failed check."}
    write_json(output / "summary.json", summary)
    return summary
