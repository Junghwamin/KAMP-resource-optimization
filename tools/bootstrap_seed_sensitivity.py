"""Seed and resample-count sensitivity of the E2 paired block bootstrap.

E2 is preregistered at 2,000 resamples with seed 42. When a 95% interval ends
close to zero, that choice alone can decide whether the interval excludes zero.
This read-only tool reruns the same statistic, MAE(RF) - MAE(final), with the
project's own ``paired_block_bootstrap`` for seeds 1-20 and once with 100,000
resamples, and writes one JSON. It fits no model and changes no E2 output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bootstrap import paired_block_bootstrap  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs/verification/bootstrap_seed_sensitivity.json"


def run(root: Path = ROOT, seeds=tuple(range(1, 21)), large: int = 100_000, large_seed: int = 42) -> dict:
    data = pd.read_csv(Path(root) / "outputs/tables/F18_src.csv", encoding="utf-8-sig")
    index = pd.DatetimeIndex(pd.to_datetime(data["datetime"]))
    columns = (data["실측"].to_numpy(float), data["Random Forest (보정)"].to_numpy(float),
               data["2단계 레짐(3분류)"].to_numpy(float))
    rows = []
    for block in (1, 2, 3):
        per_seed = [paired_block_bootstrap(index, *columns, block_days=block, n_boot=2000, seed=s) for s in seeds]
        big = paired_block_bootstrap(index, *columns, block_days=block, n_boot=large, seed=large_seed)
        rows.append({
            "block_days": block, "n_boot_per_seed": 2000, "seeds": list(seeds),
            "ci_includes_zero_count": int(sum(r["ci_includes_zero"] for r in per_seed)), "seed_count": len(per_seed),
            "ci_low_range": [min(r["ci_low"] for r in per_seed), max(r["ci_low"] for r in per_seed)],
            "p_value_range": [min(r["p_value_approx"] for r in per_seed), max(r["p_value_approx"] for r in per_seed)],
            "large_n_boot": large, "large_seed": large_seed, "large_ci": [big["ci_low"], big["ci_high"]],
            "large_p_value_approx": big["p_value_approx"], "large_ci_includes_zero": big["ci_includes_zero"],
            "diff": big["diff"],
        })
    return {"schema": "kamp.bootstrap_seed_sensitivity.v1", "source": "outputs/tables/F18_src.csv",
            "statistic": "MAE(RF) - MAE(final) on the same 336 hours; positive means the final model is better",
            "sampling": "noncircular moving calendar-day blocks, as in E2", "rows": rows,
            "interpretation": "The preregistered E2 result (2,000 resamples, seed 42) is kept; this file shows how "
                              "much the interval moves with the seed and with Monte Carlo error removed."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in result["rows"]:
        print(f"block={row['block_days']}: CI includes 0 for {row['ci_includes_zero_count']}/{row['seed_count']} seeds; "
              f"{row['large_n_boot']:,} resamples CI=[{row['large_ci'][0]:.4f}, {row['large_ci'][1]:.4f}] "
              f"p~{row['large_p_value_approx']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
