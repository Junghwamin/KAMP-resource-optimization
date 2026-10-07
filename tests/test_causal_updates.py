"""Regression tests for past-only repair, including the August outage boundary."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def repair_functions():
    path = ROOT / "src" / "s01_diagnose.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"restore_hours", "build_datetime_index", "flag_and_repair", "add_targets"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"np": np, "pd": pd, "POWER_COLS": ["15분", "30분", "45분", "60분"]}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_repair_does_not_use_next_day_observations():
    ns = repair_functions()
    raw = pd.read_csv(ROOT / "data" / "okm_augumented_2021.csv", encoding="utf-8-sig")
    indexed = ns["build_datetime_index"](ns["restore_hours"](raw))
    origin = pd.Timestamp("2021-08-29")
    full, _ = ns["flag_and_repair"](indexed)
    history, _ = ns["flag_and_repair"](indexed.loc[indexed.index < origin])
    cols = ns["POWER_COLS"] + ["평균", "풍속", "공장인원"]
    np.testing.assert_array_equal(full.loc[history.index, cols], history[cols])


def test_missing_values_use_previous_value_not_later_observation():
    ns = repair_functions()
    index = pd.date_range("2021-01-01", periods=4, freq="h")
    frame = pd.DataFrame({c: [10., 0., 100., 110.] for c in ns["POWER_COLS"]}, index=index)
    frame["평균"] = [10., 0., 100., 110.]
    frame["생산량"] = 100.
    frame["인건비"] = 1.
    frame["풍속"] = [2., np.nan, 40., 50.]
    frame["공장인원"] = [3., np.nan, 30., 40.]
    frame["강수량"] = 0.
    repaired, _ = ns["flag_and_repair"](frame)
    assert repaired.loc[index[1], "평균"] == 10.
    assert repaired.loc[index[1], "풍속"] == 2.
    assert repaired.loc[index[1], "공장인원"] == 3.
    assert bool(repaired.loc[index[1], "is_outage"])


def test_leading_outage_is_not_backfilled_from_the_future():
    ns = repair_functions()
    index = pd.date_range("2021-01-01", periods=2, freq="h")
    frame = pd.DataFrame({c: [0., 50.] for c in ns["POWER_COLS"]}, index=index)
    for column, values in {"평균": [0., 50.], "생산량": [10., 10.], "인건비": [1., 1.],
                           "풍속": [np.nan, 5.], "공장인원": [np.nan, 3.], "강수량": [0., 0.]}.items():
        frame[column] = values
    repaired, _ = ns["flag_and_repair"](frame)
    assert repaired.loc[index[0], ns["POWER_COLS"] + ["평균", "풍속", "공장인원"]].isna().all()


@pytest.mark.parametrize("target", ["2021-08-29", "2021-09-01", "2021-09-14"])
def test_batch_and_serving_use_the_same_causal_features(target):
    from serving import _core
    from serving.contract import PLAN_COLUMNS
    from serving.pipeline import build_day_ahead_features

    raw = pd.read_csv(ROOT / "data" / "okm_augumented_2021.csv", encoding="utf-8-sig")
    index_frame = _core.build_datetime_index(_core.restore_hours(raw))
    full, _ = _core.flag_and_repair(index_frame)
    full = _core.add_targets(full)
    base = _core.build_features(full, _core.build_operating_calendar(full), 187.)
    target = pd.Timestamp(target)
    history = index_frame.loc[index_frame.index < target].reset_index(drop=True)
    plan = index_frame.loc[index_frame.index.normalize() == target, PLAN_COLUMNS].reset_index(drop=True)
    got, _ = build_day_ahead_features(history, plan, columns=_core.FEATURE_COLS,
                                      theta=187., holidays=_core.HOLIDAYS_2021, min_days=8)
    expected = base.loc[base.index.normalize() == target, _core.FEATURE_COLS]
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)
