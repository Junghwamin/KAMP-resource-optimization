"""배치 CLI — 매일 24:00 익일 예측, 과거 구간 리플레이, 번들 검증.

    python -X utf8 -m serving predict --bundle DIR --history h.csv --plan p.csv --out pred.csv
    python -X utf8 -m serving replay  --bundle DIR --data data/okm_augumented_2021.csv \\
                                      --start 2021-09-01 --end 2021-09-14 [--window-days 14]
    python -X utf8 -m serving verify  --bundle DIR

종료코드: 0 정상 · 2 입력 오류 · 3 번들 오류. CSV 입출력은 utf-8-sig(Excel 한글 호환).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .bundle import load_bundle
from .contract import PLAN_COLUMNS, POWER_COLUMNS
from .errors import ServingError


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _write_csv(df: pd.DataFrame, path: str | None) -> None:
    if path:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        sys.stdout.write(df.to_csv(index=False))


def cmd_predict(args) -> int:
    bundle = load_bundle(args.bundle, allow_fast=args.allow_fast, holidays_file=args.holidays)
    out, meta = bundle.predict_day(_read_csv(args.history), _read_csv(args.plan))
    _write_csv(out, args.out)
    for w in meta["warnings"]:
        print(f"경고: {w}", file=sys.stderr)
    print(f"[OK] {meta['target_date']} 24시간 예측 · 번들 {meta['bundle_id']}", file=sys.stderr)
    return 0


def replay(bundle, raw: pd.DataFrame, start, end, window_days: int | None = None) -> pd.DataFrame:
    """원자료로 과거 구간을 하루씩 재생한다 — 각 대상일은 **그 전날까지의 이력**만 본다.

    `window_days=None` 이면 원자료 첫날부터의 이력 전체(배치 파이프라인과 비트 동일).
    """
    ymd = raw["날짜"].astype(int)
    first_day = pd.to_datetime(str(ymd.min()), format="%Y%m%d")
    parts = []
    for target in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"):
        origin = target - pd.Timedelta(days=1)
        lo = first_day if window_days is None else max(first_day, origin - pd.Timedelta(days=window_days - 1))
        hist = raw[(ymd >= int(lo.strftime("%Y%m%d"))) & (ymd <= int(origin.strftime("%Y%m%d")))]
        day = raw[ymd == int(target.strftime("%Y%m%d"))]
        out, _ = bundle.predict_day(hist, day[PLAN_COLUMNS])
        # 원자료에는 대상일 실측이 있으므로 비교용으로 붙인다(예측에는 쓰지 않았다)
        out.insert(1, "y_avg_true", day["평균"].to_numpy(dtype=float))
        out.insert(2, "y_peak_true", day[POWER_COLUMNS[:4]].max(axis=1).to_numpy(dtype=float))
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def cmd_replay(args) -> int:
    bundle = load_bundle(args.bundle, allow_fast=args.allow_fast, holidays_file=args.holidays)
    window = None if args.window_days in (None, "all") else int(args.window_days)
    out = replay(bundle, _read_csv(args.data), args.start, args.end, window)
    _write_csv(out, args.out)
    mae = float(np.mean(np.abs(out["y_avg_true"] - out["y_avg_pred"])))
    print(f"[OK] {args.start}~{args.end} {len(out)}행 · MAE {mae:.4f} · 번들 {bundle.bundle_id}", file=sys.stderr)
    return 0


def cmd_verify(args) -> int:
    bundle = load_bundle(args.bundle, allow_fast=args.allow_fast, holidays_file=args.holidays)
    m = bundle.manifest
    print(json.dumps({
        "status": "ok", "bundle_id": bundle.bundle_id, "role": bundle.role,
        "model_name": m["model_name"], "train_last": m["training"]["last"],
        "n_train_rows": m["training"]["n_rows"], "tau": bundle.tau, "tau_cls": bundle.tau_cls,
        "selftest": "ok",
    }, ensure_ascii=False, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m serving", description="KAMP 모델 번들 서빙 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--bundle", required=True, help="번들 디렉터리 (manifest.json 이 있는 곳)")
        p.add_argument("--holidays", default=None, help="운영 휴일표 JSON (기본 serving/config/holidays_kr.json)")
        p.add_argument("--allow-fast", action="store_true", help=argparse.SUPPRESS)  # 테스트 전용

    p = sub.add_parser("predict", help="이력 + 계획 → 익일 24시간 예측")
    common(p)
    p.add_argument("--history", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("replay", help="원자료로 과거 구간 재생(하루씩, 전날까지 이력만)")
    common(p)
    p.add_argument("--data", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--window-days", default=None, help="이력 창 일수 (기본: 원자료 첫날부터 전체)")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("verify", help="번들 무결성·호환성·자가검증")
    common(p)
    p.set_defaults(func=cmd_verify)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ServingError as e:
        print(f"[{e.code}] {e.message}", file=sys.stderr)
        if e.detail:
            print(json.dumps(e.detail, ensure_ascii=False), file=sys.stderr)
        return e.exit_code
