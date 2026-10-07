"""입력 계약 — 이력·계획 스키마와 검증 규칙.

컬럼명은 원본 KAMP CSV(한글)와 같다. JSON API 는 ASCII 별칭(`HISTORY_ALIASES`)을 쓴다.

| 입력 | 내용 |
|---|---|
| 이력(history) | 대상일 전날 23:00 에서 끝나는 **연속 시간별** 실측. 하루 단위로 완결. 최소 8일, 권장 14일 |
| 계획(plan) | 대상일 24행 — ERP 생산계획·공장인원·기상예보. **전력 컬럼이 있으면 거부**(누수 방지) |
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import ServingInputError

POWER_COLUMNS = ["15분", "30분", "45분", "60분", "평균"]
HISTORY_COLUMNS = ["날짜", "시간", *POWER_COLUMNS,
                   "생산량", "공장인원", "인건비", "기온", "풍속", "습도", "강수량"]
PLAN_COLUMNS = ["날짜", "시간", "생산량", "공장인원", "기온", "풍속", "습도", "강수량"]
# 결측을 허용하는 컬럼 — 원자료에도 결측이 있고, 정비 규칙(보간·0 대체)이 정해져 있다
NULLABLE = {"풍속", "강수량", "공장인원"}
# 계획에 있으면 안 되는 컬럼 — 대상일 전력을 알고 있다는 뜻이므로 누수다
FORBIDDEN_IN_PLAN = set(POWER_COLUMNS) | {"y_avg", "y_peak", "y_cls"}

# JSON API 의 ASCII 필드 → 원본 한글 컬럼
HISTORY_ALIASES = {
    "date": "날짜", "hour": "시간", "p15": "15분", "p30": "30분", "p45": "45분", "p60": "60분",
    "avg": "평균", "prod": "생산량", "headcount": "공장인원", "labor": "인건비",
    "temp": "기온", "wind": "풍속", "humid": "습도", "rain": "강수량",
}
PLAN_ALIASES = {k: v for k, v in HISTORY_ALIASES.items() if v in PLAN_COLUMNS}


def _coerce(df: pd.DataFrame, columns: list[str], what: str) -> pd.DataFrame:
    """필수 컬럼 존재·숫자형·유한값을 확인하고 필요한 컬럼만 남긴다."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ServingInputError(f"{what.upper()}_SCHEMA", f"{what} 필수 컬럼 누락: {missing}")
    out = df[columns].copy()
    for c in columns:
        try:
            out[c] = pd.to_numeric(out[c], errors="raise")
        except (ValueError, TypeError) as e:
            raise ServingInputError(f"{what.upper()}_SCHEMA", f"{what} 컬럼 '{c}' 가 숫자가 아니다") from e
        vals = out[c].to_numpy(dtype=float)
        if np.isinf(vals).any():
            raise ServingInputError(f"{what.upper()}_SCHEMA", f"{what} 컬럼 '{c}' 에 무한대 값이 있다")
        if c not in NULLABLE and np.isnan(vals).any():
            raise ServingInputError(f"{what.upper()}_SCHEMA", f"{what} 컬럼 '{c}' 에 결측이 있다")
    for c in ("날짜", "시간"):
        if (out[c] % 1 != 0).any():
            raise ServingInputError(f"{what.upper()}_SCHEMA", f"{what} 컬럼 '{c}' 는 정수여야 한다")
        out[c] = out[c].astype("int64")
    # 20210931 같은 존재하지 않는 날짜는 범위 검사를 통과하므로 달력으로 확인한다
    parsed = pd.to_datetime(out["날짜"].astype(str), format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        bad = sorted(set(out.loc[parsed.isna(), "날짜"].astype(int)))[:3]
        raise ServingInputError(f"{what.upper()}_SCHEMA", f"{what} 에 존재하지 않는 날짜가 있다: {bad}")
    return out


def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    """이력 스키마를 확인하고 필요한 컬럼만 원래 행 순서로 돌려준다."""
    if len(history) == 0:
        raise ServingInputError("HISTORY_SCHEMA", "이력이 비어 있다")
    return _coerce(history, HISTORY_COLUMNS, "history").reset_index(drop=True)


def normalize_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """계획 스키마를 확인한다. 전력 컬럼이 있으면 거부한다."""
    leaked = sorted(FORBIDDEN_IN_PLAN & set(plan.columns))
    if leaked:
        raise ServingInputError(
            "PLAN_HAS_POWER",
            f"계획에 대상일 전력 컬럼이 있다(누수): {leaked} — 계획은 생산·인원·기상만 받는다",
        )
    out = _coerce(plan, PLAN_COLUMNS, "plan").reset_index(drop=True)
    if len(out) != 24 or out["날짜"].nunique() != 1 or sorted(out["시간"]) != list(range(24)):
        raise ServingInputError("PLAN_INVALID", "계획은 대상일 하루의 0~23시 24행이어야 한다")
    # 결측 허용 컬럼이라도 24시간 전부 비면 보간이 전날 값을 하루 종일 복사할 뿐이다 → 거부
    empty = [c for c in NULLABLE if out[c].isna().all()]
    if empty:
        raise ServingInputError("PLAN_SCHEMA", f"계획 컬럼이 24시간 전부 결측이다: {sorted(empty)}")
    return out.sort_values("시간").reset_index(drop=True)


def target_of(plan: pd.DataFrame, target_date=None) -> pd.Timestamp:
    """계획의 날짜 = 대상일. `target_date` 가 주어지면 일치해야 한다."""
    t = pd.to_datetime(str(int(plan["날짜"].iloc[0])), format="%Y%m%d")
    if target_date is not None and pd.Timestamp(target_date).normalize() != t:
        raise ServingInputError("PLAN_INVALID", f"target_date({pd.Timestamp(target_date).date()}) 와 계획 날짜({t.date()})가 다르다")
    return t


def validate_history_index(idx: pd.DatetimeIndex, target: pd.Timestamp,
                           min_days: int, max_days: int | None = None) -> None:
    """이력이 대상일 전날 23:00 에서 끝나는 연속·완결 시간 격자인지 확인한다."""
    if not idx.is_unique:
        raise ServingInputError("HISTORY_INVALID", "이력에 중복 시각이 있다")
    end = target - pd.Timedelta(hours=1)
    if idx.max() != end:
        raise ServingInputError(
            "HISTORY_INVALID", f"이력은 대상일 전날 23:00({end}) 에서 끝나야 한다 (현재 {idx.max()})"
        )
    start = idx.min()
    if start.hour != 0:
        raise ServingInputError("HISTORY_INVALID", "이력은 하루 단위로 완결돼야 한다(첫 행이 00시가 아님)")
    expected = pd.date_range(start, end, freq="h")
    if len(expected) != len(idx) or not (expected == idx).all():
        raise ServingInputError("HISTORY_INVALID", "이력에 빠진 시각이 있다(1시간 간격 연속이어야 한다)")
    n_days = len(idx) // 24
    if n_days < min_days:
        raise ServingInputError(
            "HISTORY_TOO_SHORT", f"이력이 {n_days}일이다 — 최소 {min_days}일 필요",
            {"required_start": str((target - pd.Timedelta(days=min_days)).date())},
        )
    if max_days is not None and n_days > max_days:
        raise ServingInputError("HISTORY_TOO_LONG", f"이력이 {n_days}일이다 — 최대 {max_days}일")
