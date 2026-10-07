"""익일 24시간 예측 파이프라인 — 이력 + 계획 → 피처 → 번들 예측.

학습(노트북)과 **같은 원문 함수**(`_core`, src 에서 코드 생성)를 쓰되, 서빙 시점의
인과성을 지키기 위해 순서를 다음처럼 고정한다.

1. **이력에만** 정비(`restore_hours → build_datetime_index → flag_and_repair → add_targets`)를 한다.
   과거 관측값만으로 결측을 전방대체한다. 계획의 전력값은 끝까지 NaN으로 유지한다.
2. 계획 행은 전력 NaN · 결측 플래그 False · 강수량 결측 0 으로 둔다. 대상일의
   `is_erp_missing` 은 원점에서 판정할 수 없으므로 False 다.
3. 이어 붙인 뒤 가동 캘린더 → `build_features(theta=매니페스트 값)` → 대상 24행.
   θ 는 서빙에서 다시 계산하지 않는다(피처 정의의 일부다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _core
from .contract import (
    PLAN_COLUMNS,
    POWER_COLUMNS,
    normalize_history,
    normalize_plan,
    target_of,
    validate_history_index,
)
from .errors import PredictionError, ServingInputError

OUTPUT_COLUMNS = [
    "ts", "y_avg_pred", "y_peak_pred", "regime", "peak_label", "peak_prob", "peak_prob_cal",
    "peak_label_cls", "q10", "q50", "q90", "pi_lo", "pi_hi",
]


def build_day_ahead_features(history: pd.DataFrame, plan: pd.DataFrame, *, columns: list[str],
                             theta: float, holidays, min_days: int, max_days: int | None = None,
                             target_date=None) -> tuple[pd.DataFrame, dict]:
    """이력·계획(원본 한글 컬럼) → 대상일 24행 피처와 정보(dict).

    Raises
    ------
    ServingInputError
        스키마·시간격자·이력 길이·휴무 연속구간 절단·피처 결측.
    """
    hist = normalize_history(history)
    plan_n = normalize_plan(plan)
    target = target_of(plan_n, target_date)
    warnings: list[str] = []

    try:
        h = _core.restore_hours(hist)
    except ValueError as e:   # 손상 시간값인데 그 날이 24행이 아닌 경우
        raise ServingInputError("HISTORY_INVALID", str(e)) from e
    h = _core.build_datetime_index(h)
    validate_history_index(h.index, target, min_days, max_days)

    outage = h["평균"].to_numpy() == 0
    if outage[-1]:
        warnings.append("TRAILING_OUTAGE: 이력 끝이 계측정지(평균=0)다 — 마지막 유효값으로 채운다")
    if outage[0]:
        # 선두 계측정지는 과거 기준점이 없어 NaN으로 남는다.
        # 피처 참조 구간에 닿으면 더 이른 이력이 필요하다.
        first_valid = h.index[~outage][0] if (~outage).any() else h.index[-1]
        feature_start = target - pd.Timedelta(days=7)
        if first_valid > feature_start:
            raise ServingInputError(
                "HISTORY_EDGE_OUTAGE", "이력 시작의 계측정지 구간이 피처 참조 구간까지 이어진다",
                {"required_start": str((h.index[0] - pd.Timedelta(days=1)).date())},
            )
        warnings.append("LEADING_OUTAGE: 이력 시작이 계측정지다(피처 참조 구간 밖이라 영향 없음)")

    hp, _ = _core.flag_and_repair(h)
    hp = _core.add_targets(hp)

    pp = _core.build_datetime_index(plan_n[PLAN_COLUMNS])
    for c in POWER_COLUMNS:
        pp[c] = np.nan
    pp["y_avg"] = np.nan
    pp["y_peak"] = np.nan
    pp["is_outage"] = False
    pp["is_erp_missing"] = False
    pp["is_aug_inconsistent"] = False
    if pp["강수량"].isna().any():
        warnings.append("PLAN_RAIN_FILLED: 계획 강수량 결측을 0(무강수)으로 채웠다")
    pp["강수량"] = pp["강수량"].fillna(0.0)

    frame = pd.concat([hp, pp])
    for c in ("풍속", "공장인원"):
        if pp[c].isna().any():
            warnings.append(f"PLAN_FORWARD_FILLED: 계획 '{c}' 결측을 앞선 유효값으로 대체했다")
            frame[c] = frame[c].ffill()

    cal = _core.build_operating_calendar(frame)
    days = cal.index
    if bool(cal.loc[target, "is_shutdown"]) and bool(cal.loc[days[0]:target - pd.Timedelta(days=1), "is_shutdown"].all()):
        # 대상일이 속한 휴무 연속구간이 이력 시작에서 잘리면 '휴무 n일차' 를 셀 수 없다
        raise ServingInputError(
            "SHUTDOWN_RUN_TRUNCATED", "대상일의 휴무 연속구간이 이력 시작에서 잘렸다 — 더 이른 이력이 필요하다",
            {"required_start": str((days[0] - pd.Timedelta(days=7)).date())},
        )

    feats = _core.build_features(frame, cal, float(theta), holidays=holidays)
    X = feats.loc[feats.index.normalize() == target, list(columns)]
    arr = X.to_numpy(dtype=float)
    if X.shape[0] != 24 or not np.isfinite(arr).all():
        bad = [c for c in columns if not np.isfinite(X[c].to_numpy(dtype=float)).all()]
        raise ServingInputError("FEATURE_INVALID", f"대상일 피처에 결측이 있다: {bad[:5]}")
    info = {
        "target_date": str(target.date()),
        "window_start": str(h.index[0].date()),
        "history_days": int(len(h) // 24),
        "warnings": warnings,
    }
    return X, info


def assemble_output(X: pd.DataFrame, pred: dict, *, tau: float, tau_cls: float,
                    calibrator: dict, halfwidths: dict) -> pd.DataFrame:
    """번들 예측 → 서비스 출력 24행.

    - `peak_label` = `y_peak_pred >= τ` (제출 파일의 `peak_pred_label` 과 같은 규칙)
    - `peak_prob` = 보정 전 확률(제출 파일과 같음), `peak_prob_cal` = Isotonic 보정
      (`np.interp` 는 `IsotonicRegression(out_of_bounds='clip').predict` 와 정확히 같다)
    - `pi_lo/pi_hi` = 최종모델 OOF 잔차 기반 구간. 대상일 휴무 여부로 반폭을 고른다
    """
    y_avg, y_peak, prob = pred["y_avg"], pred["y_peak"], pred["prob"]
    shut = X["is_shutdown"].to_numpy() == 1
    q = np.where(shut, halfwidths["shutdown"]["qhat"], halfwidths["operating"]["qhat"])
    out = pd.DataFrame({
        "ts": X.index.strftime("%Y-%m-%d %H:%M:%S"),
        "y_avg_pred": y_avg,
        "y_peak_pred": y_peak,
        "regime": pred["regime"].astype(int),
        "peak_label": (y_peak >= tau).astype(int),
        "peak_prob": prob,
        "peak_prob_cal": np.interp(prob, calibrator["x_thresholds"], calibrator["y_thresholds"]),
        "peak_label_cls": (prob >= tau_cls).astype(int),
        "q10": pred["q10"], "q50": pred["q50"], "q90": pred["q90"],
        "pi_lo": np.maximum(y_avg - q, 0.0),
        "pi_hi": y_avg + q,
    })
    num = out.drop(columns=["ts"]).to_numpy(dtype=float)
    if not np.isfinite(num).all():
        raise PredictionError("PREDICTION_NONFINITE", "예측값에 NaN/inf 가 있다")
    return out[OUTPUT_COLUMNS]
