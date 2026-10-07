"""모델 번들 로더 — 무결성·호환성·자가검증을 통과해야만 예측기를 돌려준다.

로드 순서(하나라도 실패하면 `BundleError`):

1. 매니페스트를 바이트로 읽는다 (UTF-8 명시 — Windows 기본 인코딩 cp949 에 의존하지 않는다)
2. 스키마 버전 확인, FAST(테스트) 번들 거부
3. 파일명 화이트리스트(`^[a-z0-9][a-z0-9_.-]*$`) — 번들 디렉터리 밖을 가리키지 못하게 한다
4. 파일마다 바이트 → sha256 대조 → `lgb.Booster(model_str=...)`
   (파일 경로로 여는 LightGBM API 는 한글이 든 절대경로를 열지 못한다. 한 번 읽은 바이트로
   검증과 파싱을 모두 하므로 검증 뒤 파일이 바뀌는 틈도 없다)
5. 각 부스터의 피처 이름 == 매니페스트 계약
6. lightgbm·numpy·pandas 버전이 학습 때와 정확히 같은지 (`KAMP_ALLOW_VERSION_MISMATCH=1` 로만 우회)
7. golden 자가검증 — 원시 이력으로 피처를 다시 만들어 기대 피처와, 예측을 기대 출력과 비교
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import _core
from .errors import BundleError, ServingInputError

SCHEMA_VERSION = 1
# 순수 파일명만 — 경로 구분자 없음, 점으로 시작 금지('..'·숨김파일 차단)
FILENAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
VERSION_KEYS = ("lightgbm", "numpy", "pandas")
DEFAULT_HOLIDAYS_FILE = Path(__file__).resolve().parent / "config" / "holidays_kr.json"


def _versions_now() -> dict:
    return {"lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__}


def load_holiday_calendar(path=None) -> dict:
    """운영 휴일표 `{coverage_years: [...], dates: [...]}` 를 **기동 시점에** 검증하며 읽는다.

    명시한 경로(인자·`KAMP_HOLIDAYS_FILE`)가 없거나 형식이 틀리면 `BundleError(HOLIDAYS_INVALID)`.
    기본 파일(패키지 동봉)이 없을 때만 빈 표로 둔다. 잘못된 날짜가 요청 때마다 500 이 되지 않게 한다.
    """
    explicit = path or os.environ.get("KAMP_HOLIDAYS_FILE")
    p = Path(explicit) if explicit else DEFAULT_HOLIDAYS_FILE
    if not p.is_file():
        if explicit:
            _fail("HOLIDAYS_INVALID", f"휴일표 파일이 없다: {p.name}")
        return {"coverage_years": [], "dates": []}
    try:
        data = json.loads(p.read_bytes())
        years, dates = data.get("coverage_years", []), data.get("dates", [])
        if not isinstance(years, list) or not isinstance(dates, list):
            raise TypeError("coverage_years·dates 는 목록이어야 한다")
        years = [int(y) for y in years]
        parsed = pd.to_datetime([str(d) for d in dates], format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError, AttributeError, UnicodeDecodeError) as e:
        _fail("HOLIDAYS_INVALID", f"휴일표 형식 오류({p.name}): {type(e).__name__}")
    return {"coverage_years": years, "dates": [str(d.date()) for d in parsed]}


@dataclass
class Bundle:
    """검증된 모델 번들. 예측은 `predict_features` / `predict_day` 로 한다."""

    name: str
    manifest: dict
    parts: dict
    calibrator: dict
    holiday_calendar: dict = field(default_factory=lambda: {"coverage_years": [], "dates": []})

    # ── 매니페스트 조회 ──────────────────────────────────────────────────
    @property
    def bundle_id(self) -> str:
        return self.manifest["bundle_id"]

    @property
    def role(self) -> str:
        return self.manifest["role"]

    @property
    def columns(self) -> list[str]:
        return list(self.manifest["feature_contract"]["columns"])

    @property
    def theta(self) -> float:
        return float(self.manifest["feature_contract"]["theta"])

    @property
    def tau(self) -> float:
        return float(self.manifest["thresholds"]["tau"]["value"])

    @property
    def tau_cls(self) -> float:
        return float(self.manifest["thresholds"]["tau_cls"]["value"])

    @property
    def halfwidths(self) -> dict:
        return self.manifest["intervals"]["residual"]

    @property
    def history_days(self) -> dict:
        return dict(self.manifest["feature_contract"]["history_days"])

    # ── 휴일 ─────────────────────────────────────────────────────────────
    def holidays_for(self, target) -> pd.DatetimeIndex:
        """대상일에 쓸 공휴일 목록. 대상일이 어떤 휴일표로도 덮이지 않으면 거부한다.

        학습 휴일표(HOLIDAYS_2021)는 학습 데이터 끝(2021-09-14)까지만 검증됐다.
        그 뒤 날짜는 운영 휴일표(`config/holidays_kr.json`)의 연도 커버리지가 있어야 한다.
        """
        t = pd.Timestamp(target).normalize()
        fc = self.manifest["feature_contract"]
        in_training = pd.Timestamp(fc["holiday_coverage_start"]) <= t <= pd.Timestamp(fc["holiday_coverage_end"])
        covered = in_training or t.year in self.holiday_calendar["coverage_years"]
        if not covered:
            raise ServingInputError(
                "CALENDAR_NOT_COVERED",
                f"{t.date()} 은 공휴일표 커버리지 밖이다 — config/holidays_kr.json 에 {t.year}년 공휴일을 추가하라",
            )
        return pd.DatetimeIndex(pd.to_datetime(sorted(set(fc["holidays"]) | set(self.holiday_calendar["dates"]))))

    # ── 예측 ─────────────────────────────────────────────────────────────
    def predict_features(self, X: pd.DataFrame) -> dict:
        """피처 행렬 → 번들 예측(dict). 컬럼 순서·dtype 은 내부에서 강제한다."""
        return _core.bundle_predict(self.parts, X)

    def predict_day(self, history: pd.DataFrame, plan: pd.DataFrame, *, target_date=None,
                    max_days: int | None = None) -> tuple[pd.DataFrame, dict]:
        """이력 + 계획(원본 한글 컬럼) → 익일 24시간 예측과 메타데이터."""
        from .contract import normalize_plan, target_of
        from .pipeline import assemble_output, build_day_ahead_features

        target = target_of(normalize_plan(plan), target_date)
        holidays = self.holidays_for(target)
        X, info = build_day_ahead_features(
            history, plan, columns=self.columns, theta=self.theta, holidays=holidays,
            min_days=int(self.history_days["min"]), max_days=max_days, target_date=target,
        )
        out = assemble_output(
            X, self.predict_features(X), tau=self.tau, tau_cls=self.tau_cls,
            calibrator=self.calibrator, halfwidths=self.halfwidths,
        )
        meta = {"bundle_id": self.bundle_id, "role": self.role, **info}
        return out, meta


def _fail(code: str, msg: str):
    raise BundleError(code, msg)


_SCHEMA_ERRORS = (KeyError, TypeError, ValueError, AttributeError, IndexError, UnicodeDecodeError)


def load_bundle(bundle_dir, *, allow_fast: bool = False, check_versions: bool = True,
                selftest: bool = True, holidays_file=None) -> Bundle:
    """번들 디렉터리를 검증하며 읽는다. 실패하면 `BundleError`.

    Parameters
    ----------
    allow_fast : bool
        FAST(테스트용 축소 학습) 번들 허용. **테스트에서만** 쓴다. API 는 이 스위치를 노출하지 않는다.
    """
    d = Path(bundle_dir)
    name = d.name   # 메시지에는 디렉터리 이름만 — 절대경로(계정명)를 흘리지 않는다
    mpath = d / "manifest.json"
    if not mpath.is_file():
        _fail("BUNDLE_NOT_FOUND", f"번들 매니페스트가 없다: {name}/manifest.json")
    try:
        man = json.loads(mpath.read_bytes())
    except (ValueError, UnicodeDecodeError):
        _fail("BUNDLE_SCHEMA", f"매니페스트를 읽을 수 없다: {name}")
    if not isinstance(man, dict) or not isinstance(man.get("files_sha256"), dict):
        _fail("BUNDLE_SCHEMA", "매니페스트 형식이 아니다(객체·files_sha256 필요)")
    if man.get("schema_version") != SCHEMA_VERSION:
        _fail("BUNDLE_SCHEMA", f"지원하지 않는 번들 스키마: {man.get('schema_version')}")
    if man.get("fast_mode") and not allow_fast:
        _fail("FAST_BUNDLE", "FAST(테스트용 축소 학습) 번들은 서비스에 쓸 수 없다 — FULL 실행 번들을 쓰라")

    if check_versions and os.environ.get("KAMP_ALLOW_VERSION_MISMATCH") != "1":
        want = {k: man.get("runtime", {}).get(k) for k in VERSION_KEYS}
        have = _versions_now()
        diff = {k: f"번들 {want[k]} ≠ 현재 {have[k]}" for k in VERSION_KEYS if want[k] != have[k]}
        if diff:
            _fail("VERSION_MISMATCH", f"라이브러리 버전이 학습 때와 다르다: {diff}")

    # 매니페스트 자신의 무결성 — τ·구간 반폭처럼 매니페스트 안에만 있는 값의 변조를 잡는다
    try:
        core_ok = _core.manifest_core_sha256(man) == man["manifest_sha256"] and str(
            man["bundle_id"]).endswith(str(man["manifest_sha256"])[:12])
    except _SCHEMA_ERRORS:
        core_ok = False
    if not core_ok:
        _fail("BUNDLE_INTEGRITY", "매니페스트 sha256 불일치(손상·변조) — 노트북이 만든 원본을 쓰라")

    raw: dict[str, bytes] = {}
    for fname, sha in man["files_sha256"].items():
        if not isinstance(fname, str) or not FILENAME_RE.fullmatch(fname):
            _fail("BUNDLE_INTEGRITY", f"허용되지 않는 파일명: {fname!r}")
        p = d / fname
        if not p.is_file():
            _fail("BUNDLE_INTEGRITY", f"번들 파일이 없다: {fname}")
        b = p.read_bytes()
        if hashlib.sha256(b).hexdigest() != sha:
            _fail("BUNDLE_INTEGRITY", f"sha256 불일치(손상·변조): {fname}")
        raw[fname] = b

    def blob(fname: str) -> bytes:
        if fname not in raw:
            _fail("BUNDLE_INTEGRITY", f"매니페스트가 참조하는 파일이 무결성 목록에 없다: {fname}")
        return raw[fname]

    boosters: dict[str, lgb.Booster] = {}

    def booster(fname: str, expect_features: list[str]) -> lgb.Booster:
        if fname not in boosters:
            try:
                boosters[fname] = lgb.Booster(model_str=blob(fname).decode("utf-8"))
            except Exception:
                _fail("BUNDLE_INTEGRITY", f"LightGBM 모델을 파싱할 수 없다: {fname}")
        if boosters[fname].feature_name() != list(expect_features):
            _fail("BUNDLE_CONTRACT", f"피처 이름·순서가 매니페스트와 다르다: {fname}")
        return boosters[fname]

    try:
        cols = list(man["feature_contract"]["columns"])
        comp = man["components"]
        gate_cols = list(comp["gate"]["features"])
        parts = {
            "columns": cols,
            "gate": booster(comp["gate"]["file"], gate_cols),
            "gate_features": gate_cols,
            "classes": [int(c) for c in comp["gate"]["classes"]],
            "regs": {t: {int(r): booster(f, cols) for r, f in d_.items()} for t, d_ in comp["regressors"].items()},
            "fallbacks": {t: booster(f, cols) for t, f in comp["fallbacks"].items()},
            "peak_clf": booster(comp["peak_classifier"]["file"], cols),
            "quantiles": {float(q): booster(f, cols) for q, f in comp["quantiles"].items()},
        }
        calibrator = json.loads(blob(comp["calibrator"]))
        golden = json.loads(blob(comp["golden"]))
        for key in ("x_thresholds", "y_thresholds"):   # 필수 키 확인
            calibrator[key]
    except BundleError:
        raise
    except _SCHEMA_ERRORS as e:
        _fail("BUNDLE_SCHEMA", f"매니페스트·번들 JSON 형식 오류: {type(e).__name__}")
    # 게이트 classes 개수 == 게이트 부스터 출력 수 (2분류는 확률 1열)
    n_out = parts["gate"].num_model_per_iteration()
    if len(parts["classes"]) != (2 if n_out == 1 else n_out):
        _fail("BUNDLE_CONTRACT", "게이트 classes 개수가 게이트 모델 출력과 다르다")

    bundle = Bundle(name=name, manifest=man, parts=parts, calibrator=calibrator,
                    holiday_calendar=load_holiday_calendar(holidays_file))
    if selftest:
        run_selftest(bundle, golden)
    return bundle


def run_selftest(bundle: Bundle, golden: dict) -> dict:
    """golden 사례로 피처 코드와 부스터를 함께 검증한다. 실패하면 `BundleError`.

    - 피처: 원시 이력 21일로 다시 만든 24×44 가 기대 피처와 같아야 한다. `roll*_avg_std` 2개만
      pandas rolling 의 누적 오차(창 시작점 의존, ~1e-13) 때문에 rtol 1e-9 로 비교한다.
    - 예측: 기대 피처로 예측한 값이 기대 출력과 같아야 한다. 회귀·레짐은 정확 일치,
      확률은 OS 별 exp 구현 차이를 감안해 rtol 1e-12.
    """
    from .pipeline import build_day_ahead_features

    cols = bundle.columns
    try:
        hist = pd.DataFrame(golden["history"]["rows"], columns=golden["history"]["columns"])
        plan = pd.DataFrame(golden["plan"]["rows"], columns=golden["plan"]["columns"])
        target = pd.Timestamp(golden["target_date"])
        want_X = np.asarray(golden["expected_features"]["rows"], dtype=float)
        exp = {k: golden["expected"][k] for k in ("regime", "y_avg", "y_peak", "q10", "q50", "q90", "prob")}
    except _SCHEMA_ERRORS as e:
        _fail("BUNDLE_SCHEMA", f"golden 자가검증 사례 형식 오류: {type(e).__name__}")
    try:
        X, _ = build_day_ahead_features(
            hist, plan, columns=cols, theta=bundle.theta,
            holidays=pd.to_datetime(bundle.manifest["feature_contract"]["holidays"]),
            min_days=int(bundle.history_days["min"]), target_date=target,
        )
    except ServingInputError as e:
        _fail("SELFTEST_FEATURES", f"자가검증 사례로 피처를 만들 수 없다: {e.message}")
    got_X = X[cols].to_numpy(dtype=float)
    if want_X.shape != got_X.shape:
        _fail("SELFTEST_FEATURES", "golden 기대 피처의 형태가 다르다")
    std_idx = [i for i, c in enumerate(cols) if c.endswith("_std")]
    exact_idx = [i for i in range(len(cols)) if i not in std_idx]
    if not np.array_equal(got_X[:, exact_idx], want_X[:, exact_idx]) or not np.allclose(
        got_X[:, std_idx], want_X[:, std_idx], rtol=1e-9, atol=0
    ):
        bad = [cols[i] for i in range(len(cols)) if not np.allclose(got_X[:, i], want_X[:, i], rtol=1e-9, atol=0)]
        _fail("SELFTEST_FEATURES", f"피처 코드가 학습 때와 다른 값을 만든다: {bad[:5]}")

    pred = bundle.predict_features(pd.DataFrame(want_X, columns=cols))
    for k in ("regime", "y_avg", "y_peak", "q10", "q50", "q90"):
        if not np.array_equal(np.asarray(pred[k]), np.asarray(exp[k])):
            _fail("SELFTEST_PREDICTION", f"번들 예측이 기대 출력과 다르다: {k}")
    if np.shape(exp["prob"]) != np.shape(pred["prob"]) or not np.allclose(
        pred["prob"], exp["prob"], rtol=1e-12, atol=0
    ):
        _fail("SELFTEST_PREDICTION", "번들 예측이 기대 출력과 다르다: prob")
    return {"target_date": str(target.date()), "features": "ok", "prediction": "ok"}
