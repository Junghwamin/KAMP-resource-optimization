"""서빙 패키지 단위 테스트 — 합성 번들 + 실데이터 입력 계약.

파이프라인 fixture(s00~s10)를 쓰지 않는다. 번들은 `serving_helpers.make_synthetic_bundle` 로
tmp 디렉터리에 만들고(노트북 10.5절과 같은 매니페스트 스키마), 입력 계약은 원본 CSV 를
잘라 검증한다. 그래서 수 초 안에 끝나고 outputs/ 를 건드리지 않는다.

| 묶음 | 지키는 것 |
|---|---|
| 번들 로더 | 무결성(sha256)·FAST 거부·버전 고정·파일명 화이트리스트·피처 계약·경로 비노출 |
| 예측 | 컬럼 순서·dtype 강제, 레짐 라우팅과 fallback, 문자열 키 → int |
| 입력 계약 | 누수(계획의 전력 컬럼)·시간격자·이력 길이·휴무 연속구간 절단·계측정지 가장자리 |
| 출력 | τ 경보·Isotonic 보정·예측구간 반폭(휴무/운영) |
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

# serving/ 이 없는 환경(노트북·데이터만 받은 경우)에서는 이 모듈을 통째로 건너뛴다
pytest.importorskip("serving")

import serving_helpers as sh  # noqa: E402
from serving import BundleError, ServingInputError, _core, cli, load_bundle
from serving.errors import PredictionError
from serving.contract import PLAN_COLUMNS, POWER_COLUMNS
from serving.pipeline import OUTPUT_COLUMNS, assemble_output, build_day_ahead_features

PRED_KEYS = {"regime", "y_avg", "y_peak", "prob", "q10", "q50", "q90"}


# ══════════════════════════════════════════════════════════════════════
# 공통 도우미 · fixture
# ══════════════════════════════════════════════════════════════════════
def _same(a: dict, b: dict) -> bool:
    """두 예측 dict 가 키·값 모두 비트 단위로 같은지."""
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _arr(bundle, X: pd.DataFrame) -> np.ndarray:
    """`bundle_predict` 와 똑같이 컬럼 재정렬 + float64 로 만든 행렬."""
    return X.reindex(columns=bundle.columns).astype(np.float64).to_numpy()


def _assert_no_abs_path(err, tmp_path) -> None:
    """오류 메시지·봉투에 서버 절대경로가 없다(디렉터리 이름만 허용)."""
    texts = [str(err), err.message, json.dumps(err.to_dict(), ensure_ascii=False)]
    for path in {str(tmp_path), str(tmp_path.resolve()), tmp_path.as_posix()}:
        assert all(path not in t for t in texts), f"절대경로 노출: {texts}"
    assert all(tmp_path.name not in t for t in texts), "상위 디렉터리 이름까지 노출됐다"
    assert all(":\\" not in t and ":/" not in t for t in texts), "드라이브 경로 노출"


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> dict:
    """정상 합성 번들(레짐 0·1·2 회귀 모두 있음, FULL, golden 최소)."""
    return sh.make_synthetic_bundle(tmp_path_factory.mktemp("base") / "bundle")


@pytest.fixture(scope="module")
def bundle(base):
    """정상 합성 번들을 selftest 없이 로드한 것."""
    return load_bundle(base["dir"], selftest=False)


@pytest.fixture
def bundle_copy(base, tmp_path):
    """변조 테스트용 — 정상 번들의 사본 디렉터리(Path)."""
    d = tmp_path / "bundle"
    shutil.copytree(base["dir"], d)
    return d


@pytest.fixture(scope="module")
def drop1(tmp_path_factory):
    """레짐1 회귀기가 없는 번들(게이트는 3분류 그대로)."""
    info = sh.make_synthetic_bundle(tmp_path_factory.mktemp("drop1") / "bundle", drop_regime=1)
    return load_bundle(info["dir"], selftest=False)


@pytest.fixture(scope="module")
def golden_base(tmp_path_factory) -> dict:
    """실데이터 자가검증 사례(golden)를 가진 합성 번들 — selftest=True 로 열린다."""
    return sh.make_synthetic_bundle(tmp_path_factory.mktemp("golden") / "bundle", golden="real")


@pytest.fixture(scope="module")
def X30() -> pd.DataFrame:
    """레짐 0·1·2 가 10행씩 섞인 난수 피처 30행."""
    return sh.make_features(30, seed=7, regimes=[0, 1, 2] * 10)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """원본 KAMP CSV(utf-8-sig)."""
    return sh.load_raw()


def _days(raw: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """원자료에서 [lo, hi] 일자 행(원본 컬럼 전부)을 잘라 이력으로 쓴다."""
    ymd = raw["날짜"]
    return raw[(ymd >= lo) & (ymd <= hi)].reset_index(drop=True)


def _plan(raw: pd.DataFrame, day: int) -> pd.DataFrame:
    """원자료에서 대상일 24행의 계획 컬럼만 잘라 계획으로 쓴다."""
    return raw.loc[raw["날짜"] == day, PLAN_COLUMNS].reset_index(drop=True)


def _features(history, plan, **kw):
    """학습 계약(θ=187, HOLIDAYS_2021, 최소 8일)으로 대상일 피처를 만든다."""
    return build_day_ahead_features(
        history, plan, columns=_core.FEATURE_COLS, theta=187.0,
        holidays=pd.to_datetime(_core.HOLIDAYS_2021), min_days=8, **kw,
    )


def _input_error(code: str, history, plan, **kw) -> ServingInputError:
    """피처 생성이 주어진 코드의 입력 오류로 거부되는지 확인하고 예외를 돌려준다."""
    with pytest.raises(ServingInputError) as ei:
        _features(history, plan, **kw)
    assert ei.value.code == code, f"{ei.value.code} ≠ {code}: {ei.value.message}"
    assert ei.value.exit_code == 2
    return ei.value


# ══════════════════════════════════════════════════════════════════════
# 1. 정상 로드 · 예측 키
# ══════════════════════════════════════════════════════════════════════
def test_load_and_predict_features_keys_finite(bundle, X30):
    """정상 번들은 로드되고 예측 키 7개가 모두 유한값으로 나온다."""
    pred = bundle.predict_features(X30)
    assert set(pred) == PRED_KEYS
    for k, v in pred.items():
        assert len(v) == len(X30), k
        assert np.isfinite(np.asarray(v, dtype=float)).all(), k
    assert set(np.unique(pred["regime"])) <= {0, 1, 2}
    assert bundle.bundle_id.startswith("eval-full-")
    assert bundle.columns == list(_core.FEATURE_COLS)
    assert bundle.theta == 187.0 and bundle.tau == sh.TAU and bundle.tau_cls == sh.TAU_CLS


def test_gate_learns_regime_rule(bundle, base):
    """합성 게이트가 레짐 규칙을 학습했다 — 이후 라우팅 테스트의 전제."""
    pred = bundle.predict_features(base["X"])
    assert (pred["regime"] == base["regime"]).mean() > 0.98
    X = sh.make_features(9, regimes=[0, 1, 2] * 3)
    assert bundle.predict_features(X)["regime"].tolist() == [0, 1, 2] * 3


# ══════════════════════════════════════════════════════════════════════
# 2. 무결성 — 1바이트 변조 · 누락 · 목록 밖 참조
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("fname", ["gate.lgb", "reg_y_avg_r1.lgb", "interval_q90.lgb",
                                   "calibrator_isotonic.json", "golden.json"])
def test_one_byte_tamper_is_integrity_error(bundle_copy, fname):
    """번들 파일을 1바이트만 바꿔도 BUNDLE_INTEGRITY 로 거부한다(종료코드 3)."""
    p = bundle_copy / fname
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "BUNDLE_INTEGRITY"
    assert ei.value.exit_code == 3
    assert fname in ei.value.message


def test_missing_model_file_is_integrity_error(bundle_copy):
    """매니페스트에 있는 모델 파일이 없으면 BUNDLE_INTEGRITY."""
    (bundle_copy / "fallback_y_peak.lgb").unlink()
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "BUNDLE_INTEGRITY"


def test_component_outside_sha_list_is_integrity_error(bundle_copy):
    """구성요소가 files_sha256 에 없는 파일을 가리키면(검증 우회) BUNDLE_INTEGRITY."""
    sh.rewrite_manifest(bundle_copy, lambda m: m["files_sha256"].pop("peak_clf.lgb"))
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "BUNDLE_INTEGRITY"


# ══════════════════════════════════════════════════════════════════════
# 3. FAST 번들 거부
# ══════════════════════════════════════════════════════════════════════
def test_fast_bundle_rejected_by_default(tmp_path):
    """fast_mode 번들은 기본 거부(FAST_BUNDLE), allow_fast=True 일 때만 열린다."""
    info = sh.make_synthetic_bundle(tmp_path / "fast", fast=True)
    with pytest.raises(BundleError) as ei:
        load_bundle(info["dir"], selftest=False)
    assert ei.value.code == "FAST_BUNDLE"
    b = load_bundle(info["dir"], selftest=False, allow_fast=True)
    assert b.bundle_id.startswith("eval-fast-")
    assert set(b.predict_features(sh.make_features(5))) == PRED_KEYS


# ══════════════════════════════════════════════════════════════════════
# 4. 라이브러리 버전 고정
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("lib", ["lightgbm", "numpy", "pandas"])
def test_version_mismatch_rejected_unless_overridden(bundle_copy, monkeypatch, lib):
    """런타임 버전이 매니페스트와 다르면 VERSION_MISMATCH, 환경변수 우회 시에만 통과한다."""
    sh.rewrite_manifest(bundle_copy, lambda m: m["runtime"].__setitem__(lib, "0.0.0-other"))
    monkeypatch.delenv("KAMP_ALLOW_VERSION_MISMATCH", raising=False)
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "VERSION_MISMATCH"
    assert lib in ei.value.message
    # 우회 스위치: 정확히 "1" 일 때만
    monkeypatch.setenv("KAMP_ALLOW_VERSION_MISMATCH", "true")
    with pytest.raises(BundleError):
        load_bundle(bundle_copy, selftest=False)
    monkeypatch.setenv("KAMP_ALLOW_VERSION_MISMATCH", "1")
    assert load_bundle(bundle_copy, selftest=False).bundle_id
    monkeypatch.delenv("KAMP_ALLOW_VERSION_MISMATCH")
    assert load_bundle(bundle_copy, selftest=False, check_versions=False).bundle_id


# ══════════════════════════════════════════════════════════════════════
# 5. 파일명 화이트리스트 · 매니페스트 없음 · 경로 비노출
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("bad", ["../x.lgb", "..\\x.lgb", "sub/x.lgb", "/x.lgb", "X.lgb"])
def test_unsafe_filename_rejected(bundle_copy, tmp_path, bad):
    """files_sha256 의 파일명이 번들 밖을 가리키면(sha 가 맞아도) BUNDLE_INTEGRITY."""
    payload = b"outside the bundle"
    (tmp_path / "x.lgb").write_bytes(payload)   # '../x.lgb' 가 실제로 가리키는 파일 — sha 도 맞춰 둔다
    sh.rewrite_manifest(bundle_copy, lambda m: m["files_sha256"].__setitem__(bad, sh.sha256(payload)))
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "BUNDLE_INTEGRITY"
    assert "허용되지 않는 파일명" in ei.value.message
    _assert_no_abs_path(ei.value, tmp_path)


@pytest.mark.parametrize("make_dir", [False, True])
def test_manifest_missing_is_not_found(tmp_path, make_dir):
    """매니페스트가 없으면(디렉터리 유무 무관) BUNDLE_NOT_FOUND, 메시지에 절대경로가 없다."""
    d = tmp_path / "bundle"
    if make_dir:
        d.mkdir()
    with pytest.raises(BundleError) as ei:
        load_bundle(d, selftest=False)
    assert ei.value.code == "BUNDLE_NOT_FOUND"
    assert "bundle/manifest.json" in ei.value.message
    _assert_no_abs_path(ei.value, tmp_path)


def test_integrity_error_message_has_no_abs_path(bundle_copy, tmp_path):
    """변조 오류 메시지에도 절대경로가 없다(파일명만)."""
    (bundle_copy / "gate.lgb").write_bytes(b"tree\n")
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    _assert_no_abs_path(ei.value, tmp_path)


# ══════════════════════════════════════════════════════════════════════
# 6. 피처 계약 — 부스터 피처 이름 == 매니페스트
# ══════════════════════════════════════════════════════════════════════
def _swap01(cols: list) -> None:
    """리스트의 앞 두 원소를 맞바꾼다(순서만 다른 계약)."""
    cols[0], cols[1] = cols[1], cols[0]


@pytest.mark.parametrize("case", ["contract_order", "gate_features", "renamed_booster"])
def test_feature_name_mismatch_is_contract_error(bundle_copy, case):
    """부스터 피처 이름·순서가 매니페스트 계약과 다르면 BUNDLE_CONTRACT."""
    if case == "contract_order":
        sh.rewrite_manifest(bundle_copy, lambda m: _swap01(m["feature_contract"]["columns"]))
    elif case == "gate_features":
        sh.rewrite_manifest(bundle_copy, lambda m: _swap01(m["components"]["gate"]["features"]))
    else:
        X = sh.make_features(200).set_axis([f"f{i}" for i in range(len(_core.FEATURE_COLS))], axis=1)
        sh.replace_file(bundle_copy, "reg_y_peak_r2.lgb", sh.train_booster_bytes(X, X["f1"].to_numpy()))
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "BUNDLE_CONTRACT"


# ══════════════════════════════════════════════════════════════════════
# 7. 입력 강제 — 컬럼 순서 · dtype · 누락
# ══════════════════════════════════════════════════════════════════════
def test_shuffled_columns_give_identical_prediction(bundle, X30):
    """입력 컬럼 순서를 섞어도 예측이 비트 단위로 같다."""
    shuffled = X30[list(np.random.default_rng(3).permutation(X30.columns))]
    assert list(shuffled.columns) != list(X30.columns)
    assert _same(bundle.predict_features(shuffled), bundle.predict_features(X30))


def test_object_dtype_numbers_are_coerced_to_float64(bundle, X30):
    """숫자가 문자열(object)로 들어와도 float64 로 통일해 같은 예측을 낸다.

    `bundle_predict` 의 `astype(np.float64)` 를 지우면 LightGBM 이 object 배열을 거부해 실패한다.
    """
    Xs = X30.map(repr).astype(object)   # pandas 3 는 기본이 str dtype 이라 object 로 명시
    assert (Xs.dtypes == object).all()
    assert _same(bundle.predict_features(Xs), bundle.predict_features(X30))


def test_extra_columns_are_ignored(bundle, X30):
    """계약 밖 컬럼이 더 있어도 예측은 같다."""
    assert _same(bundle.predict_features(X30.assign(junk=1e9)), bundle.predict_features(X30))


def test_missing_feature_raises_keyerror(bundle, X30):
    """계약 피처가 하나라도 빠지면 KeyError — 조용히 NaN 으로 채우지 않는다."""
    with pytest.raises(KeyError, match="피처 누락"):
        bundle.predict_features(X30.drop(columns=["hour"]))


# ══════════════════════════════════════════════════════════════════════
# 8. 레짐 회귀가 없는 레짐 → fallback
# ══════════════════════════════════════════════════════════════════════
def test_dropped_regime_rows_use_fallback(drop1, X30):
    """레짐1 회귀가 없는 번들: 레짐1 배정 행은 fallback, 나머지는 해당 레짐 회귀(비트 동일)."""
    assert set(drop1.parts["regs"]["y_avg"]) == {0, 2}
    assert set(drop1.manifest["components"]["regressors"]["y_avg"]) == {"0", "2"}
    pred = drop1.predict_features(X30)
    regime = pred["regime"]
    assert (regime == 1).sum() >= 1 and set(np.unique(regime)) == {0, 1, 2}
    arr = _arr(drop1, X30)
    for tgt in ("y_avg", "y_peak"):
        m1 = regime == 1
        fb = drop1.parts["fallbacks"][tgt].predict(arr[m1])
        assert np.array_equal(pred[tgt][m1], fb), tgt
        # 판별력: fallback 과 레짐0 회귀는 같은 행에 다른 값을 낸다
        assert not np.array_equal(fb, drop1.parts["regs"][tgt][0].predict(arr[m1]))
        for r in (0, 2):
            m = regime == r
            assert np.array_equal(pred[tgt][m], drop1.parts["regs"][tgt][r].predict(arr[m])), (tgt, r)


# ══════════════════════════════════════════════════════════════════════
# 9. 매니페스트 문자열 키 → int 라우팅
# ══════════════════════════════════════════════════════════════════════
def test_string_regime_keys_are_routed_as_int(bundle, base, X30):
    """JSON 의 "0"/"1"/"2" 키가 int 로 바뀌어 각 레짐 행이 자기 회귀기로 간다."""
    man = sh.read_manifest(base["dir"])
    assert set(man["components"]["regressors"]["y_avg"]) == {"0", "1", "2"}
    assert set(man["components"]["quantiles"]) == {"0.1", "0.5", "0.9"}
    for tgt in ("y_avg", "y_peak"):
        assert set(bundle.parts["regs"][tgt]) == {0, 1, 2}
        assert all(type(k) is int for k in bundle.parts["regs"][tgt])
    assert set(bundle.parts["quantiles"]) == {0.1, 0.5, 0.9}
    assert bundle.parts["classes"] == [0, 1, 2]

    pred = bundle.predict_features(X30)
    arr = _arr(bundle, X30)
    for tgt in ("y_avg", "y_peak"):
        for r in (0, 1, 2):
            m = pred["regime"] == r
            assert m.any()
            assert np.array_equal(pred[tgt][m], bundle.parts["regs"][tgt][r].predict(arr[m])), (tgt, r)
        # 모든 행이 레짐 회귀로 배정됐으므로 fallback 값과는 달라야 한다
        assert not np.array_equal(pred[tgt], bundle.parts["fallbacks"][tgt].predict(arr))
    for q, key in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90")):
        assert np.array_equal(pred[key], bundle.parts["quantiles"][q].predict(arr))


# ══════════════════════════════════════════════════════════════════════
# 10. 입력 계약 (실데이터)
# ══════════════════════════════════════════════════════════════════════
def test_contract_normal_day(raw):
    """정상: 08-24~09-13 이력(21일) + 09-14 계획 → 24×44 유한 피처."""
    X, info = _features(_days(raw, 20210824, 20210913), _plan(raw, 20210914))
    assert X.shape == (24, 44)
    assert list(X.columns) == list(_core.FEATURE_COLS)
    assert np.isfinite(X.to_numpy(dtype=float)).all()
    assert (X.index.normalize() == pd.Timestamp("2021-09-14")).all()
    assert list(X.index.hour) == list(range(24))
    assert info == {"target_date": "2021-09-14", "window_start": "2021-08-24",
                    "history_days": 21, "warnings": []}


def test_contract_plan_with_power_column_rejected(raw):
    """계획에 '15분' 같은 전력 컬럼이 있으면 누수로 거부한다(PLAN_HAS_POWER)."""
    plan = _plan(raw, 20210914)
    plan["15분"] = raw.loc[raw["날짜"] == 20210914, "15분"].to_numpy()
    _input_error("PLAN_HAS_POWER", _days(raw, 20210824, 20210913), plan)


def test_contract_plan_23_rows_invalid(raw):
    """계획이 24행이 아니면 PLAN_INVALID."""
    _input_error("PLAN_INVALID", _days(raw, 20210824, 20210913), _plan(raw, 20210914).iloc[:23])


def test_contract_target_date_mismatch_invalid(raw):
    """target_date 가 계획 날짜와 다르면 PLAN_INVALID."""
    _input_error("PLAN_INVALID", _days(raw, 20210824, 20210913), _plan(raw, 20210914),
                 target_date="2021-09-15")


def test_contract_history_gap_invalid(raw):
    """이력 중간 1행이 빠지면 HISTORY_INVALID."""
    h = _days(raw, 20210824, 20210913).drop(index=100)
    _input_error("HISTORY_INVALID", h, _plan(raw, 20210914))


@pytest.mark.parametrize("case", ["drop_last_row", "ends_two_days_before"])
def test_contract_history_must_end_day_before_2300(raw, case):
    """이력이 대상일 전날 23시에 끝나지 않으면 HISTORY_INVALID."""
    if case == "drop_last_row":
        h = _days(raw, 20210824, 20210913).iloc[:-1]
    else:
        h = _days(raw, 20210824, 20210912)
    _input_error("HISTORY_INVALID", h, _plan(raw, 20210914))


def test_contract_history_too_short(raw):
    """7일 이력은 HISTORY_TOO_SHORT, 상세에 필요한 시작일이 있다."""
    err = _input_error("HISTORY_TOO_SHORT", _days(raw, 20210907, 20210913), _plan(raw, 20210914))
    assert err.detail["required_start"] == "2021-09-06"


@pytest.mark.parametrize("bad", ["string", "inf", "missing_column"])
def test_contract_history_schema(raw, bad):
    """이력 기온에 문자열·inf 가 있거나 필수 컬럼이 없으면 HISTORY_SCHEMA."""
    h = _days(raw, 20210824, 20210913)
    if bad == "string":
        h["기온"] = h["기온"].astype(object)
        h.loc[50, "기온"] = "덥다"
    elif bad == "inf":
        h.loc[50, "기온"] = np.inf
    else:
        h = h.drop(columns=["습도"])
    _input_error("HISTORY_SCHEMA", h, _plan(raw, 20210914))


def test_contract_shutdown_run_truncated(raw):
    """하계휴가 9일차(08-08)를 휴무뿐인 8일 이력으로 예측하면 SHUTDOWN_RUN_TRUNCATED."""
    err = _input_error("SHUTDOWN_RUN_TRUNCATED", _days(raw, 20210731, 20210807), _plan(raw, 20210808))
    assert "required_start" in err.detail


def test_contract_shutdown_run_full_history_ok(raw):
    """휴무 시작 전부터 이력이 있으면(07-20~) 정상이고 '휴무 n일차' 가 9로 정확히 세진다."""
    X, info = _features(_days(raw, 20210720, 20210807), _plan(raw, 20210808))
    assert X.shape == (24, 44) and np.isfinite(X.to_numpy(dtype=float)).all()
    assert (X["is_shutdown"] == 1).all()
    assert (X["shutdown_nth"] == 9).all()        # 07-31 이 1일차
    assert info["warnings"] == []


@pytest.mark.parametrize("hours, ok", [(5, True), (24, True), (25, False), (30, False)])
def test_contract_leading_outage(raw, hours, ok):
    """8일 창 시작의 계측정지: 피처 참조 구간(D-7 00시~) 밖이면 경고, 닿으면 HISTORY_EDGE_OUTAGE."""
    h = _days(raw, 20210906, 20210913)
    h.loc[: hours - 1, POWER_COLUMNS] = 0
    plan = _plan(raw, 20210914)
    if not ok:
        err = _input_error("HISTORY_EDGE_OUTAGE", h, plan)
        assert "required_start" in err.detail
        return
    X, info = _features(h, plan)
    assert any(w.startswith("LEADING_OUTAGE") for w in info["warnings"])
    # '영향 없음' 의 실증: 정지가 없던 이력과 피처가 같다(rolling std 만 누적 오차 허용)
    X0, _ = _features(_days(raw, 20210906, 20210913), plan)
    std = [c for c in X.columns if c.endswith("_std")]
    exact = [c for c in X.columns if c not in std]
    assert X[exact].equals(X0[exact])
    assert np.allclose(X[std], X0[std], rtol=1e-9, atol=0)


def test_contract_plan_missing_wind_forward_filled(raw):
    """계획 풍속 결측은 앞선 값으로 대체하고 경고를 남긴다."""
    plan = _plan(raw, 20210914)
    plan.loc[[3, 4, 10], "풍속"] = np.nan
    X, info = _features(_days(raw, 20210824, 20210913), plan)
    assert np.isfinite(X.to_numpy(dtype=float)).all()
    assert any(w.startswith("PLAN_FORWARD_FILLED") and "풍속" in w for w in info["warnings"])
    w = X["wind"].to_numpy()
    assert min(w[2], w[5]) <= w[3] <= max(w[2], w[5])   # 이웃 사이 값으로 채워졌다


# ══════════════════════════════════════════════════════════════════════
# 11. 공휴일 커버리지
# ══════════════════════════════════════════════════════════════════════
def test_holidays_for_coverage(base, tmp_path, monkeypatch):
    """학습 휴일표·운영 휴일표로 덮인 날만 통과, 밖이면 CALENDAR_NOT_COVERED."""
    monkeypatch.delenv("KAMP_HOLIDAYS_FILE", raising=False)
    b = load_bundle(base["dir"], selftest=False)
    hol = b.holidays_for("2021-09-14")
    assert set(_core.HOLIDAYS_2021) <= set(hol)
    dec = b.holidays_for("2021-12-01")                     # 기본 운영 휴일표가 2021 을 덮는다
    assert pd.Timestamp("2021-12-25") in dec and pd.Timestamp("2021-09-21") in dec
    with pytest.raises(ServingInputError) as ei:
        b.holidays_for("2026-03-01")
    assert ei.value.code == "CALENDAR_NOT_COVERED" and ei.value.exit_code == 2

    f = tmp_path / "holidays_2026.json"
    f.write_bytes(json.dumps({"coverage_years": [2026], "dates": ["2026-03-01"]}).encode("utf-8"))
    b26 = load_bundle(base["dir"], selftest=False, holidays_file=f)
    got = b26.holidays_for("2026-03-01")
    assert pd.Timestamp("2026-03-01") in got
    assert set(_core.HOLIDAYS_2021) <= set(got)          # 학습 휴일표는 항상 포함
    assert got.is_monotonic_increasing and got.is_unique


def test_holidays_for_rejects_dates_before_training_calendar(base, monkeypatch):
    """HOLIDAYS_2021 은 2021년 휴일만 있다 — 2020-03-01(삼일절)은 커버 밖이어야 한다."""
    monkeypatch.delenv("KAMP_HOLIDAYS_FILE", raising=False)
    b = load_bundle(base["dir"], selftest=False)
    with pytest.raises(ServingInputError):
        b.holidays_for("2020-03-01")


# ══════════════════════════════════════════════════════════════════════
# 12. 출력 조립
# ══════════════════════════════════════════════════════════════════════
def _assemble_case():
    """휴무 12행 + 운영 12행, τ 경계값·0 근처 y_avg 를 포함한 예측 묶음."""
    idx = pd.date_range("2021-09-14", periods=24, freq="h")
    X = pd.DataFrame({"is_shutdown": [1] * 12 + [0] * 12}, index=idx)
    y_avg = np.linspace(0.0, 200.0, 24)
    y_peak = y_avg + 30.0
    y_peak[5] = sh.TAU                                     # 경계: τ 와 같으면 경보
    prob = np.linspace(-0.05, 1.05, 24)                     # 보정표 범위 밖(클립) 포함
    pred = {"regime": np.array([0] * 12 + [2] * 12), "y_avg": y_avg, "y_peak": y_peak, "prob": prob,
            "q10": y_avg - 5, "q50": y_avg, "q90": y_avg + 5}
    halfwidths = {"operating": {"qhat": sh.QHAT["operating"]}, "shutdown": {"qhat": sh.QHAT["shutdown"]}}
    return X, pred, halfwidths


def test_assemble_output_rules():
    """경보 = y_peak≥τ, 보정확률 = np.interp, 하한 ≥ 0, 휴무행은 휴무 반폭."""
    X, pred, hw = _assemble_case()
    out = assemble_output(X, pred, tau=sh.TAU, tau_cls=sh.TAU_CLS, calibrator=sh.CALIBRATOR, halfwidths=hw)
    assert list(out.columns) == OUTPUT_COLUMNS and len(out) == 24
    assert out["ts"].iloc[0] == "2021-09-14 00:00:00" and out["ts"].iloc[-1] == "2021-09-14 23:00:00"
    assert np.array_equal(out["peak_label"], (pred["y_peak"] >= sh.TAU).astype(int))
    assert out["peak_label"].iloc[5] == 1
    assert np.array_equal(out["peak_prob"], pred["prob"])
    assert np.array_equal(out["peak_prob_cal"],
                          np.interp(pred["prob"], sh.CALIBRATOR["x_thresholds"], sh.CALIBRATOR["y_thresholds"]))
    assert out["peak_prob_cal"].between(0, 1).all()
    assert np.array_equal(out["peak_label_cls"], (pred["prob"] >= sh.TAU_CLS).astype(int))
    assert (out["pi_lo"] >= 0).all() and out["pi_lo"].iloc[0] == 0.0
    shut = X["is_shutdown"].to_numpy() == 1
    q = np.where(shut, sh.QHAT["shutdown"], sh.QHAT["operating"])
    assert np.array_equal(out["pi_hi"], pred["y_avg"] + q)
    assert np.array_equal(out["pi_lo"], np.maximum(pred["y_avg"] - q, 0.0))
    assert np.array_equal(out["regime"], pred["regime"])


def test_calibrator_interp_equals_isotonic_clip():
    """np.interp 보정이 IsotonicRegression(out_of_bounds='clip').predict 와 정확히 같다."""
    isotonic = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < x ** 2).astype(float)
    iso = isotonic.IsotonicRegression(out_of_bounds="clip").fit(x, y)
    p = np.concatenate([rng.uniform(-0.2, 1.2, 2000), iso.X_thresholds_])
    assert np.array_equal(iso.predict(p), np.interp(p, iso.X_thresholds_, iso.y_thresholds_))


def test_assemble_output_rejects_nonfinite():
    """예측에 NaN 이 섞이면 출력하지 않는다(PREDICTION_NONFINITE)."""
    X, pred, hw = _assemble_case()
    pred["q50"] = pred["q50"].copy()
    pred["q50"][3] = np.nan
    with pytest.raises(PredictionError, match="PREDICTION_NONFINITE"):
        assemble_output(X, pred, tau=sh.TAU, tau_cls=sh.TAU_CLS, calibrator=sh.CALIBRATOR, halfwidths=hw)


# ══════════════════════════════════════════════════════════════════════
# 추가 — 종단 예측 · 자가검증 · CLI 종료코드
# ══════════════════════════════════════════════════════════════════════
def test_predict_day_end_to_end(bundle, raw):
    """predict_day = 피처 생성 + predict_features + 출력 조립 (합성 번들, 실데이터 입력)."""
    hist, plan = _days(raw, 20210824, 20210913), _plan(raw, 20210914)
    out, meta = bundle.predict_day(hist, plan)
    assert list(out.columns) == OUTPUT_COLUMNS and len(out) == 24
    assert meta["target_date"] == "2021-09-14" and meta["bundle_id"] == bundle.bundle_id
    X, _ = _features(hist, plan)
    pred = bundle.predict_features(X)
    assert np.array_equal(out["y_avg_pred"], pred["y_avg"])
    assert np.array_equal(out["y_peak_pred"], pred["y_peak"])
    with pytest.raises(ServingInputError) as ei:
        bundle.predict_day(hist, plan, target_date="2021-09-15")
    assert ei.value.code == "PLAN_INVALID"


def test_selftest_passes_with_consistent_golden(golden_base):
    """golden 이 번들과 일치하면 selftest=True 로 열린다."""
    b = load_bundle(golden_base["dir"])
    assert b.bundle_id == golden_base["manifest"]["bundle_id"]


@pytest.mark.parametrize("where, code", [("expected", "SELFTEST_PREDICTION"),
                                         ("expected_features", "SELFTEST_FEATURES")])
def test_selftest_detects_golden_drift(golden_base, tmp_path, where, code):
    """golden 의 기대 출력·기대 피처가 어긋나면(sha 는 맞춰도) 자가검증이 거부한다."""
    d = tmp_path / "bundle"
    shutil.copytree(golden_base["dir"], d)
    g = json.loads((d / "golden.json").read_bytes())
    if where == "expected":
        g["expected"]["y_avg"][0] += 1e-9
    else:
        g["expected_features"]["rows"][0][g["expected_features"]["columns"].index("temp")] += 0.5
    sh.replace_file(d, "golden.json", sh.json_bytes(g))
    with pytest.raises(BundleError) as ei:
        load_bundle(d)
    assert ei.value.code == code


def test_malformed_golden_is_bundle_error(bundle_copy):
    """sha 는 맞지만 자가검증 사례 형식이 틀린 golden 은 BundleError 로 거부돼야 한다."""
    with pytest.raises(BundleError):
        load_bundle(bundle_copy)   # 기본 합성 golden 은 최소 JSON(history 등 없음)


def test_non_json_calibrator_is_bundle_error(bundle_copy):
    """sha 는 맞지만 JSON 이 아닌 보정기 파일은 BUNDLE_SCHEMA 로 거부돼야 한다."""
    sh.replace_file(bundle_copy, "calibrator_isotonic.json", b"not json")
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "BUNDLE_SCHEMA"


def test_cli_exit_codes(golden_base, raw, tmp_path, capsys):
    """CLI 종료코드: 정상 0 · 입력 오류 2 · 번들 오류 3, 오류 출력에 절대경로가 없다."""
    bdir = str(golden_base["dir"])
    hist_csv, plan_csv, bad_csv, out_csv = (tmp_path / n for n in ("h.csv", "p.csv", "bad.csv", "o.csv"))
    _days(raw, 20210824, 20210913).to_csv(hist_csv, index=False, encoding="utf-8-sig")
    _plan(raw, 20210914).to_csv(plan_csv, index=False, encoding="utf-8-sig")
    raw[raw["날짜"] == 20210914].to_csv(bad_csv, index=False, encoding="utf-8-sig")   # 전력 컬럼 포함

    assert cli.main(["verify", "--bundle", bdir]) == 0
    assert cli.main(["predict", "--bundle", bdir, "--history", str(hist_csv), "--plan", str(plan_csv),
                     "--out", str(out_csv)]) == 0
    got = pd.read_csv(out_csv, encoding="utf-8-sig")
    assert list(got.columns) == OUTPUT_COLUMNS and len(got) == 24
    capsys.readouterr()
    assert cli.main(["predict", "--bundle", bdir, "--history", str(hist_csv), "--plan", str(bad_csv)]) == 2
    assert "PLAN_HAS_POWER" in capsys.readouterr().err
    assert cli.main(["verify", "--bundle", str(tmp_path / "nope")]) == 3
    err = capsys.readouterr().err
    assert "BUNDLE_NOT_FOUND" in err and str(tmp_path) not in err



# ══════════════════════════════════════════════════════════════════════
# 리뷰 반영 — 매니페스트 무결성 · 형식 오류 · 날짜 · 휴일표 · 계획 결측
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("edit", [
    lambda m: m["thresholds"]["tau"].__setitem__("value", 1.0),
    lambda m: m["intervals"]["residual"]["operating"].__setitem__("qhat", 0.0),
    lambda m: m["feature_contract"].__setitem__("theta", 150.0),
])
def test_manifest_tamper_is_integrity_error(bundle_copy, edit):
    """매니페스트 안의 값(τ·구간 반폭·θ)만 바꿔도 BUNDLE_INTEGRITY — 파일 sha 로는 못 잡는 변조."""
    sh.rewrite_manifest(bundle_copy, edit, reseal=False)
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "BUNDLE_INTEGRITY"


def test_bundle_id_is_manifest_fingerprint(base):
    """bundle_id 의 끝 12자리 = 매니페스트 해시(모델 파일·임계값·계약 전부의 지문)."""
    man = sh.read_manifest(base["dir"])
    assert man["manifest_sha256"] == _core.manifest_core_sha256(man)
    assert man["bundle_id"].endswith(man["manifest_sha256"][:12])


@pytest.mark.parametrize("payload", [b"[1, 2]", b'{"schema_version": 1, "files_sha256": [1]}'])
def test_non_object_manifest_is_schema_error(bundle_copy, payload):
    """매니페스트가 객체가 아니거나 files_sha256 이 목록이면 BUNDLE_SCHEMA (AttributeError 아님)."""
    (bundle_copy / "manifest.json").write_bytes(payload)
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "BUNDLE_SCHEMA"


def test_gate_class_count_mismatch_is_contract_error(bundle_copy):
    """게이트 classes 개수가 게이트 모델 출력 수와 다르면 BUNDLE_CONTRACT (IndexError 아님)."""
    sh.rewrite_manifest(bundle_copy, lambda m: m["components"]["gate"].__setitem__("classes", [0, 1]))
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "BUNDLE_CONTRACT"


def test_filename_with_trailing_newline_rejected(bundle_copy):
    """'gate.lgb\\n' 같은 끝 개행 파일명도 거부한다(fullmatch)."""
    def edit(m):
        m["files_sha256"]["gate.lgb\n"] = m["files_sha256"].pop("gate.lgb")
    sh.rewrite_manifest(bundle_copy, edit)
    with pytest.raises(BundleError) as ei:
        load_bundle(bundle_copy, selftest=False)
    assert ei.value.code == "BUNDLE_INTEGRITY"


@pytest.mark.parametrize("where", ["history", "plan"])
def test_nonexistent_date_is_schema_error(raw, where):
    """20210931 처럼 존재하지 않는 날짜는 *_SCHEMA 입력 오류다(ValueError·500 아님)."""
    hist, plan = _days(raw, 20210824, 20210913), _plan(raw, 20210914)
    if where == "plan":
        plan["날짜"] = 20210931
    else:
        hist.loc[:23, "날짜"] = 20210899
    _input_error(f"{where.upper()}_SCHEMA", hist, plan)


@pytest.mark.parametrize("content, code", [
    (b"{not json", "HOLIDAYS_INVALID"),
    (b'{"coverage_years": [2026], "dates": ["2021-13-01"]}', "HOLIDAYS_INVALID"),
    (b'{"coverage_years": [2026], "dates": "2026-01-01"}', "HOLIDAYS_INVALID"),
])
def test_invalid_holiday_file_rejected_at_load(base, tmp_path, content, code):
    """휴일표 형식 오류는 요청 때 500 이 아니라 **로드 시점**에 BundleError 로 거부한다."""
    f = tmp_path / "h.json"
    f.write_bytes(content)
    with pytest.raises(BundleError) as ei:
        load_bundle(base["dir"], selftest=False, holidays_file=f)
    assert ei.value.code == code


def test_explicit_missing_holiday_file_rejected(base, tmp_path):
    """명시한 휴일표 경로가 없으면 조용히 빈 표로 두지 않고 거부한다."""
    with pytest.raises(BundleError) as ei:
        load_bundle(base["dir"], selftest=False, holidays_file=tmp_path / "nope.json")
    assert ei.value.code == "HOLIDAYS_INVALID"


@pytest.mark.parametrize("col", ["풍속", "공장인원"])
def test_plan_column_all_missing_rejected(raw, col):
    """결측 허용 컬럼이라도 24시간 전부 비면 PLAN_SCHEMA — 전날 값을 하루 종일 복사하지 않는다."""
    plan = _plan(raw, 20210914)
    plan[col] = np.nan
    _input_error("PLAN_SCHEMA", _days(raw, 20210824, 20210913), plan)
