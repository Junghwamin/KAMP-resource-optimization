"""서빙 테스트 도우미 — 파이프라인 없이 만드는 합성(synthetic) 모델 번들.

노트북 10.5절(`src/s10_package.py :: build_manifest`)과 **같은 매니페스트 스키마**로
아주 작은 LightGBM 번들을 tmp 디렉터리에 쓴다. 파이프라인 fixture(s00~s10)를 쓰지 않으므로
수 초 안에 끝나고, 학습 중인 메인 트리의 outputs/ 를 건드리지 않는다.

- 피처: `serving._core.FEATURE_COLS` 44개, 값은 [-1, 1] 난수
- 레짐: 첫 피처(`FEATURE_COLS[0]`)의 삼분위 규칙 → 게이트(multiclass 3)가 정확히 학습한다
- 부스터: 트리 5개 내외 · `deterministic` · 단일 스레드 → 같은 인자면 같은 바이트
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from serving import _core
from serving.contract import HISTORY_COLUMNS, PLAN_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "okm_augumented_2021.csv"

FEATURE_COLS = list(_core.FEATURE_COLS)
REGIME_COL = FEATURE_COLS[0]          # 레짐을 결정하는 피처
REGIME_CUTS = (-1.0 / 3.0, 1.0 / 3.0)  # 삼분위 경계 → 레짐 0/1/2
REGIME_CENTERS = {0: -0.7, 1: 0.0, 2: 0.7}  # 각 레짐 한가운데 값(경계에서 충분히 떨어짐)
REGIME_OFFSET = {0: 20.0, 1: 60.0, 2: 130.0}  # 레짐별 전력 수준(회귀기가 확연히 달라지게)
THETA = 187.0
TAU = 150.0
TAU_CLS = 0.4
QHAT = {"operating": 12.5, "shutdown": 3.25}
CALIBRATOR = {
    "x_thresholds": [0.0, 0.2, 0.5, 1.0],
    "y_thresholds": [0.0, 0.05, 0.6, 1.0],
    "out_of_bounds": "clip",
    "fit_on": "합성 번들(테스트)",
    "apply": "np.interp(prob, x_thresholds, y_thresholds)",
}
GOLDEN_ORIGIN = pd.Timestamp("2021-09-13")
GOLDEN_HISTORY_DAYS = 21

# 노트북과 같은 결정적 설정 + 아주 작은 트리
_LGB_BASE = {
    "num_leaves": 4, "min_data_in_leaf": 5, "learning_rate": 0.3,
    "deterministic": True, "force_row_wise": True, "num_threads": 1,
    "seed": 0, "verbose": -1,
}


def sha256(b: bytes) -> str:
    """바이트열의 sha256 16진 문자열."""
    return hashlib.sha256(b).hexdigest()


def json_bytes(obj) -> bytes:
    """노트북 `_json_bytes` 와 같은 결정적 JSON 바이트(키 정렬 · UTF-8 · NaN 금지)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")


def regime_of(x0) -> np.ndarray:
    """레짐 규칙 — 첫 피처의 삼분위(0/1/2)."""
    return np.digitize(np.asarray(x0, dtype=float), REGIME_CUTS)


def make_features(n: int, seed: int = 1, regimes=None, index=None) -> pd.DataFrame:
    """난수 피처 행렬(44컬럼, float64). `regimes` 를 주면 첫 피처를 그 레짐 한가운데로 둔다."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(-1.0, 1.0, size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS, index=index)
    if regimes is not None:
        X[REGIME_COL] = [REGIME_CENTERS[int(r)] for r in regimes]
    return X


def _train(X: pd.DataFrame, y, params: dict, rounds: int) -> lgb.Booster:
    """작은 부스터 하나를 결정적으로 학습한다(피처 이름 = X 컬럼)."""
    ds = lgb.Dataset(X, label=np.asarray(y, dtype=float), feature_name=list(X.columns),
                     params={"verbose": -1}, free_raw_data=False)
    return lgb.train({**_LGB_BASE, **params}, ds, num_boost_round=rounds)


def _model_bytes(bst: lgb.Booster) -> bytes:
    """부스터 → LightGBM 네이티브 텍스트 바이트(노트북 `_booster_bytes` 와 같은 형식)."""
    return bst.model_to_string().encode("utf-8")


def train_booster_bytes(X: pd.DataFrame, y, *, objective: str = "regression", rounds: int = 5) -> bytes:
    """임의 피처 이름으로 부스터를 학습해 바이트로 돌려준다(계약 위반 번들 제작용)."""
    return _model_bytes(_train(X, y, {"objective": objective}, rounds))


def load_raw() -> pd.DataFrame:
    """실데이터 CSV(utf-8-sig)."""
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def build_golden(parts: dict, raw: pd.DataFrame) -> dict:
    """노트북 `build_golden` 과 같은 구조의 자가검증 사례(원시 21일 + 계획 + 기대 피처 + 기대 출력).

    기대 피처는 서빙 파이프라인으로 만든다 — 합성 번들의 selftest 를 켤 수 있게 하는 용도다.
    """
    from serving.pipeline import build_day_ahead_features

    target = GOLDEN_ORIGIN + pd.Timedelta(days=1)
    ymd = raw["날짜"]
    lo = int((GOLDEN_ORIGIN - pd.Timedelta(days=GOLDEN_HISTORY_DAYS - 1)).strftime("%Y%m%d"))
    hist = raw.loc[(ymd >= lo) & (ymd <= int(GOLDEN_ORIGIN.strftime("%Y%m%d"))), HISTORY_COLUMNS]
    plan = raw.loc[ymd == int(target.strftime("%Y%m%d")), PLAN_COLUMNS]
    X, _ = build_day_ahead_features(hist, plan, columns=FEATURE_COLS, theta=THETA,
                                    holidays=_core.HOLIDAYS_2021, min_days=8)
    pred = _core.bundle_predict(parts, X)

    def rows(d):
        return [[None if (isinstance(v, float) and not np.isfinite(v)) else
                 (v.item() if hasattr(v, "item") else v) for v in r]
                for r in d.itertuples(index=False, name=None)]

    return {
        "origin": str(GOLDEN_ORIGIN.date()),
        "target_date": str(target.date()),
        "history": {"columns": HISTORY_COLUMNS, "rows": rows(hist)},
        "plan": {"columns": PLAN_COLUMNS, "rows": rows(plan)},
        "expected_features": {"columns": FEATURE_COLS, "rows": X.to_numpy(float).tolist()},
        "expected": {k: np.asarray(v).tolist() for k, v in pred.items()},
    }


def make_synthetic_bundle(bundle_dir, *, drop_regime: int | None = None, fast: bool = False,
                          role: str = "eval", n_rows: int = 600, seed: int = 0, rounds: int = 5,
                          runtime: dict | None = None, golden: dict | str | None = None) -> dict:
    """합성 번들을 `bundle_dir` 에 쓰고 정보(dict)를 돌려준다.

    Parameters
    ----------
    drop_regime : int, optional
        이 레짐의 회귀기를 번들에서 뺀다(파일·매니페스트 키 모두) → 그 레짐 배정 행은 fallback.
    fast : bool
        매니페스트 `fast_mode` (True 면 서빙 로더가 기본 거부한다).
    runtime : dict, optional
        매니페스트 `runtime` 덮어쓰기(버전 불일치 재현용).
    golden : dict | "real" | None
        None 이면 최소 JSON(selftest=False 로 로드할 때), "real" 이면 실데이터로 만든 자가검증 사례.

    Returns
    -------
    dict
        `dir`(Path), `manifest`, `files`({파일명: 바이트}), `X`·`regime`(학습 데이터).
    """
    d = Path(bundle_dir)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(-1.0, 1.0, size=(n_rows, len(FEATURE_COLS))), columns=FEATURE_COLS)
    regime = regime_of(X[REGIME_COL])
    x1, x2 = X[FEATURE_COLS[1]].to_numpy(), X[FEATURE_COLS[2]].to_numpy()
    y_avg = np.array([REGIME_OFFSET[int(r)] for r in regime]) + 10.0 * x1 + 5.0 * x2
    y_peak = y_avg + 25.0 + 8.0 * np.abs(x2)
    y_cls = (y_peak >= TAU).astype(int)
    assert 0 < y_cls.sum() < n_rows, "피크 분류기 학습용 양성·음성이 모두 있어야 한다"

    boosters: dict[str, lgb.Booster] = {
        "gate.lgb": _train(X, regime, {"objective": "multiclass", "num_class": 3}, rounds),
    }
    regimes = [r for r in (0, 1, 2) if r != drop_regime]
    for tgt, y in (("y_avg", y_avg), ("y_peak", y_peak)):
        for r in regimes:
            m = regime == r
            boosters[f"reg_{tgt}_r{r}.lgb"] = _train(X[m], y[m], {"objective": "regression"}, rounds)
        boosters[f"fallback_{tgt}.lgb"] = _train(X, y, {"objective": "regression"}, rounds)
    boosters["peak_clf.lgb"] = _train(X, y_cls, {"objective": "binary"}, rounds)
    for q in (0.1, 0.5, 0.9):
        boosters[f"interval_q{int(round(q * 100))}.lgb"] = _train(
            X, y_avg, {"objective": "quantile", "alpha": q}, rounds)

    files = {name: _model_bytes(b) for name, b in boosters.items()}
    files["calibrator_isotonic.json"] = json_bytes(CALIBRATOR)
    if golden == "real":
        parts = {
            "columns": FEATURE_COLS, "gate": boosters["gate.lgb"], "gate_features": FEATURE_COLS,
            "classes": [0, 1, 2],
            "regs": {t: {r: boosters[f"reg_{t}_r{r}.lgb"] for r in regimes} for t in ("y_avg", "y_peak")},
            "fallbacks": {t: boosters[f"fallback_{t}.lgb"] for t in ("y_avg", "y_peak")},
            "peak_clf": boosters["peak_clf.lgb"],
            "quantiles": {q: boosters[f"interval_q{int(round(q * 100))}.lgb"] for q in (0.1, 0.5, 0.9)},
        }
        golden = build_golden(parts, load_raw())
    files["golden.json"] = json_bytes(golden if golden is not None else {"note": "합성 번들 — selftest=False 전용"})

    files_sha = {n: sha256(b) for n, b in sorted(files.items())}
    months = [1, 2, 3, 4, 5, 6, 7, 8]
    first, last = "2021-01-08 00:00:00", "2021-08-31 23:00:00"
    manifest = {
        "schema_version": 1,
        "role": role,
        "fast_mode": bool(fast),
        "model_name": "2단계 레짐(3분류)",
        "report_final_model": "2단계 레짐(3분류)",
        "is_final_model": True,
        "training": {
            "cond": "D1",
            "excluded_masks": ["is_warmup", "is_outage", "is_erp_missing"],
            "cutoff_exclusive": "2021-09-01 00:00:00",
            "n_rows": int(n_rows),
            "first": first,
            "last": last,
            "months": months,
            "data_file": "data/okm_augumented_2021.csv",
            "data_sha256": "0" * 64,
            "used_in_report": role == "eval",
        },
        "submission": None,
        "feature_contract": {
            "columns": FEATURE_COLS,
            "dtype": "float64",
            "theta": THETA,
            "theta_note": "합성 번들",
            "safe_lags": [int(x) for x in _core.SAFE_LAGS],
            "holidays": [str(t.date()) for t in _core.HOLIDAYS_2021],
            "holiday_coverage_start": "2021-01-01",
            "holiday_coverage_end": "2021-09-14",
            "history_days": {"min": 8, "recommended": 14, "max": 62},
            "plan_columns": PLAN_COLUMNS,
            "history_columns": HISTORY_COLUMNS,
        },
        "components": {
            "gate": {
                "file": "gate.lgb", "classes": [0, 1, 2], "features": FEATURE_COLS,
                "n_estimators": int(rounds), "regime_bins": [30.0, 70.0],
            },
            "regressors": {t: {str(r): f"reg_{t}_r{r}.lgb" for r in regimes} for t in ("y_avg", "y_peak")},
            "fallbacks": {t: f"fallback_{t}.lgb" for t in ("y_avg", "y_peak")},
            "peak_classifier": {
                "file": "peak_clf.lgb", "params": {}, "n_estimators": int(rounds),
                "scale_pos_weight": 1.0, "effective_note": "합성 번들",
            },
            "quantiles": {str(q): f"interval_q{int(round(q * 100))}.lgb" for q in (0.1, 0.5, 0.9)},
            "calibrator": "calibrator_isotonic.json",
            "golden": "golden.json",
        },
        "thresholds": {
            "tau": {"value": TAU, "unit": "kW", "fold_taus": [TAU] * 4,
                    "method": "합성", "rule": "y_peak_pred >= tau"},
            "tau_cls": {"value": TAU_CLS, "method": "합성", "rule": "peak_prob >= tau_cls"},
            "inherited_from": None,
        },
        "intervals": {
            "quantile": {"calibrated": False, "fit_last": last,
                         "measured_test_coverage_q10_q90": None, "note": "합성 번들"},
            "residual": {
                "operating": {"qhat": QHAT["operating"], "n": 100},
                "shutdown": {"qhat": QHAT["shutdown"], "n": 40},
                "measured_test_coverage": 0.9, "alpha": 0.9, "center": "y_avg_pred",
                "fit_on": "합성 번들",
            },
        },
        "hyperparameters": {"lgb_base": dict(_LGB_BASE), "n_estimators": int(rounds), "regime_uses_optuna": False},
        "runtime": {
            "python": sys.version.split()[0],
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
            **(runtime or {}),
        },
        "caveats": [f"합성 번들 — month {months}"],
        "files_sha256": files_sha,
    }
    seal(manifest)
    for name, b in files.items():
        (d / name).write_bytes(b)
    (d / "manifest.json").write_bytes(json_bytes(manifest))
    return {"dir": d, "manifest": manifest, "files": files, "X": X, "regime": regime}


def seal(man: dict) -> dict:
    """노트북(10.5절)과 같은 규칙으로 매니페스트 해시·bundle_id 를 다시 매긴다."""
    man["manifest_sha256"] = _core.manifest_core_sha256(man)
    man["bundle_id"] = f"{man['role']}-{'fast' if man['fast_mode'] else 'full'}-{man['manifest_sha256'][:12]}"
    return man


def read_manifest(bundle_dir) -> dict:
    """번들의 매니페스트(UTF-8)."""
    return json.loads((Path(bundle_dir) / "manifest.json").read_bytes())


def rewrite_manifest(bundle_dir, fn, *, reseal: bool = True) -> dict:
    """매니페스트를 읽어 `fn(man)` 으로 고친 뒤 다시 쓴다.

    `reseal=True`(기본)면 매니페스트 해시를 다시 매겨 무결성 검사를 통과시킨다 — 그 뒤의 검사
    (버전·계약·파일 sha 등)를 겨냥한 테스트용. `reseal=False` 는 매니페스트 변조 자체를 재현한다.
    """
    man = read_manifest(bundle_dir)
    fn(man)
    if reseal:
        seal(man)
    (Path(bundle_dir) / "manifest.json").write_bytes(json_bytes(man))
    return man


def replace_file(bundle_dir, fname: str, data: bytes) -> None:
    """번들 파일을 바꾸고 매니페스트의 sha256 도 맞춰 준다(무결성은 통과, 내용만 다른 번들)."""
    (Path(bundle_dir) / fname).write_bytes(data)
    rewrite_manifest(bundle_dir, lambda m: m["files_sha256"].__setitem__(fname, sha256(data)))
