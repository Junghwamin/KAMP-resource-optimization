# %% tags=["nb-strip"]
from s00_env import (  # noqa: F401
    CMAP_SEQ,
    COLOR_HERO,
    COLOR_MUTED,
    FAST,
    HAS_TF,
    INK,
    INK_SOFT,
    MODEL_DIR,
    PALETTE_ADJACENT,
    SEED,
    TBL_DIR,
    display,
    keras,
    save_fig,
    save_table,
)
from s01_diagnose import THETA, TEST_START, TEST_END, df, operating_calendar  # noqa: F401
from s02_features import FEATURE_COLS, FEATURE_GROUPS, FEATURE_LABELS, feat  # noqa: F401
from s03_split import (  # noqa: F401
    ACTIVE_FOLDS,
    DATA_CONDITIONS,
    calendar_rule_predict,
    classification_metrics,
    get_fold_data,
    get_test_data,
    mae,
    regression_metrics,
    tune_tau,
)
# 4장(가이드북 재현)을 의존 체인에 포함시킨다.
# 노트북에서는 셀 순서로 자연히 실행되지만, 모듈 체인에서는 아무도 import 하지 않으면
# 4장이 통째로 실행되지 않는다(F16 미생성으로 적발됨).
from s04_baseline import (  # noqa: F401
    baseline_preds,
    corrected_tbl,
    defects_tbl,
    gate4,
    rf_original,
    rnn_original,
    silent_noop_tbl,
)
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 5. 제안모델 개발
#
# 보고서 **2.4절·2.5절**의 근거를 생성한다.
#
# 모델 수를 늘리기보다 **데이터 특성과 분석 목적이 다른 모델**을 중심으로 구성했다.
# 모든 모델은 3장에서 고정한 분할·마스크·지표를 그대로 쓰며,
# **`cond` 인자(D1/D2)를 받아** 6.5절에서 두 데이터 조건으로 재실행된다.

# %% [markdown]
# ### 5.0 공통 실행 하네스
#
# - **목적**: 모든 모델이 동일한 fold 루프·타이밍·OOF 수집 경로를 쓰게 한다.
# - **보고서 대응절**: 2.1, 2.6
# - **산출물**: `timing.csv`
#
# OOF(out-of-fold) 예측은 **7장 분석 전체의 입력**이다.
# 테스트 336시간은 피크가 28건뿐이라 조건별 분석이 불가능하므로,
# 7장은 반드시 fold 검증구간을 이어붙인 OOF 위에서 수행한다.

# %%
import lightgbm as lgb  # noqa: E402
from tools.model_persistence import predict_state, save_model_archive, state_from_output  # noqa: E402

# 재현성: 심사위원 PC의 코어 수와 무관하게 동일 결과가 나오도록 고정한다
LGB_BASE = dict(
    objective="l1",
    verbosity=-1,
    deterministic=True,
    force_row_wise=True,
    num_threads=4,
    seed=SEED,
    bagging_seed=SEED,
    feature_fraction_seed=SEED,
    data_random_seed=SEED,
)
N_TRIALS = 3 if FAST else 50
N_ESTIMATORS = 120 if FAST else 800

_TIMING: list[dict] = []


def _timed(label: str, fn, *args, **kwargs):
    """실행시간을 기록하며 함수를 호출한다 (보고서 2.6·2.9절 학습·추론시간)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _TIMING.append({"항목": label, "초": round(time.perf_counter() - t0, 3)})
    return out


def run_cv(name: str, fit_fn, cond: str = "D1", features=None, frame=None) -> dict:
    """fold 2~5를 돌며 OOF 예측과 fold별 지표를 수집한다.

    Parameters
    ----------
    name : str
        모델 표시명.
    fit_fn : callable
        `(Xtr, ytr, Xva) -> dict(pred_avg=, pred_peak=, prob=)`.
    cond : {'D1','D2'}
        데이터 조건.
    features : list[str], optional
        사용할 피처 부분집합(ablation 용).
    frame : pandas.DataFrame, optional
        피처 값을 바꾼 프레임(6.6절). 기본은 `feat`.

    Returns
    -------
    dict
        `oof`(DataFrame), `fold_metrics`(DataFrame), `tau`(float), `fit_sec`(float).
    """
    import gc

    parts, rows = [], []
    t0 = time.perf_counter()
    for fold in ACTIVE_FOLDS:
        Xtr, ytr, Xva, yva, idx = get_fold_data(fold, cond, features, frame)
        out = fit_fn(Xtr, ytr, Xva)
        pa = np.asarray(out["pred_avg"], float)
        pp = np.asarray(out.get("pred_peak", pa), float)
        prob = out.get("prob")
        part = pd.DataFrame(
            {
                "fold": fold,
                "y_avg": yva["y_avg"].to_numpy(),
                "y_peak": yva["y_peak"].to_numpy(),
                "y_cls": yva["y_cls"].to_numpy(),
                "pred_avg": pa,
                "pred_peak": pp,
                "prob": np.asarray(prob, float) if prob is not None else np.nan,
            },
            index=idx,
        )
        parts.append(part)
        # τ 는 **각 fold 검증구간에서만** 산출한다
        tau_f = tune_tau(yva["y_peak"], pp, THETA)
        m = regression_metrics(
            yva["y_avg"], pa, THETA, y_peak=yva["y_peak"], pred_peak=pp if prob is None else None
        )
        rows.append({"fold": fold, "n": len(idx), "양성": int(yva["y_cls"].sum()), "τ": tau_f, **m})
        # 다음 fold를 적합하기 전에 이전 모델/클로저를 해제해 메모리 중첩을 피한다.
        del out, Xtr, ytr, Xva, yva
        gc.collect()
    fit_sec = time.perf_counter() - t0
    _TIMING.append({"항목": f"{name} CV({cond})", "초": round(fit_sec, 3)})

    oof = pd.concat(parts)
    fold_metrics = pd.DataFrame(rows)
    # 테스트용 τ = fold τ 들의 중앙값 (테스트를 보고 정하지 않는다)
    tau = float(np.median(fold_metrics["τ"]))
    return {
        "name": name, "cond": cond, "oof": oof, "fold_metrics": fold_metrics,
        "tau": tau, "fit_sec": fit_sec / len(ACTIVE_FOLDS),
    }


def run_test(name: str, fit_fn, cond: str = "D1", features=None) -> dict:
    """학습 전체구간으로 적합해 테스트 336시간을 예측한다."""
    Xtr, ytr, Xte, yte, idx = get_test_data(cond, features)
    t0 = time.perf_counter()
    out = fit_fn(Xtr, ytr, Xte)
    infer_sec = time.perf_counter() - t0
    pa = np.asarray(out["pred_avg"], float)
    pp = np.asarray(out.get("pred_peak", pa), float)
    prob = out.get("prob")
    if features is None:
        expected = {"pred_avg": pa, "pred_peak": pp}
        if prob is not None:
            expected["prob"] = prob
        save_model_archive(
            MODEL_DIR / "comparison" / ("fast" if FAST else "full") / cond, name, state_from_output(out),
            Xtr.columns, Xte, expected,
            {"fast": FAST, "condition": cond, "seed": SEED,
             "train_start": str(Xtr.index.min()), "train_end": str(Xtr.index.max())},
        )
    return {
        "name": name, "cond": cond, "index": idx,
        "y_avg": yte["y_avg"].to_numpy(), "y_peak": yte["y_peak"].to_numpy(),
        "y_cls": yte["y_cls"].to_numpy(),
        "pred_avg": pa, "pred_peak": pp,
        "prob": np.asarray(prob, float) if prob is not None else None,
        "infer_sec": infer_sec,
        # 제출 예측을 만든 **바로 그 적합 객체**(LightGBM 계열만) — 10.5절 모델 번들이 저장한다.
        # RF·DNN 은 보관하지 않는다(메모리). 키 이름은 `_models` 가 아니다:
        # s06 measure_inference_time 이 `_models` 로 분기하므로 그 이름을 쓰면 INFER_SEC 가 바뀐다.
        "_artifacts": out.get("_artifacts"),
    }


# %% [markdown]
# ### 5.1 LightGBM 회귀 (주력) + Optuna 하이퍼파라미터 탐색
#
# - **목적**: 표 형태 소규모 데이터에서 생산량·시간·과거전력의 비선형 관계를 학습한다.
# - **보고서 대응절**: 2.4-1), 2.5
# - **산출물**: `hpo_trials.csv`, F26
#
# 탐색 공간은 `num_leaves[15,127]`, `learning_rate[0.01,0.2] log`, `max_depth[3,12]`,
# `min_child_samples[5,50]`, `feature_fraction[0.6,1.0]`, `bagging_fraction[0.6,1.0]`.
# 목적함수는 **fold 검증 MAE 평균**이며, 탐색 이력을 저장해 2.5절 "[TBD]회 탐색"의 근거로 쓴다.

# %%
import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_lgb_regressor(params: dict | None = None, n_estimators: int | None = None):
    """LightGBM 회귀 적합 함수를 만든다 (y_avg 와 y_peak 를 각각 적합)."""
    p = {**LGB_BASE, **(params or {})}
    n = n_estimators or N_ESTIMATORS

    def fit_fn(Xtr, ytr, Xva):
        models = {}
        for tgt in ("y_avg", "y_peak"):
            m = lgb.LGBMRegressor(n_estimators=n, **p)
            m.fit(Xtr, ytr[tgt])
            models[tgt] = m
        return {
            "pred_avg": models["y_avg"].predict(Xva),
            "pred_peak": models["y_peak"].predict(Xva),
            "_models": models,
            # 적합된 모델을 재사용하는 예측 클로저.
            # permutation importance 가 피처마다 재적합하지 않도록 노출한다.
            "_predict": lambda X: models["y_avg"].predict(X),
        }

    return fit_fn


def optimize_lgb(cond: str = "D1") -> tuple[dict, pd.DataFrame]:
    """Optuna TPE 로 LightGBM 회귀 하이퍼파라미터를 탐색한다.

    목적 = fold 2~5 검증 MAE 평균. 테스트는 전혀 보지 않는다.
    """

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
        }
        # 탐색 때와 선택 후 재학습 때 같은 트리 수를 사용한다.
        fit_fn = make_lgb_regressor(params, n_estimators=N_ESTIMATORS)
        scores = []
        for fold in ACTIVE_FOLDS:
            Xtr, ytr, Xva, yva, _ = get_fold_data(fold, cond)
            out = fit_fn(Xtr, ytr, Xva)
            scores.append(mae(yva["y_avg"], out["pred_avg"]))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    hist = study.trials_dataframe()[["number", "value"]].rename(
        columns={"number": "trial", "value": "fold평균MAE"}
    )
    hist["모델"] = "LightGBM 회귀"
    hist["n_estimators"] = N_ESTIMATORS
    hist["bagging_freq"] = 1
    # Optuna best_params에는 suggest_*로 탐색한 값만 들어간다.
    # 고정값을 복원하지 않으면 최종모델에서 bagging이 비활성화된다.
    return {**study.best_params, "bagging_freq": 1}, hist


lgb_best_params, hpo_reg_hist = _timed("Optuna LightGBM 회귀", optimize_lgb, "D1")
print(f"── LightGBM 회귀 Optuna {N_TRIALS}회 탐색 ──")
print(f"  최적 fold평균 MAE = {hpo_reg_hist['fold평균MAE'].min():.4f}")
for k, v in lgb_best_params.items():
    print(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}")

lgb_fit = make_lgb_regressor(lgb_best_params)

# %% [markdown]
# ### 5.2 2단계 레짐 모델
#
# - **목적**: 가동/비가동 상태를 먼저 분류한 뒤 상태별 회귀로 전력을 예측한다.
# - **보고서 대응절**: 2.4-2)
# - **산출물**: 레짐 모델 예측
#
# 전력 분포가 가동상태에 따라 **두 집단으로 갈라지는** 특성(1.3절: 생산량 0인 시간 43%,
# 평균전력 30 미만 33%)을 반영한다.
#
# 1단계는 **2분류(가동/비가동)로 시작**하고, 3분류(기저/중간/가동)는 비교 실험으로 둔다.
#
# > ⚠️ **게이트의 생산량 의존성**: 6.3절 '생산량 제외' ablation 에서는 게이트도
# > 생산량 비의존(lag 기반)으로 교체해야 검증이 성립한다. 그래서 게이트 피처를
# > 인자로 노출한다.

# %%
REGIME_CUT = 70.0  # 1.5절 EDA 의 가동/비가동 경계


def make_regime_model(n_classes: int = 2, gate_features=None):
    """2단계 레짐 모델 적합 함수를 만든다.

    Parameters
    ----------
    n_classes : {2, 3}
        1단계 분류 클래스 수. 2 = 가동/비가동, 3 = 기저/중간/가동.
    gate_features : list[str], optional
        1단계 게이트에 쓸 피처. 기본은 전체. '생산량 제외' ablation 에서
        생산량 비의존 피처만 넘겨 게이트의 독립성을 지킨다.
    """

    def regime_label(y_avg):
        if n_classes == 2:
            return (np.asarray(y_avg) >= REGIME_CUT).astype(int)
        return np.digitize(np.asarray(y_avg), [30.0, REGIME_CUT])

    def fit_fn(Xtr, ytr, Xva):
        gcols = [c for c in (gate_features or Xtr.columns) if c in Xtr.columns]
        rtr = regime_label(ytr["y_avg"])

        # 1단계 — 레짐 분류
        gate = lgb.LGBMClassifier(
            n_estimators=200 if FAST else 400,
            **{**LGB_BASE, "objective": "multiclass" if n_classes > 2 else "binary"},
        )
        gate.fit(Xtr[gcols], rtr)

        # 2단계 — 레짐별 회귀 (적합은 여기서 1회만)
        regimes = np.unique(rtr)
        regs: dict = {}
        fallbacks: dict = {}
        for tgt in ("y_avg", "y_peak"):
            regs[tgt] = {}
            for r in regimes:
                m_tr = rtr == r
                if m_tr.sum() < 20:
                    continue
                reg = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **LGB_BASE)
                reg.fit(Xtr[m_tr], ytr[tgt][m_tr])
                regs[tgt][int(r)] = reg
            fb = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **LGB_BASE)
            fb.fit(Xtr, ytr[tgt])
            fallbacks[tgt] = fb

        def predict(X, tgt: str = "y_avg") -> np.ndarray:
            """적합된 게이트·레짐별 회귀로 예측한다(재적합 없음)."""
            rr = gate.predict(X[gcols])
            out = np.empty(len(X), float)
            assigned = np.zeros(len(X), bool)
            for r, reg in regs[tgt].items():
                m = rr == r
                if m.any():
                    out[m] = reg.predict(X[m])
                    assigned |= m
            # 학습에 없던 레짐으로 배정된 행은 전체 적합 모델로 채운다
            if (~assigned).any():
                out[~assigned] = fallbacks[tgt].predict(X[~assigned])
            return out

        return {
            "pred_avg": predict(Xva, "y_avg"),
            "pred_peak": predict(Xva, "y_peak"),
            "_gate": gate,
            "_predict": lambda X: predict(X, "y_avg"),
            "_predict_peak": lambda X: predict(X, "y_peak"),
            # 모델 번들(10.5절)용 적합 객체. 클로저 `predict` 는 pickle 할 수 없으므로
            # 부스터와 라우팅 정보를 그대로 노출하고, 저장은 LightGBM 네이티브 텍스트로 한다.
            "_artifacts": {
                "kind": "regime",
                "n_classes": int(n_classes),
                "gcols": list(gcols),
                "gate": gate,
                "gate_n_estimators": int(gate.n_estimators),
                "classes": [int(c) for c in gate.classes_],
                "regs": regs,
                "fallbacks": fallbacks,
            },
        }

    return fit_fn


regime2_fit = make_regime_model(2)
regime3_fit = make_regime_model(3)

# %% [markdown]
# ### 5.3 DNN (Dense 전용 MLP)
#
# - **목적**: 참가계획서의 비교 모델을 Dense 전용 구조로 구현한다.
# - **보고서 대응절**: 2.2 표의 DNN 행, 2.4
# - **산출물**: DNN 예측
#
# 구조: `Dense(128) - BatchNorm - Dropout(0.2) - Dense(64) - Dense(1)`, Adam.
#
# > ⚠️ **`epochs` 를 반드시 명시한다.** Keras 의 `fit()` 기본값은 **1에폭**이라
# > 지정하지 않으면 학습이 사실상 이뤄지지 않는다. `epochs=300` + 조기종료.

# %%
DNN_EPOCHS = 5 if FAST else 300


def dnn_fit(Xtr, ytr, Xva):
    """Dense 전용 MLP. TensorFlow 미설치 시 None 을 반환해 건너뛴다."""
    if not HAS_TF:
        return None
    from sklearn.preprocessing import StandardScaler

    keras.utils.set_random_seed(SEED)
    xsc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = xsc.transform(Xtr), xsc.transform(Xva)

    preds, models, yscalers = {}, {}, {}
    for tgt in ("y_avg", "y_peak"):
        ysc = StandardScaler().fit(ytr[[tgt]])
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(Xtr_s.shape[1],)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
        # 조기종료용 검증집합을 **명시적으로** 자른다.
        # `validation_split=0.15` 는 셔플 전 배열의 뒤 15%를 쓰므로 여기서도
        # 결과는 같지만(Xtr 은 시간순이고 대상구간보다 앞서 끝난다 → 미래 누수 아님),
        # 암묵적 동작에 의존하면 읽는 사람이 누수로 오해한다. 경계를 눈에 보이게 둔다.
        n_es = max(int(len(Xtr_s) * 0.15), 24)
        model.fit(
            Xtr_s[:-n_es], ysc.transform(ytr[[tgt]]).ravel()[:-n_es],
            epochs=DNN_EPOCHS,          # ⚠️ 미지정 시 Keras 기본 1에폭
            batch_size=32, verbose=0,
            validation_data=(Xtr_s[-n_es:], ysc.transform(ytr[[tgt]]).ravel()[-n_es:]),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    "val_mae", patience=20, restore_best_weights=True
                )
            ],
        )
        preds[tgt] = ysc.inverse_transform(
            model.predict(Xva_s, verbose=0).reshape(-1, 1)
        ).ravel()
        models[tgt], yscalers[tgt] = model, ysc

    def predict(X, tgt):
        values = xsc.transform(X)
        return yscalers[tgt].inverse_transform(
            models[tgt].predict(values, verbose=0).reshape(-1, 1)
        ).ravel()

    return {"pred_avg": preds["y_avg"], "pred_peak": preds["y_peak"],
            "_predict": lambda X: predict(X, "y_avg"),
            "_predict_peak": lambda X: predict(X, "y_peak"),
            "_persist": {"kind": "keras", "models": models, "yscalers": yscalers, "xsc": xsc}}


# %% [markdown]
# ### 5.4 피크 직접 분류모델
#
# - **목적**: `peak15 >= θ` 여부를 LightGBM 분류로 **직접** 예측한다.
# - **보고서 대응절**: 2.4-3), 2.5
# - **산출물**: 피크 확률, `hpo_trials.csv` (분류)
#
# 피크는 전체의 약 5%로 불균형하므로 **클래스 가중치**로 양성 비중을 키운다.
# 회귀 예측값에 τ를 적용하는 방식과 성능을 비교한다.

# %%
def make_peak_classifier(params: dict | None = None):
    """피크 직접 분류기 적합 함수 (class_weight 적용)."""
    p = {**LGB_BASE, **(params or {})}
    p["objective"] = "binary"

    def fit_fn(Xtr, ytr, Xva):
        pos = max(int(ytr["y_cls"].sum()), 1)
        neg = len(ytr) - pos
        clf = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS, scale_pos_weight=neg / pos, **p
        )
        clf.fit(Xtr, ytr["y_cls"])
        prob = clf.predict_proba(Xva)[:, 1]
        # 회귀 예측이 없으므로 확률을 peak 스코어로 사용한다
        return {
            "pred_avg": np.full(len(Xva), np.nan), "pred_peak": prob, "prob": prob,
            # 모델 번들(10.5절)용. 파라미터는 모델 텍스트(6자리 반올림)가 아니라 이 dict 가 정본이다.
            "_artifacts": {
                "kind": "binary_clf", "clf": clf, "params": dict(p),
                "n_estimators": int(N_ESTIMATORS), "scale_pos_weight": float(neg / pos),
            },
        }

    return fit_fn


def optimize_peak_clf(cond: str = "D1") -> tuple[dict, pd.DataFrame]:
    """피크 분류기 하이퍼파라미터를 fold 평균 PR-AUC 최대화로 탐색한다."""
    from sklearn.metrics import average_precision_score

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
        }
        fit_fn = make_peak_classifier(params)
        scores = []
        for fold in ACTIVE_FOLDS:
            Xtr, ytr, Xva, yva, _ = get_fold_data(fold, cond)
            out = fit_fn(Xtr, ytr, Xva)
            scores.append(average_precision_score(yva["y_cls"], out["prob"]))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    hist = study.trials_dataframe()[["number", "value"]].rename(
        columns={"number": "trial", "value": "fold평균PR-AUC"}
    )
    hist["모델"] = "피크 직접분류"
    hist["n_estimators"] = N_ESTIMATORS
    hist["bagging_freq"] = 1
    return {**study.best_params, "bagging_freq": 1}, hist


clf_best_params, hpo_clf_hist = _timed("Optuna 피크분류", optimize_peak_clf, "D1")
peak_clf_fit = make_peak_classifier(clf_best_params)

hpo_trials = pd.concat([hpo_reg_hist, hpo_clf_hist], ignore_index=True)
save_table(hpo_trials, "hpo_trials")
HPO_N_TRIALS = len(hpo_trials)
print(f"── Optuna 총 탐색 횟수 = {HPO_N_TRIALS}회 (회귀 {len(hpo_reg_hist)} + 분류 {len(hpo_clf_hist)}) ──")
print(f"  분류 최적 fold평균 PR-AUC = {hpo_clf_hist['fold평균PR-AUC'].max():.4f}")


def plot_optuna_history(hist: pd.DataFrame):
    """F26 — Optuna 탐색 이력."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0))
    for ax, (mdl, col, better) in zip(
        axes,
        [("LightGBM 회귀", "fold평균MAE", "min"), ("피크 직접분류", "fold평균PR-AUC", "max")],
    ):
        sub = hist[hist["모델"] == mdl].dropna(subset=[col])
        if sub.empty:
            continue
        ax.plot(sub["trial"], sub[col], color=COLOR_MUTED, lw=1, marker="o", ms=4)
        run = sub[col].cummin() if better == "min" else sub[col].cummax()
        ax.plot(sub["trial"], run, color=COLOR_HERO, lw=2)
        ax.set_title(f"{mdl} — {col}", fontsize=10, color=INK)
        ax.set_xlabel("trial", fontsize=9)
    fig.tight_layout()
    return fig, hist


_fig, _src = plot_optuna_history(hpo_trials)
save_fig(_fig, "F26", "Optuna 탐색 이력", "5.1", source_table=_src)

# %% [markdown]
# ### 5.5 모델 레지스트리 및 교차검증 실행
#
# - **목적**: 제안모델·베이스라인을 한 레지스트리에 모아 동일 하네스로 실행한다.
# - **보고서 대응절**: 2.4, 2.6
# - **산출물**: `cv_results` (OOF), `test_results`
#
# 달력규칙 기준선(5.5)은 학습이 필요 없는 한 줄 규칙이므로 별도로 다룬다.

# %%
def rf_fit(Xtr, ytr, Xva):
    """보정 RandomForest (가이드북 계열 비교용)."""
    from sklearn.ensemble import RandomForestRegressor

    preds, models = {}, {}
    for tgt in ("y_avg", "y_peak"):
        m = RandomForestRegressor(
            n_estimators=100 if FAST else 300, max_depth=20, random_state=SEED, n_jobs=1
        )
        m.fit(Xtr, ytr[tgt])
        models[tgt] = m
        preds[tgt] = m.predict(Xva)
    return {
        "pred_avg": preds["y_avg"], "pred_peak": preds["y_peak"],
        "_models": models, "_predict": lambda X: models["y_avg"].predict(X),
    }


def naive_fit(Xtr, ytr, Xva):
    """Seasonal Naive — 168시간 전 값(이미 피처에 있다)을 그대로 쓴다."""
    return {
        "pred_avg": Xva["y_avg_lag168"].to_numpy(),
        "pred_peak": Xva["y_peak_lag168"].to_numpy(),
        "_persist": {"kind": "naive", "lag_hours": 168},
        "_predict": lambda X: X["y_avg_lag168"].to_numpy(),
    }


MODEL_REGISTRY = {
    "Seasonal Naive": naive_fit,
    "Random Forest (보정)": rf_fit,
    "LightGBM": lgb_fit,
    "2단계 레짐(2분류)": regime2_fit,
    "2단계 레짐(3분류)": regime3_fit,
    "피크 직접분류": peak_clf_fit,
}
if HAS_TF:
    MODEL_REGISTRY["DNN (MLP)"] = dnn_fit

cv_results: dict[str, dict] = {}
test_results: dict[str, dict] = {}
for _name, _fn in MODEL_REGISTRY.items():
    cv_results[_name] = run_cv(_name, _fn, "D1")
    test_results[_name] = run_test(_name, _fn, "D1")

# fold별 MAE 를 펼쳐서 본다 — 평균만 보면 가장 중요한 사실을 놓친다
fold_mae_matrix = pd.DataFrame(
    {k: v["fold_metrics"].set_index("fold")["MAE"] for k, v in cv_results.items()}
).T.round(3)
fold_mae_matrix.columns = [f"fold{c}" for c in fold_mae_matrix.columns]
save_table(fold_mae_matrix.reset_index(names="모델"), "ch5_fold_mae_matrix")

print("── 모델 × fold MAE (D1) ──")
display(fold_mae_matrix)
print(
    "\n  ⚠️ fold3·fold4 는 하계휴가(07-31~08-08)와 재가동을 포함한다.\n"
    f"     Seasonal Naive fold4 MAE = {fold_mae_matrix.loc['Seasonal Naive', 'fold4']:.1f} "
    "— lag168 참조값이 휴무일이라 완전히 붕괴한다.\n"
    f"     2단계 레짐(2분류) fold4 MAE = {fold_mae_matrix.loc['2단계 레짐(2분류)', 'fold4']:.1f} "
    "— 레짐을 먼저 분류해 붕괴를 막는다.\n"
    "     → 제안모델의 이점은 평균이 아니라 **달력이 깨지는 구간**에 집중된다(2.9절·7장 근거)."
)

print("\n── 모델별 fold 평균 성능 (D1) ──")
_summary = pd.DataFrame(
    [
        {
            "모델": k,
            "fold평균 MAE": round(float(v["fold_metrics"]["MAE"].mean()), 4),
            "fold MAE 표준편차": round(float(v["fold_metrics"]["MAE"].std()), 4),
            "τ(중앙값)": round(v["tau"], 1),
            "학습시간(초)": round(v["fit_sec"], 2),
        }
        for k, v in cv_results.items()
        if np.isfinite(v["fold_metrics"]["MAE"]).all()
    ]
).sort_values("fold평균 MAE")
display(_summary)

# %% [markdown]
# ### 5.6 앙상블 — **검증 성능이 실제로 개선될 때만** 채택
#
# - **목적**: 상위 모델의 가중평균이 단일모델을 이기는지 확인한다.
# - **보고서 대응절**: 2.4-4)
# - **산출물**: 앙상블 채택 여부
#
# "앙상블을 썼다"는 것 자체는 가점이 아니다. **검증 MAE가 실제로 낮아질 때만** 채택하고,
# 그렇지 않으면 채택하지 않았음을 명시한다.

# %%
def build_ensemble(cv_res: dict, cond: str = "D1") -> dict:
    """회귀 상위 2개 모델의 가중평균을 만들고 개선 여부를 판정한다."""
    cands = {
        k: v for k, v in cv_res.items()
        if np.isfinite(v["oof"]["pred_avg"]).all() and k != "Seasonal Naive"
    }
    ranked = sorted(cands.items(), key=lambda kv: kv[1]["fold_metrics"]["MAE"].mean())[:2]
    if len(ranked) < 2:
        return {"채택": False, "사유": "후보 부족"}

    (n1, r1), (n2, r2) = ranked
    common = r1["oof"].index.intersection(r2["oof"].index)
    y = r1["oof"].loc[common, "y_avg"]
    best = {"w": None, "mae": np.inf}
    for w in np.round(np.arange(0.0, 1.01, 0.05), 2):
        blend = w * r1["oof"].loc[common, "pred_avg"] + (1 - w) * r2["oof"].loc[common, "pred_avg"]
        m = mae(y, blend)
        if m < best["mae"]:
            best = {"w": float(w), "mae": float(m)}
    solo = float(mae(y, r1["oof"].loc[common, "pred_avg"]))
    improved = best["mae"] < solo - 1e-9
    return {
        "채택": bool(improved), "모델1": n1, "모델2": n2, "가중치1": best["w"],
        "단일 OOF MAE": round(solo, 4), "앙상블 OOF MAE": round(best["mae"], 4),
        "사유": "검증 MAE 개선" if improved else "검증 MAE 개선 없음 → 미채택",
    }


ensemble_info = build_ensemble(cv_results)
save_table(pd.DataFrame([ensemble_info]), "ch5_ensemble")
print("── 앙상블 판정 ──")
for k, v in ensemble_info.items():
    print(f"  {k}: {v}")

if ensemble_info.get("채택"):
    _n1, _n2, _w = ensemble_info["모델1"], ensemble_info["모델2"], ensemble_info["가중치1"]

    def ensemble_fit(Xtr, ytr, Xva, _n1=_n1, _n2=_n2, _w=_w):
        """상위 2개 모델의 가중평균."""
        a = MODEL_REGISTRY[_n1](Xtr, ytr, Xva)
        b = MODEL_REGISTRY[_n2](Xtr, ytr, Xva)

        def predict(X, key, callback):
            # 하위 모델의 적합 객체를 재사용한다. 중요도·추론시간 측정 중 재학습하지 않는다.
            values = [
                out[callback](X) if callable(out.get(callback))
                else predict_state(state_from_output(out), X)[key]
                for out in (a, b)
            ]
            return _w * np.asarray(values[0]) + (1 - _w) * np.asarray(values[1])

        return {
            "pred_avg": _w * np.asarray(a["pred_avg"]) + (1 - _w) * np.asarray(b["pred_avg"]),
            "pred_peak": _w * np.asarray(a["pred_peak"]) + (1 - _w) * np.asarray(b["pred_peak"]),
            "_predict": lambda X: predict(X, "pred_avg", "_predict"),
            "_predict_peak": lambda X: predict(X, "pred_peak", "_predict_peak"),
            "_persist": {"kind": "ensemble", "names": [_n1, _n2], "weight": _w,
                         "members": [state_from_output(a), state_from_output(b)]},
        }

    MODEL_REGISTRY["앙상블"] = ensemble_fit
    cv_results["앙상블"] = run_cv("앙상블", ensemble_fit, "D1")
    test_results["앙상블"] = run_test("앙상블", ensemble_fit, "D1")
    print("  → 앙상블을 레지스트리에 추가했다.")

# %% [markdown]
# ### 5.7 확률 보정 (Isotonic)
#
# - **목적**: 피크 확률이 실제 발생빈도와 일치하도록 보정한다.
# - **보고서 대응절**: 5장 창의성
# - **산출물**: `ch5_calibration.csv`, F21
#
# 경보 임계값을 확률로 운영하려면 "확률 0.7이면 실제로 70% 발생"이어야 한다.
# `class_weight` 로 양성 비중을 키우면 확률이 **체계적으로 과대추정**되므로 보정이 필요하다.
# 이전 OOF fold에만 보정기를 적합하고 다음 fold의 Brier·ECE를 비교한다.
# 확률분류기 HPO에는 같은 OOF가 쓰였으므로 완전히 독립적인 외부 검증은 아니다.
# 최종 배포 보정기는 전체 OOF로 다시 적합하며, 그 재적합 표본 점수는 보고하지 않는다.

# %%
from sklearn.isotonic import IsotonicRegression  # noqa: E402


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    """ECE — 예측확률과 실제 빈도의 가중 평균 절대차."""
    y_true, prob = np.asarray(y_true, float), np.asarray(prob, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = ((prob >= lo) if i == 0 else (prob > lo)) & (prob <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y_true[m].mean() - prob[m].mean())
    return float(ece)


def calibrate_probabilities(cv_res: dict, test_res: dict) -> tuple[pd.DataFrame, dict]:
    """이전 fold로 보정한 다음 fold의 점수와 전체 OOF 적합 보정기를 반환한다.

    첫 fold는 보정기 초기 적합에만 사용하고 이후 fold를 시간순으로 평가한다.
    ``test_res``는 호출 인터페이스를 유지하기 위한 인자이며 보정기·설정 선택에
    사용하지 않는다. 반환된 ``iso``만 전체 OOF를 사용한 최종 배포 보정기이다.
    """
    from sklearn.metrics import brier_score_loss

    oof = cv_res["피크 직접분류"]["oof"].sort_index()
    folds = list(oof["fold"].drop_duplicates())
    if len(folds) < 2 or not oof.index.is_unique:
        raise ValueError("시간순 보정검증에는 중복 없는 시각과 2개 이상의 fold가 필요합니다.")
    evaluated, fold_rows = [], []
    for i, fold in enumerate(folds[1:], start=1):
        cal = oof[oof["fold"].isin(folds[:i])]
        val = oof[oof["fold"] == fold].copy()
        if cal.index.max() >= val.index.min():
            raise ValueError("보정 fold는 평가 fold보다 시간상 앞서야 합니다.")
        iso_fold = IsotonicRegression(out_of_bounds="clip").fit(cal["prob"], cal["y_cls"])
        val["prob_cal"] = iso_fold.predict(val["prob"])
        evaluated.append(val)
        fold_rows.append({
            "평가 fold": fold, "보정 n": len(cal), "평가 n": len(val),
            "보정 시작": str(cal.index.min()), "보정 종료": str(cal.index.max()),
            "평가 시작": str(val.index.min()), "평가 종료": str(val.index.max()),
            "보정 전 Brier": brier_score_loss(val["y_cls"], val["prob"]),
            "보정 후 Brier": brier_score_loss(val["y_cls"], val["prob_cal"]),
            "보정 전 ECE": expected_calibration_error(val["y_cls"], val["prob"]),
            "보정 후 ECE": expected_calibration_error(val["y_cls"], val["prob_cal"]),
        })
    scored = pd.concat(evaluated).sort_index()
    y, p, p_cal = (scored[c].to_numpy() for c in ("y_cls", "prob", "prob_cal"))

    rows = [
        {"구분": label, "평가": "시간순 OOF 보정검증", "평가 n": len(y),
         "Brier": brier_score_loss(y, prob), "ECE": expected_calibration_error(y, prob)}
        for label, prob in (("보정 전", p), ("보정 후", p_cal))
    ]
    tbl = pd.DataFrame(rows).round(5)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof["prob"], oof["y_cls"])
    return tbl, {
        "iso": iso, "y": y, "p": p, "p_cal": p_cal,
        "fold_table": pd.DataFrame(fold_rows).round(5),
        "evaluation_predictions": scored[["fold", "y_cls", "prob", "prob_cal"]],
    }


calibration_tbl, _calib = calibrate_probabilities(cv_results, test_results)
save_table(calibration_tbl, "ch5_calibration")
save_table(_calib["fold_table"], "ch5_calibration_folds")
save_table(_calib["evaluation_predictions"].reset_index(names="시각"), "ch5_calibration_forward_predictions")
print("── 확률 보정 (이전 OOF fold 적합 → 다음 fold 평가) ──")
display(calibration_tbl)


def plot_reliability(y, p, p_cal):
    """F21 — Reliability diagram (보정 전/후 + 대각선)."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=COLOR_MUTED, lw=1.2, label="완전 보정")
    rows = []
    for arr, color, lab in ((p, PALETTE_ADJACENT[1], "보정 전"), (p_cal, COLOR_HERO, "보정 후")):
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = ((arr >= lo) if i == 0 else (arr > lo)) & (arr <= hi)
            if m.sum() < 5:
                continue
            xs.append(arr[m].mean())
            ys.append(np.asarray(y, float)[m].mean())
            rows.append({"구분": lab, "예측확률": arr[m].mean(),
                         "실제빈도": np.asarray(y, float)[m].mean(), "표본 n": int(m.sum())})
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, label=lab)
    ax.set_xlabel("예측 확률", fontsize=9)
    ax.set_ylabel("실제 발생 빈도", fontsize=9)
    ax.set_title("시간순 OOF 보정검증 Reliability diagram", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


_fig, _src = plot_reliability(_calib["y"], _calib["p"], _calib["p_cal"])
save_fig(_fig, "F21", "확률보정 Reliability", "5.7", source_table=_src)

# %% [markdown]
# ### 5.8 예측 불확실성 — 분위회귀 + Split Conformal
#
# - **목적**: 점추정만이 아니라 **예측구간**을 제시해 운영 판단에 폭을 준다.
# - **보고서 대응절**: 5장 창의성
# - **산출물**: `ch5_uncertainty.csv`, F22
#
# LightGBM 분위회귀로 q10·q50·q90을 적합하고 **실제 피복률**을 측정한다.
# 추가로 **Split Conformal** 유한표본 순서통계량으로 구간을 만든다.
# 피복 보장에는 보정·평가 점수의 교환가능성이 필요하며, 시계열 자료에서는 이를 보장하지 않는다.
# 휴무를 제외한 보정은 운영일 조건의 보조 분석이며 전체 날짜의 피복 보장으로 해석하지 않는다.

# %%
def conformal_quantile(scores, alpha: float = 0.10) -> float:
    """유한표본 보정 k=ceil((n+1)*(1-alpha))번째 비적합도 점수를 반환한다.

    표본이 적어 k>n이면 유한한 반폭으로 목표 피복을 보장할 수 없으므로 +inf이다.
    이 계산 자체가 시계열의 교환가능성이나 조건부 피복 보장을 만들어 주지는 않는다.
    """
    values = np.asarray(scores, dtype=float).ravel()
    if not 0 < alpha < 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("유한한 보정 점수와 0 < alpha < 1이 필요합니다.")
    k = int(np.ceil((len(values) + 1) * (1 - alpha)))
    return float(np.partition(values, k - 1)[k - 1]) if k <= len(values) else float("inf")


def fit_interval_models(Xtr: pd.DataFrame, ytr: pd.DataFrame, shutdown_days) -> dict:
    """분위회귀(q10·q50·q90)와 Split Conformal 을 **학습구간만으로** 적합한다.

    학습구간의 뒤 20%를 보정(calibration) 집합으로 떼어낸다. 평가(5.8절)와
    모델 번들(10.5절의 배포 번들)이 **같은 레시피**를 쓰도록 적합부만 분리한 함수다.

    Returns
    -------
    dict
        `qmodels`({0.1,0.5,0.9: 모델}), `point`, `qhat`({'전체(휴무 포함)','운영일만(휴무 제외)'}),
        `n_cal`, `n_cal_by_group`, `fit_last`(적합 구간 끝 시각).
    """
    n_cal = int(len(Xtr) * 0.2)
    if n_cal < 1 or n_cal >= len(Xtr):
        raise ValueError("분리된 적합·보정 집합을 만들 만큼 학습자료가 있어야 합니다.")
    Xfit, yfit = Xtr.iloc[:-n_cal], ytr.iloc[:-n_cal]
    Xcal, ycal = Xtr.iloc[-n_cal:], ytr.iloc[-n_cal:]

    qmodels = {}
    for q in (0.1, 0.5, 0.9):
        m = lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS, **{**LGB_BASE, "objective": "quantile", "alpha": q}
        )
        m.fit(Xfit, yfit["y_avg"])
        qmodels[q] = m

    # Split Conformal — 유한표본 순서통계량으로 목표 90%의 대칭 구간을 만든다.
    point = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **LGB_BASE)
    point.fit(Xfit, yfit["y_avg"])
    resid_all = np.abs(ycal["y_avg"].to_numpy() - point.predict(Xcal))

    # 보정집합의 휴무 상태를 제외한 변형도 제시하되, 운영일에만 평가·해석한다.
    # test 정답이나 피복률로 보정집합/alpha를 선택하지 않는다.
    # DatetimeIndex.isin() 은 Series 가 아니라 numpy 배열을 돌려준다
    cal_operating = ~np.asarray(Xcal.index.normalize().isin(shutdown_days))
    groups = {"전체(휴무 포함)": np.ones(len(resid_all), bool), "운영일만(휴무 제외)": cal_operating}
    qhat = {label: conformal_quantile(resid_all[mask]) for label, mask in groups.items()}
    return {
        "qmodels": qmodels, "point": point, "qhat": qhat,
        "n_cal": n_cal, "n_cal_by_group": {label: int(mask.sum()) for label, mask in groups.items()},
        "fit_last": str(Xfit.index.max()),
    }


def quantile_intervals(cond: str = "D1") -> tuple[pd.DataFrame, dict]:
    """분위회귀 예측구간과 Split Conformal 구간의 피복률을 비교한다."""
    Xtr, ytr, Xte, yte, idx = get_test_data(cond)
    shutdown_days = operating_calendar.index[operating_calendar["is_shutdown"]]
    fit = fit_interval_models(Xtr, ytr, shutdown_days)
    qmodels, point = fit["qmodels"], fit["point"]

    lo = qmodels[0.1].predict(Xte)
    mid = qmodels[0.5].predict(Xte)
    hi = qmodels[0.9].predict(Xte)
    cover_q = float(((yte["y_avg"] >= lo) & (yte["y_avg"] <= hi)).mean())
    p_te = point.predict(Xte)
    test_operating = ~np.asarray(idx.normalize().isin(shutdown_days))

    rows = [
        {"방법": "분위회귀 q10~q90", "보정집합": "별도 보정 없음", "평가범위": "전체 테스트",
         "평가 n": len(idx), "보정 n": 0, "목표 피복률": 0.80,
         "실제 피복률": round(cover_q, 4), "평균 구간폭": round(float(np.mean(hi - lo)), 2),
         "해석": "분위회귀 명목 80%; 경험적 피복률"},
    ]
    for label, qhat in fit["qhat"].items():
        mask = test_operating if label == "운영일만(휴무 제외)" else np.ones(len(idx), bool)
        covered = (yte["y_avg"].to_numpy() >= p_te - qhat) & (yte["y_avg"].to_numpy() <= p_te + qhat)
        cover = float(covered[mask].mean()) if mask.any() else float("nan")
        rows.append(
            {"방법": "Split Conformal", "보정집합": label, "평가범위": label,
             "평가 n": int(mask.sum()), "보정 n": fit["n_cal_by_group"][label], "목표 피복률": 0.90,
             "실제 피복률": round(cover, 4), "평균 구간폭": round(2 * qhat, 2),
             "해석": "시계열 교환가능성 미보장; " + ("운영일 한정" if label == "운영일만(휴무 제외)" else "전체 날짜")}
        )

    tbl = pd.DataFrame(rows)
    return tbl, {
        "idx": idx, "y": yte["y_avg"].to_numpy(), "lo": lo, "mid": mid, "hi": hi,
        "_fit": fit,  # 10.5절 평가 번들이 이 적합물을 그대로 저장한다
    }


uncertainty_tbl, _unc = _timed("분위회귀·Conformal", quantile_intervals, "D1")
save_table(uncertainty_tbl, "ch5_uncertainty")
print("── 예측 불확실성 ──")
display(uncertainty_tbl)


def plot_prediction_interval(u):
    """F22 — 예측구간 밴드와 실측점."""
    n = 168  # 테스트 앞 1주만 표시 (336 전부는 가독성이 떨어진다)
    idx, y, lo, mid, hi = u["idx"][:n], u["y"][:n], u["lo"][:n], u["mid"][:n], u["hi"][:n]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.fill_between(idx, lo, hi, color=COLOR_HERO, alpha=0.18, label="q10~q90 예측구간")
    ax.plot(idx, mid, color=COLOR_HERO, lw=1.6, label="q50 예측")
    out = (y < lo) | (y > hi)
    ax.scatter(idx[~out], y[~out], s=10, color=INK_SOFT, label="실측(구간 내)", zorder=3)
    ax.scatter(idx[out], y[out], s=18, color=PALETTE_ADJACENT[1], label="실측(구간 밖)", zorder=4)
    ax.set_ylabel("평균전력 (kW)", fontsize=9)
    ax.set_title("예측구간과 실측 (테스트 첫 7일)", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.20))
    fig.tight_layout()
    return fig, pd.DataFrame({"실측": y, "q10": lo, "q50": mid, "q90": hi}, index=idx)


_fig, _src = plot_prediction_interval(_unc)
save_fig(_fig, "F22", "예측구간 피복률", "5.8", source_table=_src)

timing_tbl = pd.DataFrame(_TIMING)
save_table(timing_tbl, "timing")
print(f"\n── 5장 누적 실행시간 {timing_tbl['초'].sum():.1f}초 ──")
