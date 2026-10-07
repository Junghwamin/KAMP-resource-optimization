# %% tags=["nb-strip"]
from s00_env import (  # noqa: F401
    CMAP_DIV,
    CMAP_SEQ,
    COLOR_HERO,
    COLOR_MUTED,
    FAST,
    HAS_TF,
    INK,
    INK_SOFT,
    PALETTE_ADJACENT,
    SEED,
    display,
    save_fig,
    save_table,
)
from s01_diagnose import THETA, TEST_START, TEST_END, df, operating_calendar  # noqa: F401
from s02_features import FEATURE_COLS, FEATURE_GROUPS, FEATURE_LABELS, feat  # noqa: F401
from s03_split import (  # noqa: F401
    ACTIVE_FOLDS,
    CALENDAR_RULE_FP,
    DATA_CONDITIONS,
    calendar_rule_metrics,
    calendar_rule_predict,
    classification_metrics,
    clopper_pearson,
    get_fold_data,
    get_test_data,
    mae,
    paired_bootstrap_mae,
    regression_metrics,
    tune_tau,
)
from s05_models import (  # noqa: F401
    LGB_BASE,
    MODEL_REGISTRY,
    N_ESTIMATORS,
    REGIME_CUT,
    _TIMING,
    clf_best_params,
    cv_results,
    ensemble_info,
    fold_mae_matrix,
    lgb_best_params,
    make_lgb_regressor,
    make_regime_model,
    run_cv,
    run_test,
    test_results,
    timing_tbl,
)
import time
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.model_persistence import predict_state, state_from_output
from tools.isolated_regime_cv import isolated_regime_predictions

# %% [markdown]
# ## 6. 성능평가 및 최종모델 선정
#
# 보고서 **2.6~2.9절**의 근거를 생성한다.
#
# **평가 층위를 명확히 분리한다.**
#
# | 층위 | 용도 | 표본 |
# |---|---|---|
# | **Rolling-origin OOF** | HPO·변수선택·임계값 튜닝에 재사용한 내부 검증 | 표본수는 산출표 참조 |
# | 테스트 336시간 | 최종 확인용 단일 수치 (헤드라인) | 양성 28건 |
#
# 테스트 피크 28건은 8개 날짜·16개 연속블록에만 분포해 Recall CI 폭이 0.37이다.
# 선정은 OOF 기준으로 하되, 반복 선택에 사용한 OOF를 독립 일반화 성능으로 해석하지 않는다.

# %% [markdown]
# ### 6.1 전력사용량 예측 성능 (Task A)
#
# - **목적**: 모델별 MAE·RMSE·sMAPE·Peak-MAE·학습시간을 표로 만들고 **개선의 통계적 유의성**을 검정한다.
# - **보고서 대응절**: 2.6 전력사용량 예측 성능
# - **산출물**: `ch2_regression.csv`, F17·F18·F25
#
# 검증 기준을 "MAE가 낮을 것"이 아니라 **"차이의 95% CI가 0을 포함하지 않을 것"** 으로 둔다.
# 일(day) 단위 블록 부트스트랩을 쓴다 — 시간별 오차는 하루 안에서 강하게 상관되므로
# 시간 단위 재표집은 구간을 과도하게 좁힌다.

# %%
# Task A 대상 모델 — 피크 직접분류는 회귀 예측이 없어 제외한다(pred_avg = NaN)
REGRESSION_MODELS = [
    k for k, v in cv_results.items()
    if np.isfinite(v["oof"]["pred_avg"]).all()
]
# 테스트 결과를 보고 비교기준을 고르지 않는다. 주 비교기준은 보정 RF로 고정한다.
REFERENCE_BASELINE_NAME = "Random Forest (보정)"
OOF_EVALUATION_NOTE = "HPO·변수선택·임계값 조정에 사용한 내부 검증(독립 평가 아님)"


def measure_inference_time(name: str, cond: str = "D1", n_rep: int = 10) -> float:
    """최종모델의 **336시간 배치 추론** 벽시계 시간(10회 중앙값)을 잰다.

    적합은 측정 전에 1회만 수행하고 두 타깃의 예측 함수를 재사용한다.
    예측 함수를 제공하지 않는 모델은 재학습 시간을 추론시간으로 대신 기록하지 않는다.
    """
    Xtr, ytr, Xte, _, _ = get_test_data(cond)
    fit_fn = MODEL_REGISTRY[name]
    out = fit_fn(Xtr, ytr, Xte)  # 적합은 1회만 (시간 측정 대상 아님)
    models = out.get("_models")
    predict_avg, predict_peak = out.get("_predict"), out.get("_predict_peak")
    state = None
    if not (callable(predict_avg) and callable(predict_peak)) and models is None:
        state = state_from_output(out)
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        if callable(predict_avg) and callable(predict_peak):
            predict_avg(Xte)
            predict_peak(Xte)
        elif models is not None:
            models["y_avg"].predict(Xte)
            models["y_peak"].predict(Xte)
        else:
            predict_state(state, Xte)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def build_regression_table(cond: str = "D1") -> pd.DataFrame:
    """2.6절 회귀 성능표. OOF와 테스트를 나란히 싣는다."""
    rows = []
    for name in REGRESSION_MODELS:
        cv = cv_results[name]
        te = test_results[name]
        oof = cv["oof"]
        oof_m = regression_metrics(oof["y_avg"], oof["pred_avg"], THETA,
                                   y_peak=oof["y_peak"], pred_peak=oof["pred_peak"])
        te_m = regression_metrics(te["y_avg"], te["pred_avg"], THETA,
                                  y_peak=te["y_peak"], pred_peak=te["pred_peak"])
        rows.append(
            {
                "구분": (
                    "기준" if name == "Seasonal Naive"
                    else "가이드북" if "Random Forest" in name or "RNN" in name
                    else "계획서" if "DNN" in name
                    else "제안"
                ),
                "모델": name,
                "예측지평": "Day-ahead",
                "OOF MAE": round(oof_m["MAE"], 3),
                "MAE": round(te_m["MAE"], 3),
                "RMSE": round(te_m["RMSE"], 3),
                "sMAPE": round(te_m["sMAPE"], 2),
                "Peak-MAE": round(te_m["Peak-MAE"], 3),
                "Peak15-MAE": round(te_m["Peak15-MAE"], 3),
                "Peak-n": te_m["Peak-n"],
                "High-load-MAE(legacy)": round(te_m["High-load-MAE"], 3),
                "High-load-n": te_m["High-load-n"],
                "OOF Peak-MAE": round(oof_m["Peak-MAE"], 3),
                "OOF Peak15-MAE": round(oof_m["Peak15-MAE"], 3),
                "OOF Peak-n": oof_m["Peak-n"],
                "OOF 해석": OOF_EVALUATION_NOTE,
                "fold MAE 표준편차": round(float(cv["fold_metrics"]["MAE"].std()), 3),
                "학습시간(초)": round(cv["fit_sec"], 2),
            }
        )
    return pd.DataFrame(rows).sort_values("OOF MAE").reset_index(drop=True)


regression_tbl = build_regression_table("D1")
save_table(regression_tbl, "ch2_regression")
print("── 2.6절 전력사용량 예측 성능 (D1) ──")
display(regression_tbl)

# 테스트 구간 마스킹 여부 확인 — 제출 파일은 336행을 유지해야 한다
_te_sub = feat.loc[test_results["LightGBM"]["index"]]
N_MASKED_IN_TEST = int((_te_sub["is_outage"] | _te_sub["is_erp_missing"]).sum())
print(
    f"\n  테스트 336행 중 마스크 대상 행 = {N_MASKED_IN_TEST}건 "
    "→ 마스킹 전/후 지표가 동일하다(계측정지·ERP결측이 9월에 없음). "
    "제출 파일은 336행을 그대로 유지한다."
)


def significance_vs_baseline(cond: str = "D1") -> tuple[pd.DataFrame, str, dict]:
    """사전에 고정한 보정 RF 대비 개선의 5% 주분석과 10% 사후 탐색."""
    base_name = REFERENCE_BASELINE_NAME
    base = test_results[base_name]
    rows = []
    for name in REGRESSION_MODELS:
        te = test_results[name]
        r = paired_bootstrap_mae(
            te["y_avg"], base["pred_avg"], te["pred_avg"], index=te["index"]
        )
        base_mae = mae(base["y_avg"], base["pred_avg"])
        this_mae = mae(te["y_avg"], te["pred_avg"])
        rows.append(
            {
                "모델": name,
                "MAE": round(this_mae, 3),
                f"기준({base_name}) MAE": round(base_mae, 3),
                "개선율(%)": round((base_mae - this_mae) / base_mae * 100, 2),
                "차이 95% CI": f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
                "차이 90% CI(사후 탐색)": f"[{r['ci90_low']:.3f}, {r['ci90_high']:.3f}]",
                # 표시용 문자열은 3자리로 반올림되므로, 판정 근거가 되는
                # 원시 경계값을 별도 컬럼으로 남긴다(0 근방에서 모호해지지 않게).
                "ci_low_raw": r["ci_low"],
                "ci_high_raw": r["ci_high"],
                "ci90_low_raw": r["ci90_low"],
                "ci90_high_raw": r["ci90_high"],
                "p-value": round(r["p_value"], 4),
                "p-value 산출법": r["p_method"],
                "유의": r["개선_5pct"],  # 기존 소비부는 5% 개선 판정을 유지
                "개선_5pct(주분석)": r["개선_5pct"],
                "개선_10pct(사후 탐색)": r["개선_10pct"],
                "차이 방향": "RF MAE − 비교모델 MAE (양수=개선)",
            }
        )
    return pd.DataFrame(rows).sort_values("개선율(%)", ascending=False), base_name, {}


significance_tbl, BEST_BASELINE_NAME, _ = significance_vs_baseline("D1")
save_table(significance_tbl, "ch2_significance")
BEST_BASELINE_MAE = float(
    mae(test_results[BEST_BASELINE_NAME]["y_avg"], test_results[BEST_BASELINE_NAME]["pred_avg"])
)
print(f"\n── MAE 개선 유의성 (기준 = {BEST_BASELINE_NAME}, MAE {BEST_BASELINE_MAE:.3f}) ──")
display(significance_tbl)

# %% [markdown]
# #### 그림 F17 · F18 · F25

# %%
def plot_model_mae(tbl: pd.DataFrame):
    """F17 — 모델별 Day-ahead MAE (강조형: 최우수 1색 + 나머지 회색)."""
    t = tbl.sort_values("OOF MAE")
    best = t["OOF MAE"].idxmin()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = [COLOR_HERO if i == best else COLOR_MUTED for i in t.index]
    bars = ax.bar(t["모델"], t["OOF MAE"], color=colors, width=0.62)
    for b, v in zip(bars, t["OOF MAE"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("OOF MAE (kW)", fontsize=9)
    ax.set_title("모델별 Day-ahead OOF MAE (낮을수록 우수)", fontsize=10, color=INK)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    return fig, t[["모델", "OOF MAE", "MAE"]]


_fig, _src = plot_model_mae(regression_tbl)
save_fig(_fig, "F17", "모델별 MAE", "6.1", source_table=_src)

# ⚠️ 여기서는 **잠정** 1위(OOF MAE 최소)만 잡는다.
# 최종모델은 6.4절 스코어카드(5개 기준 가중합)에서 확정되며 다를 수 있다.
# 예측 시계열·잔차 그림은 최종모델이 확정된 뒤(6.4절) 그린다.
BEST_MAE_MODEL = regression_tbl.iloc[0]["모델"]


def plot_pred_vs_actual(final: str, base: str):
    """F18 — 예측 vs 실측 시계열 (테스트 336h)."""
    te_f, te_b = test_results[final], test_results[base]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(te_f["index"], te_f["y_avg"], color=INK_SOFT, lw=1.4, label="실측")
    ax.plot(te_f["index"], te_f["pred_avg"], color=COLOR_HERO, lw=1.6, label=f"{final} 예측")
    ax.plot(te_b["index"], te_b["pred_avg"], color=COLOR_MUTED, lw=1.0, label=f"{base} 예측")
    ax.set_ylabel("평균전력 (kW)", fontsize=9)
    ax.set_title("테스트 336시간 예측 vs 실측", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    src = pd.DataFrame(
        {"실측": te_f["y_avg"], f"{final}": te_f["pred_avg"], f"{base}": te_b["pred_avg"]},
        index=te_f["index"],
    )
    return fig, src


def plot_residuals(final: str):
    """F25 — 잔차 분포 (예측값 대비)."""
    te = test_results[final]
    resid = te["y_avg"] - te["pred_avg"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.scatter(te["pred_avg"], resid, s=12, color=COLOR_HERO, alpha=0.5,
               linewidths=0.5, edgecolors="#ffffff")
    ax.axhline(0, color=PALETTE_ADJACENT[1], lw=1.2)
    ax.set_xlabel("예측값 (kW)", fontsize=9)
    ax.set_ylabel("잔차 (실측 - 예측)", fontsize=9)
    ax.set_title(f"{final} 잔차 분포", fontsize=10, color=INK)
    fig.tight_layout()
    return fig, pd.DataFrame({"예측": te["pred_avg"], "잔차": resid}, index=te["index"])


# %% [markdown]
# ### 6.2 피크 위험 탐지 성능 (Task B)
#
# - **목적**: 각 행이 **어떤 τ를 썼는지 노출**하고, **달력규칙을 기준 행으로 추가**한다.
# - **보고서 대응절**: 2.7 피크 위험 탐지 성능
# - **산출물**: `ch2_peak_detection.csv`, F19·F23
#
# 달력규칙 한 줄이 Recall 1.000·FP 82를 낸다. 제안모델의 기여는
# Recall 향상이 아니라 **동일 Recall 수준에서 FP를 줄이는 것**이다.
# 이 기준 행이 없으면 "Recall 0.9 달성"이 실제보다 대단해 보인다.

# %%
def build_peak_detection_table(cond: str = "D1") -> pd.DataFrame:
    """2.7절 피크 탐지 성능표. τ 컬럼과 달력규칙 기준 행을 포함한다."""
    rows = [
        {
            "구분": "기준(달력규칙)",
            "모델": "평일 ∧ 08≤h≤18",
            "τ": "-",
            "Precision": round(calendar_rule_metrics["Precision"], 4),
            "Recall": round(calendar_rule_metrics["Recall"], 4),
            "F1": round(calendar_rule_metrics["F1"], 4),
            "PR-AUC": np.nan,
            "TP": calendar_rule_metrics["TP"], "FP": calendar_rule_metrics["FP"],
            "FN": calendar_rule_metrics["FN"], "TN": calendar_rule_metrics["TN"],
            "산출법": "규칙 (학습 없음)",
        }
    ]
    for name, cv in cv_results.items():
        te = test_results[name]
        tau = cv["tau"]
        if te["prob"] is not None:
            # 피크 직접 분류 — 확률에 임계값 적용, PR-AUC는 확률로 계산
            score = te["prob"]
            tau_cls = tune_tau(
                cv["oof"]["y_peak"].to_numpy(), cv["oof"]["prob"].to_numpy(), THETA
            )
            pred = (score >= tau_cls).astype(int)
            tau_show, how = round(float(tau_cls), 3), "확률 임계값(OOF 산출)"
        else:
            # 회귀 → 판정: y_peak 예측에 τ 적용, PR-AUC는 연속 스코어로
            score = te["pred_peak"]
            pred = (score >= tau).astype(int)
            tau_show, how = round(tau, 1), "y_peak 예측 + τ (fold 중앙값)"
        m = classification_metrics(te["y_cls"], pred, score)
        rows.append(
            {
                "구분": "제안" if name not in ("Seasonal Naive", "Random Forest (보정)", "DNN (MLP)") else "베이스라인",
                "모델": name, "τ": tau_show,
                "Precision": round(m["Precision"], 4), "Recall": round(m["Recall"], 4),
                "F1": round(m["F1"], 4), "PR-AUC": round(m["PR-AUC"], 4),
                "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
                "Recall 95% CI": f"[{m['Recall_CI_low']:.3f}, {m['Recall_CI_high']:.3f}]",
                "산출법": how,
            }
        )
    out = pd.DataFrame(rows)
    # 달력규칙 대비 FP 감소량 — 제안모델의 실제 기여
    out["FP 감소(달력규칙 대비)"] = CALENDAR_RULE_FP - out["FP"]
    return out


peak_tbl = build_peak_detection_table("D1")
save_table(peak_tbl, "ch2_peak_detection")
print("── 2.7절 피크 위험 탐지 성능 (테스트 336h, D1) ──")
display(peak_tbl.drop(columns=["산출법"]))
print(
    "\n  ⚠️ 달력규칙이 Recall 1.000 이므로 Recall 로는 이길 수 없다.\n"
    "     Recall·미탐지·오경보를 함께 비교하며 같은 Recall이라고 가정하지 않는다."
)

# OOF 기준 피크 탐지 (모델 비교의 주 근거)
def build_peak_oof_table() -> pd.DataFrame:
    """OOF 기준 피크 탐지 성능 — 양성 156건으로 테스트보다 5.6배 많다."""
    rows = []
    for name, cv in cv_results.items():
        oof = cv["oof"]
        if oof["prob"].notna().all():
            score = oof["prob"].to_numpy()
            tau = tune_tau(oof["y_peak"].to_numpy(), score, THETA)
        else:
            score = oof["pred_peak"].to_numpy()
            tau = cv["tau"]
        m = classification_metrics(oof["y_cls"], (score >= tau).astype(int), score)
        rows.append(
            {
                "모델": name, "τ": round(float(tau), 3),
                "Precision": round(m["Precision"], 4), "Recall": round(m["Recall"], 4),
                "F1": round(m["F1"], 4), "PR-AUC": round(m["PR-AUC"], 4),
                "lift": round(m["lift"], 3), "양성률": round(m["양성률"], 4),
                "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
                "평가 성격": OOF_EVALUATION_NOTE,
            }
        )
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


peak_oof_tbl = build_peak_oof_table()
save_table(peak_oof_tbl, "ch2_peak_detection_oof")
print("\n── 피크 탐지 성능 (OOF, 양성 156건 — 비교의 주 근거) ──")
display(peak_oof_tbl)


def build_peak_temporal_threshold_table() -> pd.DataFrame:
    """앞선 OOF fold에서만 τ를 정해 다음 fold에 적용하는 보조 진단.

    임계값 선택과 평가는 분리하지만 기존 OOF 예측을 재사용하므로 HPO·모델
    선택까지 독립인 nested 검증은 아니다. 첫 fold는 임계값 보정용으로만 쓴다.
    """
    rows = []
    for name, cv in cv_results.items():
        oof = cv["oof"].sort_index()
        score_col = "prob" if oof["prob"].notna().all() else "pred_peak"
        parts = []
        for fold in sorted(oof["fold"].unique())[1:]:
            current = oof[oof["fold"] == fold]
            earlier = oof[oof.index < current.index.min()]
            tau = tune_tau(earlier["y_peak"], earlier[score_col], THETA)
            part = current[["y_cls"]].copy()
            part["pred"] = (current[score_col] >= tau).astype(int)
            part["score"] = current[score_col]
            parts.append(part)
            metrics = classification_metrics(part["y_cls"], part["pred"], part["score"])
            rows.append({"모델": name, "평가구간": f"fold{fold}", "n": len(part), "τ": tau,
                         "보정 종료": str(earlier.index.max()), "평가 시작": str(current.index.min()),
                         **metrics})
        if parts:
            pooled = pd.concat(parts)
            rows.append({"모델": name, "평가구간": "시간순 임계값 평가 합계", "n": len(pooled),
                         "τ": np.nan, "보정 종료": "이전 fold까지", "평가 시작": str(pooled.index.min()),
                         **classification_metrics(pooled["y_cls"], pooled["pred"], pooled["score"])})
    result = pd.DataFrame(rows)
    result["평가 성격"] = "임계값만 시간순 분리; HPO·모델선택은 기존 OOF 재사용"
    return result


peak_temporal_threshold_tbl = build_peak_temporal_threshold_table()
save_table(peak_temporal_threshold_tbl, "ch2_peak_threshold_temporal")


def plot_pr_curve():
    """F19 — 동일 OOF 행에서 곡선·달력규칙·무기술선을 비교한다."""
    from sklearn.metrics import precision_recall_curve

    names = [n for n in dict.fromkeys(["피크 직접분류", BEST_MAE_MODEL]) if n in cv_results]
    if not names:
        raise ValueError("F19 requires OOF predictions")
    idx = cv_results[names[0]]["oof"].index
    for name in names[1:]:
        idx = idx.intersection(cv_results[name]["oof"].index)
    idx = idx.sort_values()
    truth = cv_results[names[0]]["oof"].loc[idx, "y_cls"].to_numpy(int)
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    rows = []
    for name, color in zip(names, [COLOR_HERO, PALETTE_ADJACENT[1]]):
        oof = cv_results[name]["oof"].loc[idx]
        if not np.array_equal(truth, oof["y_cls"].to_numpy(int)):
            raise ValueError("F19 OOF target mismatch")
        score = oof["prob"].to_numpy() if oof["prob"].notna().all() else oof["pred_peak"].to_numpy()
        pr, rc, _ = precision_recall_curve(truth, score)
        ax.plot(rc, pr, color=color, lw=2, label=name)
        rows.append(pd.DataFrame({"모델": name, "Recall": rc, "Precision": pr,
                                  "평가구간": "공통 OOF", "n": len(idx)}))
    base = float(truth.mean())
    pred = ((idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour <= 18)).astype(int)
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    rc, pr = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    ax.axhline(base, color=COLOR_MUTED, lw=1.2, label=f"무기술선 ({base:.3f})")
    ax.scatter([rc], [pr], s=70, color=PALETTE_ADJACENT[2], zorder=5, label="달력규칙 (공통 OOF)")
    rows.append(pd.DataFrame([{"모델": "달력규칙", "Recall": rc, "Precision": pr,
                               "평가구간": "공통 OOF", "n": len(idx), "TP": tp, "FP": fp, "FN": fn}]))
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(f"PR 비교 (동일 OOF {len(idx):,}행)", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2)
    fig.tight_layout()
    return fig, pd.concat(rows, ignore_index=True)


_fig, _src = plot_pr_curve()
save_fig(_fig, "F19", "PR 곡선", "6.2", source_table=_src)


def plot_confusion(name: str):
    """F23 — 혼동행렬."""
    r = peak_tbl[peak_tbl["모델"] == name].iloc[0]
    cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap=CMAP_SEQ)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=13,
                    color="#ffffff" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["정상 예측", "피크 예측"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["실제 정상", "실제 피크"], fontsize=9)
    ax.set_title(f"혼동행렬 — {name}", fontsize=10, color=INK)
    ax.grid(False)
    fig.tight_layout()
    return fig, pd.DataFrame(cm, index=["실제 정상", "실제 피크"], columns=["정상 예측", "피크 예측"])


_clf_name = "피크 직접분류" if "피크 직접분류" in set(peak_tbl["모델"]) else BEST_MAE_MODEL
_fig, _src = plot_confusion(_clf_name)
save_fig(_fig, "F23", "혼동행렬", "6.2", source_table=_src)

# %% [markdown]
# ### 6.3 변수 제거 실험 (Ablation)
#
# - **목적**: 각 피처군이 실제로 기여하는지 확인한다.
# - **보고서 대응절**: 2.8 변수 제거 실험
# - **산출물**: `ch2_ablation.csv`, F20
#
# > ⚠️ **'생산량 제외' 행의 함정**: 2단계 레짐의 1단계 게이트가 생산량에 의존하면
# > 생산량을 제거해도 게이트를 통해 정보가 새어 들어온다. 그래서 이 행에서는
# > 게이트와 회귀에서 같은 생산정보 4개를 제거한다. 단 이 실험은 당일 생산량으로
# > 만든 휴무 캘린더를 유지한다. 계획 미확보 조건은 별도 확장 ablation으로 평가한다.

# %%
ABLATION_SPEC = [
    ("과거 전력 지연변수", ["과거전력"], "일·주 반복패턴의 기여"),
    ("생산량·공장인원", ["생산정보"], "생산계획 정보의 기여"),
    ("공정상태 변수", ["공정상태"], "가동 전환정보의 기여"),
    ("기상정보", ["기상정보"], "계절·냉방정보의 기여"),
]


def run_ablation(cond: str = "D1", base_model: str | None = None) -> pd.DataFrame:
    """피처군을 하나씩 제거하며 MAE·Recall 변화를 측정한다.

    기준 모델은 인자로 받는다(6.4절 확정 전이므로 잠정 1위를 쓰고,
    확정 모델이 다르면 6.4절에서 재실행한다).
    """
    FINAL_MODEL_NAME = base_model or BEST_MAE_MODEL
    full_cv = cv_results[FINAL_MODEL_NAME]
    base_mae = float(mae(full_cv["oof"]["y_avg"], full_cv["oof"]["pred_avg"]))
    base_oof = peak_oof_tbl.set_index("모델").loc[FINAL_MODEL_NAME]
    base_recall = float(base_oof["Recall"])

    base_fit = MODEL_REGISTRY[FINAL_MODEL_NAME]
    rows = []
    for label, groups, interp in ABLATION_SPEC:
        drop = {c for g in groups for c in FEATURE_GROUPS[g]}
        keep = [c for c in FEATURE_COLS if c not in drop]
        fit_fn = base_fit
        note = ""
        if "생산정보" in groups and "레짐" in FINAL_MODEL_NAME:
            # 직접 생산정보 4개는 게이트에서도 제거하되 휴무 파생변수는 유지한다.
            n_cls = 3 if "3분류" in FINAL_MODEL_NAME else 2
            fit_fn = make_regime_model(n_cls, gate_features=keep)
            note = "게이트도 직접 생산정보 제외; 생산량 유도 휴무·재가동 변수 유지"
        res = run_cv(f"ablation-{label}", fit_fn, cond, features=keep)
        oof = res["oof"]
        reg = regression_metrics(oof["y_avg"], oof["pred_avg"], THETA,
                                 y_peak=oof["y_peak"], pred_peak=oof["pred_peak"])
        a_mae = reg["MAE"]
        m = classification_metrics(
            res["oof"]["y_cls"],
            (res["oof"]["pred_peak"] >= res["tau"]).astype(int),
            res["oof"]["pred_peak"],
        )
        rows.append(
            {
                "제거 피처군": label,
                "MAE": round(a_mae, 3),
                "MAE 변화": round(a_mae - base_mae, 3),
                "Peak-MAE": round(reg["Peak-MAE"], 3),
                "Peak15-MAE": round(reg["Peak15-MAE"], 3),
                "Peak-n": reg["Peak-n"],
                "Recall": round(m["Recall"], 4),
                "Recall 변화": round(m["Recall"] - base_recall, 4),
                "해석": interp + (f" ({note})" if note else ""),
            }
        )

    # 레짐분리 제거 = 단일 LightGBM 으로 대체
    single = run_cv("ablation-레짐분리", make_lgb_regressor(lgb_best_params), cond)
    soof = single["oof"]
    sreg = regression_metrics(soof["y_avg"], soof["pred_avg"], THETA,
                              y_peak=soof["y_peak"], pred_peak=soof["pred_peak"])
    s_mae = sreg["MAE"]
    m = classification_metrics(
        single["oof"]["y_cls"],
        (single["oof"]["pred_peak"] >= single["tau"]).astype(int),
        single["oof"]["pred_peak"],
    )
    rows.append(
        {
            "제거 피처군": "레짐분리",
            "MAE": round(s_mae, 3),
            "MAE 변화": round(s_mae - base_mae, 3),
            "Peak-MAE": round(sreg["Peak-MAE"], 3),
            "Peak15-MAE": round(sreg["Peak15-MAE"], 3),
            "Peak-n": sreg["Peak-n"],
            "Recall": round(m["Recall"], 4),
            "Recall 변화": round(m["Recall"] - base_recall, 4),
            "해석": "가동·비가동 분리의 기여 (단일 LightGBM으로 대체)",
        }
    )
    out = pd.DataFrame(rows)
    out["평가 성격"] = OOF_EVALUATION_NOTE
    out.attrs["base_mae"] = base_mae
    out.attrs["base_recall"] = base_recall
    return out


def run_plan_information_ablation(base_model: str, cond: str = "D1") -> pd.DataFrame:
    """당일 생산계획 및 그 파생 캘린더까지 제거한 조건을 별도 평가한다.

    전일 휴무는 원점에서 이미 관측 가능한 과거 정보이므로 유지한다.
    기상 관측의 예보 대리 가정은 이 실험에서도 유지된다.
    """
    drop = set(FEATURE_GROUPS["생산정보"]) | {
        "is_shutdown", "shutdown_nth", "is_first_day_back", "is_state_switch",
    }
    keep = [c for c in FEATURE_COLS if c not in drop]
    fit_fn = MODEL_REGISTRY[base_model]
    if "레짐" in base_model:
        fit_fn = make_regime_model(3 if "3분류" in base_model else 2, gate_features=keep)
    result = run_cv("ablation-당일 생산계획 및 파생 캘린더", fit_fn, cond, features=keep)
    baseline = (cv_results[base_model] if cond == "D1"
                else run_cv("ablation-전체 피처", MODEL_REGISTRY[base_model], cond))
    rows = []
    for label, res in (("전체 피처", baseline),
                       ("당일 생산계획·파생 캘린더 제외", result)):
        oof = res["oof"]
        reg = regression_metrics(oof["y_avg"], oof["pred_avg"], THETA,
                                 y_peak=oof["y_peak"], pred_peak=oof["pred_peak"])
        cls = classification_metrics(oof["y_cls"], (oof["pred_peak"] >= res["tau"]).astype(int),
                                     oof["pred_peak"])
        rows.append({"조건": label, "모델": base_model, "n": len(oof), **reg, **cls})
    table = pd.DataFrame(rows)
    table["제거 변수"] = ["없음", ", ".join(sorted(drop))]
    table["유지 가정"] = "과거 생산·휴무 이력 및 실측 기상의 익일 예보 대리"
    table["평가 성격"] = OOF_EVALUATION_NOTE
    return table


ablation_tbl = run_ablation("D1", BEST_MAE_MODEL)
save_table(ablation_tbl, "ch2_ablation")
print(
    f"── 2.8절 Ablation (기준 = {BEST_MAE_MODEL}, "
    f"OOF MAE {ablation_tbl.attrs['base_mae']:.3f} / Recall {ablation_tbl.attrs['base_recall']:.3f}) ──"
)
display(ablation_tbl)


def plot_ablation(tbl: pd.DataFrame):
    """F20 — Ablation 효과 (발산형: 0 기준 ±)."""
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    for ax, col, title in (
        (axes[0], "MAE 변화", "MAE 변화 (클수록 그 피처군이 중요)"),
        (axes[1], "Recall 변화", "Recall 변화 (음수일수록 중요)"),
    ):
        vals = tbl[col].to_numpy()
        colors = [PALETTE_ADJACENT[1] if v > 0 else COLOR_HERO for v in vals]
        ax.barh(tbl["제거 피처군"], vals, color=colors, height=0.6)
        ax.axvline(0, color=INK_SOFT, lw=1)
        ax.set_xlabel("MAE 변화 (kW)" if col == "MAE 변화" else "Recall 변화 (비율)", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.tick_params(labelsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig, tbl


_fig, _src = plot_ablation(ablation_tbl)
save_fig(_fig, "F20", "Ablation 효과", "6.3", source_table=_src)

# %% [markdown]
# ### 6.4 최종모델 선정 스코어카드
#
# - **목적**: 보고서 2.9절의 5개 선정기준을 점수화해 선정 근거를 남긴다.
# - **보고서 대응절**: 2.9 최종모델 선정
# - **산출물**: `ch2_scorecard.csv`, `fold_matrix` 성능판, F24
#
# 선정기준 (보고서 2.9절 원문)
# 1) 전체구간과 피크구간에서 모두 안정적인 예측성능
# 2) 피크 미탐지를 줄일 수 있는 높은 Recall
# 3) 교차검증 구간별 성능 편차가 작을 것
# 4) 변수 중요도를 통해 예측 근거를 설명할 수 있을 것
# 5) 현장에서 매일 반복 실행할 수 있는 학습·추론시간

# %%
# 기준 5) 판정 임계 — 하루 1회 배치로 돌리기에 충분한 여유값.
# 실측 학습시간이 0.1~10초 범위이므로 경계에서 뒤집힐 위험이 없다.
DAILY_RUN_SEC_LIMIT = 60.0


def build_scorecard() -> pd.DataFrame:
    """5개 선정기준을 0~1로 정규화해 합산한다 (낮을수록 좋은 지표는 반전)."""
    rows = []
    for name in REGRESSION_MODELS:
        cv, te = cv_results[name], test_results[name]
        oof = cv["oof"]
        oof_m = regression_metrics(oof["y_avg"], oof["pred_avg"], THETA,
                                   y_peak=oof["y_peak"], pred_peak=oof["pred_peak"])
        rc = float(peak_oof_tbl.set_index("모델").loc[name, "Recall"])
        rows.append(
            {
                "모델": name,
                "1) OOF MAE": oof_m["MAE"],
                "1) OOF Peak-MAE": oof_m["Peak-MAE"],
                "참고: OOF Peak15-MAE": oof_m["Peak15-MAE"],
                "참고: 피크 n": oof_m["Peak-n"],
                "2) OOF Recall": rc,
                "3) fold MAE 표준편차": float(cv["fold_metrics"]["MAE"].std()),
                "4) 설명가능": 1.0,  # 트리계열·permutation 으로 전부 설명 가능
                # ⚠️ 기준 5)는 **벽시계 시간을 점수화하지 않는다.**
                #    연속값으로 넣으면 실행마다 종합점수가 미세하게 달라져
                #    최종모델 선정이 재현되지 않는다(재현성 검사에서 0.0067 차이로 적발됨).
                #    보고서 문구도 "현장에서 매일 반복 실행할 수 있는 시간"이라는
                #    **실행 가능성 판정**이므로 임계 기반 이진값이 더 충실하다.
                "5) 일일반복 실행가능": 1.0 if cv["fit_sec"] < DAILY_RUN_SEC_LIMIT else 0.0,
                "참고: 학습시간(초)": cv["fit_sec"],
                "평가 성격": OOF_EVALUATION_NOTE,
            }
        )
    sc = pd.DataFrame(rows).set_index("모델")

    def norm(s, lower_better=True):
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(1.0, index=s.index)
        z = (s - s.min()) / rng
        return 1 - z if lower_better else z

    score = (
        norm(sc["1) OOF MAE"]) * 0.30
        + norm(sc["1) OOF Peak-MAE"]) * 0.15
        + norm(sc["2) OOF Recall"], lower_better=False) * 0.25
        + norm(sc["3) fold MAE 표준편차"]) * 0.20
        + sc["5) 일일반복 실행가능"] * 0.10
    )
    sc["종합점수"] = score.round(4)
    return sc.sort_values("종합점수", ascending=False).round(4).reset_index()


scorecard = build_scorecard()
save_table(scorecard, "ch2_scorecard")
FINAL_MODEL_NAME = scorecard.iloc[0]["모델"]
print("── 2.9절 최종모델 선정 스코어카드 ──")
display(scorecard)
print(f"\n  ✅ 최종모델 = {FINAL_MODEL_NAME}")

if FINAL_MODEL_NAME != BEST_MAE_MODEL:
    print(
        f"  ↳ 잠정 1위(OOF MAE 최소)는 {BEST_MAE_MODEL} 였으나, 5개 기준 가중합에서는 "
        "위 모델이 선정되었다(폴드 편차·Recall·시간 반영)."
    )
    # 2.8절 Ablation 의 기준 모델을 확정 모델로 맞춘다
    ablation_tbl = run_ablation("D1", FINAL_MODEL_NAME)
    save_table(ablation_tbl, "ch2_ablation")
    _f, _s = plot_ablation(ablation_tbl)
    save_fig(_f, "F20", "Ablation 효과", "6.3", source_table=_s)

plan_ablation_tbl = run_plan_information_ablation(FINAL_MODEL_NAME)
save_table(plan_ablation_tbl, "ch2_plan_information_ablation")

# ── 최종모델이 확정된 **뒤에** 예측 시계열·잔차 그림을 그린다 ──────────
# (6.1절에서 그리면 스코어카드 결과와 다른 모델이 그려질 수 있다)
_fig, _src = plot_pred_vs_actual(FINAL_MODEL_NAME, BEST_BASELINE_NAME)
save_fig(_fig, "F18", "예측 vs 실측 시계열", "6.1", source_table=_src)
_fig, _src = plot_residuals(FINAL_MODEL_NAME)
save_fig(_fig, "F25", "잔차 분포", "6.1", source_table=_src)

# 최종 지표 확정 — 보고서 채움용 변수
_final_te = test_results[FINAL_MODEL_NAME]
_final_cv = cv_results[FINAL_MODEL_NAME]
FINAL_MAE = float(mae(_final_te["y_avg"], _final_te["pred_avg"]))
_final_metrics = regression_metrics(_final_te["y_avg"], _final_te["pred_avg"], THETA,
                                    y_peak=_final_te["y_peak"], pred_peak=_final_te["pred_peak"])
FINAL_PEAK_MAE = float(_final_metrics["Peak-MAE"])
FINAL_PEAK15_MAE = float(_final_metrics["Peak15-MAE"])
FINAL_PEAK_N = int(_final_metrics["Peak-n"])
IMPROVE_MAE_PCT = round((BEST_BASELINE_MAE - FINAL_MAE) / BEST_BASELINE_MAE * 100, 2)
_base_peak_mae = float(
    regression_metrics(
        test_results[BEST_BASELINE_NAME]["y_avg"],
        test_results[BEST_BASELINE_NAME]["pred_avg"], THETA,
        y_peak=test_results[BEST_BASELINE_NAME]["y_peak"],
        pred_peak=test_results[BEST_BASELINE_NAME]["pred_peak"],
    )["Peak-MAE"]
)
IMPROVE_PEAKMAE_PCT = round((_base_peak_mae - FINAL_PEAK_MAE) / _base_peak_mae * 100, 2)

_final_peak_row = peak_tbl[peak_tbl["모델"] == FINAL_MODEL_NAME].iloc[0]
FINAL_RECALL = float(_final_peak_row["Recall"])
FINAL_F1 = float(_final_peak_row["F1"])
_clf_row = peak_tbl[peak_tbl["모델"] == _clf_name].iloc[0]
CLF_RECALL, CLF_TP, CLF_FP, CLF_FN = (
    float(_clf_row["Recall"]), int(_clf_row["TP"]), int(_clf_row["FP"]), int(_clf_row["FN"])
)
TEST_POSITIVES = int(_final_te["y_cls"].sum())
INFER_SEC = round(measure_inference_time(FINAL_MODEL_NAME), 4)

print(f"  MAE {FINAL_MAE:.3f} (기준 {BEST_BASELINE_MAE:.3f} 대비 {IMPROVE_MAE_PCT:+.2f}%)")
print(f"  Peak-MAE {FINAL_PEAK_MAE:.3f} ({IMPROVE_PEAKMAE_PCT:+.2f}%)")
print(f"  Recall {FINAL_RECALL:.3f} / F1 {FINAL_F1:.3f}")
print(f"  336시간 배치 추론 {INFER_SEC:.4f}초 (10회 중앙값)")

# **설명용 대리모델 규칙** — 7장 SHAP 적용 경로를 여기서 확정한다
NEEDS_SURROGATE = "레짐" in FINAL_MODEL_NAME or "앙상블" in FINAL_MODEL_NAME
SURROGATE_NAME = "LightGBM" if NEEDS_SURROGATE else FINAL_MODEL_NAME
if NEEDS_SURROGATE:
    _a = _final_cv["oof"]["pred_avg"]
    _b = cv_results[SURROGATE_NAME]["oof"].reindex(_final_cv["oof"].index)["pred_avg"]
    SURROGATE_FIDELITY = float(np.corrcoef(_a, _b)[0, 1])
else:
    SURROGATE_FIDELITY = 1.0
print(
    f"\n  SHAP 적용 경로: 최종모델이 단일 트리가 아니므로 "
    f"{'대리모델 ' + SURROGATE_NAME if NEEDS_SURROGATE else '직접 적용'}"
    f" (대리 충실도 corr = {SURROGATE_FIDELITY:.4f})"
)


def plot_fold_matrix_heatmap():
    """F24 — 모델 × fold MAE 매트릭스."""
    m = fold_mae_matrix.dropna(how="all")
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    im = ax.imshow(m.to_numpy(), aspect="auto", cmap=CMAP_SEQ)
    ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns, fontsize=9)
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index, fontsize=8)
    for i in range(len(m.index)):
        for j in range(len(m.columns)):
            v = m.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if v > np.nanmax(m.to_numpy()) / 2 else INK)
    ax.set_title("모델 × fold MAE (fold3·4는 하계휴가 포함)", fontsize=10, color=INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.01).ax.tick_params(labelsize=8, colors=INK_SOFT)
    fig.tight_layout()
    return fig, m


_fig, _src = plot_fold_matrix_heatmap()
save_fig(_fig, "F24", "모델 fold 성능 매트릭스", "6.4", source_table=_src)

# %% [markdown]
# ### 6.5 데이터조건 D2 재실행 — 강건성 검증
#
# - **목적**: 복제 제거 조건(D2)에서도 **모델 순위가 보존되는지** 확인한다.
# - **보고서 대응절**: 2.6·2.7절 데이터조건 열, 2.9절 선정기준 3)
# - **산출물**: `ch2_d1_vs_d2.csv`, `ch6_rank_preservation.csv`
#
# > **중요한 실험 설계 결정**: D2에서 하이퍼파라미터를 **재탐색하지 않고 D1의 최적값을
# > 그대로 쓴다.** 재탐색하면 비교 대상이 "데이터 조건 차이"가 아니라
# > "서로 다른 HPO 결과"가 되어 강건성 주장이 성립하지 않는다.
#
# D2에서는 fold별 실제 표본수가 줄어들므로 **n을 반드시 병기**한다.

# %%
def rerun_for_d2() -> tuple[pd.DataFrame, pd.DataFrame]:
    """D1 하이퍼파라미터를 고정한 채 D2로 전 모델을 재실행한다."""
    d2_cv, d2_rows = {}, []
    for name, fit_fn in MODEL_REGISTRY.items():
        res = run_cv(name, fit_fn, "D2")
        d2_cv[name] = res
        oof = res["oof"]
        if np.isfinite(oof["pred_avg"]).all():
            metrics = regression_metrics(oof["y_avg"], oof["pred_avg"], THETA,
                                          y_peak=oof["y_peak"], pred_peak=oof["pred_peak"])
            d2_rows.append(
                {
                    "모델": name,
                    "D2 OOF MAE": round(metrics["MAE"], 3),
                    "D2 OOF Peak-MAE": round(metrics["Peak-MAE"], 3),
                    "D2 OOF Peak15-MAE": round(metrics["Peak15-MAE"], 3),
                    "D2 OOF Peak-n": metrics["Peak-n"],
                    "D2 검증 n": int(res["fold_metrics"]["n"].sum()),
                    "D2 양성": int(res["fold_metrics"]["양성"].sum()),
                }
            )
    d2_tbl = pd.DataFrame(d2_rows)

    d1_tbl = pd.DataFrame(
        [
            {
                "모델": n,
                "D1 OOF MAE": round(float(mae(cv_results[n]["oof"]["y_avg"], cv_results[n]["oof"]["pred_avg"])), 3),
                "D1 검증 n": int(cv_results[n]["fold_metrics"]["n"].sum()),
                "D1 양성": int(cv_results[n]["fold_metrics"]["양성"].sum()),
                **{
                    f"D1 OOF {key}": value
                    for key, value in regression_metrics(
                        cv_results[n]["oof"]["y_avg"], cv_results[n]["oof"]["pred_avg"], THETA,
                        y_peak=cv_results[n]["oof"]["y_peak"], pred_peak=cv_results[n]["oof"]["pred_peak"],
                    ).items() if key in ("Peak-MAE", "Peak15-MAE", "Peak-n")
                },
            }
            for n in REGRESSION_MODELS
        ]
    )
    comp = d1_tbl.merge(d2_tbl, on="모델", how="inner")
    comp["평가 성격"] = OOF_EVALUATION_NOTE
    comp["D1 순위"] = comp["D1 OOF MAE"].rank().astype(int)
    comp["D2 순위"] = comp["D2 OOF MAE"].rank().astype(int)
    comp["순위 보존"] = comp["D1 순위"] == comp["D2 순위"]

    from scipy.stats import spearmanr

    rho, pval = spearmanr(comp["D1 OOF MAE"], comp["D2 OOF MAE"])
    rank_tbl = pd.DataFrame(
        [
            {
                "지표": "Spearman 순위상관 (D1 vs D2)",
                "값": round(float(rho), 4),
                "p-value": round(float(pval), 4),
                "해석": "순위가 보존됨" if rho > 0.7 else "순위 변동 있음",
            },
            {
                "지표": "최종모델 D2 순위",
                "값": int(comp.loc[comp["모델"] == FINAL_MODEL_NAME, "D2 순위"].iloc[0]),
                "p-value": np.nan,
                "해석": "D2에서도 1위" if int(comp.loc[comp["모델"] == FINAL_MODEL_NAME, "D2 순위"].iloc[0]) == 1 else "D2에서 순위 하락",
            },
        ]
    )
    return comp, rank_tbl


d1_vs_d2, rank_preservation = rerun_for_d2()
save_table(d1_vs_d2, "ch2_d1_vs_d2")
save_table(rank_preservation, "ch6_rank_preservation")
print("── D1 vs D2 대비 (하이퍼파라미터는 D1 값 고정) ──")
display(d1_vs_d2)
print("\n── 순위 보존 검증 ──")
display(rank_preservation)

# %% [markdown]
# ### 6장 게이트 — 하한 통과 검증

# %%
def gate_chapter6() -> pd.DataFrame:
    """게이트 5 — Naive 하한(CI) 및 달력규칙 하한(FP 감소)을 통과해야 한다."""
    sig = significance_tbl[significance_tbl["모델"] == FINAL_MODEL_NAME]
    final_row = peak_tbl[peak_tbl["모델"] == FINAL_MODEL_NAME].iloc[0]
    checks = [
        ("최종모델이 사전 고정 RF보다 MAE 낮음", FINAL_MAE < BEST_BASELINE_MAE),
        ("5% 주분석: RF 대비 MAE 개선(95% CI 하한 > 0)", bool(sig["유의"].iloc[0]) if len(sig) else False),
        ("달력규칙 대비 FP 감소", int(final_row["FP"]) < CALENDAR_RULE_FP),
        ("D2에서 순위 보존(Spearman > 0.7)", float(rank_preservation.iloc[0]["값"]) > 0.7),
        ("Ablation 5행 산출", len(ablation_tbl) == 5),
        ("추론시간 측정 완료", INFER_SEC > 0),
    ]
    out = pd.DataFrame(checks, columns=["점검", "통과"])
    failed = out[~out["통과"]]
    if len(failed):
        print(f"⚠️ 게이트 5 미통과 항목: {failed['점검'].tolist()}")
    return out


gate5 = gate_chapter6()
save_table(gate5, "gate5_eval")
print("── 게이트 5 ──")
display(gate5)

# %% [markdown]
# ### 6.6 변수 제거 결과 검증 — 지연변수 제거 효과의 분해
#
# - **목적**: 6.3절에서 과거 전력 지연변수를 빼자 OOF MAE 가 오히려 좋아졌다. 이것이 최종모델을
#   바꿀 근거인지 **테스트를 보지 않고**, 결과를 보기 전에 정한 기준으로 판정한다.
# - **보고서 대응절**: 2.8, 2.9, 3.5
# - **산출물**: `ch2_lag_variants.csv`, `ch2_lag_decomposition.csv`, `ch2_lofo.csv`, `ch2_lag_d2.csv`, `ch2_gate_diagnosis.csv`
#
# **판단 기준 (먼저 고정)** — 변형이 아래를 모두 만족할 때만 최종모델 교체 후보로 본다.
#
# | 기준 | 내용 |
# |---|---|
# | ① | OOF MAE 개선의 일 단위 블록 부트스트랩 95% CI 가 0 을 포함하지 않는다 |
# | ② | 휴가·재가동이 없는 fold 2·5 각각에서 악화 폭이 잡음 폭 이하 |
# | ③ | 대상일과 7일 전의 가동 상태가 같은 날(정상일)에서 악화 폭이 잡음 폭 이하 |
# | ④ | Recall 하락 0.01 이내, Peak-MAE 악화가 Peak-MAE 잡음 폭 이하 |
# | ⑤ | 복제 제거 조건 D2 에서도 개선 방향이 같다 (①~④ 통과 변형과 주요 변형만 확인) |
#
# **잡음 폭** = 의미 없는 난수 변수 1개를 추가한 실행 5회의 |ΔMAE| 최댓값.
# 이 LightGBM 설정은 결정적(bagging·열 샘플링 없음)이라 시드를 바꿔도 결과가 같으므로,
# "아무 의미 없는 변화"가 MAE 를 얼마나 흔드는지를 이렇게 잰다.
#
# **변형** — 피처군 6개 제거, 지연 세분(24·48·168시간 각 2개), 지연+기상 동시 제거,
# 휴무 인지형 지연변수(참조일의 가동 상태가 대상일과 다르면 같은 상태의 과거일 값으로 대체) 2종,
# 잡음 기준선 5회, 44개 변수 개별 제거. 개별 제거는 44개를 동시에 비교하므로 우연히 기준을 넘는
# 변수가 나올 수 있다. 빠른 확인 모드(`KAMP_FAST=1`)에서는 개별 제거를 생략한다.
#
# > 아래 "오차 상위 2일 제외"·"토요일" 분해는 판정 기준이 아니라 **원인을 설명하는 진단**이다.

# %%
import inspect
from tools.isolated_regime_cv import isolated_regime_predictions

LAG_NOISE_SEEDS = (1000, 1001, 1002, 1003, 1004)
LAG_NORMAL_FOLDS = (2, 5)  # 하계휴가·재가동이 없는 fold
LAG_D2_FIXED = ("G-과거전력", "L168", "B+W", "SA-all", "SA-168")
LAG_DECOMP_IDS = ("A", "G-과거전력", "L48", "L168", "L24", "SA-all")


def status_matched_lag(frame, col: str, lag_h: int, step_days: int, max_steps: int):
    """참조일의 휴무 여부가 대상일과 다르면 step_days 씩 물러나 같은 상태의 날 같은 시각 값을 쓴다.

    참조일은 항상 대상일 − lag_h 이전이라 원점(대상일 00:00) 기준 인과적이다.
    끝까지 같은 상태의 날이 없으면 결측으로 둔다(LightGBM 이 결측을 처리한다).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        새 지연값, 물러난 횟수(0 = 원래 참조일 그대로, −1 = 찾지 못함).
    """
    status = operating_calendar["is_shutdown"].astype(int)
    day = frame.index.normalize()
    hour_td = pd.to_timedelta(frame.index.hour, unit="h")
    tgt = status.reindex(day).fillna(-1).to_numpy()
    y = frame[col]
    out = np.full(len(frame), np.nan)
    walked = np.full(len(frame), -1)
    todo = np.ones(len(frame), bool)
    ref = day - pd.Timedelta(hours=lag_h)
    for step in range(max_steps + 1):
        st = status.reindex(ref).fillna(-2).to_numpy()
        hit = todo & (st == tgt)
        vals = y.reindex(ref + hour_td).to_numpy()
        out[hit] = vals[hit]
        walked[hit] = step
        todo &= ~hit
        if not todo.any():
            break
        ref = ref - pd.Timedelta(days=step_days)
    return out, walked


def make_status_matched_frame(lags) -> pd.DataFrame:
    """지연변수를 휴무 인지형으로 바꾼 피처 프레임.

    24·48시간은 1일씩 최대 14일, 168시간은 같은 요일(7일)씩 최대 5주 물러난다.
    물러나지 않은 행은 원래 지연값과 정확히 같아야 한다(구현 검증).
    """
    f = feat.copy()
    for col in ("y_avg", "y_peak"):
        for lag in lags:
            step, max_steps = (7, 5) if lag == 168 else (1, 14)
            vals, walked = status_matched_lag(f, col, lag, step, max_steps)
            name = f"{col}_lag{lag}"
            same = walked == 0
            assert np.array_equal(vals[same], f[name].to_numpy()[same], equal_nan=True), name
            f[name] = vals
    return f


def lag_day_flags(index) -> dict:
    """진단용 구간 — 정상일(대상일과 7일 전의 가동 상태가 같음)과 토요일."""
    day = index.normalize()
    st = operating_calendar["is_shutdown"].astype(int)
    s0 = st.reindex(day).to_numpy()
    s7 = st.reindex(day - pd.Timedelta(days=7)).to_numpy()
    return {"정상일": s0 == s7, "토요일": index.dayofweek == 5}


def run_lag_variant(vid: str, desc: str, fit_fn=None, features=None, frame=None, cond: str = "D1") -> dict:
    """레짐 변형은 새 프로세스에서 적합하고 검증된 OOF 예측만 재사용한다.

    시드·스레드·트리 수·피처는 기존과 동일하다. 비레짐 최종모델은 기존 run_cv와
    fold별 GC를 사용한다. 캐시 적중 시 fit_sec는 최초 자식 적합시간이며 재학습시간이 아니다.
    """
    selected_fit = fit_fn or MODEL_REGISTRY[FINAL_MODEL_NAME]
    closure = inspect.getclosurevars(selected_fit).nonlocals
    isolated = None
    if "레짐" in FINAL_MODEL_NAME and "n_classes" in closure:
        folds = []
        for fold in ACTIVE_FOLDS:
            Xtr, ytr, Xva, _, _ = get_fold_data(fold, cond, features, frame)
            folds.append((fold, Xtr, ytr, Xva))
        isolated = isolated_regime_predictions(
            vid, folds, {"LGB_BASE": LGB_BASE, "N_ESTIMATORS": N_ESTIMATORS,
                         "REGIME_CUT": REGIME_CUT, "FAST": FAST},
            n_classes=closure["n_classes"], gate_features=closure.get("gate_features"),
            frame=feat if frame is None else frame, condition=cond,
        )
        predictions = iter(zip(folds, isolated["predictions"]))

        def selected_fit(Xtr, ytr, Xva):
            expected, prediction = next(predictions)
            if not Xva.index.equals(expected[3].index):
                raise ValueError(f"6.6 {vid}: isolated fold index differs from parent CV")
            return prediction

    res = run_cv(f"6.6-{vid}", selected_fit, cond, features=features, frame=frame)
    if isolated is not None:
        fit_seconds = float(sum(isolated["fit_seconds"]))
        res.update(fit_sec=fit_seconds / len(ACTIVE_FOLDS), cache_hit=isolated["cache_hit"],
                   fit_sec_source="checkpoint" if isolated["cache_hit"] else "isolated worker")
        _TIMING[-1]["초"] = round(fit_seconds, 3)
    res.update(vid=vid, desc=desc, n_feat=len(features) if features is not None else len(FEATURE_COLS))
    return res


def summarize_lag_variant(res: dict, base: dict | None, top_days) -> dict:
    """변형 하나의 지표. base 가 있으면 기준 대비 차이(변형 − 기준)와 부트스트랩 CI 를 붙인다."""
    oof = res["oof"]
    y, p = oof["y_avg"].to_numpy(), oof["pred_avg"].to_numpy()
    reg = regression_metrics(oof["y_avg"], oof["pred_avg"], THETA,
                             y_peak=oof["y_peak"], pred_peak=oof["pred_peak"])
    cls = classification_metrics(oof["y_cls"], (oof["pred_peak"] >= res["tau"]).astype(int), oof["pred_peak"])
    fm = res["fold_metrics"].set_index("fold")["MAE"]
    ae = np.abs(y - p)
    flags = lag_day_flags(oof.index)
    keep = ~oof.index.normalize().isin(top_days)
    row = {
        "ID": res["vid"], "변형": res["desc"], "조건": res["cond"], "피처수": res["n_feat"],
        "OOF MAE": reg["MAE"], "Peak-MAE": reg["Peak-MAE"], "Recall": cls["Recall"], "F1": cls["F1"],
        "Peak15-MAE": reg["Peak15-MAE"], "Peak-n": reg["Peak-n"],
        "평가 성격": OOF_EVALUATION_NOTE,
        "FP": cls["FP"], "FN": cls["FN"], "fold 표준편차": float(fm.std()),
        **{f"fold{f}": float(fm[f]) for f in ACTIVE_FOLDS},
        "정상일 MAE": float(ae[flags["정상일"]].mean()),
        "토요일 MAE": float(ae[flags["토요일"]].mean()),
        "상위2일 제외 MAE": float(ae[keep].mean()),
    }
    if base is None:
        return row
    assert base["oof"].index.equals(oof.index), "변형 간 OOF 행이 달라 짝지은 비교가 불가능하다"
    b = summarize_lag_variant(base, None, top_days)
    for k in ("OOF MAE", "Peak-MAE", "Recall", "F1", "fold 표준편차", *[f"fold{f}" for f in ACTIVE_FOLDS],
              "정상일 MAE", "토요일 MAE", "상위2일 제외 MAE"):
        row[f"Δ{k}"] = row[k] - b[k]
    # paired_bootstrap_mae 의 diff 는 MAE(기준) − MAE(변형) 이다 → 부호를 뒤집어 '변형 − 기준' 으로 적는다
    bs = paired_bootstrap_mae(y, base["oof"]["pred_avg"].to_numpy(), p, index=oof.index)
    row["ΔMAE CI 하한"], row["ΔMAE CI 상한"], row["p"] = -bs["ci_high"], -bs["ci_low"], bs["p_value"]
    row["ΔMAE 90% CI 하한(사후)"], row["ΔMAE 90% CI 상한(사후)"] = -bs["ci90_high"], -bs["ci90_low"]
    row["p 산출법"] = bs["p_method"]
    row["개선 fold 수"] = int(sum(row[f"Δfold{f}"] < 0 for f in ACTIVE_FOLDS))
    return row


def judge_lag_variants(tbl: pd.DataFrame, noise_mae: float, noise_peak: float) -> pd.DataFrame:
    """판단 기준 ①~④ 를 적용한다(잡음 폭은 기준선 실행에서 구한 값)."""
    out = tbl.copy()
    out["① CI가 0 미포함"] = out["ΔMAE CI 상한"] < 0
    out["② 정상 fold 악화 ≤ 잡음"] = np.logical_and.reduce(
        [out[f"Δfold{f}"] <= noise_mae for f in LAG_NORMAL_FOLDS]
    )
    out["③ 정상일 악화 ≤ 잡음"] = out["Δ정상일 MAE"] <= noise_mae
    out["④ Recall·Peak-MAE"] = (out["ΔRecall"] >= -0.01) & (out["ΔPeak-MAE"] <= noise_peak)
    out["①~④ 통과"] = out[["① CI가 0 미포함", "② 정상 fold 악화 ≤ 잡음", "③ 정상일 악화 ≤ 잡음", "④ Recall·Peak-MAE"]].all(axis=1)
    return out


def diagnose_gate_days(days, base_oof: pd.DataFrame) -> pd.DataFrame:
    """오차가 큰 날에 1단계 분류기가 어느 상태(레짐)를 배정했는지 직접 확인한다.

    해당 날이 검증구간인 fold 를 같은 적합 함수로 다시 학습해 게이트 출력을 읽는다
    (결정적 학습이라 교차검증 때의 게이트와 같다). 레짐 모델이 아니면 빈 표를 돌려준다.
    """
    rows = []
    fit_fn = MODEL_REGISTRY[FINAL_MODEL_NAME]
    for fold in sorted({int(base_oof.loc[base_oof.index.normalize() == d, "fold"].iloc[0]) for d in days}):
        Xtr, ytr, Xva, yva, idx = get_fold_data(fold, "D1")
        art = fit_fn(Xtr, ytr, Xva).get("_artifacts") or {}
        if art.get("kind") != "regime":
            return pd.DataFrame()
        gate = art["gate"]
        regime = pd.Series(gate.predict(Xva[art["gcols"]]), index=idx)
        splits = pd.Series(gate.booster_.feature_importance("split"), index=gate.booster_.feature_name())
        for d in days:
            m = idx.normalize() == d
            if not m.any():
                continue
            day_hours = regime[m]
            work = day_hours[(day_hours.index.hour >= 8) & (day_hours.index.hour <= 17)]
            dist = work.value_counts().sort_index()
            on_day = base_oof.index.normalize() == d
            rows.append({
                "날짜": d.strftime("%Y-%m-%d"), "요일": "월화수목금토일"[d.dayofweek], "fold": fold,
                "휴무일": bool(operating_calendar.loc[d, "is_shutdown"]),
                "실측 일평균": float(base_oof.loc[on_day, "y_avg"].mean()),
                "예측 일평균": float(base_oof.loc[on_day, "pred_avg"].mean()),
                "08~17시 배정 레짐": " · ".join(f"{int(k)}:{int(v)}시간" for k, v in dist.items()),
                "게이트 휴무일 변수 분기 수": int(splits.get("is_shutdown", 0)),
            })
    return pd.DataFrame(rows)


# ── 실행 ──────────────────────────────────────────────────────────────
_lag_t0 = time.perf_counter()
_lag_base = {**cv_results[FINAL_MODEL_NAME], "vid": "A", "desc": "최종모델 (44개)", "n_feat": len(FEATURE_COLS)}
_lag_oof = _lag_base["oof"]
_lag_daily = (_lag_oof["y_avg"] - _lag_oof["pred_avg"]).abs().groupby(_lag_oof.index.normalize()).mean()
LAG_TOP_DAYS = _lag_daily.nlargest(2).index
_lag_ae = (_lag_oof["y_avg"] - _lag_oof["pred_avg"]).abs()
LAG_TOP_SHARE = float(_lag_ae[_lag_oof.index.normalize().isin(LAG_TOP_DAYS)].sum() / _lag_ae.sum())

_lag_is_regime = "레짐" in FINAL_MODEL_NAME
_lag_n_cls = 3 if "3분류" in FINAL_MODEL_NAME else 2
lag_runs: dict = {"A": _lag_base}
for _g, _cols in FEATURE_GROUPS.items():
    _keep = [c for c in FEATURE_COLS if c not in set(_cols)]
    # 6.3절과 같이, 생산정보를 뺄 때는 1단계 분류기도 생산량 비의존으로 바꾼다
    _fit = make_regime_model(_lag_n_cls, gate_features=_keep) if (_g == "생산정보" and _lag_is_regime) else None
    lag_runs[f"G-{_g}"] = run_lag_variant(f"G-{_g}", f"{_g} {len(_cols)}개 제거", _fit, features=_keep)
for _lag in (24, 48, 168):
    _drop = {f"y_avg_lag{_lag}", f"y_peak_lag{_lag}"}
    lag_runs[f"L{_lag}"] = run_lag_variant(
        f"L{_lag}", f"{_lag}시간 전 2개 제거", features=[c for c in FEATURE_COLS if c not in _drop]
    )
_lag_drop_bw = set(FEATURE_GROUPS["과거전력"]) | set(FEATURE_GROUPS["기상정보"])
lag_runs["B+W"] = run_lag_variant("B+W", "지연 6개 + 기상 6개 제거", features=[c for c in FEATURE_COLS if c not in _lag_drop_bw])
LAG_SA_FRAMES = {"SA-all": make_status_matched_frame((24, 48, 168)), "SA-168": make_status_matched_frame((168,))}
lag_runs["SA-all"] = run_lag_variant("SA-all", "휴무 인지형 지연변수 (6개 교체)", frame=LAG_SA_FRAMES["SA-all"])
lag_runs["SA-168"] = run_lag_variant("SA-168", "휴무 인지형 지연변수 (168시간만 교체)", frame=LAG_SA_FRAMES["SA-168"])
for _i, _seed in enumerate(LAG_NOISE_SEEDS):
    _f = feat.copy()
    _f["noise"] = np.random.default_rng(_seed).standard_normal(len(_f))
    lag_runs[f"NULL-{_i}"] = run_lag_variant(
        f"NULL-{_i}", f"잡음 변수 추가 (시드 {_seed})", features=[*FEATURE_COLS, "noise"], frame=_f
    )
if not FAST:
    for _c in FEATURE_COLS:
        lag_runs[f"LOFO-{_c}"] = run_lag_variant(
            f"LOFO-{_c}", f"{FEATURE_LABELS.get(_c, _c)} 제거", features=[x for x in FEATURE_COLS if x != _c]
        )

_lag_rows = [summarize_lag_variant(r, None if k == "A" else _lag_base, LAG_TOP_DAYS) for k, r in lag_runs.items()]
_lag_all = pd.DataFrame(_lag_rows)
_lag_null = _lag_all[_lag_all["ID"].str.startswith("NULL-")]
LAG_NOISE_MAE = float(_lag_null["ΔOOF MAE"].abs().max())
LAG_NOISE_PEAK = float(_lag_null["ΔPeak-MAE"].abs().max())
_lag_judged = judge_lag_variants(_lag_all[_lag_all["ID"] != "A"], LAG_NOISE_MAE, LAG_NOISE_PEAK)
_lag_judged["구분"] = np.where(
    _lag_judged["ID"].str.startswith("NULL-"), "잡음 기준선",
    np.where(_lag_judged["ID"].str.startswith("LOFO-"), "개별 변수 제거", "변형"),
)
lag_variant_tbl = pd.concat([_lag_all[_lag_all["ID"] == "A"], _lag_judged[_lag_judged["구분"] != "개별 변수 제거"]])
lofo_tbl = _lag_judged[_lag_judged["구분"] == "개별 변수 제거"].copy()
if len(lofo_tbl):
    lofo_tbl["잡음 초과 개선·3/4 fold"] = (lofo_tbl["ΔOOF MAE"] < -LAG_NOISE_MAE) & (lofo_tbl["개선 fold 수"] >= 3)
LAG_N_VARIANTS = len(lag_runs)
LAG_GROUP_PASS = int(_lag_judged[_lag_judged["ID"].str.startswith("G-")]["①~④ 통과"].sum())

# ⑤ D2 — 주요 변형 + ①~④ 통과 변형(개별 제거 포함)
_lag_d2_ids = list(dict.fromkeys([*LAG_D2_FIXED, *_lag_judged.loc[_lag_judged["①~④ 통과"], "ID"]]))
_lag_d2_ids = [v for v in _lag_d2_ids if not v.startswith("NULL-")]
_lag_d2_base = run_lag_variant("A", "최종모델 (44개)", cond="D2")
_lag_d2_rows = [summarize_lag_variant(_lag_d2_base, None, LAG_TOP_DAYS)]
for _v in _lag_d2_ids:
    _r = lag_runs[_v]
    if _v.startswith("G-"):
        _g = _v[2:]
        _keep = [c for c in FEATURE_COLS if c not in set(FEATURE_GROUPS[_g])]
        _fit = make_regime_model(_lag_n_cls, gate_features=_keep) if (_g == "생산정보" and _lag_is_regime) else None
        _d2 = run_lag_variant(_v, _r["desc"], _fit, features=_keep, cond="D2")
    elif _v in LAG_SA_FRAMES:
        _d2 = run_lag_variant(_v, _r["desc"], frame=LAG_SA_FRAMES[_v], cond="D2")
    elif _v == "B+W":
        _d2 = run_lag_variant(_v, _r["desc"], features=[c for c in FEATURE_COLS if c not in _lag_drop_bw], cond="D2")
    elif _v.startswith("L") and _v[1:].isdigit():
        _drop = {f"y_avg_lag{_v[1:]}", f"y_peak_lag{_v[1:]}"}
        _d2 = run_lag_variant(_v, _r["desc"], features=[c for c in FEATURE_COLS if c not in _drop], cond="D2")
    else:  # LOFO-
        _c = _v[len("LOFO-"):]
        _d2 = run_lag_variant(_v, _r["desc"], features=[x for x in FEATURE_COLS if x != _c], cond="D2")
    _lag_d2_rows.append(summarize_lag_variant(_d2, _lag_d2_base, LAG_TOP_DAYS))
lag_d2_tbl = pd.DataFrame(_lag_d2_rows)
lag_d2_tbl["⑤ D2 개선 방향 유지"] = lag_d2_tbl.get("ΔOOF MAE", pd.Series(np.nan, index=lag_d2_tbl.index)) < 0

# 보고서 2.8절 표 — 지연변수 계열만 추린 분해
_lag_ref = lag_variant_tbl.set_index("ID")
lag_decomp_tbl = pd.DataFrame([
    {
        "변형": _lag_ref.loc[v, "변형"],
        "OOF MAE": _lag_ref.loc[v, "OOF MAE"],
        "상위2일 제외 MAE": _lag_ref.loc[v, "상위2일 제외 MAE"],
        "Δ상위2일 제외": _lag_ref.loc[v, "Δ상위2일 제외 MAE"] if v != "A" else 0.0,
        "토요일 MAE": _lag_ref.loc[v, "토요일 MAE"],
        "Δ토요일": _lag_ref.loc[v, "Δ토요일 MAE"] if v != "A" else 0.0,
        "ΔRecall": _lag_ref.loc[v, "ΔRecall"] if v != "A" else 0.0,
        "①~④ 통과": bool(_lag_ref.loc[v, "①~④ 통과"]) if v != "A" else None,
    }
    for v in LAG_DECOMP_IDS
])
gate_diag_tbl = diagnose_gate_days(list(_lag_daily.nlargest(4).index), _lag_oof)

save_table(lag_variant_tbl.round(6), "ch2_lag_variants")
save_table(lag_decomp_tbl.round(6), "ch2_lag_decomposition")
save_table(lofo_tbl.round(6), "ch2_lofo")
save_table(lag_d2_tbl.round(6), "ch2_lag_d2")
save_table(gate_diag_tbl, "ch2_gate_diagnosis")

print(f"── 6.6절 지연변수 제거 효과 분해 (변형 {LAG_N_VARIANTS}개 + D2 {len(lag_d2_tbl)}개, "
      f"{time.perf_counter() - _lag_t0:.0f}초) ──")
print(f"  최종모델 오차 상위 2일 = {', '.join(d.strftime('%m-%d') for d in LAG_TOP_DAYS)} "
      f"(OOF 절대오차의 {LAG_TOP_SHARE:.1%})")
print(f"  잡음 폭: MAE {LAG_NOISE_MAE:.3f} kW / Peak-MAE {LAG_NOISE_PEAK:.3f} kW")
print(f"  ①~④ 를 통과한 피처군 제거 변형: {LAG_GROUP_PASS}개")
display(lag_decomp_tbl.round(3))
if len(lofo_tbl):
    print("\n── 개별 변수 제거 — 개선 폭 상위 5 ──")
    display(lofo_tbl.nsmallest(5, "ΔOOF MAE")[["변형", "ΔOOF MAE", "ΔMAE CI 하한", "ΔMAE CI 상한", "개선 fold 수", "①~④ 통과"]].round(3))
print("\n── D2 강건성 ──")
display(lag_d2_tbl[["ID", "OOF MAE", *[c for c in ("ΔOOF MAE", "ΔMAE CI 하한", "ΔMAE CI 상한") if c in lag_d2_tbl]]].round(3))
if len(gate_diag_tbl):
    print("\n── 오차 상위 날짜의 1단계 분류(게이트) 배정 ──")
    display(gate_diag_tbl)

# %% [markdown]
# ### 6.7 결과 해석 — 보고서 서술을 실측에 맞춘다
#
# - **목적**: 유의성 검정 결과와 Ablation 결과가 보고서 초안의 서술과 어긋나는 지점을
#   찾아 **서술을 실측에 맞게 교정**한다.
# - **보고서 대응절**: 2.8, 2.9, 1.6 한계
# - **산출물**: `ch6_interpretation.csv`
#
# 여기서 나온 문장이 10.2절 문장 치환사전으로 들어간다.
# **실측과 다른 서술을 그대로 두면 감점 요인이다.**

# %%
def interpret_results() -> pd.DataFrame:
    """실측이 보고서 초안 서술과 어긋나는 지점을 정리한다."""
    rows = []

    # ① MAE 개선의 통계적 유의성
    sig = significance_tbl[significance_tbl["모델"] == FINAL_MODEL_NAME]
    is_sig = bool(sig["유의"].iloc[0]) if len(sig) else False
    is_exploratory = bool(sig["개선_10pct(사후 탐색)"].iloc[0]) if len(sig) else False
    ci = sig["차이 95% CI"].iloc[0] if len(sig) else "-"
    ci90 = sig["차이 90% CI(사후 탐색)"].iloc[0] if len(sig) else "-"
    approx_p = float(sig["p-value"].iloc[0]) if len(sig) else float("nan")
    if is_sig:
        narrative = (
            f"사전 고정 비교기준인 보정 Random Forest 대비 MAE 개선 {IMPROVE_MAE_PCT:+.2f}%는 "
            f"일 단위 블록 부트스트랩의 5% 주분석에서 확인되었다(95% CI {ci})."
        )
    else:
        narrative = (
            f"사전 고정 비교기준인 보정 Random Forest 대비 MAE 변화의 점추정은 "
            f"개선율 {IMPROVE_MAE_PCT:+.2f}%이나, 95% CI {ci}의 하한이 0을 넘지 않아 "
            "5% 주분석에서는 개선을 확인하지 못했다. "
            "평가기간은 14일이며 유의하지 않은 원인을 데이터 변형 등으로 단정하지 않는다."
        )
    narrative += (
        f" 10% 사후 탐색의 90% CI는 {ci90}이며 "
        + ("개선 방향의 차이를 보였다." if is_exploratory else "개선을 확인하지 못했다.")
        + " 사후 탐색 결과로 5% 주분석 판정을 대체하지 않는다. "
        f"양측 부트스트랩 꼬리비율 근사 p={approx_p:.4f}이며 정확검정의 p값은 아니다."
    )
    rows.append(
        {
            "항목": "MAE 개선의 유의성",
            "실측": f"{IMPROVE_MAE_PCT:+.2f}%, 95% CI {ci}, 90% CI {ci90}, 5% 개선={is_sig}, 10% 탐색={is_exploratory}",
            "보고서 서술 교정": narrative,
        }
    )
    rows.append({
        "항목": "OOF 해석 범위",
        "실측": OOF_EVALUATION_NOTE,
        "보고서 서술 교정": (
            "OOF는 하이퍼파라미터·변수구성·경보 임계값 선택에 재사용한 내부 검증이다. "
            "추가한 시간순 임계값 표는 이전 fold에서 τ를 정하고 이후 fold를 평가하지만, "
            "기존 OOF 예측을 사용하므로 HPO까지 분리한 독립 검증은 아니다."
        ),
    })
    rows.append({
        "항목": "피크 회귀 지표 정의",
        "실측": f"실제 피크 {FINAL_PEAK_N}시간, 평균전력 MAE {FINAL_PEAK_MAE:.3f}, 15분 최대전력 MAE {FINAL_PEAK15_MAE:.3f}",
        "보고서 서술 교정": (
            f"피크 탐지와 동일하게 실제 15분 최대전력이 {THETA:g} kW 이상인 시간을 선택했다. "
            f"테스트 {FINAL_PEAK_N}시간의 평균전력 MAE는 {FINAL_PEAK_MAE:.3f} kW, "
            f"15분 최대전력 MAE는 {FINAL_PEAK15_MAE:.3f} kW이다. "
            "종전 평균전력 기준 고부하 MAE는 별도 legacy 지표로 보존하였다."
        ),
    })

    # ② Ablation — 과거전력 지연변수 제거가 오히려 MAE를 낮추는가
    lag_row = ablation_tbl[ablation_tbl["제거 피처군"] == "과거 전력 지연변수"]
    if len(lag_row):
        d = float(lag_row["MAE 변화"].iloc[0])
        if d < 0:
            # 6.6절 분해 결과로 원인을 적는다(추정이 아니라 측정)
            g = lag_variant_tbl.set_index("ID").loc["G-과거전력"]
            base_row = lag_variant_tbl.set_index("ID").loc["A"]
            days = "·".join(f"{x.month}/{x.day}" for x in LAG_TOP_DAYS)
            dx = float(g["Δ상위2일 제외 MAE"])
            narrative = (
                f"과거 전력 지연변수를 제거하면 OOF MAE가 {d:+.3f} kW **개선**되지만, 6.6절 분해에서 이 개선은 "
                f"최종모델 오차 상위 2일({days}, OOF 절대오차의 {LAG_TOP_SHARE:.0%})에서 1단계 분류기가 "
                "계획상 휴무일을 가동 상태로 보낸 오분류가 사라진 효과로 확인된다. "
                f"두 날을 제외하면 제거 변형이 {dx:+.3f} kW {'나쁘고' if dx > 0 else '좋고'}, "
                f"토요일 MAE가 {base_row['토요일 MAE']:.2f} → {g['토요일 MAE']:.2f}로 변한다. "
                f"내부 검증에서 명시한 기준 ①~④를 통과한 피처군 제거 변형은 {LAG_GROUP_PASS}개이다. "
                "이 탐색은 같은 OOF를 반복 사용했으며 독립적인 효과 확인은 후속 기간 평가가 필요하다."
            )
        else:
            narrative = f"과거 전력 지연변수 제거 시 MAE가 {d:+.3f} kW 악화되어 기여가 확인된다."
        rows.append(
            {"항목": "과거전력 지연변수 Ablation", "실측": f"MAE 변화 {d:+.3f}", "보고서 서술 교정": narrative}
        )

    # ③ 레짐분리의 기여
    reg_row = ablation_tbl[ablation_tbl["제거 피처군"] == "레짐분리"]
    if len(reg_row):
        d = float(reg_row["MAE 변화"].iloc[0])
        rows.append(
            {
                "항목": "레짐분리 기여",
                "실측": f"MAE 변화 {d:+.3f}, Recall 변화 {float(reg_row['Recall 변화'].iloc[0]):+.4f}",
                "보고서 서술 교정": (
                    f"레짐분리를 단일 LightGBM으로 대체하면 OOF MAE가 {d:+.3f} kW 변한다. "
                    "단일 모델은 별도 HPO 설정을 사용하므로 순수한 구조 하나만의 인과효과로 해석하지 않는다."
                ),
            }
        )

    # ④ D2 강건성
    rho = float(rank_preservation.iloc[0]["값"])
    d2_rank = int(rank_preservation.iloc[1]["값"])
    rows.append(
        {
            "항목": "D2(복제 제거) 강건성",
            "실측": f"Spearman {rho:.4f}, 최종모델 D2 순위 {d2_rank}",
            "보고서 서술 교정": (
                f"복제를 제거한 D2 조건의 모델 순위상관은 {rho:.3f}이다. "
                + (
                    "최종모델이 D2에서도 1위를 유지한다."
                    if d2_rank == 1
                    else f"최종모델은 D2에서 {d2_rank}위로 순위가 달라졌다. "
                         "개별 모델의 우열은 데이터 조건에 따라 달라질 수 있다."
                )
            ),
        }
    )
    return pd.DataFrame(rows)


interpretation_tbl = interpret_results()
save_table(interpretation_tbl, "ch6_interpretation")
MAE_IMPROVEMENT_SIGNIFICANT = bool(
    significance_tbl[significance_tbl["모델"] == FINAL_MODEL_NAME]["유의"].iloc[0]
)
MAE_IMPROVEMENT_EXPLORATORY_10PCT = bool(
    significance_tbl[significance_tbl["모델"] == FINAL_MODEL_NAME]["개선_10pct(사후 탐색)"].iloc[0]
)
print("── 6.7절 결과 해석 (보고서 서술 교정) ──")
for _, r in interpretation_tbl.iterrows():
    print(f"\n  [{r['항목']}] {r['실측']}")
    print(f"    → {r['보고서 서술 교정']}")
