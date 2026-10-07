# %% tags=["nb-strip"]
from s00_env import (  # noqa: F401
    CMAP_DIV,
    CMAP_SEQ,
    COLOR_HERO,
    COLOR_MUTED,
    FAST,
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
    classification_metrics,
    clopper_pearson,
    get_fold_data,
    get_test_data,
    mae,
)
from s05_models import MODEL_REGISTRY, cv_results, lgb_best_params, make_lgb_regressor  # noqa: F401
from s06_eval import (  # noqa: F401
    FINAL_MODEL_NAME,
    SURROGATE_FIDELITY,
    SURROGATE_NAME,
    NEEDS_SURROGATE,
    peak_oof_tbl,
    test_results,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 7. 영향요인 및 오류분석
#
# 보고서 **3장**의 근거를 생성한다.
#
# > ### 🔒 이 장의 입력은 **전부 Rolling-origin OOF** 다
# >
# > 테스트 336시간에는 피크가 28건뿐이고 **7개 조건 × FN 6~8건 → 셀당 0~2건**이라
# > 3.4절 `"전체 미탐지의 [TBD]%"` 가 단일 사례로 결정된다.
# > 게다가 테스트는 9월뿐이라 **'여름 vs 그 외' 조건이 성립하지 않는다.**
# >
# > 따라서 **6장 성능표(헤드라인) = 테스트 336시간**, **7장 분석 전부 = OOF** 로 규칙을 고정한다.
# > 모든 조건별 표에 **표본수 n과 신뢰구간을 병기**한다.

# %%
OOF = cv_results[FINAL_MODEL_NAME]["oof"].copy()
OOF_TAU = cv_results[FINAL_MODEL_NAME]["tau"]
OOF = OOF.join(
    feat.loc[
        OOF.index,
        ["시간", "생산량", "기온", "prod_chg_ratio", "is_daytime", "is_shutdown",
         "prev_day_shutdown", "is_startup_08", "is_restart_13", "roll168_peak_max"],
    ]
)
OOF["abs_err"] = (OOF["y_avg"] - OOF["pred_avg"]).abs()
OOF["pred_cls"] = (OOF["pred_peak"] >= OOF_TAU).astype(int)

print(f"── 7장 입력: OOF {len(OOF)}행 (fold {sorted(OOF['fold'].unique())}) ──")
print(f"  구간 {OOF.index.min()} ~ {OOF.index.max()}")
print(f"  양성 {int(OOF['y_cls'].sum())}건 (테스트 28건의 {int(OOF['y_cls'].sum()) / 28:.1f}배)")
print(f"  최종모델 {FINAL_MODEL_NAME} / τ = {OOF_TAU:.1f}")

# %% [markdown]
# ### 7.1 주요 영향변수 분석 (Permutation + SHAP)
#
# - **목적**: 예측 근거가 되는 상위 변수를 찾고, 그 **방향성**까지 설명한다.
# - **보고서 대응절**: 3.1 주요 영향변수 분석
# - **산출물**: `ch3_importance.csv`, F27·F28
#
# **적용 경로를 명시적으로 나눈다.**
#
# | 분석 | 대상 | 이유 |
# |---|---|---|
# | **Permutation importance** | **파이프라인 전체** | 모델 구조와 무관하게 적용된다 → 전역 순위의 주 근거 |
# | SHAP | s06이 지정한 단일 트리 모델(필요 시 LightGBM **대리모델**) | 2단계 레짐은 설명 대상 단일 트리가 없고 base value·단위가 불일치(1단계 log-odds vs 2단계 kW) |
#
# 대리모델을 쓰는 경우 **대리 충실도**(두 모델 예측값 상관)를 반드시 함께 보고한다.

# %%
def permutation_importance_pipeline(model_name: str, cond: str = "D1", n_repeat: int = 3) -> pd.DataFrame:
    """파이프라인 전체에 대한 permutation importance (모델 구조 무관).

    각 fold에서 모델을 **1회만 적합**한 뒤, 검증셋의 컬럼을 하나씩 섞어
    MAE 증가량을 측정한다. 2단계 레짐·앙상블처럼 단일 트리가 없는 구조에도
    그대로 적용되므로 전역 변수 순위의 주 근거로 쓸 수 있다.

    ⚠️ 적합물을 재사용하는 `_predict` 클로저가 필수다. 피처마다 재적합하면
    44피처 × 3반복 × 4fold = 528회 학습이 되어 현실적으로 돌지 않는다.
    2단계 레짐의 경우 게이트까지 재적합되므로 특히 치명적이다.
    """
    fit_fn = MODEL_REGISTRY[model_name]
    rng = np.random.default_rng(SEED)
    acc: dict[str, list[float]] = {c: [] for c in FEATURE_COLS}
    for fold in ACTIVE_FOLDS:
        Xtr, ytr, Xva, yva, _ = get_fold_data(fold, cond)
        base_out = fit_fn(Xtr, ytr, Xva)
        predict = base_out.get("_predict")
        if predict is None:
            raise RuntimeError(
                f"{model_name} 이 `_predict` 클로저를 제공하지 않는다 — "
                "permutation importance 가 재적합 없이 돌 수 없다."
            )
        base_mae = mae(yva["y_avg"], base_out["pred_avg"])
        for col in FEATURE_COLS:
            deltas = []
            for _ in range(n_repeat):
                Xp = Xva.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                deltas.append(mae(yva["y_avg"], predict(Xp)) - base_mae)
            acc[col].append(float(np.mean(deltas)))
    rows = [
        {
            "변수": c,
            "한글라벨": FEATURE_LABELS.get(c, c),
            "중요도(MAE 증가)": round(float(np.mean(v)), 4),
            "표준편차": round(float(np.std(v)), 4),
        }
        for c, v in acc.items()
    ]
    return pd.DataFrame(rows).sort_values("중요도(MAE 증가)", ascending=False).reset_index(drop=True)


# permutation 은 fold×피처 만큼 예측하므로 FAST 모드에서는 반복을 줄인다
perm_importance = permutation_importance_pipeline(FINAL_MODEL_NAME, "D1", n_repeat=1 if FAST else 3)
save_table(perm_importance, "ch3_permutation")
print("── Permutation importance 상위 10 (파이프라인 전체) ──")
display(perm_importance.head(10))

TOP_FEATURES = perm_importance.head(5)["한글라벨"].tolist()
TOP_FEATURE_COLS = perm_importance.head(5)["변수"].tolist()


def compute_shap(surrogate: str, cond: str = "D1"):
    """SHAP 값을 계산한다 (필요 시 대리모델 사용).

    Returns
    -------
    (numpy.ndarray, pandas.DataFrame) | (None, None)
        SHAP 값과 대응 피처 행렬.
    """
    import shap

    Xtr, ytr, Xte, _, _ = get_test_data(cond)
    # s06에서 정한 설명 모델을 그대로 사용한다. RF를 선정한 실행을 LGB로 설명하지 않는다.
    model = MODEL_REGISTRY[surrogate](Xtr, ytr, Xte)["_models"]["y_avg"]
    # OOF 구간에서 설명한다(7장 규칙) — 표본이 많으면 샘플링
    Xexp = feat.loc[OOF.index, FEATURE_COLS]
    if len(Xexp) > 500:
        Xexp = Xexp.sample(500, random_state=SEED).sort_index()
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(Xexp), Xexp


shap_values, shap_X = compute_shap(SURROGATE_NAME, "D1")


def build_importance_table(perm: pd.DataFrame, sv, sx) -> pd.DataFrame:
    """3.1절 표 — 상위 5변수의 중요도·방향·현장 의미."""
    # SHAP 으로 방향(증가/감소)을 판정한다
    direction = {}
    if sv is not None:
        for i, c in enumerate(sx.columns):
            corr = np.corrcoef(sx[c].to_numpy(), sv[:, i])[0, 1]
            direction[c] = "증가" if corr > 0 else "감소"

    MEANING = {
        "y_avg_lag24": "공장의 일간 반복패턴을 반영한다",
        "y_peak_lag24": "전일 동시각 최대수요 수준을 반영한다",
        "y_avg_lag168": "주 단위 교대·요일 패턴을 반영한다",
        "y_peak_lag168": "주 단위 피크 재현 경향을 반영한다",
        "prod": "생산부하의 직접 효과를 반영한다",
        "prod_cum_day": "당일 누적 생산 진행도를 반영한다",
        "headcount": "투입 공수(가동 강도)를 반영한다",
        "hour": "시간대별 부하 프로파일을 반영한다",
        "is_daytime": "주간(9~17시) 교대 구간을 구분한다",
        "is_startup_08": "설비 동시 기동에 따른 부하 급증을 반영한다",
        "is_restart_13": "점심 후 재가동 부하를 반영한다",
        "is_shutdown": "휴무일 기저부하 상태를 구분한다",
        "roll24_avg_mean": "직전 24시간 평균 부하 수준을 반영한다",
        "roll168_avg_mean": "직전 1주 평균 부하 수준을 반영한다",
        "roll168_peak_max": "직전 1주 최대수요 이력을 반영한다",
        "temp": "냉난방 부하를 반영한다",
        "cdd": "여름철 냉방부하를 반영한다",
    }
    rows = []
    for i, r in perm.head(5).iterrows():
        c = r["변수"]
        rows.append(
            {
                "순위": i + 1,
                "변수": r["한글라벨"],
                "중요도": r["중요도(MAE 증가)"],
                "전력예측에 미친 방향": direction.get(c, "-"),
                "현장 의미": MEANING.get(c, "해당 피처군의 정보를 반영한다"),
            }
        )
    return pd.DataFrame(rows)


importance_tbl = build_importance_table(perm_importance, shap_values, shap_X)
save_table(importance_tbl, "ch3_importance")
print("\n── 3.1절 상위 5 영향변수 ──")
display(importance_tbl)
if NEEDS_SURROGATE:
    print(
        f"\n  ⚠️ SHAP 은 대리모델({SURROGATE_NAME})로 산출했다. "
        f"대리 충실도 corr(최종, 대리) = {SURROGATE_FIDELITY:.4f}\n"
        "     전역 변수 순위는 파이프라인 전체 permutation importance 를 따른다."
    )


def plot_permutation(perm: pd.DataFrame):
    """F27 — Permutation importance 상위 15."""
    top = perm.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = [COLOR_HERO if i >= len(top) - 3 else COLOR_MUTED for i in range(len(top))]
    ax.barh(top["한글라벨"], top["중요도(MAE 증가)"], color=colors, height=0.68,
            xerr=top["표준편차"], error_kw=dict(ecolor="#d6d5d1", lw=0.8))
    ax.set_xlabel("중요도 (섞었을 때 MAE 증가, kW)", fontsize=9)
    ax.set_title("Permutation importance 상위 15 (파이프라인 전체, OOF)", fontsize=10, color=INK)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig, perm.head(15)


_fig, _src = plot_permutation(perm_importance)
save_fig(_fig, "F27", "Permutation importance", "7.1", source_table=_src)


def plot_shap_summary(sv, sx):
    """F28 — SHAP summary (beeswarm)."""
    import shap

    fig = plt.figure(figsize=(7.4, 4.6))
    shap.summary_plot(
        sv, sx, feature_names=[FEATURE_LABELS.get(c, c) for c in sx.columns],
        max_display=15, show=False, plot_size=None,
    )
    fig = plt.gcf()
    fig.suptitle(
        f"SHAP summary ({'대리모델 ' + SURROGATE_NAME if NEEDS_SURROGATE else FINAL_MODEL_NAME})",
        fontsize=10, color=INK, y=1.01,
    )
    src = pd.DataFrame(
        {"변수": [FEATURE_LABELS.get(c, c) for c in sx.columns],
         "평균 |SHAP|": np.abs(sv).mean(axis=0).round(4)}
    ).sort_values("평균 |SHAP|", ascending=False)
    return fig, src


if shap_values is not None:
    _fig, _src = plot_shap_summary(shap_values, shap_X)
    save_fig(_fig, "F28", "SHAP summary", "7.1", source_table=_src)

# %% [markdown]
# ### 7.2 주요 변수의 상호작용
#
# - **목적**: 조건이 겹친 표본의 관측 부하를 비교한다. 인과적 상호작용 검정은 아니다.
# - **보고서 대응절**: 3.2 주요 변수의 상호작용
# - **산출물**: `ch3_interaction.csv`, F32

# %%
def analyze_interactions(o: pd.DataFrame) -> pd.DataFrame:
    """보고서 3.2절 표의 4개 상호작용을 OOF 실측으로 확인한다."""
    rows = []
    prod_hi = o["생산량"] > o["생산량"].quantile(0.75)

    # ① 고생산 집단 내 시간대 비교 — 생산량 자체를 매칭하거나 통제한 비교는 아니다
    sw = (o["is_startup_08"] == 1) | (o["is_restart_13"] == 1)
    a = float(o.loc[prod_hi & sw, "y_peak"].mean())
    b = float(o.loc[prod_hi & ~sw, "y_peak"].mean())
    rows.append(
        {
            "상호작용": "생산량 × 가동 전환시간",
            "확인할 내용": "생산량 상위 25% 시간 중 08·13시와 그 외의 관측 전력 비교",
            "고생산 & 전환시간": round(a, 1), "고생산 & 그 외": round(b, 1),
            "차이": round(a - b, 1),
            "n": int((prod_hi & sw).sum()),
            "분석 결과": "전환시간의 관측 평균이 높음(인과효과 미확인)" if a > b else "전환시간의 관측 평균 증가 미확인",
        }
    )

    # ② 생산량 × 주·야간
    a = float(o.loc[prod_hi & (o["is_daytime"] == 1), "y_peak"].mean())
    b = float(o.loc[prod_hi & (o["is_daytime"] == 0), "y_peak"].mean())
    rows.append(
        {
            "상호작용": "생산량 × 주·야간",
            "확인할 내용": "야간에 고생산 시 부하 특성이 달라지는가",
            "고생산 & 전환시간": round(a, 1), "고생산 & 그 외": round(b, 1),
            "차이": round(a - b, 1),
            "n": int(prod_hi.sum()),
            "분석 결과": "주간이 더 높다" if a > b else "야간이 더 높다",
        }
    )

    # ③ 과거 최대전력 × 생산량
    hi_hist = o["roll168_peak_max"] > o["roll168_peak_max"].median()
    a = float(o.loc[prod_hi & hi_hist, "y_peak"].mean())
    b = float(o.loc[prod_hi & ~hi_hist, "y_peak"].mean())
    rows.append(
        {
            "상호작용": "과거 최대전력 × 생산량",
            "확인할 내용": "고생산 시간 중 과거 최대부하 수준별 관측 전력 비교",
            "고생산 & 전환시간": round(a, 1), "고생산 & 그 외": round(b, 1),
            "차이": round(a - b, 1),
            "n": int((prod_hi & hi_hist).sum()),
            "분석 결과": "과거 고부하 집단의 관측 평균이 높음" if a > b else "관측 평균 증가 미확인",
        }
    )

    # ④ 기온 × 계절 — OOF 가 7~8월이라 CDD 구간으로 대체 확인
    hot = o["기온"] > o["기온"].quantile(0.75)
    a = float(o.loc[hot, "y_peak"].mean())
    b = float(o.loc[~hot, "y_peak"].mean())
    rows.append(
        {
            "상호작용": "기온 × 계절",
            "확인할 내용": "여름철 고온에서 냉방부하가 추가되는가",
            "고생산 & 전환시간": round(a, 1), "고생산 & 그 외": round(b, 1),
            "차이": round(a - b, 1),
            "n": int(hot.sum()),
            "분석 결과": (
                "고온 집단의 관측 평균이 높음(시간대·가동상태 효과 미분리)" if a > b
                else "고온 효과 미확인 (OOF가 7~8월에 한정)"
            ),
        }
    )
    return pd.DataFrame(rows)


interaction_tbl = analyze_interactions(OOF)
save_table(interaction_tbl, "ch3_interaction")
print("── 3.2절 상호작용 분석 (OOF) ──")
display(interaction_tbl)

TOP_INTERACTION = interaction_tbl.loc[interaction_tbl["차이"].idxmax(), "상호작용"]
print(f"\n  가장 큰 증가를 보인 조건: {TOP_INTERACTION}")


def plot_interaction_heatmap(o: pd.DataFrame):
    """F32 — 생산량 × 시간대 상호작용 히트맵."""
    tmp = o.copy()
    tmp["생산량구간"] = pd.qcut(tmp["생산량"].rank(method="first"), 4,
                            labels=["Q1(최저)", "Q2", "Q3", "Q4(최고)"])
    piv = tmp.pivot_table(index="생산량구간", columns="시간", values="y_peak",
                          aggfunc="mean", observed=False)
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    cmap = CMAP_SEQ.copy()
    cmap.set_bad("#ededed")
    im = ax.imshow(np.ma.masked_invalid(piv.to_numpy()), aspect="auto", cmap=cmap)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2), fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xlabel("시각 (회색 셀: 관측 없음)", fontsize=9)
    ax.set_title("생산량 구간 × 시간대 평균 peak15 (OOF)", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, pad=0.01).ax.tick_params(labelsize=8, colors=INK_SOFT)
    ax.grid(False)
    fig.tight_layout()
    return fig, piv.round(1)


_fig, _src = plot_interaction_heatmap(OOF)
save_fig(_fig, "F32", "상호작용 히트맵", "7.2", source_table=_src)

# %% [markdown]
# ### 7.3 조건별 예측오차 분석
#
# - **목적**: 모델이 **언제 실패하는지** 조건별로 분해한다.
# - **보고서 대응절**: 3.3 조건별 예측오차 분석 (표 7행 × 3열)
# - **산출물**: `ch3_condition_mae.csv`, F29
#
# 각 행에 **표본수 n과 95% 신뢰구간을 병기**한다. n이 작은 셀의 수치를
# 다른 셀과 나란히 놓으면 오독하게 된다.
#
# > ⚠️ **'계절' 행의 한계**: OOF 구간은 7~8월뿐이라 여름 대비가 성립하지 않는다.
# > 9월 테스트를 '그 외'로 쓰되 **평가 구간이 다름을 명시**한다.

# %%
def condition_error_table(o: pd.DataFrame) -> pd.DataFrame:
    """보고서 3.3절 7개 조건별 MAE 비교표를 만든다."""
    overall = float(o["abs_err"].mean())

    def ci(series):
        """일 단위 블록 재표집: 같은 날의 시간별 오차를 함께 뽑는다."""
        series = series.dropna()
        blocks = series.groupby(series.index.normalize()).agg(["sum", "count"])
        if len(blocks) < 2:
            return (float("nan"), float("nan"))
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(blocks), size=(2000, len(blocks)))
        totals = blocks["sum"].to_numpy()[draws].sum(axis=1)
        counts = blocks["count"].to_numpy()[draws].sum(axis=1)
        bounds = np.quantile(totals / counts, [0.025, 0.975])
        return tuple(float(value) for value in bounds)

    specs = [
        ("가동상태", "가동 / 비가동", o["is_shutdown"] == 0),
        ("생산량", "상위 25% / 나머지", o["생산량"] > o["생산량"].quantile(0.75)),
        ("가동전환", "08시·13시 / 그 외", (o["is_startup_08"] == 1) | (o["is_restart_13"] == 1)),
        ("주·야간", "주간 / 야간", o["is_daytime"] == 1),
        ("휴일 영향", "휴일 다음날 / 일반일", o["prev_day_shutdown"] == 1),
        ("생산량 변화", "전일 대비 ±50% 이상 / 그 외", o["prod_chg_ratio"].abs() >= 0.5),
    ]
    rows = []
    for label, cond_txt, mask in specs:
        a, b = o.loc[mask, "abs_err"], o.loc[~mask, "abs_err"]
        a_lo, a_hi = ci(a)
        rows.append(
            {
                "구분": label, "비교조건": cond_txt,
                "MAE(조건 충족)": round(float(a.mean()), 3) if len(a) else np.nan,
                "MAE(그 외)": round(float(b.mean()), 3) if len(b) else np.nan,
                "전체 평균 대비": (
                    f"{(a.mean() / overall - 1) * 100:+.1f}%" if len(a) else "-"
                ),
                "n(조건 충족)": int(len(a)), "n(그 외)": int(len(b)),
                "95% CI(조건 충족)": f"[{a_lo:.2f}, {a_hi:.2f}]" if len(a) > 1 else "-",
                "CI 방법": "일 단위 블록 bootstrap 2,000회(일 간 독립 가정)",
                "일 블록 수": int(a.index.normalize().nunique()),
                "비고": "",
            }
        )

    # 계절 — OOF(7~8월) vs 테스트(9월). 평가 구간이 다름을 명시한다.
    te = test_results[FINAL_MODEL_NAME]
    sept_err = pd.Series(np.abs(te["y_avg"] - te["pred_avg"]))
    s_lo, s_hi = ci(o["abs_err"])
    rows.append(
        {
            "구분": "계절", "비교조건": "여름(7~8월, OOF) / 그 외(9월, 테스트)",
            "MAE(조건 충족)": round(float(o["abs_err"].mean()), 3),
            "MAE(그 외)": round(float(sept_err.mean()), 3),
            "전체 평균 대비": "0.0%",
            "n(조건 충족)": int(len(o)), "n(그 외)": int(len(sept_err)),
            "95% CI(조건 충족)": f"[{s_lo:.2f}, {s_hi:.2f}]",
            "CI 방법": "일 단위 블록 bootstrap 2,000회(일 간 독립 가정)",
            "일 블록 수": int(o.index.normalize().nunique()),
            "비고": "⚠️ OOF가 7~8월에 한정되어 평가 구간이 다름 — 직접 비교 제한",
        }
    )
    out = pd.DataFrame(rows)
    out.attrs["overall"] = overall
    return out


condition_tbl = condition_error_table(OOF)
save_table(condition_tbl, "ch3_condition_mae")
print(f"── 3.3절 조건별 예측오차 (OOF 전체 평균 MAE {condition_tbl.attrs['overall']:.3f}) ──")
display(condition_tbl)

# 상위 3개 고오차 조건 — 3장 도입부와 3.3절 본문 두 곳에 같은 규칙으로 삽입한다
_cmp = condition_tbl[condition_tbl["구분"] != "계절"].copy()
_cmp["초과율"] = _cmp["MAE(조건 충족)"] / condition_tbl.attrs["overall"] - 1
_worst = _cmp.sort_values("초과율", ascending=False)
WORST_CONDITIONS = _worst.head(3)["구분"].tolist()
WORST_CONDITION_NAME = _worst.iloc[0]["구분"]
WORST_COND_EXCESS_PCT = round(float(_worst.iloc[0]["초과율"]) * 100, 1)
BEST_CONDITION_NAME = _worst.iloc[-1]["구분"]
print(f"\n  최대 오차 조건: {WORST_CONDITION_NAME} (전체 평균 대비 {WORST_COND_EXCESS_PCT:+.1f}%)")
print(f"  상위 3개 고오차 조건: {', '.join(WORST_CONDITIONS)}")
print(f"  가장 안정적인 조건: {BEST_CONDITION_NAME}")


def plot_condition_mae(tbl: pd.DataFrame, overall: float):
    """F29 — 조건별 MAE + 전체 평균 기준선 + n 병기."""
    t = tbl[tbl["구분"] != "계절"].copy()
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    vals = t["MAE(조건 충족)"].to_numpy()
    colors = [PALETTE_ADJACENT[1] if v > overall else COLOR_HERO for v in vals]
    ax.barh(t["구분"], vals, color=colors, height=0.62)
    ax.axvline(overall, color=INK_SOFT, lw=1.4, label=f"전체 평균 {overall:.2f} kW", zorder=1)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), fontsize=8, frameon=False)
    for i, (v, n) in enumerate(zip(vals, t["n(조건 충족)"])):
        ax.text(v + overall * 0.03, i, f"{v:.2f} (n={n})", va="center", fontsize=8, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=4)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (kW)", fontsize=9)
    ax.set_xlim(0, max(vals.max(), overall) * 1.40)
    ax.set_title("조건별 예측오차 (OOF, 표본수 병기)", fontsize=10, color=INK)
    fig.tight_layout()
    return fig, t


_fig, _src = plot_condition_mae(condition_tbl, condition_tbl.attrs["overall"])
save_fig(_fig, "F29", "조건별 MAE", "7.3", source_table=_src)

# %% [markdown]
# ### 7.4 피크 미탐지·오경보 분석
#
# - **목적**: TP/FN/FP/TN을 세고 **FN·FP가 집중되는 조건**을 찾는다.
# - **보고서 대응절**: 3.4 피크 미탐지와 오경보 분석
# - **산출물**: `ch3_confusion.csv`, F30
#
# 미탐지는 전력비 상승 위험을, 오경보는 불필요한 생산조정비용을 유발한다.
# 두 오류를 **분리해서** 관리해야 한다.

# %%
def confusion_analysis(o: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """혼동행렬 4구분과 각 구분의 주요 발생조건을 정리한다."""
    o = o.copy()
    o["kind"] = np.select(
        [
            (o["y_cls"] == 1) & (o["pred_cls"] == 1),
            (o["y_cls"] == 1) & (o["pred_cls"] == 0),
            (o["y_cls"] == 0) & (o["pred_cls"] == 1),
        ],
        ["정탐(TP)", "미탐지(FN)", "오경보(FP)"],
        default="정상판정(TN)",
    )

    def top_condition(sub: pd.DataFrame) -> str:
        """해당 구분에서 가장 두드러진 조건을 문장으로 만든다."""
        if len(sub) == 0:
            return "-"
        cands = {
            "08시 기동": float((sub["is_startup_08"] == 1).mean()),
            "13시 재가동": float((sub["is_restart_13"] == 1).mean()),
            "휴일 다음날": float((sub["prev_day_shutdown"] == 1).mean()),
            "주간(9~17시)": float((sub["is_daytime"] == 1).mean()),
            "고생산(상위25%)": float((sub["생산량"] > o["생산량"].quantile(0.75)).mean()),
        }
        base = {
            "08시 기동": float((o["is_startup_08"] == 1).mean()),
            "13시 재가동": float((o["is_restart_13"] == 1).mean()),
            "휴일 다음날": float((o["prev_day_shutdown"] == 1).mean()),
            "주간(9~17시)": float((o["is_daytime"] == 1).mean()),
            "고생산(상위25%)": float((o["생산량"] > o["생산량"].quantile(0.75)).mean()),
        }
        # 전체 대비 과대표집된 조건을 고른다
        lift = {k: (cands[k] / base[k] if base[k] > 0 else 0) for k in cands}
        best = max(lift, key=lift.get)
        return f"{best} ({cands[best]:.0%}, 전체 대비 {lift[best]:.1f}배)"

    rows = []
    for kind, meaning in [
        ("정탐(TP)", "실제 피크를 정상적으로 탐지"),
        ("미탐지(FN)", "실제 피크를 놓침"),
        ("오경보(FP)", "피크로 예측했으나 실제 정상"),
        ("정상판정(TN)", "정상구간을 정상으로 판정"),
    ]:
        sub = o[o["kind"] == kind]
        rows.append(
            {"구분": kind, "의미": meaning, "건수": len(sub), "주요 발생조건": top_condition(sub)}
        )
    tbl = pd.DataFrame(rows)

    fn = o[o["kind"] == "미탐지(FN)"]
    fp = o[o["kind"] == "오경보(FP)"]
    # FN 이 가장 집중된 시간대
    fn_hours = fn["시간"].value_counts()
    fn_top_hours = fn_hours.head(3).index.tolist() if len(fn_hours) else []
    fn_share = (
        round(float(fn_hours.head(3).sum() / len(fn) * 100), 1) if len(fn) else 0.0
    )
    fp_hours = fp["시간"].value_counts()
    info = {
        "FN_TOP_CONDITION": (
            f"{', '.join(f'{h}시' for h in fn_top_hours)} 시간대" if fn_top_hours else "-"
        ),
        "FN_TOP_SHARE_PCT": fn_share,
        "FP_TOP_CONDITION": (
            f"{', '.join(f'{h}시' for h in fp_hours.head(3).index)} 시간대" if len(fp_hours) else "-"
        ),
        "FN_N": len(fn), "FP_N": len(fp),
    }
    return tbl, info


confusion_tbl, fn_fp_info = confusion_analysis(OOF)
save_table(confusion_tbl, "ch3_confusion")
FN_TOP_CONDITION = fn_fp_info["FN_TOP_CONDITION"]
FN_TOP_SHARE_PCT = fn_fp_info["FN_TOP_SHARE_PCT"]
FP_TOP_CONDITION = fn_fp_info["FP_TOP_CONDITION"]
print("── 3.4절 미탐지·오경보 분석 (OOF) ──")
display(confusion_tbl)
print(f"\n  미탐지 집중 조건: {FN_TOP_CONDITION} — 전체 미탐지의 {FN_TOP_SHARE_PCT:.1f}%")
print(f"  오경보 집중 조건: {FP_TOP_CONDITION}")


def plot_fn_fp_by_hour(o: pd.DataFrame):
    """F30 — FN/FP 시간대 분포 (스택 막대)."""
    fn = o[(o["y_cls"] == 1) & (o["pred_cls"] == 0)]["시간"].value_counts().reindex(range(24), fill_value=0)
    fp = o[(o["y_cls"] == 0) & (o["pred_cls"] == 1)]["시간"].value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    ax.bar(range(24), fn.to_numpy(), color=PALETTE_ADJACENT[1], label="미탐지(FN)", width=0.7)
    ax.bar(range(24), fp.to_numpy(), bottom=fn.to_numpy(), color=COLOR_HERO,
           label="오경보(FP)", width=0.7)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시각", fontsize=9)
    ax.set_ylabel("건수", fontsize=9)
    ax.set_title("시간대별 미탐지·오경보 분포 (OOF)", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, pd.DataFrame({"FN": fn, "FP": fp})


_fig, _src = plot_fn_fp_by_hour(OOF)
save_fig(_fig, "F30", "FN FP 시간대 분포", "7.4", source_table=_src)

# %% [markdown]
# ### 7.5 대표 실패사례 분석
#
# - **목적**: 오차가 큰 사례 중 **운영상 의미가 있는** 3건을 골라 원인을 규명한다.
# - **보고서 대응절**: 3.5 대표 실패사례 분석
# - **산출물**: `ch3_failures.csv`, F33

# %%
def pick_failure_cases(o: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """오차 상위 사례 중 서로 다른 날짜에서 k건을 고른다."""
    ranked = o.sort_values("abs_err", ascending=False)
    picked, seen_days = [], set()
    for ts, r in ranked.iterrows():
        d = ts.normalize()
        if d in seen_days:
            continue
        seen_days.add(d)
        # 오차 방향과 관측 조건을 기술한다. 게이트 오분류·설비 원인은 별도 검증이 필요하다.
        direction = "과대예측" if r["pred_avg"] > r["y_avg"] else "과소예측"
        if r["is_shutdown"] == 1:
            cause = f"계획상 휴무일의 {direction} — 부하상태 배정과 참조 전력 점검 필요"
            fix = "ERP 휴무 계획과 1단계 상태 배정 대조; 강제 휴무 조건은 별도 검증"
        elif r["prev_day_shutdown"] == 1:
            cause = f"휴무 다음 가동일의 {direction} — 재가동 상태 점검 필요"
            fix = "휴무·재가동 계획과 실제 부하를 대조"
        elif abs(r["prod_chg_ratio"]) >= 0.5:
            cause = f"생산량 변화율 ±50% 이상 조건의 {direction}(인과관계 미확인)"
            fix = "ERP 계획 변경 이벤트를 별도 피처로 투입"
        elif r["is_startup_08"] == 1 or r["is_restart_13"] == 1:
            cause = f"08·13시 가동 전환시간의 {direction} — 설비별 원인은 미확인"
            fix = "설비별 기동 이력과 15분 부하를 수집해 원인 확인"
        else:
            cause = f"{direction} — 제공된 정보만으로 원인을 특정할 수 없음"
            fix = "설비별 전력센서·제품정보 추가"
        lag_values = {}
        for hours in (24, 48, 168):
            reference = ts - pd.Timedelta(hours=hours)
            value = df["y_avg"].get(reference, float("nan"))
            lag_values[f"{hours}시간 전 평균전력"] = round(float(value), 1)
        picked.append(
            {
                "사례": len(picked) + 1,
                "시점 및 조건": f"{ts:%Y-%m-%d %H시} (생산량 {r['생산량']:.0f}, 기온 {r['기온']:.1f}°C)",
                "실제값": round(float(r["y_avg"]), 1),
                "예측값": round(float(r["pred_avg"]), 1),
                "절대오차": round(float(r["abs_err"]), 1),
                "오차 방향": direction,
                **lag_values,
                "오류 원인": cause,
                "개선 방향": fix,
            }
        )
        if len(picked) == k:
            break
    return pd.DataFrame(picked)


failure_tbl = pick_failure_cases(OOF)
save_table(failure_tbl, "ch3_failures")
print("── 3.5절 대표 실패사례 ──")
display(failure_tbl)


def plot_failure_cases(o: pd.DataFrame, fails: pd.DataFrame):
    """F33 — 대표 실패사례 3건의 당일 시계열 (스몰멀티플)."""
    fig, axes = plt.subplots(1, len(fails), figsize=(11, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, (_, r) in zip(axes, fails.iterrows()):
        day = pd.Timestamp(r["시점 및 조건"].split(" ")[0])
        sub = o[o.index.normalize() == day]
        ax.plot(sub["시간"], sub["y_avg"], color=INK_SOFT, lw=1.8, marker="o", ms=3, label="실측")
        ax.plot(sub["시간"], sub["pred_avg"], color=COLOR_HERO, lw=1.8, marker="o", ms=3, label="예측")
        ax.set_title(f"사례 {int(r['사례'])} — {day:%m-%d}", fontsize=9.5, color=INK)
        ax.set_xlabel("시각", fontsize=8.5)
        rows.append(sub[["시간", "y_avg", "pred_avg"]].assign(사례=int(r["사례"])))
    axes[0].set_ylabel("평균전력 (kW)", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, pd.concat(rows) if rows else pd.DataFrame()


_fig, _src = plot_failure_cases(OOF, failure_tbl)
save_fig(_fig, "F33", "대표 실패사례", "7.5", source_table=_src)

# %% [markdown]
# ### 7.6 최대수요 피크 발생조건 도출 (깊이 3 규칙트리)
#
# - **목적**: 현장에서 바로 쓸 수 있는 **깊이 3 이하** 규칙으로 피크 조건을 단순화한다.
# - **보고서 대응절**: 3.6 최대수요 피크 발생조건 도출
# - **산출물**: `ch3_rules.csv`, F31
#
# **규칙은 분석 결과로 확인된 조건만 기재한다.** 상식적으로 예상한 조건을
# 결과처럼 적지 않는다. 도출된 고위험 규칙은 4장의 경보 기준·시나리오로 연결된다.

# %%
def extract_peak_rules(o: pd.DataFrame, max_depth: int = 3) -> tuple[pd.DataFrame, object, list]:
    """깊이 3 결정트리로 피크 발생 규칙 R1~R3을 뽑는다."""
    from sklearn.tree import DecisionTreeClassifier

    cols = ["시간", "생산량", "is_daytime", "is_shutdown", "prev_day_shutdown",
            "roll168_peak_max", "기온"]
    X = o[cols]
    y = o["y_cls"]
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=20, class_weight="balanced", random_state=SEED
    )
    tree.fit(X, y)

    t = tree.tree_
    rows = []

    def walk(node: int, conds: list):
        if t.children_left[node] == -1:  # leaf
            n = int(t.n_node_samples[node])
            counts = t.value[node][0]
            # class_weight='balanced' 라 value 는 가중치 반영값 → 원표본으로 다시 센다
            rows.append({"conds": list(conds), "n": n, "w_pos": float(counts[1] / counts.sum())})
            return
        f = cols[t.feature[node]]
        th = t.threshold[node]
        walk(t.children_left[node], conds + [(f, "<=", th)])
        walk(t.children_right[node], conds + [(f, ">", th)])

    walk(0, [])

    # 실제 표본으로 각 규칙의 피크 확률·적용 시간 비율을 다시 계산한다
    out = []
    for r in rows:
        m = pd.Series(True, index=o.index)
        parts = []
        for f, op, th in r["conds"]:
            m &= (o[f] <= th) if op == "<=" else (o[f] > th)
            label = FEATURE_LABELS.get(f, f)
            if f in ("is_daytime", "is_shutdown", "prev_day_shutdown"):
                parts.append(f"{label}={'아님' if op == '<=' else '해당'}")
            else:
                parts.append(f"{label} {op} {th:.1f}")
        if m.sum() < 20:
            continue
        out.append(
            {
                "피크 발생조건": " ∧ ".join(parts),
                "피크 확률": round(float(o.loc[m, "y_cls"].mean()), 4),
                "적용되는 시간 비율": round(float(m.mean()) * 100, 2),
                "n": int(m.sum()),
            }
        )
    rules = (
        pd.DataFrame(out)
        .sort_values("피크 확률", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )
    rules.insert(0, "규칙", [f"R{i + 1}" for i in range(len(rules))])
    return rules, tree, cols


rules_tbl, peak_tree, rule_cols = extract_peak_rules(OOF)
save_table(rules_tbl, "ch3_rules")
print("── 3.6절 피크 발생조건 규칙 (OOF, 깊이 3) ──")
display(rules_tbl)


def plot_rule_tree(tree, cols):
    """F31 — 실제 관측 피크 수를 병기한 깊이 3 규칙 트리."""
    from tools.figure_layout import draw_rule_tree

    fig = draw_rule_tree(tree, cols, OOF, FEATURE_LABELS, plt)
    return fig, rules_tbl


_fig, _src = plot_rule_tree(peak_tree, rule_cols)
save_fig(_fig, "F31", "깊이3 규칙 트리", "7.6", source_table=_src)

print("\n── 7장 보고서 채움 변수 ──")
for _k, _v in [
    ("TOP_FEATURES", TOP_FEATURES), ("WORST_CONDITIONS", WORST_CONDITIONS),
    ("WORST_CONDITION_NAME", WORST_CONDITION_NAME),
    ("WORST_COND_EXCESS_PCT", WORST_COND_EXCESS_PCT),
    ("BEST_CONDITION_NAME", BEST_CONDITION_NAME), ("TOP_INTERACTION", TOP_INTERACTION),
    ("FN_TOP_CONDITION", FN_TOP_CONDITION), ("FN_TOP_SHARE_PCT", FN_TOP_SHARE_PCT),
    ("FP_TOP_CONDITION", FP_TOP_CONDITION),
]:
    print(f"  {_k} = {_v}")
