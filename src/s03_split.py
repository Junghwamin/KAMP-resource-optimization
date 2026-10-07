# %% tags=["nb-strip"]
from s00_env import (  # noqa: F401
    COLOR_HERO,
    COLOR_MUTED,
    INK,
    INK_SOFT,
    PALETTE_ADJACENT,
    display,
    save_fig,
    save_table,
)
from s01_diagnose import THETA, TEST_START, TEST_END, df, operating_calendar, profile_hashes  # noqa: F401
from s02_features import FEATURE_COLS, FEATURE_LABELS, feat  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 3. 분할 설계 및 평가지표
#
# 보고서 **1.6절·2.1절**의 근거를 생성한다.
#
# 이 장에서 **3~7장이 공유하는 계약**이 확정된다. 이후 어떤 장도 여기서 정한
# 분할·임계값·지표 정의를 다시 만들지 않는다.

# %% [markdown]
# ### 3.1 시간순 분할과 데이터 조건 2축
#
# - **목적**: 테스트 구간을 고정하고, 복제일 처리에 대한 두 데이터 조건을 정의한다.
# - **보고서 대응절**: 1.6 학습·검증 데이터 구성, 2.1 실험 설계
# - **산출물**: `ch3_data_conditions.csv`
#
# **테스트 구간**: 2021-09-01 00:00 ~ 09-14 23:00, **336시간 고정** (가이드북 비교 가능성).
#
# **데이터 조건 2축** — 1.2절에서 확인한 복제 115일(44.7%)을 어떻게 다룰지는
# 하나로 정할 수 없는 판단이므로, **두 조건 모두 본실험**으로 운용한다.
#
# | 조건 | 정의 | 일수 |
# |---|---|---|
# | **D1** | 복제 유지 (데이터셋 원형) | 257일 |
# | **D2** | 복제 제거 (각 중복그룹에서 1일만 잔류) | 142일 |
#
# 최종모델은 **D1 기준으로 선정**하고, **D2에서도 순위가 보존되는지**를
# 강건성 근거로 2.9절에 추가한다. D2에서는 fold별 실제 표본 수가 줄어들므로
# 모든 성능표에 **표본수 n을 병기**한다.

# %%
DATA_CONDITIONS = ("D1", "D2")


def build_dedup_day_set(hashes: pd.Series) -> set:
    """D2 조건에서 **잔류시킬 날짜** 집합을 만든다.

    각 프로파일 중복그룹에서 **가장 이른 날 1일만** 남긴다(임의 선택을 피해
    시간순으로 결정론적으로 고른다).

    Returns
    -------
    set
        잔류 날짜(정수 YYYYMMDD) 집합. 142일.
    """
    keep = set()
    seen = set()
    for date in sorted(hashes.index):
        h = hashes.loc[date]
        if h not in seen:
            seen.add(h)
            keep.add(int(date))
    return keep


DEDUP_KEEP_DAYS = build_dedup_day_set(profile_hashes)


def condition_mask(d: pd.DataFrame, cond: str) -> pd.Series:
    """데이터 조건에 따라 **사용할 행**의 불리언 마스크를 만든다.

    Parameters
    ----------
    cond : {'D1', 'D2'}
        D1 = 복제 유지(전체), D2 = 복제 제거(그룹당 1일).
    """
    if cond == "D1":
        return pd.Series(True, index=d.index)
    if cond == "D2":
        ymd = d.index.strftime("%Y%m%d").astype(int)
        return pd.Series(np.isin(ymd, list(DEDUP_KEEP_DAYS)), index=d.index)
    raise ValueError(f"알 수 없는 데이터 조건: {cond}")


cond_tbl = pd.DataFrame(
    [
        {
            "조건": c,
            "정의": "복제 유지 (원형)" if c == "D1" else "복제 제거 (그룹당 1일)",
            "일수": int(condition_mask(df, c).groupby(df.index.normalize()).any().sum()),
            "시간수": int(condition_mask(df, c).sum()),
        }
        for c in DATA_CONDITIONS
    ]
)
save_table(cond_tbl, "ch3_data_conditions")
print("── 데이터 조건 2축 ──")
display(cond_tbl)

# %% [markdown]
# ### 3.2 Rolling-origin 교차검증 (+ 복제 오염 검사)
#
# - **목적**: 확장 학습창 + 24시간 갭으로 fold를 구성하고, **검증일 프로파일이
#   학습구간에 이미 존재하지 않음**을 검사한다.
# - **보고서 대응절**: 2.1 실험 설계
# - **산출물**: `ch3_folds.csv`, `ch3_contamination.csv`, F14·F15 (게이트 3)
#
# **fold1을 폐기하는 이유.** 복제 오염을 실측하면 fold1 검증 14일 중 **7일(50%)** 의
# 전력 프로파일이 이미 학습구간에 존재한다. 이 fold의 성능은 일반화가 아니라
# 암기를 측정한다. fold2~5는 오염 0%다.
#
# **분석용 피크 정의는 공통 θ로 고정한다.** fold별 학습구간 q95는 서로 다르므로
# 같은 절대 전력수준에서 위험을 비교하기 위한 선택이다. θ는 테스트 이전 전체기간에서
# 정해 과거 OOF보다 뒤의 분포도 반영한다. 사전에 외부에서 정한 운영 임계값은 아니다.

# %%
# (fold번호, 학습종료일, 검증시작일, 검증종료일) — 검증 14일, 학습종료↔검증시작 24시간 갭
FOLD_SPEC = [
    (1, "2021-06-21", "2021-06-23", "2021-07-06"),
    (2, "2021-07-05", "2021-07-07", "2021-07-20"),
    (3, "2021-07-19", "2021-07-21", "2021-08-03"),
    (4, "2021-08-02", "2021-08-04", "2021-08-17"),
    (5, "2021-08-16", "2021-08-18", "2021-08-31"),
]
# fold1은 복제 오염 50%로 폐기 (D1·D2 공통)
DROPPED_FOLDS = (1,)
ACTIVE_FOLDS = tuple(f for f, *_ in FOLD_SPEC if f not in DROPPED_FOLDS)


def fold_slices(d: pd.DataFrame, fold: int) -> tuple[pd.Series, pd.Series]:
    """fold 번호로 (학습 마스크, 검증 마스크)를 만든다.

    학습은 시작부터 `학습종료일 23:00` 까지 확장(expanding)하고,
    검증은 `검증시작일 00:00` 부터 14일이다. 사이에 하루(24시간)가 비어 있다.
    """
    spec = {f: (tr, vs, ve) for f, tr, vs, ve in FOLD_SPEC}[fold]
    train_end, val_start, val_end = (pd.Timestamp(x) for x in spec)
    train = d.index <= train_end + pd.Timedelta(hours=23)
    val = (d.index >= val_start) & (d.index <= val_end + pd.Timedelta(hours=23))
    return pd.Series(train, index=d.index), pd.Series(val, index=d.index)


def check_profile_contamination(d: pd.DataFrame, hashes: pd.Series) -> pd.DataFrame:
    """각 fold에서 **검증일 프로파일이 학습구간에 이미 존재하는 비율**을 센다.

    행 단위 중복이 0건이라 일반적인 누수 검사로는 절대 드러나지 않는다.
    """
    rows = []
    for fold, *_ in FOLD_SPEC:
        tr, va = fold_slices(d, fold)
        tr_days = pd.Index(d.index[tr].normalize().unique()).strftime("%Y%m%d").astype(int)
        va_days = pd.Index(d.index[va].normalize().unique()).strftime("%Y%m%d").astype(int)
        tr_hashes = {hashes.loc[x] for x in tr_days if x in hashes.index}
        contaminated = [x for x in va_days if hashes.get(x) in tr_hashes]
        rows.append(
            {
                "fold": fold,
                "검증일수": len(va_days),
                "오염일수": len(contaminated),
                "오염률": round(len(contaminated) / len(va_days) * 100, 1),
                "상태": "폐기" if fold in DROPPED_FOLDS else "사용",
            }
        )
    # 테스트 구간도 같은 기준으로 확인
    test_days = pd.Index(
        d.index[(d.index >= TEST_START)].normalize().unique()
    ).strftime("%Y%m%d").astype(int)
    train_all = pd.Index(
        d.index[d.index < TEST_START].normalize().unique()
    ).strftime("%Y%m%d").astype(int)
    tr_hashes = {hashes.loc[x] for x in train_all if x in hashes.index}
    cont = [x for x in test_days if hashes.get(x) in tr_hashes]
    rows.append(
        {
            "fold": "TEST",
            "검증일수": len(test_days),
            "오염일수": len(cont),
            "오염률": round(len(cont) / len(test_days) * 100, 1),
            "상태": "사용",
        }
    )
    return pd.DataFrame(rows)


def fold_summary(d: pd.DataFrame, cal: pd.DataFrame, theta: float) -> pd.DataFrame:
    """fold별 표본수·휴무비중·양성률·학습구간 q95를 정리한다.

    휴무 비중을 병기하는 이유: fold3 29%·fold4 36%가 휴무일이라, 이를 모르면
    **휴무 비중 차이로 생긴 성능 차이를 모델 편차로 오독**하게 된다.
    """
    rows = []
    for fold, tr_end, vs, ve in FOLD_SPEC:
        tr, va = fold_slices(d, fold)
        va_days = d.index[va].normalize().unique()
        shut_ratio = float(cal.reindex(va_days)["is_shutdown"].mean())
        rows.append(
            {
                "fold": fold,
                "학습종료": tr_end,
                "검증구간": f"{vs} ~ {ve}",
                "학습 n": int(tr.sum()),
                "검증 n": int(va.sum()),
                "휴무비중": f"{shut_ratio:.0%}",
                "검증 양성수": int((d.loc[va, "y_peak"] >= theta).sum()),
                "검증 양성률": f"{float((d.loc[va, 'y_peak'] >= theta).mean()):.2%}",
                "학습구간 q95": round(float(d.loc[tr, "y_peak"].quantile(0.95)), 1),
                "상태": "폐기(오염 50%)" if fold in DROPPED_FOLDS else "사용",
            }
        )
    return pd.DataFrame(rows)


contamination = check_profile_contamination(df, profile_hashes)
folds_tbl = fold_summary(df, operating_calendar, THETA)
save_table(contamination, "ch3_contamination")
save_table(folds_tbl, "ch3_folds")

print("── 복제 오염 검사 ──")
display(contamination)
print("\n── fold 구성 ──")
display(folds_tbl)
print(
    f"\n  fold별 학습구간 q95 = {folds_tbl['학습구간 q95'].tolist()}"
    f"  →  비교용 공통 피크 기준 θ={THETA:g}을 적용한다(테스트 이전 전체기간 기준)."
)


def gate_chapter3(cont: pd.DataFrame) -> pd.DataFrame:
    """게이트 3 — 사용 fold의 복제 오염이 0이어야 한다."""
    active = cont[cont["fold"].isin(ACTIVE_FOLDS)]
    test_row = cont[cont["fold"] == "TEST"]
    checks = [
        ("사용 fold 복제 오염 0%", bool((active["오염일수"] == 0).all())),
        ("테스트 구간 복제 오염 0%", bool((test_row["오염일수"] == 0).all())),
        ("fold1 오염이 실제로 높음(폐기 근거)", int(cont.loc[cont["fold"] == 1, "오염일수"].iloc[0]) >= 7),
        ("사용 fold 수 4개", len(ACTIVE_FOLDS) == 4),
    ]
    out = pd.DataFrame(checks, columns=["점검", "통과"])
    failed = out[~out["통과"]]
    if len(failed):
        raise AssertionError(f"3장 게이트 실패: {failed['점검'].tolist()}")
    return out


gate3 = gate_chapter3(contamination)
save_table(gate3, "gate3_split")
print("\n── 게이트 3 통과 ──")
display(gate3)

# %% [markdown]
# #### 그림 F14 · F15

# %%
def plot_fold_layout(folds: pd.DataFrame):
    """Rolling-origin fold 배치도 (학습 / 24h갭 / 검증)."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (_, r) in enumerate(folds.iterrows()):
        y = len(folds) - i
        tr_end = pd.Timestamp(r["학습종료"])
        vs, ve = (pd.Timestamp(x) for x in r["검증구간"].split(" ~ "))
        dropped = r["상태"].startswith("폐기")
        c_tr = COLOR_MUTED if dropped else COLOR_HERO
        c_va = COLOR_MUTED if dropped else PALETTE_ADJACENT[1]
        ax.barh(y, (tr_end - df.index.min()).days, left=df.index.min(), height=0.52, color=c_tr)
        ax.barh(y, (ve - vs).days + 1, left=vs, height=0.52, color=c_va)
        ax.text(
            ve + pd.Timedelta(days=3), y,
            f"{r['상태']} · 휴무 {r['휴무비중']} · 양성 {r['검증 양성수']}건",
            va="center", fontsize=8, color=INK_SOFT,
        )
    ax.barh(0, 14, left=TEST_START, height=0.52, color=PALETTE_ADJACENT[2])
    ax.text(TEST_END + pd.Timedelta(days=3), 0, "TEST 336h", va="center", fontsize=8, color=INK_SOFT)
    ax.set_yticks([len(folds) - i for i in range(len(folds))] + [0])
    ax.set_yticklabels([f"fold{int(r['fold'])}" for _, r in folds.iterrows()] + ["TEST"], fontsize=9)
    ax.set_ylim(-.6, len(folds) + .6)
    ax.set_xlim(df.index.min(), TEST_END + pd.Timedelta(days=80))
    ax.set_title("Rolling-origin 분할 (학습=파랑 / 검증=주황 / 회색=폐기)", fontsize=10, color=INK)
    fig.tight_layout()
    return fig, folds


_fig, _src = plot_fold_layout(folds_tbl)
save_fig(_fig, "F14", "Rolling-origin fold 배치도", "3.2", source_table=_src)


def plot_fold_positive_rate(folds: pd.DataFrame):
    """fold별 양성률 — PR-AUC 무기술선이 fold마다 다름을 보인다."""
    use = folds[folds["fold"].isin(ACTIVE_FOLDS)]
    rates = [float(x.strip("%")) for x in use["검증 양성률"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bars = ax.bar([f"fold{int(f)}" for f in use["fold"]], rates, color=COLOR_HERO, width=0.6)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}%", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("검증구간 양성률 (%)", fontsize=9)
    ax.set_title("fold별 양성률 — PR-AUC 무기술선의 차이", fontsize=10, color=INK)
    fig.tight_layout()
    return fig, use[["fold", "검증 양성수", "검증 양성률"]]


_fig, _src = plot_fold_positive_rate(folds_tbl)
save_fig(_fig, "F15", "fold별 양성률", "3.2", source_table=_src)

# %% [markdown]
# ### 3.3 평가지표 정의
#
# - **목적**: Task A(회귀)·Task B(분류) 지표와 통계적 유의성 검정을 함수로 고정한다.
# - **보고서 대응절**: 2.1 실험 설계, 2.6~2.7 성능표
# - **산출물**: 지표 함수군
#
# #### Task B 주지표 재정의 (필수)
#
# 달력규칙 한 줄(`평일 ∧ 08≤h≤18`)이 테스트에서 **Recall 1.000** 을 낸다.
# **어떤 ML 모델도 Recall 로는 이길 수 없다.** 따라서 주지표를
# **"Recall 을 유지하면서 오경보(FP)를 최소화"**, 즉 **F1 / PR-AUC** 로 재정의하고,
# 제안모델의 기여를 **"동일 Recall 에서 오경보 N건 감소"** 로 정량화한다.
#
# #### 모델 비교의 주 근거 = fold 분포 (테스트 아님)
#
# 테스트 피크 28건은 8개 날짜·16개 연속 블록에만 분포해 **유효 표본이 8~16** 에 가깝다.
# Recall 0.357 기준 Clopper-Pearson 95% CI 폭이 **0.37** 이라 두 모델을 구분할 수 없다.
# OOF는 HPO·변수선택·임계값 조정에도 재사용하는 내부 검증이며 독립 성능 추정치가 아니다.
# 테스트 336시간을 별도로 제시하고, 아래 fold_matrix에서 실제 평가 표본수를 확인한다.

# %%
from scipy import stats as _st  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_fscore_support  # noqa: E402


def mae(y, p):
    """평균절대오차 — Task A 주지표. 전력 단위(kW)로 직접 해석된다."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    """평균제곱근오차 — 큰 오차에 더 큰 벌점."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def smape(y, p):
    """대칭 평균절대백분율오차.

    분모는 `(|y|+|ŷ|)/2` 를 쓰고, 분모가 0인 시각은 제외한다
    (비가동 시간대에 0이 되어 정의되지 않는 것을 방지).
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = (np.abs(y) + np.abs(p)) / 2
    ok = denom > 0
    return float(np.mean(np.abs(y[ok] - p[ok]) / denom[ok]) * 100)


def peak_mae(y, p, theta: float, y_peak=None):
    """실제 15분 최대전력 ``y_peak >= θ`` 시간의 예측 MAE.

    이전 호출과의 호환을 위해 y_peak 생략 시 y의 고부하 구간을 사용한다.
    신규 비교에는 반드시 y_peak를 전달하여 피크 탐지와 같은 시간을 평가한다.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    peak = y if y_peak is None else np.asarray(y_peak, float)
    if y.shape != p.shape or y.shape != peak.shape:
        raise ValueError("정답·예측·피크 정답의 크기가 같아야 합니다.")
    m = peak >= theta
    return float(np.mean(np.abs(y[m] - p[m]))) if m.any() else float("nan")


def regression_metrics(y, p, theta: float, y_peak=None, pred_peak=None) -> dict:
    """평균전력 지표 및 동일한 실제 피크시간의 두 타깃 MAE를 계산한다.

    Peak-MAE는 평균전력 예측, Peak15-MAE는 15분 최대전력 예측의 오차다.
    두 지표는 y_peak >= theta인 동일한 Peak-n개 시간을 사용한다.
    종전 y_avg >= theta 지표는 High-load-MAE/n으로 별도 보존한다.
    """
    peak = np.asarray(y if y_peak is None else y_peak, float)
    return {
        "MAE": mae(y, p),
        "RMSE": rmse(y, p),
        "sMAPE": smape(y, p),
        "Peak-MAE": peak_mae(y, p, theta, y_peak),
        "Peak15-MAE": (
            peak_mae(peak, pred_peak, theta, peak)
            if y_peak is not None and pred_peak is not None else float("nan")
        ),
        "Peak-n": int((peak >= theta).sum()),
        "High-load-MAE": peak_mae(y, p, theta),
        "High-load-n": int((np.asarray(y) >= theta).sum()),
        "Peak-mask-source": "y_peak" if y_peak is not None else "y_avg (legacy)",
    }


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson 정확 이항 신뢰구간.

    양성 표본이 28건뿐이라 정규근사(Wald)는 부정확하다. 정확 구간을 쓴다.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(_st.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_st.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    """Task B 지표. PR-AUC 는 연속 스코어가 있을 때만 계산한다.

    `lift` = PR-AUC / 양성률. fold별 양성률이 5.95~17.26%(2.9배)로 달라
    무기술선이 fold마다 다르므로, 정규화 값을 함께 봐야 비교가 성립한다.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    base_rate = float(y_true.mean()) if len(y_true) else float("nan")
    out = {
        "Precision": float(pr), "Recall": float(rc), "F1": float(f1),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "양성률": base_rate,
    }
    lo, hi = clopper_pearson(tp, tp + fn)
    out["Recall_CI_low"], out["Recall_CI_high"] = lo, hi
    if y_score is not None and len(np.unique(y_true)) > 1:
        ap = float(average_precision_score(y_true, np.asarray(y_score, float)))
        out["PR-AUC"] = ap
        out["lift"] = ap / base_rate if base_rate > 0 else float("nan")
    else:
        out["PR-AUC"] = float("nan")
        out["lift"] = float("nan")
    return out


def paired_bootstrap_mae(
    y, p_a, p_b, index=None, n_boot: int = 2000, seed: int = 42
) -> dict:
    """두 모델 MAE 차이(A−B)의 일 단위 블록 부트스트랩 95%/90% 구간.

    시간별 오차는 하루 안에서 강하게 상관되므로 시간 단위로 재표집하면
    구간이 과도하게 좁아진다. **날짜 블록**으로 재표집한다.

    Returns
    -------
    dict
        양의 차이는 B의 개선이다. 95% 구간은 주분석(5%), 90%는 사후
        탐색(10%)이며 판정 기준을 바꾸어 주분석 실패를 대체하지 않는다.
        p_value는 비중심화 부트스트랩 꼬리비율의 양측 근사이며 정확검정이 아니다.
    """
    y = np.asarray(y, float)
    ea = np.abs(y - np.asarray(p_a, float))
    eb = np.abs(y - np.asarray(p_b, float))
    if y.ndim != 1 or not len(y) or ea.shape != y.shape or eb.shape != y.shape:
        raise ValueError("비교할 정답과 예측은 길이가 같은 비어 있지 않은 1차원 배열이어야 합니다.")
    if not np.isfinite(ea).all() or not np.isfinite(eb).all() or n_boot < 1:
        raise ValueError("부트스트랩에는 유한한 오차와 1회 이상의 재표집이 필요합니다.")
    diff_obs = float(ea.mean() - eb.mean())

    if index is None:
        blocks = [np.arange(len(y))]
    else:
        days = pd.Index(index).normalize()
        if len(days) != len(y) or days.hasnans:
            raise ValueError("일자 인덱스는 정답과 길이가 같고 결측이 없어야 합니다.")
        blocks = [np.where(days == d)[0] for d in days.unique()]

    rng = np.random.default_rng(seed)
    stat = np.empty(n_boot)
    nb = len(blocks)
    for i in range(n_boot):
        pick = rng.integers(0, nb, nb)
        idx = np.concatenate([blocks[j] for j in pick])
        stat[i] = ea[idx].mean() - eb[idx].mean()
    lo, hi = np.percentile(stat, [2.5, 97.5])
    lo90, hi90 = np.percentile(stat, [5.0, 95.0])
    # 비중심화 재표집 분포의 양측 꼬리비율 근사(정확한 귀무분포 p값이 아님).
    p = 2 * min((stat >= 0).mean(), (stat <= 0).mean())
    return {
        "diff": diff_obs, "ci_low": float(lo), "ci_high": float(hi),
        "ci90_low": float(lo90), "ci90_high": float(hi90),
        "p_value": float(min(1.0, p)), "p_method": "양측 부트스트랩 꼬리비율 근사",
        "유의": bool(lo > 0 or hi < 0),
        "유의_5pct": bool(lo > 0 or hi < 0),
        "유의_10pct": bool(lo90 > 0 or hi90 < 0),
        "개선_5pct": bool(lo > 0), "개선_10pct": bool(lo90 > 0),
    }


def tune_tau(y_peak_true, y_peak_pred, theta: float) -> float:
    """검증구간에서 **판정 임계값 τ** 를 F1 최대화로 고른다.

    회귀 예측값은 평균으로 수축하므로 θ를 그대로 적용하면 구조적으로 불리하다.
    실제로 동일 예측값에 θ=187 고정 시 Recall 0.143 / F1 0.235,
    τ=182 적용 시 Recall 0.500 / F1 0.538 로 3.5배 차이가 난다.
    **회귀 기반 판정 행은 전부 이 절차로 τ를 맞춘다.**
    """
    y_true = (np.asarray(y_peak_true) >= theta).astype(int)
    pred = np.asarray(y_peak_pred, float)
    if y_true.sum() == 0:
        return float(theta)
    cands = np.unique(np.round(np.quantile(pred, np.linspace(0.50, 0.999, 120)), 2))
    best_tau, best_f1 = float(theta), -1.0
    for t in cands:
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, (pred >= t).astype(int), average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_tau = float(f1), float(t)
    return best_tau


# %% [markdown]
# ### 3.4 달력규칙 기준선 — Task B의 진짜 하한
#
# - **목적**: `평일 ∧ 08≤h≤18` 한 줄 규칙의 성능을 측정해 **2.7절 표의 기준 행**으로 삼는다.
# - **보고서 대응절**: 2.7 피크 위험 탐지 성능
# - **산출물**: `ch3_calendar_rule.csv`
#
# 이 규칙이 테스트에서 **Recall 1.000**(TP 28, FN 0, FP 82, TN 226)을 낸다.
# 제안모델의 기여는 Recall 을 올리는 것이 아니라 **같은 Recall 에서 FP 82건을 줄이는 것**이다.
# 이 사실을 표에 명시하지 않으면 "Recall 0.9 달성"이 실제보다 대단해 보이는 착시가 생긴다.

# %%
def calendar_rule_predict(d: pd.DataFrame) -> np.ndarray:
    """달력규칙: 평일이고 08~18시면 피크 위험으로 판정."""
    idx = d.index
    return ((idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour <= 18)).astype(int)


_test = df.loc[TEST_START:TEST_END]
_cal_pred = calendar_rule_predict(_test)
calendar_rule_metrics = classification_metrics(_test["y_cls"], _cal_pred)
CALENDAR_RULE_FP = calendar_rule_metrics["FP"]

save_table(pd.DataFrame([calendar_rule_metrics]), "ch3_calendar_rule")
print("── 달력규칙 기준선 (테스트 336h) ──")
print(
    f"  Recall {calendar_rule_metrics['Recall']:.3f} | Precision {calendar_rule_metrics['Precision']:.3f} "
    f"| F1 {calendar_rule_metrics['F1']:.3f}"
)
print(
    f"  TP {calendar_rule_metrics['TP']} / FN {calendar_rule_metrics['FN']} / "
    f"FP {calendar_rule_metrics['FP']} / TN {calendar_rule_metrics['TN']}"
)
print("  → 제안모델은 Recall·미탐지와 오경보의 교환관계를 함께 비교한다.")

# 테스트 피크의 유효 표본 진단 — 불확실성 서술의 근거
_pk = _test.index[_test["y_cls"] == 1]
_blocks = int((pd.Series(_pk).diff() != pd.Timedelta(hours=1)).sum())
_lo, _hi = clopper_pearson(10, 28)
print(
    f"\n  테스트 피크 {len(_pk)}건 = {len(set(_pk.normalize()))}개 날짜 · {_blocks}개 연속블록"
)
print(
    f"  Recall 0.357 기준 Clopper-Pearson 95% CI 폭 = {_hi - _lo:.2f}"
    "  → 두 모델을 테스트만으로는 구분할 수 없다"
)

# %% [markdown]
# ### 3.5 fold 데이터 준비 헬퍼
#
# - **목적**: 5~7장이 공통으로 쓰는 (X, y) 추출 함수를 확정한다.
# - **보고서 대응절**: 2.1, 2.3
# - **산출물**: `get_fold_data()`, `get_test_data()`
#
# **평가·학습 제외 규칙** (1장에서 정의한 마스크 적용)
#
# | 마스크 | 학습 | 평가 |
# |---|---|---|
# | `is_warmup` (168행) | 제외 | 제외 |
# | `is_outage` (17행, 센서 정지) | 제외 | **제외** |
# | `is_erp_missing` (48행, ERP 결측) | **제외** | **제외** |

# %%
def usable_mask(d: pd.DataFrame, cond: str = "D1") -> pd.Series:
    """학습·평가에 사용 가능한 행 마스크.

    워밍업·계측정지·ERP결측을 모두 제외하고 데이터 조건을 적용한다.
    """
    ok = (~d["is_warmup"]) & (~d["is_outage"]) & (~d["is_erp_missing"])
    return ok & condition_mask(d, cond)


def get_fold_data(fold: int, cond: str = "D1", features=None, frame=None):
    """fold의 (X_train, y_train, X_val, y_val, val_index)를 돌려준다.

    Parameters
    ----------
    fold : int
        fold 번호(2~5).
    cond : {'D1','D2'}
        데이터 조건. **3장 이후 모든 함수가 이 인자를 받는다.**
    features : list[str], optional
        사용할 피처. 기본은 전체(ablation에서 부분집합을 넘긴다).
    frame : pandas.DataFrame, optional
        피처 프레임. 기본은 `feat`. 6.6절이 피처 값을 바꾼 프레임(휴무 인지형 지연변수,
        잡음 변수 추가)을 넘긴다. 행·마스크는 `feat` 와 같아야 한다.
    """
    d = feat if frame is None else frame
    cols = list(features) if features is not None else FEATURE_COLS
    tr, va = fold_slices(d, fold)
    ok = usable_mask(d, cond)
    tr, va = tr & ok, va & ok
    ytr = d.loc[tr, ["y_avg", "y_peak", "y_cls"]]
    yva = d.loc[va, ["y_avg", "y_peak", "y_cls"]]
    return d.loc[tr, cols], ytr, d.loc[va, cols], yva, d.index[va]


def get_test_data(cond: str = "D1", features=None):
    """(X_train_full, y_train_full, X_test, y_test, test_index).

    학습은 테스트 시작 이전 전체를 확장해 쓴다. 테스트 336행은
    조건·마스크와 무관하게 **전부 유지**한다(제출 파일이 336행이어야 하므로).
    """
    cols = list(features) if features is not None else FEATURE_COLS
    ok = usable_mask(feat, cond)
    tr = (feat.index < TEST_START) & ok
    te = (feat.index >= TEST_START) & (feat.index <= TEST_END)
    ytr = feat.loc[tr, ["y_avg", "y_peak", "y_cls"]]
    yte = feat.loc[te, ["y_avg", "y_peak", "y_cls"]]
    return feat.loc[tr, cols], ytr, feat.loc[te, cols], yte, feat.index[te]


def eval_mask_for(index) -> np.ndarray:
    """지표 계산에서 제외할 행(계측정지·ERP결측)을 걸러내는 마스크."""
    sub = feat.loc[index]
    return (~sub["is_outage"] & ~sub["is_erp_missing"]).to_numpy()


_sizes = []
for _c in DATA_CONDITIONS:
    for _f in ACTIVE_FOLDS:
        _Xtr, _ytr, _Xva, _yva, _ = get_fold_data(_f, _c)
        _sizes.append(
            {"조건": _c, "fold": _f, "학습 n": len(_Xtr), "검증 n": len(_Xva),
             "검증 양성": int(_yva["y_cls"].sum())}
        )
fold_matrix = pd.DataFrame(_sizes)
save_table(fold_matrix, "fold_matrix")
print("── fold × 데이터조건 표본수 ──")
display(fold_matrix.pivot(index="fold", columns="조건", values=["학습 n", "검증 n", "검증 양성"]))

_Xtr_t, _ytr_t, _Xte_t, _yte_t, _ = get_test_data("D1")
print(f"\n  테스트 학습 n = {len(_Xtr_t)} / 테스트 n = {len(_Xte_t)} (336 고정)")
print(f"  OOF 양성 합계 (fold2~5, D1) = {fold_matrix[fold_matrix['조건'] == 'D1']['검증 양성'].sum()}건")
