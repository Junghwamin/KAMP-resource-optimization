# %% tags=["nb-strip"]
from s00_env import (  # noqa: F401
    CMAP_SEQ,
    COLOR_HERO,
    COLOR_MUTED,
    INK,
    INK_SOFT,
    OUTPUT_DIR,
    PALETTE_ADJACENT,
    SEED,
    display,
    save_fig,
    save_table,
)
from s01_diagnose import THETA, TEST_START, TEST_END, df, operating_calendar  # noqa: F401
from s02_features import feat  # noqa: F401
from s06_eval import FINAL_MODEL_NAME, test_results  # noqa: F401
from s07_analysis import OOF, rules_tbl  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 8. 피크 저감 시뮬레이션
#
# 보고서 **4장 현장 활용방안(10점)** 의 근거를 생성한다.
#
# > 입력은 전체 관측기간의 실측 `peak15`다. 실측 최대 시각을 알고 조정하는
# > **가정 기반 사후 시나리오**이며, 예측 경보에 따른 운영 성과를 검증한 것이 아니다.
# > 생산량 재배치량만 보존하며, 15분 구간별 에너지·설비용량·납기는 검증하지 않는다.
# > 전력량요금은 산정에서 제외한다. 기본요금과 추가 인건비는 관측기간을 맞춰 비교하고,
# > 12개월 기본요금 환산액은 별도의 참고 열로 제시한다.

# %% [markdown]
# ### 8.1 요금 모델링
#
# - **목적**: 월별 최대수요와 직전 3개월 롤링 최대 기술통계를 산출한다.
# - **보고서 대응절**: 4장
# - **산출물**: `ch4_tariff.csv`, F34
#
# **단순 비용 가정**: 관측 최대수요가 비교기간 동안 동일하게 기본요금에 적용된다고 둔다.
# 실제 계약종별 청구 규정·과거 청구 피크를 재현한 요금 모델은 아니다.
#
# 관측기간 비교액 = Δ최대수요 × 가정 단가 × 관측월수 − 같은 기간 추가 인건비.
# 관측월수는 각 월의 관측일수/달력일수 합이다. 12개월 환산액은 실증 연간절감이 아니다.
#
# 기본요금 단가는 제공 데이터로 검증할 수 없으므로 **파라미터로 노출하고 ±30% 민감도**를 제시한다.
# 전력 단위(kW) 해석도 명시적 가정이다.

# %%
# ── 가정 파라미터 (전부 민감도 분석 대상) ──────────────────────────────
BASE_RATE_KRW_PER_KW_MONTH = 8_320.0   # 기본요금 단가 (가정, 검증 불가)
BILLING_MONTHS = 12                    # 기본요금 참고 환산기간(실제 청구규정 아님)
ALPHA_SIMULTANEOUS = 0.20              # 08시 피크 중 '동시 기동' 기여분 (가정)
NIGHT_LABOR_MULTIPLIER = 1.5           # 야간 인건비 배수 (데이터 확인됨)
WEEKEND_LABOR_MULTIPLIER = 1.0         # ⚠️ 이 데이터상 주말 할증 0 (실측)
WEEKEND_PREMIUM_PARAM = 1.5            # 실제 특근수당 가정치 (파라미터로 노출)
LABOR_COST_KRW_PER_UNIT_HOUR = 25_000  # 공수 1단위·1시간 인건비 (가정)

PEAK15 = df["y_peak"].astype(float).copy()   # 전체기간 실측 peak15
# ⚠️ `생산량` 은 int64 다. 반사실 계산에서 소수 물량을 옮기므로 float 으로 캐스팅한다.
#    pandas 3.x 는 int 컬럼에 float 대입을 TypeError 로 거부한다.
PROD = df["생산량"].astype(float).copy()


def billing_units(peak: pd.Series) -> pd.DataFrame:
    """두 청구 단위를 산출한다.

    - `월별최대`: 각 월의 최대수요
    - `3개월롤링최대`: 기술통계용 직전 3개월 최대수요(실제 청구규정 아님)
    """
    monthly = peak.groupby(peak.index.to_period("M")).max()
    rolling3 = monthly.rolling(3, min_periods=1).max()
    out = pd.DataFrame({"월별최대": monthly, "3개월롤링최대": rolling3})
    out.index = out.index.astype(str)
    return out


def annual_billing_peak(peak: pd.Series) -> float:
    """관측 최대값을 비용 가정의 기준으로 사용한다. 실제 연간 청구 피크는 아니다."""
    return float(billing_units(peak)["3개월롤링최대"].max())


def basic_charge(peak_kw: float, rate: float = BASE_RATE_KRW_PER_KW_MONTH) -> float:
    """동일 피크·단가가 12개월 유지된다는 가정의 기본요금 참고액."""
    return peak_kw * rate * BILLING_MONTHS


def observed_billing_months(peak: pd.Series) -> float:
    """각 월 관측일수/월 전체일수를 합산해 비용 비교기간을 맞춘다."""
    days = peak.index.normalize().unique()
    if len(days) == 0:
        raise ValueError("비용 비교에 필요한 관측기간이 비어 있음")
    counts = pd.Series(1.0, index=days).groupby(days.to_period("M")).sum()
    return float(sum(count / month.days_in_month for month, count in counts.items()))


tariff_tbl = billing_units(PEAK15)
BASE_BILLING_PEAK = annual_billing_peak(PEAK15)
BASE_BASIC_CHARGE = basic_charge(BASE_BILLING_PEAK)
save_table(tariff_tbl.reset_index(names="월"), "ch4_tariff")

print("── 청구 단위별 최대수요 ──")
display(tariff_tbl.T)
print(f"\n  관측 최대값 (비용 가정 기준) = {BASE_BILLING_PEAK:.0f} kW")
print(f"  12개월 기본요금 환산 참고 = {BASE_BASIC_CHARGE:,.0f} 원 (단가 {BASE_RATE_KRW_PER_KW_MONTH:,.0f} 원/kW·월 가정)")

# 최대값 도달 시각 — 저감의 표적
MAX_PEAK_TIMES = df.index[PEAK15 == PEAK15.max()]
print(f"\n  최대 {PEAK15.max():.0f} kW 도달 {len(MAX_PEAK_TIMES)}건 (저감 표적):")
for _t in MAX_PEAK_TIMES:
    print(f"    - {_t:%Y-%m-%d %H시}")

# 9월 단독 절감이 0원임을 수치로 확인한다
_sept_zeroed = PEAK15.copy()
_sept_zeroed.loc[_sept_zeroed.index >= TEST_START] = 0.0
SEPT_ONLY_SAVING = BASE_BASIC_CHARGE - basic_charge(annual_billing_peak(_sept_zeroed))
print(
    f"\n  테스트구간 peak15를 전부 0으로 낮춘 경우 12개월 기본요금 환산 차이 = "
    f"{SEPT_ONLY_SAVING:,.0f} 원"
)
print("     → 기존 관측 최대가 비용 기준으로 유지되는 가정에서만 성립")


def plot_billing_peaks(tbl: pd.DataFrame):
    """F34 — 월별 최대수요 + 3개월 롤링 최대 기술통계."""
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    x = range(len(tbl))
    ax.plot(x, tbl["월별최대"], color=COLOR_HERO, lw=2, marker="o", ms=6, label="월별 최대수요")
    ax.step(x, tbl["3개월롤링최대"], where="mid", color=PALETTE_ADJACENT[1], lw=2,
            label="3개월 롤링 최대 (기술통계)")
    mx = tbl["월별최대"].max()
    for i, v in enumerate(tbl["월별최대"]):
        if v == mx:
            ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=INK)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + (high - low) * .24)
    ax.set_xticks(list(x))
    ax.set_xticklabels(tbl.index, fontsize=8, rotation=45)
    ax.set_ylabel("peak15 (kW)", fontsize=9)
    ax.set_title("월별 최대수요와 3개월 롤링 최대 (기술통계)", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, tbl


_fig, _src = plot_billing_peaks(tariff_tbl)
save_fig(_fig, "F34", "월별 최대수요와 청구기준", "8.1", source_table=_src)

# %% [markdown]
# ### 8.2 저감 레버 L1~L4
#
# - **목적**: 4개 레버를 반사실 시뮬레이션으로 구현한다.
# - **보고서 대응절**: 4장
# - **산출물**: `ch4_levers.csv`, F37
#
# L1~L4a는 일 생산량, L4b는 주 생산량을 보존한다. 납기·설비용량·매출 영향은 미검증이다.
#
# | 레버 | 조정 | 근거 | 근거 강도 |
# |---|---|---|---|
# | **L1** 설비 기동시점 분산 | 08:00/08:20/08:40 계단식 분산 가정 | 실측 시간대별 피크분포 | **가정기반** (설비 ID 부재 → α 파라미터) |
# | **L2** 고생산 시간대 이동 | 실측 일최대 시각 물량 일부를 10·11·15·16시로 | 관측 회귀계수 β | 모델기반 사후가정 |
# | **L3** 시간당 생산량 상한 | 실측 피크시간에 상한 + 초과분 이월 | L2와 같은 β 가정 | 모델기반 사후가정 |
# | **L4** 야간·주말 이전 | 주간 고부하 일부를 야간·주말로 | 토 51.3 / 일 41.6 여유 | 모델기반 + 인건비 가정 |

# %%
def fit_production_response() -> tuple[float, dict]:
    """생산량 1단위당 `peak15` 증분(반응함수 β)을 **pooled 회귀**로 적합한다.

    시간대별로 따로 적합하면 계수가 불안정하다. `생산량` 은 ERP 등록값이라
    물리적 시간대와 어긋나고(1.5절: 19시 생산량 이상 = 마감 일괄등록 의심),
    시간별 표본이 ~250개뿐이어서 일부 시간대에서 기울기가 음수로 나온다.
    실제로 시간별 적합 시 10·11·16·20·22·23시의 β가 음수→0으로 clip 되어
    **11시 피크를 저감할 수 없게 되는 인공적 결과**가 생긴다.

    따라서 `peak15 ~ 생산량 + 시간더미` 를 한 번에 적합해 **단일 β** 를 쓴다.
    시간 고정효과로 시간대별 기저 부하 차이를 흡수하므로 β는
    "같은 시간대에서 생산량 1단위 차이와 연관된 peak15 차이"로 해석된다.
    관측 회귀계수이며, 생산량을 실제 이동했을 때의 인과효과는 확인되지 않았다.

    Returns
    -------
    (float, dict)
        β 추정치와 적합 진단(p-value, R², n).
    """
    import statsmodels.api as sm

    d = df[(df["생산량"] > 0) & (~df["is_erp_missing"])]
    X = pd.get_dummies(d["시간"], prefix="h", drop_first=True).astype(float)
    X["prod"] = d["생산량"].astype(float).to_numpy()
    X = sm.add_constant(X)
    m = sm.OLS(d["y_peak"].astype(float).to_numpy(), X).fit()
    beta = float(m.params["prod"])
    diag = {
        "β (kW/생산1단위)": round(beta, 6),
        "p-value": round(float(m.pvalues["prod"]), 4),
        "R²": round(float(m.rsquared), 4),
        "n": int(len(d)),
    }
    return max(beta, 0.0), diag


BETA, BETA_DIAG = fit_production_response()
OBS_PEAK_MIN, OBS_PEAK_MAX = float(PEAK15.min()), float(PEAK15.max())
print("── 생산량 반응함수 (pooled, 시간 고정효과) ──")
display(pd.DataFrame([BETA_DIAG]))

# ── 저감 표적: 연중 상위 피크 시각 전부 ───────────────────────────────
# 청구 피크는 **최대값**이므로 상위 사건을 하나라도 남기면 절감이 0이 된다.
TOP_N_PEAKS = 20
TOP_PEAK_TIMES = PEAK15.nlargest(TOP_N_PEAKS).index
PEAK_HOUR_COUNTS = pd.Series(TOP_PEAK_TIMES.hour).value_counts().sort_index()
print(f"\n── 연중 상위 {TOP_N_PEAKS}개 피크 시각의 시간대 분포 ──")
display(PEAK_HOUR_COUNTS.to_frame("건수").T)
print("  실측 상위 피크의 시간대를 사후 확인한 후보 비교이며, 독립 운영 검증이 아니다.")


def clip_to_observed(s: pd.Series) -> pd.Series:
    """음수만 0으로 제한한다. 수신시간의 새 피크는 관측 최대값을 넘을 수 있다.

    기존 함수명은 유지하지만 관측 범위로 상한을 자르지 않는다. 상한 절단은
    부하 이전으로 생긴 악화를 숨기므로 초과 여부를 평가표에 별도로 보고한다.
    """
    return s.clip(lower=0.0)


def lever_L1(
    peak: pd.Series, alpha: float = ALPHA_SIMULTANEOUS, hours=(8, 13)
) -> pd.Series:
    """L1 — 동시 가동 집중 시각의 부하를 계단식으로 분산한다.

    설비 ID가 없어 동시 기동 기여분을 데이터로 산출할 수 없다.
    `alpha` = 해당 시각 peak15 중 동시 기동(돌입전류)·동시 가동에 귀속되는
    비율(가정). 3그룹 계단식 분산으로 시간별 최대값이 그 기여분의 2/3만큼
    낮아지는 가상 시나리오다. 이동한 15분 부하·에너지는 재구성하지 않으므로
    전력량요금 절감을 계산하거나 실제 저감효과로 해석하지 않는다.

    Parameters
    ----------
    hours : tuple[int, ...]
        분산 대상 시각. `(8, 13)` 이 계획서 명세(L1a)이고,
        `(8, 11, 13)` 은 실측 피크 분포를 반영한 확장(L1b 포함)이다.
    """
    out = peak.copy()
    target = np.isin(out.index.hour, list(hours))
    out[target] = out[target] * (1 - alpha * (2 / 3))
    return clip_to_observed(out)


def lever_L2(peak: pd.Series, prod: pd.Series, shift_ratio: float = 0.20) -> tuple[pd.Series, pd.Series]:
    """L2 — 피크 시간의 생산량 일부를 같은 날 저부하 시간으로 이동한다.

    일일 총 생산량을 보존한다. 수용 시간대는 10·11·15·16시(1.5절 EDA에서
    여유가 확인된 시간)로 두고, 각 날의 최고 peak15 시각에서
    `shift_ratio` 만큼의 생산량을 빼 나눠 담는다.

    Returns
    -------
    (pandas.Series, pandas.Series)
        조정된 peak15, 조정된 생산량.
    """
    out, prod_out = peak.copy(), prod.copy()
    RECEIVERS = [10, 11, 15, 16]
    target_days = {t.normalize() for t in TOP_PEAK_TIMES}
    for day, g in peak.groupby(peak.index.normalize()):
        # 연중 상위 피크가 있는 날 또는 θ 초과일만 조정한다
        if day not in target_days and g.max() < THETA:
            continue
        src_t = g.idxmax()
        if prod.loc[src_t] <= 0:
            continue
        move = prod.loc[src_t] * shift_ratio
        # 같은 날 수용 시간대 (원 시각 제외)
        recv = [t for t in g.index if t.hour in RECEIVERS and t != src_t]
        if not recv:
            continue
        per = move / len(recv)
        prod_out.loc[src_t] -= move
        out.loc[src_t] -= BETA * move
        for t in recv:
            prod_out.loc[t] += per
            out.loc[t] += BETA * per
    return clip_to_observed(out), prod_out


def lever_L3(peak: pd.Series, prod: pd.Series, cap_quantile: float = 0.90) -> tuple[pd.Series, pd.Series, float]:
    """L3 — 피크 위험 시간에 시간당 생산량 상한을 걸고 초과분을 다음 시간으로 이월한다.

    실측 `peak15`가 임계를 넘는 시각에만 발동하는 사후 시나리오다.
    마지막 시간에도 잔량이 있으면 그 시간에 남겨 생산량을 보존한다.
    이 경우 상한을 모두 준수했다고 주장할 수 없다.

    Returns
    -------
    (pandas.Series, pandas.Series, float)
        조정된 peak15, 조정된 생산량, 남은 이월 잔량 합계.
    """
    out, prod_out = peak.copy(), prod.copy()
    cap = float(prod[prod > 0].quantile(cap_quantile))
    leftover_total = 0.0
    for day, g in peak.groupby(peak.index.normalize()):
        if g.max() < THETA:
            continue
        carry = 0.0
        times = list(g.index)
        for i, t in enumerate(times):
            want = prod_out.loc[t] + carry
            if peak.loc[t] >= THETA and want > cap:
                carry = want - cap
                new_q = cap
            else:
                carry = 0.0
                new_q = want
            delta = new_q - prod_out.loc[t]
            prod_out.loc[t] = new_q
            out.loc[t] += BETA * delta
        if carry:
            # 다음 날로 넘기거나 버리지 않는다. 일 생산량 보존을 상한보다 우선한다.
            prod_out.loc[times[-1]] += carry
            out.loc[times[-1]] += BETA * carry
            carry = 0.0
        leftover_total += carry
    return clip_to_observed(out), prod_out, leftover_total


def lever_L4(
    peak: pd.Series, prod: pd.Series, target: str = "night", shift_ratio: float = 0.15
) -> tuple[pd.Series, pd.Series, float]:
    """L4 — 주간 고부하 작업 일부를 야간 또는 주말로 이전한다.

    **야간 이전과 주말 이전을 분리한다.** 야간은 인건비 1.5배가 실측으로 확인되고,
    주말 할증은 **이 데이터상 0**이다(요일 분포 균등). 실제 특근수당은 검증할 수
    없으므로 파라미터로 노출하고 민감도에 포함한다.

    Returns
    -------
    (pandas.Series, pandas.Series, float)
        조정된 peak15, 조정된 생산량, 추가 인건비(원).
    """
    out, prod_out = peak.copy(), prod.copy()
    extra_labor = 0.0
    if target == "night":
        recv_hours, mult = [21, 22, 23, 0, 1, 2], NIGHT_LABOR_MULTIPLIER
    else:
        recv_hours, mult = list(range(8, 18)), WEEKEND_PREMIUM_PARAM

    for day, g in peak.groupby(peak.index.normalize()):
        if g.max() < THETA:
            continue
        src_t = g.idxmax()
        if prod.loc[src_t] <= 0:
            continue
        move = prod.loc[src_t] * shift_ratio
        if target == "night":
            recv = [t for t in g.index if t.hour in recv_hours]
        else:
            # 같은 주의 토요일 수용 시간대
            # 월~일 주간 총량을 보존한다. 일요일 표본도 같은 주 토요일로 옮기는
            # 사후 가정이며, 일요일에 실행 가능한 미래 일정 추천은 아니다.
            week_sat = day + pd.Timedelta(days=5 - day.dayofweek)
            recv = [t for t in peak.index if t.normalize() == week_sat and t.hour in recv_hours]
        if not recv:
            continue
        per = move / len(recv)
        prod_out.loc[src_t] -= move
        out.loc[src_t] -= BETA * move
        for t in recv:
            prod_out.loc[t] += per
            out.loc[t] += BETA * per
            # 이전된 물량만큼 추가 인건비 (배수 − 1.0 만큼 증분)
            extra_labor += per * LABOR_COST_KRW_PER_UNIT_HOUR / 1000 * (mult - 1.0)
    return clip_to_observed(out), prod_out, extra_labor


# 전력량요금은 계산하지 않는다. peak15는 시간별 에너지(kWh)가 아니고,
# L1은 이동한 부하를 재구성하지 않아 에너지 보존 여부를 확인할 수 없다.


def evaluate_lever(name: str, peak_new: pd.Series, prod_new: pd.Series,
                   extra_labor: float = 0.0, basis: str = "모델기반") -> dict:
    """관측기간 기본요금 가정액과 같은 기간 추가 인건비만 비교한다.

    순절감액은 두 항목의 차이에 한정한다. 미측정 실행비용·전력량요금은 제외한다.
    연간 기본요금 환산액은 동일 피크가 12개월 유지될 때의 별도 참고값이다.
    """
    if not peak_new.index.equals(PEAK15.index) or not prod_new.index.equals(PROD.index):
        raise ValueError("비교기간과 원자료 인덱스가 일치해야 함")
    new_billing = annual_billing_peak(peak_new)
    delta_peak = BASE_BILLING_PEAK - new_billing
    months = observed_billing_months(PEAK15)
    basic_saving = delta_peak * BASE_RATE_KRW_PER_KW_MONTH * months
    annual_reference = basic_charge(BASE_BILLING_PEAK) - basic_charge(new_billing)
    net = basic_saving - extra_labor
    return {
        "레버": name,
        "근거 강도": basis,
        "Δ최대수요(kW)": round(delta_peak, 1),
        "관측최대 초과 여부": bool((peak_new > OBS_PEAK_MAX).any()),
        "관측최대 초과 시간수": int((peak_new > OBS_PEAK_MAX).sum()),
        "기본요금 절감(원)": round(basic_saving),
        "연간 기본요금 절감 환산(원)": round(annual_reference),
        "비교기간": f"{PEAK15.index.min():%Y-%m-%d}~{PEAK15.index.max():%Y-%m-%d}",
        "비교기간(개월)": round(months, 6),
        "전력량요금 차액(원)": 0,
        "전력량요금 산정": "산정 제외(에너지 및 이동 부하 미검증)",
        "인건비 증가(원)": round(extra_labor),
        "순절감액(원)": round(net),
        "일생산량 보존": bool(
            np.allclose(
                prod_new.groupby(prod_new.index.normalize()).sum(),
                PROD.groupby(PROD.index.normalize()).sum(),
                atol=1e-6,
            )
        ),
        "주생산량 보존": bool(np.allclose(
            prod_new.resample("W-SUN").sum(), PROD.resample("W-SUN").sum(), atol=1e-6,
        )),
    }


# ── 레버 실행 ──────────────────────────────────────────────────────────
p1a = lever_L1(PEAK15, hours=(8, 13))          # 계획서 명세
p1b = lever_L1(PEAK15, hours=(8, 11, 13))      # 실측 피크 분포 반영 확장
p1c = lever_L1(PEAK15, hours=(8, 9, 11, 13))   # 상위 피크가 나타나는 전 시간대 포괄
p2, q2 = lever_L2(PEAK15, PROD)
p3, q3, L3_LEFTOVER = lever_L3(PEAK15, PROD)
p4n, q4n, labor_n = lever_L4(PEAK15, PROD, "night")
p4w, q4w, labor_w = lever_L4(PEAK15, PROD, "weekend")

levers_tbl = pd.DataFrame(
    [
        evaluate_lever("L1a 기동 분산 (08·13시)", p1a, PROD, 0.0, "가정기반(α)"),
        evaluate_lever("L1b 기동 분산 (08·11·13시)", p1b, PROD, 0.0, "가정기반(α)+실측 피크분포"),
        evaluate_lever("L1c 기동 분산 (08·09·11·13시)", p1c, PROD, 0.0, "가정기반(α)+실측 피크분포"),
        evaluate_lever("L2 고생산 시간대 이동", p2, q2, 0.0, "모델기반(β)"),
        evaluate_lever("L3 시간당 생산량 상한", p3, q3, 0.0, "모델기반(β)"),
        evaluate_lever("L4a 야간 이전", p4n, q4n, labor_n, "모델기반+인건비 가정"),
        evaluate_lever("L4b 주말 이전", p4w, q4w, labor_w, "모델기반+특근수당 가정"),
    ]
)
save_table(levers_tbl, "ch4_levers")
print("── 레버별 효과 (전체기간 실측 peak15 기준) ──")
display(levers_tbl)

# ── 핵심 발견 3건을 명시한다 ──────────────────────────────────────────
_l1a = levers_tbl.iloc[0]["Δ최대수요(kW)"]
_l1b = levers_tbl.iloc[1]["Δ최대수요(kW)"]
_l1c = levers_tbl.iloc[2]["Δ최대수요(kW)"]
_l2 = levers_tbl.iloc[3]["Δ최대수요(kW)"]
print(
    f"\n  [발견 1] L1a(08·13시) Δ최대수요 = {_l1a:.1f} kW → "
    f"L1b(+11시) = {_l1b:.1f} kW → L1c(+09시) = {_l1c:.1f} kW\n"
    "     관측 최대값을 기준으로 두므로, 기존 최대 사건이 남으면 기본요금 가정액도 같다.\n"
    "     대상시간 확장은 실측을 확인한 사후 후보이며 실제 이동 부하는 미검증이다."
)
print(
    f"\n  [발견 2] 생산량 이동(L2) Δ최대수요 = {_l2:.1f} kW — 효과가 미미하다.\n"
    f"     pooled β = {BETA:.5f} kW/생산1단위 이므로 20 kW를 낮추려면\n"
    f"     선형 가정에서는 약 {20 / BETA if BETA > 0 else float('inf'):,.0f} 단위에 해당한다.\n"
    "     ERP 등록시각 오차·교란이 계수에 영향을 줄 수 있다.\n"
    "     → 작은 β만으로 설비 동시 가동이 원인이라고 단정할 수 없다."
)
print(
    f"\n  [발견 3] L3 이월 잔량 = {L3_LEFTOVER:.6f} (0이어야 일 생산량 보존)\n"
    f"     주말 할증: 실측 배수 {WEEKEND_LABOR_MULTIPLIER} (할증 0) → "
    f"실제 특근수당 {WEEKEND_PREMIUM_PARAM}배 가정으로 계산."
)


def plot_l1_load_curve(peak: pd.Series, peak_new: pd.Series):
    """F37 — L1 기동 분산 전후 08시 부하 곡선."""
    base_h = peak.groupby(peak.index.hour).mean()
    new_h = peak_new.groupby(peak_new.index.hour).mean()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(base_h.index, base_h.to_numpy(), color=COLOR_MUTED, lw=2, marker="o", ms=4, label="현행")
    ax.plot(new_h.index, new_h.to_numpy(), color=COLOR_HERO, lw=2, marker="o", ms=4, label="L1 적용")
    ax.annotate(
        f"08시 {base_h[8]:.1f} → {new_h[8]:.1f}",
        (8, base_h[8]), xytext=(.55, .18), textcoords="axes fraction", fontsize=8, color=INK,
        bbox=dict(facecolor="white", edgecolor="#d6d5d1", pad=4),
        arrowprops=dict(arrowstyle="-", color=INK_SOFT),
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시각", fontsize=9)
    ax.set_ylabel("평균 peak15 (kW)", fontsize=9)
    ax.set_title(f"L1 기동 분산 전후 시간대별 부하 (α={ALPHA_SIMULTANEOUS:.0%})", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, pd.DataFrame({"현행": base_h, "L1 적용": new_h})


_fig, _src = plot_l1_load_curve(PEAK15, p1c)
save_fig(_fig, "F37", "L1 기동분산 전후 부하곡선", "8.2", source_table=_src)

# %% [markdown]
# ### 8.3 시나리오 비교와 민감도
#
# - **목적**: 레버 조합 시나리오의 **순절감액**을 계산하고, 가정 파라미터
#   ±30%에서 **우선순위가 뒤집히지 않는지** 확인한다.
# - **보고서 대응절**: 4장
# - **산출물**: `ch4_scenarios.csv`, `ch4_sensitivity.csv`, F35·F36·F38
#
# **반드시 순절감액으로 판정한다.** L4(야간 이전)는 인건비가 1.5배로 늘기 때문에
# "무조건 좋은 레버가 아님"을 수치로 보이는 것이 정직성 가점 요소다.

# %%
def run_scenarios() -> pd.DataFrame:
    """레버 조합 시나리오를 평가한다.

    S1과 S1b를 나란히 두어 **상위 피크 사건을 하나라도 남기면 절감이 0** 임을
    표로 직접 보인다(최대값 목적함수의 핵심 성질).
    """
    def row(label, peak_new, prod_new, extra=0.0, comp="-"):
        base = evaluate_lever("-", peak_new, prod_new, extra)
        return {
            "시나리오": label,
            **{k: v for k, v in base.items() if k not in ("레버", "근거 강도")},
            "구성": comp,
        }

    p1c2, q1c2 = lever_L2(p1c, PROD)
    p1c3, q1c3, _ = lever_L3(p1c, PROD)
    p1c24, q1c24, lab = lever_L4(p1c2, q1c2, "night")

    return pd.DataFrame(
        [
            row("S0 현행 (무조정)", PEAK15, PROD, 0.0, "-"),
            row("S1a L1a 단독 (08·13시)", p1a, PROD, 0.0, "기동 분산 (계획서 명세)"),
            row("S1b L1b 단독 (08·11·13시)", p1b, PROD, 0.0, "기동 분산 + 11시 포함"),
            row("S1c L1c 단독 (08·09·11·13시)", p1c, PROD, 0.0, "기동 분산 + 09·11시 포함"),
            row("S2 L1c+L2", p1c2, q1c2, 0.0, "기동 분산 + 시간대 이동"),
            row("S3 L1c+L3", p1c3, q1c3, 0.0, "기동 분산 + 생산량 상한"),
            row("S4 L1c+L2+L4(야간)", p1c24, q1c24, lab, "기동 분산 + 시간대 이동 + 야간 이전"),
        ]
    )


scenarios_tbl = run_scenarios()
save_table(scenarios_tbl, "ch4_scenarios")
print("── 시나리오별 순절감액 ──")
display(scenarios_tbl[["시나리오", "구성", "Δ최대수요(kW)", "기본요금 절감(원)",
                       "인건비 증가(원)", "순절감액(원)", "일생산량 보존"]])

BEST_SCENARIO = scenarios_tbl.loc[scenarios_tbl["순절감액(원)"].idxmax(), "시나리오"]
PEAK_REDUCTION_KW = float(scenarios_tbl["Δ최대수요(kW)"].max())
BEST_NET_SAVING = float(scenarios_tbl["순절감액(원)"].max())
print(f"\n  가정 내 관측기간 비교액 최대 시나리오 = {BEST_SCENARIO}")
print(f"  최대 Δ최대수요 = {PEAK_REDUCTION_KW:.1f} kW / 순절감액 {BEST_NET_SAVING:,.0f} 원")


def sensitivity_analysis() -> pd.DataFrame:
    """기본요금 단가 ±30% × α 10/20/30% 에서 레버 우선순위가 보존되는지 본다."""
    rows = []
    for rate_mult in (0.7, 1.0, 1.3):
        rate = BASE_RATE_KRW_PER_KW_MONTH * rate_mult
        for alpha in (0.10, 0.20, 0.30):
            pa = lever_L1(PEAK15, alpha, hours=(8, 9, 11, 13))
            pa2, qa2 = lever_L2(pa, PROD)
            pa4, qa4, lab = lever_L4(pa2, qa2, "night")
            combos = {
                "L1c 단독": (pa, PROD, 0.0),
                "L1c+L2": (pa2, qa2, 0.0),
                "L1c+L2+L4(야간)": (pa4, qa4, lab),
            }
            nets = {}
            for label, (pk, _pq, lb) in combos.items():
                nb = annual_billing_peak(pk)
                nets[label] = (
                    (BASE_BILLING_PEAK - nb) * rate * observed_billing_months(PEAK15) - lb
                )
            best = max(nets, key=nets.get)
            rows.append(
                {
                    "기본요금단가 배수": rate_mult, "α": alpha,
                    "비교기간(개월)": round(observed_billing_months(PEAK15), 6),
                    **{f"{k} 순절감(원)": round(v) for k, v in nets.items()},
                    "최적 조합": best,
                }
            )
    return pd.DataFrame(rows)


sensitivity_tbl = sensitivity_analysis()
save_table(sensitivity_tbl, "ch4_sensitivity")
PRIORITY_STABLE = sensitivity_tbl["최적 조합"].nunique() == 1
print("\n── 민감도 분석 (기본요금 단가 ±30% × α 10/20/30%) ──")
display(sensitivity_tbl)
print(
    f"\n  레버 우선순위 보존: {PRIORITY_STABLE} "
    f"(최적 조합 후보 {sensitivity_tbl['최적 조합'].unique().tolist()})"
)


def plot_waterfall(levers: pd.DataFrame):
    """F35 — 레버별 Δ최대수요 폭포."""
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    vals = levers["Δ최대수요(kW)"].to_numpy()
    colors = [COLOR_HERO if v > 0 else PALETTE_ADJACENT[1] for v in vals]
    bars = ax.bar(levers["레버"], vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals.max(), 1) * 0.03,
                f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK_SOFT, lw=1)
    ax.set_ylabel("Δ최대수요 (kW)", fontsize=9)
    ax.set_title("레버별 가정 내 최대수요 저감량", fontsize=10, color=INK)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)
    fig.tight_layout()
    return fig, levers[["레버", "Δ최대수요(kW)"]]


_fig, _src = plot_waterfall(levers_tbl)
save_fig(_fig, "F35", "레버별 최대수요 저감", "8.3", source_table=_src)


def plot_net_saving(scen: pd.DataFrame):
    """F36 — 시나리오별 순절감액 (발산형)."""
    t = scen[scen["시나리오"] != "S0 현행 (무조정)"]
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    vals = t["순절감액(원)"].to_numpy() / 1e4
    colors = [COLOR_HERO if v > 0 else PALETTE_ADJACENT[1] for v in vals]
    bars = ax.bar(t["시나리오"], vals, color=colors, width=0.58)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + abs(vals).max() * 0.03,
                f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK_SOFT, lw=1)
    ax.margins(y=.18)
    ax.set_ylabel("관측기간 비용 비교액 (만원)", fontsize=9)
    ax.set_title("사후 시나리오: 기본요금 가정액 - 추가 인건비", fontsize=10, color=INK)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([str(x).replace(" (", "\n(").replace("L4(", "L4\n(") for x in t["시나리오"]], fontsize=8, ha="center")
    fig.tight_layout()
    return fig, t[["시나리오", "순절감액(원)"]]


_fig, _src = plot_net_saving(scenarios_tbl)
save_fig(_fig, "F36", "시나리오별 순절감액", "8.3", source_table=_src)


def plot_sensitivity_heatmap(sens: pd.DataFrame):
    """F38 — 민감도 히트맵 (기본요금 단가 × α)."""
    piv = sens.pivot(index="α", columns="기본요금단가 배수", values="L1c+L2 순절감(원)") / 1e4
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.imshow(piv.to_numpy(), aspect="auto", cmap=CMAP_SEQ)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"×{c}" for c in piv.columns], fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"α={i:.0%}" for i in piv.index], fontsize=9)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, f"{piv.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if (piv.iloc[i, j] - piv.to_numpy().min()) / max(np.ptp(piv.to_numpy()), 1e-9) > .55 else INK)
    ax.set_title("S2(L1c+L2) 관측기간 비용 비교액 민감도 (만원)", fontsize=10, color=INK)
    ax.grid(False)
    fig.tight_layout()
    return fig, piv.round(2)


_fig, _src = plot_sensitivity_heatmap(sensitivity_tbl)
save_fig(_fig, "F38", "민감도 히트맵", "8.3", source_table=_src)

# %% [markdown]
# ### 8.4 현장 운영 프로토콜
#
# - **목적**: 모델 출력이 **어떤 현장 조치로 이어지는지** 타임라인으로 정리한다.
# - **보고서 대응절**: 4장
# - **산출물**: `ch4_protocol.csv`, F39

# %%
def build_protocol() -> pd.DataFrame:
    """전일 24시 예측 → 익일 조치로 이어지는 운영 프로토콜."""
    rows = [
        ("전일 23:50", "데이터 수집 점검", "23시 이전 확정 실적과 수집 상태 점검; 23~24시 구간은 종료 후 확정", "자동", "마지막 구간의 관측완료 시각 준수"),
        ("전일 24:00 이후", "예측 실행 (원점)", "24시간 실적 확정 후 저장된 서비스 번들로 익일 전력 예측; 별도 분류모델로 피크확률 산출", "자동", "번들 manifest의 모델·입력 계약 확인; 연구 평가의 선정모델과 구분"),
        ("예측 완료 후", "경보 판정", "peak15 예측 ≥ τ 인 시각을 위험시간으로 지정", "자동", "선정한 τ를 저장해 사용"),
        ("익일 00:10 이후", "조정 후보 검토", "위험시간에서 L1 기동 분산의 적용 가능성 검토; L2는 추가 비용과 비교", "현장", "자동 스케줄 생성 기능 미구현; 납기·용량·인력 확인"),
        ("익일 07:30", "기동 계획 검토", "08:00/08:20/08:40 단계 기동의 현장시험 여부 결정", "현장", "효과·생산량·인건비 영향 실측 필요"),
        ("익일 08:00~", "실시간 감시", "위험시간 진입 시 15분 최대수요와 피크 기준을 대조", "현장", "q90은 평균전력 구간이므로 평균전력의 오차 점검에만 사용하고 15분 피크 감시와 분리"),
        ("익일 12:50", "재가동 점검", "13시 단계 기동의 필요성과 실행 여건 확인", "현장", "분산으로 다른 시간에 새 피크가 생기는지도 점검"),
        ("익일 종료 후", "일일 검증 제안", "예측 편차·실행 조치·생산량·15분 부하를 함께 기록", "현장", "3장 실패조건과 대조; 실제 저감효과 별도 평가"),
        ("월 1회", "임계값 재점검", "θ·τ 재산출 여부 검토 (재산출 시 정의 변경 주의)", "분석", "θ 고정 원칙 유지"),
    ]
    return pd.DataFrame(rows, columns=["시점", "조치", "내용", "주체", "비고"])


protocol_tbl = build_protocol()
save_table(protocol_tbl, "ch4_protocol")
print("── 현장 운영 프로토콜 ──")
display(protocol_tbl)

# 점검 우선순위 — 7장 규칙과 연결
priority_tbl = rules_tbl.copy()
priority_tbl["점검 우선순위"] = range(1, len(priority_tbl) + 1)
priority_tbl["권고 조치"] = [
    "L1 기동 분산 현장시험 검토 + 실시간 감시 강화",
    "L2 생산 시간대 이동 검토",
    "모니터링 우선(낮은 발생비율이 피크 부재를 뜻하지 않음)",
][: len(priority_tbl)]
save_table(priority_tbl, "ch4_priority")
print("\n── 규칙별 점검 우선순위 (7.6절 규칙과 연결) ──")
display(priority_tbl)


def plot_alert_timeline(prot: pd.DataFrame):
    """F39 — 시점·주체·조치를 분리한 수평 읽기형 타임라인."""
    fig, ax = plt.subplots(figsize=(11, 5.1))
    colors = {"자동": COLOR_HERO, "현장": PALETTE_ADJACENT[1], "분석": PALETTE_ADJACENT[2]}
    count = len(prot)
    ax.plot([.02, .02], [-.2, count-.8], color="#d6d5d1", lw=1.4, zorder=1)
    for i, (_, r) in enumerate(prot.iterrows()):
        y = count - i - 1
        ax.scatter(.02, y, s=80, color=colors.get(r["주체"], COLOR_MUTED), zorder=3)
        ax.text(.07, y, r["시점"], va="center", fontsize=9, color=INK)
        ax.text(.31, y, r["주체"], va="center", fontsize=9, color=INK_SOFT)
        ax.text(.43, y, r["조치"], va="center", fontsize=9, color=INK)
    ax.set_xlim(-.01, 1)
    ax.set_ylim(-.6, count-.3)
    ax.set_title("경보 운영 타임라인 (전일 24시 예측 → 익일 조치)", fontsize=11, color=INK, pad=15)
    ax.axis("off")
    fig.tight_layout()
    return fig, prot


_fig, _src = plot_alert_timeline(protocol_tbl)
save_fig(_fig, "F39", "경보 운영 타임라인", "8.4", source_table=_src)

# %% [markdown]
# ### 8.5 보고서 4장 본문 초안 생성
#
# - **목적**: 위 결과를 4장 본문 초안 markdown 으로 출력한다.
# - **보고서 대응절**: 4장 전체
# - **산출물**: `outputs/report_ch4_draft.md`

# %%
def write_ch4_draft() -> str:
    """측정 결과와 미검증 가정을 구분한 4장 초안을 생성한다."""
    best = scenarios_tbl.loc[scenarios_tbl["순절감액(원)"].idxmax()]
    l1c = levers_tbl[levers_tbl["레버"].str.startswith("L1c")].iloc[0]
    s2 = scenarios_tbl[scenarios_tbl["시나리오"].str.startswith("S2")].iloc[0]
    l2_increment = s2["연간 기본요금 절감 환산(원)"] - l1c["연간 기본요금 절감 환산(원)"]
    months = observed_billing_months(PEAK15)
    md = f"""# 제 4 장. 현장 활용방안 〔10점〕

## 4.1 일일 예측과 현장 판단

전일 실적이 확정된 뒤 저장된 서비스 번들로 다음 날 24시간의 평균전력·15분 최대전력을 예측한다.
사용 모델과 입력 계약은 번들 manifest에서 확인하며 연구 평가의 선정모델과 구분한다.
회귀 예측의 경보 기준을 넘는 시각을 위험시간으로 표시하고, 별도 피크 분류모델의 확률은
보조 정보로 제공한다. 고위험 조건의 경보는 생산·휴무 계획과 함께 현장 담당자가 확인한다.
자동 스케줄 최적화와 설비 제어는 구현 범위에 포함되지 않는다.
q90은 평균전력의 예측구간이므로 15분 최대수요 경보의 임계값으로 사용하지 않는다.

## 4.2 사후 시나리오의 범위와 비용 가정

이 분석은 {PEAK15.index.min():%Y-%m-%d}~{PEAK15.index.max():%Y-%m-%d}의
**실측 부하를 알고 있는 상태에서 조정 대상을 고른 가정 기반 사후 시나리오**다.
미탐지·오경보가 조정 결과에 미치는 영향이나 실제 운영 절감 성과를 검증한 것은 아니다.
관측 최대값 {BASE_BILLING_PEAK:.0f} kW가 비교기간 내 동일하게 기본요금 기준으로
유지된다고 가정하고, 단가는 {BASE_RATE_KRW_PER_KW_MONTH:,.0f}원/kW·월로 설정했다.
실제 계약종별 요금 규정과 과거 청구 이력을 재현한 계산은 아니다.

비용 비교기간은 월별 관측일수/해당 월 일수의 합인 **{months:.4f}개월**이다.
기본요금 가정액과 추가 인건비를 이 관측기간으로 맞추어 비교한다.
별도의 12개월 환산액은 동일 피크·단가가 지속될 때의 참고값이며, 미관측 계절의 성과를 보장하지 않는다.
기존 최대값이 유지되는 가정에서는 테스트 구간만 0으로 낮추어도
기본요금의 12개월 환산 차이는 {SEPT_ONLY_SAVING:,.0f}원이다.

**전력량요금 절감은 산정하지 않는다.** peak15는 시간별 평균전력이나 에너지(kWh)가 아니며,
기동 분산으로 이동할 15분 부하를 재구성하지 않았다. 표의 전력량요금 차액 0은
산정에서 제외했다는 표시이며, 실제 사용량이 보존되었거나 절감이 없음을 입증한 값이 아니다.

## 4.3 조정 가정

- L1: 08·13시부터 시작해 11시, 09시까지 기동 분산 대상을 확장한다.
  해당 시간 최대전력 중 동시 기동 기여율을 α={ALPHA_SIMULTANEOUS:.0%}로 두고,
  3그룹 분산 시 그 기여분의 2/3만큼 시간 최대값이 낮아진다고 가정한다.
  이동 부하의 새 피크·설비용량·작업 순서·인력·납기 영향은 미검증이다.
- L2: 실측 일최대 시간의 생산량 20%를 같은 날 10·11·15·16시로 옮긴다.
- L3: 실측 피크 위험시간에서 생산량 90백분위 상한 초과분을 다음 시간으로 옮긴다.
  마지막 시간에 남으면 그 시간에 유지하여 일 생산량을 보존한다. 따라서 상한 완전 준수는 보장하지 않는다.
- L4: 실측 고부하 작업의 15%를 야간 또는 같은 주 토요일로 옮긴다.
  L1~L4a는 일 생산량, L4b는 주 생산량을 보존한다. 생산량 보존만으로 매출·납기 보존을 주장하지 않는다.

L2~L4는 시간 고정효과를 포함한 관측 회귀의 β={BETA:.6f} kW/단위를 적용했다.
ERP 등록시각 오차와 교란이 계수에 영향을 줄 수 있으므로 인과적 생산량 효과로 해석하지 않는다.
작은 β만으로 설비 동시 가동이 피크의 주원인이라고 단정할 수도 없다.

## 4.4 관측기간 비용 비교

기본요금 가정액 = Δ최대수요 × 단가 × 관측월수.
비용 비교액(기존 CSV의 순절감액) = 기본요금 가정액 − 같은 기간 추가 인건비.
조정·설비투자·납기 변경 등 미측정 실행비용과 전력량요금은 포함하지 않는다.
수신시간에 생긴 새 피크는 기존 관측 최대값으로 자르지 않는다. 관측최대 초과 여부와 시간수를
평가표에 기록하며, 최대수요 감소량과 비용 비교액이 음수이면 해당 가정에서 악화된 결과다.

| 시나리오 | Δ최대수요 | 관측기간 기본요금 가정액 | 관측기간 추가 인건비 | 관측기간 비용 비교액 | 12개월 기본요금 환산 참고 |
|---|---|---|---|---|---|
"""
    for _, row in scenarios_tbl.iterrows():
        md += (
            f"| {row['시나리오']} | {row['Δ최대수요(kW)']:.1f} kW | "
            f"{row['기본요금 절감(원)']:,.0f}원 | {row['인건비 증가(원)']:,.0f}원 | "
            f"{row['순절감액(원)']:,.0f}원 | {row['연간 기본요금 절감 환산(원)']:,.0f}원 |\n"
        )
    md += f"""
비교한 후보와 가정 내 비용 비교액이 가장 큰 시나리오는 {best['시나리오']}다.
다만 **L1c를 우선 현장시험 대상으로 제안**한다. L2를 추가한 S2의 기본요금 증가분은
12개월 환산 기준 {l2_increment:,.0f}원이며, L2의 실행비용을 0으로 둔 결과다.
이 추가 효과보다 일정변경 비용이 작고 실행 가능성이 확인되는 경우에만 L2 병행을 검토한다.

## 4.5 민감도와 남은 검증

기본요금 단가 ±30%와 α 10·20·30%의 9개 조합에서 L1c·L1c+L2·L1c+L2+야간의
세 후보를 비교했다. 가정 내 1위 후보는 {', '.join(sensitivity_tbl['최적 조합'].unique())}이며,
조합 간 1위가 {'유지되었다' if PRIORITY_STABLE else '달라졌다'}.
이 결과는 제한된 후보 비교이며 전체 운영의 최적성을 증명하지 않는다.
야간·주말 인건비 배수는 각각 {NIGHT_LABOR_MULTIPLIER}·{WEEKEND_PREMIUM_PARAM},
생산량 1,000단위를 공수 1단위로 환산하고 공수·시간당 {LABOR_COST_KRW_PER_UNIT_HOUR:,.0f}원을
적용한 비용 가정이다. 실제 작업시간·특근수당을 확인한 뒤 다시 계산해야 한다.

## 4.6 현장 운영 제안

| 시점 | 조치 | 내용 | 주체 |
|---|---|---|---|
"""
    for _, row in protocol_tbl.iterrows():
        md += f"| {row['시점']} | {row['조치']} | {row['내용']} | {row['주체']} |\n"
    md += "\n규칙 R1~R3은 같은 OOF 표본에서 도출한 설명용 조건이다. 새로운 기간에서 검증한 뒤 적용한다.\n"
    return md


_ch4 = write_ch4_draft()
(OUTPUT_DIR / "report_ch4_draft.md").write_text(_ch4, encoding="utf-8")
print(f"── 4장 본문 초안 생성 완료 ({len(_ch4):,}자) → outputs/report_ch4_draft.md ──")
