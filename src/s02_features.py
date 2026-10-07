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
from s01_diagnose import (  # noqa: F401
    THETA, TEST_START, df, df_raw, operating_calendar, restore_hours,
    build_datetime_index, flag_and_repair, add_targets, build_operating_calendar,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 2. 파생변수 구성 및 시간누수 차단
#
# 보고서 **1.5절·2.3절**의 근거를 생성한다.
#
# 전력 피처가 예측 원점 뒤의 관측값을 참조하지 않는지 검증한다.
# 생산·휴무계획의 사전 확보와 실측 기상을 예보 대리로 쓰는 가정은 별도 한계다.

# %% [markdown]
# ### 2.1 달력·공정상태 파생변수
#
# - **목적**: 요일·공휴일·가동상태 등 예측 시점에 확정적으로 아는 정보를 만든다.
# - **보고서 대응절**: 1.5 전처리 및 파생변수 구성, 2.3 입력변수 구성
# - **산출물**: 달력·공정상태 피처
#
# **설계 근거 3건** (1장 실측에서 도출)
#
# | 설계 | 근거 |
# |---|---|
# | **토·일 분리** (단일 `주말` 플래그 금지) | 토 51.3 / 일 41.6 으로 레짐이 다르다 (토=중간, 일=기저) |
# | **주야간 = `9 ≤ h ≤ 17`** | `인건비==1.0` 구간이 정확히 9~17시. **08시는 1.5 구간**이다 |
# | **휴무는 `일생산량==0`** (법정공휴일 아님) | 법정 기준이면 하계휴가 9일과 `08-09` 재가동을 못 잡는다 |

# %%
# 2021년 법정공휴일 (폐쇄망 심사환경 대응 — holidays 패키지 미사용)
HOLIDAYS_2021 = pd.to_datetime([
    "2021-01-01",                               # 신정
    "2021-02-11", "2021-02-12", "2021-02-13",   # 설날 연휴
    "2021-03-01",                               # 삼일절
    "2021-05-05",                               # 어린이날
    "2021-05-19",                               # 부처님오신날
    "2021-06-06",                               # 현충일
    "2021-08-15", "2021-08-16",                 # 광복절 + 대체공휴일
])

# 컬럼명은 ASCII 로 둔다(LightGBM·SHAP 이 특수문자 컬럼명을 거부하는 경우가 있다).
# 보고서 표에는 아래 한글 라벨로 치환해 싣는다.
FEATURE_LABELS: dict[str, str] = {}


def _label(name: str, korean: str) -> str:
    """피처명에 한글 표시 라벨을 등록하고 원래 이름을 돌려준다."""
    FEATURE_LABELS[name] = korean
    return name


def add_calendar_features(d: pd.DataFrame, cal: pd.DataFrame, holidays=None) -> pd.DataFrame:
    """달력·공정상태 파생변수를 추가한다.

    달력은 기지 정보이며, 생산량으로 유도한 휴무 상태는 사전 운영계획을
    확보한다는 가정 아래 사용한다. 실제 생산실적과 계획의 차이는 검증되지 않았다.

    Parameters
    ----------
    d : pandas.DataFrame
        datetime 인덱스를 가진 시간별 데이터.
    cal : pandas.DataFrame
        1.4절에서 도출한 일별 가동 캘린더.
    holidays : array-like of datetime, optional
        공휴일 목록. 기본은 `HOLIDAYS_2021`(학습 구간). 서비스에서 2021-09-14 이후
        날짜를 예측할 때 운영 휴일표를 주입하는 통로다(노트북 결과에는 영향 없음).
    """
    out = d.copy()
    idx = out.index
    day = idx.normalize()
    hol = HOLIDAYS_2021 if holidays is None else pd.DatetimeIndex(holidays)

    # ── 시간정보 ──────────────────────────────────────────────────────
    out[_label("hour", "시간")] = idx.hour
    out[_label("dow", "요일")] = idx.dayofweek
    out[_label("month", "월")] = idx.month
    # 토·일을 분리한다 — 단일 '주말' 플래그로는 3레짐을 표현할 수 없다
    out[_label("is_sat", "토요일")] = (idx.dayofweek == 5).astype(int)
    out[_label("is_sun", "일요일")] = (idx.dayofweek == 6).astype(int)
    out[_label("is_holiday", "공휴일")] = day.isin(hol).astype(int)
    # 주기형 인코딩 — 23시와 0시가 인접함을 모델에 알려준다
    out[_label("hour_sin", "시간 sin")] = np.sin(2 * np.pi * idx.hour / 24)
    out[_label("hour_cos", "시간 cos")] = np.cos(2 * np.pi * idx.hour / 24)
    out[_label("dow_sin", "요일 sin")] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out[_label("dow_cos", "요일 cos")] = np.cos(2 * np.pi * idx.dayofweek / 7)

    # ── 공정상태 ──────────────────────────────────────────────────────
    # 주간 = 인건비 1.0 구간(9~17시). 08시는 1.5 구간임에 유의.
    out[_label("is_daytime", "주간(9~17시)")] = idx.hour.isin(range(9, 18)).astype(int)
    # 동시 기동 시점 — 08시 피크율 24.5%(1위), 13시 17.1%
    out[_label("is_startup_08", "08시 기동")] = (idx.hour == 8).astype(int)
    out[_label("is_restart_13", "13시 재가동")] = (idx.hour == 13).astype(int)

    # 가동 캘린더 기반 (일 단위 → 시간 단위로 브로드캐스트)
    cal_h = cal.reindex(day)
    out[_label("is_shutdown", "휴무일")] = cal_h["is_shutdown"].to_numpy().astype(int)
    out[_label("shutdown_nth", "휴무 n일차")] = cal_h["휴무_n일차"].to_numpy().astype(int)
    out[_label("is_first_day_back", "휴무 직후 재가동일")] = cal_h["is_재가동일"].to_numpy().astype(int)
    # 전일이 휴무였는지 — 재가동 효과의 대리변수
    prev_shut = cal["is_shutdown"].shift(1, fill_value=False).reindex(day)
    out[_label("prev_day_shutdown", "전일 휴무")] = prev_shut.to_numpy().astype(int)
    # 가동 전환 = 전일과 가동상태가 바뀐 날
    switch = (cal["is_shutdown"] != cal["is_shutdown"].shift(1)).reindex(day)
    out[_label("is_state_switch", "가동 전환일")] = switch.fillna(False).to_numpy().astype(int)

    return out


# %% [markdown]
# ### 2.2 Day-ahead 예측 원점 규칙
#
# - **목적**: 예측 원점을 엄밀히 정의하고, 이동통계를 **원점 기준 1회 계산 후 브로드캐스트**한다.
# - **보고서 대응절**: 2.1 실험 설계, 2.3 입력변수 구성
# - **산출물**: `origin_frame` (일자 인덱스), F13
#
# $$\text{origin}(D{+}1) = D\text{일 } 24{:}00$$
#
# **왜 `D일 23:00` 이 아닌가.** `h=23` 을 예측할 때 `lag24` 의 참조 시각은 `D일 23:00` 인데,
# 이 값은 23:00~24:00 **구간의 집계**이므로 `D일 23:00` 시점에는 **아직 미완성**이다.
# 원점을 24:00으로 두면 참조 시각과 원점의 간격이 `h` 가 되어, 모든 `h` 에서
# `k ≥ 24` 인 lag 가 엄밀히 안전해진다.
#
# **이동통계 금지 패턴.** `rolling(24).mean().shift(1)` 은 `h>0` 에서 **같은 날 과거 시각**을
# 창에 포함시켜 누수가 된다. 반드시 원점에서 1회 계산해 그날 24시간에 동일 값을 뿌린다.

# %%
SAFE_LAGS = (24, 48, 168)


def build_origin_frame(d: pd.DataFrame, theta: float) -> pd.DataFrame:
    """일자 D 기준, **D일 24:00 시점에 관측이 완료된** 이동통계를 만든다.

    창의 오른쪽 끝을 `D일 23:00` 으로 두면 그 값은 D일 24:00 에 관측 완료되므로
    `origin(D+1) = D일 24:00` 에서 안전하게 사용할 수 있다.

    Returns
    -------
    pandas.DataFrame
        일자(D) 인덱스. 이 표를 `날짜 - 1일` 로 머지하면 D+1 의 피처가 된다.
    """
    avg, peak = d["y_avg"], d["y_peak"]
    is_peak = (peak >= theta).astype(float)

    cols = {}
    for win in (24, 168):
        # min_periods 를 창 크기와 같게 두어 부분 창(워밍업)을 배제한다
        cols[f"roll{win}_avg_mean"] = avg.rolling(win, min_periods=win).mean()
        cols[f"roll{win}_avg_max"] = avg.rolling(win, min_periods=win).max()
        cols[f"roll{win}_avg_std"] = avg.rolling(win, min_periods=win).std()
        cols[f"roll{win}_peak_max"] = peak.rolling(win, min_periods=win).max()
        cols[f"roll{win}_peak_cnt"] = is_peak.rolling(win, min_periods=win).sum()
    roll = pd.DataFrame(cols, index=d.index)

    # 각 날짜의 마지막 시각(23시) 행만 취한다 = 그 날 24:00 시점의 통계
    last = roll[roll.index.hour == 23].copy()
    last.index = last.index.normalize()
    last.index.name = "origin_date"

    for c in last.columns:
        FEATURE_LABELS.setdefault(
            c,
            c.replace("roll24_", "직전24h ")
            .replace("roll168_", "직전168h ")
            .replace("avg_mean", "평균전력 평균")
            .replace("avg_max", "평균전력 최대")
            .replace("avg_std", "평균전력 표준편차")
            .replace("peak_max", "peak15 최대")
            .replace("peak_cnt", "피크 발생건수"),
        )
    return last


def add_lag_features(d: pd.DataFrame) -> pd.DataFrame:
    """안전 lag(24·48·168시간)만 추가한다.

    `lag24` 의 참조 시각은 대상 시각보다 24시간 이르므로, 그 값의 **관측완료 시각**
    (참조시각 + 1시간)도 원점(= 대상일 00:00)보다 이르다. 따라서 안전하다.
    """
    out = d.copy()
    for col, kor in (("y_avg", "평균전력"), ("y_peak", "peak15")):
        for lag in SAFE_LAGS:
            name = _label(f"{col}_lag{lag}", f"{kor} {lag}시간 전")
            out[name] = out[col].shift(lag)
    return out


def add_origin_stats(d: pd.DataFrame, origin: pd.DataFrame) -> pd.DataFrame:
    """원점 기준 이동통계를 **대상일 24시간에 동일 값으로 브로드캐스트**한다.

    대상 시각 T 의 날짜에서 하루를 뺀 날(D)의 통계를 가져온다.
    같은 날 안에서는 값이 변하지 않으므로 `h>0` 누수가 원천적으로 불가능하다.
    """
    out = d.copy()
    key = out.index.normalize() - pd.Timedelta(days=1)
    merged = origin.reindex(key)
    for c in origin.columns:
        out[c] = merged[c].to_numpy()
    return out


# %% [markdown]
# ### 2.3 피처군 조립
#
# - **목적**: 6개 피처군을 조립하고, 원본 중복 컬럼을 제외한다.
# - **보고서 대응절**: 2.3 입력변수 구성
# - **산출물**: `X` 설계행렬, `ch2_feature_groups.csv`
#
# **제외 컬럼**: `day`/`d`/`m` 은 요일·일·월의 원본 중복이다. 파생변수로 재생성했으므로
# 원본은 제외한다(동일 정보의 중복 투입은 중요도 해석을 왜곡한다).
#
# **기상변수 가정**: 실측 기상을 D+1 **예보의 대리(proxy)** 로 사용한다.
# 실제 운영에서는 예보를 쓰게 되므로, 기상 ablation 결과는 "예보오차에 대한 **성능 하한**"으로
# 해석해야 한다. 이 가정을 보고서 1.5절·2.3절에 명시한다.

# %%
EXCLUDE_RAW_COLS = ["day", "d", "m"]


def add_production_weather_features(d: pd.DataFrame) -> pd.DataFrame:
    """생산정보·기상정보 파생변수를 추가한다.

    생산량은 ERP **계획값**으로 가정하므로 대상 시점 값을 사용할 수 있다.
    이 가정의 타당성은 6.3절 '생산량 제외' ablation 으로 검증한다.
    """
    out = d.copy()
    out[_label("prod", "계획생산량")] = out["생산량"].astype(float)
    out[_label("headcount", "공장인원(정규화 공수)")] = out["공장인원"].astype(float)
    # 전일 동시각 대비 생산량 변화율 (분모 0 방지)
    prev = out["생산량"].shift(24)
    out[_label("prod_chg_ratio", "생산량 전일대비 변화율")] = (
        (out["생산량"] - prev) / prev.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # 당일 누적 생산량 — ERP 계획이므로 원점에서 기지
    out[_label("prod_cum_day", "일 누적생산량")] = out.groupby(out.index.normalize())["생산량"].cumsum()

    out[_label("temp", "기온")] = out["기온"].astype(float)
    out[_label("humid", "습도")] = out["습도"].astype(float)
    out[_label("wind", "풍속")] = out["풍속"].astype(float)
    out[_label("rain", "강수량")] = out["강수량"].astype(float)
    # 냉방도일·난방도일 — 기온의 비선형 효과를 선형 항으로 분해
    out[_label("cdd", "냉방도일 CDD")] = (out["기온"] - 24).clip(lower=0)
    out[_label("hdd", "난방도일 HDD")] = (18 - out["기온"]).clip(lower=0)
    return out


FEATURE_GROUPS = {
    "과거전력": [f"{c}_lag{l}" for c in ("y_avg", "y_peak") for l in SAFE_LAGS],
    "이동통계": [
        f"roll{w}_{s}"
        for w in (24, 168)
        for s in ("avg_mean", "avg_max", "avg_std", "peak_max", "peak_cnt")
    ],
    "시간정보": [
        "hour", "dow", "month", "is_sat", "is_sun", "is_holiday",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ],
    "공정상태": [
        "is_daytime", "is_startup_08", "is_restart_13", "is_shutdown",
        "shutdown_nth", "is_first_day_back", "prev_day_shutdown", "is_state_switch",
    ],
    "생산정보": ["prod", "headcount", "prod_chg_ratio", "prod_cum_day"],
    "기상정보": ["temp", "humid", "wind", "rain", "cdd", "hdd"],
}
FEATURE_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]


def build_features(d: pd.DataFrame, cal: pd.DataFrame, theta: float, holidays=None) -> pd.DataFrame:
    """전체 피처 파이프라인. 누수 검증에서 재호출하므로 **순수 함수**로 둔다.

    Parameters
    ----------
    d : pandas.DataFrame
        1장에서 정비된 시간별 데이터.
    cal : pandas.DataFrame
        가동 캘린더.
    theta : float
        피크 정의 임계값(이동통계의 피크 발생건수에 사용).
    holidays : array-like of datetime, optional
        공휴일 목록. 기본 `HOLIDAYS_2021` (→ `add_calendar_features`).

    Returns
    -------
    pandas.DataFrame
        원본 + 파생변수. 워밍업 구간은 NaN 으로 남는다.
    """
    out = add_calendar_features(d, cal, holidays)
    out = add_lag_features(out)
    out = add_origin_stats(out, build_origin_frame(d, theta))
    out = add_production_weather_features(out)
    return out


feat = build_features(df, operating_calendar, THETA)

# 워밍업 — 168시간 창과 lag168 이 모두 채워지기 전 구간
feat["is_warmup"] = feat[FEATURE_COLS].isna().any(axis=1)
N_WARMUP = int(feat["is_warmup"].sum())

group_tbl = pd.DataFrame(
    [
        {
            "피처군": g,
            "변수 수": len(cols),
            "주요 변수": ", ".join(FEATURE_LABELS.get(c, c) for c in cols[:4])
            + (" …" if len(cols) > 4 else ""),
        }
        for g, cols in FEATURE_GROUPS.items()
    ]
)
save_table(group_tbl, "ch2_feature_groups")
save_table(
    pd.DataFrame(
        {"피처명": FEATURE_COLS, "한글라벨": [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]}
    ),
    "ch2_feature_labels",
)

print(f"피처 총 {len(FEATURE_COLS)}개 / {len(FEATURE_GROUPS)}개 군")
display(group_tbl)
print(f"\n제외한 원본 중복 컬럼: {EXCLUDE_RAW_COLS} (요일·일·월로 재생성)")
print(f"워밍업 탈락 행: {N_WARMUP}행 ({N_WARMUP / len(feat):.1%}) — 168시간 창 미충족")
print(f"실제 학습 가능 구간 시작: {feat.index[~feat['is_warmup']].min()}")

# %% [markdown]
# #### 그림 F13 — Day-ahead 원점 경계 도식

# %%
def plot_origin_boundary():
    """예측 원점 기준으로 사용 가능/금지 정보를 도식화한다."""
    fig, ax = plt.subplots(figsize=(11, 4.0))
    # D-1, D, D+1 3일 타임라인
    ax.axvspan(-24, 0, color=COLOR_MUTED, alpha=0.18)
    ax.axvspan(0, 24, color=COLOR_HERO, alpha=0.10)
    ax.axvline(0, color=PALETTE_ADJACENT[1], lw=2.4)
    ax.annotate(
        "예측 원점\norigin = D일 24:00\n(D일 24개 시간값 관측 완료)",
        xy=(0, 3.45), ha="center", va="bottom", fontsize=9, color=INK,
        bbox=dict(facecolor="white", edgecolor="#d6d5d1", boxstyle="round,pad=.4"),
    )

    items = [
        (-20, 3.0, "lag168 / lag48 / lag24", COLOR_HERO, "사용 가능"),
        (-20, 2.4, "원점 기준 이동통계 (24h·168h)", COLOR_HERO, "사용 가능"),
        (-20, 1.8, "달력 · 가동캘린더", COLOR_HERO, "사용 가능"),
        (2, 3.0, "ERP 계획생산량 · 공장인원", PALETTE_ADJACENT[2], "가정상 기지"),
        (2, 2.4, "기상 (예보 대리)", PALETTE_ADJACENT[2], "가정상 기지"),
        (2, 1.8, "대상시각 15/30/45/60분 · 평균", PALETTE_ADJACENT[1], "금지 — 누수"),
        (2, 1.2, "rolling().shift(1) · 전체구간 스케일러", PALETTE_ADJACENT[1], "금지 — 누수"),
    ]
    for x, y, text, color, kind in items:
        marker = "[O]" if kind != "금지 — 누수" else "[X]"
        ax.text(x, y, f"{marker}  {text}", fontsize=9, color=INK, va="center")

    ax.set_xlim(-26, 26)
    ax.set_ylim(0.8, 4.45)
    ax.set_xticks([-24, -12, 0, 12, 24])
    ax.set_xticklabels(["D일 00시", "D일 12시", "origin\nD일 24시", "D+1 12시", "D+1 24시"], fontsize=9)
    ax.set_yticks([])
    ax.set_title("Day-ahead 예측 원점과 정보 경계", fontsize=10, color=INK)
    fig.tight_layout()
    src = pd.DataFrame([{"구분": k, "정보": t} for _, _, t, _, k in items])
    return fig, src


_fig, _src = plot_origin_boundary()
save_fig(_fig, "F13", "Day-ahead 원점 경계", "2.2", source_table=_src)

# %% [markdown]
# ### 2.4 시간누수 자동검증
#
# - **목적**: 누수가 없음을 **기계적으로 증명**한다. 하나라도 실패하면 노트북을 중단시킨다.
# - **보고서 대응절**: 1.5, 2.1, 2.3
# - **산출물**: `ch2_leakage_check.csv` (게이트 2)
#
# 두 층위로 검증한다.
#
# **(A) 관측완료 시각 검사** — 각 피처가 참조하는 시각의 **관측완료 시각**
# (= 참조 시각 + 1시간)이 원점 이하인지 본다. "참조 시각 ≤ 원점" 으로 검사하면
# `h=23` 의 `lag24` 경계 케이스를 놓친다.
#
# **(B) 미래맹검 동등성 검사 (핵심)** — 대상일의 **전력 컬럼을 전부 NaN 으로 지우고**
# 피처를 다시 만들어, 원본으로 만든 피처와 **완전히 일치**하는지 본다.
# 타임스탬프 부기와 달리 이 검사는 `rolling().shift(1)` 이나 브로드캐스트 실수를
# **모델과 무관하게 자동 적발**한다. 값이 하나라도 달라지면 그 피처는 대상일의
# 전력을 보고 있었다는 뜻이다.

# %%
ELECTRIC_COLS = ["15분", "30분", "45분", "60분", "평균", "y_avg", "y_peak"]


def check_observation_completion(theta: float) -> pd.DataFrame:
    """(A) 각 피처의 관측완료 시각이 원점 이하인지 검사한다."""
    rows = []
    for lag in SAFE_LAGS:
        # 대상 시각 D+1 h 에 대해 참조 시각은 (D+1 h - lag)
        # 최악의 경우는 h=23 (원점에서 가장 먼 시각)
        for h in range(24):
            ref_offset_h = h - lag  # 원점(D일 24:00=0) 기준 상대 시각
            obs_complete = ref_offset_h + 1  # 관측완료 = 참조시각 + 1시간
            assert obs_complete <= 0, f"lag{lag}, h={h}: 관측완료 {obs_complete}h > 원점"
        rows.append({"피처": f"lag{lag}", "최악 관측완료시각(원점대비)": f"{23 - lag + 1}h", "판정": "안전"})
    for win in (24, 168):
        # 창의 오른쪽 끝 = D일 23:00 → 관측완료 = D일 24:00 = 원점
        rows.append({"피처": f"roll{win} (원점 브로드캐스트)", "최악 관측완료시각(원점대비)": "0h", "판정": "안전"})
    return pd.DataFrame(rows)


def check_future_blind_equivalence(
    d: pd.DataFrame, cal: pd.DataFrame, theta: float, base: pd.DataFrame, n_days: int = 12
) -> pd.DataFrame:
    """(B) 원자료를 원점에서 자르고 정비부터 다시 수행해 피처를 비교한다.

    과거 이력만 정비하고 대상일에는 생산·기상 계획 대리값만 제공한다.
    이 검사는 전력 관측의 시간 경계를 확인하며 계획의 사전 가용성을 증명하지 않는다.

    Parameters
    ----------
    n_days : int
        검사할 대상일 수. 워밍업 이후 구간에서 균등 간격으로 뽑는다.
    """
    usable = d.index.normalize().unique()
    usable = [t for t in usable if t >= d.index.min() + pd.Timedelta(days=8)
              and not bool(d.loc[d.index.normalize() == t, "is_erp_missing"].all())]
    picks = [usable[i] for i in np.linspace(0, len(usable) - 1, n_days).astype(int)]
    # 기존 양방향 보간 문제가 발생했던 원점과 최종 테스트 원점을 반드시 검사한다.
    picks = sorted(set(picks) | {t for t in usable if t >= TEST_START or t == pd.Timestamp("2021-08-29")})
    raw_indexed = build_datetime_index(restore_hours(df_raw))

    rows = []
    for target_day in picks:
        history, _ = flag_and_repair(raw_indexed.loc[raw_indexed.index < target_day])
        history = add_targets(history)
        plan = raw_indexed.loc[raw_indexed.index.normalize() == target_day].copy()
        for column in ELECTRIC_COLS:
            plan[column] = np.nan
        for column in ("is_outage", "is_erp_missing", "is_aug_inconsistent"):
            plan[column] = False
        plan["강수량"] = plan["강수량"].fillna(0.0)
        blinded = pd.concat([history, plan])
        for column in ("풍속", "공장인원"):
            blinded[column] = blinded[column].ffill()
        rebuilt = build_features(blinded, build_operating_calendar(blinded), theta)
        mask = base.index.normalize() == target_day
        a = base.loc[mask, FEATURE_COLS]
        b = rebuilt.loc[rebuilt.index.normalize() == target_day, FEATURE_COLS]

        # NaN 위치까지 포함해 완전 일치해야 한다
        same = a.equals(b)
        if not same:
            diff_cols = [c for c in FEATURE_COLS if not a[c].equals(b[c])]
        else:
            diff_cols = []
        rows.append(
            {
                "대상일": str(target_day.date()),
                "일치": same,
                "불일치 피처": ", ".join(diff_cols[:5]) if diff_cols else "-",
                "정비범위": "원점 이전 원자료부터 재정비",
                "입력가정": "대상일 생산·기상은 사전 계획·예보의 대리",
            }
        )
    return pd.DataFrame(rows)


def gate_chapter2(obs_tbl: pd.DataFrame, blind_tbl: pd.DataFrame) -> pd.DataFrame:
    """게이트 2 — 누수 검증이 전부 통과해야 한다."""
    checks = [
        ("(A) 관측완료시각 ≤ 원점", bool((obs_tbl["판정"] == "안전").all())),
        ("(B) 미래맹검 동등성 100%", bool(blind_tbl["일치"].all())),
        ("대상시각 전력이 피처에 미포함", not any(c in FEATURE_COLS for c in ELECTRIC_COLS)),
        ("원본 중복 컬럼 제외", not any(c in FEATURE_COLS for c in EXCLUDE_RAW_COLS)),
    ]
    out = pd.DataFrame(checks, columns=["점검", "통과"])
    failed = out[~out["통과"]]
    if len(failed):
        raise AssertionError(f"2장 게이트(누수) 실패: {failed['점검'].tolist()}")
    return out


obs_check = check_observation_completion(THETA)
blind_check = check_future_blind_equivalence(df, operating_calendar, THETA, feat)
gate2 = gate_chapter2(obs_check, blind_check)

save_table(obs_check, "ch2_leakage_obs")
save_table(blind_check, "ch2_leakage_blind")
save_table(gate2, "gate2_leakage")

print("── (A) 관측완료 시각 검사 ──")
display(obs_check)
print(f"\n── (B) 미래맹검 동등성 검사 ({len(blind_check)}개 대상일) ──")
display(blind_check)
print("\n── 게이트 2 통과 ──")
display(gate2)
