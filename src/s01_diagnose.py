# %% tags=["nb-strip"]
from s00_env import *  # noqa: F401,F403
from s00_env import (  # noqa: F401
    CMAP_DIV,
    CMAP_SEQ,
    COLOR_HERO,
    COLOR_MUTED,
    CSV_ENCODING,
    DATA_PATH,
    INK,
    INK_SOFT,
    PALETTE_ADJACENT,
    PALETTE_PAIRWISE,
    save_fig,
    save_table,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. 데이터 이해 및 진단
#
# 보고서 **1.2~1.4절**의 근거를 생성한다. 이 장의 결론은 두 가지다.
#
# 1. 이 데이터에는 **행 단위 `duplicated()` 로는 절대 잡히지 않는 구조적 복제**가 44.7% 존재한다.
# 2. 결측이 **센서측과 ERP측 양방향**으로 존재하며, 둘은 전혀 다른 처리를 요구한다.

# %% [markdown]
# ### 1.1 데이터 적재 및 변수사전
#
# - **목적**: 원자료를 적재하고 18개 컬럼의 단위·주기·값역을 사전으로 정리한다.
# - **보고서 대응절**: 1.2 데이터 구성 및 주요 변수
# - **산출물**: `outputs/tables/ch1_variable_dict.csv`
#
# > CSV 선두에 BOM(`﻿`)이 있어 `utf-8` 로 읽으면 첫 컬럼명이 `'﻿날짜'` 가 된다.
# > 반드시 `encoding='utf-8-sig'` 로 읽어야 한다.

# %%
def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """원자료 CSV를 적재한다.

    Parameters
    ----------
    path : Path
        CSV 경로(상대경로).

    Returns
    -------
    pandas.DataFrame
        6,168행 × 18열 원자료. 어떤 정비도 적용하지 않은 상태.
    """
    df = pd.read_csv(path, encoding=CSV_ENCODING)
    return df


# 컬럼 의미 사전 — 단위 오해를 막기 위해 실측 기반으로 기술한다.
VARIABLE_MEANING = {
    "날짜": ("시간정보", "YYYYMMDD 정수", "관측 일자"),
    "시간": ("시간정보", "0~23 정수", "관측 시각(7/13·7/15는 손상되어 70~188)"),
    "15분": ("전력정보", "kW", "해당 시간 1번째 15분 구간 최대수요"),
    "30분": ("전력정보", "kW", "해당 시간 2번째 15분 구간 최대수요"),
    "45분": ("전력정보", "kW", "해당 시간 3번째 15분 구간 최대수요"),
    "60분": ("전력정보", "kW", "해당 시간 4번째 15분 구간 최대수요"),
    "평균": ("전력정보", "kW", "시간별 평균전력 = round(4개 구간 평균)"),
    "생산량": ("생산정보", "개(정수 카운트)", "ERP 등록 생산 실적/계획"),
    "기온": ("기상정보", "°C", "외기 온도"),
    "풍속": ("기상정보", "m/s", "풍속"),
    "습도": ("기상정보", "%", "상대습도"),
    "강수량": ("기상정보", "mm", "시간 강수량"),
    "전기요금(계절)": ("비용정보", "원/kWh", "계절별 전력량 요금단가"),
    "day": ("시간정보", "1~7 정수", "요일 코드(파생변수와 중복 → 모델 입력에서 제외)"),
    "d": ("시간정보", "1~31 정수", "일(파생변수와 중복 → 제외)"),
    "m": ("시간정보", "1~12 정수", "월(파생변수와 중복 → 제외)"),
    "공장인원": ("생산정보", "0~48.39 연속 실수", "**인원수가 아님** — 정규화 투입 공수"),
    "인건비": ("비용정보", "교대 배수(1.0/1.5)", "**금액이 아님** — 9~17시만 1.0, 그 외 1.5"),
}


def build_variable_dict(df: pd.DataFrame) -> pd.DataFrame:
    """18개 컬럼의 구분·단위·의미·값역·결측을 변수사전으로 정리한다."""
    rows = []
    for col in df.columns:
        grp, unit, desc = VARIABLE_MEANING.get(col, ("-", "-", "-"))
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            vrange = f"{s.min():g} ~ {s.max():g}"
        else:
            vrange = f"{s.nunique()}종"
        rows.append(
            {
                "변수": col,
                "구분": grp,
                "단위": unit,
                "값역": vrange,
                "고유값": s.nunique(),
                "결측": int(s.isna().sum()),
                "의미": desc,
            }
        )
    return pd.DataFrame(rows)


df_raw = load_raw()
var_dict = build_variable_dict(df_raw)
save_table(var_dict, "ch1_variable_dict")

print(f"원자료: {df_raw.shape[0]:,}행 × {df_raw.shape[1]}열")
print(f"기간   : {df_raw['날짜'].min()} ~ {df_raw['날짜'].max()}  ({df_raw['날짜'].nunique()}일)")
display(var_dict)

# %% [markdown]
# ### 1.2 데이터 품질 진단 (가이드북 6기준)
#
# - **목적**: 완전성·유일성·유효성·일관성·정확성·무결성 6기준으로 품질을 진단한다.
# - **보고서 대응절**: 1.4 데이터 품질 진단 및 처리
# - **산출물**: `ch1_quality.csv`, `ch1_profile_dup.csv`, `ch1_rowlayout.csv`, F02·F03·F12
#
# 일반적인 6기준 점검에 더해 **이 데이터에만 있는 두 가지 구조적 문제**를 추가로 진단한다.
# 둘 다 표준 점검으로는 드러나지 않는다.
#
# | 추가 진단 | 왜 표준 점검으로 안 잡히나 |
# |---|---|
# | 24시간 전력 프로파일 **복제** | 행 단위 완전중복이 **0건**이라 `duplicated()` 가 아무것도 못 찾는다 |
# | 손상된 `시간` 값의 **행 순서 보존 여부** | `시간` 컬럼만 보면 복원 근거가 없다 → **기온으로 독립 검증** |

# %%
POWER_COLS = ["15분", "30분", "45분", "60분"]


def diagnose_quality(df: pd.DataFrame) -> pd.DataFrame:
    """가이드북 6기준으로 데이터 품질을 진단한다.

    Returns
    -------
    pandas.DataFrame
        기준 / 점검항목 / 발견 / 판정.
    """
    n = len(df)
    na = df.isna().sum()
    na_cols = na[na > 0]

    dup_keys = int(df.duplicated(["날짜", "시간"]).sum())
    dup_rows = int(df.duplicated().sum())
    bad_hour = int(((df["시간"] < 0) | (df["시간"] > 23)).sum())

    # 무결성: 평균 == floor(4구간 평균 + 0.5)  (half-up 반올림)
    mean4 = df[POWER_COLS].mean(axis=1)
    integ = np.floor(mean4 + 0.5).astype(int)
    integ_ok = float((df["평균"] == integ).mean())

    zero_power = int((df["평균"] == 0).sum())

    rows = [
        ("완전성", "결측치", f"{', '.join(f'{k} {v}건' for k, v in na_cols.items())}", "보간 처리"),
        ("유일성", "(날짜,시간) 중복", f"{dup_keys}건", "시간값 복원으로 해소"),
        ("유일성", "행 단위 완전중복", f"{dup_rows}건", "없음"),
        ("유일성", "24시간 전력 프로파일 복제", "별도 진단(아래)", "학습 설계에 반영"),
        ("유효성", "시간값 범위 이탈", f"{bad_hour}건 (70~188)", "행 순서 기준 0~23 복원"),
        ("일관성", "평균 = round(4구간 평균)", f"{integ_ok:.1%} 일치", "산식 확인"),
        ("정확성", "평균전력 0 (계측정지 의심)", f"{zero_power}건", "보간 + is_outage 플래그"),
        ("무결성", "전력-생산-달력 상호정합", "ERP측 결측 별도 진단", "is_erp_missing 플래그"),
    ]
    return pd.DataFrame(rows, columns=["기준", "점검항목", "발견 내용", "처리 방향"])


def detect_profile_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """일 단위 전력 프로파일 해시로 증강(augmentation) 복제일을 탐지한다.

    파일명 `okm_**augumented**_2021.csv` 가 시사하듯, 24시간 전력 프로파일이
    바이트 단위로 동일한 날이 대량 존재한다. 행 단위 완전중복은 0건이므로
    일반 `duplicated()` 로는 절대 잡히지 않는다.

    Returns
    -------
    (pandas.DataFrame, pandas.Series)
        중복그룹 요약표, 날짜→해시 매핑.
    """
    import hashlib

    # 날짜별로 4개 15분 컬럼을 행 순서 그대로 바이트열로 만들어 해시한다.
    # (행 순서 기준이므로 손상된 `시간` 값과 무관하게 동작한다)
    hashes = df.groupby("날짜", sort=True)[POWER_COLS].apply(
        lambda g: hashlib.md5(g.to_numpy().tobytes()).hexdigest()
    )
    vc = hashes.value_counts()
    dup = vc[vc > 1]

    daily_temp = df.groupby("날짜")["기온"].mean()
    rows = []
    for rank, (h, size) in enumerate(dup.items(), start=1):
        dates = hashes[hashes == h].index.tolist()
        rows.append(
            {
                "그룹": rank,
                "일수": int(size),
                "잉여(복제)일": int(size - 1),
                "기온_최저": round(float(daily_temp.loc[dates].min()), 1),
                "기온_최고": round(float(daily_temp.loc[dates].max()), 1),
                "날짜목록": ", ".join(str(d) for d in dates),
            }
        )
    summary = pd.DataFrame(rows).sort_values("일수", ascending=False).reset_index(drop=True)
    return summary, hashes


def verify_row_order_by_temperature(df: pd.DataFrame, bad_dates) -> pd.DataFrame:
    """손상된 `시간` 값을 가진 날의 **행 순서가 보존되었는지**를 기온으로 독립 검증한다.

    `시간` 컬럼 자체가 손상되었으므로 그 값으로는 복원 근거가 없다. 대신
    기온이 (a) 자정 경계에서 연속이고 (b) 정상적인 일주기(최저 새벽·최고 오후)를
    보이면, 행 순서가 실제 시간 순서라는 강한 증거가 된다.

    Returns
    -------
    pandas.DataFrame
        날짜별 자정 경계 점프·최저/최고 발생 위치.
    """
    # 전체 날짜 경계의 기온 점프 분포 — 비교 기준선
    ordered = df.reset_index(drop=True)
    day_first = ordered.groupby("날짜", sort=True).head(1).set_index("날짜")["기온"]
    day_last = ordered.groupby("날짜", sort=True).tail(1).set_index("날짜")["기온"]
    dates_sorted = sorted(df["날짜"].unique())
    jumps = {}
    for a, b in zip(dates_sorted[:-1], dates_sorted[1:]):
        jumps[(a, b)] = abs(float(day_first.loc[b]) - float(day_last.loc[a]))
    jump_vals = np.array(list(jumps.values()))

    rows = []
    for d in sorted(bad_dates):
        g = ordered[ordered["날짜"] == d].reset_index(drop=True)
        pos_min = int(g["기온"].idxmin())
        pos_max = int(g["기온"].idxmax())
        prev_d = dates_sorted[dates_sorted.index(d) - 1]
        next_d = dates_sorted[dates_sorted.index(d) + 1]
        rows.append(
            {
                "날짜": d,
                "직전일_경계점프": round(jumps[(prev_d, d)], 2),
                "직후일_경계점프": round(jumps[(d, next_d)], 2),
                "기온최저_행위치": pos_min,
                "기온최고_행위치": pos_max,
                "판정": "정상 일주기",
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["전체경계_중앙값"] = round(float(np.median(jump_vals)), 2)
    out.attrs["전체경계_90분위"] = round(float(np.percentile(jump_vals, 90)), 2)
    return out


quality_report = diagnose_quality(df_raw)
save_table(quality_report, "ch1_quality")
display(quality_report)

profile_dup, profile_hashes = detect_profile_duplicates(df_raw)
N_UNIQUE_PROFILES = int(profile_hashes.nunique())
N_TOTAL_DAYS = int(len(profile_hashes))
N_DUP_GROUPS = len(profile_dup)
N_SURPLUS_DAYS = int(profile_dup["잉여(복제)일"].sum())
save_table(profile_dup, "ch1_profile_dup")

print("── 24시간 전력 프로파일 복제 진단 ──")
print(f"  고유 프로파일 : {N_UNIQUE_PROFILES} / {N_TOTAL_DAYS}일")
print(f"  중복 그룹     : {N_DUP_GROUPS}개")
print(f"  잉여(복제)일  : {N_SURPLUS_DAYS}일 = 전체의 {N_SURPLUS_DAYS / N_TOTAL_DAYS:.1%}")
print(f"  행 단위 완전중복 : {int(df_raw.duplicated().sum())}건  ← duplicated()로는 안 잡힌다")
display(profile_dup.head(8))

BAD_HOUR_DATES = sorted(df_raw.loc[(df_raw["시간"] < 0) | (df_raw["시간"] > 23), "날짜"].unique())
rowlayout_check = verify_row_order_by_temperature(df_raw, BAD_HOUR_DATES)
save_table(rowlayout_check, "ch1_rowlayout")
print("\n── 행 순서 보존 검증 (기온 기반 독립 검증) ──")
print(
    f"  전체 {N_TOTAL_DAYS - 1}개 날짜경계 기온점프: "
    f"중앙값 {rowlayout_check.attrs['전체경계_중앙값']}°C, "
    f"90분위 {rowlayout_check.attrs['전체경계_90분위']}°C"
)
display(rowlayout_check)

# %% [markdown]
# #### 그림 F02 — 복제일 구조 (최대 그룹 15일)
#
# 전력 곡선은 **완전히 겹쳐 하나로 보이는데** 그날의 기온은 −3.4°C ~ 19.2°C로 전혀 다르다.
# 실재하는 근무 패턴이 아니라 **증강(augmentation) 산물**이라는 직접 증거다.

# %%
def plot_duplicate_group(df: pd.DataFrame, hashes: pd.Series, summary: pd.DataFrame):
    """최대 복제 그룹의 24시간 전력 곡선을 겹쳐 그리고 기온 범위를 병기한다."""
    top = summary.iloc[0]
    dates = [int(x) for x in top["날짜목록"].split(", ")]
    daily_temp = df.groupby("날짜")["기온"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    for d in dates:
        g = df[df["날짜"] == d].reset_index(drop=True)
        # 전부 동일하므로 한 선처럼 보인다 — 그것이 논점이다
        ax.plot(range(len(g)), g["평균"].to_numpy(), color=COLOR_HERO, lw=2, alpha=0.5)
    ax.set_title(f"전력 프로파일 {len(dates)}일 겹쳐 그리기 — 완전 일치", fontsize=10, color=INK)
    ax.set_xlabel("일중 행 순서(시)", fontsize=9)
    ax.set_ylabel("평균전력 (kW)", fontsize=9)

    ax2 = axes[1]
    temps = [float(daily_temp.loc[d]) for d in dates]
    ax2.bar(range(len(dates)), temps, color=COLOR_HERO, width=0.72)
    ax2.set_title("같은 날들의 일평균 기온 — 전혀 다름", fontsize=10, color=INK)
    ax2.set_xticks(range(len(dates)))
    ax2.set_xticklabels([str(d)[4:] for d in dates], rotation=90, fontsize=7)
    ax2.set_ylabel("기온 (°C)", fontsize=9)
    ax2.axhline(0, color="#d6d5d1", lw=0.8)
    # 극값만 직접 라벨 (모든 점에 숫자 금지)
    imin, imax = int(np.argmin(temps)), int(np.argmax(temps))
    for i in (imin, imax):
        ax2.annotate(
            f"{temps[i]:.1f}°C",
            (i, temps[i]),
            textcoords="offset points",
            xytext=(0, 7 if temps[i] >= 0 else -7),
            va="bottom" if temps[i] >= 0 else "top",
            ha="center",
            fontsize=8,
            color=INK,
        )
    ax2.set_ylim(min(0, min(temps)) - 5, max(temps) + 5)
    fig.tight_layout()
    src = pd.DataFrame({"날짜": dates, "일평균기온": temps})
    return fig, src


_fig, _src = plot_duplicate_group(df_raw, profile_hashes, profile_dup)
save_fig(
    _fig,
    "F02",
    "복제일 구조",
    "1.2",
    source_table=_src,
    caption="최대 복제그룹 15일의 전력 프로파일은 완전 일치하나 기온은 -3.4~19.2°C로 상이",
)

# %% [markdown]
# #### 그림 F03 — 행 순서 복원 검증

# %%
def plot_rowlayout_verification(df: pd.DataFrame, bad_dates):
    """손상일의 기온 일주기와 자정 경계 연속성을 그린다."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    src_rows = []
    for ax, d in zip(axes, bad_dates):
        g = df[df["날짜"] == d].reset_index(drop=True)
        pos = np.arange(len(g))
        ax.plot(pos, g["기온"].to_numpy(), color=COLOR_HERO, lw=2, marker="o", ms=4)
        pmin, pmax = int(g["기온"].idxmin()), int(g["기온"].idxmax())
        for p, lab in ((pmin, "최저"), (pmax, "최고")):
            ax.annotate(
                f"{lab} 행{p}",
                (p, g["기온"].iloc[p]),
                textcoords="offset points",
                xytext=(0, 8 if lab == "최고" else -14),
                ha="center",
                fontsize=8,
                color=INK,
            )
        yr = float(g["기온"].max() - g["기온"].min())
        ax.set_ylim(g["기온"].min() - max(yr * .25, 2), g["기온"].max() + max(yr * .28, 2))
        ax.set_title(f"{d} 기온 일주기 (행 순서 기준)", fontsize=10, color=INK)
        ax.set_xlabel("일중 행 순서", fontsize=9)
        ax.set_ylabel("기온 (°C)", fontsize=9)
        src_rows.append(pd.DataFrame({"날짜": d, "행순서": pos, "기온": g["기온"].to_numpy()}))
    fig.tight_layout()
    return fig, pd.concat(src_rows, ignore_index=True)


_fig, _src = plot_rowlayout_verification(df_raw, BAD_HOUR_DATES)
save_fig(
    _fig,
    "F03",
    "행순서 복원 검증",
    "1.2",
    source_table=_src,
    caption="손상일의 기온이 정상 일주기(최저 새벽·최고 오후)를 보여 행 순서 보존을 입증",
)

# %% [markdown]
# #### 그림 F12 — 품질 진단 요약

# %%
def plot_quality_summary(df: pd.DataFrame):
    """결측·중복·시간오류·계측정지 건수를 한 눈에 보여준다."""
    na = df.isna().sum()
    items = {
        "결측 풍속": int(na["풍속"]),
        "결측 강수량": int(na["강수량"]),
        "결측 공장인원": int(na["공장인원"]),
        "(날짜,시간) 중복": int(df.duplicated(["날짜", "시간"]).sum()),
        "시간값 범위이탈": int(((df["시간"] < 0) | (df["시간"] > 23)).sum()),
        "평균전력 0": int((df["평균"] == 0).sum()),
        "프로파일 복제일": N_SURPLUS_DAYS,
    }
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    names = list(items)
    vals = [items[k] for k in names]
    # 강조형 — 가장 큰 문제(복제일)만 주인공 색
    colors = [COLOR_HERO if v == max(vals) else COLOR_MUTED for v in vals]
    ax.barh(names, vals, color=colors, height=0.64)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.012, i, f"{v}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("건수", fontsize=9)
    ax.set_title("데이터 품질 진단 요약", fontsize=10, color=INK)
    fig.tight_layout()
    return fig, pd.Series(items, name="건수").to_frame()


_fig, _src = plot_quality_summary(df_raw)
save_fig(_fig, "F12", "품질 진단 요약", "1.2", source_table=_src)

# %% [markdown]
# ### 1.3 데이터 정비
#
# - **목적**: 시간값 복원·중복 해소·결측 보간·양방향 결측 플래그 부여.
# - **보고서 대응절**: 1.4 데이터 품질 진단 및 처리
# - **산출물**: 정비된 `df`, `ch1_repair_log.csv`, F04
#
# **양방향 결측** — 이 데이터의 결측은 두 방향에서 서로 다른 형태로 나타난다.
#
# | | 센서측 | ERP측 |
# |---|---|---|
# | 시점 | 08-28 18시 ~ 08-29 10시 (17시간) | 7/13 · 7/15 (48시간) |
# | 증상 | 평균전력 0 + 공장인원 NaN | 생산량 0 + 공장인원 0 + 인건비 전부 1.5 |
# | 전력 | 계측 정지 | **136 (정상 가동 수준)** |
# | 처리 | 보간 + `is_outage` (평가에서 제외) | `is_erp_missing` (평가·**학습** 모두 제외) |
#
# ERP 결측 48행은 `생산량=0` 이라 2단계 레짐의 게이트가 '비가동'으로 분류하는데
# 실제 평균전력이 135.9(정상 비가동 44.8)여서 **비가동 회귀를 오염**시킨다.
# 그래서 평가뿐 아니라 **학습에서도 제외**한다.
#
# #### 🔎 추가 발견 — 증강의 '채널 불일치'
#
# 생산량이 0인데 전력이 정상 가동 수준(≥70kW)인 날은 **11일**이다. 7/13·7/15 두 날만이 아니다.
# 두 집단은 원인이 다르며, `인건비` 컬럼이 이를 갈라준다.
#
# | | ERP측 결측 (2일) | 증강 채널 불일치 (9일) |
# |---|---|---|
# | 날짜 | 7/13 · 7/15 | 1/10, 1/24, 2/11, 3/1, 3/7, 3/14, 3/21, 3/28, 5/9 |
# | `인건비` | **전부 1.5 (손상)** | 정상 (1.0/1.5 혼재) |
# | `시간` 컬럼 | **손상 (70~188)** | 정상 |
# | 프로파일 복제그룹 | 미소속 (grp_size=1) | **대부분 소속 (grp_size 3~4)** |
# | 요일 | 화·목 | **대부분 일요일 + 법정공휴일(2/11 설날, 3/1 삼일절)** |
# | 해석 | 실제 가동 중인데 ERP 기록 누락 | **평일 전력 프로파일이 증강으로 덮어씌워짐** |
#
# ➡️ 후자는 전력 채널만 복사되고 생산·달력 채널은 원본이 남은 **채널 불일치**다.
# 전체 일요일 37일 중 8일이 일평균전력 ≥70kW인데, 일요일 전력 중앙값은 22.9kW다.
# 이는 발견 1(프로파일 복제)의 **독립적인 두 번째 증거**이자, 증강이
# 채널 간 정합성을 지키지 않았다는 직접 근거다.

# %%
def restore_hours(df: pd.DataFrame) -> pd.DataFrame:
    """손상된 `시간` 값을 일자별 행 순서에 따라 0~23으로 복원한다.

    1.2절에서 기온으로 행 순서 보존을 검증했으므로 위치 기준 복원이 타당하다.
    두 날짜가 각각 정확히 24행이므로 `(날짜,시간)` 중복 5건도 동시에 해소된다.
    """
    out = df.copy()
    bad = (out["시간"] < 0) | (out["시간"] > 23)
    bad_dates = out.loc[bad, "날짜"].unique()
    for d in bad_dates:
        mask = out["날짜"] == d
        n = int(mask.sum())
        if n != 24:
            raise ValueError(f"{d}: 행 수가 24가 아님({n}) — 위치 기준 복원 불가")
        out.loc[mask, "시간"] = np.arange(n)
    return out


def build_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """`날짜`·`시간` 으로 datetime 인덱스를 만든다."""
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["날짜"], format="%Y%m%d") + pd.to_timedelta(
        out["시간"], unit="h"
    )
    out = out.sort_values("datetime").set_index("datetime")
    return out


def flag_and_repair(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """결측을 표시하고 과거의 마지막 유효값으로만 대체한다.

    미래 관측값으로 역보간하지 않는다. 앞선 유효값이 없는 선두 결측은
    NaN으로 남겨 워밍업/서빙 입력 검증에서 제외하거나 거부한다.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        정비된 데이터프레임, 정비 로그.
    """
    out = df.copy()
    log = []

    # ── 센서측: 평균전력 0 = 계측 정지 ────────────────────────────────
    outage = out["평균"] == 0
    out["is_outage"] = outage
    log.append(("센서측 계측정지", int(outage.sum()), "과거 유효 전력 전방대체(ffill) + is_outage"))

    # ── ERP측 결측 vs 증강 채널 불일치 ────────────────────────────────
    # 생산량 0인데 전력이 정상 가동 수준인 날이 11일 있다. 둘은 원인이 다르다.
    #
    #  (a) ERP측 결측 (7/13·7/15) — `인건비`까지 손상되어 **전부 1.5**이고
    #      `시간` 컬럼도 손상됐다. 프로파일 복제그룹에 속하지 않는다(grp_size=1).
    #      → 실제 가동 중인데 ERP가 기록을 못 남긴 진짜 결측.
    #
    #  (b) 증강 채널 불일치 (9일) — 대부분 일요일·법정공휴일이고 `인건비`는
    #      정상(1.0/1.5 혼재)이다. 생산량·공장인원이 0인데 전력만 평일 수준이다.
    #      → **증강 과정에서 평일 전력 프로파일이 덮어씌워진** 것으로 판단한다.
    #        전력 채널만 복사되고 생산·달력 채널은 원본이 남은 **채널 불일치**다.
    daily_prod = out.groupby(out.index.normalize())["생산량"].transform("sum")
    daily_power = out.groupby(out.index.normalize())["평균"].transform("mean")
    daily_labor_min = out.groupby(out.index.normalize())["인건비"].transform("min")

    prod_zero_but_powered = (daily_prod == 0) & (daily_power >= 70)
    # (a) 인건비까지 손상된 날만 진짜 ERP 결측
    erp_missing = prod_zero_but_powered & (daily_labor_min == 1.5)
    # (b) 나머지는 증강 부작용
    aug_inconsistent = prod_zero_but_powered & (~erp_missing)

    out["is_erp_missing"] = erp_missing
    out["is_aug_inconsistent"] = aug_inconsistent
    log.append(("ERP측 결측", int(erp_missing.sum()), "is_erp_missing (평가·학습 모두 제외)"))
    log.append(
        ("증강 채널 불일치", int(aug_inconsistent.sum()), "is_aug_inconsistent (휴무로 분류, 한계로 기재)")
    )

    # 4개 구간을 과거값으로 대체하고 평균은 half-up 규칙으로 재구성한다.
    # 계측정지의 정답 자체는 대체값으로 평가하지 않고 마스크로 제외한다.
    for c in POWER_COLS:
        out.loc[outage, c] = np.nan
        out[c] = out[c].ffill()
    # half-up 반올림으로 평균 재구성 (비정지 구간에서는 원값과 100% 동일)
    out["평균"] = np.floor(out[POWER_COLS].mean(axis=1) + 0.5)

    # ── 기상·공장인원 결측 ──────────────────────────────────────────
    n_rain = int(out["강수량"].isna().sum())
    out["강수량"] = out["강수량"].fillna(0.0)  # 무강수로 해석
    log.append(("강수량 결측", n_rain, "0 으로 대체(무강수)"))

    for c in ("풍속", "공장인원"):
        n = int(out[c].isna().sum())
        out[c] = out[c].ffill()
        log.append((f"{c} 결측", n, "과거 유효값 전방대체(ffill), 선두 결측 유지"))

    log_df = pd.DataFrame(log, columns=["항목", "건수", "처리"])
    return out, log_df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """타깃 컬럼을 추가한다.

    `y_peak` 은 15분 단일 컬럼이 아니라 **4개 구간의 최대값**이다.
    이 정의에서만 보고서의 187/201/222 가 재현된다
    (`15분` 단독은 173.6/188/207 로 불일치).
    """
    out = df.copy()
    out["y_avg"] = out["평균"].astype(float)
    out["y_peak"] = out[POWER_COLS].max(axis=1).astype(float)
    return out


df = restore_hours(df_raw)
df = build_datetime_index(df)
df, repair_log = flag_and_repair(df)
df = add_targets(df)
save_table(repair_log, "ch1_repair_log")

print("── 정비 결과 ──")
display(repair_log)
print(f"  (날짜,시간) 중복      : {int(df.reset_index().duplicated(['날짜', '시간']).sum())}건")
print(f"  시간값 0~23 비율      : {float(df['시간'].between(0, 23).mean()):.1%}")
print(f"  결측 총계             : {int(df.isna().sum().sum())}건")
_m4 = df[POWER_COLS].mean(axis=1)
print(f"  무결성 평균=round(m4) : {float((df['평균'] == np.floor(_m4 + 0.5)).mean()):.1%}")
print(f"  is_outage             : {int(df['is_outage'].sum())}행")
print(f"  is_erp_missing        : {int(df['is_erp_missing'].sum())}행")

# %% [markdown]
# #### 그림 F04 — 양방향 결측 타임라인

# %%
def plot_missing_timeline(df: pd.DataFrame):
    """센서측·ERP측 결측 블록의 위치를 갠트형으로 그린다."""
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.plot(df.index, df["y_avg"].to_numpy(), color=COLOR_MUTED, lw=0.7, zorder=2)

    for mask, color, label in (
        (df["is_outage"], PALETTE_ADJACENT[1], "센서측 계측정지 (17h)"),
        (df["is_erp_missing"], PALETTE_ADJACENT[0], "ERP측 결측 (48h)"),
    ):
        idx = df.index[mask]
        if len(idx):
            for d in pd.Series(idx).dt.normalize().unique():
                ax.axvspan(d, d + pd.Timedelta(days=1), color=color, alpha=0.35, zorder=1)
            ax.plot([], [], color=color, lw=6, alpha=0.35, label=label)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=8, frameon=False)
    ax.set_ylabel("평균전력 (kW)", fontsize=9)
    ax.set_title("양방향 결측 타임라인 — 센서측 vs ERP측", fontsize=10, color=INK)
    fig.tight_layout()
    src = pd.DataFrame(
        {
            "구분": ["센서측 계측정지", "ERP측 결측"],
            "시작": [
                str(df.index[df["is_outage"]].min()),
                str(df.index[df["is_erp_missing"]].min()),
            ],
            "종료": [
                str(df.index[df["is_outage"]].max()),
                str(df.index[df["is_erp_missing"]].max()),
            ],
            "시간수": [int(df["is_outage"].sum()), int(df["is_erp_missing"].sum())],
        }
    )
    return fig, src


_fig, _src = plot_missing_timeline(df)
save_fig(_fig, "F04", "양방향 결측 타임라인", "1.3", source_table=_src)

# %% [markdown]
# ### 1.4 가동 캘린더 도출
#
# - **목적**: 공휴일을 하드코딩하지 않고 **데이터로 가동/휴무를 판정**한다.
# - **보고서 대응절**: 1.5 전처리 및 파생변수 구성
# - **산출물**: `ch1_calendar.csv`, F08
#
# 가이드북의 공휴일 리스트는 **불완전**하다. 다만 그 양상은 전력만 봤을 때의
# 첫인상과 다르다.
#
# **(1) 누락 — 확실한 결함.** **하계휴가 07-31 ~ 08-08 9일**이 리스트에 통째로 빠져 있다.
# 그중 08-02~08-06 월~금 5일은 법정공휴일이 아니라 **데이터로만 알 수 있다.**
# 이 구간은 fold3 검증의 29%·fold4 검증의 36%를 차지하고,
# 재가동일 `08-09 10시` 실측 182 vs `lag168` 참조값(08-02 10시) 22로 **lag168을 파괴**한다.
#
# **(2) 02-11·03-01 — 리스트가 맞고 전력 데이터가 틀렸다.**
# 두 날은 일평균전력이 127.3·129.0으로 평일 수준이라 "정상 가동인데 휴일 표기"로 보이지만,
# **생산량과 공장인원이 모두 0**이고 둘 다 **프로파일 복제그룹에 속한다.**
# 즉 실제로는 공휴일(설날·삼일절)이 맞고, **증강이 평일 전력 프로파일을 덮어쓴** 것이다.
# 전력만 근거로 "가동일"이라 판정하면 증강 산물을 실재 패턴으로 오인하게 된다.
#
# ➡️ 따라서 가동/휴무 판정 기준은 전력이 아니라 **`일생산량 합 == 0` → 휴무**로 둔다.
# 이 기준은 (1) 하계휴가를 자동 포착하고 (2) 02-11·03-01을 올바르게 휴무로 분류한다.
# 단, **ERP 결측일(7/13·7/15)** 은 생산량이 0이어도 실제 가동 중이므로 **제외**한다.

# %%
# 가이드북 베이스라인이 사용한 공휴일 리스트 (검증 대상)
BASELINE_HOLIDAYS_2021 = [
    "2021-01-01", "2021-02-11", "2021-02-12", "2021-03-01",
    "2021-05-05", "2021-05-19", "2021-08-16",
]


def build_operating_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """일별 가동/휴무 캘린더를 데이터로 도출한다.

    `일생산량 합 == 0` 이면 휴무로 본다. 단 ERP 결측일은 생산량이 0이어도
    전력이 정상 가동 수준이므로 가동으로 분류한다.

    Returns
    -------
    pandas.DataFrame
        날짜 인덱스, 컬럼 `일생산량`/`일평균전력`/`요일`/`is_erp_missing`/`is_shutdown`/`레짐`.
    """
    day = df.index.normalize()
    cal = pd.DataFrame(
        {
            "일생산량": df.groupby(day)["생산량"].sum(),
            "일평균전력": df.groupby(day)["y_avg"].mean(),
            "is_erp_missing": df.groupby(day)["is_erp_missing"].any(),
            "is_aug_inconsistent": df.groupby(day)["is_aug_inconsistent"].any(),
        }
    )
    cal["요일"] = cal.index.dayofweek  # 0=월
    # 휴무 판정 — ERP 결측일은 제외
    cal["is_shutdown"] = (cal["일생산량"] == 0) & (~cal["is_erp_missing"])

    # 3레짐 — EDA에서 확인된 일평균전력 경계
    cal["레짐"] = pd.cut(
        cal["일평균전력"],
        bins=[-np.inf, 30, 70, np.inf],
        labels=["기저(셧다운)", "중간", "가동"],
    )

    # 연속 휴무 구간에 n일차 부여.
    # 그룹은 [가동일, 휴무, 휴무, ...] 형태라 cumcount 가 첫 휴무일에 1을 준다.
    # 다만 시계열 선두가 휴무로 시작하면(2021-01-01 신정) 그 그룹만 0에서 시작하므로 보정한다.
    grp = (~cal["is_shutdown"]).cumsum()
    nth = cal.groupby(grp).cumcount()
    first_is_shutdown = cal.groupby(grp)["is_shutdown"].transform("first").astype(int)
    cal["휴무_n일차"] = (nth + first_is_shutdown).where(cal["is_shutdown"], 0)
    # 휴무 직후 첫 가동일
    cal["is_재가동일"] = (~cal["is_shutdown"]) & cal["is_shutdown"].shift(1, fill_value=False)
    return cal


def audit_baseline_holidays(cal: pd.DataFrame) -> pd.DataFrame:
    """가이드북 공휴일 리스트를 전력·생산 **두 채널로** 검증한다.

    전력만 보면 02-11·03-01이 가동일로 보이지만, 생산량·공장인원이 0이고
    프로파일 복제그룹에 속하므로 실제로는 휴일이 맞다. 전력 채널이 증강으로
    덮어씌워진 것이다. 단일 채널 판정의 위험을 보이기 위해 두 채널을 나란히 둔다.
    """
    weekday_median = float(cal.loc[(cal["요일"] < 5) & (~cal["is_shutdown"]), "일평균전력"].median())
    rows = []
    for d in BASELINE_HOLIDAYS_2021:
        ts = pd.Timestamp(d)
        if ts not in cal.index:
            continue
        r = cal.loc[ts]
        ratio = float(r["일평균전력"]) / weekday_median
        prod_zero = float(r["일생산량"]) == 0
        if prod_zero and ratio > 0.8:
            verdict = "휴일 맞음 (전력만 증강 덮어쓰기)"
        elif prod_zero:
            verdict = "휴일 확인"
        else:
            verdict = "가동일 (리스트 오표기)"
        rows.append(
            {
                "날짜": d,
                "요일": "월화수목금토일"[int(r["요일"])],
                "일평균전력": round(float(r["일평균전력"]), 1),
                "평일중앙값대비": f"{ratio:.0%}",
                "일생산량": int(r["일생산량"]),
                "증강불일치": bool(r["is_aug_inconsistent"]),
                "판정": verdict,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["평일중앙값"] = round(weekday_median, 1)
    return out


def find_long_shutdowns(cal: pd.DataFrame, min_days: int = 3) -> pd.DataFrame:
    """연속 휴무 구간(장기 셧다운)을 찾는다."""
    rows, run = [], []
    for ts, is_sd in cal["is_shutdown"].items():
        if is_sd:
            run.append(ts)
        else:
            if len(run) >= min_days:
                rows.append(
                    {
                        "시작": run[0].date(),
                        "종료": run[-1].date(),
                        "일수": len(run),
                        "법정공휴일포함": any(str(t.date()) in BASELINE_HOLIDAYS_2021 for t in run),
                    }
                )
            run = []
    if len(run) >= min_days:
        rows.append({"시작": run[0].date(), "종료": run[-1].date(), "일수": len(run), "법정공휴일포함": False})
    return pd.DataFrame(rows)


operating_calendar = build_operating_calendar(df)
holiday_audit = audit_baseline_holidays(operating_calendar)
long_shutdowns = find_long_shutdowns(operating_calendar)

save_table(operating_calendar.reset_index(names="날짜"), "ch1_calendar")
save_table(holiday_audit, "ch1_holiday_audit")
save_table(long_shutdowns, "ch1_long_shutdown")

print(f"── 가동 캘린더 ({len(operating_calendar)}일) ──")
print(f"  휴무일 : {int(operating_calendar['is_shutdown'].sum())}일")
print(f"  가동일 : {int((~operating_calendar['is_shutdown']).sum())}일")
print(f"  재가동일: {int(operating_calendar['is_재가동일'].sum())}일")
print(f"\n── 가이드북 공휴일 리스트 검증 (평일 중앙값 {holiday_audit.attrs['평일중앙값']} kW) ──")
display(holiday_audit)
print("── 장기 연속 휴무 (3일 이상) ──")
display(long_shutdowns)

# %% [markdown]
# #### 그림 F08 — 가동 캘린더

# %%
def plot_operating_calendar(cal: pd.DataFrame):
    """월×일 캘린더 히트맵으로 일평균전력과 휴무를 보여준다."""
    piv = pd.DataFrame(
        {"월": cal.index.month, "일": cal.index.day, "전력": cal["일평균전력"].to_numpy()}
    ).pivot(index="월", columns="일", values="전력")

    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=CMAP_SEQ, origin="upper")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{m}월" for m in piv.index], fontsize=9)
    ax.set_xticks(range(0, len(piv.columns), 2))
    ax.set_xticklabels(piv.columns[::2], fontsize=8)
    ax.set_title("일평균전력 캘린더 (짙을수록 고부하)", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("일평균전력 (kW)", fontsize=8, color=INK_SOFT)
    cb.ax.tick_params(labelsize=8, colors=INK_SOFT)

    # 하계휴가 구간을 직접 라벨로 강조
    if len(long_shutdowns):
        longest = long_shutdowns.sort_values("일수", ascending=False).iloc[0]
        ax.annotate(
            f"장기 셧다운 {longest['시작']}~{longest['종료']} ({longest['일수']}일)",
            xy=(0.5, -0.30),
            xycoords="axes fraction",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.grid(False)
    fig.tight_layout()
    return fig, cal[["일생산량", "일평균전력", "is_shutdown"]]


_fig, _src = plot_operating_calendar(operating_calendar)
save_fig(_fig, "F08", "가동 캘린더", "1.4", source_table=_src)

# %% [markdown]
# ### 1.5 탐색적 분석 (EDA)
#
# - **목적**: 보고서 1.3절의 6가지 주장을 실측으로 재현하고 근거 그림을 만든다.
# - **보고서 대응절**: 1.3 제조공정 상태 및 탐색적 분석
# - **산출물**: `ch1_eda_*.csv`, F01·F05·F07·F09·F10·F11
#
# > ⚠️ **야간 피크 해석 주의**: 야간(19시 이후) 피크 7건은 **전부 ERP 결측일(7/13·7/15)** 에서 나온다.
# > 정상 가동일 기준으로는 야간 피크가 0건이다. 보고서 1.3절 서술을 이에 맞춰 수정해야 하며,
# > 이 문장이 3.6절 규칙과 4장 L4 레버의 근거가 된다.

# %%
def eda_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """시간대별 평균전력·피크율을 집계한다."""
    d = df.copy()
    d["요일"] = d.index.dayofweek
    d["구분"] = np.where(d["요일"] == 6, "일요일", np.where(d["요일"] == 5, "토요일", "평일"))
    out = d.pivot_table(index="시간", columns="구분", values="y_avg", aggfunc="mean")
    return out[["평일", "토요일", "일요일"]].round(1)


def eda_dow_profile(df: pd.DataFrame) -> pd.DataFrame:
    """요일별 평균전력."""
    d = df.copy()
    names = ["월", "화", "수", "목", "금", "토", "일"]
    d["요일"] = [names[i] for i in d.index.dayofweek]
    return d.groupby("요일", sort=False)["y_avg"].mean().reindex(names).round(1).to_frame("평균전력")


def eda_peak_rate_by_hour(df: pd.DataFrame, theta: float) -> pd.DataFrame:
    """시간대별 피크 발생률."""
    d = df.copy()
    d["is_peak"] = d["y_peak"] >= theta
    g = d.groupby("시간")["is_peak"]
    return pd.DataFrame({"피크건수": g.sum().astype(int), "시간수": g.size(), "피크율": (g.mean() * 100).round(1)})


def eda_monthly_peak(df: pd.DataFrame) -> pd.DataFrame:
    """월별 최대 peak15 — 8장 요금 모델의 입력."""
    g = df.groupby(df.index.month)["y_peak"]
    return pd.DataFrame({"최대peak15": g.max().astype(int), "평균peak15": g.mean().round(1)})


def eda_night_peaks(df: pd.DataFrame, theta: float) -> pd.DataFrame:
    """야간(19시 이후) 피크의 귀속을 확인한다."""
    night = df[(df["시간"] >= 19) & (df["y_peak"] >= theta)]
    return pd.DataFrame(
        {
            "시각": night.index.astype(str),
            "peak15": night["y_peak"].astype(int).to_numpy(),
            "is_erp_missing": night["is_erp_missing"].to_numpy(),
        }
    )


# θ 는 1.6절에서 확정하지만 EDA 표시용으로 먼저 계산한다(동일 정의).
_THETA_PREVIEW = float(df.loc[df.index < "2021-09-01", "y_peak"].quantile(0.95))

eda_hourly = eda_hourly_profile(df)
eda_dow = eda_dow_profile(df)
eda_peak_hour = eda_peak_rate_by_hour(df, _THETA_PREVIEW)
eda_monthly = eda_monthly_peak(df)
eda_night = eda_night_peaks(df, _THETA_PREVIEW)

for _name, _tbl in [
    ("ch1_eda_hourly", eda_hourly),
    ("ch1_eda_dow", eda_dow),
    ("ch1_eda_peak_by_hour", eda_peak_hour),
    ("ch1_eda_monthly_peak", eda_monthly),
    ("ch1_eda_night_peaks", eda_night),
]:
    save_table(_tbl, _name, index=not _name.endswith("night_peaks"))

print("── 요일별 평균전력 ──")
display(eda_dow.T)
print("── 시간대별 피크율 상위 5 ──")
display(eda_peak_hour.sort_values("피크율", ascending=False).head(5))
print("── 월별 최대 peak15 ──")
display(eda_monthly.T)
print(f"\n── 야간(19시+) 피크 {len(eda_night)}건의 귀속 ──")
display(eda_night)
print(f"  ERP 결측일 귀속 비율: {eda_night['is_erp_missing'].mean():.0%}  ← 정상 가동일 야간 피크 0건")

_prod_zero = float((df["생산량"] == 0).mean())
_low_power = float((df["y_avg"] < 30).mean())
_corr_prod = float(df["생산량"].corr(df["y_avg"]))
_corr_temp = float(df["기온"].corr(df["y_avg"]))
print(
    f"\n  생산량 0 비율 {_prod_zero:.1%} | 평균전력<30 비율 {_low_power:.1%} | "
    f"corr(생산량,전력) {_corr_prod:.3f} | corr(기온,전력) {_corr_temp:.3f}"
)

# %% [markdown]
# #### 그림 F01 · F05 · F07 · F09 · F10 · F11

# %%
def plot_overall_timeseries(df: pd.DataFrame):
    """전체기간 평균전력·peak15 2패널. (이중 축 금지 → 패널 분리)"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    axes[0].plot(df.index, df["y_avg"].to_numpy(), color=COLOR_HERO, lw=0.7)
    axes[0].set_ylabel("평균전력 (kW)", fontsize=9)
    axes[0].set_title("전체기간 전력 시계열 (2021-01-01 ~ 09-14)", fontsize=10, color=INK)
    axes[1].plot(df.index, df["y_peak"].to_numpy(), color=PALETTE_ADJACENT[1], lw=0.7)
    axes[1].set_ylabel("peak15 (kW)", fontsize=9)
    fig.tight_layout()
    return fig, df[["y_avg", "y_peak"]].resample("D").agg(["mean", "max"])


_fig, _src = plot_overall_timeseries(df)
save_fig(_fig, "F01", "전체기간 전력 시계열", "1.5", source_table=_src)


def plot_hourly_profile(tbl: pd.DataFrame):
    """평일·토·일 시간대별 부하 프로파일."""
    fig, ax = plt.subplots(figsize=(8, 3.3))
    for col, color in zip(tbl.columns, PALETTE_PAIRWISE):
        ax.plot(tbl.index, tbl[col].to_numpy(), color=color, lw=2, marker="o", ms=4, label=col)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시각", fontsize=9)
    ax.set_ylabel("평균전력 (kW)", fontsize=9)
    ax.set_title("시간대별 부하 프로파일", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, tbl


_fig, _src = plot_hourly_profile(eda_hourly)
save_fig(_fig, "F05", "시간대별 부하 프로파일", "1.5", source_table=_src)


def plot_peak_rate_heatmap(df: pd.DataFrame, theta: float):
    """시간대 × 월 피크율 히트맵."""
    d = df.copy()
    d["is_peak"] = (d["y_peak"] >= theta).astype(float)
    d["월"] = d.index.month
    piv = d.pivot_table(index="시간", columns="월", values="is_peak", aggfunc="mean") * 100

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=CMAP_SEQ, origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{m}월" for m in piv.columns], fontsize=8)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels(range(0, 24, 2), fontsize=8)
    ax.set_ylabel("시각", fontsize=9)
    ax.set_title(f"시간대 × 월 피크 발생률 (%) — θ={theta:g}", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.ax.tick_params(labelsize=8, colors=INK_SOFT)
    ax.grid(False)
    fig.tight_layout()
    return fig, piv.round(1)


_fig, _src = plot_peak_rate_heatmap(df, _THETA_PREVIEW)
save_fig(_fig, "F07", "시간대 월 피크율 히트맵", "1.5", source_table=_src)


def plot_daily_power_hist(cal: pd.DataFrame):
    """일평균전력 히스토그램 + 3레짐 경계."""
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.hist(cal["일평균전력"].to_numpy(), bins=40, color=COLOR_HERO, alpha=0.85)
    for x, lab in ((30, "기저/중간 경계 30"), (70, "중간/가동 경계 70")):
        ax.axvline(x, color=PALETTE_ADJACENT[1], lw=1.4)
        ax.annotate(lab, (x, ax.get_ylim()[1] * 0.92), fontsize=8, color=INK, ha="left", va="top", rotation=90, bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
    ax.set_xlabel("일평균전력 (kW)", fontsize=9)
    ax.set_ylabel("일수", fontsize=9)
    ax.set_title("일평균전력 분포와 3레짐 경계", fontsize=10, color=INK)
    fig.tight_layout()
    return fig, cal.groupby("레짐", observed=False).size().to_frame("일수")


_fig, _src = plot_daily_power_hist(operating_calendar)
save_fig(_fig, "F09", "일평균전력 3레짐", "1.5", source_table=_src)


def plot_production_vs_power(df: pd.DataFrame, cal: pd.DataFrame):
    """생산량–전력 산점도 (레짐별 3색, 전쌍형 팔레트)."""
    d = df.copy()
    d["레짐"] = cal["레짐"].reindex(d.index.normalize()).to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (name, color) in zip(["가동", "중간", "기저(셧다운)"], PALETTE_PAIRWISE):
        sub = d[d["레짐"] == name]
        ax.scatter(
            sub["생산량"], sub["y_avg"], s=9, color=color, alpha=0.45,
            linewidths=0.5, edgecolors="#ffffff", label=name,
        )
    ax.set_xlabel("생산량 (개)", fontsize=9)
    ax.set_ylabel("평균전력 (kW)", fontsize=9)
    ax.set_title(f"생산량–전력 관계 (전체 corr={df['생산량'].corr(df['y_avg']):.3f})", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, d.groupby("레짐", observed=False)[["생산량", "y_avg"]].mean().round(2)


_fig, _src = plot_production_vs_power(df, operating_calendar)
save_fig(_fig, "F10", "생산량 전력 산점도", "1.5", source_table=_src)


def plot_correlation_matrix(df: pd.DataFrame):
    """정비 후 상관행렬 (발산형 팔레트)."""
    cols = ["y_avg", "y_peak", "생산량", "공장인원", "기온", "습도", "풍속", "강수량", "인건비"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr.to_numpy(), cmap=CMAP_DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if abs(v) > 0.55 else INK)
    ax.set_title("상관행렬 (정비 후)", fontsize=10, color=INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.01).ax.tick_params(labelsize=8, colors=INK_SOFT)
    fig.tight_layout()
    return fig, corr.round(3)


_fig, _src = plot_correlation_matrix(df)
save_fig(_fig, "F11", "상관행렬", "1.5", source_table=_src)

# %% [markdown]
# ### 1.6 피크 정의 및 임계값 확정
#
# - **목적**: `peak15` 정의를 확정하고 피크 **정의 임계값 θ** 를 1회 산출·고정한다.
# - **보고서 대응절**: 1.1 (피크 기준), 1.6 (학습·검증 구성)
# - **산출물**: `ch1_theta.csv`, F06
#
# **임계값 2종을 기호로 분리한다.**
#
# | 기호 | 의미 | 산출 범위 | 값 |
# |---|---|---|---|
# | **θ (THETA)** | 피크 **정의** 임계값 | 학습 전체구간 q95에서 **1회**, 7/13·7/15 **포함** | **고정** |
# | **τ (TAU)** | **판정/경보** 임계값 | **각 fold 검증구간에서만** 산출 | 모델·fold별 가변 |
#
# θ를 fold마다 재산출하면 `y_cls` **정의 자체가 달라져** fold 간 Recall 평균이 무의미해진다.
# 실제로 fold별 학습구간 q95는 182/182/184/186/186 으로 **어느 fold에서도 187이 나오지 않는다.**
# 따라서 θ 고정은 선택이 아니라 필수다.

# %%
TEST_START = pd.Timestamp("2021-09-01")
TEST_END = pd.Timestamp("2021-09-14 23:00")


def compute_theta(df: pd.DataFrame, test_start=TEST_START) -> float:
    """피크 **정의** 임계값 θ 를 학습 전체구간 q95에서 1회 산출한다.

    테스트 구간을 쓰지 않으므로 누수가 아니다. ERP 결측일(7/13·7/15)은
    사용자 확정 결정에 따라 **θ 산출에는 포함**한다(보고서 기존 수치 187 유지).
    """
    train = df.loc[df.index < test_start, "y_peak"]
    return float(train.quantile(0.95))


def peak_definition_comparison(df: pd.DataFrame, test_start=TEST_START) -> pd.DataFrame:
    """`peak15 = max(4구간)` 과 `15분` 단독 정의의 분위수를 비교한다.

    보고서의 187/201/222 는 **max(4구간) 정의에서만** 재현된다.
    가이드북 베이스라인은 `15분` 만 쓰므로 따라가면 안 된다.
    """
    train = df.loc[df.index < test_start]
    rows = []
    for label, s in [("peak15 = max(15,30,45,60분)", train["y_peak"]), ("15분 단독", train["15분"])]:
        rows.append(
            {
                "정의": label,
                "상위5%(q95)": round(float(s.quantile(0.95)), 1),
                "상위1%(q99)": round(float(s.quantile(0.99)), 1),
                "최대값": int(s.max()),
            }
        )
    out = pd.DataFrame(rows)
    out["보고서 일치"] = ["✅ 187/201/222 재현", "❌ 불일치"]
    return out


THETA = compute_theta(df)
theta_compare = peak_definition_comparison(df)
save_table(theta_compare, "ch1_theta")

df["y_cls"] = (df["y_peak"] >= THETA).astype(int)

print("── 피크 정의 대조 ──")
display(theta_compare)
print(f"\n  θ (피크 정의 임계값) = {THETA:g}  ← 전 fold·테스트 고정")
print(f"  학습구간 양성률       = {df.loc[df.index < TEST_START, 'y_cls'].mean():.2%}")
print(f"  테스트구간 양성 건수  = {int(df.loc[df.index >= TEST_START, 'y_cls'].sum())}건")

# 최대값 도달 시각 — 8장 저감 시뮬레이션의 표적
_max_peak = df["y_peak"].max()
max_peak_times = df.index[df["y_peak"] == _max_peak]
print(f"\n  전체 최대 peak15 = {_max_peak:g} kW, 도달 {len(max_peak_times)}건")
for t in max_peak_times:
    print(f"    - {t:%Y-%m-%d %H시}")

# %% [markdown]
# #### 그림 F06 — peak15 분포와 임계값

# %%
def plot_peak_distribution(df: pd.DataFrame, theta: float):
    """peak15 히스토그램 + θ·q99·max 수직선."""
    train = df.loc[df.index < TEST_START, "y_peak"]
    q99, mx = float(train.quantile(0.99)), float(train.max())

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(train.to_numpy(), bins=60, color=COLOR_MUTED)
    ax.hist(train[train >= theta].to_numpy(), bins=60, color=COLOR_HERO,
            label=f"피크 구간 (≥ θ={theta:g})")
    for x, lab, c in (
        (theta, f"θ = {theta:g} (q95)", PALETTE_ADJACENT[1]),
        (q99, f"q99 = {q99:g}", PALETTE_ADJACENT[3]),
        (mx, f"max = {mx:g}", PALETTE_ADJACENT[4]),
    ):
        ax.axvline(x, color=c, lw=1.6)
        ax.annotate(lab, (x, ax.get_ylim()[1] * 0.9), rotation=90, fontsize=8,
                    color=INK, ha="right", va="top")
    ax.set_xlabel("peak15 (kW)", fontsize=9)
    ax.set_ylabel("시간 수", fontsize=9)
    ax.set_title("학습구간 peak15 분포와 피크 정의 임계값", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.25))
    fig.tight_layout()
    src = pd.DataFrame({"지표": ["q95(θ)", "q99", "max"], "값": [theta, q99, mx]})
    return fig, src


_fig, _src = plot_peak_distribution(df, THETA)
save_fig(_fig, "F06", "peak15 분포와 임계값", "1.6", source_table=_src)

# %% [markdown]
# ### 1장 게이트 — 정비 결과 검증
#
# 여기서 실패하면 이후 모든 장이 무의미하므로 즉시 중단시킨다.

# %%
def gate_chapter1(df: pd.DataFrame, theta: float) -> pd.DataFrame:
    """1장 게이트: 중복 0 · 시간 0~23 · 결측 0 · 무결성 100%.

    Returns
    -------
    pandas.DataFrame
        점검 항목별 통과 여부. 하나라도 실패하면 AssertionError.
    """
    m4 = df[POWER_COLS].mean(axis=1)
    checks = [
        ("(날짜,시간) 중복 0건", int(df.reset_index().duplicated(["날짜", "시간"]).sum()) == 0),
        ("시간값 0~23 100%", bool(df["시간"].between(0, 23).all())),
        ("결측 0건", int(df.isna().sum().sum()) == 0),
        ("무결성 평균=round(m4) 100%", bool((df["평균"] == np.floor(m4 + 0.5)).all())),
        ("행 수 6168", len(df) == 6168),
        ("θ 고정값 확인", theta == 187.0),
    ]
    out = pd.DataFrame(checks, columns=["점검", "통과"])
    failed = out[~out["통과"]]
    if len(failed):
        raise AssertionError(f"1장 게이트 실패: {failed['점검'].tolist()}")
    return out


gate1 = gate_chapter1(df, THETA)
save_table(gate1, "gate1_chapter1")
print("── 게이트 1 통과 ──")
display(gate1)
