# %% tags=["nb-strip"]
from s00_env import (  # noqa: F401
    FAST,
    FIG_DIR,
    FIGURE_INDEX_PATH,
    FONT_NAME,
    HAS_TF,
    OUTPUT_DIR,
    SEED,
    TBL_DIR,
    display,
    env_versions,
    save_table,
)
from s01_diagnose import (  # noqa: F401
    N_SURPLUS_DAYS,
    N_TOTAL_DAYS,
    N_UNIQUE_PROFILES,
    THETA,
    TEST_END,
    TEST_START,
    df,
    operating_calendar,
)
from s00_env import DATA_PATH, MODEL_DIR  # noqa: F401
from s01_diagnose import df_raw  # noqa: F401
from s02_features import FEATURE_COLS, HOLIDAYS_2021, N_WARMUP, SAFE_LAGS, feat  # noqa: F401
from s03_split import ACTIVE_FOLDS, CALENDAR_RULE_FP, calendar_rule_metrics, tune_tau, usable_mask  # noqa: F401
from s05_models import (  # noqa: F401
    HPO_N_TRIALS,
    LGB_BASE,
    MODEL_REGISTRY,
    N_ESTIMATORS,
    _calib,
    _unc,
    cv_results,
    fit_interval_models,
    conformal_quantile,
    test_results,
)
from s06_eval import (  # noqa: F401
    BEST_BASELINE_MAE,
    BEST_BASELINE_NAME,
    CLF_FN,
    CLF_FP,
    CLF_RECALL,
    CLF_TP,
    FINAL_F1,
    FINAL_MAE,
    FINAL_MODEL_NAME,
    FINAL_PEAK_MAE,
    FINAL_PEAK15_MAE,
    FINAL_PEAK_N,
    FINAL_RECALL,
    IMPROVE_MAE_PCT,
    IMPROVE_PEAKMAE_PCT,
    INFER_SEC,
    MAE_IMPROVEMENT_SIGNIFICANT,
    MAE_IMPROVEMENT_EXPLORATORY_10PCT,
    SURROGATE_FIDELITY,
    TEST_POSITIVES,
    interpretation_tbl,
    peak_tbl,
    rank_preservation,
    significance_tbl,
)
from s07_analysis import (  # noqa: F401
    BEST_CONDITION_NAME,
    FN_TOP_CONDITION,
    FN_TOP_SHARE_PCT,
    FP_TOP_CONDITION,
    TOP_FEATURES,
    TOP_INTERACTION,
    WORST_CONDITIONS,
    WORST_CONDITION_NAME,
    WORST_COND_EXCESS_PCT,
    OOF,
)
from s08_simulation import (  # noqa: F401
    BETA,
    PEAK_REDUCTION_KW,
    SEPT_ONLY_SAVING,
    scenarios_tbl,
)
from s09_creativity import creativity_tbl  # noqa: F401
import re
import numpy as np
import pandas as pd

# %% [markdown]
# ## 10. 산출물·재현성·제출 패키징
#
# 보고서 **6장 코드 구성 및 재현성(10점)** 의 근거를 생성하고, 제출물을 만든다.

# %% [markdown]
# ### 10.1 테스트 예측결과 파일
#
# - **목적**: 제출물 "테스트데이터 예측결과 파일"을 규격에 맞게 생성한다.
# - **보고서 대응절**: 6장
# - **산출물**: `outputs/predictions_test_336h.csv`
#
# 규격: **336행 · 결측 0 · `utf-8-sig`**, 컬럼
# `datetime, y_avg_true, y_avg_pred, y_peak_true, y_peak_pred, peak_prob, peak_pred_label, peak_true_label, model_name`

# %%
PREDICTION_PATH = OUTPUT_DIR / "predictions_test_336h.csv"


def build_prediction_file() -> pd.DataFrame:
    """제출용 예측결과 파일을 만든다.

    피크 확률은 '피크 직접분류' 모델에서, 판정 라벨은 최종모델의 τ 적용 결과에서 가져온다.
    테스트 336행은 어떤 마스크와도 무관하게 **전부 유지**한다.
    """
    te = test_results[FINAL_MODEL_NAME]
    tau = cv_results[FINAL_MODEL_NAME]["tau"]
    clf = test_results.get("피크 직접분류")
    prob = clf["prob"] if clf is not None and clf["prob"] is not None else np.full(len(te["index"]), np.nan)

    out = pd.DataFrame(
        {
            "datetime": pd.DatetimeIndex(te["index"]).strftime("%Y-%m-%d %H:%M:%S"),
            "y_avg_true": np.round(te["y_avg"], 4),
            "y_avg_pred": np.round(te["pred_avg"], 4),
            "y_peak_true": np.round(te["y_peak"], 4),
            "y_peak_pred": np.round(te["pred_peak"], 4),
            "peak_prob": np.round(prob, 6),
            "peak_pred_label": (te["pred_peak"] >= tau).astype(int),
            "peak_true_label": te["y_cls"].astype(int),
            "model_name": FINAL_MODEL_NAME,
        }
    )
    return out


predictions = build_prediction_file()
predictions.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")
print(f"── 예측결과 파일 생성: {PREDICTION_PATH.name} ──")
print(f"  행 수 {len(predictions)} / 결측 {int(predictions.isna().sum().sum())}건 / 인코딩 utf-8-sig")
display(predictions.head(3))

# %% [markdown]
# ### 10.2 보고서 채움표 · 문장 치환사전 · 장별 PASS/FAIL
#
# - **목적**: 보고서의 모든 `[TBD]` 를 실측값으로 채우고, **장별 충족 여부**를 판정한다.
# - **보고서 대응절**: 전 장
# - **산출물**: `outputs/report_tbd_filled.md`, `ch6_tbd_table.csv`
#
# > ### ⚠️ 빈칸 점검은 **2단계**여야 한다
# >
# > **① 정규식 점검만으로는 부족하다.** 단순 `[TBD]` 리터럴 grep 은
# > `[TBDd]`·`[TBDDd]`·`[Tbdd]`·`[TBdd]`·대괄호 없는 `TBd`·내부 공백이 있는 `[최종모델명 ]` 을
# > 전부 놓친다. 정규식은 `\[?T[Bb][Dd]+\]?` 와 `\[[^\]]*\]` 두 개를 함께 쓴다.
# >
# > **② 4·5·6장은 `[TBD]` 토큰이 0개다.** 따라서 정규식만 돌리면
# > **30점이 통째로 비어 있어도 통과한다.** 장별 충족 체크리스트가 반드시 필요하다.

# %%
def build_tbd_table() -> pd.DataFrame:
    """보고서 플레이스홀더 → 실측값 채움표."""
    rows = [
        # 1쪽 내용요약
        ("1쪽 내용요약", "[TBD]% (MAE 개선)", f"{IMPROVE_MAE_PCT:.2f}%", "6.1 유의성 검정"),
        ("1쪽 내용요약", "[TBD] (피크 Recall)", f"{FINAL_RECALL:.3f}", "6.2 피크탐지표"),
        ("1쪽 내용요약", "[TBD] (피크 F1)", f"{FINAL_F1:.3f}", "6.2 피크탐지표"),
        ("1쪽 내용요약", "[TBD] kW (최대수요 저감)", f"{PEAK_REDUCTION_KW:.1f} kW", "8.3 시나리오 (전체기간 기준)"),
        # 2장 도입
        ("2장 도입", "[최종모델명 ]", FINAL_MODEL_NAME, "6.4 스코어카드"),
        ("2장 도입", "[TBDd]% (MAE 개선)", f"{IMPROVE_MAE_PCT:.2f}%", "6.1"),
        ("2장 도입", "[TBDDd] (Recall)", f"{FINAL_RECALL:.3f}", "6.2"),
        # 2.5절
        ("2.5절", "[TBDd]회 탐색", f"{HPO_N_TRIALS}회", "5.1·5.4 Optuna 이력"),
        # 2.6절
        ("2.6절 표", "[최종모델]", FINAL_MODEL_NAME, "6.4"),
        ("2.6절 본문", "[최종모델명 ]", FINAL_MODEL_NAME, "6.4"),
        ("2.6절 본문", "[TBDD] (최종 MAE)", f"{FINAL_MAE:.3f}", "6.1"),
        ("2.6절 본문", "[TBDd] (최선 베이스라인 MAE)", f"{BEST_BASELINE_MAE:.3f} ({BEST_BASELINE_NAME})", "6.1"),
        ("2.6절 본문", "[Tbdd]% (개선율)", f"{IMPROVE_MAE_PCT:.2f}%", "6.1"),
        ("2.6절 본문", "[TBDd]% (Peak-MAE 감소)", f"{IMPROVE_PEAKMAE_PCT:.2f}%", "6.1"),
        # 2.7절
        ("2.7절 본문", "[TBDd] (직접분류 Recall)", f"{CLF_RECALL:.3f}", "6.2"),
        ("2.7절 본문", "[TBD] (전체 피크 건수)", f"{TEST_POSITIVES}건", "1.6 θ 적용"),
        ("2.7절 본문", "[TBDd] (탐지 건수)", f"{CLF_TP}건", "6.2"),
        ("2.7절 본문", "[TBDd] (오경보)", f"{CLF_FP}건", "6.2"),
        ("2.7절 본문", "[TBdd] (미탐지)", f"{CLF_FN}건", "6.2"),
        # 2.9절
        ("2.9절", "[최종모델명 ]", FINAL_MODEL_NAME, "6.4"),
        ("2.9절", "[TBD]% (MAE 개선)", f"{IMPROVE_MAE_PCT:.2f}%", "6.1"),
        ("2.9절", "[TBDd] (Recall)", f"{FINAL_RECALL:.3f}", "6.2"),
        ("2.9절", "[TBDd]초 (추론시간)", f"{INFER_SEC:.2f}초", "6.4 (336h 배치, 10회 중앙값)"),
        # 3장 도입
        ("3장 도입", "[상위변수 1]", TOP_FEATURES[0], "7.1 permutation"),
        ("3장 도입", "[상위변수 2]", TOP_FEATURES[1], "7.1 permutation"),
        ("3장 도입", "[상위변수 3]", TOP_FEATURES[2], "7.1 permutation"),
        ("3장 도입", "[가동전환, 휴일 다음날, …]", ", ".join(WORST_CONDITIONS), "7.3 조건별 오차"),
        ("3장 도입", "[TBDd]% (초과율)", f"{WORST_COND_EXCESS_PCT:.1f}%", "7.3"),
        # 3.1·3.2절
        ("3.1절", "[예:24시간 전 전력]", TOP_FEATURES[0], "7.1"),
        ("3.1절", "[예: 생산량]", TOP_FEATURES[1], "7.1"),
        ("3.2절", "[실제 확인된 상호작용]", TOP_INTERACTION, "7.2"),
        # 3.3·3.4절
        ("3.3절", "[실제 조건] (최대오차)", WORST_CONDITION_NAME, "7.3"),
        ("3.3절", "[TBD]% (평균 대비)", f"{WORST_COND_EXCESS_PCT:.1f}%", "7.3"),
        ("3.3절", "[실제 조건] (안정)", BEST_CONDITION_NAME, "7.3"),
        ("3.4절", "[실제 조건] (FN 집중)", FN_TOP_CONDITION, "7.4"),
        ("3.4절", "[TBDd]% (FN 비중)", f"{FN_TOP_SHARE_PCT:.1f}%", "7.4"),
        ("3.4절", "[실제조건] (FP 집중)", FP_TOP_CONDITION, "7.4"),
    ]
    return pd.DataFrame(rows, columns=["위치", "플레이스홀더", "채움값", "산출 근거(노트북 절)"])


tbd_tbl = build_tbd_table()
save_table(tbd_tbl, "ch6_tbd_table")
print(f"── 보고서 채움표 {len(tbd_tbl)}항목 ──")
display(tbd_tbl)


def scan_placeholders(text: str) -> list[str]:
    """2단계 빈칸 점검의 ① — 정규식으로 잔여 플레이스홀더를 찾는다.

    단순 `[TBD]` grep 이 놓치는 변형을 모두 잡는다.
    """
    WHITELIST = {"[O]", "[X]", "[발견 1]", "[발견 2]", "[발견 3]"}
    hits = []
    for pat in (r"\[?T[Bb][Dd]+\]?", r"\[[^\]\n]{0,40}\]"):
        for m in re.finditer(pat, text):
            tok = m.group()
            if tok in WHITELIST:
                continue
            # 마크다운 링크·각주는 제외
            if re.fullmatch(r"\[\d+\]", tok):
                continue
            hits.append(tok)
    return sorted(set(hits))


CHAPTER_REQUIREMENTS = {
    "1장 데이터 이해 및 진단": [
        ("변수사전 18컬럼", lambda: (TBL_DIR / "ch1_variable_dict.csv").exists()),
        ("품질 6기준 진단표", lambda: (TBL_DIR / "ch1_quality.csv").exists()),
        ("프로파일 복제 진단", lambda: N_SURPLUS_DAYS == 115),
        ("가동 캘린더", lambda: (TBL_DIR / "ch1_calendar.csv").exists()),
        ("θ 확정", lambda: THETA == 187.0),
    ],
    "2장 AI 예측모델 개발 및 성능평가": [
        ("2.2 베이스라인 결함표", lambda: (TBL_DIR / "ch4_defects.csv").exists()),
        ("2.6 회귀 성능표", lambda: (TBL_DIR / "ch2_regression.csv").exists()),
        ("2.7 피크탐지표 + τ + 달력규칙 행", lambda: "τ" in pd.read_csv(TBL_DIR / "ch2_peak_detection.csv", encoding="utf-8-sig").columns),
        ("2.8 Ablation 5행", lambda: len(pd.read_csv(TBL_DIR / "ch2_ablation.csv", encoding="utf-8-sig")) == 5),
        ("2.9 스코어카드", lambda: (TBL_DIR / "ch2_scorecard.csv").exists()),
        ("D1/D2 대비표", lambda: (TBL_DIR / "ch2_d1_vs_d2.csv").exists()),
    ],
    "3장 영향요인 및 오류분석": [
        ("3.1 상위 5변수", lambda: len(pd.read_csv(TBL_DIR / "ch3_importance.csv", encoding="utf-8-sig")) == 5),
        ("3.2 상호작용 4종", lambda: len(pd.read_csv(TBL_DIR / "ch3_interaction.csv", encoding="utf-8-sig")) == 4),
        ("3.3 조건별 오차 7행", lambda: len(pd.read_csv(TBL_DIR / "ch3_condition_mae.csv", encoding="utf-8-sig")) == 7),
        ("3.4 혼동행렬 4구분", lambda: len(pd.read_csv(TBL_DIR / "ch3_confusion.csv", encoding="utf-8-sig")) == 4),
        ("3.5 실패사례 3건", lambda: len(pd.read_csv(TBL_DIR / "ch3_failures.csv", encoding="utf-8-sig")) == 3),
        ("3.6 규칙 R1~R3", lambda: len(pd.read_csv(TBL_DIR / "ch3_rules.csv", encoding="utf-8-sig")) >= 1),
    ],
    "4장 현장 활용방안 (10점)": [
        ("본문 초안 생성", lambda: (OUTPUT_DIR / "report_ch4_draft.md").exists()),
        ("레버 L1~L4 전부", lambda: len(pd.read_csv(TBL_DIR / "ch4_levers.csv", encoding="utf-8-sig")) >= 4),
        ("순절감액 산출", lambda: "순절감액(원)" in pd.read_csv(TBL_DIR / "ch4_scenarios.csv", encoding="utf-8-sig").columns),
        ("민감도 ±30%", lambda: (TBL_DIR / "ch4_sensitivity.csv").exists()),
        ("운영 프로토콜", lambda: (TBL_DIR / "ch4_protocol.csv").exists()),
    ],
    "5장 창의성 및 차별성 (10점)": [
        ("본문 초안 생성", lambda: (OUTPUT_DIR / "report_ch5_draft.md").exists()),
        ("차별점 5종", lambda: len(creativity_tbl) == 5),
        ("2·3장 결과 연결", lambda: creativity_tbl["대응 절"].notna().all()),
    ],
    "6장 코드 구성 및 재현성 (10점)": [
        ("본문 초안 생성", lambda: (OUTPUT_DIR / "report_ch6_draft.md").exists()),
        ("예측결과 파일 336행", lambda: len(predictions) == 336),
        ("requirements.txt", lambda: (OUTPUT_DIR.parent / "requirements.txt").exists()),
        ("환경 증거(requirements_generated)", lambda: (OUTPUT_DIR / "requirements_generated.txt").exists()),
        ("그림 인덱스", lambda: FIGURE_INDEX_PATH.exists()),
        ("채움표", lambda: len(tbd_tbl) >= 30),
    ],
}


def check_chapters() -> pd.DataFrame:
    """2단계 빈칸 점검의 ② — 장별 충족 체크리스트."""
    rows = []
    for chapter, checks in CHAPTER_REQUIREMENTS.items():
        for label, fn in checks:
            try:
                ok = bool(fn())
            except Exception as e:
                ok = False
                label = f"{label} (오류: {type(e).__name__})"
            rows.append({"장": chapter, "요건": label, "충족": ok})
    out = pd.DataFrame(rows)
    summary = out.groupby("장")["충족"].all().rename("PASS")
    out.attrs["summary"] = summary
    return out


chapter_checks = check_chapters()
save_table(chapter_checks, "ch6_chapter_checklist")
print("\n── 장별 충족 체크리스트 ──")
display(chapter_checks.attrs["summary"].to_frame())
_failed = chapter_checks[~chapter_checks["충족"]]
if len(_failed):
    print("\n  ⚠️ 미충족 요건:")
    display(_failed)
else:
    print("\n  ✅ 전 장 요건 충족")

# %% [markdown]
# ### 10.3 `report_tbd_filled.md` 생성 — 채움표 + 치환사전 + plan 대비 정정

# %%
# plan.md 예상치와 실측이 다른 항목 — 보고서가 plan 수치로 쓰여 있으면 어긋난다
PLAN_DELTAS = [
    ("fold3 휴무 비중", "29%", f"{float(pd.read_csv(TBL_DIR / 'ch3_folds.csv', encoding='utf-8-sig').set_index('fold').loc[3, '휴무비중'].strip('%')):.0f}%",
     "가동 캘린더를 `일생산량==0` 으로 판정하면 일요일·증강불일치일이 휴무로 더 많이 잡힌다"),
    ("fold4 휴무 비중", "36%", f"{float(pd.read_csv(TBL_DIR / 'ch3_folds.csv', encoding='utf-8-sig').set_index('fold').loc[4, '휴무비중'].strip('%')):.0f}%", "동일"),
    ("OOF 양성 합계", "185건", f"{int(OOF['y_cls'].sum())}건",
     "계측정지 17행·ERP결측 48행을 학습·평가에서 제외하고 fold1을 폐기한 결과"),
    ("토요일 평균전력", "51.1", "51.3", "08-28~29 계측정지 17행을 시간보간한 결과 (+0.2)"),
    ("일요일 평균전력", "41.3", "41.6", "동일 (+0.3)"),
    ("08시 피크율", "24.7%", "24.5%", "동일"),
    ("2/11·3/1 판정", "정상 가동일(공휴일 리스트 오류)", "공휴일 맞음 — 전력만 증강 덮어쓰기",
     "두 날 모두 생산량·공장인원이 0이고 프로파일 복제그룹에 속한다 → 전력 단일 채널 판정은 위험"),
]


def write_tbd_filled() -> str:
    """채움표·치환사전·정정사항·장별 PASS 를 하나의 문서로 만든다."""
    md = f"""# 보고서 빈칸 채움표 및 문장 치환사전

> 생성 시점의 노트북 실행 결과다. 노트북을 다시 실행하면 이 문서도 갱신된다.
> **FAST 모드 실행 여부: {FAST}** (True 면 탐색·에폭이 축소된 값이므로 제출 전 FULL 재실행 필요)

## 1. 최종 결과 요약

| 항목 | 값 |
|---|---|
| 최종모델 | **{FINAL_MODEL_NAME}** |
| 고정 비교기준 | {BEST_BASELINE_NAME} (MAE {BEST_BASELINE_MAE:.3f}) |
| 테스트 MAE | {FINAL_MAE:.3f} ({IMPROVE_MAE_PCT:+.2f}%) |
| 실제 피크 시간 평균전력 MAE | {FINAL_PEAK_MAE:.3f} ({IMPROVE_PEAKMAE_PCT:+.2f}%), n={FINAL_PEAK_N} |
| 실제 피크 시간 최대전력 MAE | {FINAL_PEAK15_MAE:.3f} kW |
| 테스트 Recall / F1 | {FINAL_RECALL:.3f} / {FINAL_F1:.3f} |
| MAE 개선 5% 주분석 | {'개선 확인' if MAE_IMPROVEMENT_SIGNIFICANT else '개선 미확인'}, 95% CI {significance_tbl.set_index('모델').loc[FINAL_MODEL_NAME, '차이 95% CI']} |
| MAE 개선 10% 사후 보조분석 | {'개선 방향 차이' if MAE_IMPROVEMENT_EXPLORATORY_10PCT else '개선 미확인'}, 90% CI {significance_tbl.set_index('모델').loc[FINAL_MODEL_NAME, '차이 90% CI(사후 탐색)']} |
| 336시간 배치 추론 | {INFER_SEC:.2f}초 (10회 중앙값) |
| 최대수요 저감 (전체기간) | {PEAK_REDUCTION_KW:.1f} kW |
| HPO 탐색 횟수 | {HPO_N_TRIALS}회 |
| θ (피크 정의) | {THETA:g} (고정) |

## 2. 플레이스홀더 치환표 ({len(tbd_tbl)}항목)

| 위치 | 플레이스홀더 | → 채움값 | 산출 근거 |
|---|---|---|---|
"""
    for _, r in tbd_tbl.iterrows():
        md += f"| {r['위치']} | `{r['플레이스홀더']}` | **{r['채움값']}** | {r['산출 근거(노트북 절)']} |\n"

    md += """
## 3. 본문 문장 교정 (실측과 어긋나는 서술)

보고서 초안에 이미 쓰여 있으나 **실측과 다른** 서술이다. 반드시 고쳐야 한다.

"""
    fixes = [
        ("1.1절", "피크 기준 서술",
         f"`peak15 = max(15분, 30분, 45분, 60분)` 정의를 각주로 명시한다. "
         f"`15분` 단일 컬럼으로는 q95/q99/max 가 173.4/188/207 로 보고서의 187/201/222 와 불일치한다."),
        ("1.3절", "야간 피크 서술",
         "'야간·심야 피크 0건' → **'정상 가동일 기준 야간 피크 0건 "
         "(야간 피크 7건은 전부 ERP 결측일 7/13·7/15에 귀속)'**. "
         "이 문장이 3.6절 규칙과 4장 L4 레버의 근거이므로 그대로 두면 안 된다."),
        ("1.4절", "유일성 행",
         f"'날짜와 시간이 중복된 5건' → **'(날짜,시간) 중복 5건 + 24시간 전력 프로파일 복제 "
         f"{N_SURPLUS_DAYS}일({N_SURPLUS_DAYS / N_TOTAL_DAYS:.1%})'**"),
        ("1.4절", "무결성 지수 서술",
         "'무결성 지수는 33.3%에서 100%로 개선' → **'복제 구조를 진단하고 학습 설계에 반영'**. "
         "복제는 오류가 아니라 증강 구조이므로 '개선'으로 표현하면 부정확하다."),
        ("1.4절", "결측 서술 보강",
         "**'결측이 센서측(08-28 18시~08-29 10시, 17시간)과 ERP측(7/13·7/15, 48시간)의 "
         "양방향으로 존재함을 확인'** 을 추가한다. 양방향 결측을 대칭적으로 진단한 것이 "
         "서면평가 1번(검증전략)의 강한 근거다."),
        ("1.2절", "변수 단위 정정",
         "`공장인원`은 인원수가 아니라 **0~48.39 연속 실수(정규화 투입 공수)**, "
         "`인건비`는 금액이 아니라 **교대 배수(9~17시 1.0, 그 외 1.5)** 임을 명시한다."),
        ("1.5절", "기상변수 가정",
         "기상 실측값을 **D+1 예보의 대리(proxy)** 로 사용했음을 명시하고, "
         "기상 ablation 결과를 '예보오차에 대한 성능 하한'으로 해석한다."),
        ("1.6절", "학습 표본 한계",
         f"복제로 인해 **실질 학습 표본이 명목의 약 {N_UNIQUE_PROFILES / N_TOTAL_DAYS:.0%}** 임을 명시하고, "
         f"워밍업 탈락 {N_WARMUP}행을 반영한 실제 학습 구간을 기재한다."),
        ("2.1절", "교차검증 fold 수",
         f"'Rolling-origin 5-fold' → **'Rolling-origin {len(ACTIVE_FOLDS)}-fold "
         "(fold1은 검증일 50%가 학습구간과 동일 전력 프로파일이라 폐기)'**"),
        ("2.1절", "Task B 주지표",
         f"'Task B 주지표 Recall' → **'Task B 주지표 F1 / PR-AUC'**. "
         f"달력규칙 한 줄이 Recall 1.000(FP {CALENDAR_RULE_FP})을 내므로 Recall 로는 "
         "어떤 모델도 이길 수 없다. 주지표를 'Recall 유지 + 오경보 최소화'로 재정의한다."),
        ("4장", "절감액 해석 한계",
         f"**'테스트 구간(9월) 단독으로는 연간 기본요금에 영향이 없다'** 를 명시한다. "
         f"9월 peak15 를 0으로 낮추는 극단 가정에서도 절감액은 {SEPT_ONLY_SAVING:,.0f}원이다."),
    ]
    md += "| 절 | 대상 | 교정 내용 |\n|---|---|---|\n"
    for sec, target, fix in fixes:
        md += f"| {sec} | {target} | {fix} |\n"

    md += "\n### 3.1 6장에서 도출된 추가 교정\n\n"
    for _, r in interpretation_tbl.iterrows():
        md += f"- **{r['항목']}** (실측: {r['실측']})\n  → {r['보고서 서술 교정']}\n\n"

    md += """## 4. plan.md 예상치 대비 실측 정정

`plan.md` 에 적힌 예상 수치와 실제 실행 결과가 다른 항목이다.
보고서가 plan 수치로 쓰여 있으면 노트북 출력과 어긋나므로 **아래 실측값으로 맞춘다.**

| 항목 | plan.md | 실측 | 원인 |
|---|---|---|---|
"""
    for item, planv, actual, cause in PLAN_DELTAS:
        md += f"| {item} | {planv} | **{actual}** | {cause} |\n"

    md += "\n## 5. 장별 충족 판정 (PASS/FAIL)\n\n"
    md += "> ⚠️ 4·5·6장은 `[TBD]` 토큰이 0개라 정규식 점검만으로는 통째로 비어도 통과한다.\n"
    md += "> 그래서 아래 요건 기반 판정이 필수다.\n\n"
    md += "| 장 | 판정 |\n|---|---|\n"
    for ch, ok in chapter_checks.attrs["summary"].items():
        md += f"| {ch} | {'✅ PASS' if ok else '❌ FAIL'} |\n"
    md += "\n<details><summary>요건별 상세</summary>\n\n| 장 | 요건 | 충족 |\n|---|---|---|\n"
    for _, r in chapter_checks.iterrows():
        md += f"| {r['장']} | {r['요건']} | {'✅' if r['충족'] else '❌'} |\n"
    md += "\n</details>\n\n"

    md += """## 6. 사람이 수행할 항목 (노트북 밖)

1. 설문 응답 후 완료화면 캡처 → 보고서 10쪽 삽입
2. 발표자료 PDF·PPT 최종화 (`outputs/발표자료.pptx` 골격 활용)
3. 보고서 "작성 요령 ※ 작성 후 삭제" 블록 **전부 삭제**
4. 1쪽 서명·날짜 기입
5. 블라인드 규정 최종 확인 (소속·학교·로고 없음, 성명·팀명만)
6. 그림 39장 렌더 육안 점검 (라벨 충돌·축 잘림·한글 깨짐·범례 겹침)
7. 보고서 PDF 내보내기 후 `/Author` 메타데이터 확인·제거 (HWP 내보내기 시 계정명이 박힌다)
"""
    return md


_tbd_md = write_tbd_filled()
(OUTPUT_DIR / "report_tbd_filled.md").write_text(_tbd_md, encoding="utf-8")
_residual = scan_placeholders(_tbd_md.split("## 2. 플레이스홀더 치환표")[0])
print(f"── report_tbd_filled.md 생성 ({len(_tbd_md):,}자) ──")
print(f"  요약부 잔여 플레이스홀더: {_residual if _residual else '0건'}")

# %% [markdown]
# ### 10.4 requirements.txt · README(= 6장 본문)
#
# - **목적**: 심사위원이 전처리부터 결과생성까지 실행할 수 있는 문서를 만든다.
# - **보고서 대응절**: 6장 전체
# - **산출물**: `requirements.txt`, `README.md`, `outputs/report_ch6_draft.md`

# %%
PROJECT_ROOT = OUTPUT_DIR.parent


def write_requirements() -> str:
    """버전을 `==` 로 완전 핀한 requirements.txt.

    `torch`·`ortools` 는 실사용이 없으므로 제외한다.
    공휴일은 외부 패키지 대신 2021 상수 리스트를 쓴다(폐쇄망 심사환경 대응).
    """
    v = env_versions().set_index("항목")["버전"].to_dict()
    lines = ["# 제6회 K-인공지능 제조데이터 분석 경진대회 ⑤ 자원 최적화",
             f"# Python {v.get('python', '3.11')}", ""]
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
                "lightgbm", "optuna", "shap", "statsmodels"):
        ver = v.get(pkg, "")
        if ver and ver != "미설치":
            lines.append(f"{pkg}=={ver}")
    if v.get("tensorflow", "미설치") != "미설치" and not v["tensorflow"].startswith("미설치"):
        lines.append(f"tensorflow=={v['tensorflow']}")
    lines += ["python-pptx==1.0.2", "jupyter", "nbformat", "",
              "# 공휴일은 외부 패키지(holidays/workalendar) 대신 2021 상수 리스트를 사용한다",
              "# (폐쇄망 심사환경 대응)"]
    return "\n".join(lines) + "\n"


# ⚠️ 패키지 루트의 `requirements.txt` 를 덮어쓰지 않는다.
#    루트 파일은 사람이 관리하며 `ipykernel` 등 노트북 실행에 필요한 항목이 더 들어 있다.
#    여기서는 **현재 환경의 실제 설치 버전 증거**를 outputs/ 에 남기는 것이 목적이다.
#    (보고서 6장에서 "환경을 이렇게 고정했다"는 근거로 쓴다)
_req = write_requirements()
(OUTPUT_DIR / "requirements_generated.txt").write_text(_req, encoding="utf-8")
print("── outputs/requirements_generated.txt (실행 환경 증거) ──")
print(_req)


def write_readme() -> str:
    """Reuse the maintained package README instead of regenerating obsolete notebook commands."""
    return (OUTPUT_DIR.parent / "README.md").read_text(encoding="utf-8")


# ⚠️ 루트에 README.md 를 쓰지 않는다. 패키지의 진입점은 `00_README.md` 이며
#    사람이 관리한다. 여기서 생성하는 것은 **보고서 6장 본문 초안**이다.
_readme = write_readme()
(OUTPUT_DIR / "report_ch6_draft.md").write_text(
    "# 제 6 장. 코드 구성 및 재현성 〔10점〕\n\n" + _readme.split("# 제조 생산데이터")[-1].split("\n", 1)[1],
    encoding="utf-8",
)
print(f"── report_ch6_draft.md 생성 완료 (6장 본문 {len(_readme):,}자, 루트 README 는 쓰지 않음) ──")

# %% [markdown]
# ### 10.5 모델 저장 — 서비스 번들 (평가 · 배포)
#
# - **목적**: 학습된 모델을 파일로 저장해, 실서비스(매일 24:00 익일 24시간 예측)에서
#   **학습 시점과 비트 단위로 같은 예측**을 재현할 수 있게 한다.
# - **보고서 대응절**: 6장 (코드 구성 및 재현성)
# - **산출물**: `outputs/models/{full|fast}/{eval|deploy}/`, `ch6_model_bundles.csv`
#
# | 번들 | 학습 구간 | 용도 |
# |---|---|---|
# | `eval` | 2021-01-08 ~ 08-31 (테스트 이전) | 제출 예측을 만든 **바로 그 모델**. 재로드 예측이 제출 파일과 비트 단위로 같은지 검증 |
# | `deploy` | 2021-01-08 ~ 09-14 (전 구간) | 배포용 재학습. **보고서의 어떤 수치에도 쓰지 않는다** |
#
# 번들 구성: 최종모델(2단계 레짐 3분류 — 게이트 + 레짐별 회귀 + fallback) · 피크 직접분류기 ·
# Isotonic 보정기 · 예측구간(분위 q10/q50/q90, 최종모델 OOF 잔차 기반 구간) · τ·θ·피처 계약.
#
# **저장 형식을 이렇게 정한 이유**
#
# | 결정 | 근거 |
# |---|---|
# | pickle 을 쓰지 않는다 | 레짐 모델의 예측 함수가 클로저라 pickle 자체가 불가하다. 노트북에서 정의한 객체는 `__main__` 에 묶여 서비스에서 로드할 수도 없다 |
# | LightGBM 네이티브 텍스트 + JSON 매니페스트 | 라이브러리 표준 형식이고, 결정적 설정에서 재적합하면 **바이트까지 같다** → sha256 로 재현성을 증명한다 |
# | 바이트로 읽고 sha256 대조 후 `model_str` 로 연다 | `Booster(model_file=...)` 는 한글이 든 절대경로를 열지 못한다 |
# | 예측 직전 컬럼 재정렬 + float64 통일 | LightGBM 은 컬럼 순서가 바뀌어도 **오류 없이 틀린 값**을 낸다. 입력은 float64 로 통일한다(float32 로 한 번 내려간 값은 정밀도가 손실돼 예측이 달라진다) |
# | 매니페스트도 해시한다 | τ·구간 반폭은 매니페스트 안에만 있다. 파일 sha 만으로는 매니페스트 변조를 못 잡으므로 `manifest_sha256` 을 두고 `bundle_id` 가 그 앞 12자리다 |
# | 하이퍼파라미터는 dict → JSON | 모델 텍스트의 파라미터는 6자리로 반올림돼 재학습 레시피로 쓸 수 없다 |
# | FAST 번들은 `fast/` 에 분리 | 테스트 실행이 FULL 번들을 덮어쓰지 못하게 한다. 서빙 로더는 FAST 번들을 거부한다 |
#
# 배포 번들의 τ·τ_cls·보정기·예측구간 반폭은 교차검증 산출물이라 재학습 구간에서 다시 만들 수 없다.
# 그래서 평가 번들 값을 **묶음으로 상속**하고 출처를 매니페스트에 적는다.

# %%
def bundle_predict(parts: dict, X: pd.DataFrame) -> dict:
    """모델 번들의 부스터로 예측한다.

    노트북의 재로드 검증과 서빙 패키지(`serving/_core.py`, 코드 생성)가 **같은 함수**를 쓴다.
    5.2절 레짐 모델의 `predict` 와 같은 규칙이다: 게이트가 레짐을 고르고, 그 레짐의
    회귀 모델이 예측하며, 학습에 없던 레짐으로 배정된 행은 fallback 이 채운다.

    Parameters
    ----------
    parts : dict
        `columns`(피처 순서), `gate`, `gate_features`, `classes`, `regs`({타깃: {레짐: 부스터}}),
        `fallbacks`({타깃: 부스터}), 선택적으로 `peak_clf`, `quantiles`({분위: 부스터}).
    X : pandas.DataFrame
        피처 행렬. 컬럼 순서와 dtype 은 여기서 강제한다.

    Returns
    -------
    dict[str, numpy.ndarray]
        `regime`, `y_avg`, `y_peak`, (있으면) `prob`, `q10`·`q50`·`q90`.
    """
    cols = list(parts["columns"])
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise KeyError(f"피처 누락: {missing[:5]}")
    # LightGBM 은 컬럼 순서가 바뀌어도 오류 없이 틀린 값을 낸다 → 계약 순서로 재정렬, float64 로 통일
    Xf = X.reindex(columns=cols).astype(np.float64)
    arr = Xf.to_numpy()

    gate_p = parts["gate"].predict(Xf[list(parts["gate_features"])].to_numpy())
    classes = np.asarray(parts["classes"])
    if gate_p.ndim == 1:
        # 2분류 게이트: sklearn predict 와 같다(p > 0.5 일 때만 두 번째 클래스)
        regime = classes[(gate_p > 0.5).astype(int)]
    else:
        regime = classes[np.argmax(gate_p, axis=1)]
    out = {"regime": regime.astype(int)}

    for tgt in ("y_avg", "y_peak"):
        pred = np.empty(len(Xf), float)
        assigned = np.zeros(len(Xf), bool)
        for r, bst in parts["regs"][tgt].items():
            m = regime == int(r)
            if m.any():
                pred[m] = bst.predict(arr[m])
                assigned |= m
        if (~assigned).any():
            pred[~assigned] = parts["fallbacks"][tgt].predict(arr[~assigned])
        out[tgt] = pred

    if parts.get("peak_clf") is not None:
        out["prob"] = parts["peak_clf"].predict(arr)
    for q, bst in (parts.get("quantiles") or {}).items():
        out[f"q{int(round(float(q) * 100))}"] = bst.predict(arr)
    return out


def manifest_core_sha256(manifest: dict) -> str:
    """매니페스트 자체의 무결성 해시.

    `bundle_id`·`manifest_sha256` 두 키를 뺀 나머지를 결정적 JSON(키 정렬·UTF-8·NaN 금지)으로
    직렬화한 sha256 이다. τ·구간 반폭처럼 매니페스트 안에만 있는 값의 변조를 잡는다.
    노트북 검증과 서빙 로더가 **같은 함수**로 계산한다(코드 생성).
    """
    core = {k: v for k, v in manifest.items() if k not in ("bundle_id", "manifest_sha256")}
    text = json.dumps(core, sort_keys=True, ensure_ascii=False, indent=1, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# %%
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

import lightgbm as lgb  # noqa: E402
import sklearn  # noqa: E402

BUNDLE_SCHEMA_VERSION = 1
SERVICE_MODEL_NAME = "2단계 레짐(3분류)"   # 사용자 확정: 서비스 번들은 항상 이 모델
CLF_MODEL_NAME = "피크 직접분류"
BUNDLE_BASE = MODEL_DIR / ("fast" if FAST else "full")
HISTORY_DAYS = {"min": 8, "recommended": 14, "max": 62}
PLAN_COLUMNS = ["날짜", "시간", "생산량", "공장인원", "기온", "풍속", "습도", "강수량"]
HISTORY_COLUMNS = ["날짜", "시간", "15분", "30분", "45분", "60분", "평균",
                   "생산량", "공장인원", "인건비", "기온", "풍속", "습도", "강수량"]
INTERVAL_ALPHA = 0.9
GOLDEN_ORIGIN = pd.Timestamp("2021-09-13")   # 자가검증 사례: 09-13 24:00 원점 → 09-14 예측
GOLDEN_HISTORY_DAYS = 21


def _sha256(b: bytes) -> str:
    """바이트열의 sha256 16진 문자열."""
    return hashlib.sha256(b).hexdigest()


def _json_bytes(obj) -> bytes:
    """결정적 JSON 바이트 — 키 정렬 · UTF-8 · NaN 금지. OS·로캘과 무관하게 같은 바이트가 나온다."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")


def _native(v):
    """numpy 스칼라·결측을 JSON 호환 파이썬 값으로 바꾼다(결측은 None)."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def _booster_bytes(model) -> bytes:
    """sklearn 래퍼 → LightGBM 네이티브 텍스트 바이트."""
    return model.booster_.model_to_string().encode("utf-8")


def component_files(art: dict) -> dict:
    """적합 객체 묶음을 `{파일명: LightGBM 텍스트 바이트}` 로 바꾼다."""
    reg, clf, itv = art["regime"], art["clf"], art["interval"]
    files = {"gate.lgb": _booster_bytes(reg["gate"])}
    for tgt in ("y_avg", "y_peak"):
        for r, m in sorted(reg["regs"][tgt].items()):
            files[f"reg_{tgt}_r{int(r)}.lgb"] = _booster_bytes(m)
        files[f"fallback_{tgt}.lgb"] = _booster_bytes(reg["fallbacks"][tgt])
    files["peak_clf.lgb"] = _booster_bytes(clf["clf"])
    for q, m in sorted(itv["qmodels"].items()):
        files[f"interval_q{int(round(q * 100))}.lgb"] = _booster_bytes(m)
    return files


def training_frame(cutoff) -> tuple:
    """`cutoff` 이전의 사용 가능 행(D1 — 워밍업·계측정지·ERP결측 제외)."""
    tr = usable_mask(feat, "D1") & (feat.index < cutoff)
    return feat.loc[tr, FEATURE_COLS], feat.loc[tr, ["y_avg", "y_peak", "y_cls"]]


def fit_service_artifacts(cutoff) -> dict:
    """서비스 번들 구성요소를 `cutoff` 이전 데이터로 적합한다.

    적합 로직을 복제하지 않는다. 5장과 **같은 함수 객체**(`MODEL_REGISTRY`,
    `fit_interval_models`)를 학습 데이터만 바꿔 호출한다.
    """
    Xtr, ytr = training_frame(cutoff)
    dummy = Xtr.iloc[:24]  # fit_fn 은 검증 X 를 요구한다 — 그 예측은 쓰지 않는다
    shutdown_days = operating_calendar.index[operating_calendar["is_shutdown"]]
    return {
        "regime": MODEL_REGISTRY[SERVICE_MODEL_NAME](Xtr, ytr, dummy)["_artifacts"],
        "clf": MODEL_REGISTRY[CLF_MODEL_NAME](Xtr, ytr, dummy)["_artifacts"],
        "interval": fit_interval_models(Xtr, ytr, shutdown_days),
        "X": Xtr,
    }


def parts_from_artifacts(art: dict) -> dict:
    """메모리의 적합 객체 → `bundle_predict` 입력(부스터 묶음)."""
    reg = art["regime"]
    return {
        "columns": list(FEATURE_COLS),
        "gate": reg["gate"].booster_,
        "gate_features": list(reg["gcols"]),
        "classes": list(reg["classes"]),
        "regs": {t: {int(r): m.booster_ for r, m in d.items()} for t, d in reg["regs"].items()},
        "fallbacks": {t: m.booster_ for t, m in reg["fallbacks"].items()},
        "peak_clf": art["clf"]["clf"].booster_,
        "quantiles": {q: m.booster_ for q, m in art["interval"]["qmodels"].items()},
    }


def interval_halfwidths(oof: pd.DataFrame, alpha: float = INTERVAL_ALPHA) -> dict:
    """최종모델 OOF 절대잔차로 휴무일·운영일별 예측구간 반폭(qhat)을 구한다.

    5.8절 Split Conformal 은 최종모델이 아닌 별도 point 모델이 중심이라 서비스 출력에 쓰지 않는다.
    여기서는 **최종모델 예측 y_avg 를 중심**으로 두고, 휴무 여부별로 따로 보정한다
    유한표본 보정 분위를 쓰되, 모델 선택에 재사용한 OOF와 시계열 의존성 때문에
    운영 환경에서 90% 피복을 보장하는 구간으로 해석하지 않는다.
    """
    resid = (oof["y_avg"] - oof["pred_avg"]).abs().to_numpy()
    shut = operating_calendar["is_shutdown"].reindex(oof.index.normalize()).to_numpy().astype(bool)
    out = {}
    for key, m in (("operating", ~shut), ("shutdown", shut)):
        r = np.sort(resid[m])
        qhat = conformal_quantile(r, alpha=1.0 - alpha)
        if not np.isfinite(qhat):
            raise ValueError(f"{key}: 예측구간 보정 표본이 부족하다")
        out[key] = {"qhat": float(qhat), "n": int(len(r))}
    return out


def interval_bounds(y_avg_pred, is_shutdown, halfwidths: dict) -> tuple:
    """예측구간 [lo, hi]. 대상일 휴무 여부(피처 `is_shutdown`)로 반폭을 고른다. 하한은 0."""
    q = np.where(np.asarray(is_shutdown) == 1,
                 halfwidths["shutdown"]["qhat"], halfwidths["operating"]["qhat"])
    return np.maximum(y_avg_pred - q, 0.0), y_avg_pred + q


def build_golden(parts: dict) -> dict:
    """서빙 기동 자가검증 사례 — 원시 이력 21일 + 계획 24행 + 기대 피처 + 기대 출력.

    서빙은 이 원시 입력으로 피처를 다시 만들어 기대 피처와 비교하고(피처 코드의 의미 검증),
    기대 출력과 예측을 비교한다(부스터 로드 검증).
    """
    target = GOLDEN_ORIGIN + pd.Timedelta(days=1)
    ymd = df_raw["날짜"]
    lo = int((GOLDEN_ORIGIN - pd.Timedelta(days=GOLDEN_HISTORY_DAYS - 1)).strftime("%Y%m%d"))
    hist = df_raw.loc[(ymd >= lo) & (ymd <= int(GOLDEN_ORIGIN.strftime("%Y%m%d"))), HISTORY_COLUMNS]
    plan = df_raw.loc[ymd == int(target.strftime("%Y%m%d")), PLAN_COLUMNS]
    X = feat.loc[feat.index.normalize() == target, FEATURE_COLS]
    pred = bundle_predict(parts, X)

    def rows(d):
        return [[_native(v) for v in r] for r in d.itertuples(index=False, name=None)]

    return {
        "origin": str(GOLDEN_ORIGIN.date()),
        "target_date": str(target.date()),
        "history": {"columns": HISTORY_COLUMNS, "rows": rows(hist)},
        "plan": {"columns": PLAN_COLUMNS, "rows": rows(plan)},
        "expected_features": {"columns": list(FEATURE_COLS), "rows": X.to_numpy(float).tolist()},
        "expected": {k: np.asarray(v).tolist() for k, v in pred.items()},
    }


def build_manifest(role: str, art: dict, files: dict, shared: dict) -> dict:
    """매니페스트 — 타임스탬프·호스트명·절대경로 없이 결정적으로 만든다."""
    reg, clf = art["regime"], art["clf"]
    X = art["X"]
    files_sha = {n: _sha256(b) for n, b in sorted(files.items())}
    man = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "role": role,
        "fast_mode": bool(FAST),
        "model_name": SERVICE_MODEL_NAME,
        "report_final_model": FINAL_MODEL_NAME,
        "is_final_model": FINAL_MODEL_NAME == SERVICE_MODEL_NAME,
        "training": {
            "cond": "D1",
            "excluded_masks": ["is_warmup", "is_outage", "is_erp_missing"],
            "cutoff_exclusive": str(shared["cutoff"][role]),
            "n_rows": int(len(X)),
            "first": str(X.index.min()),
            "last": str(X.index.max()),
            "months": sorted(int(m) for m in X.index.month.unique()),
            "data_file": "data/okm_augumented_2021.csv",
            "data_sha256": shared["data_sha256"],
            "used_in_report": role == "eval",
        },
        "submission": shared["submission"] if role == "eval" else None,
        "feature_contract": {
            "columns": list(FEATURE_COLS),
            "dtype": "float64",
            "theta": float(THETA),
            "theta_note": "피크 정의 임계값(피처 roll*_peak_cnt 에도 쓰임). 서빙에서 다시 계산하지 않는다",
            "safe_lags": [int(x) for x in SAFE_LAGS],
            "holidays": [str(d.date()) for d in HOLIDAYS_2021],
            "holiday_coverage_start": str(df.index.min().date()),
            "holiday_coverage_end": str(TEST_END.date()),
            "history_days": HISTORY_DAYS,
            "plan_columns": PLAN_COLUMNS,
            "history_columns": HISTORY_COLUMNS,
        },
        "components": {
            "gate": {
                "file": "gate.lgb", "classes": list(reg["classes"]),
                "features": list(reg["gcols"]), "n_estimators": int(reg["gate_n_estimators"]),
                "regime_bins": [30.0, 70.0] if reg["n_classes"] == 3 else [70.0],
            },
            "regressors": {
                t: {str(int(r)): f"reg_{t}_r{int(r)}.lgb" for r in sorted(d)}
                for t, d in reg["regs"].items()
            },
            "fallbacks": {t: f"fallback_{t}.lgb" for t in reg["fallbacks"]},
            "peak_classifier": {
                "file": "peak_clf.lgb",
                "params": clf["params"],
                "n_estimators": clf["n_estimators"],
                "scale_pos_weight": clf["scale_pos_weight"],
                "effective_note": "탐색과 최종 적합 모두 bagging_freq=1을 유지하여 bagging_fraction을 적용",
            },
            "quantiles": {str(q): f"interval_q{int(round(q * 100))}.lgb" for q in sorted(art["interval"]["qmodels"])},
            "calibrator": "calibrator_isotonic.json",
            "golden": "golden.json",
        },
        "thresholds": shared["thresholds"],
        "intervals": {
            "quantile": {
                "calibrated": False,
                "fit_last": art["interval"]["fit_last"],
                "measured_test_coverage_q10_q90": shared["quantile_coverage"] if role == "eval" else None,
                "note": "분위회귀는 미보정이다. 실제 테스트 피복률은 기록된 값으로 확인하며 운영 피복을 보장하지 않음",
            },
            "residual": shared["residual"],
        },
        "hyperparameters": {
            "lgb_base": dict(LGB_BASE),
            "n_estimators": int(N_ESTIMATORS),
            "regime_uses_optuna": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "caveats": [
            f"학습 데이터의 month 범위는 {sorted(int(m) for m in X.index.month.unique())} 이다. 그 밖의 달은 외삽이다",
            "공휴일표 HOLIDAYS_2021 은 2021-09-14 까지만 검증됐다. 이후 날짜는 운영 휴일표를 주입해야 한다",
            "Isotonic 보정기는 fold 모델의 OOF 확률로 적합됐고, 보정 후 ECE 0 은 적합 데이터 위의 값이다",
            "경보 라벨은 보정 전 확률과 tau_cls, 또는 y_peak 예측과 tau 로 정한다",
        ],
        "files_sha256": files_sha,
    }
    # 매니페스트 자신까지 해시한다 — bundle_id 는 모델 파일과 임계값·계약 전부의 지문이다
    man["manifest_sha256"] = manifest_core_sha256(man)
    man["bundle_id"] = f"{role}-{'fast' if FAST else 'full'}-{man['manifest_sha256'][:12]}"
    return man


def write_bundle(bundle_dir, files: dict, manifest: dict) -> None:
    """번들 디렉터리를 새로 쓴다(이전 실행의 잔여 파일이 섞이지 않도록 먼저 지운다)."""
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    for name, b in files.items():
        (bundle_dir / name).write_bytes(b)
    (bundle_dir / "manifest.json").write_bytes(_json_bytes(manifest))


def read_bundle_parts(bundle_dir) -> tuple:
    """번들을 검증하며 읽는다(노트북 재로드 검증용 최소 로더).

    파일을 **바이트로** 읽어 sha256 을 대조한 뒤 `Booster(model_str=...)` 로 연다.
    서빙 로더(`serving/bundle.py`)는 여기에 버전·FAST·자가검증 정책을 더한다.

    Returns
    -------
    (dict, dict, dict)
        매니페스트, `bundle_predict` 입력, `{파일명: 트리 수}`.
    """
    man = json.loads((bundle_dir / "manifest.json").read_bytes())
    if manifest_core_sha256(man) != man["manifest_sha256"] or not man["bundle_id"].endswith(man["manifest_sha256"][:12]):
        raise AssertionError("매니페스트 sha256 불일치")
    raw = {}
    for name, sha in man["files_sha256"].items():
        b = (bundle_dir / name).read_bytes()
        if _sha256(b) != sha:
            raise AssertionError(f"sha256 불일치: {name}")
        raw[name] = b
    boosters = {n: lgb.Booster(model_str=b.decode("utf-8")) for n, b in raw.items() if n.endswith(".lgb")}
    comp = man["components"]
    parts = {
        "columns": man["feature_contract"]["columns"],
        "gate": boosters[comp["gate"]["file"]],
        "gate_features": comp["gate"]["features"],
        "classes": comp["gate"]["classes"],
        "regs": {t: {int(r): boosters[f] for r, f in d.items()} for t, d in comp["regressors"].items()},
        "fallbacks": {t: boosters[f] for t, f in comp["fallbacks"].items()},
        "peak_clf": boosters[comp["peak_classifier"]["file"]],
        "quantiles": {float(q): boosters[f] for q, f in comp["quantiles"].items()},
    }
    return man, parts, {n: int(b.num_trees()) for n, b in boosters.items()}


def text_is_clean(b: bytes) -> bool:
    """매니페스트·JSON 에 절대경로나 실행 계정명이 없는지 본다(블라인드 평가)."""
    text = b.decode("utf-8")
    if any(t in text for t in (":\\", ":/", "/home/", "\\Users", "/Users/")):
        return False
    # 계정명은 **경로 문맥**에서만 본다 — 'app'·'test'·'data' 같은 흔한 계정명이
    # 매니페스트의 일반 단어("apply", "data_file")에 걸려 노트북을 멈추지 않게 한다
    user = Path.home().name
    return not (user and re.search(r"[\\/]" + re.escape(user) + r"(?:[\\/\"]|$)", text))


def outputs_equal(a: dict, b: dict) -> bool:
    """두 `bundle_predict` 결과가 비트 단위로 같은지."""
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def run_bundle_section() -> dict:
    """평가·배포 번들을 만들고 검증한다. 실패하면 AssertionError 로 멈춘다."""
    checks: dict = {}
    te_s, te_c = test_results[SERVICE_MODEL_NAME], test_results[CLF_MODEL_NAME]
    Xte = feat.loc[te_s["index"], FEATURE_COLS]
    is_final = FINAL_MODEL_NAME == SERVICE_MODEL_NAME

    # ── 교차검증 산출물(τ·τ_cls·보정기·구간 반폭) — 두 번들이 공유 ────────────
    clf_oof = cv_results[CLF_MODEL_NAME]["oof"]
    tau = float(cv_results[SERVICE_MODEL_NAME]["tau"])
    tau_cls = float(tune_tau(clf_oof["y_peak"].to_numpy(), clf_oof["prob"].to_numpy(), THETA))
    halfwidths = interval_halfwidths(cv_results[SERVICE_MODEL_NAME]["oof"])
    lo, hi = interval_bounds(te_s["pred_avg"], Xte["is_shutdown"].to_numpy(), halfwidths)
    y_te = te_s["y_avg"]
    halfwidths["measured_test_coverage"] = round(float(((y_te >= lo) & (y_te <= hi)).mean()), 4)
    halfwidths.update({
        "alpha": INTERVAL_ALPHA, "center": "y_avg_pred",
        "fit_on": f"{SERVICE_MODEL_NAME} OOF 절대잔차 (fold2~5)",
    })
    iso = _calib["iso"]
    calibrator = {
        "x_thresholds": [float(x) for x in iso.X_thresholds_],
        "y_thresholds": [float(y) for y in iso.y_thresholds_],
        "out_of_bounds": "clip",
        "fit_on": f"{CLF_MODEL_NAME} OOF 확률 (fold2~5)",
        "apply": "np.interp(prob, x_thresholds, y_thresholds)",
    }
    thresholds = {
        "tau": {"value": tau, "unit": "kW",
                "fold_taus": [float(x) for x in cv_results[SERVICE_MODEL_NAME]["fold_metrics"]["τ"]],
                "method": "fold2~5 검증구간 tune_tau(F1 최대) 의 중앙값", "rule": "y_peak_pred >= tau"},
        "tau_cls": {"value": tau_cls, "method": "피크 직접분류 OOF 확률에 tune_tau", "rule": "peak_prob >= tau_cls"},
        "inherited_from": None,
    }
    pred_path = OUTPUT_DIR / "predictions_test_336h.csv"
    shared = {
        "cutoff": {"eval": TEST_START, "deploy": TEST_END + pd.Timedelta(hours=1)},
        "data_sha256": _sha256(Path(DATA_PATH).read_bytes()),
        "submission": {
            "file": "outputs/predictions_test_336h.csv",
            "sha256": _sha256(pred_path.read_bytes()),
            "bound": is_final,
        },
        "thresholds": thresholds,
        "residual": halfwidths,
        "quantile_coverage": round(float(((y_te >= _unc["lo"]) & (y_te <= _unc["hi"])).mean()), 4),
    }
    calib_bytes = _json_bytes(calibrator)

    # ── ① 평가 번들: 제출 예측을 만든 바로 그 적합 객체를 저장한다 ─────────────
    eval_art = {
        "regime": te_s["_artifacts"], "clf": te_c["_artifacts"], "interval": _unc["_fit"],
        "X": training_frame(TEST_START)[0],
    }
    eval_files = component_files(eval_art)
    eval_files["calibrator_isotonic.json"] = calib_bytes
    eval_files["golden.json"] = _json_bytes(build_golden(parts_from_artifacts(eval_art)))
    eval_man = build_manifest("eval", eval_art, eval_files, shared)
    eval_dir = BUNDLE_BASE / "eval"
    write_bundle(eval_dir, eval_files, eval_man)

    # 재로드 → 메모리 예측(제출 예측을 만든 값)과 비트 비교
    man_e, parts_e, trees_e = read_bundle_parts(eval_dir)
    got = bundle_predict(parts_e, Xte)
    checks["eval_bitwise"] = bool(
        np.array_equal(got["y_avg"], te_s["pred_avg"])
        and np.array_equal(got["y_peak"], te_s["pred_peak"])
        and np.array_equal(got["prob"], te_c["prob"])
        and np.array_equal(got["q10"], _unc["lo"])
        and np.array_equal(got["q50"], _unc["mid"])
        and np.array_equal(got["q90"], _unc["hi"])
    )
    if is_final:
        sub = predictions
        checks["submission_match"] = bool(
            np.array_equal(np.round(got["y_avg"], 4), sub["y_avg_pred"].to_numpy())
            and np.array_equal(np.round(got["y_peak"], 4), sub["y_peak_pred"].to_numpy())
            and np.array_equal((got["y_peak"] >= tau).astype(int), sub["peak_pred_label"].to_numpy())
            and np.array_equal(np.round(got["prob"], 6), sub["peak_prob"].to_numpy())
        )
    else:
        checks["submission_match"] = True  # 해당 없음: 제출 파일은 다른 모델로 만들어졌다

    # ── ② 재학습 경로의 결정성: 같은 cutoff 로 다시 적합하면 바이트까지 같아야 한다 ──
    refit_eval = fit_service_artifacts(TEST_START)
    refit_files = component_files(refit_eval)
    checks["refit_sha"] = bool(
        {n: _sha256(b) for n, b in refit_files.items()}
        == {n: _sha256(b) for n, b in eval_files.items() if n.endswith(".lgb")}
        and refit_eval["interval"]["qhat"] == _unc["_fit"]["qhat"]
    )

    # ── ③ 배포 번들: 전 구간(~09-14) 재학습. CV 산출물은 평가 번들에서 상속한다 ──
    deploy_art = fit_service_artifacts(TEST_END + pd.Timedelta(hours=1))
    deploy_shared = {**shared, "thresholds": {**thresholds, "inherited_from": man_e["bundle_id"]},
                     "residual": {**halfwidths, "inherited_from": man_e["bundle_id"]}}
    deploy_files = component_files(deploy_art)
    deploy_files["calibrator_isotonic.json"] = calib_bytes
    deploy_parts_mem = parts_from_artifacts(deploy_art)
    deploy_files["golden.json"] = _json_bytes(build_golden(deploy_parts_mem))
    deploy_man = build_manifest("deploy", deploy_art, deploy_files, deploy_shared)
    deploy_dir = BUNDLE_BASE / "deploy"
    write_bundle(deploy_dir, deploy_files, deploy_man)
    man_d, parts_d, trees_d = read_bundle_parts(deploy_dir)
    checks["deploy_reload"] = outputs_equal(bundle_predict(parts_d, Xte), bundle_predict(deploy_parts_mem, Xte))

    # ── ④ 무결성·개인정보 ───────────────────────────────────────────────────
    checks["integrity"] = bool(checks["deploy_reload"]) and man_e["files_sha256"] == {
        n: _sha256(b) for n, b in sorted(eval_files.items())
    }
    json_files = [p for p in BUNDLE_BASE.rglob("*.json")]
    checks["manifest_clean"] = all(text_is_clean(p.read_bytes()) for p in json_files)

    # 순수 추론시간(336h) — 출력만 한다. 보고서의 INFER_SEC 는 바꾸지 않는다.
    times = []
    for _ in range(10):
        t0 = _time.perf_counter()
        bundle_predict(parts_e, Xte)
        times.append(_time.perf_counter() - t0)

    rows = []
    for man, trees, bdir in ((man_e, trees_e, eval_dir), (man_d, trees_d, deploy_dir)):
        for name, sha in list(man["files_sha256"].items()) + [
            ("manifest.json", _sha256((bdir / "manifest.json").read_bytes()))
        ]:
            rows.append({
                "번들": man["role"], "모드": "FAST" if man["fast_mode"] else "FULL",
                "bundle_id": man["bundle_id"], "파일": name, "sha256": sha,
                "bytes": len((bdir / name).read_bytes()),
                "트리수": trees.get(name, 0), "학습행수": man["training"]["n_rows"],
            })
    return {
        "checks": checks, "table": pd.DataFrame(rows),
        "eval_id": man_e["bundle_id"], "deploy_id": man_d["bundle_id"],
        "infer_sec": float(np.median(times)), "tau": tau, "tau_cls": tau_cls,
        "halfwidths": halfwidths, "is_final": is_final,
        "n_eval": man_e["training"]["n_rows"], "n_deploy": man_d["training"]["n_rows"],
    }


if os.environ.get("KAMP_SKIP_BUNDLE") == "1":
    # 비상 탈출구 — 번들 버그 때문에 제출 노트북 실행 전체를 잃지 않게 한다. 최종 게이트는 실패로 남는다.
    BUNDLE_CHECKS = {k: False for k in ("integrity", "eval_bitwise", "submission_match", "refit_sha", "manifest_clean")}
    BUNDLE_SUMMARY = None
    # 이전 실행의 번들·표가 남아 있으면 오래된 산출물을 가리키게 되므로 지운다
    if BUNDLE_BASE.exists():
        shutil.rmtree(BUNDLE_BASE)
    (TBL_DIR / "ch6_model_bundles.csv").unlink(missing_ok=True)
    print("⚠️ KAMP_SKIP_BUNDLE=1 — 모델 번들 저장을 건너뛰었다 (최종 게이트 미통과로 기록)")
else:
    BUNDLE_SUMMARY = run_bundle_section()
    BUNDLE_CHECKS = BUNDLE_SUMMARY["checks"]
    save_table(BUNDLE_SUMMARY["table"], "ch6_model_bundles")
    BUNDLE_INFER_SEC = BUNDLE_SUMMARY["infer_sec"]
    print(f"── 모델 번들 저장 ({'FAST' if FAST else 'FULL'}) ──")
    print(f"  평가 번들 {BUNDLE_SUMMARY['eval_id']}  (학습 {BUNDLE_SUMMARY['n_eval']:,}행, ~08-31)")
    print(f"  배포 번들 {BUNDLE_SUMMARY['deploy_id']}  (학습 {BUNDLE_SUMMARY['n_deploy']:,}행, ~09-14)")
    print(f"  τ = {BUNDLE_SUMMARY['tau']!r} kW (반올림 없음) · τ_cls = {BUNDLE_SUMMARY['tau_cls']!r}")
    _hw = BUNDLE_SUMMARY["halfwidths"]
    print(
        f"  예측구간 반폭 운영일 ±{_hw['operating']['qhat']:.2f} / 휴무일 ±{_hw['shutdown']['qhat']:.2f} kW"
        f" → 테스트 피복률 {_hw['measured_test_coverage']:.3f} (목표 {INTERVAL_ALPHA})"
    )
    print(f"  번들 순수 추론 336시간 = {BUNDLE_INFER_SEC * 1000:.1f} ms (10회 중앙값, 보고서 INFER_SEC 와 별개)")
    display(pd.DataFrame(list(BUNDLE_CHECKS.items()), columns=["점검", "통과"]))
    if not all(BUNDLE_CHECKS.values()):
        raise AssertionError(f"모델 번들 검증 실패: {[k for k, v in BUNDLE_CHECKS.items() if not v]}")

# %% [markdown]
# ### 10.6 발표자료 골격 · 개인정보 스캔
#
# - **목적**: 발표자료 pptx 골격을 만들고, **제출 전 개인정보를 스캔**한다.
# - **보고서 대응절**: 6장, 제출물
# - **산출물**: `outputs/발표자료.pptx`, `ch6_privacy_scan.csv`
#
# 블라인드 평가이므로 소속·학교·로고 등 식별정보가 있으면 안 된다.
# **1건이라도 걸리면 빌드를 중단**해야 한다.
#
# > ### ⚠️ 이 셀이 볼 수 없는 것 — 자기 실행 출력
# >
# > `nbconvert --execute --inplace` 는 **모든 셀이 끝난 뒤에** 파일을 쓴다.
# > 따라서 이 셀이 읽는 노트북 파일에는 **아직 출력이 비어 있다.**
# > 서드파티 경고가 stderr 로 내보내는 설치 경로(사용자명 포함)는
# > 실행이 끝난 뒤에야 파일에 박히므로 여기서는 절대 잡을 수 없다.
# >
# > 그래서 제출 직전에 **반드시** 다음을 실행한다.
# >
# > ```bash
# > python tools/finalize_notebook.py
# > ```
# >
# > 이 스크립트가 stderr 출력을 제거하고 잔여 경로를 치환한 뒤 후스캔한다.
# > (이 한계는 실제로 사용자명 6건이 출력에 남아 적발되면서 확인되었다)

# %%
BETA_STR = f"{BETA:.5f}"


def build_pptx() -> object:
    """발표자료 골격을 만든다 (핵심 그림 위주)."""
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except Exception as e:  # pragma: no cover
        print(f"  python-pptx 미설치로 건너뜀 ({type(e).__name__})")
        return None

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    def add_title_slide(title, subtitle):
        s = prs.slides.add_slide(prs.slide_layouts[0])
        s.shapes.title.text = title
        s.placeholders[1].text = subtitle
        return s

    def add_fig_slide(title, fid, bullet):
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = title
        hits = sorted(FIG_DIR.glob(f"{fid}_*.png"))
        if hits:
            s.shapes.add_picture(str(hits[0]), Inches(0.7), Inches(1.5), width=Inches(8.2))
        tb = s.shapes.add_textbox(Inches(9.2), Inches(1.6), Inches(3.6), Inches(4.5))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = bullet
        for p in tf.paragraphs:
            p.font.size = Pt(13)
        return s

    add_title_slide(
        "제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석",
        "제6회 K-인공지능 제조데이터 분석 경진대회 · 과제 ⑤ 자원 최적화",
    )
    add_fig_slide(
        "데이터의 44.7%가 합성 복제일",
        "F02",
        f"· 고유 프로파일 {N_UNIQUE_PROFILES}/{N_TOTAL_DAYS}일\n"
        f"· 복제 {N_SURPLUS_DAYS}일 (행 단위 중복 0건)\n"
        "· 전력은 동일, 기온은 -3.4~19.2°C\n"
        "· 오염 fold 폐기 → 성능 과대추정 차단",
    )
    add_fig_slide(
        "양방향 결측을 대칭 진단",
        "F04",
        "· 센서측 17시간 (평균전력 0)\n"
        "· ERP측 48시간 (생산량 0, 전력 정상)\n"
        "· 야간 피크 7건이 전부 ERP 결측일\n"
        "· 서로 다른 처리 규칙 적용",
    )
    add_fig_slide(
        "Day-ahead 누수 차단",
        "F13",
        "· origin = D일 24:00\n"
        "· 안전 lag 24/48/168 만\n"
        "· 이동통계는 원점 1회 계산 후 브로드캐스트\n"
        "· 미래맹검 동등성 검사로 기계적 증명",
    )
    add_fig_slide(
        "가이드북 베이스라인 결함 증명",
        "F16",
        "· 셀 73 `iloc[0]` → 실질 피처 1개\n"
        "· 학습 MSE > 테스트 MSE (과적합 0)\n"
        "· 무음 미반영 3곳 (Weekend/Vacation 전부 0)\n"
        "· 결함 13건을 표로 정리",
    )
    add_fig_slide(
        f"최종모델: {FINAL_MODEL_NAME}",
        "F17",
        f"· 테스트 MAE {FINAL_MAE:.2f} ({IMPROVE_MAE_PCT:+.1f}%)\n"
        f"· Recall {FINAL_RECALL:.3f} / F1 {FINAL_F1:.3f}\n"
        f"· 추론 {INFER_SEC:.2f}초 (336시간 배치)\n"
        f"· D2 순위상관 {float(rank_preservation.iloc[0]['값']):.3f}",
    )
    add_fig_slide(
        "달력규칙 기준선이 Recall 1.000",
        "F19",
        f"· 평일 ∧ 08~18시 → Recall 1.000, FP {CALENDAR_RULE_FP}\n"
        "· Recall 로는 어떤 모델도 이길 수 없다\n"
        "· 주지표를 F1/PR-AUC 로 재정의\n"
        "· Recall과 오경보를 함께 비교(동일 Recall을 가정하지 않음)",
    )
    add_fig_slide(
        "피크 저감: 동시성 분산이 핵심",
        "F35",
        "· 08·13시만 분산 → Δ최대수요 0 kW\n"
        "· 09·11시까지 포함 → 13 kW\n"
        "· 청구 피크는 최대값 → 상위 사건 전부 덮어야\n"
        f"· 생산량 이동 β={BETA_STR} kW/단위 (지렛대 미미)",
    )
    add_fig_slide(
        "현장 운영 프로토콜",
        "F39",
        "· 전일 24:00 예측 → 24:10 조정안\n"
        "· 익일 07:30 계단식 기동 지시\n"
        "· 위험시간 실시간 감시\n"
        "· 규칙 R1~R3 과 연결된 점검 우선순위",
    )
    return prs


_prs = build_pptx()
if _prs is not None:
    _prs.save(str(OUTPUT_DIR / "발표자료.pptx"))
    print(f"── 발표자료.pptx 생성 ({len(_prs.slides._sldIdLst)}슬라이드) ──")


# ⚠️ 사용자명을 **소스에 literal 로 쓰지 않는다.**
# 여기에 이름을 적으면 그 이름이 노트북·스캔결과 CSV 에 그대로 실려
# 블라인드 평가 위반이 된다(실제로 그렇게 적발되었다).
# 실행 시점의 홈 디렉터리명에서 유도하면 어떤 심사 PC 에서도 동작한다.
from pathlib import Path as _Path  # noqa: E402

_USER = _Path.home().name

PRIVACY_PATTERNS = {
    "사용자명": re.escape(_USER),
    # 역슬래시는 `\\+` 로 써야 잡힌다. `\+` 는 리터럴 '+' 라 실제 윈도 사용자 경로를 한 번도 못 잡았다.
    # (이 주석에 예시 경로를 쓰지 않는다 — finalize 가 셀 소스를 그대로 스캔한다)
    "Windows 절대경로": r"[Cc]:\\+[Uu]sers",
    "홈 경로": r"/(?:home|Users)/[A-Za-z0-9_가-힣]+",
    "이메일": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "절대경로 출력": r"os\.getcwd\(\)",
}


def _mask(text: str) -> str:
    """발견된 문자열을 마스킹한다.

    스캔 결과 CSV 는 제출물에 포함되므로, 발견 내용을 원문 그대로 적으면
    스캐너가 오히려 개인정보를 유출한다. 첫 글자만 남기고 가린다.
    """
    text = text[:60]
    return text[:1] + "*" * max(len(text) - 1, 0)


ALLOWED_TEAM_NAME = "정종묵"   # 팀명·성명은 허용


# 제출 zip 에 포함되지 않는 개발 전용 문서 (스캔은 하되 차단 사유로 삼지 않는다)
DEV_ONLY = {"PRD.md", "CHECKLIST.md", "plan.md"}
# 스캐너 자신과 그 출력은 검사 대상에서 제외한다.
# 패턴 정의 문자열과 스캔 결과표가 매번 '발견'으로 잡히는 자기참조를 막는다.
SCANNER_SELF = {"s10_package.py", "ch6_privacy_scan.csv", "ch6_forbidden_scan.csv"}


def _scan_text(text: str, source: str, kind: str, rows: list) -> None:
    """한 덩어리 텍스트에 패턴을 적용해 결과를 rows 에 쌓는다."""
    for label, pat in PRIVACY_PATTERNS.items():
        # `os.getcwd()` 는 설명 문구에서도 등장하므로 코드에서만 위반으로 본다
        if label == "절대경로 출력" and kind != "code":
            continue
        for m in re.finditer(pat, text, re.MULTILINE):
            rows.append(
                {"파일": source, "위치": kind, "유형": label, "발견": _mask(m.group())}
            )


def _strip_comments(src: str) -> str:
    """파이썬 주석 줄을 제거한다 (설명 문구를 위반으로 오탐하지 않게)."""
    return chr(10).join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )


def scan_privacy() -> pd.DataFrame:
    """제출 대상 파일에서 개인정보·절대경로를 스캔한다.

    노트북은 **JSON 원문을 그대로 훑지 않는다.** JSON 에서는 모든 소스 줄이
    따옴표로 감싸져 있어 주석 판별이 불가능하고, 이스케이프 때문에 패턴도
    어긋난다. `nbformat` 으로 파싱해 다음 세 곳을 구분해서 본다.

    - 코드 셀 소스 (주석 제거 후)
    - 마크다운 셀 소스 (설명이므로 PII 만 검사)
    - **셀 실행 출력** — 서드파티 경고에 설치 경로가 박혀 들어오는 주 경로다

    소스만 보면 실행 출력에 각인된 절대경로를 놓친다.
    """
    rows: list[dict] = []

    nb_path = PROJECT_ROOT / "자원최적화_제안모델.ipynb"
    if nb_path.exists():
        import nbformat

        nb = nbformat.read(nb_path, as_version=4)
        for i, cell in enumerate(nb.cells):
            src = cell.get("source", "") or ""
            if cell.cell_type == "code":
                _scan_text(_strip_comments(src), nb_path.name, "code", rows)
            else:
                _scan_text(src, nb_path.name, "markdown", rows)
            for out in cell.get("outputs", []) or []:
                chunks = []
                if "text" in out:
                    chunks.append(out["text"])
                for key in ("data",):
                    d = out.get(key) or {}
                    if isinstance(d, dict):
                        chunks.append(str(d.get("text/plain", "")))
                if out.get("output_type") == "error":
                    chunks.append(chr(10).join(out.get("traceback", []) or []))
                for c in chunks:
                    _scan_text(str(c), nb_path.name, "output", rows)

    targets = []
    targets += sorted(PROJECT_ROOT.glob("*.md"))
    targets += sorted(PROJECT_ROOT.glob("*.txt"))
    targets += sorted(OUTPUT_DIR.glob("*.md"))
    targets += sorted(TBL_DIR.glob("*.csv"))
    targets += sorted(PROJECT_ROOT.glob("src/*.py"))
    targets += sorted(PROJECT_ROOT.glob("tests/*.py"))
    # 10.5절 모델 번들의 매니페스트·자가검증 JSON (절대경로·계정명이 들어가면 안 된다)
    targets += sorted(MODEL_DIR.rglob("*.json"))

    for f in targets:
        if f.name in SCANNER_SELF:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        kind = "code" if f.suffix == ".py" else "doc"
        body = _strip_comments(text) if f.suffix == ".py" else text
        _scan_text(body, str(f.relative_to(PROJECT_ROOT)), kind, rows)

    out = pd.DataFrame(rows, columns=["파일", "위치", "유형", "발견"])
    if len(out):
        out["제출대상"] = ~(
            out["파일"].isin(DEV_ONLY) | out["파일"].str.startswith("tests")
        )
    else:
        out["제출대상"] = pd.Series(dtype=bool)
    return out


privacy_scan = scan_privacy()
save_table(privacy_scan, "ch6_privacy_scan")
# 제출 zip 에 실제로 들어가는 파일에서 발견된 것만 차단 사유로 삼는다.
# PRD.md·CHECKLIST.md·plan.md·tests/ 는 개발 전용이라 zip 에 포함하지 않는다.
privacy_blocking = (
    privacy_scan[privacy_scan["제출대상"]]
    if len(privacy_scan) and "제출대상" in privacy_scan.columns
    else privacy_scan.iloc[0:0]
)
print(
    f"\n── 개인정보 스캔: 전체 {len(privacy_scan)}건 / "
    f"**제출 대상 {len(privacy_blocking)}건** (개발 전용 문서는 zip 미포함) ──"
)
if len(privacy_scan) and {"유형", "파일"} <= set(privacy_scan.columns):
    display(privacy_scan.groupby(["유형", "파일", "위치", "제출대상"]).size().to_frame("건수").head(30))
    if len(privacy_blocking):
        print("\n  ⚠️ 제출 대상 파일에서 발견 — **zip 빌드 전에 반드시 제거**해야 한다.")
    else:
        print("\n  ✅ 제출 대상 파일에는 0건 (발견분은 전부 개발 전용 문서)")
else:
    print("  ✅ 개인정보·절대경로 0건")

# 노트북 내 금지 패턴 점검
# 문서·주석에 등장하는 '언급'이 아니라 **실제 호출**만 잡도록 좁힌다.
# (4.4절 결함표가 `freq='H'` 를 설명 문구로 담고 있어 단순 grep 은 오탐한다)
FORBIDDEN_IN_NOTEBOOK = {
    "이중 축 twinx()": r"^(?!\s*#).*\.twinx\(\)",
    "pandas 3.x 크래시 freq='H'": (
        r"^(?!\s*#).*(?:date_range|resample|period_range|Grouper)\s*\([^)]*"
        r"freq\s*=\s*['\"]H['\"]"
    ),
}


def scan_forbidden() -> pd.DataFrame:
    """노트북·stage 소스에서 금지 패턴을 찾는다."""
    rows = []
    for f in sorted(PROJECT_ROOT.glob("src/*.py")):
        if f.name in SCANNER_SELF:
            continue  # 패턴 정의 자체를 잡지 않는다
        text = f.read_text(encoding="utf-8")
        for label, pat in FORBIDDEN_IN_NOTEBOOK.items():
            hits = len(re.findall(pat, text, re.MULTILINE))
            if hits:
                rows.append({"파일": f.name, "금지 패턴": label, "건수": hits})
    return pd.DataFrame(rows, columns=["파일", "금지 패턴", "건수"])


forbidden_scan = scan_forbidden()
save_table(forbidden_scan, "ch6_forbidden_scan")
print(f"\n── 금지 패턴 스캔: {len(forbidden_scan)}건 ──")
if len(forbidden_scan):
    display(forbidden_scan)
else:
    print("  ✅ 이중 축(twinx)·freq='H' 0건")

# %% [markdown]
# ### 최종 게이트 — 제출 준비 상태 점검

# %%
# requirements.txt·README 가 10.4에서 생성되므로 장별 체크를 **여기서 다시** 돌린다.
# (10.2 시점에는 아직 파일이 없어 6장이 FAIL 로 잡힌다)
chapter_checks = check_chapters()
save_table(chapter_checks, "ch6_chapter_checklist")
_tbd_md = write_tbd_filled()
(OUTPUT_DIR / "report_tbd_filled.md").write_text(_tbd_md, encoding="utf-8")
print("── 장별 충족 체크리스트 (최종 재계산) ──")
display(chapter_checks.attrs["summary"].to_frame())


def final_gate() -> pd.DataFrame:
    """최종 게이트 — 제출 전 확인 항목."""
    nfig = len(list(FIG_DIR.glob("F*.png")))
    fig_idx = pd.read_csv(FIGURE_INDEX_PATH, encoding="utf-8-sig") if FIGURE_INDEX_PATH.exists() else pd.DataFrame()
    checks = [
        ("예측결과 336행·결측 0", len(predictions) == 336 and int(predictions.isna().sum().sum()) == 0),
        ("예측결과 9컬럼 규격", list(predictions.columns) == [
            "datetime", "y_avg_true", "y_avg_pred", "y_peak_true", "y_peak_pred",
            "peak_prob", "peak_pred_label", "peak_true_label", "model_name"]),
        ("그림 생성 수 == 인덱스 행 수", nfig == len(fig_idx)),
        ("장별 요건 전부 충족", bool(chapter_checks["충족"].all())),
        ("4·5·6장 본문 초안 존재", all(
            (OUTPUT_DIR / f"report_ch{i}_draft.md").exists() for i in (4, 5, 6))),
        ("requirements.txt 존재", (PROJECT_ROOT / "requirements.txt").exists()),
        ("진입 문서(00_README.md) 존재", (PROJECT_ROOT / "00_README.md").exists()),
        ("환경 증거 생성", (OUTPUT_DIR / "requirements_generated.txt").exists()),
        ("이중 축·freq='H' 0건", len(forbidden_scan) == 0),
        ("제출 대상 개인정보 0건", len(privacy_blocking) == 0),
        ("채움표 30항목 이상", len(tbd_tbl) >= 30),
        # 10.5절 모델 번들 — 번들 절이 실패하면 이미 예외로 멈추므로 여기서는 결과만 기록한다
        ("서비스 번들 2종 생성·sha256 무결성", bool(BUNDLE_CHECKS.get("integrity"))),
        ("평가 번들 재로드 예측 == 메모리 예측(비트)", bool(BUNDLE_CHECKS.get("eval_bitwise"))),
        ("번들 예측 == 제출 파일 (최종모델=번들모델일 때)", bool(BUNDLE_CHECKS.get("submission_match"))),
        ("재학습 경로 sha 재현 · 번들 JSON 경로·계정명 0건",
         bool(BUNDLE_CHECKS.get("refit_sha")) and bool(BUNDLE_CHECKS.get("manifest_clean"))),
    ]
    out = pd.DataFrame(checks, columns=["점검", "통과"])
    return out


gate_final = final_gate()
save_table(gate_final, "gate6_final")
print("── 최종 게이트 ──")
display(gate_final)
_nf = gate_final[~gate_final["통과"]] if "통과" in gate_final.columns else gate_final.iloc[0:0]
if len(_nf):
    print(f"\n  ⚠️ 미통과: {_nf['점검'].tolist()}")
else:
    print("\n  ✅ 최종 게이트 전 항목 통과")

print(f"\n{'=' * 64}")
print("제출 준비 요약")
print("=" * 64)
print(f"  최종모델        : {FINAL_MODEL_NAME}")
print(f"  테스트 MAE      : {FINAL_MAE:.3f} ({IMPROVE_MAE_PCT:+.2f}% vs {BEST_BASELINE_NAME})")
print(f"  Recall / F1     : {FINAL_RECALL:.3f} / {FINAL_F1:.3f}")
print(f"  최대수요 저감   : {PEAK_REDUCTION_KW:.1f} kW (전체기간 기준)")
print(f"  그림            : {len(list(FIG_DIR.glob('F*.png')))}장")
print(f"  표              : {len(list(TBL_DIR.glob('*.csv')))}개")
print(f"  FAST 모드       : {FAST}  {'← 제출 전 FULL 재실행 필요' if FAST else ''}")
if BUNDLE_SUMMARY is not None:
    print(f"  모델 번들       : {BUNDLE_SUMMARY['eval_id']} / {BUNDLE_SUMMARY['deploy_id']}")
print("=" * 64)
