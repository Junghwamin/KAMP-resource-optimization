# %% tags=["nb-strip"]
from s00_env import COLOR_HERO, INK, OUTPUT_DIR, display, save_table  # noqa: F401
from s01_diagnose import (  # noqa: F401
    N_DUP_GROUPS,
    N_SURPLUS_DAYS,
    N_TOTAL_DAYS,
    N_UNIQUE_PROFILES,
    THETA,
    df,
)
from s03_split import CALENDAR_RULE_FP, calendar_rule_metrics  # noqa: F401
from s05_models import calibration_tbl, ensemble_info, uncertainty_tbl  # noqa: F401
from s06_eval import (  # noqa: F401
    BEST_BASELINE_NAME,
    FINAL_MODEL_NAME,
    FINAL_MAE,
    FINAL_RECALL,
    IMPROVE_MAE_PCT,
    MAE_IMPROVEMENT_SIGNIFICANT,
    ablation_tbl,
    interpretation_tbl,
    peak_tbl,
    rank_preservation,
)
from s07_analysis import OOF, TOP_FEATURES, WORST_CONDITION_NAME, WORST_COND_EXCESS_PCT, rules_tbl  # noqa: F401
from s08_simulation import (  # noqa: F401
    BETA,
    PEAK_REDUCTION_KW,
    SEPT_ONLY_SAVING,
    levers_tbl,
    scenarios_tbl,
)
import pandas as pd

# %% [markdown]
# ## 9. 창의성·차별성 근거 산출
#
# 보고서 **5장 창의성 및 차별성(10점)** 의 근거를 생성한다.
#
# > 채점 기준은 "새로운 알고리즘을 썼다"가 아니라
# > **"제안한 방법이 성능·오류 감소·현장 활용에 어떤 기여를 했는지를 2·3장 결과에 연결"** 하는 것이다.
# > 따라서 이 장은 새 실험을 하지 않고, 앞 장에서 **이미 측정된 수치**를 차별점으로 재구성한다.

# %% [markdown]
# ### 9.1 ~ 9.5 차별점 5종 정리
#
# - **목적**: 각 차별점을 "무엇을 했는가 → 어떤 수치로 확인되는가 → 어디에 기여하는가"로 정리한다.
# - **보고서 대응절**: 5장
# - **산출물**: `ch5_creativity.csv`

# %%
def build_creativity_table() -> pd.DataFrame:
    """실행 결과에 따른 차별점과 한계를 함께 정리한다."""
    final = peak_tbl[peak_tbl["모델"] == FINAL_MODEL_NAME].iloc[0]
    regime = ablation_tbl[ablation_tbl["제거 피처군"] == "레짐분리"]
    regime_text = (
        f"레짐분리 제거 시 OOF MAE 변화는 {float(regime['MAE 변화'].iloc[0]):+.3f} kW였다. "
        if len(regime) else "레짐분리 제거 결과는 해당 실행의 ablation 표를 확인한다. "
    )
    calib = calibration_tbl.set_index("구분")
    l1a = levers_tbl[levers_tbl["레버"].str.startswith("L1a")].iloc[0]
    l1c = levers_tbl[levers_tbl["레버"].str.startswith("L1c")].iloc[0]
    final_fp = int(final["FP"])
    return pd.DataFrame([
        {
            "차별점": "① 반복 전력 프로파일 진단",
            "무엇을 했는가": "일 단위 전력 프로파일 해시로 반복 구조와 분할 간 프로파일 공유를 점검했다.",
            "측정된 근거": (
                f"고유 프로파일 {N_UNIQUE_PROFILES}/{N_TOTAL_DAYS}일, 중복그룹 {N_DUP_GROUPS}개, "
                f"반복일 {N_SURPLUS_DAYS}일({N_SURPLUS_DAYS / N_TOTAL_DAYS:.1%}). "
                f"D2 순위상관은 {float(rank_preservation.iloc[0]['값']):.3f}다."
            ),
            "기여": "시간 분할만으로 남을 수 있는 프로파일 공유를 확인하고, 복제 유지·제거 조건에서 결론의 민감도를 비교했다.",
            "대응 절": "1장, 2장 데이터 조건 비교",
        },
        {
            "차별점": "② 피크 정의와 경보 판정의 분리",
            "무엇을 했는가": (
                f"실제 피크 기준 θ={THETA:g} kW와 모델의 경보 판정 기준 τ를 구분했다. "
                "θ의 산출 구간과 τ의 선정·평가 구간은 분할표에 명시한다."
            ),
            "측정된 근거": "모델별 τ와 혼동행렬을 함께 제시해 피크 정의와 경보 기준의 차이를 확인할 수 있게 했다.",
            "기여": "높은 탐지율과 오경보 부담의 절충을 같은 실제 피크 정의에서 비교했다. 임계값 선정에 사용한 자료의 성능은 독립 평가로 보지 않는다.",
            "대응 절": "1.6, 2.5, 2.7",
        },
        {
            "차별점": "③ 달력규칙과 오경보·미탐지 비교",
            "무엇을 했는가": "평일 08~18시 경보 규칙을 피크 탐지 기준선에 포함했다.",
            "측정된 근거": (
                f"달력규칙 Recall {calendar_rule_metrics['Recall']:.3f}, "
                f"TP {calendar_rule_metrics['TP']} / FN {calendar_rule_metrics['FN']} / "
                f"FP {calendar_rule_metrics['FP']} / TN {calendar_rule_metrics['TN']}. "
                f"최종모델 Recall {float(final['Recall']):.3f}, "
                f"TP {int(final['TP'])} / FN {int(final['FN'])} / FP {final_fp}."
            ),
            "기여": (
                f"달력규칙 대비 FP 변화는 {final_fp - CALENDAR_RULE_FP:+d}건, "
                f"FN 변화는 {int(final['FN']) - calendar_rule_metrics['FN']:+d}건이다. "
                "Recall이 같다고 가정하지 않고 탐지 누락과 경보 업무량을 함께 비교한다."
            ),
            "대응 절": "2.1, 2.7, 4.1",
        },
        {
            "차별점": "④ 확률 보정과 예측구간 평가",
            "무엇을 했는가": "Isotonic 확률 보정과 분위회귀·Split Conformal 예측구간을 평가했다.",
            "측정된 근거": (
                f"Brier {calib.loc['보정 전', 'Brier']:.5f} → {calib.loc['보정 후', 'Brier']:.5f}, "
                f"ECE {calib.loc['보정 전', 'ECE']:.5f} → {calib.loc['보정 후', 'ECE']:.5f}. "
                "예측구간 표에 실제 피복률과 폭을 보고했다."
            ),
            "기여": (
                "점예측의 오차와 확률 신뢰도를 별도로 점검했다. 보정만으로 예측확률과 실제 빈도의 "
                "정확한 일치나 미래 구간의 피복률을 보장하지 않으며, 분포이동 시 재평가가 필요하다."
            ),
            "대응 절": "2장 보정·불확실성, 4장 운영 제안",
        },
        {
            "차별점": "⑤ 부하상태 모델과 운영 가설의 연결",
            "무엇을 했는가": "부하상태별 예측모델을 비교하고, 과거 실측 위에서 기동 분산·생산량 이동 시나리오를 계산했다.",
            "측정된 근거": (
                regime_text + f"생산량의 관측 회귀계수 β={BETA:.6f} kW/단위다. "
                f"L1a·L1c의 가정 내 최대수요 감소량은 각각 {l1a['Δ최대수요(kW)']:.1f}·"
                f"{l1c['Δ최대수요(kW)']:.1f} kW다."
            ),
            "기여": (
                "최대값을 남기는 사건에 따라 저감 상한이 달라지는 점을 확인했다. L1c를 현장시험 후보로 제시하며, "
                "생산량 회귀계수를 인과효과로 보거나 가정 기반 저감량을 실제 성과로 해석하지 않는다."
            ),
            "대응 절": "2.4, 2.8, 4장",
        },
    ])


creativity_tbl = build_creativity_table()
save_table(creativity_tbl, "ch5_creativity")
print("── 5장 차별점 5종 ──")
for _, r in creativity_tbl.iterrows():
    print(f"\n[{r['차별점']}] (보고서 {r['대응 절']})")
    print(f"  무엇: {r['무엇을 했는가']}")
    print(f"  근거: {r['측정된 근거']}")
    print(f"  기여: {r['기여']}")

# %% [markdown]
# ### 9.6 보고서 5장 본문 초안 생성
#
# - **목적**: 차별점 5종을 5장 본문 초안 markdown 으로 출력한다.
# - **보고서 대응절**: 5장 전체
# - **산출물**: `outputs/report_ch5_draft.md`

# %%
def write_ch5_draft() -> str:
    """5장 초안: 실행 수치와 검증 한계를 함께 제시한다."""
    final = peak_tbl[peak_tbl["모델"] == FINAL_MODEL_NAME].iloc[0]
    positives = int(final["TP"] + final["FN"])
    test_n = int(sum(final[column] for column in ("TP", "FN", "FP", "TN")))
    lag = ablation_tbl[ablation_tbl["제거 피처군"] == "과거 전력 지연변수"]
    lag_text = (
        f"과거 전력 지연변수 제거 시 OOF MAE 변화는 {float(lag['MAE 변화'].iloc[0]):+.3f} kW였다."
        if len(lag) else "해당 실행의 변수 제거 표를 확인한다."
    )
    md = "# 제 5 장. 창의성 및 차별성 〔10점〕\n\n제안 방법의 기여를 성능·오류분석·운영 가설과 연결한다.\n\n"
    for i, (_, row) in enumerate(creativity_tbl.iterrows(), start=1):
        md += f"""## 5.{i} {row['차별점'][2:]}

**방법.** {row['무엇을 했는가']}

**결과.** {row['측정된 근거']}

**해석.** {row['기여']} ({row['대응 절']})

"""
    md += f"""## 5.6 한계와 추가 검증

1. 기준선 대비 최종모델의 MAE 개선율은 {IMPROVE_MAE_PCT:+.2f}%다.
   일 단위 블록 bootstrap의 95% 신뢰구간은
   {'0을 포함하지 않았다' if MAE_IMPROVEMENT_SIGNIFICANT else '0을 포함하여 유의수준 5%에서 차이를 확인하지 못했다'}.
   비유의 원인을 데이터 정비 탓으로 단정하지 않는다. 효과 크기와 구간을 함께 확인한다.
2. 테스트는 {test_n}시간, 실제 피크 {positives}건이며 최종모델 Recall은 {float(final['Recall']):.3f}다.
   OOF는 {len(OOF)}시간, 실제 피크 {int(OOF['y_cls'].sum())}건이다.
   연속 시간의 의존성과 기간·계절 제한을 고려해야 하며 장기 운영 성능은 추가 검증이 필요하다.
3. {lag_text} 전체 평균과 휴무·토요일 등 조건별 변화를 함께 검토한다.
   선택 기준과 후속 진단을 구분하고, 결과를 본 뒤 만든 가설을 사전에 정한 기준으로 표현하지 않는다.
4. 피크 저감은 실측을 알고 계산한 사후 시나리오다. 기존 최대값 유지 가정에서 테스트만 0으로
   낮췄을 때의 12개월 기본요금 차이는 {SEPT_ONLY_SAVING:,.0f}원이다.
   실제 예측 경보를 사용한 일정조정·현장 효과는 검증하지 않았다.
5. 비용 비교는 관측기간의 기본요금 가정액과 추가 인건비만 사용하며, 12개월 환산액은 참고로 분리한다.
   전력량요금은 산정에서 제외한다. L1의 이동 부하·설비용량·납기·실행비용을 확인한 뒤 현장시험이 필요하다.
"""
    return md


_ch5 = write_ch5_draft()
(OUTPUT_DIR / "report_ch5_draft.md").write_text(_ch5, encoding="utf-8")
print(f"\n── 5장 본문 초안 생성 완료 ({len(_ch5):,}자) → outputs/report_ch5_draft.md ──")
