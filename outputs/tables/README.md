이 폴더는 `run_all.py`가 계산한 실험표와 그림 원자료(CSV)가 저장되는 곳이며, 보고서 수치의 근거가 된다.

# tables — 실험표와 그림 원자료

**결론**: 보고서의 표·수치를 확인하려면 아래 '보고서 표 ← CSV' 를 보고 해당 파일을 연다.
터미널에서는 `python show_results.py --table ch2_regression` 처럼 이름으로, 또는 `--table 2-2` 처럼 보고서 표 번호로 볼 수 있다.
기존 파일이 남아 있더라도 최신 코드의 FULL 실행이 완료됐는지 `outputs/steps/index.json`으로 확인한다.

- FULL 필수 CSV는 110개(실험표 71개 + 그림 원자료 `F01_src`~`F39_src` 39개)다. 실제 파일 수는 실행 옵션에 따라 달라지며, `--skip-bundle`이면 `ch6_model_bundles`가 생성되지 않는다.
- 형식: UTF-8(BOM 있음) — 엑셀에서 한글이 깨지지 않는다. 줄바꿈은 실행한 OS 를 따른다(리눅스 LF, Windows CRLF).

## 보고서 표 ← CSV

| 보고서 | 파일 | 만드는 단계 |
|---|---|---|
| 1.2절 변수 사전 | `ch1_variable_dict.csv` | 1.1 |
| 1.4절 품질 진단 | `ch1_quality.csv` | 1.2 |
| 표1-1 교차검증 fold 구성 | `ch3_folds.csv` | 3.2 |
| 표2-1 가이드북 결함 | `ch4_defects.csv` (함께: `ch4_rf_original`·`ch4_corrected`·`ch4_naive`) | 4.4 (4.1·4.3·4.5) |
| 2.3절 입력변수 구성 | `ch2_feature_groups.csv` | 2.3 |
| 2.5절 하이퍼파라미터 탐색 | `hpo_trials.csv` (FULL 100행, FAST 6행 — 회귀 탐색 5.1 + 분류 탐색 5.4) | 5.4 |
| 표2-2 전력 예측 성능 | `ch2_regression.csv` | 6.1 |
| RF 대비 MAE 감소량의 95% 기본·90% 사후 보조 CI | `ch2_significance.csv` | 6.1 |
| 시간순 교차적합 확률 보정 평가 | `ch5_calibration.csv` | 5.7 |
| 확률 보정 fold별 학습·평가 경계 | `ch5_calibration_folds.csv` | 5.7 |
| 확률 보정 시간순 검증 예측 | `ch5_calibration_forward_predictions.csv` | 5.7 |
| 표2-3 fold × 모델 MAE | `ch5_fold_mae_matrix.csv` | 5.5 |
| 표2-4 피크 탐지 성능(테스트) | `ch2_peak_detection.csv` | 6.2 |
| 표2-5 피크 탐지 성능(OOF) | `ch2_peak_detection_oof.csv` | 6.2 |
| 피크 임계값 시간순 분리 보조평가 | `ch2_peak_threshold_temporal.csv` | 6.2 |
| 2.8절 Ablation | `ch2_ablation.csv` | 6.3 |
| 표2-6 지연변수 제거 효과 분해 | `ch2_lag_decomposition.csv` | 6.6 |
| 표2-7 최종모델 스코어카드 | `ch2_scorecard.csv` | 6.4 |
| 생산·가동계획 정보 제거 비교 | `ch2_plan_information_ablation.csv` | 6.4 |
| 표3-1 주요 영향변수 | `ch3_importance.csv` | 7.1 |
| 3.2절 상호작용 | `ch3_interaction.csv` | 7.2 |
| 3.3절 조건별 오차 | `ch3_condition_mae.csv` | 7.3 |
| 3.4절 미탐지·오경보 | `ch3_confusion.csv` | 7.4 |
| 3.5절 대표 실패사례 | `ch3_failures.csv` | 7.5 |
| 3.6절 피크 발생 규칙 | `ch3_rules.csv` | 7.6 |
| 표4-1 저감 레버 | `ch4_levers.csv` | 8.2 |
| 표4-2 시나리오 비교 | `ch4_scenarios.csv` | 8.3 |
| 4장 운영 프로토콜 | `ch4_protocol.csv` | 8.4 |
| 6장 실행 환경·모델 번들 | `env_versions.csv` · `ch6_model_bundles.csv` | 0.6 · 10.5 |

## 파일 이름의 접두 — 주의

접두는 **보고서 장**을 따른 것이라 코드 파일과 1:1 이 아니다.

| 접두 | 만드는 코드 |
|---|---|
| `ch1_` | `s01` (1장 데이터 진단) |
| `ch2_` | `s02`(피처 구성·누수 검증) 와 `s06`(성능평가·Ablation·스코어카드) |
| `ch3_` | `s03`(분할·fold·달력규칙: `ch3_data_conditions`·`ch3_contamination`·`ch3_folds`·`ch3_calendar_rule`) 와 `s07`(영향요인·오류분석) |
| **`ch4_`** | **두 곳**: `s04` 가이드북 베이스라인(`ch4_rf_original`·`ch4_defects`·`ch4_silent_noop`·`ch4_corrected`·`ch4_naive` → 보고서 2.2절) 와 `s08` 피크 저감 시뮬레이션(`ch4_tariff`·`ch4_levers`·`ch4_scenarios`·`ch4_sensitivity`·`ch4_protocol`·`ch4_priority` → 보고서 4장) |
| `ch5_` | `s05` (앙상블·확률 보정·불확실성·fold 행렬) |
| `ch6_` | `s06`(`ch6_rank_preservation`) 와 `s10`(`ch6_model_bundles`) |
| 접두 없음 | `env_versions`(s00) · `fold_matrix`(s03) · `hpo_trials`·`timing`(s05) · 게이트 표 |
| `F01_src` ~ `F39_src` | 같은 번호 그림의 원자료(`outputs/figures/`). 그림을 다시 그리거나 값을 확인할 때 쓴다 |

어느 단계가 어느 표를 썼는지는 `outputs/steps/index.json` 의 단계별 `files.tables` 목록에 정확히 남는다.

## 게이트 표 — 장마다 스스로 점검한 결과

| 파일 | 장 | 실패하면 |
|---|---|---|
| `gate1_chapter1.csv` | 1장 (중복·시간값·결측·평균 무결성·행 수·θ=187) | 그 자리에서 멈춘다 |
| `gate2_leakage.csv` | 2장 (시간누수 — 대상일 전력을 지워도 피처가 같은가) | 멈춘다 |
| `gate3_split.csv` | 3장 (fold·테스트의 복제 오염) | 멈춘다 |
| `gate4_baseline.csv` | 4장 (가이드북 결함 재현) | 멈춘다 |
| `gate5_eval.csv` | 6장 게이트(6.G) — 성능 하한 | **표로만 남긴다** |

1~4의 필수 점검은 모두 통과해야 한다. `gate5_eval.csv`는 해당 실행의 성능 기준 충족 여부를 그대로 기록하므로 `False` 개수를 미리 정하지 않는다. RF 대비 개선의 기본 판정은 기존 95% 신뢰구간이며, 90% 구간은 사후 보조분석으로만 해석한다. 유의성 기준 미충족 자체와 코드 실행 오류를 구분한다.

## 수정본 표를 읽는 기준

- `ch2_regression`: `Peak-MAE`는 실제 `y_peak ≥ θ` 시간의 평균전력 예측오차, `Peak15-MAE`는 같은 시간의 최대전력 예측오차다. 테스트 피크 표본은 28시간이다. 평균전력 `y_avg ≥ θ`인 2시간은 별도 고부하 열로 기록한다.
- 현재 3분류 모델의 위 실제 피크 오차는 8.144·10.245kW이고 `High-load-MAE(legacy)`는 15.878kW(2시간)다. 전체 테스트 MAE는 5.310kW, OOF MAE는 11.044kW다.
- `ch2_significance`: 감소량은 `RF MAE − 최종모델 MAE`이며 양수가 개선이다. 일 단위 블록 2,000회, 시드 42의 95% 기본 구간과 90% 사후 보조 구간 및 근사 p값을 함께 본다. 수정 전 p=0.067을 새 실행의 p값으로 옮겨 쓰지 않는다.
- 현재 RF 대비 3분류 모델의 감소량은 1.019kW(16.11%), 95% CI [0.008, 2.527]kW, 90% 사후 보조 CI [0.099, 2.257]kW, 근사 p=0.046이다. 기존 5% 기준을 충족하지만 새 외부 독립 데이터로 확인한 결과는 아니다.
- `ch5_calibration`: 과거 OOF로 적합한 보정기를 뒤 구간에 적용한 시간순 교차적합 성능이다. 전체 OOF로 적합한 최종 서빙용 보정기의 학습내 성능과 구분한다.
- `ch5_calibration_folds`·`ch5_calibration_forward_predictions`: 위 확률 보정 평가의 시간 경계와 예측을 확인하는 근거다. 첫 OOF 구간은 이전 보정 자료가 없으므로 평가 적용 범위를 확인한다.
- 현재 보정 평가 991시간에서 Brier는 0.08475→0.07745, ECE는 0.07290→0.05747이다. HPO·모델 선택이 같은 데이터에 의존하는 한계는 남아 있다.
- `ch2_peak_threshold_temporal`: 임계값을 이전 OOF에서만 선택하고 뒤 구간에 적용한 보조평가다. HPO·모델 선택은 기존 OOF를 재사용하므로 완전한 중첩 검증 결과가 아니다.
- `ch2_plan_information_ablation`: 생산 관련 원변수뿐 아니라 가동·휴무 등 계획 파생정보의 의존성도 구분해 비교한다. 단순히 원변수 4개를 제거한 결과를 계획정보 없이 운영 가능한 증거로 해석하지 않는다.
- `ch4_levers`·`ch4_scenarios`: 전력량요금 절감은 산정하지 않아 0원으로 둔다. 기본요금 절감·인건비 증가·순절감은 2021-01-01~09-14의 관측기간(월수 환산 `8 + 14/30`)으로 맞춘다. 12개월 기본요금 환산액은 별도 참고값이며 실제 청구 절감이 아니다.
- 부하 이동 후 새 피크가 관측 최대 222kW를 넘을 수 있다. 이를 상한으로 잘라내지 않고 계산하며, 시나리오 표의 관측 최대 초과 여부·시간 수를 함께 제시한다.
- 순절감 최선 시나리오와 최대수요 감소량 최선 시나리오는 다를 수 있다. FULL 생성표와 독립 계산이 일치했으며, S2는 순절감 924,176원으로 가장 크고 최대수요 감소는 13.1kW다. S4의 감소는 13.2kW지만 관측기간 인건비 94,720원을 차감한 순절감은 834,509원이다. 두 시나리오의 값을 혼합하지 않는다. 실행비용은 측정하지 않아 실제 한전 청구 절감액으로 제시할 수 없다.
- `ch4_tariff`: 과거 222kW 청구 기준이 유지된다는 시나리오 가정에서는 9월만 저감해도 기본요금 절감이 0원이다. 이 결과를 실제 요금제 전반의 청구 규칙으로 일반화하지 않는다.
- `ch3_failures`: 과대예측·과소예측은 실측과 예측의 차이 부호로 설명한다. 휴무라는 조건만으로 오차 방향이나 인과관계를 단정하지 않는다.
- `ch2_d1_vs_d2`·`ch6_rank_preservation`: 현재 D2 OOF에서 레짐 2분류 12.693kW(1위), 3분류 12.828kW(2위)이며 전체 순위상관은 0.9429다. 최종모델의 1위가 완전히 보존된 것도, 일반화가 입증된 것도 아니다.

## 실행마다 달라지는 표 (정상)

| 표 | 무엇이 달라지나 |
|---|---|
| `env_versions.csv` | 실행 환경의 파이썬·라이브러리 버전 |
| `timing.csv` | 단계별 벽시계 시간 |
| `ch2_regression.csv`·`ch2_scorecard.csv` 의 '학습시간(초)' 열 | 벽시계 시간. 성능값은 코드·데이터·학습 설정이 같을 때 비교 |
| DNN (MLP)·Simple RNN 이 들어간 행 (`ch2_regression`·`ch2_peak_detection_oof`·`ch2_scorecard`·`ch4_corrected`·`ch5_fold_mae_matrix` 등) | TensorFlow 버전·GPU 에 따라 수치가 조금 다르다. 최종모델 선정·제출 예측과는 무관 |
| `ch6_model_bundles.csv` | 번들 이름·sha256 — 번들에 OS·라이브러리 버전이 기록된다 |
| `ch1_profile_dup.csv` | 그룹 **번호**와 행 순서만(동점 정렬 순서가 pandas 버전마다 다르다). 그룹 내용은 같다 |
