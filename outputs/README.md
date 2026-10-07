이 폴더는 `run_all.py` 가 계산한 결과(제출 예측·그림·표·모델·단계 로그)가 저장되는 곳이다.

# outputs — 실행 결과

**결론**: 이 폴더의 파일은 모두 `python run_all.py` 가 만든다. 사람이 고치지 않는다.
보는 방법은 `python show_results.py` (보고서 순서로 넘겨 보기) 또는 각 하위 폴더의 README.

- 기존 산출물이 남아 있어도 코드 수정 후에는 FULL 실행으로 갱신한다. 수정 전 보고서 수치를 현재 결과로 간주하지 않는다.
- **실행 후**: 아래 파일이 채워지고, 예측 파일도 이번 실행 결과로 다시 쓴다. `steps/index.json`의 상태·모드·코드 해시로 결과의 출처를 확인한다.
  결과 묶음 zip(`python run_all.py --pack-results`)에는 이 폴더의 예측·그림·표·단계 로그·모델 번들(full/eval·deploy)이 들어간다.

## 구성

| 경로 | 내용 | 만드는 단계 |
|---|---|---|
| `predictions_test_336h.csv` | **제출 예측 파일** (아래 규격) | 10.1 |
| `figures/` | 그림 **39장**(F01~F39) + 그림 색인 `figure_index.csv` | 각 절 |
| `tables/` | FULL 필수 CSV 110개(실험표 71개 + 그림 원자료 39개). 실제 목록은 단계 기록으로 확인 | 각 절 |
| `models/` | 모델 번들 `full/{eval,deploy}` (FAST 면 `fast/`) · 비교 모델 `baselines/`·`comparison/`(pickle 기반이라 제출 ZIP에서는 제외, 실행하면 다시 생성) | 4장·5장·10.5 |
| `steps/` | 단계별 로그 · 진행/결과 기록 `index.json` · 서빙 검증 증거 `serving_*` | 모든 단계 |
| `baseline_repro/` | 실행하면 만들어지지만 비어 있다(원본 노트북 호환용 폴더) | 0.4 |

## 제출 예측 파일 규격 — `predictions_test_336h.csv`

- **336행 × 9열**, 2021-09-01 00:00 ~ 2021-09-14 23:00 매시 한 행, 결측 없음, UTF-8(BOM 있음).
- 열:

| 열 | 뜻 |
|---|---|
| `datetime` | 시각 (`YYYY-MM-DD HH:MM:SS`) |
| `y_avg_true` · `y_avg_pred` | 시간 평균전력 실측 · 예측 (kW) |
| `y_peak_true` · `y_peak_pred` | peak15(15분 최대수요의 시간 내 최대) 실측 · 예측 (kW) |
| `peak_prob` | 피크 확률 (피크 직접분류 모델) |
| `peak_pred_label` | 피크 경보 (1 = `y_peak_pred ≥ τ`, τ는 해당 실행의 평가표·번들에 저장) |
| `peak_true_label` | 실제 피크 여부 (1 = `y_peak_true ≥ θ = 187 kW`) |
| `model_name` | 최종모델 이름 (`2단계 레짐(3분류)`) |

완료 점검(C)이 이 규격을 자동으로 확인한다. 리눅스와 Windows 는 줄바꿈이 달라 파일 sha256 이 다르다 — 비교는 LF 로 맞춘 뒤 한다(루트 README 6절).

## 헤드라인 수치가 있는 표

| 확인할 값 | 표 (`tables/`) |
|---|---|
| 최종모델 선정 결과와 OOF 근거 | `ch2_scorecard.csv` |
| 전체 MAE, 실제 피크 시간의 Peak-MAE·Peak15-MAE·표본 수, 별도 평균전력 고부하 지표 | `ch2_regression.csv` |
| RF 대비 MAE 감소량·95% 기본 CI·90% 사후 보조 CI·근사 p값 | `ch2_significance.csv` |
| Recall·F1·τ·정탐·미탐·오경보 | `ch2_peak_detection.csv`·`ch3_confusion.csv` |
| 임계값만 시간순 분리한 보조평가 | `ch2_peak_threshold_temporal.csv` |
| 생산·가동계획 정보 제거 비교 | `ch2_plan_information_ablation.csv` |
| 시간순 교차적합으로 평가한 확률 보정 성능·fold 경계·예측 | `ch5_calibration.csv`·`ch5_calibration_folds.csv`·`ch5_calibration_forward_predictions.csv` |
| 관측기간 순절감과 별도 연간 기본요금 환산액 | `ch4_levers.csv`·`ch4_scenarios.csv` |

기존 보고서의 평균전력 고부하 2시간에 대한 오차와 새 실제 피크 28시간의 오차는 표본이 다르다. 또한 시나리오의 전력량요금 절감은 산정 제외(0원)이며, 관측기간 순절감과 12개월 기본요금 환산액을 합산하지 않는다. 운영 활용은 L1c 기동 분산의 현장시험을 우선 검토하는 제안이다.

현재 재계산된 3분류 모델은 MAE 5.310kW, RF 대비 감소량 1.019kW(16.11%), 95% CI [0.008, 2.527]kW, 90% 사후 보조 CI [0.099, 2.257]kW, 근사 p=0.046이다. 피크 탐지는 정탐 23·오경보 43·미탐 5다. 같은 2021년 9월 데이터를 다시 평가한 것이며 새로운 외부 독립 검증은 아니다. FULL 62단계·번들 검증·336행 메모리 리플레이·123개 테스트가 완료됐고 기록은 `steps/index.json`과 `serving_*` 파일에 있다.

비용표는 독립 계산과 대조했다. S1c·S2·S4 관측기간 순절감은 각각 915,755원·924,176원·834,509원이며, 별도 연간 기본요금 환산액은 1,297,920원·1,309,856원·1,317,017원이다. 이는 미측정 실행비용을 제외한 사후 시나리오 추정액이다.

시나리오에서 이동 후 부하가 관측 최대 222kW를 넘더라도 상한으로 잘라내지 않는다. `ch4_levers`·`ch4_scenarios`의 관측 최대 초과 여부와 시간 수도 확인한다.

9월만의 피크 저감에 기본요금 절감 0원이 나온다면 이는 과거 222kW 청구 기준이 유지된다는 계산 가정의 결과다. 실제 요금제·계약·청구서를 검증한 일반적인 결론이 아니다.

## 실행마다 달라지는 파일 (정상)

- `tables/env_versions.csv` — 실행 환경의 라이브러리 버전.
- `tables/timing.csv`, 표의 '학습시간(초)' 열(`ch2_regression`·`ch2_scorecard` 등) — 벽시계 시간.
- TensorFlow 비교 모델(DNN (MLP)·Simple RNN)의 행 — TF 버전·GPU 에 따라 수치가 조금 다르다. 최종모델·제출 예측과는 무관하다.
- `tables/ch6_model_bundles.csv` 의 해시·번들 이름 — 번들에 OS·라이브러리 버전이 기록되기 때문이다.
- `steps/` 전체 — 실행 시각·소요시간이 들어 있다.
