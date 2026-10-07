# KAMP 자원최적화 — TensorFlow 팟 실행 README

## 추가 검증 제출본 실행 안내

기존 62단계 분석에 E1~E6 추가 검증과 그림 재생성·엄격 검증을 연결하였다.
각 실행의 실제 완료 여부는 `outputs/additional/validated_run/verification_summary.json`에서 확인한다.
원시 예측과 그림 수정본은 기존 헤드라인 실험과 추가 분석의 구분을 유지한다.

완성 ZIP을 KAMP-NOTE에 옮긴 뒤에는 Windows 기준 snapshot을 남기고 **새 경로**로 실행한다.
환경 준비 및 공유 팟의 패키지 보존 절차는 [PORTABILITY.md](PORTABILITY.md)를 먼저 따른다.
새로 압축을 푼 작업 폴더에서 실행한다. 코어의 `outputs/steps`, 표·그림·모델은 이번 실행으로
갱신되므로, Windows 원본 제출 ZIP은 보관한다. 별도 snapshot 경로는 비교용 원시 예측을 보존한다.

```bash
python run_submission.py --profile full --cpu --snapshot outputs/kamp_analysis_snapshot --run-dir outputs/additional/kamp_run --figures outputs/kamp_figures --reference outputs/analysis_snapshot
python show_results.py --additional --run-dir outputs/additional/kamp_run
```

계산과 파일 검증이 끝나도 새 그림을 직접 검토하지 않았다면 종료코드 3과
`COMPUTE_OK_VISUAL_PENDING`이 나온다. 검토한 이미지 해시에 연결된 시각 QA 기록을 전달해야
로컬 범위의 `FULL_OK`가 가능하다. Windows의 완료 기록은 KAMP 실행 완료를 뜻하지 않는다.
`--reference`는 기준 OOF·테스트 예측을 비교하며, 추가 E1~E6의 플랫폼 간 대조는 별도로 수행한다.
KAMP 실행이 끝나면 다음 명령으로 6개 추가 실험의 설정·원행·선정 결과·지표를 대조한다.

```bash
python tools/compare_supplemental.py --reference outputs/additional/validated_run --current outputs/additional/kamp_run --output outputs/verification/kamp_supplemental_comparison.json
```

시간·캐시·운영체제 차이는 허용하되 정답·라벨·선정 모델은 같아야 하고, 실수 결과에는
`atol=1e-6`, `rtol=1e-6`을 적용한다. 부분 실행이나 불일치는 실패로 기록한다.

| 추가 항목 | 계산 내용 | 결과 폴더 |
|---|---|---|
| E1 | 조건별 분모·미탐률·오경보율과 날짜 단위 구간 | `diagnostics/` |
| E2 | 같은 14일 테스트의 1·2·3일 블록 민감도 | `bootstrap/` |
| E3 | 동결 평가모델에 대한 생산계획·인원·기상 입력 오차 | `sensitivity/` |
| E4 | 같은 데이터·회귀설정의 단일 모델과 레짐 비교 | `controlled/` |
| E5 | 각 평가기간 이전 정보로 튜닝·선정·보정한 시간순 평가 | `temporal/` |
| E6 | 시간순 예측에 따른 경보·생산조정의 가정 모의 | `policy/` |

공통 경로는 `outputs/additional/<실행명>/`이다. 실행 설정은 `experiments/config.json`에 고정하였다.
E5는 기존 자료의 후향적 평가이며 새 독립 테스트가 아니다. E6는 실제 절감이나 폐루프 운영의 증거가 아니다.
1일 블록의 결과(2,000회, seed 42)는 seed에 민감하고(seed 1~20 중 17개에서 95% 구간이 0을 포함) 2·3일 블록에서도 0을 포함하므로
5% 유의성을 주장하지 않는다. 모든 사전 지정 결과를 함께 보고한다. seed·반복수 민감도는 다음 명령으로 재현한다(학습 없음, 약 1분).

```bash
python tools/bootstrap_seed_sensitivity.py   # outputs/verification/bootstrap_seed_sensitivity.json
```

그림만 다시 만들 때는 다음 명령을 사용한다. 모델을 다시 학습하지 않는다.

```bash
python tools/render_figures.py --all --snapshot outputs/analysis_snapshot --experiments outputs/additional/validated_run --output outputs/figures_rerendered
```

고정 Windows 환경 기록은 `verification/environment_windows.json`에 있으며,
기존 의존성의 재현 제약은 `verification/reproduction_constraints.txt`에 제공한다.
전체 Windows freeze에는 Windows 전용 패키지가 있으므로 Linux에 그대로 설치하지 않는다.

## 1. 목적과 적용 조건

이 문서는 전체 실험을 실행하고 핵심 코드의 역할을 확인하는 데 사용한다.

| 항목 | 적용 조건 |
|---|---|
| 실행 환경 | KAMP-NOTE의 TensorFlow 팟 |
| 실행 위치 | 팟의 터미널 |
| 작업 폴더 | `~/KAMP_정종묵` |
| 준비 상태 | 전체 프로젝트를 작업 폴더에 배치한 상태 |
| 실행 모드 | FULL, CPU 사용 |
| 입력 파일 | `data/okm_augumented_2021.csv` |

프로그램은 다음 날 24시간의 평균전력과 15분 최대수요를 예측한다. 예측 결과로 피크 경보를 생성한다.

최종모델은 **LightGBM 기반 2단계 레짐(3분류) 모델**이다.
TensorFlow는 비교 모델인 **DNN과 SimpleRNN** 학습에 사용한다.

## 2. 필수 코드와 역할

아래 표는 설명 대상을 요약한다. 실행에는 전체 프로젝트 구성이 필요하다.

| 파일 또는 폴더 | 역할 |
|---|---|
| `check_env.sh` | Python과 패키지 상태를 확인한다. 환경을 변경하지 않는다. |
| `setup_pod.sh` | 없는 패키지를 설치한다. `--with-tests`를 사용하면 테스트 실행에 필요한 패키지도 준비한다. |
| `run_all.py` | 전체 계산을 제어한다. 모델 저장, 서빙 검증, 완료 점검을 실행한다. |
| `src/` | 데이터 처리, 모델 학습, 성능 평가, 피크 저감 시뮬레이션을 수행한다. |
| `tools/` | 모델 저장, 서빙 코어 생성, 레짐 교차검증을 보조한다. |
| `serving/` | 저장된 모델로 예측한다. 모델을 다시 학습하지 않는다. |
| `tests/`, `pytest.ini` | 단위·회귀테스트와 테스트 실행 설정을 제공한다. |
| `show_results.py` | 저장된 결과를 표시한다. 결과를 다시 계산하지 않는다. |
| `requirements.txt` | 프로젝트의 패키지 버전을 기록한다. 팟 설치에는 `setup_pod.sh`를 사용한다. |

## 3. 핵심 처리 규칙

### 3.1 입력과 전처리

`src/`는 입력 CSV를 읽고 모델 입력을 만든다.

결측 전력, 풍속, 공장인원은 이전 관측값으로 채운다.
이 처리를 `ffill`이라고 한다.
예측 시점 이후의 관측값으로 결측값을 채우지 않는다.

학습과 서빙에는 같은 전처리 규칙을 적용한다.
미래 생산·휴무계획을 사용하는 입력은 예측 전에 확보해야 한다.

### 3.2 학습과 테스트 분리

`src/s03_split.py`는 시간 기준으로 데이터를 분리한다.

| 구간 | 용도 |
|---|---|
| 2021-09-01 이전 | 학습, 교차검증, 튜닝, 모델 선정 |
| 2021-09-01 00시 ~ 09-14 23시 | 최종 테스트 336시간 |

튜닝과 모델 선정에는 교차검증 예측인 **OOF 예측**을 사용한다.
테스트 구간의 실측값을 학습이나 튜닝에 사용하지 않는다.

### 3.3 모델 평가와 경보

`src/`는 비교 모델과 레짐 모델을 학습한다.
`src/s06_eval.py`는 예측 성능과 피크 탐지 성능을 평가한다.

모델 선정에는 OOF 오차, 피크 오차, 탐지 성능, 검증 구간별 편차를 함께 사용한다.

실제 피크는 `y_peak ≥ 187 kW`인 시간이다.
이 값은 실제 피크의 정의다.
예측 경보 임계값과 구분해서 사용한다.

### 3.4 모델 저장과 검증

`run_all.py`는 모델과 필요한 정보를 모델 번들로 저장한다.

| 번들 경로 | 용도 |
|---|---|
| `outputs/models/full/eval/` | 테스트 예측 재현 |
| `outputs/models/full/deploy/` | 저장 모델을 이용한 추론 |

FULL 실행은 다음 항목을 확인한다.

1. 모델 번들의 무결성과 자체 검증.
2. 테스트 예측의 재현 결과.
3. 이력·계획 CSV를 사용한 익일 예측 데모.
4. 단위·회귀테스트.

## 4. 실행 절차

### 4.1 환경 준비

1. 작업 폴더로 이동하십시오.

   ```bash
   cd ~/KAMP_정종묵
   ```

2. 환경 점검을 실행하십시오.

   ```bash
   bash check_env.sh
   ```

   **확인:** TensorFlow 버전이 표시되어야 한다.
   TensorFlow가 없으면 KAMP-NOTE에서 TensorFlow 팟을 선택하십시오.

> **주의:** 공유 팟에서 `pip install -r requirements.txt`를 실행하지 마십시오. 기존 패키지를 변경하면 설치 오류가 발생할 수 있다.
>
> 팟에 설치된 TensorFlow를 유지하십시오. `setup_pod.sh`는 TensorFlow를 설치하지 않는다.

3. 필요한 패키지를 준비하십시오.

   ```bash
   bash setup_pod.sh --with-tests
   ```

### 4.2 FULL 실행

> **주의:** FULL 실행은 `outputs/`의 결과를 갱신한다. 기존 결과가 필요하면 실행 전에 별도로 보관하십시오.

1. 전체 실행을 시작하십시오.

   ```bash
   nohup python -u -X utf8 run_all.py --cpu > ~/full_run.out 2>&1 < /dev/null &
   ```

   | 명령 요소 | 기능 |
   |---|---|
   | `--cpu` | GPU 사용을 끈다. |
   | `-u` | Python 출력의 버퍼링을 끈다. |
   | `-X utf8` | Python의 UTF-8 모드를 켠다. |
   | `nohup`, `&`, `< /dev/null` | 터미널 입력 없이 백그라운드에서 실행한다. |
   | `> ~/full_run.out 2>&1` | 일반 출력과 오류 출력을 같은 로그에 저장한다. |

   **참고:** 팟이 정지되면 실행도 정지된다.

2. 실행 로그를 표시하십시오.

   ```bash
   tail -f ~/full_run.out
   ```

3. 이번 실행의 마지막 로그를 확인하십시오.

   **정상 판정:**

   ```text
   RUN_ALL_OK mode=FULL
   ```

   이 문구가 없으면 완료로 판정하지 마십시오.

4. 정상 판정 후 `Ctrl+C`를 누르십시오.

   `tail`의 로그 표시가 종료된다. 백그라운드 학습 프로세스에는 중단 명령을 보내지 않는다.

5. 결과를 표시하십시오.

   ```bash
   python show_results.py
   ```

   다음 쪽은 Enter로 연다. 종료는 `q`를 사용한다.

## 5. 출력 확인과 저장 모델 사용

### 5.1 확인할 출력

| 경로 | 확인 내용 |
|---|---|
| `outputs/predictions_test_336h.csv` | 테스트 336시간의 예측과 실측값 |
| `outputs/tables/ch2_regression.csv` | 전력 예측 오차 |
| `outputs/tables/ch2_peak_detection.csv` | 피크 탐지 성능 |
| `outputs/models/full/` | 평가용·추론용 모델 번들 |
| `outputs/steps/index.json` | 실행 상태와 코드 해시 |
| `outputs/steps/` | 단계 로그와 서빙 검증 기록 |

예측 CSV의 주요 열은 다음과 같다.

| 열 | 의미 |
|---|---|
| `datetime` | 예측 대상 시각 |
| `y_avg_pred` | 평균전력 예측, kW |
| `y_peak_pred` | 15분 최대수요 예측, kW |
| `peak_prob` | 피크 확률 |
| `peak_pred_label` | 피크 경보. 1은 경보, 0은 경보 없음 |
| `y_avg_true`, `y_peak_true`, `peak_true_label` | 평가에 사용하는 실측값과 실제 피크 라벨 |
| `model_name` | 예측에 사용한 모델 이름 |

기존 출력 파일의 존재만으로 이번 실행의 성공을 판정하지 않는다.
이번 실행 로그와 완료 점검 결과를 함께 확인한다.

### 5.2 저장 모델 예측 데모

**시작 조건:** FULL 실행이 정상 종료되었다.
아래 명령은 FULL 실행에서 생성한 데모 입력을 사용한다.

1. 저장 모델로 예측하십시오.

   ```bash
   python -X utf8 -m serving predict \
     --bundle outputs/models/full/deploy \
     --history outputs/steps/serving_demo_history.csv \
     --plan outputs/steps/serving_demo_plan.csv \
     --out pred.csv
   ```

2. `pred.csv`를 확인하십시오.

`--history`는 과거 이력 CSV를 지정한다.
`--plan`은 예측 대상일의 계획 CSV를 지정한다.
실제 운영 입력의 필수 열과 형식은 프로젝트의 `serving/README.md`를 확인하십시오.

## 6. 이상 시 조치

| 증상 | 조치 |
|---|---|
| TensorFlow를 찾을 수 없음 | KAMP-NOTE에서 TensorFlow 팟을 선택하십시오. |
| `optuna` 또는 `shap`을 찾을 수 없음 | `bash setup_pod.sh --with-tests`를 실행하십시오. |
| 종료코드 `1` | 실패 단계의 로그를 확인하십시오. |
| 종료코드 `2` | 사전점검 메시지에서 실패 원인을 확인하십시오. |
| 종료코드 `3` | `outputs/steps/`에서 완료 점검 C의 실패 항목을 확인하십시오. |
| `VERSION_MISMATCH` | 번들을 만든 환경과 같은 numpy·pandas·lightgbm 버전을 사용하십시오. |
| 6.6 단계에서 출력이 적음 | 2분 간격의 경과 표시를 확인하십시오. 경과 표시가 나오면 계산이 진행 중이다. |
| S.4 테스트가 생략됨 | 6.1절의 테스트 복구 절차를 수행하십시오. |

### 6.1 테스트 복구

1. `bash setup_pod.sh --with-tests`로 테스트 패키지를 준비하십시오.
2. `python run_all.py --check-only`로 기존 결과를 검증하십시오.

### 6.2 팟 중단 후 복구

1. `bash check_env.sh`로 환경을 다시 점검하십시오.
2. 누락된 패키지가 있으면 `bash setup_pod.sh --with-tests`를 실행하십시오.
3. 4.2절을 다시 수행하십시오.

재실행은 전체 단계의 이어하기가 아니다.
앞선 단계는 다시 계산하며, 검증된 6.6 체크포인트만 재사용할 수 있다.

## 7. 코드 수정 시 확인

전처리, 입력변수, 번들 예측 코드를 수정한 경우 다음 절차를 수행하십시오.

1. 서빙 코어를 다시 생성하십시오.

   ```bash
   python tools/build_serving_core.py
   ```

2. 학습 코드와 서빙 코어의 동기화를 확인하십시오.

   ```bash
   python tools/build_serving_core.py --check
   ```

3. 4.2절의 FULL 실행을 다시 수행하십시오.

수정 후 생성한 예측, 평가표, 모델 번들, 검증 로그를 함께 사용한다.

---

### 문서 근거

- 코드 역할과 실행 명령: 제공된 `README.md`의 빠른 시작 및 1~9절.
- 작성 원칙: [ASD-STE100 공식 안내](https://asd-ste100.org/)와 [공식 FAQ](https://asd-ste100.org/STE_faq.html).
- ASD-STE100은 영문 기술문서 작성 표준이다. 이 문서는 짧은 문장, 일관된 용어, 직접적인 지시 원칙을 한국어에 적용했다.
- 이번 문서 작성에서는 실제 Python 소스 검토와 TensorFlow 팟 실행 검증을 수행하지 않았다.
