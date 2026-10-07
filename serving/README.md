이 폴더는 `run_all.py` 가 저장한 모델 번들을 검증하며 읽어 익일 24시간을 예측하는 추론 전용 코드(`python -m serving`)가 있는 곳이다.

# serving — 모델 번들 서빙 (운영 런북)

`python run_all.py`의 **10.5 단계**가 저장한 모델 번들을 검증하며 읽어, 전일 24:00 기준 **익일 24시간**의 전력(평균·peak15)과 피크 위험을 예측한다. 학습은 `run_all.py`와 `src/`에서 수행하며 이 폴더는 추론을 담당한다.

`python run_all.py --pack-results`의 결과 묶음에는 `full/eval`·`full/deploy` 번들이 포함된다. 새 환경에서 번들이 없거나 코드·전처리가 바뀌었으면 FULL 실행으로 다시 생성한다. 예측·경보를 출력하는 기능이며 생산일정 자동 최적화나 실제 절감 효과를 검증한 서비스는 아니다.

이번 FULL에서 두 번들의 무결성 검증, 336행 리플레이의 메모리 예측 배열 일치, eval·deploy 각 24행 CLI 예측과 누수 입력 거부를 확인했다. S.4는 123개 테스트를 통과했다. 배포 번들의 9월 14일 예측은 학습기간 안의 데모이며 독립 성능평가가 아니다.

---

## 1. 번들

`python run_all.py`를 실행하면 `outputs/models/{full|fast}/{eval|deploy}/`에 번들 두 개가 생긴다.

| 번들 | 학습 구간 | 용도 |
|---|---|---|
| `full/eval` | 2021-01-08 ~ 08-31 | 해당 실행의 제출 예측을 만든 모델. 재로드·리플레이 일치는 10.5·S.2 기록으로 확인 |
| `full/deploy` | 2021-01-08 ~ 09-14 | **배포용**. 같은 하이퍼파라미터로 전 구간 재학습. τ·보정기·구간 반폭은 eval 에서 상속 |
| `fast/*` | (테스트 실행 산출물) | 축소 학습 — 서빙 로더가 **거부**한다 |

| 파일 | 내용 |
|---|---|
| `manifest.json` | 피처 계약(44컬럼 순서·θ·휴일)·임계값(τ, τ_cls)·구간 반폭·런타임 버전·파일별 sha256·**매니페스트 자신의 sha256**(`bundle_id` 끝 12자리) |
| `gate.lgb` · `reg_y_*_r{0,1,2}.lgb` · `fallback_y_*.lgb` | 최종모델(2단계 레짐 3분류) — LightGBM 네이티브 텍스트 |
| `peak_clf.lgb` | 피크 직접분류기 |
| `interval_q{10,50,90}.lgb` | 분위회귀(미보정 — 참고용) |
| `calibrator_isotonic.json` | 피크 확률 보정 임계점 (`np.interp` 로 적용) |
| `golden.json` | 기동 자가검증 사례(원시 이력 21일 + 계획 + 기대 피처 + 기대 출력) |

pickle 은 쓰지 않는다. 로더는 파일을 **바이트로 읽어 sha256 을 대조한 뒤** `Booster(model_str=...)`
로 연다(파일 경로로 여는 LightGBM API 는 한글이 든 절대경로를 열지 못한다).

## 2. 설치

KAMP의 공유 conda 환경에서는 패키지 루트에서 다음 명령을 사용한다. 기존 numpy·pandas·TensorFlow를 임의로 교체하지 않는다.

```bash
bash check_env.sh
bash setup_pod.sh --with-tests
```

배치 명령(verify·replay·predict)은 numpy·pandas·lightgbm을 사용한다. 선택 기능인 REST API가 필요하면 `bash setup_pod.sh --with-api`로 fastapi·uvicorn·httpx를 준비한다. 별도의 격리 환경을 직접 구성하는 경우에만 `serving/requirements.txt`의 버전 목록을 사용한다.

lightgbm·numpy·pandas 는 **학습 때와 같은 버전**이어야 로더가 기동한다(매니페스트 `runtime`).
다르면 `VERSION_MISMATCH` 로 거부한다. 불가피하면 `KAMP_ALLOW_VERSION_MISMATCH=1` 로 우회하되,
그래도 golden 자가검증은 통과해야 한다.

## 3. 배포 — 번들은 outputs 밖으로 복사한다

```bash
mkdir -p /srv/kamp/bundles
cp -r outputs/models/full/deploy /srv/kamp/bundles/$(python -c "import json;print(json.load(open('outputs/models/full/deploy/manifest.json',encoding='utf-8'))['bundle_id'])")
```

`python run_all.py`의 10.5 단계는 해당 실행 모드의 번들을 지우고 다시 쓴다. 서빙 중인 번들이
바뀌지 않도록 **불변 사본**을 가리킨다. 롤백 = 이전 사본 경로로 바꾸고 재시작.

## 4. 배치 예측 (매일 24:00)

```bash
python -X utf8 -m serving verify  --bundle /srv/kamp/bundles/<id>
python -X utf8 -m serving predict --bundle /srv/kamp/bundles/<id> \
       --history history.csv --plan plan.csv --out pred_YYYYMMDD.csv
```

**입력** (UTF-8 BOM CSV, 원본 KAMP 한글 컬럼)

| 파일 | 컬럼 | 규칙 |
|---|---|---|
| `history.csv` | 날짜, 시간, 15분, 30분, 45분, 60분, 평균, 생산량, 공장인원, 인건비, 기온, 풍속, 습도, 강수량 | 대상일 **전날 23:00** 에서 끝나는 연속 시간별, 하루 단위 완결. **최소 8일 · 권장 14일** (API 는 최대 62일) |
| `plan.csv` | 날짜, 시간, 생산량, 공장인원, 기온, 풍속, 습도, 강수량 | 대상일 0~23시 24행. 생산량 = ERP 계획, 기상 = 예보. **전력 컬럼이 있으면 거부**(누수) |

- 결측 허용: 풍속·강수량·공장인원. 강수량은 0, 풍속·공장인원은 과거값 전파(`ffill`)로 처리하고 경고를 남긴다. 계획값에 관한 세부 입력 계약도 적용한다.
- 계측정지(평균=0)의 이력은 미래값을 사용하지 않는 과거값 전파로 처리한다. 유효한 과거값이 없는 입력 경계는 계약 검증 결과에 따른다.
- θ(=187) 는 다시 계산하지 않는다 — 매니페스트 값이 피처 정의의 일부다

**출력** 24행

| 컬럼 | 의미 |
|---|---|
| `y_avg_pred`, `y_peak_pred` | 평균전력·peak15 예측 (kW) |
| `regime` | 게이트가 고른 가동 레짐 (0 기저 · 1 중간 · 2 가동) |
| `peak_label` | `y_peak_pred >= τ` — 제출 파일 `peak_pred_label` 과 같은 규칙 |
| `peak_prob` / `peak_prob_cal` | 피크 확률 (보정 전 / 시간순 교차적합 평가 후 전체 OOF로 적합한 Isotonic 보정) |
| `peak_label_cls` | `peak_prob >= τ_cls` |
| `q10` · `q50` · `q90` | 분위회귀 (미보정 — 피복률은 해당 실행의 `ch5_uncertainty.csv`에서 확인) |
| `pi_lo` · `pi_hi` | 최종모델 OOF 잔차 기반 90% 구간 (휴무일/운영일 반폭 따로) |

과거 구간 재생(검증용):

```bash
python -X utf8 -m serving replay --bundle outputs/models/full/eval --data data/okm_augumented_2021.csv \
       --start 2021-09-01 --end 2021-09-14 --out replay.csv
```

- `--bundle` 은 번들 **디렉터리 경로**다(bundle_id 가 아니다).
- `--out` 을 빼면 결과 CSV 를 표준출력으로 낸다. `--window-days` 를 빼면 원자료 첫날부터의 이력 전체를 쓴다(배치 파이프라인과 비트 동일).
- `--window-days 14`를 추가하면 이력 창을 14일로 제한한다. 기본 전체 이력 재생 결과와 제출 예측의 일치는 S.2의 `outputs/steps/serving_replay_check.txt`로 확인한다.

출력의 90% 예측구간(`pi_lo`·`pi_hi`)은 개별 시간의 전력 불확실성을 나타내며, 모델 비교에서 쓰는 MAE 감소량의 90% 신뢰구간과 다르다.

## 5. REST API

```bash
KAMP_BUNDLE_DIR=/srv/kamp/bundles/<id> KAMP_API_KEY=<비밀> \
  uvicorn --factory serving.api:create_app --workers 1 --host 127.0.0.1 --port 8000
```

| 엔드포인트 | 설명 |
|---|---|
| `GET /healthz` | `{status, bundle_id, role, train_last}` |
| `GET /v1/model` | 피처 계약·τ·휴일 커버리지·주의사항 |
| `POST /v1/predict/day-ahead` | `{target_date, history[8~62일], plan[24]}` → `{meta, rows[24]}` |

JSON 필드는 ASCII 별칭이다: `date hour p15 p30 p45 p60 avg prod headcount labor temp wind humid rain`
(계획은 전력 필드 없이 `date hour prod headcount temp wind humid rain`).

- 기동 시 번들을 즉시 검증·로드한다. 실패하면 **뜨지 않는다**(fail-fast).
- 워커 1개. 번들 교체는 재시작으로 한다(핫 리로드 없음).
- `KAMP_API_KEY` 가 있으면 `/v1/*` 에 `X-API-Key` 헤더가 필요하다.
- 본문 2MB 초과 → 413, `Content-Length` 없는 요청(chunked) → 411. 모든 수치는 유한값·물리 범위 검증, 모르는 필드는 거부.
- 모든 오류는 한 봉투 `{"error": {code, message, detail?, request_id}}` (404·500 포함). 응답 헤더 `X-Request-ID`.
- 요청마다 stderr 에 JSON 로그 1줄(요청 id·경로·상태·지연). 페이로드는 남기지 않는다.

## 6. 오류 코드

| 코드 | HTTP / CLI | 뜻과 대처 |
|---|---|---|
| `SCHEMA`, `HISTORY_SCHEMA`, `PLAN_SCHEMA` | 422 / 2 | 필드 누락·숫자 아님·NaN/inf·범위 밖·존재하지 않는 날짜·계획 컬럼 24시간 전부 결측 |
| `PLAN_HAS_POWER` | 422 / 2 | 계획에 대상일 전력이 있다(누수) — 빼고 보내라 |
| `PLAN_INVALID` | 422 / 2 | 계획이 대상일 0~23시 24행이 아니다 |
| `HISTORY_INVALID` | 422 / 2 | 빠진 시각·중복·전날 23시에 끝나지 않음·하루 미완결 |
| `HISTORY_TOO_SHORT` / `HISTORY_TOO_LONG` | 422 / 2 | 8일 미만 / API 62일 초과. `detail.required_start` 참고 |
| `LENGTH_REQUIRED` · `BODY_TOO_LARGE` · `UNAUTHORIZED` · `NOT_FOUND` | 411 · 413 · 401 · 404 | Content-Length 없음 · 본문 2MB 초과 · API 키 · 없는 경로 |
| `METHOD_NOT_ALLOWED` | 405 | 허용되지 않은 메서드 (예: `GET /v1/predict/day-ahead`) |
| `SHUTDOWN_RUN_TRUNCATED` | 422 / 2 | 대상일이 휴무인데 휴무 연속구간이 이력 시작에서 잘렸다 — 더 이른 이력을 보내라 |
| `HISTORY_EDGE_OUTAGE` | 422 / 2 | 이력 시작의 계측정지가 피처 참조 구간(최근 7일)까지 이어진다 |
| `CALENDAR_NOT_COVERED` | 422 / 2 | 대상일 연도의 공휴일이 `config/holidays_kr.json` 에 없다 |
| `FEATURE_INVALID` | 422 / 2 | 대상일 피처에 결측 — 입력을 확인하라 |
| `BUNDLE_NOT_FOUND` · `BUNDLE_SCHEMA` · `BUNDLE_INTEGRITY` · `BUNDLE_CONTRACT` | 기동 실패 / 3 | 번들 없음·형식 오류·파일 또는 **매니페스트** 손상·변조·피처/게이트 계약 불일치 |
| `HOLIDAYS_INVALID` | 기동 실패 / 3 | 운영 휴일표 JSON 형식 오류·잘못된 날짜·명시한 파일 없음 |
| `FAST_BUNDLE` | 기동 실패 / 3 | 테스트용 축소 번들 — `full/` 번들을 쓰라 |
| `VERSION_MISMATCH` | 기동 실패 / 3 | lightgbm·numpy·pandas 버전이 학습 때와 다르다 |
| `SELFTEST_FEATURES` · `SELFTEST_PREDICTION` | 기동 실패 / 3 | 이 환경에서 피처·예측이 학습 때와 다르다 |
| `PREDICTION_NONFINITE` · `INTERNAL` | 500 / 1 | 예측값이 유한하지 않다 · 처리되지 않은 오류(서버 로그의 request_id 로 추적) |

> 무결성 검사는 **손상·실수로 인한 변경**을 잡는다. 번들 디렉터리에 쓸 수 있는 사람은 매니페스트까지 다시 쓸 수 있으므로,
> 보안은 서비스 계정에 번들 디렉터리 **읽기 전용** 권한만 주는 것으로 지킨다.

## 7. 공휴일 — 운영자가 관리한다

학습에 쓴 `HOLIDAYS_2021` 은 2021-08-16 까지만 있다. `serving/config/holidays_kr.json` 이
연도 커버리지(`coverage_years`)와 날짜를 보충한다(현재 2021년 전체). 새 연도를 서비스하려면
**공식 출처로 확인한** 공휴일·대체공휴일·임시공휴일을 추가하고 연도를 `coverage_years` 에 넣는다.
커버리지 밖 대상일은 `CALENDAR_NOT_COVERED` 로 거부한다 — 휴일을 모르는 채 조용히 평일로
예측하는 것보다 안전하다.

다른 휴일표를 쓰려면 CLI `--holidays <json>` 또는 환경변수 `KAMP_HOLIDAYS_FILE`(API 포함)로 지정한다.
지정한 파일이 없거나 형식이 틀리면 `HOLIDAYS_INVALID` 로 기동하지 않는다.

## 8. 한계 (모델 자체)

- 학습 데이터가 2021-01~09 뿐이다. month 10~12 와 다른 해는 **외삽**이다 (`/v1/model` 의 caveats).
- 배포 번들의 τ·τ_cls·보정기·구간 반폭은 평가 번들의 교차검증 값을 상속한다(재학습 구간에는 CV fold 가 없다).
- 기상은 학습 때 실측을 예보의 대리로 썼다. 운영에서는 예보 오차만큼 성능이 떨어질 수 있다.
- 새 데이터로 재학습하려면 `src/`의 분석 파이프라인을 새 데이터에 맞게 수정하고 `python run_all.py`로 검증해야 한다(2021 데이터에 묶인 게이트·상수가 있다).
- 확률 보정 성능은 시간순 교차적합으로 평가하지만 모델·임계값 선택에도 OOF를 사용하므로 전체 선택 절차가 완전히 독립적인 평가는 아니다.
- 사후 실측 부하 시나리오의 절감액을 이 서비스의 실제 운영 절감액으로 해석하지 않는다.

## 9. 테스트

| 명령 | 범위 |
|---|---|
| `python -X utf8 -B -m pytest tests/test_serving_unit.py -q -p no:cacheprovider` | 서빙 입력 계약·번들 무결성 단위테스트 |
| `python -X utf8 -B -m pytest tests -q -p no:cacheprovider` | 통계·학습 설정·시뮬레이션·인과성 수정 검증을 포함한 전체 테스트 |
| `python run_all.py` | FULL 학습·평가·번들 생성과 S.1~S.4·C 검증. `outputs/` 갱신 |
| `python run_all.py --check-only` | 저장된 산출물의 서빙·완료 점검. 모델을 재학습하지 않음 |

공통 전처리·피처·예측 로직은 `src/`에서 수정한 뒤 `python tools/build_serving_core.py`로 `serving/_core.py`를 재생성한다. `python tools/build_serving_core.py --check`의 동기화 점검과 `tests/test_causal_updates.py`·FULL 리플레이를 함께 확인한다. 실제 통과 개수와 생략 여부는 테스트 출력 및 `outputs/steps/index.json`에 따른다. 이 패키지의 단위테스트를 REST API의 운영 부하 검증으로 표현하지 않는다.
