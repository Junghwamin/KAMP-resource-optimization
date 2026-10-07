이 폴더는 비교 모델 저장 도우미 `model_persistence.py`, 서빙 코어 생성 도구 `build_serving_core.py`, 레짐 CV 분리 도구 `isolated_regime_cv.py`, 패키지 초기화 파일 `__init__.py`가 있는 곳이다. 추가 검증용 도구(`render_figures.py`, `figure_layout.py`, `publish_figures.py`, `verify_submission.py`, `verify_code_state.py`, `compare_supplemental.py`, `analysis_snapshot.py`, E2의 seed·반복수 민감도를 다시 계산하는 `bootstrap_seed_sensitivity.py`)의 실행 방법은 루트 `README.md`와 `PORTABILITY.md`에 있다.

# tools — 모델 저장·서빙 코어 생성·레짐 CV 분리

`model_persistence.py`는 `run_all.py`가 4·5장을 돌릴 때 사용한다. `build_serving_core.py`는 학습 코드의 공통 전처리·피처·예측 함수를 서빙 코드로 동기화할 때 직접 실행한다.

## build_serving_core.py — 학습·서빙 공통 함수

패키지 루트에서:

```bash
python tools/build_serving_core.py
python tools/build_serving_core.py --check
```

첫 명령은 `src/s01_diagnose.py`, `src/s02_features.py`, `src/s10_package.py`의 지정된 함수·상수를 AST로 추출하여 `serving/_core.py`를 다시 생성한다. 분석 모듈을 import하거나 모델을 학습하지 않는다. 두 번째 명령은 파일을 변경하지 않고 생성 예상값과 비교하며 일치하면 `SERVING_CORE_SYNC_OK`를 출력한다.

원본 함수는 `src/`에서 수정하고 생성된 `serving/_core.py`를 따로 편집하지 않는다. 동기화 확인 후 인과성 테스트와 FULL 번들 리플레이로 실제 동작을 검증한다.

## isolated_regime_cv.py — 6.6 레짐 변형의 프로세스 분리

Windows 장시간 학습 중 LightGBM 네이티브 DLL에서 프로세스 종료가 관측돼 해당 실행 경로를 분리했다. 정확한 원인은 확정하지 않았으며 분리 후 이번 FULL 검증에서는 재발하지 않았다. 6.6에서 2·3분류 레짐 변형마다 새 프로세스를 생성해 4개 fold를 학습하고 OOF 숫자 배열만 반환한다. 학습 모델 객체나 pickle을 프로세스 사이에 전달하지 않는다. 실행은 `run_all.py`가 관리하며 별도의 수동 실행은 필요 없다.

- 시드·트리 수·스레드 4개 등 기존 학습 설정을 유지한다. 2·3분류 각각에서 기존 방식과 분리 방식의 예측 배열이 `array_equal`로 일치하는 테스트를 통과했다.
- `KAMP_WORK_DIR/regime_cv`에 체크포인트를 저장한다. 환경변수를 지정하지 않으면 패키지 `.cache/regime_cv`를 쓴다.
- 소스·데이터·입력 프레임·설정·런타임·바이너리 지문과 결과 NPZ의 해시, 완료 JSON이 모두 일치해야 재사용한다. 불완전하거나 손상되거나 조건이 다른 캐시는 계산 근거로 사용하지 않는다.
- 동일 요청은 한 번 재시도하고 다시 실패하면 실패로 기록한다. 6.6 이전 단계는 재실행하므로 전체 파이프라인 이어하기 기능은 아니다.
- 작업 캐시는 제출·결과 ZIP에서 제외한다. 필요하면 실행 전에 충분한 여유 공간이 있는 폴더로 `KAMP_WORK_DIR`를 지정한다. 이는 결과 `outputs/`의 위치를 변경하지 않는다.

## model_persistence.py — 왜 있나

- 4장(가이드북 베이스라인)과 5장(제안모델 비교군)은 학습한 모델을 **joblib 아카이브**로 저장한다:
  `outputs/models/baselines/…`, `outputs/models/comparison/…` 의 `model.joblib` (보통 11개 — 완료 점검 C 가 개수를 info 로 알려 준다).
- 저장할 때 전처리 상태와 함께 담고, 곧바로 다시 읽어 **예측이 같은지 왕복 확인**한다. 어긋나면 멈춘다.
- 저장 형식과 검증 방법은 현재 패키지 코드가 기준이다. 코드 수정 후에는 `python run_all.py`로 아카이브를 다시 생성하고 해당 실행의 왕복 검증 결과를 확인한다.

**주의 — joblib 은 pickle 이다.** pickle 파일은 여는 순간 그 안의 코드가 실행될 수 있다.
그래서 **내가 이 컴퓨터에서 직접 만든 파일만** 연다. 남이 준 `.joblib` 은 열지 않는다.
이 때문에 비교 모델 아카이브는 결과 묶음(`python run_all.py --pack-results`)과 제출 zip 에 넣지 않는다.

최종모델과 서빙용 모델은 이 방식을 쓰지 않는다. `outputs/models/full/{eval,deploy}/` 번들은 **pickle 없이**
LightGBM 텍스트 파일 + JSON 매니페스트(파일별 sha256)로 저장되고, 서빙 코드가 해시를 검증하며 읽는다(`outputs/models/README.md`).

## __init__.py — 왜 있나 (가려짐 방지)

`src` 코드는 `from tools.model_persistence import …` 로 이 폴더를 찾는다.
폴더에 `__init__.py` 가 없으면 파이썬은 이 폴더를 우선순위가 낮은 '이름공간 패키지'로 취급한다.
그러면 공유 conda 환경에 `tools` 라는 이름의 다른 패키지가 설치돼 있을 때 **그쪽이 먼저 잡혀** import 가 실패한다.
설명 한 줄뿐인 `__init__.py` 를 두면 실행 폴더의 이 `tools` 가 항상 먼저 쓰인다. `bash check_env.sh` 9절이 이것을 확인한다.

통계 검정·피크 지표·비용 계산은 이 폴더가 아니라 `src/`에서 수행한다. 최신 수치는 `outputs/tables/`의 생성표를 사용하며, 저장된 이전 모델 아카이브와 새 전처리 코드를 섞어 사용하지 않는다.
