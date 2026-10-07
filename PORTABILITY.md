# 추가 실험 제출본: Windows·KAMP-NOTE 실행 안내

이 문서는 기존 `README.md`의 환경·기준모델 설명에 추가되는 통합 실행 안내다. 모든 명령은 압축을 푼 프로젝트에서 `run_submission.py`, `run_all.py`, `requirements.txt`가 있는 루트를 기준으로 한다. 코드·데이터·모델·설정·검증 도구를 포함한 프로젝트 전체를 옮긴다.

**문서 작성 시점의 상태:** Windows 전체 계산과 최종 검증은 진행 중이다. 실제 KAMP-NOTE 실행은 아직 검증하지 않았다. 이 문서의 명령이나 Windows에서 저장한 결과만으로 KAMP 재현 완료를 뜻하지 않는다. 최종 실행 여부·환경·수치는 배포본에 동봉되는 실행 로그와 검증 결과를 확인한다.

## 1. 환경 원칙

- Windows 기준 실행은 **Python 3.11.9**이다. Python 3.11 계열을 사용하되 엄격한 재사용 검증은 패치 버전까지 대조한다. 실행하는 `python`이 의도한 환경의 인터프리터인지 `python --version`과 `python -c "import sys; print(sys.executable)"`로 확인한다.
- 프로젝트가 이미 사용하는 `requirements.txt`, `setup_pod.sh`, 테스트 도구를 사용한다. 추가 실험을 위해 새로운 라이브러리를 요구하지 않는다.
- `requirements.txt`에 고정된 주요 버전은 아래와 같다. 독립 환경 설치에는 기존 의존성의 버전을 함께 제한하는 `verification/reproduction_constraints.txt`를 사용한다. 전체 실제 설치 환경은 `verification/environment_windows.json`과 `verification/environment_windows_freeze.txt`에 기록되어 있다.

| 패키지 | 기준 버전 |
|---|---:|
| numpy | 1.26.4 |
| pandas | 2.1.4 |
| scipy | 1.11.4 |
| scikit-learn | 1.4.2 |
| matplotlib | 3.9.2 |
| lightgbm | 4.5.0 |
| optuna | 5.0.0 |
| statsmodels | 0.14.2 |
| tensorflow | 2.17.0 |
| shap | 0.49.1 |
| keras | 3.15.1 |
| joblib | 1.6.0 |
| pytest | 9.0.2 |

TensorFlow는 기존 DNN·SimpleRNN 비교모델에 필요하다. 최종 레짐 모델이 LightGBM이라고 해서 TensorFlow 비교모델이나 테스트를 생략한 실행을 전체 완료로 표시하지 않는다.

`keras`는 기존 TensorFlow 의존성이고 `pytest`는 기존 테스트 도구다. 제약 파일은 Windows에서 실제 사용한 이 버전들까지 명시한다. 모든 전이 의존성을 완전히 고정한 플랫폼 공통 lock 파일은 아니므로, 설치 후 실제 환경도 기록해 대조한다.

Windows의 가상환경 폴더를 Linux로 복사하지 않는다. **Windows `pip freeze`에 있는 `tensorflow-intel` 등 플랫폼 전용 배포 패키지를 Linux에 그대로 설치하지 않는다.** Windows에서는 같은 TensorFlow 버전이 플랫폼에 맞는 구현 패키지로 설치될 수 있다. Linux에서는 프로젝트의 `tensorflow==2.17.0` 요구와 Linux용 배포 패키지를 기준으로 환경을 준비한다. 전체 `pip freeze`는 실행 환경의 증거이며, 다른 운영체제의 설치 파일을 대신하지 않는다.

## 2. Windows PC

이미 검증용 Python 환경을 준비했다면 해당 환경의 `python`을 사용한다. 새 환경이 필요하면 프로젝트 바깥에 별도 가상환경을 만든다. 다음 설치 명령은 **새로 만든 독립 가상환경에만** 적용한다.

```powershell
py -3.11 -m venv ..\.venv_kamp
..\.venv_kamp\Scripts\python.exe -m pip install -r requirements.txt -c verification/reproduction_constraints.txt
..\.venv_kamp\Scripts\python.exe -m pip install pytest==9.0.2 -c verification/reproduction_constraints.txt
```

실행 파일 경로를 직접 지정하면 PowerShell의 활성화 스크립트 설정에 의존하지 않는다. 완성 ZIP에는 Windows 기준 `outputs/analysis_snapshot`이 이미 포함되어 있으므로, **새 전체 계산에는 새 snapshot·실험·그림 폴더를 지정한다.** 아래 경로가 이미 사용된 경우에는 다른 새 이름을 사용한다.

```powershell
..\.venv_kamp\Scripts\python.exe -u -X utf8 run_submission.py --profile full --cpu --snapshot outputs/windows_rerun_snapshot --run-dir outputs/additional/windows_rerun --figures outputs/windows_rerun_figures --reference outputs/analysis_snapshot
```

이미 해당 가상환경이 활성화되어 있다면 아래 명령도 같은 실행이다.

```powershell
python -u -X utf8 run_submission.py --profile full --cpu --snapshot outputs/windows_rerun_snapshot --run-dir outputs/additional/windows_rerun --figures outputs/windows_rerun_figures --reference outputs/analysis_snapshot
```

Windows에서는 무거운 모델 적합을 순차 격리 프로세스로 실행하고 LightGBM 스레드 수를 4로 고정한다. 실행 도중 다른 폴더의 `src/`, 데이터, 설정을 덮어쓰지 않는다. 전체 소요시간은 실제 로그로 확인하며, 고정된 분 단위 완료시간을 보장하지 않는다.

## 3. KAMP-NOTE

### 기존 TensorFlow 팟 사용

1. 완성 제출 ZIP을 새 작업 폴더에 풀고 프로젝트 루트로 이동한다. Windows 원본 ZIP과 기준 산출물은 비교를 위해 보존한다.
2. Python 3.11과 TensorFlow가 제공되는 팟인지 확인한다.
3. 기존 환경 점검과 설치 도우미를 실행한다.

```bash
python --version
python -c "import sys; print(sys.executable)"
bash check_env.sh
bash setup_pod.sh --with-tests
```

`setup_pod.sh`는 이미 설치된 패키지를 유지하고 부족한 패키지를 준비한다. 이 때문에 실행 가능해졌다는 사실만으로 기준 버전과 모두 일치한다고 판단하지 않는다. 실행 전 실제 버전을 `verification/environment_windows.json` 및 `verification/reproduction_constraints.txt`와 대조한다. 버전이 다르면 차이를 기록하고 동일 조건 재현으로 판정하지 않는다. 기준과 다른 공유 패키지를 교체하지 말고 아래의 독립 환경을 사용한다.

**공유 conda 팟에 `pip install -r requirements.txt`를 직접 실행하지 않는다.** 기존 공용 패키지를 교체하려다 권한·의존성 문제가 생길 수 있다. 기존 팟의 TensorFlow를 유지하는 `setup_pod.sh` 경로를 먼저 사용한다. `setup_pod.sh --venv`는 기존 패키지를 상속하는 방식이므로, 버전이 다른 패키지까지 기준 버전으로 맞추는 기능은 아니다.

### 기준 버전을 맞춘 별도 환경

팟의 기본 버전이 달라 독립 환경을 준비할 수 있다면, Python 3.11의 **공유 패키지를 상속하지 않는 새 가상환경**에서 기존 요구사항을 설치한다.

```bash
python3.11 -m venv ../.venv_kamp_linux
source ../.venv_kamp_linux/bin/activate
python -m pip install -r requirements.txt -c verification/reproduction_constraints.txt
python -m pip install pytest==9.0.2 -c verification/reproduction_constraints.txt
```

위 명령은 생성한 독립 환경 안에서만 사용한다. Python 3.11이나 해당 패키지의 설치가 지원되지 않으면 설치 로그를 남기고 환경 준비 실패로 기록한다. 다른 버전이나 모델 생략으로 조용히 대체하지 않는다.

### KAMP에서 실제 전체 재실행

제출 ZIP의 Windows 기준 snapshot을 그대로 보존하고, **KAMP에서 새로 계산할 경로**를 지정한다. 아래 명령은 기존 기준 학습과 추가 실험을 실제로 다시 실행한다. `--reuse-core`는 사용하지 않는다.

```bash
python -u -X utf8 run_submission.py --profile full --cpu --snapshot outputs/kamp_analysis_snapshot --run-dir outputs/additional/kamp_run --figures outputs/kamp_figures --reference outputs/analysis_snapshot
```

`outputs/kamp_analysis_snapshot`은 없거나 비어 있어야 한다. 같은 경로로 새 기준 실행을 반복하면 덮어쓰기를 거부하므로, 다음 새 실행에는 `kamp_analysis_snapshot_2`, `kamp_run_2`, `kamp_figures_2`처럼 새 경로를 지정한다. 기존 기준본을 지워서 통과시키지 않는다.

`--reference outputs/analysis_snapshot`은 새 KAMP 기준 예측을 ZIP에 동봉된 Windows 기준 예측과 비교한다. **E1~E6의 플랫폼 간 결과 비교는 이 옵션의 범위에 포함되지 않는다.** 두 환경의 추가 실험이 모두 완료되면 다음 전용 비교 명령을 별도로 실행한다.

```bash
python tools/compare_supplemental.py --reference outputs/additional/validated_run --current outputs/additional/kamp_run --output outputs/verification/kamp_supplemental_comparison.json
```

비교 결과 JSON의 `status`가 `passed`이고 종료코드가 0이어야 추가 6개 실험의 수치 대조를 통과한 것이다. 미완료 실험·누락·무결성 오류·선정 변경·허용오차 초과는 실패로 기록하고 종료코드 1을 반환한다. 이 명령은 학습을 실행하지 않는다.

실행한 환경의 정보, 시작·종료 상태, 모든 실험 결과와 테스트 로그를 보관한다. 직접 시각 검토가 아직 없으면 계산 완료 후 `COMPUTE_OK_VISUAL_PENDING`과 종료코드 3이 나올 수 있다. 이는 아래에서 설명하는 시각 검토 대기 상태다.

## 4. `--reuse-core`와 중간 결과 재사용

```text
python -u -X utf8 run_submission.py --profile full --cpu --reuse-core
```

`--reuse-core`는 검증된 기존 기준 실행과 `outputs/analysis_snapshot`을 재사용하여 **기존 62단계 계산만 생략**하는 옵션이다. 추가 실험 E1~E6, 그림 생성, 테스트·산출물 검증의 완료 요구는 그대로 적용된다.

재사용하려면 snapshot의 완료 상태와 저장 파일 해시, 데이터·설정·관련 계산 소스 및 실행환경에 대한 검증을 통과해야 한다. 현재 Python 패치 버전과 기록된 패키지 버전도 대조한다. 필요한 파일이 없거나 불완전·손상·불일치 상태이면 재사용 성공으로 처리하지 않는다. 불일치 시에는 새 snapshot 경로로 기준 계산부터 실행한다. 통합 추가 코드의 출처·해시는 별도 실행 manifest로 기록한다.

Windows snapshot을 KAMP로 가져와 검사하거나 재사용한 것은 **이미 계산된 산출물 검사**다. KAMP에서 기준 학습을 새로 실행한 증거가 아니므로, KAMP 전체 재현 확인에는 `--reuse-core`를 사용하지 않은 실제 전체 실행이 필요하다. 플랫폼별 런타임·바이너리 차이 때문에 Windows 모델 적합 캐시가 Linux에서 그대로 재사용된다고 기대하지 않는다.

추가 실험은 해시로 확인한 수치 체크포인트를 재사용할 수 있다. 설정·행·피처 순서·소스·런타임이 바뀌면 해당 적합의 캐시는 다시 계산해야 한다. 실행 중단이나 실패 상태의 파일이 존재하는 것만으로 완료로 판단하지 않는다.

## 5. 전체 프로필과 결과 위치

E4는 동일 회귀설정의 단일 LightGBM·3레짐 비교다. E5는 고정된 44개 피처와 다섯 후보를 사용하며, O2~O5 네 평가구간에서 회귀·분류 HPO를 각각 50회 수행한다. 800 trees, seed 42, 24시간 gap, 14일 보정구간과 사전 후보 순서를 유지한다.

최종 모델과 피크 기준은 core에서 정하고, 별도 보정구간에서 경보 임계값·확률보정을 정한다. 보정구간을 합쳐 모델을 다시 학습하지 않는다. 측정된 학습시간은 자원 기록에만 사용하고 후보선정 점수에는 넣지 않으므로 운영체제의 속도 차이가 선정 결과를 바꾸지 않는다.

등록한 피처·기간·예산·설정을 축소하거나 바꾼 실험은 별도 변경 이력과 `partial` 상태로 구분한다. 점수가 낮거나 신뢰구간이 0을 포함하는 결과 자체는 실행 실패가 아니다.

| 상대경로 | 내용 |
|---|---|
| `src/`, `data/`, `serving/`, `tests/`, `tools/` | 기존 제출 파이프라인과 입력·검증 코드 |
| `experiments/config.json` | 사전에 고정한 추가 실험 설정 |
| `outputs/analysis_snapshot/` | 검증된 기준 실행의 피처·예측·설정 snapshot |
| `outputs/kamp_analysis_snapshot/` | 위 명령으로 KAMP에서 새로 계산한 기준 snapshot |
| `outputs/additional/<run_id>/bootstrap/` | E2 블록 길이 민감도 |
| `outputs/additional/<run_id>/diagnostics/` | E1 조건별 오류율·교차 분석 |
| `outputs/additional/<run_id>/sensitivity/` | E3 계획·기상 입력오차 민감도 |
| `outputs/additional/<run_id>/controlled/` | E4 동일 회귀설정 비교 |
| `outputs/additional/<run_id>/temporal/` | E5 시간순 후보선정·보정·원행 예측 |
| `outputs/additional/<run_id>/policy/` | E6 예측 경보 기반의 사후 정책 모의 |
| 각 실험의 manifest·summary·로그 | 완료/부분/실패 상태, 설정과 입력·산출물 연결 |
| `outputs/figures_rerendered/` | 통합 실행에서 다시 생성한 그림 |
| `outputs/additional/<run_id>/submission_pipeline.json` | 통합 실행 단계별 상태 |
| `outputs/additional/<run_id>/verification_summary.json` | 계산·그림 검증의 최종 상태와 근거 |
| `outputs/verification/current_code_test_manifest.json` | 현재 코드·테스트·문서와 연결된 전체 테스트 실행 증거 |
| `outputs/verification/kamp_supplemental_comparison.json` | 전용 비교 명령으로 생성한 Windows·KAMP 추가 6개 실험 대조 결과 |

기본 `<run_id>`는 `validated_run`이고 위 KAMP 명령은 `kamp_run`을 사용한다. `--run-dir`, `--snapshot`, `--figures`로 바꾼 경우 실제 통합 실행 로그의 경로를 따른다. 실제 Windows 환경은 `verification/environment_windows.json`과 `verification/environment_windows_freeze.txt`를 확인한다. 폴더 이름에 `full`이 있거나 기존 `RUN_ALL_OK`가 찍혔다는 사실만으로 추가 실험과 그림 검증까지 완료되었다고 판단하지 않는다.

위 KAMP 통합 실행 후 같은 산출물을 다시 검증하려면 다음 명령을 사용한다. Windows에서 검증할 때에는 해당 Windows 실행의 snapshot·실험·그림 경로를 지정한다.

```text
python tools/verify_submission.py --mode strict --snapshot outputs/kamp_analysis_snapshot --run-dir outputs/additional/kamp_run --figures outputs/kamp_figures --config experiments/config.json --reference outputs/analysis_snapshot
```

| 검증 상태 | 의미 |
|---|---|
| `COMPUTE_OK_VISUAL_PENDING` / 종료코드 3 | 계산 검증은 통과했으나 최종 이미지의 직접 시각 검토가 남아 있음 |
| `FULL_OK` / 종료코드 0 | 계산 검증과 해당 그림 파일 해시에 연결된 직접 시각 검토가 모두 통과함 |

직접 시각 검토 기록은 `--visual-review <QA파일>`로 전달한다. QA 파일은 검토한 실제 그림의 해시에 연결되어야 한다. Windows에서 검토한 기록을 그림이 바뀐 KAMP 산출물에 그대로 적용하지 않는다. 필요한 검증을 생략하거나 QA 상태 문자열만 바꾸어 `FULL_OK`로 만들지 않는다.

통합 실행기는 그림 생성 뒤에 현재 코드 상태의 전체 테스트도 자동으로 실행한다. 그 뒤 코드·테스트·문서·환경 제약 파일이 변경되면 이전 테스트 기록을 그대로 최종 증거로 사용할 수 없다. 최종 검증 전에 아래 명령으로 다시 수집한다.

```text
python tools/verify_code_state.py --record-code-tests
```

이 명령은 실제 전체 테스트를 실행하고 코드·환경의 실행 전후 해시 및 로그를 기록한다. 테스트 기록을 갱신한 다음 해당 KAMP 출력 경로로 `verify_submission.py --mode strict`를 다시 실행한다.

이 통합 판정의 범위는 **로컬 코드·결과·그림(`local_code_results_and_figures`)**이다. 최종 보고서 PDF 검토와 실제 KAMP 재현 여부는 별도 상태이므로, Windows의 `FULL_OK`가 PDF·KAMP 확인까지 끝났다는 의미는 아니다.

## 6. Windows·KAMP 결과 대조 기준

`run_submission.py`와 `verify_submission.py`의 `--reference`에는 Windows 기준 **snapshot 폴더 전체**를 지정한다. 이 폴더에는 `metadata.json`과 `payload/`의 모든 파일이 있어야 한다. 제출본에서는 `outputs/analysis_snapshot`이 이 기준본이다. 다른 위치로 옮겼다면 해당 폴더를 지정한다.

이 옵션은 baseline `oof_long`·`test_long`의 원행 키·정답·라벨을 정확히 비교하고, 예측·확률·임계값은 `atol=1e-6`, `rtol=1e-6`으로 비교한다.

E1~E6는 앞서 제시한 **`tools/compare_supplemental.py`**로 대조한다. 이 도구의 `--reference`·`--current`에는 snapshot이 아니라 **추가 6개 실험을 담은 실행 폴더**를 지정한다. 각 실행의 자체 검증만으로 플랫폼 간 대조가 자동 완료되지 않으므로 이 명령을 실행한 결과도 보관한다.

| 비교 도구 | Windows 기준 경로 | KAMP 결과 경로 | 비교 범위 |
|---|---|---|---|
| `verify_submission.py --reference` | `outputs/analysis_snapshot` | `--snapshot outputs/kamp_analysis_snapshot` | 기준 OOF·테스트 원행 예측 |
| `compare_supplemental.py` | `--reference outputs/additional/validated_run` | `--current outputs/additional/kamp_run` | E1~E6 설정·선정·보정·원행·지표·신뢰구간 |

전용 비교기는 양쪽의 6개 실험 완료 상태, 실행 manifest와 모든 산출물 해시, 지원 스키마를 먼저 확인한다. 원행은 키로 정렬하므로 저장 행 순서만 바뀐 것은 차이로 보지 않는다. 설정·후보·피처 순서·선정 모델·정답·라벨·품질 마스크는 정확히 비교하고, 실수 예측·확률·지표·신뢰구간은 `atol=1e-6`, `rtol=1e-6`을 적용한다. 결측 패턴 차이와 중복 키도 실패 사유다. 실행시간·캐시 사용·OS·출처 해시값 자체의 동일성은 요구하지 않지만 양쪽 출처 기록을 결과 JSON에 남긴다.

비교기의 성공은 **제공한 두 결과 폴더의 대조 통과**다. Windows 파일을 KAMP 결과 폴더에 복사해 비교한 것은 실제 KAMP 실행의 증거가 아니므로, 새 KAMP 전체 실행 로그를 함께 보관해야 한다. 현재 실제 KAMP 재현은 여전히 미검증이다. 시각 결과도 KAMP에서 생성한 실제 그림으로 확인한다.

- 입력 파일·설정·피처 순서·분할 경계·정답과 경보 라벨을 먼저 비교한다. 플랫폼이 다르다는 이유로 데이터나 후보를 바꾸지 않는다.
- 같은 환경에서 결정론적으로 재실행한 원시 예측은 기본 절대 허용오차 `1e-8`로 확인한다. 플랫폼 간 원시 예측은 사전 기준 `atol=1e-6`, `rtol=1e-6`으로 대조하고, 초과하면 원인을 조사한다. 결과에 맞춰 허용오차를 사후에 넓히지 않는다.
- 모델선정·경보·분모·보고서 표시 정밀도의 핵심 지표는 별도로 일치 여부를 확인한다. 소수점 이하의 작은 예측 차이도 임계값 주변에서 경보를 바꿀 수 있다.
- 운영체제별 글꼴과 렌더러 차이가 있으므로 PNG 픽셀 전체의 동일성을 요구하지 않는다. 모든 그림의 데이터·단위·문구를 대조하고 실제 생성 이미지와 최종 PDF에서 글자 겹침·잘림·누락을 확인한다.
- Windows 실행 기록을 KAMP의 실행 기록으로 재명명하지 않는다. KAMP 실제 실행이 없으면 상태는 `미검증`이다.

최종 배포에는 상대경로로 구성한 코드·데이터·필수 모델·예측·추가 결과·검증 기록을 포함한다. 가상환경, `__pycache__`, 개인 절대경로를 담은 적합 캐시와 임시 작업 상태는 제출 ZIP의 실행 의존물로 취급하지 않는다.

## 7. 해석 범위

E5는 기존 자료를 이용한 **회고적 시간순 후보선정 평가**이며 새 독립 테스트가 아니다. `plan_known`과 가동 캘린더에는 사후 ERP 결측 진단이 사용되므로, 이를 예측 시점에 확보한 계획 확인정보의 대리값으로 간주한다. 실제 원점의 정보 가용성을 검증한 데이터는 아니다.

E6는 관측 이력으로 만든 예측을 고정한 가정 기반 사후 모의다. 생산 조정 뒤의 전력 이력을 다음날 모델에 다시 넣는 폐루프 운영이나 실제 현장 절감, 연간 청구액 절감을 입증한 결과로 해석하지 않는다.
