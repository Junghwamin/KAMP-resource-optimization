# tests — 서빙·분석 회귀테스트

서빙 입력 계약·번들 무결성과 통계·학습 설정·시뮬레이션·전처리의 수정 동작을 검증한다. 테스트 파일과 실제 수집 개수는 현재 패키지가 기준이며, 통과 여부는 실행 결과로 판단한다.

이번 FULL 파이프라인의 S.4에서 **123개가 17.42초에 통과**했다. 62개 단계도 생략 없이 완료됐으며 필수 점검 15개와 보조 점검 4개가 통과했다. 실제 현장 절감 효과나 새로운 외부 데이터에서의 일반화 성능을 증명하는 수치는 아니다.

## 실행

패키지 루트에서 전체 테스트를 실행한다.

```bash
python -X utf8 -B -m pytest tests -q -p no:cacheprovider
```

서빙 입력 계약만 확인하려면:

```bash
python -X utf8 -B -m pytest tests/test_serving_unit.py -q -p no:cacheprovider
```

- pytest가 없으면 `bash setup_pod.sh --with-tests`로 준비한다.
- `-p no:cacheprovider`는 pytest 캐시 생성을 끄고, `-B`는 Python 바이트코드 캐시 생성을 끈다.
- `python run_all.py`의 S.4 테스트 기록은 `outputs/steps/serving_unit_tests.txt`에 남는다. 파일명은 호환성을 위해 유지하며 실제 실행 범위·개수는 로그로 확인한다.
- 테스트 생략(`--no-tests`)이나 skip은 통과와 다르다. FULL 파이프라인의 완료·번들 리플레이까지 확인해야 최종 산출물 검증을 마친 것이다.

## 범위

| 파일·분류 | 확인할 내용 |
|---|---|
| `test_serving_unit.py` | 번들 로드·예측, 변조·파일 누락, FAST 번들, 라이브러리 버전, 입력 계약(이력·계획), 오류 코드 등 기존 서빙 검증 |
| `test_statistics_updates.py` | 일 단위 부트스트랩의 95% 기본·90% 사후 보조 구간, 실제 피크 표본의 오차와 별도 평균전력 고부하 지표 |
| `test_model_training_updates.py` | 탐색·최종 학습 설정 일치, 확률 보정의 시간순 교차적합 경계, DNN·앙상블의 평가 동작 |
| `test_simulation_updates.py` | 전력량요금 절감의 산정 제외, 비용·편익 기간 일치, 연간 기본요금 환산값 분리, 이동 후 새 피크를 상한으로 잘라내지 않는 계산 |
| `test_causal_updates.py` | 6개 사례: 미래 관측값이 이전 원점의 전처리·피처에 영향을 주지 않는지와 학습·서빙 정합성 |
| `test_isolated_cv.py` | 레짐 2·3분류의 기존/분리 프로세스 예측 배열 일치, 체크포인트 검증, CV 모델 참조 해제 |
| `test_runner_import_boundary.py` | `run_all.py`의 실제 6.6 실행 셀에 프로세스 분리 도구 등 필요한 import가 남아 있는지 검증 |
| `serving_helpers.py` | 임시 폴더의 작은 합성 번들 생성. 실제 FULL 번들 없이 서빙 계약 검사 가능 |
| `conftest.py` | 결정성 환경변수·작업 폴더 등 pytest 공통 설정 |

전체 목록은 `python -X utf8 -m pytest tests --collect-only -q -p no:cacheprovider`로 확인한다. 테스트 이름이나 이전 `76 passed` 기록을 새 실행의 전체 통과 수로 사용하지 않는다. S.4도 `tests/` 전체를 실행하며 실제 결과를 로그에 기록한다.

## 전체 실행과의 관계

단위·회귀테스트는 특정 경계와 계산 규칙을 확인하며, 최종 모델 성능을 대신 검증하지 않는다. `run_all.py`의 S.1~S.3은 생성된 FULL 번들로 무결성·리플레이·예측 데모를 확인하고, C는 산출물 규격과 필수 점검을 확인한다.

`conftest.py`의 `s00`~`s10` fixture는 해당 장까지 파이프라인을 실행할 수 있다. 신규 테스트가 이를 사용하게 되면 `outputs/` 영향과 소요시간을 먼저 확인한다. 실제 API 서비스의 운영 부하나 현장 절감 효과를 이 테스트의 통과만으로 주장하지 않는다.

분리 CV의 일치 테스트는 작은 합성 데이터에서 시드·학습 설정을 맞춰 `array_equal`로 확인한다. 실제 전체 실행·번들·336행 리플레이도 이번 FULL에서 별도로 통과했다. 프로세스 분리 후 이번 검증에서는 네이티브 DLL 종료가 재발하지 않았지만 원인을 확정하거나 모든 환경에서 재발하지 않는다고 보장한 것은 아니다.
