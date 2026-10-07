이 폴더는 `run_all.py` 가 저장하는 모델(서빙용 번들과 비교 모델 아카이브)이 들어가는 곳이다.

# models — 모델 번들과 비교 모델

**결론**: 서빙에 쓰는 것은 `full/eval`·`full/deploy` 두 번들뿐이다. pickle 이 없고, 파일마다 sha256 으로 검증하며 읽는다.
기존 번들이 있어도 전처리·학습 설정 수정 후에는 `python run_all.py`로 다시 생성한다. 수정 전 번들을 새 코드의 평가·서빙 근거로 사용하지 않는다.

## 구성 (실행 후)

| 경로 | 내용 | 결과 묶음(zip)에 |
|---|---|---|
| `full/eval/` | **평가 번들** — 해당 실행의 제출 예측을 만든 모델(2021-08-31까지 학습). 재로드·리플레이 일치는 10.5·S.2 기록으로 확인 | 넣는다 |
| `full/deploy/` | **배포 번들** — 같은 방법으로 2021-09-14 까지 다시 학습한 모델. 경보 임계 τ·확률 보정·예측구간 폭은 평가 번들에서 물려받는다 | 넣는다 |
| `fast/…` | `--fast` 로 돌렸을 때의 축소 번들. **서빙 로더가 거부한다**(`FAST_BUNDLE`) — 동작 확인용일 뿐이다 | 넣지 않는다 |
| `baselines/…` · `comparison/…` | 4장 베이스라인·5장 비교 모델의 joblib 아카이브(보통 11개). joblib 은 pickle 이라 **내가 만든 파일만** 연다(`tools/README.md`) | 넣지 않는다 |

## 번들 형식 (`full/eval`, `full/deploy`, 각 약 25MB)

- `manifest.json` — 피처 44개의 순서·θ(187)·임계값 τ·구간 반폭·**실행 환경(runtime) 버전**·파일별 sha256·매니페스트 자신의 sha256.
- `*.lgb` — LightGBM 모델을 **텍스트**로 저장한 것(최종모델 2단계 레짐의 분류기·레짐별 회귀기, 피크 분류기, 분위회귀).
- `calibrator_isotonic.json` · `golden.json` — 확률 보정 임계점 · 기동 자가검증 사례.
- 10.5 단계가 저장 직후 다시 읽어 예측이 비트 단위로 같은지, 다시 학습한 해시가 같은지 확인하고 어긋나면 멈춘다.

확률 보정의 성능 평가는 과거 OOF로 보정하고 뒤 구간을 평가하는 시간순 교차적합으로 수행한다. 번들에는 평가를 마친 뒤 전체 OOF로 적합한 최종 보정기를 저장한다. `pi_lo`·`pi_hi`의 90% **예측구간**과 RF 대비 MAE 감소량의 90% **신뢰구간**은 목적과 계산 대상이 다르다.

## 검증 명령

```bash
python -X utf8 -m serving verify --bundle outputs/models/full/eval
python -X utf8 -m serving verify --bundle outputs/models/full/deploy
```

`run_all.py` 의 S.1 단계가 같은 검증을 하고 결과를 `outputs/steps/serving_verify_eval.json`·`serving_verify_deploy.json` 에 남긴다.

이번 FULL에서는 eval·deploy 두 번들이 모두 검증을 통과했다. S.2 리플레이 336행도 파이프라인의 메모리 예측 배열과 비트 단위로 일치하고 경보 라벨 비교를 통과했다. CSV에는 표시 자릿수 반올림이 있으므로 메모리 배열의 비트 일치와 CSV 파일 바이트 일치는 구분한다.

**버전 고정** — 번들은 만든 환경의 numpy·pandas·lightgbm 버전(매니페스트 `runtime`)과 **같은 환경에서만** 열린다.
KAMP 기준은 numpy 1.26.4 · pandas 2.1.4 · lightgbm 4.5.0 (`requirements.txt`·`serving/requirements.txt` 의 핀과 같다). 다르면 `VERSION_MISMATCH`.
같은 이유로 번들 이름과 해시(`outputs/tables/ch6_model_bundles.csv`)는 OS·버전이 다르면 달라진다.

## 왜 README 가 이 폴더에만 있나

`full/`·`fast/` 아래 번들 폴더는 10.5 단계가 **통째로 지우고 다시 쓴다**(`--skip-bundle` 이어도 해당 모드 폴더를 지운다).
`baselines/`·`comparison/` 도 실행 때마다 새로 만든다. 그래서 설명은 지워지지 않는 이 README 한 곳에 둔다.
번들을 서비스에 쓸 때는 이 폴더 밖으로 **복사한 사본**을 가리킨다(`serving/README.md` 3절) — 다시 실행하면 이 안의 번들이 바뀐다.
