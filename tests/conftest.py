"""pytest 공통 설정.

TensorFlow 결정성 환경변수는 **TF import 이전**에 설정되어야 하므로
이 파일 최상단에서 처리한다. (PYTHONHASHSEED는 인터프리터 시작 시점에만
유효하므로 러너 스크립트 run_tests.ps1 / run_tests.sh 에서 설정한다.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── TF import 전에 반드시 설정 ────────────────────────────────────────────
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# 테스트는 항상 FAST 모드(노트북 본실행만 FULL)
os.environ.setdefault("KAMP_FAST", "1")

import pytest  # noqa: E402

# ── 경로 설정 ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES_DIR = PROJECT_ROOT / "src"

# stage 모듈을 import 가능하게
if str(STAGES_DIR) not in sys.path:
    sys.path.insert(0, str(STAGES_DIR))
# 서빙 패키지(serving/)를 `pytest` 단독 실행에서도 import 가능하게
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# stage 모듈은 CWD 기준 상대경로('data/...')를 쓴다 → CWD를 프로젝트 루트로 고정
os.chdir(PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def s00():
    """0장 — 환경·재현성 설정 모듈."""
    import s00_env

    return s00_env


@pytest.fixture(scope="session")
def s01():
    """1장 — 데이터 이해 및 진단 모듈 (import 시 해당 장이 실제로 실행된다)."""
    import s01_diagnose

    return s01_diagnose


@pytest.fixture(scope="session")
def s02():
    """2장 — 파생변수 구성 및 시간누수 차단."""
    import s02_features

    return s02_features


@pytest.fixture(scope="session")
def s03():
    """3장 — 분할 설계 및 평가지표."""
    import s03_split

    return s03_split


@pytest.fixture(scope="session")
def s04():
    """4장 — 가이드북 베이스라인 재현."""
    import s04_baseline

    return s04_baseline


@pytest.fixture(scope="session")
def s05():
    """5장 — 제안모델 개발."""
    import s05_models

    return s05_models


@pytest.fixture(scope="session")
def s06():
    """6장 — 성능평가 및 최종모델 선정."""
    import s06_eval

    return s06_eval


@pytest.fixture(scope="session")
def s07():
    """7장 — 영향요인 및 오류분석."""
    import s07_analysis

    return s07_analysis


@pytest.fixture(scope="session")
def s08():
    """8장 — 피크 저감 시뮬레이션."""
    import s08_simulation

    return s08_simulation


@pytest.fixture(scope="session")
def s09():
    """9장 — 창의성·차별성 근거 산출."""
    import s09_creativity

    return s09_creativity


@pytest.fixture(scope="session")
def s10():
    """10장 — 산출물·재현성·제출 패키징."""
    import s10_package

    return s10_package
