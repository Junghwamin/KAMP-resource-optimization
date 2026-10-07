# %% [markdown]
# # 제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석
#
# **제6회 K-인공지능 제조데이터 분석 경진대회 — 과제 ⑤ 자원 최적화**
#
# 선박엔진용 볼트·너트 제조공장의 2021년 1~9월 전력·생산·기상 데이터를 사용하여
# ① 다음 날 시간별 전력사용량 예측(Task A)과 ② 최대수요 피크 위험 사전 탐지(Task B)를 수행한다.
#
# 본 노트북은 `Restart Kernel → Run All` 한 번으로 전처리 → 학습 → 추론 → 결과생성이
# 자동 완료되며, 보고서 1~6장의 모든 표·수치·본문 초안이 `outputs/` 에 생성된다.
#
# | 노트북 장 | 보고서 대응 |
# |---|---|
# | 0. 실행 환경 및 재현성 설정 | 6장 |
# | 1. 데이터 이해 및 진단 | 1.2~1.4 |
# | 2. 파생변수 구성 및 시간누수 차단 | 1.5, 2.3 |
# | 3. 분할 설계 및 평가지표 | 1.6, 2.1 |
# | 4. 가이드북 베이스라인 재현 | 2.2 |
# | 5. 제안모델 개발 | 2.4, 2.5 |
# | 6. 성능평가 및 최종모델 선정 | 2.6~2.9 |
# | 7. 영향요인 및 오류분석 | 3장 |
# | 8. 피크 저감 시뮬레이션 | 4장 |
# | 9. 창의성·차별성 근거 산출 | 5장 |
# | 10. 산출물·재현성·제출 패키징 | 6장 |

# %% [markdown]
# ## 0. 실행 환경 및 재현성 설정
#
# 보고서 6장(코드 구성 및 재현성)의 근거가 되는 절이다.

# %% [markdown]
# ### 0.1 환경변수·시드 설정
#
# - **목적**: 실행마다 동일한 결과가 나오도록 난수원(random source)을 고정한다.
# - **보고서 대응절**: 6장
# - **산출물**: 없음(부수효과만)
#
# > ⚠️ 이 셀은 **반드시 모든 import보다 먼저** 실행되어야 한다.
# > `PYTHONHASHSEED`는 인터프리터 시작 시점에만, TensorFlow 결정성 플래그는
# > TF import 시점에만 반영되기 때문이다.

# %%
import os

# 해시 기반 자료구조의 순회 순서 고정 (인터프리터 시작 시점에만 유효)
os.environ.setdefault("PYTHONHASHSEED", "42")
# oneDNN 커스텀 연산은 연산 순서에 따라 부동소수점 결과가 달라진다 → 비활성화
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# TF 연산을 결정적 구현으로 강제
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
# TF 로그 억제(INFO/WARNING) — 노트북 출력 가독성
# 3 = ERROR 까지 억제. 2로 두면 absl/node_def 메시지가 셀 출력에 남고
# 그 메시지에 **설치 경로(사용자명 포함)** 가 박혀 블라인드 평가 위반이 된다.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SEED = 42

# 테스트 러너는 KAMP_FAST=1 로 탐색·에폭 수를 축소한다. 노트북 본실행은 FULL.
FAST = os.environ.get("KAMP_FAST", "0") == "1"

# %% [markdown]
# ### 0.2 라이브러리 임포트·버전 출력
#
# - **목적**: 사용 라이브러리를 한곳에서 임포트하고, 재현에 필요한 버전을 출력한다.
# - **보고서 대응절**: 6장 (`requirements.txt` 와 대조 가능)
# - **산출물**: 버전 표 (`outputs/tables/env_versions.csv`)

# %%
import random
import re
import sys
import time
import warnings
from pathlib import Path

# 서드파티 경고 메시지에는 **설치 경로가 그대로 포함**된다
# (예: ...\AppData\Local\Programs\Python\...\shap\plots\colors\_colors.py).
# 이 출력이 노트북 셀에 저장되면 블라인드 평가 규정을 위반하므로,
# import 하기 전에 경로가 새는 범주의 경고를 끈다.
# (무음 실패 승격은 0.6절에서 ChainedAssignmentError 로 별도 처리한다)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
warnings.filterwarnings("ignore", message=".*copy_on_write.*")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 재현성: 파이썬·NumPy 난수 고정
random.seed(SEED)
np.random.seed(SEED)

# TensorFlow 는 선택적 의존성으로 취급한다.
# 미설치 환경에서도 SimpleRNN·DNN 만 건너뛰고 파이프라인이 완주해야 한다.
try:
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    HAS_TF = True
    TF_VERSION = tf.__version__
except Exception as _e:  # pragma: no cover - 설치 환경에 따라 달라짐
    tf = None
    keras = None
    HAS_TF = False
    TF_VERSION = f"미설치({type(_e).__name__})"


def env_versions() -> pd.DataFrame:
    """재현에 필요한 런타임·라이브러리 버전을 표로 만든다.

    Returns
    -------
    pandas.DataFrame
        컬럼 `항목`, `버전`.
    """
    import sklearn

    rows = [("python", sys.version.split()[0])]
    # 핵심 라이브러리는 import 가능한 것만 기록(폐쇄망 심사환경 대비)
    for name, mod in [
        ("numpy", np),
        ("pandas", pd),
        ("matplotlib", matplotlib),
        ("scikit-learn", sklearn),
    ]:
        rows.append((name, mod.__version__))
    for name in ("lightgbm", "optuna", "shap", "scipy", "statsmodels"):
        try:
            rows.append((name, __import__(name).__version__))
        except Exception:
            rows.append((name, "미설치"))
    rows.append(("tensorflow", TF_VERSION))
    return pd.DataFrame(rows, columns=["항목", "버전"])


# %% [markdown]
# ### 0.3 전역 설정 — 경로·한글폰트·표시옵션
#
# - **목적**: 데이터 경로를 **상대경로**로 고정하고(제출 zip 단독 실행 보장),
#   한글 폰트 폴백 체인을 적용한다.
# - **보고서 대응절**: 6장
# - **산출물**: 선택된 폰트명 출력
#
# > 블라인드 평가 규정상 절대경로·사용자명이 노출되면 안 되므로
# > `os.getcwd()` 출력이나 절대경로 하드코딩을 하지 않는다.

# %%
# 데이터는 제출 zip 안에 동봉된 상대경로만 사용한다.
DATA_PATH = Path("data") / "okm_augumented_2021.csv"
OUTPUT_DIR = Path("outputs")

# CSV 에 BOM(﻿)이 있어 utf-8 로 읽으면 첫 컬럼명이 '﻿날짜' 가 된다.
CSV_ENCODING = "utf-8-sig"


def setup_korean_font() -> str:
    """한글 폰트 폴백 체인을 적용하고 선택된 폰트명을 반환한다.

    가이드북 경로(`NanumGothic.otf`)가 없는 로컬 환경을 위해
    `Malgun Gothic` → `Noto Sans KR` 순으로 폴백하며, 모두 없으면
    경고 후 기본 폰트를 사용한다.

    Returns
    -------
    str
        실제 적용된 폰트 패밀리명.
    """
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Malgun Gothic", "NanumGothic", "Noto Sans KR", "AppleGothic"):
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
            return candidate
    warnings.warn("한글 폰트를 찾지 못해 기본 폰트로 대체합니다.", RuntimeWarning)
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.family"][0]


FONT_NAME = setup_korean_font()

# 노트북에서는 인라인 백엔드, 스크립트/테스트에서는 무시
try:
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821
except (NameError, AttributeError):
    matplotlib.use("Agg")

# `display()` 는 노트북 전용 내장 함수다. 스크립트/테스트 실행에서도
# 동일 코드가 돌도록 폴백을 정의한다(노트북에서는 원래 것이 우선).
try:
    display  # noqa: B018,F821
except NameError:

    def display(*objs):  # noqa: D103
        for o in objs:
            print(o)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

# %% [markdown]
# ### 0.4 출력 디렉터리 준비
#
# - **목적**: 보고서·발표자료에 쓸 산출물 경로를 만든다.
# - **보고서 대응절**: 6장
# - **산출물**: `outputs/` 하위 디렉터리

# %%
FIG_DIR = OUTPUT_DIR / "figures"
TBL_DIR = OUTPUT_DIR / "tables"
REPRO_DIR = OUTPUT_DIR / "baseline_repro"
MODEL_DIR = OUTPUT_DIR / "models"

for _d in (OUTPUT_DIR, FIG_DIR, TBL_DIR, REPRO_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FIGURE_INDEX_PATH = FIG_DIR / "figure_index.csv"


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """표를 `outputs/tables/` 에 UTF-8(BOM) CSV 로 저장한다.

    Parameters
    ----------
    df : pandas.DataFrame
        저장할 표.
    name : str
        확장자 없는 파일명.
    index : bool
        인덱스 포함 여부.

    Returns
    -------
    Path
        저장된 파일 경로.
    """
    path = TBL_DIR / f"{name}.csv"
    # Excel 에서 한글이 깨지지 않도록 BOM 포함 저장
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


# %% [markdown]
# ### 0.5 시각화 유틸리티
#
# - **목적**: 그림을 **저장 → 인덱싱 → 인라인 출력** 순서로 일괄 처리한다.
# - **보고서 대응절**: 전 장 (그림↔절 매핑표가 6장 README 로 들어간다)
# - **산출물**: `outputs/figures/*.png`, `figure_index.csv`
#
# **팔레트**(인쇄 흰 배경 기준 대비 검증 완료)
#
# | 용도 | 색 |
# |---|---|
# | 범주형·인접형 최대 5계열 | `#2a78d6` `#eb6834` `#1baf7a` `#eda100` `#e87ba4` |
# | 범주형·전쌍형 최대 3계열 | `#2a78d6` `#eb6834` `#1baf7a` |
# | 순차형 | blue 단일 색조 `#cde2fb`→`#0d366b` |
# | 발산형 | blue ↔ red, 중립 `#f0efec` |
# | 강조 | 주인공 `#2a78d6` + 나머지 회색 `#b8b7b2` |
#
# **규칙**: 이중 축(`twinx`) 금지 · 모든 점에 숫자 금지(끝점·극값·주인공만) ·
# 격자는 실선 헤어라인 · 텍스트는 잉크색만(계열색을 글자에 입히지 않음).

# %%
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# 범주형 팔레트 — 인접형(선/막대/스택)은 5계열까지, 전쌍형(산점도)은 3계열까지
PALETTE_ADJACENT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
PALETTE_PAIRWISE = ["#2a78d6", "#eb6834", "#1baf7a"]
# 강조형 — 주인공 1색 + 나머지 회색
COLOR_HERO = "#2a78d6"
COLOR_MUTED = "#b8b7b2"
# 잉크색 — 텍스트 전용
INK = "#0b0b0b"
INK_SOFT = "#52514e"
# 순차형 / 발산형
CMAP_SEQ = LinearSegmentedColormap.from_list("kamp_seq", ["#cde2fb", "#0d366b"])
CMAP_DIV = LinearSegmentedColormap.from_list(
    "kamp_div", ["#0d366b", "#7fb0e8", "#f0efec", "#e8907a", "#a3271a"]
)


def apply_axis_style(ax) -> None:
    """축 스타일을 통일한다 — 헤어라인 실선 격자, 상·우 spine 제거."""
    ax.set_facecolor("#ffffff")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#d6d5d1")
    # 점선 격자는 금지 — 실선 헤어라인
    # 각 plot의 grid 설정을 보존한다.
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SOFT, length=0)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(INK_SOFT)


_FIGURE_INDEX: list[dict] = []


def save_fig(fig, fid: str, title: str, section: str, source_table=None, caption: str = ""):
    """그림을 저장·인덱싱하고 노트북에 인라인 출력한다.

    처리 순서가 중요하다. inline 백엔드는 `show()` 이후 figure 를 비우므로
    **저장을 show() 보다 먼저** 해야 빈 PNG 가 되지 않는다.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        저장할 figure.
    fid : str
        그림 ID (예: ``"F01"``).
    title : str
        그림 제목(파일명·캡션에 사용).
    section : str
        보고서 대응 절 (예: ``"1.5"``).
    source_table : pandas.DataFrame, optional
        그림의 소스 데이터. 주어지면 `outputs/tables/{fid}_src.csv` 로 동반 저장한다.
        대비가 낮은 계열색(#1baf7a/#eda100/#e87ba4)을 쓴 그림은 이 표가 있어야
        접근성 요건을 충족한다.
    caption : str
        보고서 캡션 문구.

    Returns
    -------
    Path
        저장된 PNG 경로.
    """
    # 파일명은 ASCII 안전하게 — 한글 제목은 fid 로만 식별
    slug = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", title).strip("_")[:40]
    path = FIG_DIR / f"{fid}_{slug}.png"

    # 저장 단계에서는 grid/tick/font/레이아웃을 변경하지 않는다.
    fig.patch.set_facecolor("#ffffff")

    # 1) 저장이 먼저 (show() 이후에는 figure 가 비워진다)
    from tools.figure_layout import save_figure_with_qa
    save_figure_with_qa(fig, path, fid=fid, dpi=300)

    # 2) 소스 표 동반 저장
    src_path = ""
    if source_table is not None:
        src_path = str(save_table(pd.DataFrame(source_table), f"{fid}_src", index=True))

    # 3) 인덱스에 1행 추가
    _FIGURE_INDEX.append(
        {
            "fid": fid,
            "파일명": path.name,
            "보고서절": section,
            "제목": title,
            "캡션": caption or title,
            "소스표": Path(src_path).name if src_path else "",
        }
    )
    pd.DataFrame(_FIGURE_INDEX).to_csv(FIGURE_INDEX_PATH, index=False, encoding="utf-8-sig")

    # 4) 인라인 출력
    plt.show()
    return path


# %% [markdown]
# ### 0.6 무음 실패 승격 — 조용히 반영되지 않는 코드를 즉시 중단시킨다
#
# - **목적**: 가이드북 베이스라인이 겪은 **무음 미반영**(silent no-op)을 자동 적발한다.
# - **보고서 대응절**: 2.2절 (원본 vs 보정)
# - **산출물**: 없음(부수효과만)
#
# 가이드북에는 `df['Weekend'][mask] = 1` 같은 **연쇄 대입(chained assignment)** 이 3곳 있는데,
# 이는 사본에 값을 쓰고 버려서 결과적으로 해당 컬럼이 **전부 0** 이 된다.
# 예외를 발생시키지 않기 때문에 "무오류 완주"만 보면 놓친다.
# pandas 3.x 의 Copy-on-Write 는 이 패턴을 `ChainedAssignmentError` 로 잡아준다.

# %%
def promote_silent_failures() -> None:
    """무음 실패를 예외로 승격시킨다.

    pandas 3.x 는 Copy-on-Write 가 기본이라 연쇄 대입이 이미 오류로 잡히지만,
    버전에 무관하게 동작하도록 명시적으로 설정한다.
    """
    try:
        pd.options.mode.copy_on_write = True
    except Exception:
        pass
    # SettingWithCopyWarning 계열을 오류로
    try:
        warnings.filterwarnings("error", category=pd.errors.ChainedAssignmentError)
    except AttributeError:
        pass


promote_silent_failures()


def _print_env_summary() -> None:
    """환경 요약을 출력하고 버전 표를 저장한다."""
    versions = env_versions()
    save_table(versions, "env_versions")
    print("=" * 62)
    print("실행 환경")
    print("=" * 62)
    for _, r in versions.iterrows():
        print(f"  {r['항목']:<14s} {r['버전']}")
    print(f"  {'한글폰트':<14s} {FONT_NAME}")
    print(f"  {'시드':<14s} {SEED}")
    print(f"  {'FAST 모드':<14s} {FAST}  (테스트=True, 본실행=False)")
    print("=" * 62)


_print_env_summary()
