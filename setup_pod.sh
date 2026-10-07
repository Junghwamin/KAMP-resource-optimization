#!/usr/bin/env bash
# KAMP 1회 준비 — 공유 conda 환경 · 비root 를 기본으로 가정한다.
#
#   bash setup_pod.sh                 # 없는 패키지만 설치 (optuna·shap 등)
#   bash setup_pod.sh --with-tests    # + pytest (서빙 단위테스트 76개용, 권장)
#   bash setup_pod.sh --with-api      # + fastapi · uvicorn · httpx (REST API 를 띄울 때만)
#   bash setup_pod.sh --venv          # ~/.venvs/kamp-exp 가상환경을 만들어 거기에 설치
#   bash setup_pod.sh --dry-run       # 무엇을 할지만 보여 준다 (설치·삭제 없음. pip 가 풀이용 파일을 임시로 받을 수 있다)
#
# 설계 원칙
#   1. 남의 환경을 건드리지 않는다 — base 에 쓰기 권한이 없으면 --user 나 venv 로 간다.
#   2. 이미 있는 것은 건드리지 않는다 — 설치된 핵심 패키지 버전을 constraints 로 묶고
#      없는 것만 설치한다. 그래서 새 패키지가 numpy·TensorFlow 를 조용히 바꾸지 못한다.
#   3. 'pip install -r requirements.txt' 와 'pip install --upgrade pip' 은 쓰지 않는다
#      (공유 conda 에서 이미 있는 numpy 를 바꾸려다 Permission denied 로 막힌 적이 있다).
set -uo pipefail
cd "$(dirname "$0")" || exit 2

DRY=0; WITH_TESTS=0; WITH_API=0; FORCE_VENV=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)    DRY=1 ;;
        --with-tests) WITH_TESTS=1 ;;
        --with-api)   WITH_API=1 ;;
        --venv)       FORCE_VENV=1 ;;
        -h|--help)    sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "모르는 옵션: $1  (쓸 수 있는 옵션: --dry-run --with-tests --with-api --venv)"; exit 2 ;;
    esac
    shift
done

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg TF_CPP_MIN_LOG_LEVEL=3
unset PYTHONSAFEPATH 2>/dev/null || true

ok()   { printf '  \033[32m✔\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✘\033[0m %s\n' "$1"; }
step() { printf '\n\033[1m[%s]\033[0m %s\n' "$1" "$2"; }

FAILS=""                       # 실패를 모아 두었다가 마지막에 보고한다
fail() { bad "$1"; FAILS="${FAILS}
   - $1"; }
SOFT=""                        # 코드 실행은 막지 않는 문제(한글 폰트·선택 패키지) — 따로 알린다
soft() { warn "$1"; SOFT="${SOFT}
   - $1"; }

PY=python
command -v "$PY" >/dev/null 2>&1 || PY=python3
command -v "$PY" >/dev/null 2>&1 || { echo "python 명령이 없다"; exit 1; }

# 경로의 홈 부분을 '~' 로 보여 준다
tilde() { printf '%s' "${1/#$HOME/\~}"; }

[ "$DRY" = "1" ] && printf '\033[1m--dry-run: 아무것도 설치·삭제하지 않는다(pip 가 의존성 풀이용 파일을 임시로 받을 수 있다). 할 일만 보여 준다.\033[0m\n'

# ──────────────────────────────────────────────────────────────────────
step 1/6 "설치 위치 정하기"
WRITABLE=$("$PY" -c "import os,sysconfig;print(1 if os.access(sysconfig.get_paths()['purelib'],os.W_OK) else 0)")
USERSITE=$("$PY" -c "import site;print(1 if site.ENABLE_USER_SITE else 0)")
IN_VENV=$("$PY" -c "import sys;print(1 if sys.prefix != sys.base_prefix else 0)")
PIP_USER=""
VENV_DIR="$HOME/.venvs/kamp-exp"
USE_VENV=0

if [ "$FORCE_VENV" = "1" ] || { [ "$WRITABLE" = "0" ] && { [ "$USERSITE" = "0" ] || [ "$IN_VENV" = "1" ]; }; }; then
    USE_VENV=1
    # 공유 환경의 큰 패키지(TF 등)는 그대로 재사용하고 부족한 것만 venv 에 얹는다
    if [ ! -d "$VENV_DIR" ]; then
        if [ "$DRY" = "1" ]; then
            warn "(dry-run) 가상환경을 만들 예정: $(tilde "$VENV_DIR") (--system-site-packages)"
        else
            "$PY" -m venv --system-site-packages "$VENV_DIR" || { bad "venv 생성 실패"; exit 1; }
        fi
    fi
    if [ -x "$VENV_DIR/bin/python" ]; then
        PY="$VENV_DIR/bin/python"
    elif [ -x "$VENV_DIR/Scripts/python" ]; then
        PY="$VENV_DIR/Scripts/python"
    fi
    ok "가상환경 사용: $(tilde "$VENV_DIR") (base 패키지 상속)"
    echo "     run_all.py 를 돌리기 전에 매번: source ~/.venvs/kamp-exp/bin/activate"
elif [ "$WRITABLE" = "1" ]; then
    ok "현재 환경에 직접 설치 (쓰기 권한 있음)"
else
    PIP_USER="--user"
    ok "사용자 영역에 설치 (--user) — base 환경은 건드리지 않는다"
fi

# ──────────────────────────────────────────────────────────────────────
step 2/6 "지금 있는 패키지"
# 'pip이름 import이름 KAMP핀' — 핀이 '-' 면 버전을 지정하지 않는다
REQUIRED="numpy numpy 1.26.4
pandas pandas 2.1.4
scipy scipy 1.11.4
scikit-learn sklearn 1.4.2
matplotlib matplotlib 3.9.2
lightgbm lightgbm 4.5.0
optuna optuna 5.0.0
statsmodels statsmodels 0.14.2
shap shap 0.49.1
joblib joblib -"
OPTIONAL=""   # 넷째 칸 opt = 선택 패키지 — 필수와 따로 설치한다
[ "$WITH_TESTS" = "1" ] && OPTIONAL="${OPTIONAL}pytest pytest - opt
"
[ "$WITH_API" = "1" ] && OPTIONAL="${OPTIONAL}fastapi fastapi - opt
uvicorn uvicorn - opt
httpx httpx - opt
"

has_mod() { "$PY" -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('$1') else 1)" 2>/dev/null; }
pkg_ver() { "$PY" -c "import importlib.metadata as m
try: print(m.version('$1'))
except Exception: print('?')" 2>/dev/null; }

PINNED=""     # 필수: 핀을 붙인 설치 대상  (예: optuna==5.0.0)
UNPINNED=""   # 필수: 핀 없이 적은 설치 대상 (재시도용)
OPT_PKGS=""   # 선택(pytest·API): 필수 설치가 끝난 뒤 따로 설치한다 — 실패해도 optuna·shap 은 남는다
MISSING=""    # 지금 없는 모듈의 import 이름 (dry-run 점검에서 '설치 예정' 으로 보인다)
while read -r pipn modn pin kind; do
    [ -z "${pipn:-}" ] && continue
    if has_mod "$modn"; then
        ok "$(printf '%-13s %s (그대로 둔다)' "$pipn" "$(pkg_ver "$pipn")")"
    else
        warn "$(printf '%-13s 없음 — 설치 대상' "$pipn")"
        MISSING="$MISSING $modn"
        if [ "${kind:-}" = "opt" ]; then OPT_PKGS="$OPT_PKGS $pipn"; continue; fi
        if [ "$pin" = "-" ]; then PINNED="$PINNED $pipn"; else PINNED="$PINNED $pipn==$pin"; fi
        UNPINNED="$UNPINNED $pipn"
    fi
done <<EOF
$REQUIRED
$OPTIONAL
EOF
if has_mod tensorflow; then
    ok "$(printf '%-13s %s (그대로 둔다 · 절대 업그레이드하지 않는다)' tensorflow "$(pkg_ver tensorflow)")"
else
    warn "tensorflow    없음 — 설치하지 않는다. DNN·RNN 비교 모델만 건너뛰고 나머지는 그대로 돈다"
fi

# ──────────────────────────────────────────────────────────────────────
step 3/6 "기존 버전 고정(constraints)"
CONS=$(mktemp 2>/dev/null || echo "$HOME/.kamp_constraints.txt")
FIXED="numpy pandas scipy scikit-learn matplotlib lightgbm statsmodels tensorflow keras numba llvmlite joblib protobuf h5py ml-dtypes"
snapshot() {   # 'pip이름==버전' 줄 목록 (Windows 파이썬의 CR 은 지운다)
    "$PY" - $FIXED <<'PY' | tr -d '\r'
import sys
import importlib.metadata as md
for p in sys.argv[1:]:
    v = None
    for name in (p, p.replace("-", "_")):
        try:
            v = md.version(name)
            break
        except Exception:
            pass
    if v:
        print(f"{p}=={v}")
PY
}
snapshot > "$CONS"
BEFORE=$(cat "$CONS")
sed 's/^/     /' "$CONS"

# ──────────────────────────────────────────────────────────────────────
step 4/6 "없는 것만 설치"
PIP_DRY=""; PIP_RUN=1
if [ "$DRY" = "1" ]; then
    # pip 22.2 이상은 --dry-run 으로 설치 없이 의존성 풀이만 해 본다. 그보다 낮으면 명령만 보여 준다.
    if "$PY" -m pip install --help 2>/dev/null | grep -q -- "--dry-run"; then PIP_DRY="--dry-run"; else PIP_RUN=0; fi
fi
pip_install() {   # 인자: 공백으로 구분한 설치 대상 목록 한 개
    [ "$PIP_RUN" = "1" ] || { echo "     (dry-run) pip 가 --dry-run 을 몰라 실행하지 않았다"; return 0; }
    if [ -n "$PIP_DRY" ]; then
        # 풀이 결과('Would install …')와 오류 줄만 보여 준다 (설치 경로·재시도 경고 줄은 생략)
        # shellcheck disable=SC2086
        "$PY" -m pip install $PIP_DRY --disable-pip-version-check --no-cache-dir $PIP_USER -c "$CONS" $1 2>&1 \
            | awk '/^Would install|^ERROR/'
    else
        # shellcheck disable=SC2086
        "$PY" -m pip install --quiet --disable-pip-version-check --no-cache-dir $PIP_USER -c "$CONS" $1
    fi
}

done_msg() {   # dry-run 이면 설치하지 않았다고 알린다
    if [ "$DRY" = "1" ]; then ok "(dry-run) 설치하지 않았다"; else ok "$1"; fi
}
if [ -z "${PINNED// /}" ]; then
    ok "필수 패키지는 설치할 것이 없다"
else
    echo "  설치 대상:$PINNED"
    echo "  명령: python -m pip install ${PIP_DRY:+$PIP_DRY }--no-cache-dir ${PIP_USER:+$PIP_USER }-c <constraints>$PINNED"
    if pip_install "$PINNED"; then
        done_msg "완료 (기존 패키지는 그대로)"
    else
        warn "핀 버전으로는 풀리지 않았다 → 핀 없이 constraints 만 걸고 다시 시도"
        if pip_install "$UNPINNED"; then
            done_msg "완료 (핀 없이 설치 — 버전은 위 constraints 에 맞춰 pip 가 골랐다)"
        else
            fail "패키지 설치 실패 — 기존 버전과 충돌했거나 권한·네트워크 문제"
        fi
    fi
fi
# 선택 패키지는 따로 설치한다 — 이것이 실패해도 위 필수 설치(optuna·shap)는 되돌려지지 않는다
if [ -n "${OPT_PKGS// /}" ]; then
    echo "  선택 설치 대상:$OPT_PKGS"
    echo "  명령: python -m pip install ${PIP_DRY:+$PIP_DRY }--no-cache-dir ${PIP_USER:+$PIP_USER }-c <constraints>$OPT_PKGS"
    if pip_install "$OPT_PKGS"; then
        done_msg "선택 패키지 완료"
    else
        soft "선택 패키지 설치 실패:$OPT_PKGS — run_all.py 는 돈다(pytest 가 없으면 S.4 단위테스트만 건너뛴다)"
        echo "     네트워크·권한을 확인한 뒤 같은 명령(bash setup_pod.sh --with-tests 등)을 다시 실행한다"
    fi
fi

# ──────────────────────────────────────────────────────────────────────
step 5/6 "설치 후 점검"
AFTER=$(snapshot)
# 원래 있던 패키지의 버전이 그대로인지 한 줄씩 본다. 새로 생긴 패키지(예: shap 이 끌어온 numba)는 알림만 한다.
CHANGED=""
while read -r line; do
    [ -z "$line" ] && continue
    printf '%s\n' "$AFTER" | grep -qxF "$line" || CHANGED="$CHANGED ${line%%==*}"
done <<EOF
$BEFORE
EOF
ADDED=""
while read -r line; do
    [ -z "$line" ] && continue
    printf '%s\n' "$BEFORE" | grep -q "^${line%%==*}==" || ADDED="$ADDED $line"
done <<EOF
$AFTER
EOF
if [ -z "$CHANGED" ]; then
    ok "고정 패키지 버전 변화 없음"
else
    fail "고정 패키지 버전이 바뀌었다!:$CHANGED (공유 환경의 다른 사용자에게도 영향)"
    diff <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$AFTER") | sed 's/^/     /'
fi
[ -n "$ADDED" ] && echo "     새로 설치된 의존 패키지(정상):$ADDED"
rm -f "$CONS" 2>/dev/null

PLANNED=""; [ "$DRY" = "1" ] && PLANNED="$MISSING"   # dry-run: 아직 설치하지 않은 것은 실패로 세지 않는다
# shellcheck disable=SC2086
"$PY" - $PLANNED <<'PY'
import importlib, sys
need = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "lightgbm",
        "optuna", "statsmodels", "shap", "joblib"]
planned = set(sys.argv[1:])
miss = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"  ✔ import {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        if m in planned:
            print(f"  - import {m:12s} (dry-run) 설치 예정")
            continue
        miss.append(m)
        print(f"  ✘ import {m:12s} {type(e).__name__}")
sys.exit(1 if miss else 0)
PY
[ $? -eq 0 ] || fail "필수 패키지 import 실패 (위 ✘ 참고)"

LEFT=$("$PY" -c "
import site, sysconfig, pathlib
ds = {sysconfig.get_paths()['purelib'], site.getusersitepackages()}
print(' '.join(p.name for d in ds if pathlib.Path(d).is_dir() for p in pathlib.Path(d).glob('~*')))" 2>/dev/null)
if [ -n "${LEFT// /}" ]; then
    warn "이전 설치 실패의 잔해가 있다: $LEFT"
    echo "     numpy·pandas 가 정상 import 되는지 확인한 뒤 site-packages 안의 그 폴더를 지우면 된다."
else
    ok "잔해(~ 로 시작하는 폴더) 없음"
fi

# ──────────────────────────────────────────────────────────────────────
step 6/6 "한글 폰트 (s00 가 찾는 정확한 이름 4개)"
font_ok() {   # matplotlib 이 없으면 조용히 '없음'
    "$PY" - 2>/dev/null <<'PY'
import sys
from matplotlib import font_manager
names = {f.name for f in font_manager.fontManager.ttflist}
for n in ("Malgun Gothic", "NanumGothic", "Noto Sans KR", "AppleGothic"):
    if n in names:
        print(n)
        sys.exit(0)
sys.exit(1)
PY
}
clear_mpl_cache() {
    CACHE=$("$PY" -c "import matplotlib;print(matplotlib.get_cachedir())" 2>/dev/null)
    if [ -n "$CACHE" ] && [ -d "$CACHE" ]; then
        if [ "$DRY" = "1" ]; then
            echo "     (dry-run) matplotlib 폰트 캐시를 지울 예정: $(tilde "$CACHE")"
        else
            rm -f "$CACHE"/fontlist-*.json 2>/dev/null || true
        fi
    fi
}

if FOUND=$(font_ok); then
    ok "한글 폰트 있음: $FOUND"
else
    warn "s00 가 찾는 이름이 없다 → 캐시를 비우고 다시 확인한다"
    clear_mpl_cache
    if [ "$DRY" = "0" ] && FOUND=$(font_ok); then
        ok "캐시를 비우니 잡혔다: $FOUND"
    elif [ "$DRY" = "1" ]; then
        echo "     (dry-run) 그래도 없으면 NanumGothic 을 ~/.fonts 에 받을 예정"
    else
        warn "홈에 NanumGothic 을 내려받는다 (~/.fonts)"
        mkdir -p "$HOME/.fonts"
        URL="https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        DEST="$HOME/.fonts/NanumGothic.ttf"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL "$URL" -o "$DEST" 2>/dev/null
        else
            wget -q "$URL" -O "$DEST" 2>/dev/null
        fi
        # 받은 것이 진짜 폰트인지 확인 — 404 HTML 을 폰트로 착각하면 조용히 깨진다
        if [ -s "$DEST" ] && [ "$(wc -c < "$DEST")" -gt 50000 ]; then
            command -v fc-cache >/dev/null 2>&1 && fc-cache -f "$HOME/.fonts" >/dev/null 2>&1
            clear_mpl_cache
            # 새 프로세스에서 다시 확인 (matplotlib 은 캐시를 프로세스 시작 때 읽는다)
            if FOUND=$(font_ok); then
                ok "홈에 폰트 설치 후 확인: $FOUND"
            else
                rm -f "$DEST"
                soft "받은 폰트를 matplotlib 이 인식하지 못했다 (그림 한글만 깨지고 코드는 돈다)"
            fi
        else
            rm -f "$DEST" 2>/dev/null
            soft "폰트를 받지 못했다 (그림 한글만 □□□ 로 깨지고 코드는 돈다)"
        fi
        if [ -z "${FOUND:-}" ]; then
            echo "     손으로: NanumGothic .ttf 를 ~/.fonts 에 두고  rm -f ~/.cache/matplotlib/fontlist-*.json  뒤 다시 실행"
        fi
    fi
fi

# ──────────────────────────────────────────────────────────────────────
if [ -n "$SOFT" ]; then
    printf '\n\033[33m주의 (코드 실행은 된다):\033[0m%s\n' "$SOFT"
fi
if [ -n "$FAILS" ] && [ "$DRY" = "1" ]; then
    printf '\n\033[31m(dry-run) 실제로 실행하면 막힐 수 있는 것:\033[0m%s\n' "$FAILS"
    printf '(dry-run 끝 — 아무것도 설치하지 않았다)\n'
    exit 1
fi
if [ -n "$FAILS" ]; then
    printf '\n\033[31m준비가 끝나지 않았다:\033[0m%s\n' "$FAILS"
    cat <<'EOS'

── 수동 3줄 (자동 설치가 막혔을 때 — KAMP 에서 실제로 통한 방법) ─────────
 1) numpy 가 멀쩡한지 확인 (위에 '잔해' 가 나왔다면 그 폴더를 지운다)
      python -c "import numpy, pandas; print(numpy.__version__, pandas.__version__)"
 2) 지금 버전을 constraints 로 적는다
      python -m pip list --format=freeze | grep -iE '^(numpy|pandas|scipy|scikit-learn|matplotlib|lightgbm|statsmodels|tensorflow|keras|joblib)==' > ~/kamp_constraints.txt
 3) 없는 것만 설치한다 (Permission denied 가 나면 --user 를 붙인다)
      python -m pip install --no-cache-dir -c ~/kamp_constraints.txt optuna shap
 -c 가 핵심이다. 이것이 shap 을 numpy 에 맞는 버전으로 스스로 내려가게 해서 numpy·TensorFlow 를 지킨다.
 폐쇄망이라 pip 가 안 되면: 관리자에게 optuna·shap 설치를 요청한다.
──────────────────────────────────────────────────────────────────────────
EOS
    exit 1
fi

if [ "$DRY" = "1" ]; then
    printf '\n(dry-run 끝 — 실제로 하려면 --dry-run 을 빼고 다시 실행)\n'
    exit 0
fi
printf '\n준비 끝. 다음:\n'
[ "$USE_VENV" = "1" ] && echo "  source ~/.venvs/kamp-exp/bin/activate"
echo "  python run_all.py --list     (계획만 보기)"
echo "  python run_all.py            (FULL — 오래 걸리면 README 의 nohup 명령)"
exit 0
