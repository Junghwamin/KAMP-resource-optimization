#!/usr/bin/env bash
# 실행 환경 진단 — 읽기만 한다. 아무것도 설치하지 않고 패키지 파일도 고치지 않는다.
# (matplotlib 을 처음 import 하면 matplotlib 이 자기 폰트 캐시를 만드는 것만 예외)
#
#   bash check_env.sh
#
# 끝에 '종합 판정'이 나온다. 필수 항목이 빠졌으면 종료코드 1, 아니면 0.
# 계정 이름·호스트 이름은 출력하지 않는다. 경로는 홈을 '~' 로 가려서 보여 준다.
set -uo pipefail
cd "$(dirname "$0")" 2>/dev/null || true

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg TF_CPP_MIN_LOG_LEVEL=3
unset PYTHONSAFEPATH 2>/dev/null || true

PY=""
for c in python python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done

FAIL=""      # 필수 항목 실패 목록 (공백 구분)
NOTE=""      # 알림 목록 (실패는 아님)

sec() { printf '\n════════ %s ════════\n' "$1"; }

# 파이썬 블록 공통 머리말: 경로의 홈 부분을 가리는 mask()
PY_PRE='
import os, re, sys
from pathlib import Path
_H = str(Path.home())
def mask(s):
    s = str(s)
    for h in {_H, _H.replace("\\", "/")}:
        if len(h) > 3:
            s = s.replace(h, "~")
    s = re.sub(r"(?i)[A-Za-z]:[\\/]+users[\\/]+[^\\/\s\"]+", "<홈>", s)
    s = re.sub(r"(?i)/[a-z]/users/[^/\s\"]+", "<홈>", s)
    s = re.sub(r"/(home|Users)/[^/\s\"]+", r"/\1/<사용자>", s)
    return s
'
py() { { printf '%s\n' "$PY_PRE"; cat; } | "$PY" - ; }

sec "1. 기본"
case "$PWD/" in
    "$HOME"/*) echo "패키지 폴더 : 홈 아래 · 폴더 이름 $(basename "$PWD")" ;;
    *)         echo "패키지 폴더 : 홈 밖 · 폴더 이름 $(basename "$PWD")" ;;
esac
if [ -z "$PY" ]; then
    echo "  ✘ python 명령이 없다"
    echo
    echo "종합 판정: ✘ python 이 없어 진단을 계속할 수 없다"
    exit 1
fi
py <<'PY'
import platform
v = sys.version_info
ok = v >= (3, 10)
print(f"python      : {platform.python_version()}  {'✔' if ok else '✘ 3.10 이상 필요'}")
print(f"실행 파일   : {mask(sys.executable)}")
print(f"OS          : {platform.system()} {platform.machine()}")
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || FAIL="$FAIL python3.10이상"
PIPV=$("$PY" -m pip --version 2>/dev/null | awk '{print $1" "$2}')
echo "pip         : ${PIPV:-없음}"
[ -n "$PIPV" ] || NOTE="$NOTE pip없음"
echo "conda 환경  : ${CONDA_DEFAULT_ENV:-없음}"

sec "2. 설치 위치와 권한"
py <<'PY'
import site, sysconfig
sp = sysconfig.get_paths()["purelib"]
w = os.access(sp, os.W_OK)
u = bool(site.ENABLE_USER_SITE)
venv = sys.prefix != sys.base_prefix
print(f"site-packages : {mask(sp)}")
print(f"  쓰기 가능   : {'예' if w else '아니오'}")
print(f"user site     : {mask(site.getusersitepackages())}  (사용 가능: {'예' if u else '아니오'})")
print(f"가상환경 안   : {'예' if venv else '아니오'}")
if w:
    how = "현재 환경에 직접"
elif u:
    how = "사용자 영역(--user)"
else:
    how = "홈의 가상환경(--venv)"
print(f"→ setup_pod.sh 가 고를 설치 방식: {how}")
PY
if [ -w "$HOME" ]; then echo "홈 쓰기     : 가능"; else echo "홈 쓰기     : 불가"; fi

sec "3. 자원"
AVAIL_K=$(df -Pk . 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "${AVAIL_K:-}" ]; then
    echo "디스크 여유 : $((AVAIL_K / 1024 / 1024)) GB (이 폴더가 있는 곳, 실행에 약 1GB 필요)"
else
    echo "디스크 여유 : 확인 불가"
fi
if [ -r /proc/meminfo ]; then
    MT=$(awk '/^MemTotal:/ {print int($2/1024/1024)}' /proc/meminfo)
    MA=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
    echo "메모리      : 전체 ${MT:-?} GB · 사용 가능 ${MA:-?} GB"
else
    echo "메모리      : 확인 불가(/proc/meminfo 없음)"
fi
if command -v nproc >/dev/null 2>&1; then
    echo "CPU 코어    : $(nproc)"
else
    echo "CPU 코어    : $("$PY" -c 'import os; print(os.cpu_count())')"
fi

sec "4. 패키지"
py <<'PY'
import importlib.metadata as md
import importlib.util as iu
# (pip 이름, import 이름, KAMP 기준 버전)
REQ = [("numpy", "numpy", "1.26.4"), ("pandas", "pandas", "2.1.4"), ("scipy", "scipy", "1.11.4"),
       ("scikit-learn", "sklearn", "1.4.2"), ("matplotlib", "matplotlib", "3.9.2"),
       ("lightgbm", "lightgbm", "4.5.0"), ("optuna", "optuna", "5.0.0"),
       ("statsmodels", "statsmodels", "0.14.2"), ("shap", "shap", "0.49.1"), ("joblib", "joblib", "")]
COND = [("tensorflow", "tensorflow", "2.17.0"), ("keras", "keras", "")]
OPT = [("pytest", "pytest", ""), ("fastapi", "fastapi", ""), ("uvicorn", "uvicorn", ""), ("httpx", "httpx", "")]

def ver(pip_name):
    try:
        return md.version(pip_name)
    except Exception:
        return "?"

def show(title, rows, note_missing):
    print(title)
    missing = []
    for p, m, ref in rows:
        if iu.find_spec(m) is None:
            missing.append(p)
            print(f"  ✘ {p:13s} 없음 — {note_missing}")
            continue
        v = ver(p)
        mark = ""
        if ref and v != ref:
            mark = f"   (KAMP 기준 {ref} 와 다름)"
        print(f"  ✔ {p:13s} {v}{mark}")
    return missing

miss = show("[필수] 없으면 run_all.py 가 시작하지 않는다", REQ, "bash setup_pod.sh 로 설치")
show("[조건] 없으면 DNN·RNN 비교 모델만 건너뛴다", COND, "설치하지 말 것(GPU 연동이 깨질 수 있다)")
show("[선택] 단위테스트(pytest)·REST API(fastapi 등)", OPT, "필요할 때 setup_pod.sh --with-tests / --with-api")
if iu.find_spec("optuna") is not None and ver("optuna") != "5.0.0":
    print(f"\n  알림: optuna {ver('optuna')} — 검증한 버전은 5.0.0 이다. 탐색 결과(hpo_trials)가 달라질 수 있다.")
print("\n  numpy·pandas·lightgbm 버전은 모델 번들에 기록된다. 번들은 같은 버전 환경에서만 열린다.")
sys.exit(1 if miss else 0)
PY
[ $? -eq 0 ] || FAIL="$FAIL 필수패키지"

sec "5. 이전 설치 실패의 잔해(~ 로 시작하는 폴더)"
py <<'PY'
import site, sysconfig
dirs = {sysconfig.get_paths()["purelib"], site.getusersitepackages()}
left = []
for d in dirs:
    p = Path(d)
    if p.is_dir():
        left += [mask(x) for x in p.glob("~*")]
if left:
    print("  ! 잔해가 있다(pip 가 교체 도중 멈춘 흔적). 'Ignoring invalid distribution' 경고의 원인:")
    for x in left:
        print(f"     {x}")
    print("    numpy·pandas 가 정상 import 되는지 확인한 뒤 위 폴더를 지우면 된다.")
else:
    print("  ✔ 없음")
PY

sec "6. 한글 폰트 (s00 규칙)"
py <<'PY'
try:
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
except Exception as e:
    print(f"  matplotlib 확인 불가: {type(e).__name__}")
    sys.exit(0)
# s00_env.py 의 setup_korean_font() 와 같은 순서·같은 정확한 이름
WANT = ("Malgun Gothic", "NanumGothic", "Noto Sans KR", "AppleGothic")
fonts = {}
for f in font_manager.fontManager.ttflist:
    fonts.setdefault(f.name, f.fname)
chosen = next((n for n in WANT if n in fonts), None)
if chosen:
    try:
        has_ga = 0xAC00 in FT2Font(fonts[chosen]).get_charmap()
    except Exception:
        has_ga = None
    g = {True: "'가' 글리프 있음", False: "'가' 글리프 없음 — 그림 한글이 깨질 수 있다", None: "글리프 확인 불가"}[has_ga]
    print(f"  ✔ s00 가 쓸 폰트: {chosen}  ({g})")
else:
    print("  ✘ s00 가 찾는 이름(Malgun Gothic / NanumGothic / Noto Sans KR / AppleGothic)이 없다")
    print("    코드는 멈추지 않지만 그림의 한글이 □□□ 로 깨진다 → bash setup_pod.sh 가 NanumGothic 을 받아 준다")
keys = ("nanum", "cjk", " kr", "malgun", "gulim", "batang", "dotum", "gungsuh", "apple sd", "applegothic")
near = sorted(n for n in fonts if n not in WANT and any(k in n.lower() for k in keys))
if near:
    print("  근접 이름(s00 가 쓰지 않음): " + ", ".join(near[:8]) + (" …" if len(near) > 8 else ""))
PY
if command -v fc-list >/dev/null 2>&1; then
    echo "  fc-list 한국어 폰트 수: $(fc-list :lang=ko 2>/dev/null | wc -l)"
else
    echo "  fc-list: 없음"
fi

sec "7. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
        | sed 's/^/  /' || echo "  nvidia-smi 실행 실패"
else
    echo "  nvidia-smi: 없음"
fi
TFOUT=$("$PY" -c "
import tensorflow as tf
g = tf.config.list_physical_devices('GPU')
print('tensorflow', tf.__version__, '· GPU', len(g), '개')
" 2>/dev/null | tail -1)
echo "  ${TFOUT:-tensorflow: 없음 또는 import 실패 (DNN·RNN 비교 모델만 건너뛴다)}"
echo "  GPU 는 DNN·RNN 비교 모델에만 쓰인다. 없으면 CPU 로 돌고 결과 판단에는 영향이 없다."

sec "8. 데이터"
py <<'PY'
import hashlib
p = Path("data") / "okm_augumented_2021.csv"
WANT = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
if not p.is_file():
    print("  ✘ data/okm_augumented_2021.csv 없음 — 패키지 폴더 안에서 실행했는지 확인")
    sys.exit(1)
b = p.read_bytes()
h = hashlib.sha256(b).hexdigest()
print(f"  크기 {len(b):,} B")
if h == WANT:
    print("  ✔ sha256 일치 (원본 그대로)")
    sys.exit(0)
print(f"  ! sha256 불일치: {h[:16]}…  (기대 {WANT[:16]}…)")
print("    엑셀 등으로 다시 저장하면 바이트가 바뀐다. 원본 파일로 교체할 것 — 결과가 보고서와 달라진다.")
sys.exit(2)
PY
RC=$?
[ $RC -eq 1 ] && FAIL="$FAIL 데이터없음"
[ $RC -eq 2 ] && NOTE="$NOTE 데이터sha불일치"

sec "9. 이름 가려짐 (tools · serving)"
py <<'PY'
import importlib.util as iu
root = Path.cwd().resolve()
if str(root) not in sys.path and "" not in sys.path:
    sys.path.insert(0, str(root))
bad = 0
for name in ("tools", "serving"):
    spec = iu.find_spec(name)
    want = root / name / "__init__.py"
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    if origin == want.resolve():
        print(f"  ✔ {name:8s} → ./{name}/__init__.py (이 패키지 것이 쓰인다)")
    elif spec is None:
        print(f"  ✘ {name:8s} 를 찾지 못했다 — 패키지 폴더 안에서 실행했는지 확인")
        bad = 1
    else:
        print(f"  ✘ {name:8s} → {mask(spec.origin)} (다른 곳의 같은 이름이 먼저 잡힌다)")
        bad = 1
    # 다른 경로에 같은 이름이 있는지(있어도 위가 ✔ 이면 문제없다)
    others = []
    for d in sys.path:
        if not d or Path(d).resolve() == root:
            continue
        q = Path(d) / name
        if (q / "__init__.py").is_file() or q.with_suffix(".py").is_file():
            others.append(mask(q))
    if others:
        print(f"    참고: 다른 경로에도 '{name}' 이 있다: {', '.join(others[:3])}")
sys.exit(bad)
PY
[ $? -eq 0 ] || NOTE="$NOTE 이름가려짐"

sec "10. 네트워크 (패키지 저장소)"
TMPD=$(mktemp -d 2>/dev/null || echo "")
if [ -n "$TMPD" ]; then
    if command -v timeout >/dev/null 2>&1; then TO="timeout 40"; else TO=""; fi
    if $TO "$PY" -m pip download --no-deps --no-cache-dir -q -d "$TMPD" "optuna==5.0.0" >/dev/null 2>&1; then
        echo "  ✔ 접근 가능 (optuna 5.0.0 을 받을 수 있다)"
    else
        echo "  ! 접근 실패 — 폐쇄망이거나 저장소 설정 문제. setup_pod.sh 가 설치하지 못할 수 있다"
        NOTE="$NOTE 패키지저장소접근실패"
    fi
    rm -rf "$TMPD" 2>/dev/null || true
else
    echo "  확인 불가(임시 폴더를 만들지 못함)"
fi

sec "종합 판정"
if [ -z "$FAIL" ]; then
    echo "  ✔ 필수 항목 통과 — 다음: bash setup_pod.sh --with-tests  (이미 다 있으면 설치 없이 끝난다)"
    [ -n "$NOTE" ] && echo "  알림:$NOTE"
    exit 0
else
    echo "  ✘ 부족한 항목:$FAIL"
    [ -n "$NOTE" ] && echo "  알림:$NOTE"
    echo "  → bash setup_pod.sh 로 설치한 뒤 이 진단을 다시 실행한다."
    exit 1
fi
