#!/usr/bin/env python3
"""KAMP 터미널 실행기 — 노트북 대신 `src/s00~s10` 을 노트북과 **같은 방식**으로 실행하고 기록한다.

    python run_all.py                    # FULL(보고서 수치) · 멈춤 없음
    python run_all.py --fast             # FAST 동작 확인(수치는 최종값 아님) — 복사본에서
    python run_all.py --pause section    # 단계마다 Enter 대기 (chapter = 장이 끝날 때마다)
    python run_all.py --check-only       # 파이프라인 없이 기존 outputs 로 서빙·완료 점검만
    python run_all.py --list             # 실행 계획만 출력 (실행·쓰기 없음)
    python run_all.py --pack-results     # 결과 zip (기본 ~/kamp_results_full.zip · FAST 결과면 ~/kamp_results_fast.zip)

실행 방식: src 를 텍스트로 파싱해 `# %%` 셀로 나누고(nb-strip 셀은 버린다 = 노트북 빌드와 같다),
한 namespace 에서 셀을 차례로 exec 한다. 마지막 식의 값은 `Out:` 으로 찍고, 셀마다 그림을 닫는다.
보고서 생성 절(6.7 · 8.5 · 9장 · 10.2 · 10.3 · 10.4 · 10.6 · 최종 게이트)을 뺀 57단계(68셀) 뒤에
서빙 점검 S.1~S.4 와 완료 점검 C 를 한다. 단계 로그와 index.json 은 outputs/steps/ 에 남는다.

종료코드: 0 성공 · 1 단계 실패 · 2 사용법/사전점검 · 3 완료 점검(hard) 실패 · 130 중단.
최상위 import 는 표준 라이브러리뿐이다(numpy·pandas·serving 은 자식 프로세스의 함수 안에서만 쓴다).
"""
from __future__ import annotations

import argparse
import ast
import builtins
import datetime as _dt
import difflib
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import site
import subprocess
import sys
import threading
import time
import traceback
import types
import zipfile
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
STEPS_DIR = ROOT / "outputs" / "steps"
INDEX_PATH = STEPS_DIR / "index.json"
LOCK_PATH = STEPS_DIR / ".run_all.lock"
DATA_REL = "data/okm_augumented_2021.csv"
PRED_REL = "outputs/predictions_test_336h.csv"

# ── PLAN.md 에서 검증된 값 — src 가 바뀌면 단언이 실패한다 ─────────────────────
MARK_CODE, MARK_MD, MARK_STRIP = "# %%", "# %% [markdown]", '# %% tags=["nb-strip"]'
N_CODE_CELLS, N_EXCLUDED_CELLS, N_PLAN_CELLS, N_PLAN_STEPS = 77, 9, 68, 57
EXPECTED_STEPS = """0.1 0.2 0.3 0.4 0.5 0.6 1.1 1.2:4 1.3:2 1.4:2 1.5:2 1.6:2 1.G 2.1 2.2 2.3:2 2.4
3.1 3.2:2 3.3 3.4 3.5 4.1 4.2 4.3 4.4 4.5 4.G 5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8
6.1:2 6.2 6.3 6.4 6.5 6.G 6.6 7.0 7.1 7.2 7.3 7.4 7.5 7.6 8.1 8.2 8.3 8.4 10.1 10.5:2""".split()
# 뺀 절 = 보고서·제출물 생성 부분(실험 결과에는 영향 없음). 키가 src 에 없으면 즉시 실패한다.
EXCLUDE = {
    "s06_eval": {"6.7": "보고서 서술 교정 — 10.2절 문장 치환사전 입력용"},
    "s08_simulation": {"8.5": "보고서 4장 본문 초안(md) 생성"},
    "s09_creativity": {"*": "창의성·차별성 근거 정리 + 보고서 5장 초안 — 실험이 아니라 보고서 재료"},
    "s10_package": {
        "10.2": "보고서 채움표 · 문장 치환사전 · 장별 PASS/FAIL",
        "10.3": "report_tbd_filled.md 생성",
        "10.4": "requirements.txt · README(6장 본문) 생성",
        "10.6": "발표자료 골격(pptx) · 개인정보 스캔",
        "최종 게이트 — 제출 준비 상태 점검": "제출 준비 점검",
    },
}
REPORT_OVERRIDE = {
    "0.5": ["6"], "0.6": ["6", "2.2"], "1.G": ["1.4"], "4.G": ["2.2"], "6.G": ["2.9"],
    "5.1": ["2.5", "2.4"], "7.0": ["3"], "10.5": ["6", "4"],
    "S.1": ["6"], "S.2": ["6"], "S.4": ["6"], "C": ["6"], "S.3": ["4", "6"],
}
# 개발 PC FULL 소요시간(초, 진행 표시용). 나머지 단계는 10초 미만
DEV_SEC = {"4.2": 149, "4.5": 26, "5.1": 77, "5.4": 72, "5.5": 271, "6.3": 60, "6.4": 48,
           "6.5": 122, "6.6": 1108, "7.1": 25, "10.5": 17}
EXTRA_STEPS = [  # (단계, 종류, 제목, 근거 파일)
    ("S.1", "serving", "S.1 번들 검증 — serving verify (eval · deploy)", "serving/cli.py"),
    ("S.2", "serving", "S.2 리플레이 — 평가 번들로 09-01~09-14 재생, 테스트 예측과 비트 비교", "serving/cli.py"),
    ("S.3", "serving", "S.3 CLI 익일 예측(09-14) · 누수 차단", "serving/cli.py"),
    ("S.4", "serving", "S.4 단위·회귀테스트 (pytest)", "tests"),
    ("C", "check", "C 완료 점검 — 예측 · 평가표 · 그림 · 번들 · 서빙", "run_all.py"),
]
PIPELINE_KINDS = ("section", "gate", "intro")
EXPECTED_TABLES = [f"F{i:02d}_src" for i in range(1, 40)] + """
ch1_calendar ch1_eda_dow ch1_eda_hourly ch1_eda_monthly_peak ch1_eda_night_peaks ch1_eda_peak_by_hour
ch1_holiday_audit ch1_long_shutdown ch1_profile_dup ch1_quality ch1_repair_log ch1_rowlayout ch1_theta
ch1_variable_dict ch2_ablation ch2_d1_vs_d2 ch2_feature_groups ch2_feature_labels ch2_gate_diagnosis
ch2_lag_d2 ch2_lag_decomposition ch2_lag_variants ch2_leakage_blind ch2_leakage_obs ch2_lofo
ch2_peak_detection ch2_peak_detection_oof ch2_regression ch2_scorecard ch2_significance ch3_calendar_rule
ch3_condition_mae ch3_confusion ch3_contamination ch3_data_conditions ch3_failures ch3_folds
ch3_importance ch3_interaction ch3_permutation ch3_rules ch4_corrected ch4_defects ch4_levers ch4_naive
ch4_priority ch4_protocol ch4_rf_original ch4_scenarios ch4_sensitivity ch4_silent_noop ch4_tariff
ch5_calibration ch5_ensemble ch5_fold_mae_matrix ch5_uncertainty ch6_model_bundles ch6_rank_preservation
env_versions fold_matrix gate1_chapter1 gate2_leakage gate3_split gate4_baseline gate5_eval hpo_trials
timing ch2_peak_threshold_temporal ch2_plan_information_ablation
ch5_calibration_folds ch5_calibration_forward_predictions""".split()
PRED_COLUMNS = ["datetime", "y_avg_true", "y_avg_pred", "y_peak_true", "y_peak_pred",
                "peak_prob", "peak_pred_label", "peak_true_label", "model_name"]
DATA_SHA256 = "8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830"
SERVICE_MODEL, CLF_MODEL = "2단계 레짐(3분류)", "피크 직접분류"
REQUIRED = ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "lightgbm", "optuna",
            "statsmodels", "shap", "joblib")
VERSION_DISTS = [("numpy", ("numpy",)), ("pandas", ("pandas",)), ("scipy", ("scipy",)),
                 ("scikit-learn", ("scikit-learn",)), ("matplotlib", ("matplotlib",)),
                 ("lightgbm", ("lightgbm",)), ("optuna", ("optuna",)), ("statsmodels", ("statsmodels",)),
                 ("shap", ("shap",)), ("joblib", ("joblib",)),
                 ("tensorflow", ("tensorflow", "tensorflow-cpu", "tensorflow-intel", "tensorflow-gpu")),
                 ("keras", ("keras",)), ("pytest", ("pytest",))]
PROV_FILES = ("run_all.py", "show_results.py", "check_env.sh", "setup_pod.sh", "pytest.ini", "requirements.txt")
PROV_GLOBS = ("src/*.py", "tools/*.py", "serving/**/*.py", "serving/config/*.json", "tests/*.py", "data/*.csv")
CHILD_ENV = {"PYTHONHASHSEED": "42", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
             "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "TF_ENABLE_ONEDNN_OPTS": "0",
             "TF_DETERMINISTIC_OPS": "1", "TF_CPP_MIN_LOG_LEVEL": "3", "KAMP_RUNALL_CHILD": "1"}

RE_CHAPTER = re.compile(r"^##\s+(\d+)\.\s*(.*?)\s*$")
RE_SECTION = re.compile(r"^###\s+(.*?)\s*$")
RE_SEC_NUM = re.compile(r"^(\d+(?:\.\d+)*)\s+")       # 절 키 — 제목 앞 번호(없으면 제목 전체)
RE_TAG = re.compile(r"^-\s*\*\*보고서 대응절\*\*\s*[:：]\s*(.*)$")
RE_REF = re.compile(r"(\d+)\.(\d+)(?:\s*~\s*\d+\.(\d+))?|(\d+)\s*장|전\s*장")


# ══════════════════════════════════════════════════════════════════════
# 공용 도우미
# ══════════════════════════════════════════════════════════════════════
def _redaction_rules() -> list:
    """로그에 쓸 때 가릴 것: 패키지 경로 → `.`, 홈 → `~`, 사용자 경로·pytest 임시폴더의 계정명.

    파이썬 설치·임시 폴더도 가린다 — Windows 는 이 폴더들이 홈 아래에 있어 서드파티 경고 경로에 그대로 박힌다.
    아래 정규식은 `[/]`·`[-]` 로 적는다 — 패키지 스캐너가 이 규칙 문자열 자체를 경로로 잡지 않게 한다.
    """
    flags = re.IGNORECASE if os.name == "nt" else 0
    bases = [(ROOT, "."), (Path(sys.prefix), "<python>"), (Path(sys.base_prefix), "<python>"), (Path.home(), "~")]
    tmp = os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP")
    if tmp:
        bases.append((Path(tmp), "<tmp>"))
    try:
        bases.append((Path(site.getusersitepackages()), "<user-site>"))
    except Exception:
        pass
    lits = []
    for base, repl in bases:
        s = str(base)
        for v in {s, base.as_posix(), s.replace("\\", "\\\\")}:   # 원형 · 슬래시형 · repr 의 겹역슬래시
            if len(v) > 3 and v not in ("/usr", "/tmp", "/var", "/opt"):   # 짧은 공용 경로는 그대로 둔다
                lits.append((len(v), re.compile(re.escape(v), flags), repl))
    rules = [(rx, repl) for _, rx, repl in sorted(lits, key=lambda t: -t[0])]
    return rules + [
        (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", re.IGNORECASE), "<홈>"),
        (re.compile(r"[/]home[/][^/\s\"']+"), "/home/<사용자>"),
        (re.compile(r"[/]Users[/][^/\s\"']+"), "/Users/<사용자>"),
        (re.compile(r"pytest[-]of[-][^/\s]+"), "pytest-of-<사용자>"),
    ]


_RULES = _redaction_rules()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def redact(text: str) -> str:
    """ANSI 제거 · CRLF→LF · 경로 가림. 로그·증거 파일·index 에 쓰는 모든 텍스트에 적용한다."""
    text = _ANSI.sub("", str(text)).replace("\r\n", "\n")
    for rx, repl in _RULES:
        text = rx.sub(repl, text)
    return text


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def fmt_sec(x) -> str:
    x = float(x or 0)
    return f"{x:.1f}초" if x < 90 else f"{int(x // 60)}분 {int(x % 60):02d}초"


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lf_sha(p: Path) -> str:
    """줄바꿈을 LF 로 맞춘 sha256 — pandas CSV 줄바꿈이 OS 따라 달라도 같은 값이 나온다."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def public(d: dict) -> dict:
    """내부 키(`_` 로 시작)를 뺀 사본 — index.json 에 쓰는 모양."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _jsonable(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if o is None or isinstance(o, (str, int, bool)):
        return o
    try:                      # numpy 스칼라
        return _jsonable(o.item())
    except Exception:
        return str(o)


def read_index():
    try:
        d = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("schema") == 1 else None


def versions() -> dict:
    """설치 버전(패키지를 import 하지 않고 메타데이터로 읽는다). 없으면 None."""
    out = {"python": platform.python_version()}
    for name, dists in VERSION_DISTS:
        out[name] = None
        for d in dists:
            try:
                out[name] = metadata.version(d)
                break
            except metadata.PackageNotFoundError:
                continue
    return out


def provenance() -> tuple:
    """패키지 코드·설정·데이터 파일의 sha256 (README·outputs 제외) → (index 용 dict, `sha  path` 목록)."""
    files = {}
    for p in [ROOT / f for f in PROV_FILES] + [p for g in PROV_GLOBS for p in sorted(ROOT.glob(g))]:
        if p.is_file() and "__pycache__" not in p.parts:
            files[p.relative_to(ROOT).as_posix()] = file_sha(p)
    text = "".join(sorted(f"{h}  {r}\n" for r, h in files.items()))
    return {"files": dict(sorted(files.items())),
            "code_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}, text


# ══════════════════════════════════════════════════════════════════════
# 실행 계획 — src 를 텍스트로 파싱 (jupytext 불필요)
# ══════════════════════════════════════════════════════════════════════
class PlanError(Exception):
    """계획이 PLAN.md 의 검증값과 다르다 — 종료코드 2."""


def read_cells(path: Path) -> list:
    """percent 파일 → 셀 목록. 마커는 세 가지뿐, 셀 끝 빈 줄과 빈 셀·nb-strip 셀은 버린다."""
    lines = path.read_text(encoding="utf-8").splitlines()
    marks = [i for i, s in enumerate(lines, 1) if s.startswith("# %%")]
    cells = []
    for j, m in enumerate(marks):
        nxt = marks[j + 1] if j + 1 < len(marks) else len(lines) + 1
        head = lines[m - 1].rstrip()
        if head not in (MARK_CODE, MARK_MD, MARK_STRIP):
            raise PlanError(f"알 수 없는 셀 마커: src/{path.name}:{m}  {head}")
        body = lines[m:nxt - 1]
        while body and not body[-1].strip():
            body.pop()
        if head == MARK_STRIP or not any(s.strip() for s in body):
            continue
        if head == MARK_MD:   # 마크다운은 줄마다 `# ` 접두
            body = [s[2:] if s.startswith("# ") else ("" if s == "#" else s) for s in body]
        cells.append({"type": "markdown" if head == MARK_MD else "code", "mark": m, "end": nxt - 1,
                      "text": "\n".join(body)})
    return cells


def report_refs(tag: str) -> list:
    """`보고서 대응절` 텍스트 → 보고서 절·장 목록(첫 값이 대표). 괄호·백틱 안은 무시한다."""
    text = re.sub(r"\([^)]*\)|`[^`]*`", " ", tag or "")
    refs = []
    for m in RE_REF.finditer(text):
        if m.group(1):
            a, b = int(m.group(2)), int(m.group(3) or m.group(2))
            refs += [f"{m.group(1)}.{k}" for k in range(a, max(a, b) + 1)]
        else:
            refs.append(m.group(4) or "*")
    return list(dict.fromkeys(refs))


def _new_step(sid, kind, chapter, title, file, tag) -> dict:
    return {"seq": 0, "id": sid, "kind": kind, "chapter": chapter, "title": title, "file": file,
            "lines": None, "cells": 0, "report": list(REPORT_OVERRIDE.get(sid) or report_refs(tag)),
            "tag_raw": tag, "status": "pending", "seconds": 0.0, "log": None, "log_lines": 0,
            "files": {"tables": [], "figures": [], "models": [], "other": []},
            "dev_sec": DEV_SEC.get(sid), "error": None, "_cells": []}


def build_plan() -> tuple:
    """src → (단계 목록, 제외 목록). 장 `## N.`(도입 코드는 N.0) · 절 `### …` · 장 게이트 N.G."""
    files = sorted(SRC_DIR.glob("s[0-9][0-9]_*.py"))
    missing_files = [s for s in EXCLUDE if not (SRC_DIR / f"{s}.py").is_file()]
    if missing_files:
        raise PlanError(f"제외 대상 파일이 없다: {missing_files}")
    steps, excluded, found = [], [], set()
    n_code = n_excl = 0
    for path in files:
        spec = EXCLUDE.get(path.stem, {})
        cur = dropped = None
        chapter = None
        for c in read_cells(path):
            if c["type"] == "markdown":
                head = c["text"].split("\n", 1)[0].strip()
                mc, ms = RE_CHAPTER.match(head), RE_SECTION.match(head)
                if mc:
                    chapter, title = int(mc.group(1)), mc.group(2)
                    cur = dropped = None
                    if "*" in spec:
                        found.add((path.stem, "*"))
                        dropped = {"file": path.name, "key": "*", "title": f"{chapter}. {title}",
                                   "reason": spec["*"], "_cells": 0}
                        excluded.append(dropped)
                    else:
                        cur = _new_step(f"{chapter}.0", "intro", chapter, f"{chapter}.0 {title} — 장 도입",
                                        f"src/{path.name}", "")
                elif ms and "*" not in spec:
                    title = ms.group(1)
                    num = RE_SEC_NUM.match(title)
                    key = num.group(1) if num else title
                    cur = dropped = None
                    if key in spec:
                        found.add((path.stem, key))
                        dropped = {"file": path.name, "key": key, "title": title, "reason": spec[key], "_cells": 0}
                        excluded.append(dropped)
                    else:
                        gate = re.match(r"^(\d+)장 게이트", title)
                        first = re.match(r"^\d+\.\d+", title)      # "9.1 ~ 9.5 …" → 9.1
                        sid = f"{gate.group(1)}.G" if gate else (first.group(0) if first else key)
                        tag = next((m.group(1).strip() for m in map(RE_TAG.match, c["text"].splitlines()) if m), "")
                        cur = _new_step(sid, "gate" if gate else "section", chapter, title, f"src/{path.name}", tag)
                continue
            n_code += 1
            if dropped is not None:
                dropped["_cells"] += 1
                n_excl += 1
                continue
            if cur is None:
                raise PlanError(f"장·절 제목 밖의 코드셀: src/{path.name}:{c['mark']}")
            if not cur["_cells"]:
                cur["lines"] = [c["mark"], c["end"]]
                steps.append(cur)
            cur["lines"][1] = c["end"]
            cur["_cells"].append({"line": c["mark"] + 1, "code": c["text"]})

    lost = [f"{s}:{k}" for s, spec in EXCLUDE.items() for k in spec if (s, k) not in found]
    if lost:
        raise PlanError(f"제외 키가 src 에 없다: {lost}")
    got = [(s["id"], len(s["_cells"])) for s in steps]
    want = [(t.split(":")[0], int(t.split(":")[1]) if ":" in t else 1) for t in EXPECTED_STEPS]
    n_cells = sum(n for _, n in got)
    if (n_code, n_excl, n_cells, len(steps)) != (N_CODE_CELLS, N_EXCLUDED_CELLS, N_PLAN_CELLS, N_PLAN_STEPS) \
            or got != want:
        i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), min(len(got), len(want)))
        raise PlanError(f"계획이 검증값과 다르다 — 코드셀 {n_code}(기대 {N_CODE_CELLS}) · 제외 {n_excl}"
                        f"(기대 {N_EXCLUDED_CELLS}) · {n_cells}셀/{len(steps)}단계(기대 {N_PLAN_CELLS}/"
                        f"{N_PLAN_STEPS}) · 첫 차이 #{i + 1}: {got[i:i + 1]} vs 기대 {want[i:i + 1]}")
    for sid, kind, title, f in EXTRA_STEPS:
        steps.append(_new_step(sid, kind, "S" if kind == "serving" else "C", title, f, ""))
    for seq, s in enumerate(steps, 1):
        s.update(seq=seq, cells=len(s["_cells"]), log=f"outputs/steps/{seq:02d}_{s['id']}.txt")
    return steps, excluded


# ══════════════════════════════════════════════════════════════════════
# 셀 실행 (Jupyter 와 동등) · 기록(Tee)
# ══════════════════════════════════════════════════════════════════════
def _display(*objs, **_kw):
    """노트북 display() 대체. s00 은 display 가 이미 있으면 대체 함수를 만들지 않으므로 이것이 쓰인다."""
    pd = sys.modules.get("pandas")
    for o in objs:
        if pd is not None and isinstance(o, (pd.DataFrame, pd.Series)):
            with pd.option_context("display.unicode.east_asian_width", True, "display.width", 160):
                print(o)
        else:
            print(o)


def run_cell(ns: dict, fname: str, cell: dict) -> None:
    """셀 하나 — 줄번호 유지 · 마지막 식의 값은 `Out:` · 셀이 끝나면 그림을 모두 닫는다(inline 과 같다).

    exec/eval 은 이 패키지의 src 파일(노트북과 같은 코드)만 실행한다 — 외부 입력은 실행하지 않는다.
    dont_inherit=True: 이 파일 맨 위의 `from __future__ import annotations` 가 셀 코드에 옮지 않게 한다
    (노트북처럼 함수 주석을 바로 평가한다).
    """
    tree = ast.parse("\n" * (cell["line"] - 1) + cell["code"], filename=fname)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, fname, "exec", dont_inherit=True), ns)
    if last is not None:
        val = eval(compile(ast.Expression(last.value), fname, "eval", dont_inherit=True), ns)
        if val is not None:
            print(f"Out: {val!r}")
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")


def src_traceback(exc: BaseException) -> tuple:
    """트레이스백에서 src/ 프레임만 → (출력 텍스트, 마지막 src 프레임 `src/sNN_x.py:줄`)."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if f.filename.replace("\\", "/").startswith("src/")]
    out = ["Traceback (src 프레임만):"]
    for f in frames:
        out.append(f'  File "{f.filename}", line {f.lineno}, in {f.name}')
        if f.line:
            out.append(f"    {f.line.strip()}")
    out += [x.rstrip("\n") for x in traceback.format_exception_only(type(exc), exc)]
    return "\n".join(out), (f"{frames[-1].filename}:{frames[-1].lineno}" if frames else None)


def _where(exc: BaseException):
    tb = traceback.extract_tb(exc.__traceback__)
    if not tb:
        return None
    p = Path(tb[-1].filename)
    try:
        r = p.resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        r = p.name
    return f"{r}:{tb[-1].lineno}"


def _error(exc: BaseException, where) -> dict:
    return {"type": type(exc).__name__, "message": redact(str(exc)).strip()[:800], "where": where}


class Hub:
    """Tee 공용 상태 — 현재 단계 로그(없으면 화면만), 마지막 출력 시각."""

    def __init__(self, screen):
        self.lock = threading.RLock()
        self.screen = screen
        self.log = None
        self.lines = 0
        self.tees = []
        self.last_output = time.monotonic()

    def write_log(self, text: str) -> None:
        text = redact(text)
        self.log.write(text)
        self.lines += text.count("\n")

    def open_log(self, path: Path) -> None:
        with self.lock:
            self.close_log()
            self.log = open(path, "w", encoding="utf-8", newline="\n", buffering=1)

    def close_log(self) -> int:
        with self.lock:
            for t in self.tees:
                t.drain()
            n = self.lines
            if self.log is not None:
                self.log.close()
            self.log, self.lines = None, 0
            return n


class Tee:
    """sys.stdout/stderr 대체(한 번만 설치). 화면에는 그대로, 로그에는 완성된 줄을 가려서 쓴다.

    optuna 로깅 핸들러가 설치 시점의 스트림을 붙잡으므로 단계마다 바꾸지 않고 로그 대상만 바꾼다.
    """

    def __init__(self, hub: Hub, screen):
        self._hub, self._screen, self._pending = hub, screen, ""

    def write(self, s) -> int:
        s = str(s)
        with self._hub.lock:
            try:
                self._screen.write(s)
            except Exception:   # 화면(파이프)이 끊겨도 로그는 계속 쓴다
                pass
            self._hub.last_output = time.monotonic()
            if self._hub.log is not None and s:
                self._pending += s
                if "\n" in self._pending:
                    done, self._pending = self._pending.rsplit("\n", 1)
                    self._hub.write_log(done + "\n")
        return len(s)

    def writelines(self, lines) -> None:
        for s in lines:
            self.write(s)

    def drain(self) -> None:
        if self._pending and self._hub.log is not None:
            self._hub.write_log(self._pending + "\n")
        self._pending = ""

    def flush(self) -> None:
        try:
            self._screen.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return self._screen.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._screen, name)


def _watch(hub: Hub, state: dict, stop: threading.Event, fast: bool) -> None:
    """2분 넘게 출력이 없으면 화면에만 진행 상황을 알린다(로그에는 쓰지 않는다)."""
    while not stop.wait(10):
        with hub.lock:
            s = state.get("step")
            now = time.monotonic()
            if s is None or now - hub.last_output < 120 or now - state["notice"] < 120:
                continue
            dev, pc = s.get("dev_sec"), "개발 PC FULL" if fast else "개발 PC"   # 개발 PC 값은 FULL 기준
            ref = (f" ({pc} 약 {round(dev / 60)}분)" if dev and dev >= 90
                   else f" ({pc} 약 {dev}초)" if dev else f" ({pc} 10초 미만)")
            try:
                hub.screen.write(f"… [{s['id']}] 실행 중 · {int((now - state['t0']) // 60)}분 경과{ref}\n")
                hub.screen.flush()
            except Exception:
                pass
            state["notice"] = now


def snapshot() -> dict:
    """outputs/(steps 제외) 파일의 (mtime, 크기) — 단계 전후 비교로 새로 쓴 파일을 찾는다."""
    out = {}
    for p in (ROOT / "outputs").rglob("*"):
        try:
            if p.is_file():
                r = p.relative_to(ROOT).as_posix()
                if not r.startswith("outputs/steps/"):
                    st = p.stat()
                    out[r] = (st.st_mtime_ns, st.st_size)
        except OSError:
            pass
    return out


def classify(paths) -> dict:
    f = {"tables": [], "figures": [], "models": [], "other": []}
    for r in sorted(paths):
        if r.startswith("outputs/tables/") and r.endswith(".csv"):
            f["tables"].append(r)
        elif r.startswith("outputs/figures/") and r.endswith(".png"):
            f["figures"].append(r)
        elif r.startswith("outputs/models/"):
            f["models"].append(r)
        else:
            f["other"].append(r)
    return f


# ══════════════════════════════════════════════════════════════════════
# 서빙 단계 S.1~S.4 (증거는 outputs/steps/serving_* 평면 파일)
# ══════════════════════════════════════════════════════════════════════
def _sub(tail: list, module: str = "serving"):
    """`python -X utf8 -B -m <module> …` 을 패키지 루트에서 상대경로로 실행하고 출력을 로그에 남긴다."""
    print(f"$ python -X utf8 -B -m {module} {' '.join(tail)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    r = subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", module, *tail], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    for text in (r.stdout, r.stderr):
        if text.strip():
            print(text.rstrip())
    print(f"  → 종료코드 {r.returncode}")
    return r


def _report(items: list) -> bool:
    for name, ok in items:
        print(f"  {'✓' if ok else '✗'} {name}")
    return bool(items) and all(ok for _, ok in items)


def serving_verify(run) -> tuple:
    """S.1 — eval·deploy 번들 `verify`: 종료코드 0 · status ok · selftest ok."""
    info, files, items = {}, [], []
    for role in ("eval", "deploy"):
        r = _sub(["verify", "--bundle", run.bundle_rel(role), *run.fast_flag])
        out = STEPS_DIR / f"serving_verify_{role}.json"
        out.write_bytes(redact(r.stdout).encode("utf-8"))
        files.append(rel(out))
        try:
            d = json.loads(r.stdout)
        except ValueError:
            d = {}
        items.append((f"{role}: 종료코드 0 · status ok · selftest ok",
                      r.returncode == 0 and d.get("status") == "ok" and d.get("selftest") == "ok"))
        info[f"{role}_bundle_id"] = d.get("bundle_id")
        if role == "eval":
            info.update(tau=d.get("tau"), tau_cls=d.get("tau_cls"))
    return _report(items), info, files


def serving_replay(run) -> tuple:
    """S.2 — 평가 번들 리플레이(인프로세스). 파이프라인 실행이면 메모리 값과 비트 비교
    (근거 tests/test_serving_replay.py:83-110), --check-only 면 제출 파일 정밀도(4·6자리) 비교."""
    import numpy as np
    import pandas as pd
    from serving import load_bundle
    from serving.cli import replay

    ns = run.ns
    bundle = load_bundle(Path(run.bundle_rel("eval")), allow_fast=run.fast)
    raw = ns["df_raw"].copy() if ns is not None else pd.read_csv(DATA_REL, encoding="utf-8-sig")
    out = replay(bundle, raw, "2021-09-01", "2021-09-14")
    csv_path = STEPS_DIR / "serving_replay_eval.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    def col(c):
        return out[c].to_numpy()

    if ns is not None:
        S = SERVICE_MODEL
        te_s, te_c, unc = ns["test_results"][S], ns["test_results"][CLF_MODEL], ns["_unc"]
        X = ns["feat"].loc[te_s["index"], ns["FEATURE_COLS"]]
        lo, hi = ns["interval_bounds"](te_s["pred_avg"], X["is_shutdown"].to_numpy(),
                                       ns["BUNDLE_SUMMARY"]["halfwidths"])
        label = (te_s["pred_peak"] >= ns["cv_results"][S]["tau"]).astype(int)
        how = "메모리 비트 비교"
        items = [("행 336", len(out) == 336),
                 ("ts", list(out["ts"]) == list(pd.DatetimeIndex(te_s["index"]).strftime("%Y-%m-%d %H:%M:%S"))),
                 ("y_avg_pred", np.array_equal(col("y_avg_pred"), te_s["pred_avg"])),
                 ("y_peak_pred", np.array_equal(col("y_peak_pred"), te_s["pred_peak"])),
                 ("peak_prob", np.array_equal(col("peak_prob"), te_c["prob"])),
                 ("q10", np.array_equal(col("q10"), unc["lo"])),
                 ("q50", np.array_equal(col("q50"), unc["mid"])),
                 ("q90", np.array_equal(col("q90"), unc["hi"])),
                 ("peak_label = y_peak_pred ≥ τ", np.array_equal(col("peak_label"), label)),
                 ("pi_lo", np.array_equal(col("pi_lo"), lo)),
                 ("pi_hi", np.array_equal(col("pi_hi"), hi))]
        if ns["FINAL_MODEL_NAME"] == S:
            sub = pd.read_csv(PRED_REL, encoding="utf-8-sig")
            items.append(("peak_label == 제출 파일 peak_pred_label",
                          np.array_equal(col("peak_label"), sub["peak_pred_label"].to_numpy())))
    else:
        sub = pd.read_csv(PRED_REL, encoding="utf-8-sig", float_precision="round_trip")
        how = "제출 파일 정밀도 비교(4·6자리)"
        items = [("행 336", len(out) == 336 == len(sub)),
                 ("ts == 제출 datetime", list(out["ts"]) == list(sub["datetime"].astype(str)))]
        if set(sub["model_name"].astype(str)) == {SERVICE_MODEL}:
            items += [("y_avg_pred (4자리)", np.array_equal(np.round(col("y_avg_pred"), 4), sub["y_avg_pred"].to_numpy())),
                      ("y_peak_pred (4자리)", np.array_equal(np.round(col("y_peak_pred"), 4), sub["y_peak_pred"].to_numpy())),
                      ("peak_prob (6자리)", np.array_equal(np.round(col("peak_prob"), 6), sub["peak_prob"].to_numpy())),
                      ("peak_label == peak_pred_label", np.array_equal(col("peak_label"), sub["peak_pred_label"].to_numpy()))]
        else:
            print("  (제출 파일의 모델이 서비스 모델과 달라 값 비교는 해당 없음)")
    mae = float(np.mean(np.abs(out["y_avg_true"] - out["y_avg_pred"])))
    print(f"S.2 리플레이 — {how} · 번들 {bundle.bundle_id} · {len(out)}행 · MAE {mae:.4f}")
    ok = _report(items)
    check = STEPS_DIR / "serving_replay_check.txt"
    text = [f"S.2 리플레이 — {how}", f"번들 {bundle.bundle_id} · 2021-09-01~09-14 · {len(out)}행 · MAE {mae:.4f}"]
    text += [f"{'✓' if k else '✗'} {n}" for n, k in items] + [f"판정: {'통과' if ok else '실패'}"]
    check.write_bytes(("\n".join(text) + "\n").encode("utf-8"))
    return ok, {"compare": how, "rows": len(out), "mae": round(mae, 4),
                "mismatch": [n for n, k in items if not k]}, [rel(csv_path), rel(check)]


def serving_predict(run) -> tuple:
    """S.3 — CLI 익일 예측: 이력 08-31~09-13 + 계획 09-14 → 평가·배포 번들 예측, 누수 차단 확인."""
    import numpy as np
    import pandas as pd
    from serving.contract import HISTORY_COLUMNS, PLAN_COLUMNS

    raw = run.ns["df_raw"] if run.ns is not None else pd.read_csv(DATA_REL, encoding="utf-8-sig")
    ymd = raw["날짜"].astype(int)
    hist = raw.loc[(ymd >= 20210831) & (ymd <= 20210913), HISTORY_COLUMNS]
    plan = raw.loc[ymd == 20210914, PLAN_COLUMNS]
    S = "outputs/steps/"
    hist.to_csv(ROOT / S / "serving_demo_history.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(ROOT / S / "serving_demo_plan.csv", index=False, encoding="utf-8-sig")
    files = [S + "serving_demo_history.csv", S + "serving_demo_plan.csv"]
    inputs = ["--history", S + "serving_demo_history.csv", "--plan", S + "serving_demo_plan.csv"]
    items, info = [], {}

    def predicted(role):
        name = f"serving_predict_{role}_20210914.csv"
        (ROOT / S / name).unlink(missing_ok=True)
        r = _sub(["predict", "--bundle", run.bundle_rel(role), *run.fast_flag, *inputs, "--out", S + name])
        if r.returncode != 0 or not (ROOT / S / name).is_file():
            return None
        files.append(S + name)
        return pd.read_csv(ROOT / S / name, encoding="utf-8-sig", float_precision="round_trip")

    got = predicted("eval")
    items.append(("평가 번들 predict: 종료코드 0 · 09-14 24행",
                  got is not None and len(got) == 24 and got["ts"].astype(str).str.startswith("2021-09-14").all()))
    if got is not None and run.final_model() == SERVICE_MODEL:
        sub = pd.read_csv(PRED_REL, encoding="utf-8-sig", float_precision="round_trip")
        sub = sub[sub["datetime"].astype(str).str.startswith("2021-09-14")]
        same = (len(sub) == 24 and len(got) == 24 and list(got["ts"]) == list(sub["datetime"])
                and np.array_equal(np.round(got["y_avg_pred"].to_numpy(), 4), sub["y_avg_pred"].to_numpy())
                and np.array_equal(np.round(got["y_peak_pred"].to_numpy(), 4), sub["y_peak_pred"].to_numpy())
                and np.array_equal(np.round(got["peak_prob"].to_numpy(), 6), sub["peak_prob"].to_numpy())
                and np.array_equal(got["peak_label"].to_numpy(), sub["peak_pred_label"].to_numpy()))
        items.append(("09-14 예측 == 제출 파일 (4·6자리 반올림 · 라벨)", same))
    elif got is not None:
        print("  (최종모델이 서비스 모델과 달라 제출 파일 비교는 해당 없음)")
    dep = predicted("deploy")
    print("  ※ 배포 번들은 09-14 까지 학습했다 — 이 09-14 예측은 학습 구간 안의 데모이고 성능 근거가 아니다")
    items.append(("배포 번들 predict: 종료코드 0 · 24행 (데모)", dep is not None and len(dep) == 24))
    # 누수 차단 — 계획에 대상일 전력('평균')이 있으면 입력 오류(종료코드 2)이고 출력 파일이 없어야 한다
    plan.assign(평균=100).to_csv(ROOT / S / "serving_demo_plan_leak.csv", index=False, encoding="utf-8-sig")
    files.append(S + "serving_demo_plan_leak.csv")
    leak_out = ROOT / S / "serving_predict_leak.csv"
    leak_out.unlink(missing_ok=True)
    r = _sub(["predict", "--bundle", run.bundle_rel("eval"), *run.fast_flag, "--history",
              S + "serving_demo_history.csv", "--plan", S + "serving_demo_plan_leak.csv",
              "--out", S + "serving_predict_leak.csv"])
    items.append(("누수 차단: 종료코드 2 · PLAN_HAS_POWER · 출력 파일 없음",
                  r.returncode == 2 and "PLAN_HAS_POWER" in r.stderr and not leak_out.exists()))
    info.update(eval_rows=None if got is None else len(got), deploy_rows=None if dep is None else len(dep),
                leak_exit=r.returncode, deploy_note="배포 번들 09-14 예측은 학습 구간 안의 데모")
    return _report(items), info, files


def serving_tests(run) -> tuple:
    """S.4 — 서빙·통계·학습설정·비용·인과성의 단위 및 회귀테스트."""
    r = _sub(["tests", "-q", "-p", "no:cacheprovider"], module="pytest")
    text = redact(r.stdout + ("\n" + r.stderr if r.stderr.strip() else ""))
    out = STEPS_DIR / "serving_unit_tests.txt"
    out.write_bytes(text.encode("utf-8"))
    summary = next((s.strip() for s in reversed(text.splitlines()) if s.strip()), "")
    ok = _report([("종료코드 0 · 테스트 통과", r.returncode == 0 and re.search(r"\b[1-9]\d* passed\b", text) is not None)])
    return ok, {"summary": summary}, [rel(out)]


SERVING = {"S.1": serving_verify, "S.2": serving_replay, "S.3": serving_predict, "S.4": serving_tests}


# ══════════════════════════════════════════════════════════════════════
# 완료 점검 C
# ══════════════════════════════════════════════════════════════════════
def completion_checks(run) -> dict:
    import pandas as pd

    res = {"hard": [], "warn": [], "info": []}

    def add(grade, name, ok, detail=""):
        res[grade].append({"name": name, "ok": bool(ok), "detail": redact(detail)})

    def table(name):
        p = ROOT / "outputs" / "tables" / f"{name}.csv"
        return pd.read_csv(p, encoding="utf-8-sig") if p.is_file() else None

    full, final = run.mode == "FULL", run.final_model()
    # ── hard ──
    pred_p = ROOT / PRED_REL
    if pred_p.is_file():
        pred = pd.read_csv(pred_p, encoding="utf-8-sig")
        add("hard", "예측 파일 336행 × 9열 규격", pred.shape == (336, 9) and list(pred.columns) == PRED_COLUMNS,
            f"{pred.shape[0]}행 × {pred.shape[1]}열")
        na = int(pred.isna().sum().sum())
        add("hard", "예측 파일 결측 0", na == 0, f"결측 {na}")
        want = [(_dt.datetime(2021, 9, 1) + _dt.timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S") for h in range(336)]
        ts = pred["datetime"].astype(str).tolist() if "datetime" in pred else []
        add("hard", "2021-09-01 00:00 ~ 09-14 23:00 매시", ts == want, f"{ts[0]} ~ {ts[-1]}" if ts else "datetime 없음")
        names = sorted(set(pred["model_name"].astype(str))) if "model_name" in pred else []
        add("hard", "model_name 이 모두 최종모델", final is not None and names == [final], f"{names} · 최종모델 {final}")
    else:
        add("hard", "예측 파일 존재", False, f"{PRED_REL} 없음")
    expected = [t for t in EXPECTED_TABLES if not (run.args.skip_bundle and t == "ch6_model_bundles")]
    if run.args.check_only:
        have, how = {p.stem for p in (ROOT / "outputs" / "tables").glob("*.csv")}, "outputs/tables 에 있음"
    else:
        have, how = {Path(f).stem for s in run.steps for f in s["files"]["tables"]}, "이번 실행에서 씀"
    miss = [t for t in expected if t not in have]
    add("hard", f"기대 표 {len(expected)}개 ({how})", not miss,
        f"누락 {len(miss)}개: {', '.join(miss[:8])}" if miss else f"모두 있음 · 전체 {len(have)}개")
    pngs = list((ROOT / "outputs" / "figures").glob("*.png"))
    fidx = ROOT / "outputs" / "figures" / "figure_index.csv"
    nfid = int(pd.read_csv(fidx, encoding="utf-8-sig")["fid"].nunique()) if fidx.is_file() else 0
    add("hard", "그림 PNG 39장 · figure_index 고유 fid 39", len(pngs) == 39 and nfid == 39, f"PNG {len(pngs)} · fid {nfid}")
    if run.args.skip_bundle:
        add("info", "번들 manifest 점검", True, "--skip-bundle — 생략")
    else:
        miss_b = [r for r in ("eval", "deploy") if not (ROOT / run.bundle_rel(r) / "manifest.json").is_file()]
        add("hard", f"번들 manifest.json ({run.bundle_rel('{eval,deploy}')})", not miss_b,
            "둘 다 있음" if not miss_b else f"없음: {miss_b}")
    for sid, name in (("S.1", "S.1 번들 verify"), ("S.2", "S.2 리플레이 비교"), ("S.3", "S.3 CLI 예측·누수 차단"),
                      ("S.4", "S.4 단위·회귀테스트")):
        st = run.by_id[sid]["status"]
        add("info" if st == "skipped" else "hard", name, st == "ok", st)
    reg, sig = table("ch2_regression"), table("ch2_significance")
    if reg is not None and pred_p.is_file():
        row = reg.loc[reg["모델"] == final]
        observed_mae = float((pred["y_avg_true"] - pred["y_avg_pred"]).abs().mean())
        recorded_mae = float(row["MAE"].iloc[0]) if len(row) else math.nan
        add("hard", "보고 MAE == 제출 CSV 재계산", abs(observed_mae-recorded_mae) <= 0.0006,
            f"CSV {observed_mae:.6f} / 표 {recorded_mae:.3f}")
        expected_n = int(pred["peak_true_label"].sum())
        add("hard", "피크 회귀 평가 표본 == 실제 피크 라벨", len(row) == 1 and int(row["Peak-n"].iloc[0]) == expected_n,
            f"피크 {expected_n}시간")
    ci_columns = {"ci_low_raw", "ci_high_raw", "ci90_low_raw", "ci90_high_raw", "개선_5pct(주분석)", "개선_10pct(사후 탐색)"}
    ci_ok = sig is not None and ci_columns <= set(sig.columns)
    if ci_ok:
        ci_ok = bool(((sig["ci_low_raw"] <= sig["ci90_low_raw"]) &
                      (sig["ci90_low_raw"] <= sig["ci90_high_raw"]) &
                      (sig["ci90_high_raw"] <= sig["ci_high_raw"])).all())
    add("hard", "95% 주분석 및 90% 사후 보조구간", ci_ok, "구간 순서와 필수 열 확인")
    costs = table("ch4_scenarios")
    cost_ok = costs is not None and {"기본요금 절감(원)", "인건비 증가(원)", "순절감액(원)", "전력량요금 차액(원)", "비교기간(개월)"} <= set(costs.columns)
    if cost_ok:
        cost_ok = bool((costs["전력량요금 차액(원)"] == 0).all() and
                       (costs["비교기간(개월)"] > 0).all() and
                       ((costs["기본요금 절감(원)"] - costs["인건비 증가(원)"] - costs["순절감액(원)"]).abs() <= 1).all())
    add("hard", "비용 기간 통일 · 미검증 에너지 절감 제외", cost_ok, "같은 관측기간의 기본요금 차액에서 추가 인건비 차감")
    # ── warn ──
    add("warn", f"최종모델 == {SERVICE_MODEL}", final == SERVICE_MODEL, str(final))
    add("info", "결과 수치의 기준", True, "이번 실행의 평가표·예측파일·provenance를 기준으로 확인; 수정 전 수치를 강제하지 않음")
    hpo = table("hpo_trials")
    n_hpo = -1 if hpo is None else len(hpo)
    add("warn", f"hpo_trials {100 if full else 6}행", n_hpo == (100 if full else 6), f"{n_hpo}행")
    s4 = run.by_id["S.4"]
    add("warn", "S.4 단위·회귀테스트", s4["status"] == "ok",
        (run.index["serving"].get("S.4") or {}).get("summary") or (run.index["serving"].get("S.4") or {}).get("reason")
        or s4["status"])
    data_ok = (ROOT / DATA_REL).is_file() and file_sha(ROOT / DATA_REL) == DATA_SHA256
    add("warn", "데이터 sha256 == 정본", data_ok, "일치" if data_ok else "다름 — 엑셀로 다시 저장하지 않았는지 확인")
    # ── info ──
    base = ROOT / "outputs" / "models"
    n_job = sum(1 for sub in ("baselines", "comparison") for _ in (base / sub / run.mode.lower()).rglob("model.joblib"))
    add("info", "joblib 아카이브 (baselines·comparison, 기대 11)", n_job == 11, f"{n_job}개")
    removed = 0
    for p in sorted(ROOT.rglob("__pycache__"), reverse=True):
        if p.is_dir() and not any(x.startswith(".") for x in p.relative_to(ROOT).parts):
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    add("info", "__pycache__ 정리", True, f"{removed}개 삭제" if removed else "없음")

    print("── 완료 점검 ──")
    for grade, bad in (("hard", "✗"), ("warn", "!"), ("info", "·")):
        for c in res[grade]:
            print(f"  [{grade}] {'✓' if c['ok'] and grade != 'info' else bad} {c['name']} — {c['detail']}")
    return res


# ══════════════════════════════════════════════════════════════════════
# 실행기 (자식 프로세스)
# ══════════════════════════════════════════════════════════════════════
class _Quit(Exception):
    """멈춤 프롬프트에서 q."""


def decide_pause(req: str) -> str:
    """stdin·stdout 이 모두 TTY 이고(POSIX 는 포그라운드 프로세스 그룹일 때만) 멈춤을 켠다."""
    if req == "none":
        return "none"
    ok = bool(sys.stdin and sys.stdin.isatty() and sys.stdout.isatty())
    if ok and os.name == "posix":
        try:
            ok = os.tcgetpgrp(sys.stdin.fileno()) == os.getpgrp()
        except OSError:
            ok = False
    if not ok:
        print(f"※ --pause {req}: 대화형 터미널(포그라운드)이 아니라 멈춤 없이 진행한다")
        return "none"
    return req


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def guards(args) -> tuple:
    """사전점검 → (실패 목록, 알림 목록). `--force` 는 FULL 덮어쓰기·잠금 가드만 푼다."""
    problems, notes = [], []
    if (ROOT / ".git").exists():
        problems.append("개발 저장소(.git 있음)에서는 실행하지 않는다 — 커밋된 outputs 를 덮는다. 패키지 사본에서 실행하라")
    if sys.version_info < (3, 10):
        problems.append(f"Python 3.10 이상이 필요하다 (현재 {platform.python_version()})")
    if not (ROOT / DATA_REL).is_file():
        problems.append(f"데이터가 없다: {DATA_REL}")
    try:
        from tools.build_serving_core import generated_source
        if (ROOT / "serving" / "_core.py").read_text(encoding="utf-8") != generated_source():
            problems.append("학습·서빙 코어가 다르다 — python tools/build_serving_core.py 로 재생성")
    except (OSError, ImportError, ValueError, SyntaxError) as exc:
        problems.append(f"서빙 코어 동기화 확인 실패: {type(exc).__name__}")

    def has(m):
        try:
            return find_spec(m) is not None
        except (ImportError, ValueError):
            return False

    missing = [m for m in REQUIRED if not has(m)]
    if missing:
        problems.append(f"필수 패키지가 없다: {', '.join(missing)} — bash setup_pod.sh 로 설치")
    if not has("tensorflow"):
        notes.append("tensorflow 가 없다 — SimpleRNN·DNN 만 건너뛰고 나머지는 그대로 돈다")
    old = read_index()
    if args.fast and not args.check_only and old and old.get("mode") == "FULL" and old.get("status") == "ok":
        msg = "outputs 에 FULL 결과(status ok)가 있다 — --fast 는 패키지 복사본에서 실행하라"
        (notes if args.force else problems).append(msg + (" (--force: 덮어쓴다)" if args.force else " (덮어쓰려면 --force)"))
    try:
        pid = int(LOCK_PATH.read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        pid = None
    if pid is not None:
        if os.name != "posix":
            notes.append(f"잠금파일이 남아 있다(PID {pid}) — Windows 는 확인하지 않는다. 다른 실행이 없으면 무시해도 된다")
        elif _pid_alive(pid):
            msg = f"다른 run_all 이 실행 중이다(PID {pid}, outputs/steps/.run_all.lock)"
            (notes if args.force else problems).append(msg + (" — --force 로 무시한다" if args.force else ""))
        else:
            notes.append(f"지난 실행의 잠금파일(PID {pid})은 프로세스가 없어 무시한다")
    return problems, notes


class Runner:
    def __init__(self, args):
        self.args = args
        self.mode, self.fast, self.pause = "FULL", False, "none"
        self.ns = None
        self.index, self.base = None, None
        self.steps, self.excluded, self.by_id = [], [], {}
        self.hub = None
        self.watch = {"step": None, "t0": 0.0, "notice": 0.0}
        self.t0, self.snap = 0.0, None
        self.sig_name, self.sigs = None, []   # 중단시킨 신호 이름 · 처리기를 단 신호

    @property
    def fast_flag(self) -> list:
        return ["--allow-fast"] if self.fast else []   # FAST 번들은 CLI 숨김 옵션으로만 열린다

    def bundle_rel(self, role: str) -> str:
        return f"outputs/models/{self.mode.lower()}/{role}"

    def final_model(self):
        """최종모델 — 파이프라인 실행이면 메모리 값, --check-only 면 스코어카드 1위(s06 과 같은 규칙)."""
        if self.ns is not None and "FINAL_MODEL_NAME" in self.ns:
            return str(self.ns["FINAL_MODEL_NAME"])
        import pandas as pd
        p = ROOT / "outputs" / "tables" / "ch2_scorecard.csv"
        return str(pd.read_csv(p, encoding="utf-8-sig")["모델"].iloc[0]) if p.is_file() else None

    # ── 기록 ────────────────────────────────────────────────────────────
    def write_index(self) -> None:
        """index.json 원자적 교체(os.replace). 읽는 쪽이 파일을 잡고 있으면 잠시 기다렸다 다시 한다."""
        self.index["steps"] = [public(s) for s in self.steps]
        data = json.dumps(_jsonable(self.index), ensure_ascii=False, indent=1, allow_nan=False) + "\n"
        tmp = STEPS_DIR / ".index.json.tmp"
        tmp.write_bytes(data.encode("utf-8"))
        for _ in range(20):
            try:
                os.replace(tmp, INDEX_PATH)
                return
            except PermissionError:
                time.sleep(0.25)
        os.replace(tmp, INDEX_PATH)

    def screen(self, text: str) -> None:
        """화면 전용 출력(로그에는 쓰지 않는다)."""
        with self.hub.lock:
            try:
                self.hub.screen.write(text + "\n")
                self.hub.screen.flush()
            except Exception:
                pass

    def new_index(self) -> dict:
        a = self.args
        return {"schema": 1, "tool": "run_all", "mode": self.mode, "status": "running",
                "started": now_iso(), "finished": None, "total_sec": 0.0, "current": None,
                "flags": {"fast": a.fast, "pause": a.pause, "skip_serving": a.skip_serving, "no_tests": a.no_tests,
                          "skip_bundle": a.skip_bundle, "cpu": a.cpu, "check_only": a.check_only},
                "python": platform.python_version(), "platform": platform.platform(), "versions": versions(),
                "provenance": provenance()[0], "excluded": [public(e) for e in self.excluded], "steps": [],
                "checks": {"hard": [], "warn": [], "info": []}, "serving": {k: {} for k in SERVING}}

    # ── 시작 ────────────────────────────────────────────────────────────
    def main(self) -> int:
        a = self.args
        if getattr(a, "export_analysis", None) and a.check_only:
            print("[사전점검 실패] --export-analysis 는 학습 namespace가 필요하므로 --check-only 와 함께 쓸 수 없습니다.", file=sys.stderr)
            return 2
        if getattr(a, "export_analysis", None):
            target = Path(a.export_analysis).resolve()
            if target.exists() and (not target.is_dir() or any(target.iterdir())):
                print("[사전점검 실패] --export-analysis 대상은 새 폴더 또는 빈 폴더여야 합니다.", file=sys.stderr)
                return 2
            a.export_analysis = str(target)
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        sys.dont_write_bytecode = True
        self.pause = decide_pause(a.pause)
        problems, notes = guards(a)
        for n in notes:
            print(f"※ {n}")
        if problems:
            for p in problems:
                print(f"[사전점검 실패] {p}", file=sys.stderr)
            return 2
        try:
            self.steps, self.excluded = build_plan()
        except PlanError as e:
            print(f"[계획 실패] {e}", file=sys.stderr)
            return 2
        old = read_index()
        if a.check_only:
            old_mode = (old or {}).get("mode")
            if a.fast and old_mode == "FULL":
                print("[사전점검 실패] index.json 은 FULL 실행 기록이다 — --check-only 에 --fast 를 붙이지 마라", file=sys.stderr)
                return 2
            self.mode = "FAST" if a.fast else (old_mode if old_mode in ("FULL", "FAST") else "FULL")
            # 건너뛰기 옵션이 지난 실행에서 통과한 서빙 단계를 건너뛰게 하면, 그 증거(단계 로그·serving_*)만
            # 지워지고 다시 만들어지지 않는다 — 지우기 전에 거부한다(pytest 가 없는 등 환경 때문이면 해당 없음)
            lost = [s.get("id") for s in (old or {}).get("steps") or []
                    if isinstance(s, dict) and s.get("id") in SERVING and s.get("status") == "ok"
                    and self._skip_reason(s.get("id"), flags_only=True)]
            if lost:
                print(f"[사전점검 실패] --check-only 에 건너뛰기 옵션을 붙이면 지난 실행에서 통과한 {', '.join(lost)} 의 "
                      "증거(단계 로그·outputs/steps/serving_*)가 지워지고 다시 만들어지지 않는다 — "
                      "--skip-serving·--skip-bundle·--no-tests 를 빼고 다시 실행하라", file=sys.stderr)
                return 2
        else:
            self.mode = "FAST" if a.fast else "FULL"
        self.fast = self.mode == "FAST"
        STEPS_DIR.mkdir(parents=True, exist_ok=True)
        if a.check_only:
            self._prepare_check_only(old)
        else:   # 지난 실행 기록을 지운다(README.md 만 남긴다)
            for p in STEPS_DIR.iterdir():
                if p.is_file() and p.name != "README.md":
                    p.unlink()
            self.index = self.new_index()
        self.by_id = {s["id"]: s for s in self.steps}
        LOCK_PATH.write_bytes(f"{os.getpid()}\n".encode("utf-8"))
        self.write_index()
        return self._run()

    def _prepare_check_only(self, old) -> None:
        """--check-only: 파이프라인 단계 기록·로그는 그대로 두고 S.1~S.4 · C 만 새로 쓴다."""
        n = N_PLAN_STEPS
        if old and [s.get("id") for s in (old.get("steps") or [])[:n]] == [s["id"] for s in self.steps[:n]]:
            self.base, self.index = old.get("status"), old
            self.steps[:n] = old["steps"][:n]
        else:
            self.index = self.new_index()
            for s in self.steps[:n]:
                s.update(status="skipped", log=None)
        for s in self.steps[n:]:
            (ROOT / s["log"]).unlink(missing_ok=True)
        for p in STEPS_DIR.glob("serving_*"):
            p.unlink()
        self.index.update(status="running", current=None, checks={"hard": [], "warn": [], "info": []},
                          serving={k: {} for k in SERVING})

    def _header(self) -> None:
        pipe = [s for s in self.steps[:N_PLAN_STEPS]]
        what = ("기존 outputs 점검(--check-only)" if self.args.check_only
                else f"{len(pipe)}단계({sum(s['cells'] for s in pipe)}셀)")
        print("━" * 64)
        print(f"KAMP run_all · {self.mode} · {what} + 서빙 S.1~S.4 + 완료 점검 C")
        print(f"  Python {platform.python_version()} · {platform.platform()} · 시작 {now_iso()}")
        print("  제외(보고서 생성 절): " + " · ".join(
            f"{int(e['file'][1:3])}장 전체" if e["key"] == "*" else e["key"].split(" — ")[0] for e in self.excluded))
        print("  기록: outputs/steps/ (단계 로그 · index.json) — 결과 보기: python show_results.py")
        if self.fast:
            print("  ※ FAST — 동작 확인용 축소 실행이다. 수치는 보고서 값이 아니다")
        print("━" * 64)

    def _run(self) -> int:
        a = self.args
        self._header()
        real_out, real_err = sys.stdout, sys.stderr
        self.hub = Hub(real_out)
        sys.stdout, sys.stderr = Tee(self.hub, real_out), Tee(self.hub, real_err)
        self.hub.tees = [sys.stdout, sys.stderr]
        stop = threading.Event()
        threading.Thread(target=_watch, args=(self.hub, self.watch, stop, self.fast), daemon=True).start()
        runner_main = sys.modules.get("__main__")
        if not a.check_only:
            # 셀 namespace — IPython 처럼 사용자 모듈을 __main__ 으로 둔다(get_ipython 없음 → s00 이 Agg)
            user = types.ModuleType("__main__")
            user.__dict__.update({"__builtins__": builtins, "display": _display})
            self.ns = user.__dict__
            sys.modules["__main__"] = user
        started = time.monotonic()
        rc, status, failed = 0, "ok", None
        order = [s for s in self.steps if not (a.check_only and s["kind"] in PIPELINE_KINDS)]
        # SIGTERM(서버 중지 등)·SIGHUP(터미널 닫힘)도 Ctrl+C 와 같은 중단 경로로 기록을 남기고 끝낸다.
        # 기본 동작이면 finally 없이 죽어 index.json 이 running 으로 굳는다. nohup 이 무시시킨 SIGHUP 은 그대로 둔다
        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None and signal.getsignal(sig) is not signal.SIG_IGN:
                signal.signal(sig, self._on_signal)
                self.sigs.append(sig)
        try:
            for i, s in enumerate(order):
                if s["kind"] in PIPELINE_KINDS:
                    if not self._pipeline_step(s):
                        rc, status, failed = 1, "failed", s
                        break
                elif s["kind"] == "serving":
                    self._serving_step(s)
                elif not self._check_step(s):
                    rc, status = 3, "check_failed"
                if i + 1 < len(order) and self._want_pause(s, order[i + 1]) and self._prompt() == "q":
                    raise _Quit()
            if rc == 0 and getattr(a, "export_analysis", None):
                if not self._export_analysis():
                    rc, status = 1, "export_failed"
        except (KeyboardInterrupt, _Quit):
            rc, status = 130, "interrupted"
            cur = self.by_id.get(self.index.get("current") or "")
            if cur is not None and cur["status"] == "running":
                sig = self.sig_name
                print(f"\n중단됨 ({sig + ' — 서버 중지·터미널 종료 등' if sig else 'Ctrl+C'})", file=sys.stderr)
                cur.update(status="failed", seconds=round(time.monotonic() - self.t0, 2),
                           error={"type": sig or "KeyboardInterrupt",
                                  "message": f"{sig} 신호로 중단" if sig else "사용자 중단", "where": None})
        finally:
            for sig in self.sigs:   # 기록을 쓰는 동안 오는 두 번째 신호가 쓰기를 끊지 않게
                signal.signal(sig, signal.SIG_IGN)
            stop.set()
            self.watch["step"] = None
            if not a.check_only:
                sys.modules["__main__"] = runner_main
            cur = self.by_id.get(self.index.get("current") or "")
            n = self.hub.close_log()
            if cur is not None:
                cur["log_lines"] = n
            sys.stdout, sys.stderr = real_out, real_err
            if a.check_only and self.base is not None:
                # 파이프라인이 끝나지 않은 기록(failed 등)을 점검 통과로 올리지 않는다
                if self.base not in ("ok", "check_failed") and status != "interrupted":
                    status = self.base
                self.index["checks"]["info"].append({"name": "--check-only 재점검", "ok": rc == 0, "detail": now_iso()})
            else:
                self.index.update(finished=now_iso(), total_sec=round(time.monotonic() - started, 2))
            self.index.update(status=status, current=None)
            self.write_index()
            try:
                LOCK_PATH.unlink()
            except OSError:
                pass
        self._summary(rc, failed, time.monotonic() - started)
        return rc

    def _export_analysis(self) -> bool:
        """Optional post-success export; does not add or alter the 62 steps."""
        target = Path(self.args.export_analysis).resolve()
        self.index["analysis_export"] = {"status": "running"}
        self.write_index()
        try:
            from tools.analysis_snapshot import export_snapshot
            metadata = export_snapshot(self.ns, target, root=ROOT, run_metadata={
                "mode": self.mode, "pipeline_step_count": len(self.steps),
                "run_started": self.index.get("started"),
                "pipeline_status": "ok", "final_model": self.final_model(),
            })
        except KeyboardInterrupt:
            self.index["analysis_export"] = {"status": "interrupted"}
            raise
        except Exception as exc:
            self.index["analysis_export"] = {"status": "failed", "error": type(exc).__name__}
            print(redact(traceback.format_exc()).rstrip(), file=sys.stderr)
            self.write_index()
            return False
        self.index["analysis_export"] = {"status": "complete", "fingerprint": metadata["fingerprint"]}
        self.write_index()
        self.screen("  ✓ 분석 snapshot 저장·스키마·해시 검증 완료")
        return True

    def _on_signal(self, signum, frame) -> None:
        """SIGTERM·SIGHUP → KeyboardInterrupt — 셀·서빙·점검 어디서 받아도 _run 의 중단 기록 경로를 탄다."""
        self.sig_name = signal.Signals(signum).name
        raise KeyboardInterrupt

    # ── 단계 ────────────────────────────────────────────────────────────
    def _begin(self, s: dict) -> None:
        s["status"] = "running"
        self.index["current"] = s["id"]
        self.write_index()
        loc = f"  ({s['file']}:{s['lines'][0]})" if s.get("lines") else ""
        self.screen(f"\n━━ [{s['seq']:02d}/{len(self.steps)}] {s['title']}{loc} ━━")
        self.snap = snapshot() if s["kind"] in PIPELINE_KINDS else None
        self.hub.open_log(ROOT / s["log"])
        self.t0 = time.monotonic()
        self.hub.last_output = self.t0
        self.watch.update(step=s, t0=self.t0, notice=self.t0)

    def _end(self, s: dict, status: str) -> None:
        s["seconds"] = round(time.monotonic() - self.t0, 2)
        self.watch["step"] = None
        s["log_lines"] = self.hub.close_log()
        if self.snap is not None:
            s["files"] = classify(k for k, v in snapshot().items() if self.snap.get(k) != v)
        s["status"] = status
        self.index["current"] = None
        self.write_index()
        f = s["files"]
        made = " · ".join(f"{n} {len(f[k])}" for k, n in (("tables", "표"), ("figures", "그림"), ("models", "모델")) if f[k])
        self.screen(f"  {'✓' if status == 'ok' else '✗'} {s['id']} {status} · {fmt_sec(s['seconds'])}"
                    + (f" · {made}" if made else "") + f" · 로그 {s['log']}")

    def _pipeline_step(self, s: dict) -> bool:
        self._begin(s)
        try:
            for cell in s["_cells"]:
                run_cell(self.ns, s["file"], cell)
        except KeyboardInterrupt:
            raise
        except BaseException as e:   # SystemExit 도 단계 실패로 기록한다
            text, where = src_traceback(e)
            print(text, file=sys.stderr)
            s["error"] = _error(e, where)
            self._end(s, "failed")
            return False
        self._end(s, "ok")
        return True

    def _skip_reason(self, sid: str, flags_only: bool = False):
        """서빙 단계를 건너뛸 이유(없으면 None). flags_only 면 옵션 때문인 것만 본다."""
        a = self.args
        if a.skip_serving:
            return "--skip-serving"
        if sid in ("S.1", "S.2", "S.3") and a.skip_bundle:
            return "--skip-bundle (10.5절 번들을 만들지 않았다)"
        if sid == "S.4":
            if a.no_tests:
                return "--no-tests"
            if not flags_only and find_spec("pytest") is None:
                return "pytest 없음 — bash setup_pod.sh --with-tests 로 설치하면 돈다"
        return None

    def _serving_step(self, s: dict) -> None:
        reason = self._skip_reason(s["id"])
        if reason:
            (ROOT / s["log"]).write_bytes(f"건너뜀: {reason}\n".encode("utf-8"))
            s.update(status="skipped", seconds=0.0, log_lines=1)
            self.index["serving"][s["id"]] = {"status": "skipped", "reason": reason}
            self.write_index()
            self.screen(f"\n━━ [{s['seq']:02d}/{len(self.steps)}] {s['title']} ━━\n  – 건너뜀: {reason}")
            return
        self._begin(s)
        try:
            ok, info, files = SERVING[s["id"]](self)
        except KeyboardInterrupt:
            raise
        except Exception as e:   # 실행기 쪽 오류도 단계 실패로 남기고 다음 단계로 간다
            print(redact(traceback.format_exc()).rstrip(), file=sys.stderr)
            ok, info, files = False, {}, []
            s["error"] = _error(e, _where(e))
        s["files"]["other"] = files
        self.index["serving"][s["id"]] = {"status": "ok" if ok else "failed", **info}
        self._end(s, "ok" if ok else "failed")

    def _check_step(self, s: dict) -> bool:
        self._begin(s)
        try:
            checks = completion_checks(self)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(redact(traceback.format_exc()).rstrip(), file=sys.stderr)
            s["error"] = _error(e, _where(e))
            checks = {"hard": [{"name": "완료 점검 실행", "ok": False, "detail": f"{type(e).__name__}"}],
                      "warn": [], "info": []}
        self.index["checks"] = checks
        ok = all(c["ok"] for c in checks["hard"])
        self._end(s, "ok" if ok else "failed")
        return ok

    def _want_pause(self, s: dict, nxt: dict) -> bool:
        if self.pause == "section":
            return True
        return self.pause == "chapter" and s["chapter"] != nxt["chapter"]

    def _prompt(self) -> str:
        with self.hub.lock:
            self.hub.screen.write("[Enter] 다음 단계 · c 멈춤 해제 · q 중단 > ")
            self.hub.screen.flush()
        line = sys.stdin.readline()
        ans = line.strip().lower() if line else "c"      # EOF 는 c
        if ans == "c":
            self.pause = "none"
            self.screen("  멈춤 해제 — 끝까지 이어서 실행한다")
        elif ans == "q":
            self.screen("  중단 — index.json 을 기록하고 끝낸다")
        return ans

    def _summary(self, rc: int, failed, sec: float) -> None:
        cnt = {k: sum(1 for s in self.steps if s["status"] == k) for k in ("ok", "failed", "skipped", "pending")}
        ch = self.index["checks"]
        print("━" * 64)
        print(f"  모드 {self.mode} · 상태 {self.index['status']} · {fmt_sec(sec)} · 단계 ok {cnt['ok']}"
              f" · 실패 {cnt['failed']} · 건너뜀 {cnt['skipped']}" + (f" · 미실행 {cnt['pending']}" if cnt["pending"] else ""))
        if ch["hard"]:
            print(f"  완료 점검 hard {sum(c['ok'] for c in ch['hard'])}/{len(ch['hard'])}"
                  f" · warn {sum(c['ok'] for c in ch['warn'])}/{len(ch['warn'])}")
        print("  기록: outputs/steps/ (단계 로그 · index.json) — 결과 보기: python show_results.py")
        if rc == 0:
            print(f"RUN_ALL_OK mode={self.mode}")
        elif rc == 1:
            err = failed.get("error") or {}
            print(f"RUN_ALL_FAILED mode={self.mode} step={failed['id']} at={err.get('where')} log={failed['log']}")
        elif rc == 3:
            bad = [c["name"] for c in self.index["checks"]["hard"] if not c["ok"]]
            print(f"RUN_ALL_CHECK_FAILED mode={self.mode} hard={bad}")
        else:
            print(f"RUN_ALL_INTERRUPTED mode={self.mode}")


# ══════════════════════════════════════════════════════════════════════
# 재실행 없이 바로 처리하는 명령: --list · --compare-notebook · --pack-results
# ══════════════════════════════════════════════════════════════════════
def cmd_list() -> int:
    try:
        steps, excluded = build_plan()
    except PlanError as e:
        print(f"[계획 실패] {e}", file=sys.stderr)
        return 2
    pipe = steps[:N_PLAN_STEPS]
    n_cells, n_ex = sum(s["cells"] for s in pipe), sum(e["_cells"] for e in excluded)
    print(f"실행 계획 — {len(pipe)}단계 · {n_cells}셀 (코드셀 {n_cells + n_ex} − 제외 {n_ex}) · "
          f"이어서 서빙 S.1~S.4 · 완료 점검 C → 총 {len(steps)}단계")
    print("  순번 단계   셀  개발PC   보고서       제목  (코드 위치)")
    for s in steps:
        loc = f"{s['file']}:{s['lines'][0]}-{s['lines'][1]}" if s["lines"] else s["file"]
        dev = fmt_sec(s["dev_sec"]) if s["dev_sec"] else "-"
        print(f"  {s['seq']:02d}   {s['id']:<5} {s['cells'] or '-':>2}  {dev:>7}  {','.join(s['report']):<12} "
              f"{s['title']}  ({loc})")
    print(f"\n제외 {n_ex}셀 / {len(excluded)}키 — 보고서·제출물 생성 절(실험 결과에는 영향 없음):")
    for e in excluded:
        print(f"  {e['file']:<18} {e['key']:<6} {e['_cells']}셀  {e['title']} — {e['reason']}")
    print(f"\n개발 PC FULL 합계 약 {round(sum(DEV_SEC.values()) / 60)}분(10초 넘는 단계만 표시) · KAMP 예상 45~75분 · "
          "FAST 약 10~15분")
    print("※ 계획만 출력했다 — 실행·파일 쓰기 없음")
    return 0


def _same(label: str, a: list, b: list) -> bool:
    if a == b:
        return True
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    print(f"[불일치] {label}: 개수 {len(a)} vs {len(b)} · 첫 차이 #{i + 1}", file=sys.stderr)
    x = a[i] if i < len(a) else ("(없음)", "")
    y = b[i] if i < len(b) else ("(없음)", "")
    diff = difflib.unified_diff(f"[{x[0]}]\n{x[1]}".splitlines(), f"[{y[0]}]\n{y[1]}".splitlines(),
                                "왼쪽", "오른쪽", lineterm="", n=2)
    for line in list(diff)[:60]:
        print(line, file=sys.stderr)
    return False


def cmd_compare(nb_path: str) -> int:
    """정적 동등성: ① src 셀 == 노트북 셀 ② 노트북 제목으로 9장·제외 키를 뺀 코드셀 == 실행 계획."""
    try:
        steps, _ = build_plan()
        nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    except PlanError as e:
        print(f"[계획 실패] {e}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print(f"노트북을 읽을 수 없다: {type(e).__name__}", file=sys.stderr)
        return 2

    def src(c):
        s = c.get("source", "")
        return "".join(s) if isinstance(s, list) else str(s)

    nb_cells = [(c.get("cell_type"), src(c).rstrip()) for c in nb.get("cells", [])]
    src_cells = [(c["type"], c["text"].rstrip()) for p in sorted(SRC_DIR.glob("s[0-9][0-9]_*.py"))
                 for c in read_cells(p)]
    if not _same("① src 셀 ↔ 노트북 셀", src_cells, nb_cells):
        return 1
    drop_ch = {int(stem[1:3]) for stem, spec in EXCLUDE.items() if "*" in spec}
    drop_key = {(int(stem[1:3]), k) for stem, spec in EXCLUDE.items() for k in spec if k != "*"}
    kept, chapter, key = [], None, None
    for t, s in nb_cells:
        if t == "markdown":
            head = s.split("\n", 1)[0].strip()
            mc, ms = RE_CHAPTER.match(head), RE_SECTION.match(head)
            if mc:
                chapter, key = int(mc.group(1)), None
            elif ms:
                num = RE_SEC_NUM.match(ms.group(1))
                key = num.group(1) if num else ms.group(1)
        elif t == "code" and chapter not in drop_ch and (chapter, key) not in drop_key:
            kept.append(("code", s))
    plan_cells = [("code", c["code"].rstrip()) for s in steps for c in s["_cells"]]
    if not _same("② 노트북(제외 후) ↔ 실행 계획", kept, plan_cells):
        return 1
    print(f"노트북 동등성 OK — ① 셀 {len(src_cells)}개 일치 · ② 코드셀 {len(plan_cells)}개 일치 "
          f"({N_PLAN_STEPS}단계)")
    return 0


def cmd_pack(out_arg) -> int:
    """결과 zip — 예측·그림·표·단계 기록(+FULL 번들)과 지금 시점 provenance·버전."""
    idx = read_index()
    if not idx or idx.get("status") != "ok":
        print(f"[묶기 실패] outputs/steps/index.json 이 status=ok 가 아니다 ({(idx or {}).get('status', '없음')})"
              " — 먼저 python run_all.py 를 끝까지 실행하라", file=sys.stderr)
        return 2
    mode = idx.get("mode")
    # 기본 이름은 모드별 — FAST 결과가 FULL 결과 zip 이름으로 나가거나 그것을 덮지 않게 한다
    default = Path.home() / ("kamp_results_fast.zip" if mode == "FAST" else "kamp_results_full.zip")
    out = (Path(out_arg).expanduser() if out_arg else default).resolve()
    o = ROOT / "outputs"
    cands = [o / "predictions_test_336h.csv", o / "figures" / "figure_index.csv", *o.glob("figures/*.png"),
             *o.glob("tables/*.csv"), *[p for p in (o / "steps").glob("*") if p.name != "README.md"]]
    if mode == "FULL":
        cands += [p for role in ("eval", "deploy") for p in (o / "models" / "full" / role).rglob("*")]
    members = set()
    for p in cands:
        if not p.is_file():
            continue
        r = p.relative_to(ROOT).as_posix()
        if any(x.startswith(".") or x == "__pycache__" for x in r.split("/")) or r.endswith(".pyc"):
            continue
        members.add(r)
    ver = versions()
    virtual = {"_provenance/code_sha256.txt": provenance()[1],
               "_provenance/versions.txt": f"# run_all 결과 · mode={mode} · {platform.platform()}\n"
                                           + "".join(f"{k}=={v or '(없음)'}\n" for k, v in ver.items())}
    entries = sorted(members | set(virtual))
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
        for r in entries:
            if r in virtual:
                z.writestr(r, virtual[r].encode("utf-8"))
            else:
                z.write(ROOT / r, r)
    os.replace(part, out)
    try:
        shown = "~/" + out.relative_to(Path.home().resolve()).as_posix()
    except ValueError:
        shown = out.name
    print(f"결과 zip: {shown} · 파일 {len(entries)}개 · {out.stat().st_size / 1e6:.2f} MB (모드 {mode})")
    return 0


# ══════════════════════════════════════════════════════════════════════
# 진입점 — 환경을 고정해 자기 자신을 다시 실행한다
# ══════════════════════════════════════════════════════════════════════
def reexec(args) -> int:
    """PYTHONHASHSEED 등은 인터프리터 시작 때만 먹으므로 고정 환경으로 자신을 다시 띄운다.

    모드는 플래그로만 정한다(셸에 남은 KAMP_FAST·KAMP_SKIP_BUNDLE 은 무시).
    서빙 우회 변수(KAMP_ALLOW_VERSION_MISMATCH·KAMP_HOLIDAYS_FILE)도 빼서 S.1~S.3 이 버전 핀 검사와
    동봉 휴일표(serving/config/holidays_kr.json)로 검증하게 한다.
    """
    env = dict(os.environ)
    env.update(CHILD_ENV)
    env["KAMP_FAST"] = "1" if args.fast else "0"
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    if args.skip_bundle:
        env["KAMP_SKIP_BUNDLE"] = "1"
    else:
        env.pop("KAMP_SKIP_BUNDLE", None)
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTHONSAFEPATH", None)
    for k in ("KAMP_ALLOW_VERSION_MISMATCH", "KAMP_HOLIDAYS_FILE"):
        if env.pop(k, None) is not None:
            print(f"※ 셸의 {k} 는 이 실행에서 쓰지 않는다 — 서빙 검증은 기본 설정(버전 핀 검사·동봉 휴일표)으로 한다")
    cmd = [sys.executable, "-X", "utf8", "-u", "-B", str(ROOT / "run_all.py"), *sys.argv[1:]]
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "posix":
        os.execve(sys.executable, cmd, env)      # PID 유지 — nohup·kill 친화
    proc = subprocess.Popen(cmd, env=env)
    rc = None
    while rc is None:
        try:
            rc = proc.wait()
        except KeyboardInterrupt:   # 자식도 Ctrl+C 를 받는다 — 기록을 마치고 끝날 때까지 기다린다
            pass
    return rc


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python run_all.py",
                                 description="KAMP 자원최적화 — src 를 노트북과 같은 방식으로 실행하고 outputs/steps/ 에 기록한다")
    ap.add_argument("--fast", action="store_true", help="FAST(KAMP_FAST=1) 동작 확인용 — 수치는 최종값 아님")
    ap.add_argument("--pause", choices=["none", "chapter", "section"], default="none",
                    help="chapter = 장이 끝날 때마다, section = 단계마다 Enter 대기 (기본 none)")
    ap.add_argument("--skip-serving", action="store_true", help="서빙 점검 S.1~S.4 생략")
    ap.add_argument("--no-tests", action="store_true", help="서빙 단위테스트 S.4 생략")
    ap.add_argument("--skip-bundle", action="store_true", help="KAMP_SKIP_BUNDLE=1 (10.5절 번들 생략 → S.1~S.3 생략)")
    ap.add_argument("--check-only", action="store_true", help="파이프라인 없이 기존 outputs 로 S.1~S.4 · C 만")
    ap.add_argument("--export-analysis", metavar="SNAPSHOT_DIR", default=None,
                    help="전체 실행 성공 후 원행 OOF·test·피처·그림 원자료를 새 폴더에 저장 (--check-only 불가)")
    ap.add_argument("--list", action="store_true", help="실행 계획만 출력(쓰기 없음)")
    ap.add_argument("--cpu", action="store_true", help='GPU 를 쓰지 않는다(CUDA_VISIBLE_DEVICES="")')
    ap.add_argument("--force", action="store_true", help="FULL 결과 덮어쓰기·잠금 가드를 무시")
    ap.add_argument("--pack-results", action="store_true", help="결과 zip 만들기 (index.json status=ok 필요)")
    ap.add_argument("--out", metavar="PATH", default=None,
                    help="--pack-results 출력 경로 (기본 ~/kamp_results_full.zip · FAST 결과면 ~/kamp_results_fast.zip)")
    ap.add_argument("--compare-notebook", metavar="PATH", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.out and not args.pack_results:
        ap.error("--out 은 --pack-results 와 함께 쓴다")
    if args.export_analysis and (args.check_only or args.list or args.pack_results or args.compare_notebook):
        ap.error("--export-analysis 는 실제 파이프라인 실행에만 사용합니다 (--check-only/--list/--pack-results 불가)")
    return args


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")    # 한글을 못 쓰는 콘솔에서도 죽지 않게
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    if args.list:
        return cmd_list()
    if args.compare_notebook:
        return cmd_compare(args.compare_notebook)
    if args.pack_results:
        return cmd_pack(args.out)
    if os.environ.get("KAMP_RUNALL_CHILD") != "1":
        return reexec(args)
    return Runner(args).main()


if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)
