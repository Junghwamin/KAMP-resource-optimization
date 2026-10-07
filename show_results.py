"""저장된 실행 결과를 보고서 순서(요약 + 24쪽)로 넘겨 보는 읽기 전용 뷰어.

    python show_results.py                 # 대화형 (Enter 다음 · p 이전 · 2.6 이동 · l 목록 · t 표 · f 그림 · o 로그 · q 끝)
    python show_results.py --all           # 모든 쪽을 한 번에 출력
    python show_results.py --section 2.6   # 한 쪽만
    python show_results.py --table 2-2     # 표 보기 (이름 · 접두 · 보고서 표 번호 · pred · F18)
    python show_results.py --figures       # 그림 목록
    python show_results.py --list          # 쪽 목록

읽기만 한다: 모델·src·serving 을 import 하지 않고 outputs/ 에 아무것도 쓰지 않는다.
outputs/steps/index.json 이 없으면(아직 run_all.py 를 돌리지 않았으면) 표·그림 존재 여부와 표 보기만 하는 정적 모드로 동작한다.
"""
from __future__ import annotations

import argparse
import codecs
import json
import os
import pydoc
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
STEPS_DIR = OUT / "steps"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
PRED = OUT / "predictions_test_336h.csv"
FIG_INDEX = FIGS / "figure_index.csv"

# ── 쪽 구성 (PLAN §5): (쪽 id, 보고서 제목, 주 단계, 참고 단계, 자산[(보고서 번호, 종류, 이름)])
#    종류: table = outputs/tables/이름.csv · fig = 그림 fid · pred = 예측 파일 · figidx = figure_index.csv
PAGES = [
    ("1.1", "1.1 분석 배경 및 문제 정의", ["1.6"], [], []),
    ("1.2", "1.2 데이터 구성 및 주요 변수", ["1.1"], [], [("", "table", "ch1_variable_dict")]),
    ("1.3", "1.3 제조공정 상태 및 탐색적 분석(EDA)", ["1.5"], [], [("그림1", "fig", "F05")]),
    ("1.4", "1.4 데이터 품질 진단 및 처리", ["1.2", "1.3", "1.G"], [],
     [("", "table", "ch1_quality"), ("그림2", "fig", "F02")]),
    ("1.5", "1.5 전처리 및 파생변수 구성", ["1.4", "2.1", "2.4"], [], []),
    ("1.6", "1.6 학습·검증 데이터 구성 및 한계", ["3.1"], ["1.6"],
     [("표1-1", "table", "ch3_folds"), ("그림3", "fig", "F14")]),
    ("2.1", "2.1 실험 설계", ["2.2", "3.2", "3.3", "3.5", "5.0"], ["2.4", "3.1"], []),
    ("2.2", "2.2 가이드북 베이스라인 재현", ["4.1", "4.2", "4.3", "4.4", "4.5", "4.G", "5.3"], ["0.6"],
     [("표2-1", "table", "ch4_defects"), ("", "table", "ch4_rf_original"), ("", "table", "ch4_corrected"),
      ("", "table", "ch4_naive")]),
    ("2.3", "2.3 입력변수 구성", ["2.3"], ["2.1", "2.2", "2.4", "3.5"], [("", "table", "ch2_feature_groups")]),
    ("2.4", "2.4 제안모델 구성", ["5.2", "5.4", "5.5", "5.6"], ["5.1", "5.3"], []),
    ("2.5", "2.5 학습 및 최적화 방법", ["5.1"], ["5.4"], [("", "table", "hpo_trials"), ("", "fig", "F26")]),
    ("2.6", "2.6 전력사용량 예측 성능", ["6.1", "6.5"], ["3.3", "4.1", "4.5", "5.0", "5.5"],
     [("표2-2", "table", "ch2_regression"), ("표2-3", "table", "ch5_fold_mae_matrix"),
      ("", "table", "ch2_significance"),
      ("그림4", "fig", "F18"), ("그림5", "fig", "F24")]),
    ("2.7", "2.7 피크 위험 탐지 성능", ["3.4", "6.2"], ["3.3", "6.5"],
     [("표2-4", "table", "ch2_peak_detection"), ("표2-5", "table", "ch2_peak_detection_oof"),
      ("", "table", "ch2_peak_threshold_temporal"), ("그림6", "fig", "F19")]),
    ("2.8", "2.8 변수 제거 실험(Ablation)", ["6.3", "6.6"], [],
     [("", "table", "ch2_ablation"), ("", "table", "ch2_plan_information_ablation"),
      ("그림7", "fig", "F20"), ("표2-6", "table", "ch2_lag_decomposition")]),
    ("2.9", "2.9 최종모델 선정", ["6.4", "6.G"], ["6.5", "6.6"], [("표2-7", "table", "ch2_scorecard")]),
    ("3.1", "3.1 주요 영향변수 분석", ["7.0", "7.1"], [], [("표3-1", "table", "ch3_importance"), ("그림8", "fig", "F27")]),
    ("3.2", "3.2 주요 변수의 상호작용", ["7.2"], [], [("", "table", "ch3_interaction")]),
    ("3.3", "3.3 조건별 예측오차 분석", ["7.3"], [], [("", "table", "ch3_condition_mae"), ("그림9", "fig", "F29")]),
    ("3.4", "3.4 피크 미탐지와 오경보 분석", ["7.4"], [], [("", "table", "ch3_confusion"), ("그림10", "fig", "F30")]),
    ("3.5", "3.5 대표 실패사례 분석", ["7.5"], ["6.6"], [("", "table", "ch3_failures"), ("그림11", "fig", "F33")]),
    ("3.6", "3.6 최대수요 피크 발생조건 도출", ["7.6"], [], [("", "table", "ch3_rules"), ("그림12", "fig", "F31")]),
    ("4", "4장 현장 활용방안", ["8.1", "8.2", "8.3", "8.4", "S.3"], ["10.5"],
     [("표4-1", "table", "ch4_levers"), ("그림13", "fig", "F35"), ("표4-2", "table", "ch4_scenarios"),
      ("", "table", "ch4_protocol")]),
    ("5", "5장 창의성 및 차별성", ["5.7", "5.8"], [],
     [("그림14", "fig", "F21"), ("", "table", "ch5_calibration_folds"), ("", "table", "ch5_uncertainty")]),
    ("6", "6장 코드 구성 및 재현성",
     ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "10.1", "10.5", "S.1", "S.2", "S.4", "C"], ["S.3"],
     [("", "table", "env_versions"), ("", "pred", ""), ("", "table", "ch6_model_bundles"), ("", "figidx", "")]),
]
assert len(PAGES) == 24, "보고서 쪽은 24개여야 한다"

# 보고서 표·그림 번호 → 파일 (위 PAGES 에서 만든다: 표 번호 '2-2' → ch2_regression, 그림 '4' → F18)
TABLE_NO = {lab[1:]: name for p in PAGES for lab, kind, name in p[4] if kind == "table" and lab}
FIG_NO = {lab[2:]: name for p in PAGES for lab, kind, name in p[4] if kind == "fig" and lab}
FIG_LABEL = {fid: "그림" + no for no, fid in FIG_NO.items()}

# 서빙·완료 점검 단계 제목 (정적 모드용 — 실행 뒤에는 index.json 의 제목을 쓴다)
EXTRA_TITLES = {"S.1": "S.1 서빙 번들 검증 (verify)", "S.2": "S.2 리플레이 비트 동일 확인 (replay)",
                "S.3": "S.3 CLI 예측 (predict)", "S.4": "S.4 서빙 단위테스트 (pytest)", "C": "C 완료 점검"}

REPORT_FINAL = "2단계 레짐(3분류)"
TABLES_EXPECTED, PNG_EXPECTED = 110, 39
OK, NG, ELL = "✓", "✗", "…"
# 출력 인코딩(cp949 등)에 없는 기호의 대체 문자 — 그 밖의 문자는 ? 로 바꾼다
_FALLBACK = {"✓": "O", "✗": "X", "…": "~", "—": "-", "■": "*", "Δ": "d"}


# ── 출력 보조 ─────────────────────────────────────────────────────────────
def _replace_err(e):
    return "".join(_FALLBACK.get(c, "?") for c in e.object[e.start:e.end]), e.end


def _setup_stdio():
    """cp949 콘솔·리다이렉트에서도 죽지 않게: 인코딩할 수 없는 문자는 대체 문자로 바꿔 쓴다."""
    codecs.register_error("show_results_replace", _replace_err)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="show_results_replace")
        except (AttributeError, ValueError):
            pass


def _dw(ch):
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def clip(s, width):
    """표시 폭(한글 2칸) 기준으로 잘라 한 줄에 맞춘다."""
    s = s.expandtabs(4)
    if sum(_dw(c) for c in s) <= width:
        return s
    w, out = 0, []
    for c in s:
        w += _dw(c)
        if w > width - 1:
            break
        out.append(c)
    return "".join(out) + ELL


def fmt_sec(v):
    if v is None:
        return "-"
    v = float(v)
    return f"{v:.1f}초" if v < 60 else f"{int(v // 60)}분 {v % 60:.0f}초"


def _ts(s):
    return str(s)[:19].replace("T", " ") if s else "-"


def _mark(ok):
    return OK if ok else NG


def _rel(p):
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def _pd_opts(width, max_rows=None):
    import pandas as pd
    return pd.option_context("display.unicode.east_asian_width", True, "display.width", width,
                             "display.max_columns", None, "display.max_colwidth", 40,
                             "display.max_rows", max_rows)


def _read_csv(path):
    import pandas as pd
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None


# ── 입력 읽기 ─────────────────────────────────────────────────────────────
def load_index():
    """index.json → (dict | None, 경고 문자열)."""
    p = STEPS_DIR / "index.json"
    if not p.exists():
        return None, ""
    try:
        return json.loads(p.read_text(encoding="utf-8-sig")), ""
    except Exception as e:
        return None, f"index.json 을 읽지 못했습니다({type(e).__name__}) — 정적 모드로 봅니다"


def run_was_cut(idx) -> bool:
    """status=running 인데 도는 run_all 이 없다 — 서버 중지·강제 종료로 기록이 중간에 끊겼다.

    run_all 은 잠금 파일을 index 보다 먼저 쓰고 마지막 기록 뒤에 지운다. 그래서 잠금이 없으면 끊긴 기록이다.
    잠금 PID 가 살아 있는지는 POSIX 에서만 본다(Windows 의 os.kill 은 그 프로세스를 끝내 버린다).
    """
    if not isinstance(idx, dict) or idx.get("status") != "running":
        return False
    try:
        pid = int((STEPS_DIR / ".run_all.lock").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return True
    if os.name != "posix":
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.is_file():            # 리눅스: 그 PID 가 다른 프로그램에 다시 쓰였는지도 본다
        try:
            return b"run_all.py" not in cmdline.read_bytes()
        except OSError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:                  # 권한 없음 등 — 살아 있다고 본다
        return False
    return False


def scan_src_titles():
    """정적 모드용: src/*.py 의 마크다운 제목을 텍스트로만 읽어 단계 id → (제목, 파일, 줄)."""
    titles = {}
    for f in sorted((ROOT / "src").glob("s[0-9][0-9]_*.py")):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for no, line in enumerate(lines, 1):
            m = re.match(r"# ## (\d+)\.\s*(.*)", line)
            if m:      # 장 머리 → 도입 코드 N.0 (### N.0 이 따로 있으면 그것이 덮는다)
                titles.setdefault(f"{m.group(1)}.0", (f"{m.group(1)}.0 {m.group(2).strip()} (도입)", f"src/{f.name}", no))
                continue
            m = re.match(r"# ### (\d+\.\d+)\s*(.*)", line)
            if m:
                titles[m.group(1)] = (f"{m.group(1)} {m.group(2).strip()}", f"src/{f.name}", no)
                continue
            m = re.match(r"# ### (\d+)장 게이트\s*(.*)", line)
            if m:
                titles[f"{m.group(1)}.G"] = (f"{m.group(1)}.G {m.group(1)}장 게이트 {m.group(2).strip()}".rstrip(),
                                             f"src/{f.name}", no)
    return titles


def list_tables():
    return sorted(p.stem for p in TABLES.glob("*.csv")) if TABLES.exists() else []


def figure_rows():
    """figure_index.csv 의 행 목록 [{fid, 파일명, 보고서절, 제목}] (없으면 PNG 파일명으로 대신)."""
    df = _read_csv(FIG_INDEX) if FIG_INDEX.exists() else None
    if df is not None and "fid" in df.columns:
        return [{k: str(r.get(k, "")) for k in ("fid", "파일명", "보고서절", "제목")} for r in df.to_dict("records")]
    return [{"fid": p.name.split("_")[0], "파일명": p.name, "보고서절": "", "제목": ""}
            for p in sorted(FIGS.glob("*.png"))] if FIGS.exists() else []


class View:
    def __init__(self, args):
        self.idx, self.warn = load_index()
        self.cut = run_was_cut(self.idx)
        self.steps = {s.get("id"): s for s in (self.idx or {}).get("steps") or [] if isinstance(s, dict)}
        self.titles = scan_src_titles()
        self.width = args.width or (shutil.get_terminal_size((160, 24)).columns if sys.stdout.isatty() else 160)
        self.width = max(60, self.width)
        self.lines, self.max_rows = max(0, args.lines), max(1, args.max_rows)
        self.figs = {r["fid"]: r for r in figure_rows()}

    # ── 쪽 머리 ──
    def head(self, i, title):
        bar = "=" * min(self.width, 100)
        L = [bar, f"[{i}/24] {title}", bar]
        if self.idx is None:
            L.append("  (정적 모드: 실행 기록 없음 — python run_all.py 로 실행하면 단계 로그가 함께 보입니다)")
        else:
            if str(self.idx.get("mode")) == "FAST":
                L.append("  !! FAST 결과 — 동작 확인용 축소 실행입니다. 수치는 보고서 값이 아닙니다 (FULL: python run_all.py) !!")
            st = self.idx.get("status")
            if st != "ok":
                cur = self.idx.get("current")
                if self.cut:
                    L.append("  !! 실행이 중간에 끊긴 기록입니다 (status=running 이지만 도는 run_all 이 없음"
                             + (f" · 마지막 단계 {cur}" if cur else "") + ") — python run_all.py 로 다시 실행하세요 !!")
                else:
                    L.append(f"  !! 실행 상태: {st}" + (f" (현재/마지막 단계 {cur})" if cur else "") + " !!")
        return L + [""]

    # ── 단계 한 개 ──
    def step_title(self, sid):
        s = self.steps.get(sid)
        if s:
            t = str(s.get("title") or sid)
            return t if t.startswith(sid) else f"{sid} {t}"
        return (self.titles.get(sid) or (EXTRA_TITLES.get(sid, sid), "", 0))[0]

    def step_loc(self, sid):
        s = self.steps.get(sid)
        if s and s.get("file"):
            ln = s.get("lines") or []
            return f"{s['file']}:{ln[0]}-{ln[-1]}" if ln else str(s["file"])
        t = self.titles.get(sid)
        return f"{t[1]}:{t[2]}" if t else ""

    def log_path(self, s):
        if s.get("log"):
            return ROOT / s["log"]
        return STEPS_DIR / f"{int(s.get('seq', 0)):02d}_{s.get('id')}.txt"

    def read_log(self, s):
        p = self.log_path(s)
        try:
            return p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None

    def step_block(self, sid):
        loc = self.step_loc(sid)
        L = [f"■ 근거 코드: {self.step_title(sid)}" + (f" ({loc})" if loc else "")]
        s = self.steps.get(sid)
        if s is None:
            L.append("    (실행 기록 없음)" if self.idx is None else "    (index.json 에 이 단계가 없습니다)")
            return L + [""]
        dev = f" (개발 PC 약 {fmt_sec(s['dev_sec'])})" if s.get("dev_sec") else ""
        L.append(f"    상태 {s.get('status', '-')} · {fmt_sec(s.get('seconds'))}{dev} · 로그 {_rel(self.log_path(s))}")
        if s.get("error"):
            L += ["    오류: " + clip(line, self.width - 10) for line in _err_lines(s["error"])]
        log = self.read_log(s)
        if log is None:
            L.append("    (로그 파일 없음)")
        elif self.lines:
            L += ["    | " + clip(x, self.width - 6) for x in log[: self.lines]]
            if len(log) > self.lines:
                L.append(f"    | {ELL} 이하 {len(log) - self.lines}줄 생략 (대화형 o 로 전체 보기)")
        for key, name in (("tables", "표"), ("figures", "그림"), ("models", "모델"), ("other", "기타")):
            got = (s.get("files") or {}).get(key) or []
            if got:
                # 표·그림은 파일 이름만, 모델·기타는 이름이 겹치므로(model.joblib 등) outputs/ 아래 경로로
                names = [Path(str(x)).name if key in ("tables", "figures") else re.sub(r"^outputs/", "", str(x).replace("\\", "/"))
                         for x in got]
                more = f" 외 {len(names) - 12}개" if len(names) > 12 else ""
                L.append(clip(f"    만든 {name} {len(names)}개: " + ", ".join(names[:12]) + more, self.width))
        return L + [""]

    # ── 자산 ──
    def asset_line(self, lab, kind, name):
        if kind == "table":
            p = TABLES / f"{name}.csv"
            key = lab[1:] if lab else name
            return f"  {lab or '표'} ← {_rel(p)}  {_mark(p.exists())}   (t {key} 로 전체 보기)"
        if kind == "fig":
            row = self.figs.get(name)
            p = FIGS / row["파일명"] if row else next(iter(sorted(FIGS.glob(f"{name}_*.png"))), FIGS / f"{name}.png")
            title = f" — {row['제목']}" if row and row.get("제목") else ""
            return f"  {lab or '그림'}({name}) ← {_rel(p)}  {_mark(p.exists())}{title}   (원자료 t {name})"
        if kind == "pred":
            return f"  예측 파일 ← {_rel(PRED)}  {_mark(PRED.exists())}   (t pred 로 보기)"
        return f"  그림 색인 ← {_rel(FIG_INDEX)}  {_mark(FIG_INDEX.exists())}   (f 로 목록 보기)"

    def ref_line(self, sid):
        s = self.steps.get(sid)
        st = f" — {s.get('status', '-')} {fmt_sec(s.get('seconds'))}" if s else ""
        loc = self.step_loc(sid)
        return clip(f"  · {self.step_title(sid)}{st}" + (f" ({loc})" if loc else ""), self.width)

    # ── 보고서 쪽 ──
    def page(self, i):
        if i == 0:
            return "\n".join(self.summary())
        pid, title, mains, refs, assets = PAGES[i - 1]
        L = self.head(i, title)
        for sid in mains:
            L += self.step_block(sid)
        if pid == "5":
            L.append("  ※ 9장(s09 창의성 산출)은 보고서 생성용이라 이 패키지 실행에서 제외되었습니다.")
            for e in (self.idx or {}).get("excluded") or []:
                if isinstance(e, dict) and str(e.get("file", "")).startswith("s09"):
                    L.append(clip(f"    - {e.get('file')} {e.get('key')} {e.get('title', '')}", self.width))
            L.append("")
        if pid == "6" and self.idx:
            vers = self.idx.get("versions") or {}
            L.append(clip("  실행 환경: python " + str(self.idx.get("python", "-")) + " · " + " · ".join(
                f"{k} {v}" for k, v in vers.items() if v and k != "python"), self.width))
            code = (self.idx.get("provenance") or {}).get("code_sha256")
            if code:
                L.append(f"  코드 지문(code_sha256): {str(code)[:16]}{ELL}")
            L.append("")
        if assets:
            L.append("[보고서 표·그림 ← 파일]")
            L += [self.asset_line(*a) for a in assets] + [""]
        if refs:
            L.append("[참고 단계]")
            L += [self.ref_line(sid) for sid in refs] + [""]
        return "\n".join(L)

    # ── 요약 쪽 ──
    def summary(self):
        L = self.head(0, "요약")
        idx = self.idx
        if self.warn:
            L.append("  " + self.warn)
        if idx is None:
            L += ["  아직 실행하지 않았습니다 — python run_all.py", "  지금은 저장된 표·그림의 존재 여부와 표 보기만 할 수 있습니다.", ""]
        else:
            bad = [s for s in self.steps.values() if s.get("status") == "failed"]
            for s in bad:     # 실패 단계는 맨 위에
                L.append(f"  !! 실패 단계: {self.step_title(s.get('id'))} ({self.step_loc(s.get('id'))})")
                L += ["     " + clip(x, self.width - 6) for x in _err_lines(s.get("error"))]
            if bad:
                L.append("")
            cnt = Counter(str(s.get("status")) for s in self.steps.values())
            flags = [k for k, v in (idx.get("flags") or {}).items() if v not in (False, None, "none")]
            L += [f"  모드 {idx.get('mode', '-')} · 상태 {idx.get('status', '-')} · 시작 {_ts(idx.get('started'))} · "
                  f"종료 {_ts(idx.get('finished'))} · 총 {fmt_sec(idx.get('total_sec'))}",
                  "  단계 " + " · ".join(f"{k} {v}" for k, v in sorted(cnt.items())) + f" (총 {len(self.steps)})"
                  + (f" · 옵션 {', '.join(flags)}" if flags else "")]
            exc = [e for e in idx.get("excluded") or [] if isinstance(e, dict)]
            if exc:
                L.append("  제외 절: " + clip(", ".join(f"{e.get('file', '')} {e.get('key', '')}" for e in exc), self.width - 12))
            L += [""] + self.check_lines() + self.serving_lines()
            slow = sorted(self.steps.values(), key=lambda s: -float(s.get("seconds") or 0))[:5]
            L.append("[가장 느린 5단계]")
            L += [clip(f"  {fmt_sec(s.get('seconds')):>10}  {self.step_title(s.get('id'))}", self.width) for s in slow]
            L.append("")
        n_tab, n_png = len(list_tables()), len(list(FIGS.glob("*.png"))) if FIGS.exists() else 0
        L += [f"  산출물: 표 {n_tab}개 (기대 {TABLES_EXPECTED}개 이상) · 그림 PNG {n_png}장 (기대 {PNG_EXPECTED}) · "
              f"예측 파일 {_mark(PRED.exists())} · 그림 색인 {_mark(FIG_INDEX.exists())}", ""]
        L.append("[헤드라인 수치 — outputs/tables 에서 읽어 보고서 값과 대조]")
        L += self.headline() + [""]
        L.append("  이동: Enter/n 다음 · p 이전 · 2.6 같은 절 번호 · l 쪽 목록 · t 이름 [N] 표 · f 그림 · o 로그 · q 끝")
        return L

    def check_lines(self):
        ch = (self.idx or {}).get("checks") or {}
        if not any(ch.get(g) for g in ("hard", "warn", "info")):
            return ["[완료 점검] 아직 기록 없음", ""]
        L = ["[완료 점검]"]
        for grade in ("hard", "warn", "info"):
            items = ch.get(grade) or []
            if not items:
                continue
            oks = [it for it in items if isinstance(it, dict) and it.get("ok") is True]
            L.append(f"  {grade}: {len(oks)}/{len(items)} 통과" if grade != "info" else f"  info: {len(items)}건")
            for it in items:
                if not isinstance(it, dict):
                    L.append(clip(f"    - {it}", self.width))
                elif grade == "info" or it.get("ok") is not True:
                    L.append(clip(f"    {_mark(it.get('ok') is True) if grade != 'info' else '-'} "
                                  f"{it.get('name', '')}: {it.get('detail', '')}", self.width))
        return L + [""]

    def serving_lines(self):
        sv = (self.idx or {}).get("serving") or {}
        L = ["[서빙]"]
        for k in ("S.1", "S.2", "S.3", "S.4"):
            s = self.steps.get(k) or {}
            d = sv.get(k)
            body = ", ".join(f"{a}={json.dumps(b, ensure_ascii=False) if isinstance(b, (dict, list)) else b}"
                             for a, b in d.items()) if isinstance(d, dict) else (str(d) if d else "")
            L.append(clip(f"  {k} {s.get('status', '-')}" + (f" · {body}" if body else ""), self.width))
        return L + [""]

    def headline(self):
        """최신 수치를 표시하고 수정 전 보고서와의 차이를 별도로 보여 준다."""
        import pandas as pd
        reg, sig, peak = _tab("ch2_regression"), _tab("ch2_significance"), _tab("ch2_peak_detection")
        sc, scen = _tab("ch2_scorecard"), _tab("ch4_scenarios")
        final, how = None, ""
        pred = _read_csv(PRED) if PRED.exists() else None
        if pred is not None and "model_name" in pred.columns and pred["model_name"].nunique() == 1:
            final, how = str(pred["model_name"].iloc[0]), "예측 파일 model_name"
        if sc is not None and {"모델", "종합점수"} <= set(sc.columns):
            top = str(sc.loc[pd.to_numeric(sc["종합점수"], errors="coerce").idxmax(), "모델"])
            if final is None:
                final, how = top, "ch2_scorecard 종합점수 1위"
            elif top != final:
                how += f" (스코어카드 1위 {top} 와 다름)"
            else:
                how += " = 스코어카드 1위"

        def pick(df, col, key, fmt, starts=False):
            try:
                keys = df.iloc[:, 0] if starts else df["모델"]
                row = df[keys.astype(str).str.startswith(key)] if starts else df[keys.astype(str) == key]
                return fmt.format(float(row[col].iloc[0]))
            except Exception:
                return "-"

        kw = next((c for c in (scen.columns if scen is not None else []) if "최대수요" in c), "")
        rows = [("최종모델", final or "-", REPORT_FINAL, how or "-"),
                ("MAE (최종모델)", pick(reg, "MAE", final, "{:.3f}"), "5.220", "ch2_regression"),
                ("MAE (RF 보정)", pick(reg, "MAE", "Random Forest (보정)", "{:.3f}"), "6.039", "ch2_regression"),
                ("개선율 %", pick(sig, "개선율(%)", final, "{:.2f}"), "13.56", "ch2_significance"),
                ("p-value", pick(sig, "p-value", final, "{:.3f}"), "0.067", "ch2_significance"),
                ("Recall", pick(peak, "Recall", final, "{:.3f}"), "0.929", "ch2_peak_detection"),
                ("F1", pick(peak, "F1", final, "{:.3f}"), "0.525", "ch2_peak_detection"),
                ("권고안 S2 ΔkW", pick(scen, kw, "S2 ", "{:.1f}", True), "13.1", "ch4_scenarios"),
                ("S4 ΔkW", pick(scen, kw, "S4 ", "{:.1f}", True), "13.2", "ch4_scenarios")]
        if sig is not None and final is not None:
            selected = sig.loc[sig["모델"] == final]
            if len(selected):
                for label, column in (("95% CI(주분석)", "차이 95% CI"),
                                      ("90% CI(사후 보조)", "차이 90% CI(사후 탐색)")):
                    rows.append((label, str(selected.iloc[0].get(column, "-")), "-", "ch2_significance"))
        df = pd.DataFrame([{"항목": a, "현재 실행 결과": b, "수정 전 보고서": c,
                            "비교": "미생성" if b == "-" else "신규" if c == "-" else "동일" if b == c else "보고서 갱신 필요",
                            "출처": d} for a, b, c, d in rows])
        with _pd_opts(self.width):
            return ["  " + x for x in df.to_string(index=False).splitlines()]

    # ── 목록·표·그림·로그 ──
    def page_list(self):
        L = ["쪽 목록 (절 번호를 입력하면 이동)", "   0  요약"]
        for i, (pid, title, mains, _, _) in enumerate(PAGES, 1):
            sts = [str((self.steps.get(s) or {}).get("status", "")) for s in mains]
            m = "" if self.idx is None else (NG if "failed" in sts else OK if all(x in ("ok", "skipped") for x in sts) else "·")
            L.append(f"  {i:2d}  {m:1s} {title}  (단계 {', '.join(mains)})")
        return "\n".join(L)

    def figures(self):
        import pandas as pd
        rows = [{"fid": fid, "보고서": FIG_LABEL.get(fid, ""), "코드 절": r.get("보고서절", ""),
                 "있음": _mark((FIGS / r["파일명"]).exists()), "파일": f"outputs/figures/{r['파일명']}", "제목": r.get("제목", "")}
                for fid, r in sorted(self.figs.items())]
        if not rows:
            return "그림이 없습니다 (outputs/figures/)"
        with _pd_opts(self.width):
            return f"그림 {len(rows)}개 (figure_index.csv 기준, '코드 절'은 src 절 번호)\n" + \
                pd.DataFrame(rows).to_string(index=False) + "\n원자료 표는 t F18 처럼 fid 로 봅니다."

    def table(self, query, n=None):
        path, msg = resolve_table(query)
        if path is None:
            return msg
        df = _read_csv(path) if path.exists() else None
        if df is None:
            return f"{'읽을 수 없습니다' if path.exists() else '파일이 없습니다'}: {_rel(path)}"
        n = n or self.max_rows
        head = f"표: {_rel(path)} ({len(df)}행 × {len(df.columns)}열)" + (f" — {msg}" if msg else "")
        with _pd_opts(self.width):
            body = str(df.head(n)) if len(df) else "(빈 표)"
        tail = f"\n{ELL} 이하 {len(df) - n}행 생략 (t 이름 N 또는 --max-rows N)" if len(df) > n else ""
        return f"{head}\n{body}{tail}"

    def full_logs(self, i):
        sids = ["C"] if i == 0 else PAGES[i - 1][2]
        parts = []
        for sid in sids:
            s = self.steps.get(sid)
            log = self.read_log(s) if s else None
            parts.append(f"===== {self.step_title(sid)} =====\n" + ("\n".join(log) if log is not None else "(로그 없음)"))
        return "\n\n".join(parts)


def _tab(name):
    return _read_csv(TABLES / f"{name}.csv") if (TABLES / f"{name}.csv").exists() else None


def _err_lines(err):
    """index 의 error: 문자열 또는 {type, message, where …} 사전 → 줄 목록."""
    if not err:
        return []
    if isinstance(err, dict):
        return [f"{k}: {v}" for k, v in err.items() if v not in (None, "")]
    return str(err).splitlines()[-6:]


def find_page(key):
    k = key.strip().replace("장", "").rstrip(".")
    if k in ("0", "요약"):
        return 0
    return next((i for i, p in enumerate(PAGES, 1) if p[0] == k), None)


def resolve_table(query):
    """표 이름 해석 → (경로 | None, 설명). 정확 > 표 번호 > pred > 그림 fid > 접두 > 부분."""
    q = query.strip()
    q = q[:-4] if q.lower().endswith(".csv") else q
    names = list_tables()
    low = {n.lower(): n for n in names}
    if q.lower() in low:
        return TABLES / f"{low[q.lower()]}.csv", ""
    m = re.fullmatch(r"(?:표\s*)?(\d+-\d+)", q)
    if m:
        name = TABLE_NO.get(m.group(1))
        return (TABLES / f"{name}.csv", f"보고서 표{m.group(1)}") if name else (None, f"보고서 표 번호 {q} 없음: " + ", ".join(TABLE_NO))
    if q.lower() in ("pred", "예측", "predictions", "predictions_test_336h"):
        return (PRED, "제출 예측 파일") if PRED.exists() else (None, "예측 파일이 없습니다")
    if q.lower() in ("figure_index", "그림색인"):
        return (FIG_INDEX, "그림 색인") if FIG_INDEX.exists() else (None, "figure_index.csv 가 없습니다")
    m = re.fullmatch(r"그림\s*(\d+)", q) or re.fullmatch(r"[Ff](\d{1,2})(?:_src)?", q)
    if m:
        fid = FIG_NO.get(m.group(1)) if q.startswith("그림") else f"F{int(m.group(1)):02d}"
        if fid and f"{fid}_src".lower() in low:
            return TABLES / f"{fid}_src.csv", f"{FIG_LABEL.get(fid, fid)} 원자료"
        return None, f"{q} 의 원자료 표가 없습니다"
    for cands in ([n for n in names if n.lower().startswith(q.lower())], [n for n in names if q.lower() in n.lower()]):
        if len(cands) == 1:
            return TABLES / f"{cands[0]}.csv", ""
        if cands:
            return None, f"'{q}' 후보 {len(cands)}개 — 하나를 골라 주세요:\n  " + "\n  ".join(cands[:40])
    return None, f"'{q}' 표를 찾지 못했습니다. 이름·접두·표 번호(2-2)·pred·F18 로 찾습니다 (표 {len(names)}개)"


def handle(cmd, cur, view):
    """대화형 명령 한 개 → (새 쪽, 출력 종류, 텍스트). 종류: page · text · pager · quit."""
    c = cmd.strip()
    if c in ("", "n"):
        return (cur + 1, "page", "") if cur < 24 else (cur, "text", "마지막 쪽입니다 (p 이전 · q 끝)")
    if c == "p":
        return (cur - 1, "page", "") if cur > 0 else (cur, "text", "첫 쪽입니다")
    if c == "q":
        return cur, "quit", ""
    if c == "l":
        return cur, "text", view.page_list()
    if c == "f":
        return cur, "text", view.figures()
    if c == "o":
        return cur, "pager", view.full_logs(cur)
    if c == "t" or c.startswith("t "):
        parts = c.split()[1:]
        if not parts:
            return cur, "text", "표 {}개: ".format(len(list_tables())) + ", ".join(list_tables())
        n = int(parts.pop()) if len(parts) > 1 and parts[-1].isdigit() else None
        return cur, "text", view.table(" ".join(parts), n)
    i = find_page(c)
    if i is not None:
        return i, "page", ""
    return cur, "text", "모르는 명령입니다. Enter/n 다음 · p 이전 · 2.6 이동 · l 목록 · t 이름 [N] · f 그림 · o 로그 · q 끝"


def interactive(view):
    cur = 0
    print(view.page(cur))
    while True:
        try:
            cmd = input(f"\n[{cur}/24] Enter 다음 · p · 절번호 · l · t 이름 · f · o · q > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        cur, kind, text = handle(cmd, cur, view)
        if kind == "quit":
            return
        if kind == "pager":
            try:
                pydoc.pager(text)
            except Exception:
                print(text)
        else:
            print(view.page(cur) if kind == "page" else text)


def main(argv=None):
    _setup_stdio()
    ap = argparse.ArgumentParser(description="저장된 실행 결과를 보고서 순서로 보는 읽기 전용 뷰어")
    ap.add_argument("--all", action="store_true", help="요약 + 24쪽 전체 출력")
    ap.add_argument("--section", action="append", metavar="X", help="한 쪽만 출력 (예 2.6, 4, 0=요약). 여러 번 가능")
    ap.add_argument("--table", action="append", metavar="NAME", help="표 보기 (이름·접두·2-2·pred·F18). 여러 번 가능")
    ap.add_argument("--figures", action="store_true", help="그림 목록")
    ap.add_argument("--list", action="store_true", help="쪽 목록")
    ap.add_argument("--additional", action="store_true", help="추가 실험 6개와 최종 검증의 저장 상태")
    ap.add_argument("--run-dir", type=Path, default=OUT / "additional/validated_run", help="추가 실험 결과 폴더")
    ap.add_argument("--lines", type=int, default=40, help="쪽마다 보여 줄 단계 로그 줄 수 (기본 40)")
    ap.add_argument("--width", type=int, default=0, help="출력 폭 (기본: 터미널 폭, 터미널이 아니면 160)")
    ap.add_argument("--max-rows", type=int, default=60, help="표 보기 최대 행 수 (기본 60)")
    args = ap.parse_args(argv)
    if args.additional:
        print("추가 검증 결과 (저장된 상태를 표시하며 재계산하지 않음)")
        for label, name in (("E1 조건별 오류", "diagnostics"), ("E2 블록 길이", "bootstrap"),
                            ("E3 계획·예보 오차", "sensitivity"), ("E4 같은 설정의 구조 비교", "controlled"),
                            ("E5 시간순 후보선정", "temporal"), ("E6 경보 기반 정책 모의", "policy")):
            path = args.run_dir / name / "executor_manifest.json"
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            print(f"  {label}: {data.get('status', '미실행')} ({name}/summary.json)")
        path = args.run_dir / "verification_summary.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        print(f"최종 로컬 검증: {data.get('status', '미검증')}")
        print("실제 KAMP 실행 여부는 이 저장 상태만으로 입증되지 않습니다. PORTABILITY.md를 확인하세요.")
        return 0
    view = View(args)
    if view.warn:
        print(view.warn)
    rc, did = 0, False
    if args.list:
        print(view.page_list()); did = True
    for key in args.section or []:
        i = find_page(key)
        if i is None:
            print(f"없는 절입니다: {key} (--list 로 목록 확인)"); rc = 2
        else:
            print(view.page(i))
        did = True
    for name in args.table or []:
        print(view.table(name)); did = True
    if args.figures:
        print(view.figures()); did = True
    if did:
        return rc
    if args.all or not (sys.stdin.isatty() and sys.stdout.isatty()):
        if not args.all:
            print("(터미널 입력이 아니므로 --all 처럼 전체 쪽을 출력합니다)")
        print("\n\n".join(view.page(i) for i in range(25)))
        return 0
    interactive(view)
    return 0


if __name__ == "__main__":
    sys.exit(main())
