"""FastAPI 앱 — 익일 24시간 예측 REST API (공장 1곳 · 일일 배치 규모).

실행 (번들 디렉터리는 **outputs 밖에 복사한 불변 사본**을 가리킨다):

    KAMP_BUNDLE_DIR=/srv/kamp/bundles/deploy-full-xxxx \\
      uvicorn --factory serving.api:create_app --workers 1 --host 127.0.0.1 --port 8000

| 엔드포인트 | 내용 |
|---|---|
| `GET /healthz` | 상태·번들 id |
| `GET /v1/model` | 피처 계약·임계값·휴일 커버리지·주의사항 (경로는 싣지 않는다) |
| `POST /v1/predict/day-ahead` | 이력 + 계획 → 24행 예측 |

설계 원칙
- **기동 즉시 로드(fail-fast)** — 번들이 손상·불일치면 앱이 뜨지 않는다. "번들 미로드" 상태가 없다.
- 워커 1개 + `threading.Lock` — 요청이 하루 1~수 회라 동시성이 필요 없다. 리로드는 재시작으로 한다.
- 모든 float 는 유한값·물리 범위 검증(`allow_inf_nan=False`), 모르는 필드는 거부(`extra="forbid"`).
- 오류 봉투는 하나: `{"error": {"code", "message", "detail"?}}`. 입력값을 되돌려 주지 않는다
  (NaN/inf 를 되돌려 주다 직렬화 500 이 나는 것을 막고, 페이로드가 로그·응답에 새지 않게 한다).
"""
from __future__ import annotations

import datetime as _dt
import hmac
import json
import logging
import os
import sys
import threading
import time
import uuid

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .bundle import Bundle, load_bundle
from .contract import HISTORY_ALIASES, PLAN_ALIASES
from .errors import BundleError, ServingError, ServingInputError

MAX_BODY_BYTES = 2_000_000

_log = logging.getLogger("kamp.serving.api")
if not _log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)


def _f(lo: float, hi: float, **kw):
    """유한값 + 물리 범위 float 필드."""
    return Field(..., ge=lo, le=hi, allow_inf_nan=False, **kw)


class HistoryRow(BaseModel):
    """이력 1시간 — 원본 KAMP 컬럼의 ASCII 별칭."""

    model_config = ConfigDict(extra="forbid")
    date: int = Field(..., ge=19000101, le=21001231, description="YYYYMMDD")
    hour: int = Field(..., ge=0, le=23)
    p15: float = _f(0, 5000)
    p30: float = _f(0, 5000)
    p45: float = _f(0, 5000)
    p60: float = _f(0, 5000)
    avg: float = _f(0, 5000)
    prod: float = _f(0, 1e7)
    headcount: float | None = _f(0, 1e4)
    labor: float = _f(0, 10)
    temp: float = _f(-60, 60)
    wind: float | None = _f(0, 100)
    humid: float = _f(0, 100)
    rain: float | None = _f(0, 1000)


class PlanRow(BaseModel):
    """대상일 계획 1시간 — 생산계획·공장인원·기상예보. 전력 필드는 받지 않는다."""

    model_config = ConfigDict(extra="forbid")
    date: int = Field(..., ge=19000101, le=21001231)
    hour: int = Field(..., ge=0, le=23)
    prod: float = _f(0, 1e7)
    headcount: float | None = _f(0, 1e4)
    temp: float = _f(-60, 60)
    wind: float | None = _f(0, 100)
    humid: float = _f(0, 100)
    rain: float | None = _f(0, 1000)


class PredictRequest(BaseModel):
    """익일 예측 요청. 이력은 대상일 전날 23:00 에서 끝나는 연속 8~62일.

    길이 판정(8일 미만·62일 초과)은 파이프라인이 하므로 여기서는 느슨하게 둔다 —
    그래야 `HISTORY_TOO_SHORT`(+ `required_start`) 같은 구체적 오류 코드가 나간다.
    """

    model_config = ConfigDict(extra="forbid")
    target_date: _dt.date
    history: list[HistoryRow] = Field(..., min_length=1, max_length=24 * 100)
    plan: list[PlanRow] = Field(..., min_length=24, max_length=24)


def _error(status: int, code: str, message: str, detail=None, request_id: str | None = None) -> JSONResponse:
    body = {"code": code, "message": message}
    if detail:
        body["detail"] = detail
    if request_id:
        body["request_id"] = request_id
    return JSONResponse(status_code=status, content={"error": body})


def _to_frame(rows: list[BaseModel], aliases: dict) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows])
    return df.rename(columns=aliases).astype({"날짜": "int64", "시간": "int64"})


def _rows_json(out: pd.DataFrame) -> list[dict]:
    """응답 행 — numpy 스칼라를 파이썬 값으로 명시 변환한다(직렬화 500 방지)."""
    rows = []
    for r in out.itertuples(index=False):
        d = r._asdict()
        rows.append({
            k: (v if isinstance(v, str) else int(v) if k in ("regime", "peak_label", "peak_label_cls") else float(v))
            for k, v in d.items()
        })
    return rows


def create_app(bundle_dir: str | None = None, *, allow_fast: bool = False,
               api_key: str | None = None, max_body_bytes: int = MAX_BODY_BYTES,
               holidays_file: str | None = None) -> FastAPI:
    """앱 팩토리. 번들을 **즉시** 검증·로드하고, 실패하면 예외를 던져 기동을 막는다.

    Parameters
    ----------
    bundle_dir : str, optional
        없으면 환경변수 `KAMP_BUNDLE_DIR` (필수).
    allow_fast : bool
        테스트 전용. 환경변수로는 켤 수 없다.
    api_key : str, optional
        없으면 환경변수 `KAMP_API_KEY`. 설정되면 `X-API-Key` 헤더를 요구한다.
    """
    bundle_dir = bundle_dir or os.environ.get("KAMP_BUNDLE_DIR")
    if not bundle_dir:
        raise BundleError("BUNDLE_NOT_FOUND", "KAMP_BUNDLE_DIR 환경변수로 번들 디렉터리를 지정하라")
    bundle: Bundle = load_bundle(bundle_dir, allow_fast=allow_fast, holidays_file=holidays_file)
    api_key = api_key if api_key is not None else os.environ.get("KAMP_API_KEY") or None
    api_key_bytes = api_key.encode("utf-8") if api_key else b""
    lock = threading.Lock()
    hd = bundle.history_days

    app = FastAPI(
        title="KAMP 자원최적화 — 익일 전력·피크 위험 예측",
        version="1.0.0",
        docs_url=None, redoc_url=None, openapi_url=None,   # 운영에서는 스키마 페이지를 열지 않는다
    )
    app.state.bundle = bundle

    @app.middleware("http")
    async def guard(request: Request, call_next):
        rid = uuid.uuid4().hex[:16]
        request.state.rid = rid
        t0 = time.perf_counter()
        length = request.headers.get("content-length")
        if request.method == "POST" and (length is None or not length.isdigit()):
            # 본문 길이를 모르는 요청(chunked)은 크기 제한을 우회할 수 있으므로 받지 않는다
            resp = _error(411, "LENGTH_REQUIRED", "Content-Length 헤더가 필요하다", request_id=rid)
        elif length is not None and length.isdigit() and int(length) > max_body_bytes:
            resp = _error(413, "BODY_TOO_LARGE", f"요청 본문이 {max_body_bytes:,} 바이트를 넘는다", request_id=rid)
        elif api_key and request.url.path.startswith("/v1/") and not hmac.compare_digest(
            # 헤더는 latin-1 로 디코딩된 문자열이다 — 바이트로 비교해야 비ASCII 키에서 500 이 나지 않는다
            request.headers.get("x-api-key", "").encode("latin-1", "replace"), api_key_bytes
        ):
            resp = _error(401, "UNAUTHORIZED", "X-API-Key 가 없거나 틀리다", request_id=rid)
        else:
            try:
                resp = await call_next(request)
            except Exception as exc:   # 처리되지 않은 예외도 요청 id·접근 로그를 남긴다
                _log.error(json.dumps({"request_id": rid, "error": type(exc).__name__}, ensure_ascii=False))
                resp = _error(500, "INTERNAL", "내부 오류", request_id=rid)
        resp.headers["X-Request-ID"] = rid
        _log.info(json.dumps({
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "request_id": rid, "path": request.url.path, "status": resp.status_code,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "bundle_id": bundle.bundle_id,
        }, ensure_ascii=False))
        return resp

    @app.exception_handler(RequestValidationError)
    async def on_schema(request: Request, exc: RequestValidationError):
        # 입력값(input)·ctx 는 싣지 않는다 — NaN/inf 를 되돌려 주면 응답 직렬화가 500 이 된다
        detail = [{"loc": [str(x) for x in e.get("loc", ())], "type": e.get("type"), "msg": e.get("msg")}
                  for e in exc.errors()[:20]]
        return _error(422, "SCHEMA", "요청 스키마 오류", detail, getattr(request.state, "rid", None))

    @app.exception_handler(ServingInputError)
    async def on_input(request: Request, exc: ServingInputError):
        return _error(422, exc.code, exc.message, exc.detail or None, getattr(request.state, "rid", None))

    @app.exception_handler(ServingError)
    async def on_serving(request: Request, exc: ServingError):
        # 입력 오류가 아닌 서빙 오류(예: PREDICTION_NONFINITE) — 서버 쪽 실패
        return _error(500, exc.code, exc.message, request_id=getattr(request.state, "rid", None))

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, exc: StarletteHTTPException):
        # 404·405 도 같은 오류 봉투로 돌려준다
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, f"HTTP_{exc.status_code}")
        return _error(exc.status_code, code, str(exc.detail), request_id=getattr(request.state, "rid", None))

    @app.exception_handler(Exception)
    async def on_internal(request: Request, exc: Exception):
        _log.error(json.dumps({"request_id": getattr(request.state, "rid", None),
                               "error": type(exc).__name__}, ensure_ascii=False))
        return _error(500, "INTERNAL", "내부 오류", request_id=getattr(request.state, "rid", None))

    @app.get("/healthz")
    def healthz():
        m = bundle.manifest
        return {"status": "ok", "bundle_id": bundle.bundle_id, "role": bundle.role,
                "train_last": m["training"]["last"]}

    @app.get("/v1/model")
    def model_info():
        m = bundle.manifest
        return {
            "bundle_id": bundle.bundle_id, "role": bundle.role, "model_name": m["model_name"],
            "feature_columns": bundle.columns, "theta": bundle.theta,
            "tau": bundle.tau, "tau_cls": bundle.tau_cls,
            "history_days": hd,
            "holiday_coverage": {
                "training_holidays_until": m["feature_contract"]["holiday_coverage_end"],
                "operator_years": bundle.holiday_calendar["coverage_years"],
            },
            "intervals": {
                "pi": {k: bundle.halfwidths[k] for k in ("alpha", "operating", "shutdown", "measured_test_coverage")},
                "quantile_calibrated": m["intervals"]["quantile"]["calibrated"],
            },
            "training": {k: m["training"][k] for k in ("first", "last", "n_rows", "months")},
            "caveats": m["caveats"],
        }

    @app.post("/v1/predict/day-ahead")
    def predict(req: PredictRequest, request: Request):
        history = _to_frame(req.history, HISTORY_ALIASES)
        plan = _to_frame(req.plan, PLAN_ALIASES)
        with lock:
            out, meta = bundle.predict_day(history, plan, target_date=req.target_date, max_days=int(hd["max"]))
        # 유한성은 assemble_output 이 보장한다(아니면 PREDICTION_NONFINITE → 500)
        return {"meta": {**meta, "request_id": request.state.rid}, "rows": _rows_json(out)}

    return app
