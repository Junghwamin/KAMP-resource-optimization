"""서빙 오류 — 입력 오류(422 / 종료코드 2)와 번들 오류(기동 실패 / 종료코드 3).

메시지에는 서버의 절대경로·계정명을 넣지 않는다. 오류 응답이 그대로 클라이언트에 가기 때문이다.
"""
from __future__ import annotations


class ServingError(Exception):
    """코드·메시지·상세를 가진 서빙 오류의 공통 부모."""

    exit_code = 1

    def __init__(self, code: str, message: str, detail: dict | None = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict:
        """API 오류 봉투의 `error` 부분."""
        out = {"code": self.code, "message": self.message}
        if self.detail:
            out["detail"] = self.detail
        return out


class ServingInputError(ServingError):
    """요청 입력(이력·계획·대상일)이 계약을 어겼다. API 422, CLI 종료코드 2."""

    exit_code = 2


class PredictionError(ServingError):
    """예측값이 유한하지 않다 등 서버 쪽 예측 실패. API 500, CLI 종료코드 1."""

    exit_code = 1


class BundleError(ServingError):
    """번들이 없거나 손상됐거나 이 런타임과 맞지 않는다. 기동 거부, CLI 종료코드 3."""

    exit_code = 3
