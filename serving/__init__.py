"""KAMP 자원최적화 — 모델 번들 서빙 패키지 (추론 전용).

노트북 10.5절이 만든 번들(`outputs/models/full/{eval|deploy}/`)을 검증하며 읽어
익일 24시간 전력·피크 위험을 예측한다. 학습 코드는 두지 않는다 — 학습은 노트북만 한다.

    from serving import load_bundle
    bundle = load_bundle("deploy_bundle_dir")
    pred, meta = bundle.predict_day(history_df, plan_df)

제출 zip·포드 제출 패키지에 포함한다(번들은 노트북 실행으로 만든다). 사용법은 serving/README.md.
"""
from .bundle import Bundle, load_bundle
from .errors import BundleError, ServingError, ServingInputError

__version__ = "1.0.0"
__all__ = ["Bundle", "BundleError", "ServingError", "ServingInputError", "load_bundle"]
