"""Gen 3 — DL 재구성/예측 (2015–2020). 전부 논문 기반 자체 재구현.

라이선스 감사 결과 원저장소 다수가 라이선스 불명 또는 비호환
(OmniAnomaly: 라이선스 파일 부재 이력, donut: 불명 등)이므로 코드를 복사하지 않고
논문 수식으로 재구현했다. 각 모듈 docstring에 원 논문과의 차이를 명시한다.
"""

from tsad_forge.models.gen3_dl_recon.ae import AEDetector
from tsad_forge.models.gen3_dl_recon.dagmm import DAGMMDetector
from tsad_forge.models.gen3_dl_recon.lstm import LSTMADDetector, LSTMPredictorDetector
from tsad_forge.models.gen3_dl_recon.omni import OmniAnomalyDetector
from tsad_forge.models.gen3_dl_recon.usad import USADDetector
from tsad_forge.models.gen3_dl_recon.vae import DonutVAEDetector

__all__ = [
    "AEDetector",
    "DAGMMDetector",
    "DonutVAEDetector",
    "LSTMADDetector",
    "LSTMPredictorDetector",
    "OmniAnomalyDetector",
    "USADDetector",
]
