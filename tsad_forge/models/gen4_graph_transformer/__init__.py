"""Gen 4 — Graph/Transformer (2020–2023). 논문 기반 자체 재구현.

라이선스 감사: TranAD 원저장소는 BSD-3(코드 참조 가능)이나 일관성을 위해 전 모델을
논문 수식 기반으로 재구현했다. 각 모듈 docstring에 원 논문과의 차이를 명시한다.
"""

from tsad_forge.models.gen4_graph_transformer.anomaly_transformer import (
    AnomalyTransformerDetector,
)
from tsad_forge.models.gen4_graph_transformer.dcdetector import DCdetectorDetector
from tsad_forge.models.gen4_graph_transformer.gdn import GDNDetector
from tsad_forge.models.gen4_graph_transformer.mtad_gat import MTADGATDetector
from tsad_forge.models.gen4_graph_transformer.timesnet import TimesNetDetector
from tsad_forge.models.gen4_graph_transformer.tranad import TranADDetector

__all__ = [
    "AnomalyTransformerDetector",
    "DCdetectorDetector",
    "GDNDetector",
    "MTADGATDetector",
    "TimesNetDetector",
    "TranADDetector",
]
