"""denselinkage — record linkage with dense blocking and LLM matching.

Curated prelude: the symbols a typical script needs without reaching into
submodules. Ports and heavy adapters stay in their submodules
(``denselinkage.core.ports``, ``denselinkage.embedding``, ...) on purpose —
the top level is the orchestration entry points plus the canonical
reference serializers and result types.
"""

from denselinkage.clustering import connected_components
from denselinkage.core.models import Source
from denselinkage.core.results import ClusteringResult, LabeledPairs, LinkageResult
from denselinkage.linkage import DenseLinker, LinkageIndex
from denselinkage.metrics import BlockingMetrics, ClusteringMetrics, LinkageMetrics
from denselinkage.serializing import (
    FieldwiseSerializer,
    TemplateSerializer,
    WholeRowSerializer,
)

__version__ = "0.1.0"

__all__ = [
    "BlockingMetrics",
    "ClusteringMetrics",
    "ClusteringResult",
    "DenseLinker",
    "FieldwiseSerializer",
    "LabeledPairs",
    "LinkageIndex",
    "LinkageMetrics",
    "LinkageResult",
    "Source",
    "TemplateSerializer",
    "WholeRowSerializer",
    "connected_components",
]
