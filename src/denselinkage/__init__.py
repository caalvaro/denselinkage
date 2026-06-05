"""denselinkage — record linkage with dense blocking and LLM matching.

Curated prelude: the symbols a typical script needs without reaching into
submodules. Ports and heavy adapters stay in their submodules
(``denselinkage.core.ports``, ``denselinkage.embedding``, ...) on purpose —
the top level is the orchestration entry points plus the canonical
reference serializers and result types.
"""

from importlib.metadata import PackageNotFoundError, version

from denselinkage.clustering import connected_components
from denselinkage.core.models import Source
from denselinkage.core.results import ClusteringResult, LabeledPairs, LinkageResult
from denselinkage.linkage import (
    DenseLinker,
    LinkageIndex,
    candidate_pairs_from_frame,
)
from denselinkage.metrics import BlockingMetrics, ClusteringMetrics, LinkageMetrics
from denselinkage.serializing import (
    FieldwiseSerializer,
    TemplateSerializer,
    WholeRowSerializer,
)

try:
    __version__ = version("denselinkage")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

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
    "candidate_pairs_from_frame",
    "connected_components",
]
