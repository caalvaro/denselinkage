"""denselinkage — record linkage with dense blocking and LLM matching.

Curated prelude: the symbols a typical script needs without reaching into
submodules. Ports and heavy adapters stay in their submodules
(``denselinkage.core.ports``, ``denselinkage.embedding``, ...) on purpose —
the top level is the orchestration entry points plus the canonical
reference serializers and result types.
"""

from denselinkage.cluster import connected_components
from denselinkage.core.models import Source
from denselinkage.core.results import (
    BlockingMetrics,
    Clustering,
    LabeledPairs,
    LinkageMetrics,
    LinkageResult,
)
from denselinkage.linker import DenseLinker, LinkageIndex
from denselinkage.serialize import (
    FieldwiseSerializer,
    TemplateSerializer,
    WholeRowSerializer,
)

__version__ = "0.1.0"

__all__ = [
    "BlockingMetrics",
    "Clustering",
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
