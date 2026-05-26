"""Clustering — a swappable strategy over a ``LinkageResult``.

``Clusterer`` (in ``denselinkage.core.ports``) is the port; this module ships
the dependency-free reference algorithm in two forms: ``connected_components``,
the prelude convenience function, and ``ConnectedComponentsClusterer``, the
adapter that declares the ``Clusterer`` port (so it is mypy-completeness-checked
and appears in the adapter contract test). Alternative algorithms (e.g.
agglomerative, incremental) implement ``Clusterer`` in their own modules.
"""

from denselinkage.core.ports import Clusterer
from denselinkage.core.results import Clustering, LinkageResult


def connected_components(result: LinkageResult) -> Clustering:
    """Connected-components clustering: transitively close the matched pairs in
    ``result`` and label each record with its component id. Convenience form of
    :class:`ConnectedComponentsClusterer`; kept in the prelude."""
    ...


class ConnectedComponentsClusterer(Clusterer):
    """Reference ``Clusterer`` adapter; delegates to :func:`connected_components`."""

    def cluster(self, result: LinkageResult) -> Clustering: ...


__all__ = ["ConnectedComponentsClusterer", "connected_components"]
