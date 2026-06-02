"""Clustering — a swappable strategy over a ``LinkageResult``.

``Clusterer`` (in ``denselinkage.core.ports``) is the port; this package ships
the dependency-free reference algorithm in two forms: ``connected_components``,
the prelude convenience function, and ``ConnectedComponentsClusterer``, the
adapter that declares the ``Clusterer`` port (so it is mypy-completeness-checked
and appears in the adapter contract test). Alternative algorithms (e.g.
agglomerative, incremental) implement ``Clusterer`` in their own modules.

This package is a façade: implementations live in sibling modules; import the
public names here.
"""

from denselinkage.clustering.connected_components import connected_components
from denselinkage.clustering.connected_components_clusterer import (
    ConnectedComponentsClusterer,
)

__all__ = ["ConnectedComponentsClusterer", "connected_components"]
