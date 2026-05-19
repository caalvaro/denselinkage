"""Clustering — a separate, pluggable function returning a typed result."""

from denselinkage.core.results import Clustering, LinkageResult


def connected_components(result: LinkageResult) -> Clustering: ...


__all__ = ["connected_components"]
