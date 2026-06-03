"""Candidate-pair filtering — a second comparison-space reduction distinct from
blocking (its own port/module, parallel to ``blocking``/``indexing``). The
dependency-free reference adapter ``SimilarityThresholdFilter`` implements the
``denselinkage.core.ports.Filter`` port.

This package is a façade: the implementation lives in
``similarity_threshold_filter``; import the public name here.
"""

from denselinkage.filtering.similarity_threshold_filter import (
    SimilarityThresholdFilter,
)

__all__ = ["SimilarityThresholdFilter"]
