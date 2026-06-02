"""Matchers. ``LangChainMatcher`` is the heavy adapter (extra:
``[langchain]``). The user prompt is only the question; the matcher owns
output and returns typed ``MatchDecision``s.

This package is a façade: implementations live in sibling modules; import the
public names here.
"""

from denselinkage.matching.langchain_matcher import LangChainMatcher
from denselinkage.matching.retry_policy import RetryPolicy
from denselinkage.matching.threshold_matcher import ThresholdMatcher

__all__ = ["LangChainMatcher", "RetryPolicy", "ThresholdMatcher"]
