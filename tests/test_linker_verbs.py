"""Behavioural tests for the ``match_pairs`` and ``dedupe`` verbs and the shared
``assemble_linkage_result`` helper (Batch 1)."""

import pandas as pd
import pytest

from denselinkage import (
    DenseLinker,
    LabeledPairs,
    Source,
    TemplateSerializer,
    connected_components,
)
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.linkage._assembly import assemble_linkage_result
from denselinkage.matching import ThresholdMatcher
from denselinkage.metrics import clustering_metrics


class _CountBreakingMatcher:
    """Returns the wrong number of outcomes — exercises the alignment guard."""

    def match(self, pairs: object) -> list[object]:
        return []


# --- assemble_linkage_result ------------------------------------------------


def test_assemble_rejects_misaligned_matcher() -> None:
    pair = CandidatePair(Record("a", ""), Record("b", ""), similarity_score=0.9)
    with pytest.raises(ValueError, match="one outcome per input pair"):
        assemble_linkage_result([pair], _CountBreakingMatcher())


# --- match_pairs ------------------------------------------------------------


def test_match_pairs_needs_no_blocker() -> None:
    linker = DenseLinker(matcher=ThresholdMatcher(threshold=0.5))  # no blocker
    candidates = [
        CandidatePair(Record("a", ""), Record("b", ""), similarity_score=0.9),
        CandidatePair(Record("c", ""), Record("d", ""), similarity_score=0.1),
    ]
    result = linker.match_pairs(candidates)
    decided = {(p.record_a.id, p.record_b.id): d.is_match for p, d in result.decisions}
    assert decided == {("a", "b"): True, ("c", "d"): False}
    assert result.errors == ()


def test_match_pairs_unscored_pair_goes_to_errors() -> None:
    linker = DenseLinker(matcher=ThresholdMatcher(threshold=0.5))
    candidates = [
        CandidatePair(Record("a", ""), Record("b", ""), similarity_score=None)
    ]
    result = linker.match_pairs(candidates)
    assert result.decisions == ()
    assert len(result.errors) == 1
    pair, err = result.errors[0]
    assert isinstance(err, MatchError)
    assert (pair.record_a.id, pair.record_b.id) == ("a", "b")


def test_match_pairs_empty() -> None:
    result = DenseLinker(matcher=ThresholdMatcher()).match_pairs([])
    assert result.decisions == ()
    assert result.errors == ()


# --- dedupe -----------------------------------------------------------------


def _dedupe_source() -> Source:
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "name": ["Apple Inc", "Apple Incorporated", "Microsoft Corp"],
            "city": ["Cupertino", "Cupertino", "Redmond"],
        }
    )
    return Source(df, id_column="id")


def test_dedupe_requires_a_blocker() -> None:
    with pytest.raises(ValueError, match="requires a blocker"):
        DenseLinker(matcher=ThresholdMatcher()).dedupe(_dedupe_source())


def test_dedupe_suppresses_self_and_symmetric_pairs() -> None:
    result = DenseLinker.with_defaults().dedupe(_dedupe_source())
    all_pairs = [p for p, _ in result.decisions] + [p for p, _ in result.errors]
    assert all(p.record_a.id != p.record_b.id for p in all_pairs)  # no self-pairs
    keys = [frozenset((p.record_a.id, p.record_b.id)) for p in all_pairs]
    assert len(keys) == len(set(keys))  # no symmetric duplicates


def test_dedupe_finds_duplicate_and_clusters_it() -> None:
    result = DenseLinker.with_defaults().dedupe(_dedupe_source())
    # Couples to the lexical default stack: "Apple Inc"/"Apple Incorporated" must
    # clear the 0.5 threshold. The identical-text cases above are the robust anchors.
    matched = {
        frozenset((p.record_a.id, p.record_b.id))
        for p, d in result.decisions
        if d.is_match
    }
    assert frozenset(("1", "2")) in matched  # the two Apple rows
    clusters = connected_components(result)
    assert clusters.labels["1"] == clusters.labels["2"]
    assert clusters.labels["3"] != clusters.labels["1"]
    assert clusters.n_clusters == 2


def test_link_and_index_require_a_blocker() -> None:
    linker = DenseLinker(matcher=ThresholdMatcher())
    src = Source(pd.DataFrame({"id": ["1"]}), id_column="id")
    with pytest.raises(ValueError, match="requires a blocker"):
        linker.index(src)
    with pytest.raises(ValueError, match="requires a blocker"):
        linker.link(src, src)


def test_threshold_matcher_boundary_is_inclusive() -> None:
    # A similarity exactly at the threshold is a match (>=, not >).
    pair = CandidatePair(Record("a", ""), Record("b", ""), similarity_score=0.5)
    [decision] = ThresholdMatcher(threshold=0.5).match([pair])
    assert isinstance(decision, MatchDecision)
    assert decision.is_match is True


def test_dedupe_to_clustering_metrics_end_to_end() -> None:
    # The full dependency-free dedup pipeline (example 04): dedupe ->
    # connected_components -> B3 clustering_metrics, on the lexical stack. This
    # pins the result the CI example smoke-run only crash-checks.
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "name": ["Apple Inc", "Apple Incorporated", "Microsoft Corp"],
            "city": ["Cupertino", "Cupertino", "Redmond"],
        }
    )
    src = Source(df, id_column="id", serializer=TemplateSerializer("{name} — {city}"))
    clusters = connected_components(DenseLinker.with_defaults().dedupe(src))
    cm = clustering_metrics(clusters, gold=LabeledPairs.from_pairs([("1", "2")]))
    assert (cm.b3_precision, cm.b3_recall, cm.b3_f1) == (1.0, 1.0, 1.0)
    assert (cm.n_clusters, cm.n_gold_clusters) == (2, 2)
