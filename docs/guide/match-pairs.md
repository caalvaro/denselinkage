# Matching pre-blocked pairs

If you already have candidate pairs — from rule-based blocking, an external
system, or a join — skip dense blocking and decide them directly with
`match_pairs`. This is Resolvi's "matching without blocking" variant.

```python
from denselinkage import DenseLinker
from denselinkage.core.models import CandidatePair, Record
from denselinkage.matching import LangChainMatcher

pairs = [
    CandidatePair(
        record_a=Record(id="A1", text="Apple Inc, Cupertino"),
        record_b=Record(id="B1", text="Apple Incorporated, Cupertino"),
    ),
    # ...
]

linker = DenseLinker(matcher=LangChainMatcher(llm=...))   # no blocker needed
result = linker.match_pairs(pairs)                        # -> LinkageResult
```

## Similarity is optional here

A {class}`~denselinkage.core.models.CandidatePair` from dense blocking carries a
`similarity_score`; pairs you supply from elsewhere have none, so
`similarity_score` defaults to `None`.

This matters for the matcher you choose:

- {class}`~denselinkage.matching.LangChainMatcher` reads the record text, so it
  works whether or not a similarity is present.
- {class}`~denselinkage.matching.ThresholdMatcher` gates *on* the carried
  similarity. Given a pair with `similarity_score=None` it cannot decide and
  returns a {class}`~denselinkage.core.models.MatchError` — so either supply a
  `similarity_score` yourself or use a content-aware matcher.

## See also

- [Custom components](custom-components) — write your own `Matcher`.
- [Evaluation](evaluation) — score the decided pairs.
