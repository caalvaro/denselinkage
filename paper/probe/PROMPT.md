# Contract-implementability probe

A first out-of-sample test of QR1 (extensibility): can a party that did **not**
design the contract implement a new adapter from the public ports and value
types alone? Five independent LLM agents were each given **only** the contract
below (no access to any existing adapter) and asked to implement one `Matcher`
adapter. Each result is then verified mechanically — `mypy --strict`, a
port-conformance + functional harness (`verify_probe.py`), and a `git diff`
confined to the new file. This is an *extensibility / implementability* probe,
not a usability study: an LLM agent is not an independent human author, but it
is a party that did not place the contract, so a clean implementation from the
ports alone is evidence the surface is self-sufficient. (See Zan et al., *When
Language Model Meets Private Library*, EMNLP Findings 2022, for the established
task of implementing against an unseen library from its documentation.)

## Protocol
- **n = 5** independent attempts, each a fresh agent with the identical prompt.
- **Withheld:** every existing adapter and the two examples that implement
  adapters (`03_custom_embedder.py`, `05_failure_accounting.py`). The agents
  received the contract inline (the value types + the `Matcher` Protocol) and
  were instructed not to read any files.
- **Single-shot:** each agent returns the module; no human edits, no
  prompt-tuning to force a pass. All outcomes are reported, including failures.
- **Generator model:** the orchestrating session's model (Claude Opus 4.8).
- **Verification (all must pass):** (C1) `mypy --strict` on the module; (C2)
  explicit `Matcher` subclass; (C3) `git diff` touches no existing file; (C4)
  the failure contract (normal pair → `MatchDecision`, empty text →
  `MatchError`, one outcome per pair) and an end-to-end run inside a real
  `DenseLinker`.

## Contract given to the agents (the only API they could use)

```
# core/models.py : value types
RecordId = str

@dataclass(frozen=True, slots=True)
class Record:
    id: RecordId
    text: str
    fields: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class CandidatePair:
    record_a: Record
    record_b: Record
    similarity_score: float | None = None

@dataclass(frozen=True, slots=True)
class MatchDecision:        # is_match is always a real bool
    is_match: bool
    confidence: float | None = None
    rationale: str | None = None

@dataclass(frozen=True, slots=True)
class MatchError:           # a pair the matcher could not decide
    reason: str

# core/ports.py : the Matcher Protocol
@runtime_checkable
class Matcher(Protocol):
    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        # one outcome per pair, position-aligned; an undecidable pair yields a
        # MatchError, never raising into the batch.
        ...
```

## Task

Implement a `Matcher` adapter `TokenJaccardMatcher` in a single self-contained
module: decide each pair by the Jaccard similarity of the whitespace token sets
of the two records' lowercased `text` (J = |∩| / |∪|); constructor keyword-only
`threshold: float = 0.5`; return `MatchDecision(is_match = J >= threshold,
confidence = J)`, or a `MatchError` if either text has no tokens; one outcome
per pair, in order. Import only from `denselinkage.core.*` and the stdlib;
subclass `Matcher` explicitly; type-check under `mypy --strict`.

## Results

See `RESULTS.md` (generated after verification).
