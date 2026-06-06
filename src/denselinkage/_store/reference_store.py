"""Reference Store — persist/reload the dependency-free blocking index (a
``DenseBlocker`` over ``NumpyFlatIndex``) as a portable on-disk bundle.

Layout (a directory):

- ``vectors.npy`` — the indexed embeddings (float32, ``n x dim``), the costly
  artifact a reload avoids recomputing.
- ``meta.json`` — provenance (``model_id`` / ``embedding_dim``), blocker config
  (``top_k`` / ``similarity_threshold``), the record ids, and the reference
  records (``id`` / ``text`` / ``fields``).

No ``pickle``: the embedder and matcher are re-supplied at load time and the
stored ``model_id`` / ``embedding_dim`` are validated against the re-supplied
embedder, so a persisted index refuses a query embedded by a different model.
"""

import json
from pathlib import Path

import numpy as np

from denselinkage.blocking.dense_blocking_index import DenseBlockingIndex
from denselinkage.core.errors import IncompatibleStore
from denselinkage.core.models import Record
from denselinkage.core.ports import BlockingIndex, Embedder
from denselinkage.indexing.numpy_searchable_index import NumpySearchableIndex

_FORMAT = 1
_VECTORS_FILE = "vectors.npy"
_META_FILE = "meta.json"
_UNSUPPORTED = (
    "the reference store persists the dependency-free numpy stack "
    "(DenseBlocker over NumpyFlatIndex); cannot persist a {name}"
)
_REQUIRED_KEYS = frozenset(
    {
        "format",
        "model_id",
        "embedding_dim",
        "top_k",
        "similarity_threshold",
        "ids",
        "records",
    }
)


def save_reference_index(blocking: BlockingIndex, path: str | Path) -> None:
    """Persist ``blocking`` (the dependency-free reference stack) to ``path``.

    ``path`` is a directory dedicated to this store; ``save`` writes
    ``vectors.npy`` and ``meta.json`` into it (overwriting those two on re-save).
    Raises ``NotImplementedError`` if ``blocking`` is not a ``DenseBlockingIndex``
    over a ``NumpySearchableIndex`` (e.g. a FAISS-backed index — persistence for
    that ships with the FAISS adapter).
    """
    if not isinstance(blocking, DenseBlockingIndex):
        raise NotImplementedError(_UNSUPPORTED.format(name=type(blocking).__name__))
    searchable = blocking.searchable
    if not isinstance(searchable, NumpySearchableIndex):
        raise NotImplementedError(_UNSUPPORTED.format(name=type(searchable).__name__))

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / _VECTORS_FILE, searchable.vectors)
    meta = {
        "format": _FORMAT,
        "model_id": blocking.embedder.model_id,
        "embedding_dim": blocking.embedder.embedding_dim,
        "top_k": blocking.top_k,
        "similarity_threshold": blocking.similarity_threshold,
        "ids": list(searchable.ids),
        "records": [
            {"id": record.id, "text": record.text, "fields": dict(record.fields)}
            for record in blocking.records.values()
        ],
    }
    with (directory / _META_FILE).open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, default=str)


def load_reference_index(path: str | Path, *, embedder: Embedder) -> DenseBlockingIndex:
    """Reload a blocking index from ``path``, re-supplying the live ``embedder``.

    Raises ``IncompatibleStore`` if the store is malformed or unsupported, or the
    re-supplied ``embedder``'s ``model_id`` / ``embedding_dim`` does not match the
    stored provenance.
    """
    directory = Path(path)
    with (directory / _META_FILE).open(encoding="utf-8") as handle:
        meta = json.load(handle)

    if not isinstance(meta, dict):
        raise IncompatibleStore("store meta.json is not a JSON object")
    missing = _REQUIRED_KEYS - meta.keys()
    if missing:
        raise IncompatibleStore(f"store meta.json is missing keys: {sorted(missing)}")
    if meta["format"] != _FORMAT:
        raise IncompatibleStore(
            f"unsupported store format {meta['format']!r} (expected {_FORMAT})"
        )
    if embedder.model_id != meta["model_id"]:
        raise IncompatibleStore(
            f"stored index was built with embedder {meta['model_id']!r}, "
            f"but got {embedder.model_id!r}"
        )
    if embedder.embedding_dim != meta["embedding_dim"]:
        raise IncompatibleStore(
            f"stored index has embedding_dim {meta['embedding_dim']}, "
            f"but the embedder produces {embedder.embedding_dim}"
        )

    vectors = np.load(directory / _VECTORS_FILE)
    expected_shape = (len(meta["ids"]), meta["embedding_dim"])
    if vectors.shape != expected_shape:
        raise IncompatibleStore(
            f"stored vectors have shape {vectors.shape}, expected {expected_shape}"
        )
    records_by_id = {
        str(entry["id"]): Record(
            id=str(entry["id"]), text=entry["text"], fields=entry["fields"]
        )
        for entry in meta["records"]
    }
    searchable = NumpySearchableIndex(
        vectors, [str(record_id) for record_id in meta["ids"]]
    )
    return DenseBlockingIndex(
        searchable=searchable,
        embedder=embedder,
        records_by_id=records_by_id,
        top_k=meta["top_k"],
        similarity_threshold=meta["similarity_threshold"],
    )
