"""Internal persistence for the dependency-free reference stack (the Reference
Store / Memento). The public entry points are ``LinkageIndex.save`` /
``LinkageIndex.load``; this package holds the on-disk format and the
provenance check.
"""

from denselinkage._store.reference_store import (
    load_reference_index,
    save_reference_index,
)

__all__ = ["load_reference_index", "save_reference_index"]
