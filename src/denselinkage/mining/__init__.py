"""Active-learning material — mine hard negatives from scored candidates.

Dependency-free and forward-looking: the v2 ``denselinkage.training`` adapters
consume what this produces. The façade re-exports the public name.
"""

from denselinkage.mining.hard_negatives import mine_hard_negatives

__all__ = ["mine_hard_negatives"]
