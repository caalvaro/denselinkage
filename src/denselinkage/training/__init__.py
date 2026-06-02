"""Reserved namespace for v2 training (behind the ``[train]`` extra).

Intentionally empty in v1. ``EmbedderTrainer`` (symmetric in-batch
contrastive bi-encoder) and ``CrossEncoderTrainer`` (ce / tsallis / focal)
land here in v2 and implement the ``denselinkage.core.ports.Trainer``
protocol (locked in Phase A). The package path and the ``[train]`` extra are
reserved now so v2 adds them without a breaking import change. LLM
fine-tuning is intentionally out of scope.
"""

__all__: list[str] = []
