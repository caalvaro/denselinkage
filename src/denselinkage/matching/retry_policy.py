"""``RetryPolicy`` — operational config for retrying matcher calls."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_seconds: float = 0.0
