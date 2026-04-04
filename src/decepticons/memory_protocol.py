"""Protocol for optional memory attachment to substrate families.

A memory attachment provides a residual probability correction to the
substrate's base prediction.  The kernel defines the protocol; descendants
own the runtime implementation.

Memory kinds:
  - "none"              no memory attachment (default)
  - "ngram"             uses NgramMemoryConfig from this package
  - "exact_context"     uses ExactContextConfig from this package
  - "statistical_backoff" uses StatisticalBackoffConfig from this package
"""
from __future__ import annotations

from dataclasses import dataclass

MEMORY_KINDS = ("none", "ngram", "exact_context", "statistical_backoff")


@dataclass(frozen=True)
class MemoryAttachmentConfig:
    """Configuration for an optional memory attachment.

    The kernel provides the config shape.  Descendants own the implementation:
    how the memory is built, trained, mixed with the substrate prediction, and
    packed into an artifact.
    """

    kind: str = "none"
    vocabulary_size: int = 256
    max_order: int = 3
    alpha: float = 0.05
    trigram_bucket_count: int = 4096
    mix_mode: str = "learned"  # "learned" or "fixed"

    def __post_init__(self) -> None:
        if self.kind not in MEMORY_KINDS:
            raise ValueError(f"Unknown memory kind: {self.kind!r}; expected one of {MEMORY_KINDS}")
        if self.vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        if self.max_order < 1:
            raise ValueError("max_order must be >= 1")
        if self.trigram_bucket_count < 1:
            raise ValueError("trigram_bucket_count must be positive")
