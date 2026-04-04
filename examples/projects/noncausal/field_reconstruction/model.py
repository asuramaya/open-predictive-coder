from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decepticons.artifacts import ArtifactAccounting, ArtifactMetadata
from decepticons.noncausal_reconstructive import (
    NoncausalReconstructiveAdapter,
    NoncausalReconstructiveConfig,
    NoncausalReconstructiveFitReport,
    NoncausalReconstructiveReport,
    NoncausalReconstructiveTrace,
)


@dataclass(frozen=True)
class FieldReconstructionConfig:
    vocabulary_size: int = 256
    exact_max_order: int = 3
    exact_alpha: float = 0.05
    bidirectional_left_order: int = 2
    bidirectional_right_order: int = 2
    blend_temperature: float = 1.0
    agreement_threshold: float = 0.75
    replay_threshold: float = 0.55
    min_replay_span: int = 2
    artifact_name: str = "field_reconstruction"
    metadata: ArtifactMetadata | None = None

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.exact_max_order < 1:
            raise ValueError("exact_max_order must be >= 1")
        if self.exact_alpha < 0.0:
            raise ValueError("exact_alpha must be >= 0")
        if self.bidirectional_left_order < 0:
            raise ValueError("bidirectional_left_order must be >= 0")
        if self.bidirectional_right_order < 0:
            raise ValueError("bidirectional_right_order must be >= 0")
        if self.blend_temperature <= 0.0:
            raise ValueError("blend_temperature must be > 0")
        if not 0.0 <= self.agreement_threshold <= 1.0:
            raise ValueError("agreement_threshold must be in [0, 1]")
        if not 0.0 <= self.replay_threshold <= 1.0:
            raise ValueError("replay_threshold must be in [0, 1]")
        if self.min_replay_span < 1:
            raise ValueError("min_replay_span must be >= 1")

    def to_adapter_config(self) -> NoncausalReconstructiveConfig:
        return NoncausalReconstructiveConfig(
            vocabulary_size=self.vocabulary_size,
            exact_max_order=self.exact_max_order,
            exact_alpha=self.exact_alpha,
            bidirectional_left_order=self.bidirectional_left_order,
            bidirectional_right_order=self.bidirectional_right_order,
            blend_temperature=self.blend_temperature,
            agreement_threshold=self.agreement_threshold,
            replay_threshold=self.replay_threshold,
            min_replay_span=self.min_replay_span,
        )


FieldReconstructionFitReport = NoncausalReconstructiveFitReport
FieldReconstructionTrace = NoncausalReconstructiveTrace
FieldReconstructionReport = NoncausalReconstructiveReport


class FieldReconstructionModel:
    def __init__(self, config: FieldReconstructionConfig | None = None):
        self.config = config or FieldReconstructionConfig()
        self.adapter = NoncausalReconstructiveAdapter(
            self.config.to_adapter_config(),
            artifact_name=self.config.artifact_name,
            metadata=self.config.metadata,
        )

    @classmethod
    def build(cls, **kwargs: object) -> "FieldReconstructionModel":
        return cls(FieldReconstructionConfig(**kwargs))

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> FieldReconstructionFitReport:
        return self.adapter.fit(data)

    def trace(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> FieldReconstructionTrace:
        return self.adapter.trace(sequence)

    def reconstruct(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> np.ndarray:
        return self.adapter.reconstruct(sequence)

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> FieldReconstructionReport:
        return self.adapter.score(sequence)

    def accounting(self) -> ArtifactAccounting:
        return self.adapter.accounting()


__all__ = [
    "FieldReconstructionConfig",
    "FieldReconstructionFitReport",
    "FieldReconstructionModel",
    "FieldReconstructionReport",
    "FieldReconstructionTrace",
]
