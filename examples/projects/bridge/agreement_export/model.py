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

from decepticons import BridgeFeatureArrays, BridgeFeatureConfig, bridge_feature_arrays, ensure_tokens


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _normalized_matrix(rng: np.random.Generator, rows: int, cols: int, scale: float) -> np.ndarray:
    matrix = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
    return (matrix / np.sqrt(max(cols, 1))) * scale


@dataclass(frozen=True)
class AgreementExportConfig:
    vocabulary_size: int = 256
    hidden_dim: int = 48
    agreement_window: int = 6
    bridge: BridgeFeatureConfig = BridgeFeatureConfig(candidate_count=4)
    seed: int = 29

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.hidden_dim < 4:
            raise ValueError("hidden_dim must be >= 4")
        if self.agreement_window < 1:
            raise ValueError("agreement_window must be >= 1")


@dataclass(frozen=True)
class AgreementExportTrace:
    tokens: int
    left_probs: np.ndarray
    right_probs: np.ndarray
    features: BridgeFeatureArrays

    @property
    def steps(self) -> int:
        return int(self.left_probs.shape[0])


@dataclass(frozen=True)
class AgreementExportReport:
    tokens: int
    steps: int
    mean_entropy: float
    mean_agreement: float
    mean_agreement_mass: float
    mean_consensus_ratio: float
    mean_disagreement: float


class AgreementExportModel:
    def __init__(self, config: AgreementExportConfig | None = None):
        self.config = config or AgreementExportConfig()
        rng = np.random.default_rng(self.config.seed)
        self.embeddings = rng.normal(
            loc=0.0,
            scale=0.4,
            size=(self.config.vocabulary_size, self.config.hidden_dim),
        ).astype(np.float64)
        self.recurrent = _normalized_matrix(rng, self.config.hidden_dim, self.config.hidden_dim, scale=0.75)
        self.left_head = _normalized_matrix(rng, self.config.vocabulary_size, self.config.hidden_dim, scale=0.9)
        self.right_head = _normalized_matrix(rng, self.config.vocabulary_size, self.config.hidden_dim, scale=0.9)
        self.bias = rng.normal(loc=0.0, scale=0.04, size=(self.config.vocabulary_size,)).astype(np.float64)

    def _coerce_tokens(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> np.ndarray:
        if isinstance(data, (str, bytes, bytearray, memoryview, np.ndarray)):
            return ensure_tokens(data)
        if isinstance(data, Sequence):
            return ensure_tokens(data)
        return ensure_tokens(data)

    def _scan_probabilities(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.zeros(self.config.hidden_dim, dtype=np.float64)
        left_window: list[np.ndarray] = []
        right_window: list[np.ndarray] = []
        left_probs: list[np.ndarray] = []
        right_probs: list[np.ndarray] = []

        for token in tokens:
            embedded = self.embeddings[int(token)]
            hidden = np.tanh((self.recurrent @ hidden) + embedded)
            left_window.append(hidden.copy())
            right_window.append(np.roll(hidden, 1))
            if len(left_window) > self.config.agreement_window:
                left_window.pop(0)
            if len(right_window) > self.config.agreement_window:
                right_window.pop(0)

            left_state = np.mean(np.vstack(left_window), axis=0)
            right_state = np.mean(np.vstack(right_window), axis=0)
            left_logits = (self.left_head @ left_state) + self.bias
            right_logits = (self.right_head @ right_state) + self.bias

            left_probs.append(_softmax(left_logits))
            right_probs.append(_softmax(right_logits))

        if not left_probs:
            empty = np.zeros((0, self.config.vocabulary_size), dtype=np.float64)
            return empty, empty
        return np.vstack(left_probs), np.vstack(right_probs)

    def trace(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> AgreementExportTrace:
        tokens = self._coerce_tokens(data)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")
        left_probs, right_probs = self._scan_probabilities(tokens[:-1])
        features = bridge_feature_arrays(left_probs, right_probs, self.config.vocabulary_size, config=self.config.bridge)
        return AgreementExportTrace(
            tokens=int(tokens.size),
            left_probs=left_probs,
            right_probs=right_probs,
            features=features,
        )

    def report(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> AgreementExportReport:
        trace = self.trace(data)
        features = trace.features
        agreement = np.asarray(features.agreement, dtype=np.float64)
        agreement_mass = np.asarray(features.agreement_mass, dtype=np.float64)
        candidate4 = np.asarray(features.candidate4, dtype=np.float64)
        consensus_ratio = np.divide(
            agreement_mass,
            np.maximum(candidate4, 1e-12),
            out=np.zeros_like(agreement_mass),
            where=candidate4 > 0.0,
        )
        disagreement = 1.0 - agreement
        return AgreementExportReport(
            tokens=trace.tokens,
            steps=trace.steps,
            mean_entropy=float(np.mean(features.entropy)),
            mean_agreement=float(np.mean(agreement)),
            mean_agreement_mass=float(np.mean(agreement_mass)),
            mean_consensus_ratio=float(np.mean(consensus_ratio)),
            mean_disagreement=float(np.mean(disagreement)),
        )

    def summary(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> dict[str, float]:
        report = self.report(data)
        return {
            "tokens": float(report.tokens),
            "steps": float(report.steps),
            "mean_entropy": report.mean_entropy,
            "mean_agreement": report.mean_agreement,
            "mean_agreement_mass": report.mean_agreement_mass,
            "mean_consensus_ratio": report.mean_consensus_ratio,
            "mean_disagreement": report.mean_disagreement,
        }


__all__ = [
    "AgreementExportConfig",
    "AgreementExportModel",
    "AgreementExportReport",
    "AgreementExportTrace",
]
