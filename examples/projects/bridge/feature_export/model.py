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
class FeatureExportConfig:
    vocabulary_size: int = 256
    hidden_dim: int = 48
    proxy_window: int = 4
    bridge: BridgeFeatureConfig = BridgeFeatureConfig(candidate_count=4)
    seed: int = 17

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.hidden_dim < 4:
            raise ValueError("hidden_dim must be >= 4")
        if self.proxy_window < 1:
            raise ValueError("proxy_window must be >= 1")


@dataclass(frozen=True)
class FeatureExportTrace:
    tokens: int
    source_probs: np.ndarray
    proxy_probs: np.ndarray
    features: BridgeFeatureArrays

    @property
    def steps(self) -> int:
        return int(self.source_probs.shape[0])


@dataclass(frozen=True)
class FeatureExportReport:
    tokens: int
    steps: int
    mean_entropy: float
    mean_peak: float
    mean_candidate4: float
    mean_agreement: float
    mean_agreement_mass: float


class FeatureExportModel:
    def __init__(self, config: FeatureExportConfig | None = None):
        self.config = config or FeatureExportConfig()
        rng = np.random.default_rng(self.config.seed)
        self.embeddings = rng.normal(
            loc=0.0,
            scale=0.4,
            size=(self.config.vocabulary_size, self.config.hidden_dim),
        ).astype(np.float64)
        self.recurrent = _normalized_matrix(rng, self.config.hidden_dim, self.config.hidden_dim, scale=0.7)
        self.source_head = _normalized_matrix(rng, self.config.vocabulary_size, self.config.hidden_dim, scale=0.9)
        self.proxy_head = _normalized_matrix(rng, self.config.vocabulary_size, self.config.hidden_dim, scale=0.9)
        self.bias = rng.normal(loc=0.0, scale=0.05, size=(self.config.vocabulary_size,)).astype(np.float64)

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
        window: list[np.ndarray] = []
        source_probs: list[np.ndarray] = []
        proxy_probs: list[np.ndarray] = []

        for token in tokens:
            embedded = self.embeddings[int(token)]
            hidden = np.tanh((self.recurrent @ hidden) + embedded)
            window.append(hidden.copy())
            if len(window) > self.config.proxy_window:
                window.pop(0)

            source_logits = (self.source_head @ hidden) + self.bias
            proxy_state = np.mean(np.vstack(window), axis=0)
            proxy_logits = (self.proxy_head @ proxy_state) + self.bias

            source_probs.append(_softmax(source_logits))
            proxy_probs.append(_softmax(proxy_logits))

        if not source_probs:
            empty = np.zeros((0, self.config.vocabulary_size), dtype=np.float64)
            return empty, empty
        return np.vstack(source_probs), np.vstack(proxy_probs)

    def export(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> FeatureExportTrace:
        tokens = self._coerce_tokens(data)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")
        source_probs, proxy_probs = self._scan_probabilities(tokens[:-1])
        features = bridge_feature_arrays(source_probs, proxy_probs, self.config.vocabulary_size, config=self.config.bridge)
        return FeatureExportTrace(
            tokens=int(tokens.size),
            source_probs=source_probs,
            proxy_probs=proxy_probs,
            features=features,
        )

    def report(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> FeatureExportReport:
        trace = self.export(data)
        features = trace.features
        return FeatureExportReport(
            tokens=trace.tokens,
            steps=trace.steps,
            mean_entropy=float(np.mean(features.entropy)),
            mean_peak=float(np.mean(features.peak)),
            mean_candidate4=float(np.mean(features.candidate4)),
            mean_agreement=float(np.mean(features.agreement)),
            mean_agreement_mass=float(np.mean(features.agreement_mass)),
        )

    def summary(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> dict[str, float]:
        report = self.report(data)
        return {
            "tokens": float(report.tokens),
            "steps": float(report.steps),
            "mean_entropy": report.mean_entropy,
            "mean_peak": report.mean_peak,
            "mean_candidate4": report.mean_candidate4,
            "mean_agreement": report.mean_agreement,
            "mean_agreement_mass": report.mean_agreement_mass,
        }


__all__ = [
    "FeatureExportConfig",
    "FeatureExportModel",
    "FeatureExportReport",
    "FeatureExportTrace",
]
