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

from open_predictive_coder.artifacts import ArtifactAccounting
from open_predictive_coder.bidirectional_context import BidirectionalContextConfig, BidirectionalContextProbe, BidirectionalContextStats
from open_predictive_coder.bridge_export import BridgeExportAdapter, BridgeExportConfig
from open_predictive_coder.bridge_features import BridgeFeatureArrays, BridgeFeatureConfig, bridge_feature_arrays
from open_predictive_coder.codecs import ensure_tokens
from open_predictive_coder.metrics import bits_per_byte_from_probabilities
from open_predictive_coder.probability_diagnostics import top2_margin


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _normalized_matrix(rng: np.random.Generator, rows: int, cols: int, scale: float) -> np.ndarray:
    matrix = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
    return (matrix / np.sqrt(max(cols, 1))) * scale


@dataclass(frozen=True)
class TeacherExportConfig:
    vocabulary_size: int = 256
    hidden_dim: int = 48
    proxy_window: int = 4
    attack_stride: int = 5
    attack_shift: int = 17
    bridge: BridgeFeatureConfig = BridgeFeatureConfig(candidate_count=4)
    bridge_export: BridgeExportConfig = BridgeExportConfig(
        candidate_count=4,
        source_names=("teacher", "student"),
    )
    bidirectional_context: BidirectionalContextConfig = BidirectionalContextConfig(left_order=2, right_order=2)
    seed: int = 43

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.hidden_dim < 4:
            raise ValueError("hidden_dim must be >= 4")
        if self.proxy_window < 1:
            raise ValueError("proxy_window must be >= 1")
        if self.attack_stride < 1:
            raise ValueError("attack_stride must be >= 1")
        if self.attack_shift < 1:
            raise ValueError("attack_shift must be >= 1")


@dataclass(frozen=True)
class TeacherExportTrace:
    tokens: int
    clean_tokens: np.ndarray
    attacked_tokens: np.ndarray
    teacher_probs: np.ndarray
    student_probs: np.ndarray
    attacked_probs: np.ndarray
    teacher_labels: np.ndarray
    student_labels: np.ndarray
    attacked_labels: np.ndarray
    features: BridgeFeatureArrays
    clean_context: BidirectionalContextStats
    attacked_context: BidirectionalContextStats

    @property
    def steps(self) -> int:
        return int(self.teacher_probs.shape[0])


@dataclass(frozen=True)
class TeacherExportReport:
    tokens: int
    steps: int
    teacher_bits_per_byte: float
    student_bits_per_byte: float
    attack_bits_per_byte: float
    mean_entropy: float
    mean_peak: float
    mean_candidate4: float
    mean_agreement: float
    mean_agreement_mass: float
    mean_teacher_margin: float
    mean_attack_margin: float
    label_flip_rate: float
    student_label_disagreement: float
    attack_mutation_rate: float
    clean_deterministic_fraction: float
    attacked_deterministic_fraction: float
    deterministic_fraction_drop: float
    accounting: ArtifactAccounting


class TeacherExportModel:
    def __init__(self, config: TeacherExportConfig | None = None):
        self.config = config or TeacherExportConfig()
        rng = np.random.default_rng(self.config.seed)
        self.embeddings = rng.normal(
            loc=0.0,
            scale=0.35,
            size=(self.config.vocabulary_size, self.config.hidden_dim),
        ).astype(np.float64)
        self.recurrent = _normalized_matrix(rng, self.config.hidden_dim, self.config.hidden_dim, scale=0.75)
        self.teacher_head = _normalized_matrix(rng, self.config.vocabulary_size, self.config.hidden_dim, scale=0.95)
        self.student_head = _normalized_matrix(rng, self.config.vocabulary_size, self.config.hidden_dim, scale=0.85)
        self.teacher_bias = rng.normal(loc=0.0, scale=0.03, size=(self.config.vocabulary_size,)).astype(np.float64)
        self.student_bias = rng.normal(loc=0.0, scale=0.03, size=(self.config.vocabulary_size,)).astype(np.float64)
        self.bridge = BridgeExportAdapter(self.config.bridge_export)
        self.context_probe = BidirectionalContextProbe(self.config.bidirectional_context)

    def _coerce_tokens(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> np.ndarray:
        if isinstance(data, (str, bytes, bytearray, memoryview, np.ndarray)):
            return ensure_tokens(data)
        if isinstance(data, Sequence):
            return ensure_tokens(data)
        return ensure_tokens(data)

    def _attack_tokens(self, tokens: np.ndarray) -> np.ndarray:
        attacked = np.asarray(tokens, dtype=np.int64).copy()
        if attacked.size == 0:
            return attacked
        attack_positions = np.arange(self.config.attack_stride - 1, attacked.size, self.config.attack_stride)
        if attack_positions.size == 0:
            return attacked
        attacked[attack_positions] = (attacked[attack_positions] + self.config.attack_shift + attack_positions) % self.config.vocabulary_size
        return attacked

    def _scan_probabilities(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.zeros(self.config.hidden_dim, dtype=np.float64)
        window: list[np.ndarray] = []
        teacher_probs: list[np.ndarray] = []
        student_probs: list[np.ndarray] = []

        for token in tokens:
            embedded = self.embeddings[int(token)]
            hidden = np.tanh((self.recurrent @ hidden) + embedded)
            window.append(hidden.copy())
            if len(window) > self.config.proxy_window:
                window.pop(0)

            teacher_logits = (self.teacher_head @ hidden) + self.teacher_bias
            student_state = np.mean(np.vstack(window), axis=0)
            student_logits = (self.student_head @ student_state) + self.student_bias

            teacher_probs.append(_softmax(teacher_logits))
            student_probs.append(_softmax(student_logits))

        if not teacher_probs:
            empty = np.zeros((0, self.config.vocabulary_size), dtype=np.float64)
            return empty, empty
        return np.vstack(teacher_probs), np.vstack(student_probs)

    def export(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> TeacherExportTrace:
        tokens = self._coerce_tokens(data)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        clean_tokens = tokens.astype(np.int64, copy=False)
        attacked_tokens = self._attack_tokens(clean_tokens)
        clean_context = self.context_probe.scan(clean_tokens)
        attacked_context = self.context_probe.scan(attacked_tokens)

        teacher_probs, student_probs = self._scan_probabilities(clean_tokens[:-1])
        attacked_probs, _ = self._scan_probabilities(attacked_tokens[:-1])
        teacher_labels = np.argmax(teacher_probs, axis=-1)
        student_labels = np.argmax(student_probs, axis=-1)
        attacked_labels = np.argmax(attacked_probs, axis=-1)

        features = bridge_feature_arrays(
            teacher_probs,
            student_probs,
            self.config.vocabulary_size,
            config=self.config.bridge,
        )
        return TeacherExportTrace(
            tokens=int(clean_tokens.size),
            clean_tokens=clean_tokens,
            attacked_tokens=attacked_tokens,
            teacher_probs=teacher_probs,
            student_probs=student_probs,
            attacked_probs=attacked_probs,
            teacher_labels=teacher_labels,
            student_labels=student_labels,
            attacked_labels=attacked_labels,
            features=features,
            clean_context=clean_context,
            attacked_context=attacked_context,
        )

    def report(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> TeacherExportReport:
        trace = self.export(data)
        bridge_report = self.bridge.export(
            trace.teacher_probs,
            trace.student_probs,
            targets=trace.teacher_labels,
            source_names=self.config.bridge_export.source_names,
        )
        attack_bits_per_byte = bits_per_byte_from_probabilities(trace.attacked_probs, trace.teacher_labels)
        teacher_margin = top2_margin(trace.teacher_probs)
        attack_margin = top2_margin(trace.attacked_probs)
        label_flip_rate = float(np.mean(trace.attacked_labels != trace.teacher_labels))
        student_label_disagreement = float(np.mean(trace.student_labels != trace.teacher_labels))
        attack_mutation_rate = float(np.mean(trace.clean_tokens != trace.attacked_tokens))
        clean_deterministic_fraction = float(trace.clean_context.deterministic_fraction)
        attacked_deterministic_fraction = float(trace.attacked_context.deterministic_fraction)
        deterministic_fraction_drop = clean_deterministic_fraction - attacked_deterministic_fraction

        return TeacherExportReport(
            tokens=trace.tokens,
            steps=trace.steps,
            teacher_bits_per_byte=float(
                bridge_report.base_bits_per_byte if bridge_report.base_bits_per_byte is not None else 0.0
            ),
            student_bits_per_byte=float(
                bridge_report.proxy_bits_per_byte if bridge_report.proxy_bits_per_byte is not None else 0.0
            ),
            attack_bits_per_byte=float(attack_bits_per_byte),
            mean_entropy=float(np.mean(trace.features.entropy)),
            mean_peak=float(np.mean(trace.features.peak)),
            mean_candidate4=float(np.mean(trace.features.candidate4)),
            mean_agreement=float(np.mean(trace.features.agreement)),
            mean_agreement_mass=float(np.mean(trace.features.agreement_mass)),
            mean_teacher_margin=float(np.mean(teacher_margin)) if teacher_margin.size else 0.0,
            mean_attack_margin=float(np.mean(attack_margin)) if attack_margin.size else 0.0,
            label_flip_rate=label_flip_rate,
            student_label_disagreement=student_label_disagreement,
            attack_mutation_rate=attack_mutation_rate,
            clean_deterministic_fraction=clean_deterministic_fraction,
            attacked_deterministic_fraction=attacked_deterministic_fraction,
            deterministic_fraction_drop=deterministic_fraction_drop,
            accounting=bridge_report.accounting,
        )

    def summary(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> dict[str, float]:
        report = self.report(data)
        return {
            "tokens": float(report.tokens),
            "steps": float(report.steps),
            "teacher_bits_per_byte": report.teacher_bits_per_byte,
            "student_bits_per_byte": report.student_bits_per_byte,
            "attack_bits_per_byte": report.attack_bits_per_byte,
            "mean_entropy": report.mean_entropy,
            "mean_peak": report.mean_peak,
            "mean_candidate4": report.mean_candidate4,
            "mean_agreement": report.mean_agreement,
            "mean_agreement_mass": report.mean_agreement_mass,
            "mean_teacher_margin": report.mean_teacher_margin,
            "mean_attack_margin": report.mean_attack_margin,
            "label_flip_rate": report.label_flip_rate,
            "student_label_disagreement": report.student_label_disagreement,
            "attack_mutation_rate": report.attack_mutation_rate,
            "clean_deterministic_fraction": report.clean_deterministic_fraction,
            "attacked_deterministic_fraction": report.attacked_deterministic_fraction,
            "deterministic_fraction_drop": report.deterministic_fraction_drop,
        }


__all__ = [
    "TeacherExportConfig",
    "TeacherExportModel",
    "TeacherExportReport",
    "TeacherExportTrace",
]
