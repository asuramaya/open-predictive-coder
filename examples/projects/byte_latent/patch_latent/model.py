from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_predictive_coder import (
    ByteLatentFeatureView,
    FitReport,
    LatentConfig,
    LatentObservation,
    LearnedSegmenter,
    LearnedSegmenterConfig,
    PatchPooler,
    PatchPoolerConfig,
    RidgeReadout,
    SequenceReport,
    ensure_tokens,
)
from open_predictive_coder.metrics import bits_per_byte_from_logits, softmax
from open_predictive_coder.patch_latent_blocks import (
    GlobalLocalBridge,
    GlobalLocalBridgeConfig,
    LocalByteEncoder,
    LocalByteEncoderConfig,
)
from open_predictive_coder.patching import AdaptiveSegmenter


def _normalized_matrix(rng: np.random.Generator, rows: int, cols: int, scale: float = 1.0) -> np.ndarray:
    matrix = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
    return (matrix / np.sqrt(max(cols, 1))) * scale


@dataclass(frozen=True)
class PatchLatentConfig:
    vocabulary_size: int = 256
    token_view_dim: int = 24
    encoder_state_dim: int = 48
    segmenter: LearnedSegmenterConfig = field(
        default_factory=lambda: LearnedSegmenterConfig(
            target_patch_size=4,
            min_patch_size=2,
            max_patch_size=10,
            threshold=0.52,
            target_regularization=0.1,
        )
    )
    latent: LatentConfig = field(
        default_factory=lambda: LatentConfig(
            latent_dim=24,
            global_dim=24,
            reservoir_features=24,
            bridge_scale=0.25,
            global_update_scale=0.3,
            readout_l2=1e-4,
        )
    )
    pooler: PatchPoolerConfig = field(
        default_factory=lambda: PatchPoolerConfig(mode="mix", mix_weight=0.5)
    )
    encoder_output_l2: float = 1e-4
    boundary_warmup_epochs: int = 4
    boundary_refine_epochs: int = 2
    seed: int = 7

    def __post_init__(self) -> None:
        if self.vocabulary_size != 256:
            raise ValueError("PatchLatentConfig expects byte vocabulary size 256")
        if self.token_view_dim < 4:
            raise ValueError("token_view_dim must be >= 4")
        if self.encoder_state_dim < self.token_view_dim:
            raise ValueError("encoder_state_dim must be >= token_view_dim")
        if self.latent.reservoir_features != self.token_view_dim:
            raise ValueError("latent.reservoir_features must match token_view_dim")
        if self.latent.latent_dim != self.token_view_dim:
            raise ValueError("latent.latent_dim must match token_view_dim")
        if self.boundary_warmup_epochs < 0:
            raise ValueError("boundary_warmup_epochs must be >= 0")
        if self.boundary_refine_epochs < 0:
            raise ValueError("boundary_refine_epochs must be >= 0")
        if self.encoder_output_l2 < 0.0:
            raise ValueError("encoder_output_l2 must be >= 0")


@dataclass(frozen=True)
class PatchLatentTrace:
    features: np.ndarray
    targets: np.ndarray
    boundaries: np.ndarray
    tokens: int
    patches: int
    mean_patch_size: float
    compression_ratio: float
    mean_surprise: float
    mean_boundary_probability: float


@dataclass
class PatchLatentState:
    global_state: np.ndarray
    local_state: np.ndarray
    previous_local: np.ndarray | None
    patch_views: list[np.ndarray]
    patch_length: int
    last_latent: np.ndarray
    steps: int
    patches: int


@dataclass(frozen=True)
class _TraceRun:
    trace: PatchLatentTrace
    bridge_inputs: np.ndarray
    local_targets: np.ndarray


class PatchLatentByteModel:
    def __init__(self, config: PatchLatentConfig | None = None):
        self.config = config or PatchLatentConfig()
        self.encoder = LocalByteEncoder(
            LocalByteEncoderConfig(
                vocabulary_size=self.config.vocabulary_size,
                local_dim=self.config.token_view_dim,
                state_dim=self.config.encoder_state_dim,
                output_dim=self.config.token_view_dim,
                output_l2=self.config.encoder_output_l2,
                seed=self.config.seed + 11,
            )
        )
        self.segmenter = LearnedSegmenter(self.config.segmenter)
        self.pooler = PatchPooler(self.config.pooler)
        self.bridge = GlobalLocalBridge(
            GlobalLocalBridgeConfig(
                global_dim=self.config.latent.global_dim,
                latent_dim=self.config.latent.latent_dim,
                local_dim=self.config.token_view_dim,
                seed=self.config.seed + 13,
            )
        )
        rng = np.random.default_rng(self.config.seed + 17)
        self.global_recurrent = _normalized_matrix(
            rng,
            rows=self.config.latent.global_dim,
            cols=self.config.latent.global_dim,
            scale=0.8,
        )
        self.global_input = _normalized_matrix(
            rng,
            rows=self.config.latent.global_dim,
            cols=self.config.latent.latent_dim,
            scale=self.config.latent.global_update_scale,
        )
        self.feature_view = ByteLatentFeatureView(max_patch_size=self.config.segmenter.max_patch_size)
        self.readout = RidgeReadout(
            input_dim=self.feature_view.feature_dim(self.config.latent),
            output_dim=self.config.vocabulary_size,
            alpha=self.config.latent.readout_l2,
        )

    @property
    def feature_dim(self) -> int:
        return self.feature_view.feature_dim(self.config.latent)

    def _coerce_sequences(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> tuple[np.ndarray, ...]:
        if isinstance(data, (str, bytes, bytearray, memoryview, np.ndarray)):
            return (ensure_tokens(data),)
        if isinstance(data, Sequence) and data and all(isinstance(item, int) for item in data):
            return (ensure_tokens(data),)
        if isinstance(data, Sequence):
            return tuple(ensure_tokens(item) for item in data)
        return (ensure_tokens(data),)

    def _initial_state(self) -> PatchLatentState:
        return PatchLatentState(
            global_state=np.zeros(self.config.latent.global_dim, dtype=np.float64),
            local_state=self.encoder.initial_state(),
            previous_local=None,
            patch_views=[],
            patch_length=0,
            last_latent=np.zeros(self.config.latent.latent_dim, dtype=np.float64),
            steps=0,
            patches=0,
        )

    def _current_patch_summary(self, state: PatchLatentState) -> np.ndarray:
        if not state.patch_views:
            return np.zeros(self.config.token_view_dim, dtype=np.float64)
        return self.pooler.pool(np.vstack(state.patch_views))

    def _step(
        self,
        state: PatchLatentState,
        token: int,
        *,
        train_segmenter: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, LatentObservation, float]:
        local_view, next_local_state = self.encoder.step(int(token), state.local_state)
        bridge_input = self.bridge.stack_state(state.global_state, state.last_latent)
        predicted_view = self.bridge.predict_batch(bridge_input[None, :])[0]

        if state.previous_local is None:
            novelty = 0.0
        else:
            novelty = float(np.mean(np.abs(local_view - state.previous_local)))
        prediction_error = local_view - predicted_view
        surprise = float(np.mean(np.abs(prediction_error)))

        state.patch_views.append(local_view.copy())
        state.patch_length += 1
        patch_summary = self._current_patch_summary(state)

        decision = self.segmenter.decide(
            state.patch_length,
            novelty=novelty,
            surprise=surprise,
            train=train_segmenter,
        )
        boundary = decision.boundary
        latent = state.last_latent.copy()
        pre_boundary_patch_length = state.patch_length

        if boundary:
            latent = patch_summary.copy()
            state.global_state = np.tanh((self.global_recurrent @ state.global_state) + (self.global_input @ latent))
            state.last_latent = latent.copy()
            state.patch_views.clear()
            state.patch_length = 0
            state.patches += 1

        observation = LatentObservation(
            local_view=local_view.copy(),
            predicted_view=predicted_view.copy(),
            prediction_error=prediction_error.copy(),
            patch_summary=patch_summary.copy(),
            global_state=state.global_state.copy(),
            latent=state.last_latent.copy(),
            novelty=novelty,
            patch_length=pre_boundary_patch_length,
            boundary=boundary,
        )
        feature = self.feature_view.encode(observation)
        state.local_state = next_local_state
        state.previous_local = local_view.copy()
        state.steps += 1
        return feature, bridge_input, observation, decision.probability

    def _trace_sequence(
        self,
        sequence: np.ndarray,
        *,
        train_segmenter: bool = False,
        collect_bridge: bool = False,
    ) -> _TraceRun:
        tokens = ensure_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        state = self._initial_state()
        features: list[np.ndarray] = []
        boundaries: list[float] = []
        surprises: list[float] = []
        boundary_probabilities: list[float] = []
        bridge_inputs: list[np.ndarray] = []
        local_targets: list[np.ndarray] = []

        for token in tokens[:-1]:
            feature, bridge_input, observation, probability = self._step(
                state,
                int(token),
                train_segmenter=train_segmenter,
            )
            features.append(feature)
            boundaries.append(1.0 if observation.boundary else 0.0)
            surprises.append(float(np.mean(np.abs(observation.prediction_error))))
            boundary_probabilities.append(probability)
            if collect_bridge:
                bridge_inputs.append(bridge_input)
                local_targets.append(observation.local_view.copy())

        total_tokens = int(tokens.size)
        segment_stats = AdaptiveSegmenter.summarize(total_tokens, state.patches)
        trace = PatchLatentTrace(
            features=np.vstack(features),
            targets=tokens[1:].astype(np.int64, copy=False),
            boundaries=np.asarray(boundaries, dtype=np.float64),
            tokens=total_tokens,
            patches=state.patches,
            mean_patch_size=segment_stats.mean_patch_size,
            compression_ratio=segment_stats.compression_ratio,
            mean_surprise=float(np.mean(surprises)),
            mean_boundary_probability=float(np.mean(boundary_probabilities)),
        )
        if collect_bridge:
            bridge_input_array = np.vstack(bridge_inputs) if bridge_inputs else np.zeros((0, self.bridge.input_dim))
            local_target_array = np.vstack(local_targets) if local_targets else np.zeros((0, self.config.token_view_dim))
        else:
            bridge_input_array = np.zeros((0, self.bridge.input_dim), dtype=np.float64)
            local_target_array = np.zeros((0, self.config.token_view_dim), dtype=np.float64)
        return _TraceRun(
            trace=trace,
            bridge_inputs=bridge_input_array,
            local_targets=local_target_array,
        )

    def _fit_local_encoder(self, sequences: tuple[np.ndarray, ...]) -> None:
        hidden_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        for sequence in sequences:
            tokens = ensure_tokens(sequence)
            if tokens.size < 2:
                continue
            hidden, _ = self.encoder.hidden_states(tokens[:-1])
            next_embeddings = self.encoder.embedding[tokens[1:]]
            hidden_batches.append(hidden)
            target_batches.append(next_embeddings)
        if hidden_batches:
            self.encoder.fit_output(
                np.concatenate(hidden_batches, axis=0),
                np.concatenate(target_batches, axis=0),
            )

    def _warm_segmenter(self, sequences: tuple[np.ndarray, ...], *, epochs: int) -> None:
        if epochs < 1:
            return
        for _ in range(epochs):
            for sequence in sequences:
                self._trace_sequence(sequence, train_segmenter=True, collect_bridge=False)

    def trace(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> PatchLatentTrace:
        return self._trace_sequence(ensure_tokens(sequence)).trace

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> FitReport:
        sequences = self._coerce_sequences(data)
        self._fit_local_encoder(sequences)
        self._warm_segmenter(sequences, epochs=self.config.boundary_warmup_epochs)

        bridge_inputs: list[np.ndarray] = []
        bridge_targets: list[np.ndarray] = []
        for sequence in sequences:
            run = self._trace_sequence(sequence, collect_bridge=True)
            if run.bridge_inputs.shape[0] > 0:
                bridge_inputs.append(run.bridge_inputs)
                bridge_targets.append(run.local_targets)
        if bridge_inputs:
            self.bridge.fit(
                np.concatenate(bridge_inputs, axis=0),
                np.concatenate(bridge_targets, axis=0),
            )

        self._warm_segmenter(sequences, epochs=self.config.boundary_refine_epochs)

        feature_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        total_tokens = 0
        total_patches = 0
        for sequence in sequences:
            run = self._trace_sequence(sequence)
            feature_batches.append(run.trace.features)
            target_batches.append(run.trace.targets)
            total_tokens += run.trace.tokens
            total_patches += run.trace.patches

        design = np.concatenate(feature_batches, axis=0)
        targets = np.concatenate(target_batches, axis=0)
        self.readout.fit(design, targets)
        logits = self.readout.logits(design)
        compression = total_tokens / max(total_patches, 1)
        return FitReport(
            sequences=len(sequences),
            tokens=total_tokens,
            patches=total_patches,
            mean_patch_size=compression,
            compression_ratio=compression,
            train_bits_per_byte=bits_per_byte_from_logits(logits, targets),
        )

    def score(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> SequenceReport:
        trace = self.trace(sequence)
        logits = self.readout.logits(trace.features)
        return SequenceReport(
            tokens=trace.tokens,
            patches=trace.patches,
            mean_patch_size=trace.mean_patch_size,
            compression_ratio=trace.compression_ratio,
            bits_per_byte=bits_per_byte_from_logits(logits, trace.targets),
        )

    def predict_proba(self, prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
        tokens = ensure_tokens(prompt)
        if tokens.size < 1:
            raise ValueError("prompt must contain at least one token")

        state = self._initial_state()
        feature = None
        for token in tokens:
            feature, _, _, _ = self._step(state, int(token))
        assert feature is not None
        return self.readout.probabilities(feature[None, :])[0]

    def generate(
        self,
        prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
        *,
        steps: int,
        temperature: float = 1.0,
        greedy: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        if steps < 0:
            raise ValueError("steps must be >= 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        tokens = ensure_tokens(prompt)
        if tokens.size < 1:
            raise ValueError("prompt must contain at least one token")

        rng = np.random.default_rng(seed)
        state = self._initial_state()
        feature = None
        output = tokens.astype(np.uint8, copy=True).tolist()
        for token in tokens:
            feature, _, _, _ = self._step(state, int(token))
        assert feature is not None

        for _ in range(steps):
            logits = self.readout.logits(feature[None, :])[0]
            if greedy:
                next_token = int(np.argmax(logits))
            else:
                probs = softmax((logits / temperature)[None, :], axis=-1)[0]
                next_token = int(rng.choice(self.config.vocabulary_size, p=probs))
            output.append(next_token)
            feature, _, _, _ = self._step(state, next_token)
        return np.asarray(output, dtype=np.uint8)


__all__ = [
    "PatchLatentByteModel",
    "PatchLatentConfig",
    "PatchLatentTrace",
]
