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

from open_predictive_coder import (  # noqa: E402
    ArtifactAccounting,
    BidirectionalContextConfig,
    BidirectionalContextProbe,
    BidirectionalContextStats,
    ReplaySpan,
    SpanSelectionConfig,
    ensure_tokens,
    make_artifact_accounting,
    make_replay_span,
    replay_spans_from_scores,
)


def _coerce_tokens(data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> np.ndarray:
    return ensure_tokens(data).astype(np.uint8, copy=False)


def _empty_positions() -> np.ndarray:
    return np.empty(0, dtype=np.int32)


def _score_neighborhoods(stats: BidirectionalContextStats) -> np.ndarray:
    neighborhoods = tuple(sorted(stats.neighborhoods, key=lambda item: item.position))
    if not neighborhoods:
        return np.empty(0, dtype=np.float64)

    scores = np.empty(len(neighborhoods), dtype=np.float64)
    for index, neighborhood in enumerate(neighborhoods):
        candidate_confidence = 1.0 / float(max(neighborhood.candidate_count, 1))
        support_total = float(neighborhood.left_support + neighborhood.right_support)
        pair_support = float(neighborhood.pair_support)
        support_ratio = pair_support / max(support_total, 1.0)
        determinism_bonus = 0.1 if neighborhood.deterministic else 0.0
        score = 0.55 * candidate_confidence + 0.35 * support_ratio + determinism_bonus
        scores[index] = float(np.clip(score, 0.0, 1.0))
    return scores


def _positions_from_spans(spans: tuple[ReplaySpan, ...]) -> np.ndarray:
    if not spans:
        return _empty_positions()
    positions = [np.arange(span.start, span.stop, dtype=np.int32) for span in spans if span.stop > span.start]
    if not positions:
        return _empty_positions()
    return np.concatenate(positions)


def _layout_bytes(positions: np.ndarray, tokens: np.ndarray, scores: np.ndarray, spans: tuple[ReplaySpan, ...]) -> int:
    dictionary_bytes = int(positions.size * 16)
    span_bytes = int(len(spans) * 16)
    return int(positions.nbytes + tokens.nbytes + scores.nbytes + dictionary_bytes + span_bytes)


def _compress_sparse_dictionary(tokens: np.ndarray, scores: np.ndarray) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    dictionary: list[tuple[int, int]] = []
    lookup: dict[tuple[int, int], int] = {}
    indices = np.empty(tokens.size, dtype=np.int32)

    for index, (token, score) in enumerate(zip(tokens, scores)):
        signature = (int(token), int(round(float(score) * 1000.0)))
        dictionary_index = lookup.get(signature)
        if dictionary_index is None:
            dictionary_index = len(dictionary)
            lookup[signature] = dictionary_index
            dictionary.append(signature)
        indices[index] = dictionary_index

    return tuple(dictionary), indices


@dataclass(frozen=True)
class PayloadChoiceConfig:
    vocabulary_size: int = 256
    left_order: int = 2
    right_order: int = 2
    sparse_threshold: float = 0.62
    sparse_min_span: int = 2
    sparse_max_gap: int = 1
    payload_bytes_scale: float = 4096.0
    dense_bias: float = 0.0
    sparse_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.left_order < 0:
            raise ValueError("left_order must be >= 0")
        if self.right_order < 0:
            raise ValueError("right_order must be >= 0")
        if not 0.0 <= self.sparse_threshold <= 1.0:
            raise ValueError("sparse_threshold must be within [0, 1]")
        if self.sparse_min_span < 1:
            raise ValueError("sparse_min_span must be >= 1")
        if self.sparse_max_gap < 0:
            raise ValueError("sparse_max_gap must be >= 0")
        if self.payload_bytes_scale <= 0.0:
            raise ValueError("payload_bytes_scale must be > 0")


@dataclass(frozen=True)
class PayloadChoiceLayout:
    kind: str
    positions: np.ndarray
    tokens: np.ndarray
    scores: np.ndarray
    spans: tuple[ReplaySpan, ...]
    dictionary: tuple[tuple[int, int], ...]
    dictionary_entries: int
    payload_bytes: int
    mean_score: float
    density: float


@dataclass(frozen=True)
class PayloadChoiceTrace:
    tokens: np.ndarray
    position_scores: np.ndarray
    bidirectional_context: BidirectionalContextStats
    dense_layout: PayloadChoiceLayout
    sparse_layout: PayloadChoiceLayout
    selected_layout: PayloadChoiceLayout
    dense_utility: float
    sparse_utility: float
    selection_margin: float


@dataclass(frozen=True)
class PayloadChoiceReport:
    tokens: int
    selected_layout: str
    dense_payload_bytes: int
    sparse_payload_bytes: int
    selected_payload_bytes: int
    dense_position_count: int
    sparse_position_count: int
    selected_position_count: int
    dense_dictionary_entries: int
    sparse_dictionary_entries: int
    selected_dictionary_entries: int
    dense_mean_score: float
    sparse_mean_score: float
    selection_margin: float
    deterministic_fraction: float
    mean_candidate_count: float
    coverage_ratio: float
    accounting: ArtifactAccounting


class PayloadChoiceModel:
    def __init__(self, config: PayloadChoiceConfig | None = None) -> None:
        self.config = config or PayloadChoiceConfig()
        self.bidirectional = BidirectionalContextProbe(
            BidirectionalContextConfig(
                left_order=self.config.left_order,
                right_order=self.config.right_order,
            )
        )
        self.span_selection = SpanSelectionConfig(
            threshold=self.config.sparse_threshold,
            min_span=self.config.sparse_min_span,
            max_gap=self.config.sparse_max_gap,
        )

    @classmethod
    def build(cls, config: PayloadChoiceConfig | None = None) -> PayloadChoiceModel:
        return cls(config=config)

    def _dense_layout(self, tokens: np.ndarray, scores: np.ndarray) -> PayloadChoiceLayout:
        positions = np.arange(tokens.size, dtype=np.int32)
        dense_spans = (make_replay_span(0, int(tokens.size), label="dense"),) if tokens.size else ()
        dictionary = tuple((int(position), int(token)) for position, token in zip(positions, tokens))
        payload_bytes = _layout_bytes(positions, tokens, scores, dense_spans)
        return PayloadChoiceLayout(
            kind="dense",
            positions=positions,
            tokens=tokens.copy(),
            scores=scores.copy(),
            spans=dense_spans,
            dictionary=dictionary,
            dictionary_entries=len(dictionary),
            payload_bytes=payload_bytes,
            mean_score=float(np.mean(scores)) if scores.size else 0.0,
            density=1.0 if tokens.size else 0.0,
        )

    def _sparse_layout(self, tokens: np.ndarray, scores: np.ndarray) -> PayloadChoiceLayout:
        spans = replay_spans_from_scores(scores, self.span_selection, label="sparse")
        if not spans and scores.size:
            best = int(np.argmax(scores))
            spans = (make_replay_span(best, best + 1, label="sparse"),)
        positions = _positions_from_spans(spans)
        if positions.size:
            selected_tokens = tokens[positions]
            selected_scores = scores[positions]
        else:
            selected_tokens = np.empty(0, dtype=np.uint8)
            selected_scores = np.empty(0, dtype=np.float64)
        dictionary, dictionary_indices = _compress_sparse_dictionary(selected_tokens, selected_scores)
        payload_bytes = int(dictionary_indices.nbytes + len(dictionary) * 16 + len(spans) * 16)
        return PayloadChoiceLayout(
            kind="sparse",
            positions=positions,
            tokens=selected_tokens,
            scores=selected_scores,
            spans=spans,
            dictionary=dictionary,
            dictionary_entries=len(dictionary),
            payload_bytes=payload_bytes,
            mean_score=float(np.mean(selected_scores)) if selected_scores.size else 0.0,
            density=(positions.size / float(tokens.size)) if tokens.size else 0.0,
        )

    def trace(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> PayloadChoiceTrace:
        tokens = _coerce_tokens(data)
        stats = self.bidirectional.scan(tokens)
        scores = _score_neighborhoods(stats)
        dense_layout = self._dense_layout(tokens, scores)
        sparse_layout = self._sparse_layout(tokens, scores)
        dense_utility = dense_layout.mean_score + self.config.dense_bias - (dense_layout.payload_bytes / self.config.payload_bytes_scale)
        sparse_utility = sparse_layout.mean_score + self.config.sparse_bias - (sparse_layout.payload_bytes / self.config.payload_bytes_scale)
        selected_layout = dense_layout if dense_utility >= sparse_utility else sparse_layout
        return PayloadChoiceTrace(
            tokens=tokens,
            position_scores=scores,
            bidirectional_context=stats,
            dense_layout=dense_layout,
            sparse_layout=sparse_layout,
            selected_layout=selected_layout,
            dense_utility=float(dense_utility),
            sparse_utility=float(sparse_utility),
            selection_margin=float(abs(dense_utility - sparse_utility)),
        )

    def report(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> PayloadChoiceReport:
        trace = self.trace(data)
        accounting = make_artifact_accounting(
            artifact_name="payload_choice",
            artifact_bytes=trace.dense_layout.payload_bytes,
            replay_bytes=trace.selected_layout.payload_bytes,
            replay_spans=trace.selected_layout.spans,
            layout=trace.selected_layout.kind,
            dense_payload_bytes=trace.dense_layout.payload_bytes,
            sparse_payload_bytes=trace.sparse_layout.payload_bytes,
            selected_position_count=int(trace.selected_layout.positions.size),
        )
        return PayloadChoiceReport(
            tokens=int(trace.tokens.size),
            selected_layout=trace.selected_layout.kind,
            dense_payload_bytes=trace.dense_layout.payload_bytes,
            sparse_payload_bytes=trace.sparse_layout.payload_bytes,
            selected_payload_bytes=trace.selected_layout.payload_bytes,
            dense_position_count=int(trace.dense_layout.positions.size),
            sparse_position_count=int(trace.sparse_layout.positions.size),
            selected_position_count=int(trace.selected_layout.positions.size),
            dense_dictionary_entries=int(trace.dense_layout.dictionary_entries),
            sparse_dictionary_entries=int(trace.sparse_layout.dictionary_entries),
            selected_dictionary_entries=int(trace.selected_layout.dictionary_entries),
            dense_mean_score=trace.dense_layout.mean_score,
            sparse_mean_score=trace.sparse_layout.mean_score,
            selection_margin=trace.selection_margin,
            deterministic_fraction=float(trace.bidirectional_context.deterministic_fraction),
            mean_candidate_count=float(trace.bidirectional_context.mean_candidate_size),
            coverage_ratio=float(accounting.coverage_ratio),
            accounting=accounting,
        )

    def fit(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> PayloadChoiceReport:
        return self.report(data)

    def score(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> PayloadChoiceReport:
        return self.report(data)

    def summary(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> dict[str, float]:
        report = self.report(data)
        return {
            "tokens": float(report.tokens),
            "selected_layout_dense": 1.0 if report.selected_layout == "dense" else 0.0,
            "selected_layout_sparse": 1.0 if report.selected_layout == "sparse" else 0.0,
            "dense_payload_bytes": float(report.dense_payload_bytes),
            "sparse_payload_bytes": float(report.sparse_payload_bytes),
            "selected_payload_bytes": float(report.selected_payload_bytes),
            "dense_dictionary_entries": float(report.dense_dictionary_entries),
            "sparse_dictionary_entries": float(report.sparse_dictionary_entries),
            "selected_dictionary_entries": float(report.selected_dictionary_entries),
            "selection_margin": report.selection_margin,
            "deterministic_fraction": report.deterministic_fraction,
            "mean_candidate_count": report.mean_candidate_count,
            "coverage_ratio": report.coverage_ratio,
        }


__all__ = [
    "PayloadChoiceConfig",
    "PayloadChoiceLayout",
    "PayloadChoiceModel",
    "PayloadChoiceReport",
    "PayloadChoiceTrace",
]
