from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BridgeFeatureConfig:
    candidate_count: int = 4
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")


@dataclass(frozen=True)
class BridgeFeatureArrays:
    entropy: np.ndarray
    peak: np.ndarray
    candidate4: np.ndarray
    agreement: np.ndarray
    agreement_mass: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "entropy": self.entropy,
            "peak": self.peak,
            "candidate4": self.candidate4,
            "agreement": self.agreement,
            "agreement_mass": self.agreement_mass,
        }


def _coerce_probabilities(probabilities: np.ndarray | list[float] | tuple[float, ...], vocab_size: int) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("probability arrays must have at least one dimension")
    if array.shape[-1] != vocab_size:
        raise ValueError("last dimension must match vocab_size")
    if np.any(array < 0.0):
        raise ValueError("probabilities must be non-negative")
    totals = array.sum(axis=-1, keepdims=True)
    normalized = np.divide(
        array,
        totals,
        out=np.full_like(array, 1.0 / float(vocab_size)),
        where=totals > 0.0,
    )
    return normalized


def _normalized_entropy(probabilities: np.ndarray, *, epsilon: float) -> np.ndarray:
    vocab_size = probabilities.shape[-1]
    log_vocab = np.log(float(vocab_size))
    entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=-1)
    return entropy / log_vocab if log_vocab > 0.0 else np.zeros_like(entropy)


def _topk_indices(probabilities: np.ndarray, candidate_count: int) -> np.ndarray:
    vocab_size = probabilities.shape[-1]
    k = min(candidate_count, vocab_size)
    order = np.argsort(-probabilities, axis=-1, kind="mergesort")
    return order[..., :k]


def _topk_mass(probabilities: np.ndarray, candidate_count: int) -> np.ndarray:
    indices = _topk_indices(probabilities, candidate_count)
    selected = np.take_along_axis(probabilities, indices, axis=-1)
    return np.sum(selected, axis=-1)


def _topk_mask(probabilities: np.ndarray, candidate_count: int) -> np.ndarray:
    indices = _topk_indices(probabilities, candidate_count)
    mask = np.zeros_like(probabilities, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=-1)
    return mask


def bridge_feature_arrays(
    base_probs: np.ndarray | list[float] | tuple[float, ...],
    proxy_probs: np.ndarray | list[float] | tuple[float, ...],
    vocab_size: int,
    *,
    config: BridgeFeatureConfig | None = None,
) -> BridgeFeatureArrays:
    config = config or BridgeFeatureConfig()
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")

    base = _coerce_probabilities(base_probs, vocab_size)
    proxy = _coerce_probabilities(proxy_probs, vocab_size)
    if base.shape != proxy.shape:
        raise ValueError("base_probs and proxy_probs must have the same shape")

    entropy = 0.5 * (
        _normalized_entropy(base, epsilon=config.epsilon) + _normalized_entropy(proxy, epsilon=config.epsilon)
    )
    peak = 0.5 * (np.max(base, axis=-1) + np.max(proxy, axis=-1))

    candidate_count = min(config.candidate_count, vocab_size)
    candidate4 = 0.5 * (_topk_mass(base, candidate_count) + _topk_mass(proxy, candidate_count))

    agreement = np.sum(np.minimum(base, proxy), axis=-1)

    base_mask = _topk_mask(base, candidate_count)
    proxy_mask = _topk_mask(proxy, candidate_count)
    shared_mask = base_mask & proxy_mask
    agreement_mass = 0.5 * (
        np.sum(base * shared_mask, axis=-1) + np.sum(proxy * shared_mask, axis=-1)
    )

    return BridgeFeatureArrays(
        entropy=np.asarray(entropy, dtype=np.float64),
        peak=np.asarray(peak, dtype=np.float64),
        candidate4=np.asarray(candidate4, dtype=np.float64),
        agreement=np.asarray(agreement, dtype=np.float64),
        agreement_mass=np.asarray(agreement_mass, dtype=np.float64),
    )


__all__ = [
    "BridgeFeatureArrays",
    "BridgeFeatureConfig",
    "bridge_feature_arrays",
]
