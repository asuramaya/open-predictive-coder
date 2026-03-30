from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

FAMILY_ROOT = Path(__file__).resolve().parents[1]
if str(FAMILY_ROOT) not in sys.path:
    sys.path.insert(0, str(FAMILY_ROOT))

from open_predictive_coder import FrozenReadoutExpert, OscillatoryMemoryConfig, OscillatoryMemorySubstrate

from shared import CausalReplicaBase, ExpertMixtureModel, build_linear_memory_expert


class OscillatoryStabilityView:
    def __init__(self, config: OscillatoryMemoryConfig):
        self.config = config
        self._decay_width = len(config.decay_rates) * config.embedding_dim
        self.feature_dim = 9

    def encode(self, state: np.ndarray, previous_state: np.ndarray | None = None) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.config.state_dim,):
            raise ValueError("state does not match oscillatory substrate state_dim")
        decay = state[: self._decay_width].reshape(len(self.config.decay_rates), self.config.embedding_dim)
        oscillatory = state[self._decay_width :].reshape(
            self.config.oscillatory_bank_count,
            2,
            self.config.embedding_dim,
        )
        oscillatory_abs = np.abs(oscillatory)
        previous = None if previous_state is None else np.asarray(previous_state, dtype=np.float64)
        drift = 0.0 if previous is None else float(np.mean(np.abs(state - previous)))
        return np.asarray(
            [
                float(np.mean(decay)),
                float(np.mean(np.square(decay))),
                float(np.mean(oscillatory[:, 0])),
                float(np.mean(oscillatory[:, 1])),
                float(np.mean(np.square(oscillatory))),
                float(np.mean(np.abs(oscillatory[:, 0] - oscillatory[:, 1]))),
                drift,
                float(np.max(oscillatory_abs)) if oscillatory.size else 0.0,
                float(np.mean(np.linalg.norm(oscillatory, axis=2))) if oscillatory.size else 0.0,
            ],
            dtype=np.float64,
        )


def build_oscillatory_stability_expert(
    *,
    name: str,
    seed: int = 43,
    alpha: float = 1e-4,
) -> FrozenReadoutExpert:
    config = OscillatoryMemoryConfig(
        vocabulary_size=256,
        embedding_dim=12,
        decay_rates=(0.32, 0.58, 0.8, 0.92),
        oscillatory_modes=4,
        seed=seed,
    )
    substrate = OscillatoryMemorySubstrate(config)
    view = OscillatoryStabilityView(config)
    return FrozenReadoutExpert(
        name=name,
        substrate=substrate,
        feature_dim=view.feature_dim,
        vocabulary_size=config.vocabulary_size,
        feature_fn=view.encode,
        alpha=alpha,
    )


@dataclass
class MemoryStabilityModel(CausalReplicaBase):
    model: ExpertMixtureModel

    @classmethod
    def build(cls) -> "MemoryStabilityModel":
        compressor = build_linear_memory_expert(
            name="memory_path",
            embedding_dim=12,
            decays=(0.32, 0.58, 0.8, 0.92),
            seed=41,
            alpha=1e-4,
        )
        stability = build_oscillatory_stability_expert(
            name="stability_path",
            seed=43,
            alpha=1e-4,
        )
        return cls(model=ExpertMixtureModel((compressor, stability), alpha=1e-4))
