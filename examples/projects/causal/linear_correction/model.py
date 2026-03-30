from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

FAMILY_ROOT = Path(__file__).resolve().parents[1]
if str(FAMILY_ROOT) not in sys.path:
    sys.path.insert(0, str(FAMILY_ROOT))

from shared import CausalReplicaBase, ExpertMixtureModel, build_echo_correction_expert, build_linear_memory_expert


@dataclass
class LinearCorrectionModel(CausalReplicaBase):
    model: ExpertMixtureModel

    @classmethod
    def build(cls) -> "LinearCorrectionModel":
        linear = build_linear_memory_expert(
            name="linear_path",
            embedding_dim=14,
            decays=(0.22, 0.45, 0.72, 0.88, 0.96),
            seed=51,
            alpha=1e-4,
        )
        correction = build_echo_correction_expert(
            name="correction_path",
            size=48,
            seed=53,
            alpha=1e-4,
        )
        return cls(model=ExpertMixtureModel((linear, correction), alpha=1e-4))
