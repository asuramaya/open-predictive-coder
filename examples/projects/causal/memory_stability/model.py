from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

FAMILY_ROOT = Path(__file__).resolve().parents[1]
if str(FAMILY_ROOT) not in sys.path:
    sys.path.insert(0, str(FAMILY_ROOT))

from shared import CausalReplicaBase, ExpertMixtureModel, build_hierarchical_stability_expert, build_linear_memory_expert


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
        stability = build_hierarchical_stability_expert(
            name="stability_path",
            seed=43,
            alpha=1e-4,
        )
        return cls(model=ExpertMixtureModel((compressor, stability), alpha=1e-4))
