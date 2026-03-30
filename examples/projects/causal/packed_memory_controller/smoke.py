from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from model import PackedMemoryControllerModel


def main() -> None:
    model = PackedMemoryControllerModel.build()
    corpus = (
        "memory first controllers need confidence, not just larger tables.\n"
        "agreement between packed memory and exact repair is a useful signal.\n"
    ) * 6
    fit = model.fit(corpus)
    score = model.score(corpus)
    print("project: packed_memory_controller")
    print(f"train tokens: {fit.ngram.tokens}")
    print(f"prior bits/byte: {score.prior_bits_per_byte:.4f}")
    print(f"exact bits/byte: {score.exact_bits_per_byte:.4f}")
    print(f"mixed bits/byte: {score.mixed_bits_per_byte:.4f}")
    print(f"mean memory trust: {score.mean_memory_trust:.4f}")
    print(f"mean agreement mass: {score.mean_agreement_mass:.4f}")
    print(f"mean candidate4: {score.mean_candidate4:.4f}")


if __name__ == "__main__":
    main()
