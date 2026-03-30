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

from model import ProgramControllerModel


def main() -> None:
    model = ProgramControllerModel.build()
    corpus = (
        "the controller should choose between prior, exact repair, and local repair spans.\n"
        "when exact context is strong, the repair program can take over.\n"
        "when local statistics dominate, the prior route should stay active.\n"
    ) * 4
    fit = model.fit(corpus)
    score = model.score(corpus[:240])
    print("project: program_controller")
    print(f"train tokens: {fit.ngram.tokens}")
    print(f"route accuracy: {fit.route_accuracy:.4f}")
    print(f"repair accuracy: {fit.repair_accuracy:.4f}")
    print(f"mean repair strength: {fit.mean_repair_strength:.4f}")
    print(f"repair spans: {fit.repair_span_count}")
    print(f"prior bits/byte: {score.prior_bits_per_byte:.4f}")
    print(f"exact bits/byte: {score.exact_bits_per_byte:.4f}")
    print(f"repair bits/byte: {score.repair_bits_per_byte:.4f}")
    print(f"mixed bits/byte: {score.mixed_bits_per_byte:.4f}")
    print(f"route entropy: {score.mean_route_entropy:.4f}")


if __name__ == "__main__":
    main()
