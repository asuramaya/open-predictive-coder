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
        "controller logic should reveal where repair is useful.\n"
        "controller logic should reveal where repair is useful.\n"
        "the exact path can dominate whenever support is repeated.\n"
    ) * 4
    fit = model.fit(corpus)
    score = model.score(corpus)
    print("project: program_controller")
    print(f"train tokens: {fit.ngram.tokens}")
    print(f"route accuracy: {fit.route_accuracy:.4f}")
    print(f"repair spans: {score.repair_span_count}")
    print(f"mean route entropy: {score.mean_route_entropy:.4f}")
    print(f"mixed bits/byte: {score.mixed_bits_per_byte:.4f}")


if __name__ == "__main__":
    main()
