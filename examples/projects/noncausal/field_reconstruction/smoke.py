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

from model import FieldReconstructionModel


def main() -> None:
    model = FieldReconstructionModel.build()
    corpus = (
        "the field is reconstructed from both directions.\n"
        "the field is reconstructed from both directions.\n"
        "the field is reconstructed from both directions.\n"
    )
    fit = model.fit(corpus)
    report = model.score(corpus)

    print("project:", "field_reconstruction")
    print("tokens:", report.tokens)
    print("replay spans:", report.replay_span_count)
    print("agreement rate:", round(report.agreement_rate, 4))
    print("replay rate:", round(report.replay_rate, 4))
    print("blended bits/byte:", round(report.blended_bits_per_byte, 4))
    print("forward train tokens:", fit.forward.tokens)
    print("reverse train tokens:", fit.reverse.tokens)


if __name__ == "__main__":
    main()
