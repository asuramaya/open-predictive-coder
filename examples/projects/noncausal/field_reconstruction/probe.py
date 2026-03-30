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
        "noncausal reconstruction uses both left and right context.\n"
        "noncausal reconstruction uses both left and right context.\n"
        "the replay span should capture repeated field structure.\n"
    ) * 4
    fit = model.fit(corpus)
    report = model.score(corpus[:192])

    print("project:", "field_reconstruction")
    print("forward contexts:", sum(fit.forward.contexts_by_order))
    print("reverse contexts:", sum(fit.reverse.contexts_by_order))
    print("bidirectional neighborhoods:", fit.bidirectional_context.neighborhood_count)
    print("replay spans:", report.replay_span_count)
    print("agreement rate:", round(report.agreement_rate, 4))
    print("blended bits/byte:", round(report.blended_bits_per_byte, 4))
    print("reconstructed prefix:", report.reconstructed_text[:48])


if __name__ == "__main__":
    main()
