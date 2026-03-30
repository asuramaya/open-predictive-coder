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
        "packed memory prefers reusable local counts over a flat prior.\n"
        "exact repair speaks when the local continuation is highly supported.\n"
        "the controller should trust memory more when agreement and support are high.\n"
    ) * 4
    fit = model.fit(corpus)
    score = model.score(corpus[:240])
    print(f"train tokens: {fit.ngram.tokens}")
    print(f"exact win rate: {fit.exact_win_rate:.4f}")
    for name, weight in zip(fit.feature_names, fit.controller_weights):
        print(f"{name}: {weight:.4f}")
    print(f"prior bits/byte: {score.prior_bits_per_byte:.4f}")
    print(f"exact bits/byte: {score.exact_bits_per_byte:.4f}")
    print(f"mixed bits/byte: {score.mixed_bits_per_byte:.4f}")
    print(f"mean memory trust: {score.mean_memory_trust:.4f}")


if __name__ == "__main__":
    main()
