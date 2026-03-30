from __future__ import annotations

from model import StatisticalMemoryModel


SMOKE_CORPUS = (
    "the same fragment repeats, then a local variation appears.\n"
    "the same fragment repeats, then a local variation appears.\n"
    "the same fragment repeats, then a local variation appears.\n"
) * 2


def main() -> None:
    model = StatisticalMemoryModel.build()
    fit = model.fit(SMOKE_CORPUS)
    score = model.score(SMOKE_CORPUS[:192])

    print("project:", "statistical_memory")
    print("ngram train tokens:", fit.ngram.tokens)
    print("exact train tokens:", fit.exact.tokens)
    print("ngram bits/byte:", round(score.ngram_bits_per_byte, 4))
    print("exact bits/byte:", round(score.exact_bits_per_byte, 4))
    print("mixed bits/byte:", round(score.mixed_bits_per_byte, 4))


if __name__ == "__main__":
    main()
