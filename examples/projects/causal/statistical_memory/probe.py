from __future__ import annotations

from model import StatisticalMemoryModel


PROBE_CORPUS = (
    "statistical memory combines broad n-gram pressure with exact context repair.\n"
    "statistical memory combines broad n-gram pressure with exact context repair.\n"
    "the exact branch should lock onto repeated phrases while the n-gram branch stays smooth.\n"
)


def main() -> None:
    model = StatisticalMemoryModel.build()
    fit = model.fit(PROBE_CORPUS)
    trace = model.trace(PROBE_CORPUS[:160])
    score = model.score(PROBE_CORPUS[:160])

    print("project:", "statistical_memory")
    print("ngram tokens:", fit.ngram.tokens)
    print("exact contexts:", sum(fit.exact.contexts_by_order))
    print("trace steps:", trace.steps)
    print("mixed bits/byte:", round(score.mixed_bits_per_byte, 4))
    print("exact order:", score.exact_order)


if __name__ == "__main__":
    main()
