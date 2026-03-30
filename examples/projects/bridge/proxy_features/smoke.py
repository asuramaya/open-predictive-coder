from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import BridgeProxyConfig, BridgeProxyModel


def main() -> None:
    model = BridgeProxyModel(BridgeProxyConfig())
    corpus = (
        "bridge features should summarize how a causal stream differs from a proxy stream.\n"
        "the example stays simple and project-local.\n"
    ) * 6
    report = model.report(corpus)

    print("tokens:", report.tokens)
    print("steps:", report.steps)
    print("mean entropy:", round(report.mean_entropy, 4))
    print("mean peak:", round(report.mean_peak, 4))
    print("mean candidate4:", round(report.mean_candidate4, 4))
    print("mean agreement:", round(report.mean_agreement, 4))
    print("mean agreement mass:", round(report.mean_agreement_mass, 4))


if __name__ == "__main__":
    main()
