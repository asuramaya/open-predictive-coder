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
    sample = (
        "bridge features compare a causal probability stream against a proxy stream.\n"
        "the example is intentionally small and project-local.\n"
    )
    report = model.report(sample)
    config = model.config

    print("vocabulary_size:", config.vocabulary_size)
    print("hidden_dim:", config.hidden_dim)
    print("proxy_window:", config.proxy_window)
    print("candidate_count:", config.bridge.candidate_count)
    print("tokens:", report.tokens)
    print("steps:", report.steps)
    print("mean_entropy:", round(report.mean_entropy, 4))
    print("mean_peak:", round(report.mean_peak, 4))
    print("mean_candidate4:", round(report.mean_candidate4, 4))
    print("mean_agreement:", round(report.mean_agreement, 4))
    print("mean_agreement_mass:", round(report.mean_agreement_mass, 4))


if __name__ == "__main__":
    main()
