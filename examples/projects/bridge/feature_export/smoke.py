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

from model import FeatureExportModel


def main() -> None:
    model = FeatureExportModel()
    corpus = (
        "the export path compares a source stream and a proxy stream.\n"
        "the shared bridge transform remains local and reusable.\n"
    ) * 6
    report = model.report(corpus)
    print(f"tokens: {report.tokens}")
    print(f"steps: {report.steps}")
    print(f"mean entropy: {report.mean_entropy:.4f}")
    print(f"mean peak: {report.mean_peak:.4f}")
    print(f"mean candidate4: {report.mean_candidate4:.4f}")
    print(f"mean agreement: {report.mean_agreement:.4f}")
    print(f"mean agreement mass: {report.mean_agreement_mass:.4f}")


if __name__ == "__main__":
    main()
