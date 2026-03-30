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

from model import AgreementExportModel


def main() -> None:
    model = AgreementExportModel()
    corpus = (
        "the agreement export tracks overlap between two local probability streams.\n"
        "the bridge transform stays reusable and does not widen the kernel.\n"
    ) * 6
    report = model.report(corpus)
    print(f"tokens: {report.tokens}")
    print(f"steps: {report.steps}")
    print(f"mean entropy: {report.mean_entropy:.4f}")
    print(f"mean agreement: {report.mean_agreement:.4f}")
    print(f"mean agreement mass: {report.mean_agreement_mass:.4f}")
    print(f"mean consensus ratio: {report.mean_consensus_ratio:.4f}")
    print(f"mean disagreement: {report.mean_disagreement:.4f}")


if __name__ == "__main__":
    main()
