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

from model import PayloadChoiceModel


def main() -> None:
    model = PayloadChoiceModel.build()
    corpus = (
        "the payload choice example compares dense and sparse layout policies.\n"
        "the payload choice example compares dense and sparse layout policies.\n"
    ) * 3
    report = model.score(corpus)

    print("project:", "payload_choice")
    print("tokens:", report.tokens)
    print("selected layout:", report.selected_layout)
    print("dense payload bytes:", report.dense_payload_bytes)
    print("sparse payload bytes:", report.sparse_payload_bytes)
    print("coverage ratio:", round(report.coverage_ratio, 4))
    print("selection margin:", round(report.selection_margin, 4))


if __name__ == "__main__":
    main()
