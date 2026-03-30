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
        "payload choice keeps dictionary layout policy local.\n"
        "payload choice keeps dictionary layout policy local.\n"
        "dense and sparse payloads should both remain visible.\n"
    ) * 4
    trace = model.trace(corpus)
    report = model.report(corpus)

    print("project:", "payload_choice")
    print("selected layout:", report.selected_layout)
    print("dense payload bytes:", report.dense_payload_bytes)
    print("sparse payload bytes:", report.sparse_payload_bytes)
    print("selection margin:", round(report.selection_margin, 4))
    print("deterministic fraction:", round(report.deterministic_fraction, 4))
    print("mean candidate count:", round(report.mean_candidate_count, 4))
    print("dense positions:", trace.dense_layout.positions.size)
    print("sparse positions:", trace.sparse_layout.positions.size)


if __name__ == "__main__":
    main()
