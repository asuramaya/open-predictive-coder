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

from model import TeacherExportConfig, TeacherExportModel


def main() -> None:
    model = TeacherExportModel(TeacherExportConfig(hidden_dim=16))
    corpus = (
        "attack-aware bridge export compares teacher labels to attacked labels.\n"
        "the local bridge keeps policy contained to the example.\n"
    ) * 2
    report = model.report(corpus)
    print(f"tokens: {report.tokens}")
    print(f"steps: {report.steps}")
    print(f"teacher bits/byte: {report.teacher_bits_per_byte:.4f}")
    print(f"student bits/byte: {report.student_bits_per_byte:.4f}")
    print(f"attack bits/byte: {report.attack_bits_per_byte:.4f}")
    print(f"mean peak: {report.mean_peak:.4f}")
    print(f"mean candidate4: {report.mean_candidate4:.4f}")
    print(f"label flip rate: {report.label_flip_rate:.4f}")
    print(f"clean deterministic fraction: {report.clean_deterministic_fraction:.4f}")
    print(f"attacked deterministic fraction: {report.attacked_deterministic_fraction:.4f}")


if __name__ == "__main__":
    main()
