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
        "teacher export emits labels from a local teacher stream.\n"
        "the bridge path stays generic while the attack policy stays local.\n"
    )
    report = model.report(corpus)
    print(f"tokens: {report.tokens}")
    print(f"steps: {report.steps}")
    print(f"teacher bits/byte: {report.teacher_bits_per_byte:.4f}")
    print(f"student bits/byte: {report.student_bits_per_byte:.4f}")
    print(f"attack bits/byte: {report.attack_bits_per_byte:.4f}")
    print(f"mean entropy: {report.mean_entropy:.4f}")
    print(f"mean agreement mass: {report.mean_agreement_mass:.4f}")
    print(f"label flip rate: {report.label_flip_rate:.4f}")
    print(f"attack mutation rate: {report.attack_mutation_rate:.4f}")
    print(f"clean deterministic fraction: {report.clean_deterministic_fraction:.4f}")
    print(f"attacked deterministic fraction: {report.attacked_deterministic_fraction:.4f}")


if __name__ == "__main__":
    main()
