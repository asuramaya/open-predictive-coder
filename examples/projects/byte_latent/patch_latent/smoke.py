from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_predictive_coder import ByteCodec
from model import PatchLatentByteModel, PatchLatentConfig


def main() -> None:
    model = PatchLatentByteModel(PatchLatentConfig())
    corpus = (
        "patch-latent coding compresses bytes into patch latents, then mixes them globally.\n"
        "short internal streams should carry the main burden.\n"
    ) * 24

    fit_report = model.fit(corpus)
    score_report = model.score(corpus)
    trace = model.trace(corpus[:160])

    print("train bits/byte:", round(fit_report.train_bits_per_byte, 4))
    print("score bits/byte:", round(score_report.bits_per_byte, 4))
    print("mean patch size:", round(float(trace.mean_patch_size), 4))
    print("compression ratio:", round(float(trace.compression_ratio), 4))
    print("mean surprise:", round(float(trace.mean_surprise), 4))
    print("mean boundary probability:", round(float(trace.mean_boundary_probability), 4))
    print("patches:", trace.patches)
    print("tokens:", trace.tokens)

    prompt = ByteCodec.encode_text("patch ")
    sample = model.generate(prompt, steps=48, greedy=True)
    print(ByteCodec.decode_text(sample))


if __name__ == "__main__":
    main()
