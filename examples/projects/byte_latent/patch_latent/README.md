# patch_latent

This folder is a runnable project-layer example for a learned patch-latent byte path.

It is a reconstruction inside this repo, and its public lineage notes live in [`docs/lineage.md`](../../../docs/lineage.md).

The point is to mirror the real shape, without turning the example into a large training stack:

- bytes in
- learned boundary scoring and patch segmentation
- learned local byte encoding
- patch commit into a shorter latent stream
- recurrent global mixing
- learned bridge back to local byte prediction

The example uses current kernel primitives where they fit:

- `LearnedSegmenterConfig` and `LearnedSegmenter`
- `LocalByteEncoder`
- `PatchPooler`
- `GlobalLocalBridge`
- `LatentConfig`
- `ByteLatentFeatureView`
- `RidgeReadout`
- `ByteCodec`

## What Is Faithful

- the causal byte interface
- the patch/commit boundary with learned boundary probabilities
- the shorter latent stream and recurrent global state
- the bridge from latent state back to byte prediction features
- bits-per-byte scoring

## What Is Simplified

- no large transformer stack
- no entropy transformer or rate-distortion controller
- no quantized export pipeline
- no distributed training or benchmark harness
- no second compression stage or QAT path

The example is intended as a structural/dev target: enough to test the architecture from scratch, but small enough to run quickly.

## Entry Points

- `probe.py`: prints the model, encoding, and patching dimensions
- `smoke.py`: fits on a small corpus, scores it, prints patch metrics, and samples from a prompt

## Run

From the repository root:

```bash
PYTHONPATH=src python3 examples/projects/byte_latent/patch_latent/probe.py
```

```bash
PYTHONPATH=src python3 examples/projects/byte_latent/patch_latent/smoke.py
```
