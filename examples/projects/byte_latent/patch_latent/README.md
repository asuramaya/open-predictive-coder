# patch_latent

This folder is a runnable project-layer example for a patch-latent byte path.

It is a reconstruction inside this repo, and its public lineage notes live in [`docs/lineage.md`](../../../docs/lineage.md).

The point is to mirror the real shape, not to reimplement the full training system:

- bytes in
- patch segmentation
- patch commit into a shorter latent stream
- recurrent global mixing
- bridge back to local byte prediction

The example uses current kernel primitives where they fit:

- `SegmenterConfig` and `AdaptiveSegmenter`
- `LatentConfig` and `LatentCommitter`
- `ByteLatentFeatureView`
- `RidgeReadout`
- `ByteCodec`

## What Is Faithful

- the causal byte interface
- the patch/commit boundary
- the shorter latent stream and recurrent global state
- the bridge from latent state back to byte prediction features
- bits-per-byte scoring

## What Is Simplified

- no large transformer stack
- no learned segmentation head
- no quantized export pipeline
- no distributed training or benchmark harness
- no full rate-distortion controller

The example is intended as a structural/dev target: enough to test the architecture from scratch, but small enough to run quickly.

## Entry Points

- `probe.py`: prints the model and patching dimensions
- `smoke.py`: fits on a small corpus, scores it, and samples from a prompt

## Run

From the repository root:

```bash
PYTHONPATH=src python3 examples/projects/byte_latent/patch_latent/probe.py
```

```bash
PYTHONPATH=src python3 examples/projects/byte_latent/patch_latent/smoke.py
```
