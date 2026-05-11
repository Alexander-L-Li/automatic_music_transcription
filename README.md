# Seeded Diffusion Refinement for Image-to-Image Piano Transcription

This repo is a runnable scaffold for a small-budget test of whether a frozen
MIDI-only diffusion prior can help piano audio-to-MIDI transcription on a tiny
MAESTRO v3.0.0 subset.

The experiment is intentionally staged:

1. Data sanity: fetch metadata/MIDI, choose fixed held-out pieces, render piano rolls.
2. Audio alignment: byte-range fetch only selected WAV files, cache log-mel+CQT features, plot audio against MIDI.
3. One-clip overfit: prove the supervised path can memorize one 4-second target.
4. Tiny baseline: train a small direct spectrogram-to-piano-roll CNN.
5. MIDI diffusion prior: train a small MIDI-only denoising model and inspect unconditional samples.
6. Frozen-prior audio branch: freeze the prior and train the audio branch.

No result should be treated as real until it is regenerated from the scripts and
recorded in `runs/results.tsv`.

## Local Setup

Torch is intentionally not pinned in `pyproject.toml`, because Colab and this
machine usually already provide a suitable build. Install the rest:

```bash
uv pip install --system -e .
```

Then run the CPU-safe stages:

```bash
fpamt data-sanity --pieces-per-split 2
fpamt audio-alignment --max-pieces 1
fpamt overfit-one --steps 250
```

For GPU work, use `notebooks/colab_poc.py` as a copy-pasteable Colab driver.

## Integrity Rules

- Preserve MAESTRO train/validation/test split boundaries by piece.
- Pick demo validation/test clips before inspecting predictions.
- Log every run to `runs/results.tsv`.
- Stop early if data alignment or one-clip overfit fails.
- Compare ControlNet-prior against the local supervised baseline, not against
  published AMT systems unless those are actually run under comparable settings.

* This README was written with the help of Claude Code. *
