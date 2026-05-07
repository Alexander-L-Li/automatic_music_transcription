from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import MIDI_MIN


def plot_piano_roll(active: np.ndarray, frame_hz: int, out: Path, title: str = "Piano roll") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    seconds = active.shape[1] / frame_hz
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(
        active,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, seconds, MIDI_MIN, MIDI_MIN + active.shape[0] - 1],
        cmap="Greys",
    )
    ax.set_title(title)
    ax.set_xlabel("seconds")
    ax.set_ylabel("MIDI pitch")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_alignment(
    logmel: np.ndarray,
    onset: np.ndarray,
    frame_hz: int,
    out: Path,
    title: str = "Audio/MIDI alignment",
    max_seconds: float = 12.0,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    max_frames = min(logmel.shape[-1], onset.shape[-1], int(max_seconds * frame_hz))
    seconds = max_frames / frame_hz
    onset_curve = onset[:, :max_frames].sum(axis=0)
    onset_curve = onset_curve / max(1.0, onset_curve.max(initial=1.0))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[3, 1])
    axes[0].imshow(
        logmel[:, :max_frames],
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, seconds, 0, logmel.shape[0]],
        cmap="magma",
    )
    axes[0].set_title(title)
    axes[0].set_ylabel("mel bin")
    axes[1].plot(np.arange(max_frames) / frame_hz, onset_curve, color="black", lw=1.0)
    axes[1].fill_between(np.arange(max_frames) / frame_hz, onset_curve, alpha=0.25, color="black")
    axes[1].set_ylabel("MIDI onsets")
    axes[1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_prediction_grid(
    panels: dict[str, np.ndarray],
    frame_hz: int,
    out: Path,
    title: str = "Prediction comparison",
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    names = list(panels.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    seconds = next(iter(panels.values())).shape[1] / frame_hz
    for ax, name in zip(axes, names):
        arr = panels[name]
        ax.imshow(
            arr,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[0, seconds, MIDI_MIN, MIDI_MIN + arr.shape[0] - 1],
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_ylabel(name)
    axes[0].set_title(title)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

