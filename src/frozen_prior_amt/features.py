from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import AudioConfig, Paths
from .maestro import local_maestro_path


def _pad_or_trim_time(x: np.ndarray, frames: int) -> np.ndarray:
    if x.shape[-1] == frames:
        return x.astype(np.float32, copy=False)
    if x.shape[-1] > frames:
        return x[..., :frames].astype(np.float32, copy=False)
    pad = [(0, 0)] * x.ndim
    pad[-1] = (0, frames - x.shape[-1])
    return np.pad(x, pad, mode="constant").astype(np.float32, copy=False)


def _resize_freq(x: np.ndarray, target_bins: int) -> np.ndarray:
    if x.shape[0] == target_bins:
        return x
    src = np.linspace(0.0, 1.0, x.shape[0])
    dst = np.linspace(0.0, 1.0, target_bins)
    out = np.empty((target_bins, x.shape[1]), dtype=np.float32)
    for t in range(x.shape[1]):
        out[:, t] = np.interp(dst, src, x[:, t])
    return out


def _standardize(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-6:
        std = 1.0
    return ((x - mean) / std).astype(np.float32)


def audio_features_from_wav(
    wav_path: Path,
    cfg: AudioConfig,
    expected_frames: int | None = None,
) -> dict[str, np.ndarray]:
    y, sr = librosa.load(wav_path, sr=cfg.sample_rate, mono=True)
    if expected_frames is None:
        expected_frames = int(np.ceil(len(y) / cfg.hop_length))

    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel_power, ref=np.max)
    logmel = _pad_or_trim_time(logmel, expected_frames)

    cqt = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=cfg.hop_length,
            fmin=librosa.midi_to_hz(cfg.fmin_midi),
            n_bins=cfg.n_cqt_bins,
            bins_per_octave=cfg.cqt_bins_per_octave,
        )
    )
    logcqt = librosa.amplitude_to_db(cqt, ref=np.max)
    logcqt = _pad_or_trim_time(logcqt, expected_frames)
    logcqt_resized = _resize_freq(logcqt, cfg.n_mels)

    stacked = np.stack([_standardize(logmel), _standardize(logcqt_resized)], axis=0)
    return {
        "x": stacked.astype(np.float32),
        "logmel": logmel.astype(np.float32),
        "logcqt": logcqt.astype(np.float32),
        "sample_rate": np.asarray(sr),
        "hop_length": np.asarray(cfg.hop_length),
        "frame_hz": np.asarray(cfg.frame_hz),
    }


def cache_features(paths: Paths, row: pd.Series, cfg: AudioConfig, force: bool = False) -> Path:
    out = paths.cache / "features" / f"{row.piece_id}.npz"
    if out.exists() and not force:
        return out
    expected_frames = int(np.ceil(float(row.duration) * cfg.frame_hz))
    wav_path = local_maestro_path(paths, row.audio_filename)
    feats = audio_features_from_wav(wav_path, cfg, expected_frames=expected_frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **feats,
        duration=float(row.duration),
        piece_id=row.piece_id,
        audio_filename=row.audio_filename,
    )
    return out


def cache_feature_set(paths: Paths, rows: pd.DataFrame, cfg: AudioConfig, force: bool = False) -> list[Path]:
    out = []
    for _, row in tqdm(list(rows.iterrows()), desc="cache audio features", mininterval=10.0, maxinterval=30.0):
        out.append(cache_features(paths, row, cfg, force=force))
    return out


def load_features(paths: Paths, piece_id: str) -> dict[str, np.ndarray]:
    path = paths.cache / "features" / f"{piece_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
