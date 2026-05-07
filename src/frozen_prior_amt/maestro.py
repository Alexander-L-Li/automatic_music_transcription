from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pretty_midi
import requests
from remotezip import RemoteZip
from tqdm import tqdm

from .config import (
    MAESTRO_CSV_URL,
    MAESTRO_FULL_ZIP_URL,
    MAESTRO_MIDI_ZIP_URL,
    MAESTRO_ZIP_PREFIX,
    MIDI_MAX,
    MIDI_MIN,
    N_KEYS,
    AudioConfig,
    Paths,
)
from .utils import atomic_json, slugify


def fetch_metadata(paths: Paths, refresh: bool = False) -> pd.DataFrame:
    paths.ensure()
    csv_path = paths.raw / "maestro-v3.0.0.csv"
    if refresh or not csv_path.exists():
        response = requests.get(MAESTRO_CSV_URL, timeout=60)
        response.raise_for_status()
        csv_path.write_bytes(response.content)
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)
    df["piece_id"] = df.apply(
        lambda r: f"{r.split}_{int(r.row_id):04d}_{slugify(r.canonical_composer)}_{slugify(r.canonical_title, 48)}",
        axis=1,
    )
    return df


def select_subset(
    metadata: pd.DataFrame,
    pieces_per_split: int = 2,
    min_duration: float = 20.0,
    max_duration: float | None = None,
) -> pd.DataFrame:
    rows = []
    for split in ["train", "validation", "test"]:
        part = metadata[(metadata["split"] == split) & (metadata["duration"] >= min_duration)]
        if max_duration is not None:
            part = part[part["duration"] <= max_duration]
        part = part.sort_values(["duration", "canonical_composer", "canonical_title", "row_id"])
        rows.append(part.head(pieces_per_split))
    subset = pd.concat(rows, ignore_index=True)
    return subset


def write_subset(paths: Paths, subset: pd.DataFrame, name: str = "tiny_subset") -> Path:
    out = paths.cache / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out, index=False)
    manifest = {
        "name": name,
        "rows": len(subset),
        "split_counts": subset["split"].value_counts().to_dict(),
        "piece_ids": subset["piece_id"].tolist(),
    }
    atomic_json(paths.cache / f"{name}.manifest.json", manifest)
    return out


def local_maestro_path(paths: Paths, filename: str) -> Path:
    return paths.raw / filename


def _copy_member(zf: RemoteZip, member: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with zf.open(member) as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    tmp.replace(out_path)


def download_maestro_files(
    paths: Paths,
    rows: pd.DataFrame,
    include_audio: bool,
    force: bool = False,
) -> list[Path]:
    """Fetch selected MIDI and optionally WAV files from MAESTRO ZIP archives.

    The full archive is large, but it is a ZIP that supports HTTP range requests.
    `remotezip` reads only the central directory and the selected members.
    """

    paths.ensure()
    needed_midi = sorted(set(rows["midi_filename"].tolist()))
    needed_audio = sorted(set(rows["audio_filename"].tolist())) if include_audio else []
    written: list[Path] = []

    midi_missing = [name for name in needed_midi if force or not local_maestro_path(paths, name).exists()]
    if midi_missing:
        with RemoteZip(MAESTRO_MIDI_ZIP_URL) as zf:
            for rel in tqdm(midi_missing, desc="fetch MIDI", mininterval=10.0, maxinterval=30.0):
                member = f"{MAESTRO_ZIP_PREFIX}/{rel}"
                out_path = local_maestro_path(paths, rel)
                _copy_member(zf, member, out_path)
                written.append(out_path)

    audio_missing = [name for name in needed_audio if force or not local_maestro_path(paths, name).exists()]
    if audio_missing:
        with RemoteZip(MAESTRO_FULL_ZIP_URL) as zf:
            for rel in tqdm(audio_missing, desc="fetch audio", mininterval=10.0, maxinterval=30.0):
                member = f"{MAESTRO_ZIP_PREFIX}/{rel}"
                out_path = local_maestro_path(paths, rel)
                _copy_member(zf, member, out_path)
                written.append(out_path)

    return written


def piano_roll_from_midi(
    midi_path: Path,
    frame_hz: int,
    duration: float | None = None,
    start: float = 0.0,
    seconds: float | None = None,
) -> dict[str, np.ndarray]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    end_time = duration if duration is not None else midi.get_end_time()
    if seconds is not None:
        end_time = min(end_time, start + seconds)
    n_frames = max(1, int(np.ceil((end_time - start) * frame_hz)))
    active = np.zeros((N_KEYS, n_frames), dtype=np.float32)
    onset = np.zeros((N_KEYS, n_frames), dtype=np.float32)
    velocity = np.zeros((N_KEYS, n_frames), dtype=np.float32)

    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if note.pitch < MIDI_MIN or note.pitch > MIDI_MAX:
                continue
            note_start = note.start - start
            note_end = note.end - start
            if note_end <= 0 or note_start >= (end_time - start):
                continue
            p = note.pitch - MIDI_MIN
            s = int(np.floor(max(0.0, note_start) * frame_hz))
            e = int(np.ceil(min(end_time - start, note_end) * frame_hz))
            if s >= n_frames:
                continue
            e = max(s + 1, min(e, n_frames))
            active[p, s:e] = 1.0
            onset[p, s] = 1.0
            velocity[p, s:e] = max(velocity[p, s:e].max(initial=0.0), note.velocity / 127.0)

    return {"active": active, "onset": onset, "velocity": velocity}


def cache_roll(paths: Paths, row: pd.Series, audio_cfg: AudioConfig, force: bool = False) -> Path:
    out = paths.cache / "rolls" / f"{row.piece_id}.npz"
    if out.exists() and not force:
        return out
    midi_path = local_maestro_path(paths, row.midi_filename)
    roll = piano_roll_from_midi(midi_path, frame_hz=audio_cfg.frame_hz, duration=float(row.duration))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        active=roll["active"],
        onset=roll["onset"],
        velocity=roll["velocity"],
        frame_hz=audio_cfg.frame_hz,
        duration=float(row.duration),
        piece_id=row.piece_id,
        midi_filename=row.midi_filename,
        audio_filename=row.audio_filename,
    )
    return out


def cache_rolls(paths: Paths, rows: pd.DataFrame, audio_cfg: AudioConfig, force: bool = False) -> list[Path]:
    out = []
    for _, row in tqdm(list(rows.iterrows()), desc="cache rolls", mininterval=10.0, maxinterval=30.0):
        out.append(cache_roll(paths, row, audio_cfg, force=force))
    return out


def load_roll(paths: Paths, piece_id: str) -> dict[str, np.ndarray]:
    path = paths.cache / "rolls" / f"{piece_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def first_note_window_start(
    active: np.ndarray,
    frame_hz: int,
    window_frames: int,
    min_start_seconds: float = 1.0,
) -> int:
    note_frames = np.where(active.sum(axis=0) > 0)[0]
    if len(note_frames) == 0:
        return int(min_start_seconds * frame_hz)
    start = int(note_frames[0])
    start = max(start - int(0.25 * frame_hz), int(min_start_seconds * frame_hz))
    return min(start, max(0, active.shape[1] - window_frames))


def manifest_from_rows(rows: pd.DataFrame) -> list[dict[str, object]]:
    keys = [
        "piece_id",
        "split",
        "canonical_composer",
        "canonical_title",
        "duration",
        "midi_filename",
        "audio_filename",
    ]
    return [{k: row[k] for k in keys} for _, row in rows.iterrows()]


def save_run_manifest(path: Path, rows: pd.DataFrame, audio_cfg: AudioConfig, extra: dict[str, object]) -> None:
    payload = {
        "audio_config": asdict(audio_cfg),
        "pieces": manifest_from_rows(rows),
        **extra,
    }
    atomic_json(path, payload)
