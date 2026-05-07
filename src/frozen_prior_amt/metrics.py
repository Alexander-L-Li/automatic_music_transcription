from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MIDI_MIN


@dataclass(frozen=True)
class Note:
    pitch: int
    onset: float
    offset: float

    @property
    def duration(self) -> float:
        return max(0.0, self.offset - self.onset)


def binarize(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (x >= threshold).astype(np.float32)


def extract_notes(active: np.ndarray, frame_hz: int, threshold: float = 0.5) -> list[Note]:
    roll = binarize(active, threshold=threshold)
    notes: list[Note] = []
    for p in range(roll.shape[0]):
        values = roll[p]
        padded = np.pad(values, (1, 1), mode="constant")
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            if e <= s:
                continue
            notes.append(Note(MIDI_MIN + p, s / frame_hz, e / frame_hz))
    return notes


def postprocess_roll(
    active: np.ndarray,
    frame_hz: int,
    threshold: float,
    min_note_seconds: float = 0.08,
    merge_gap_seconds: float = 0.04,
) -> np.ndarray:
    """Threshold, merge tiny gaps, and remove tiny fragments per pitch."""

    roll = binarize(active, threshold=threshold)
    min_frames = max(1, int(round(min_note_seconds * frame_hz)))
    gap_frames = max(0, int(round(merge_gap_seconds * frame_hz)))
    out = np.zeros_like(roll, dtype=np.float32)
    for p in range(roll.shape[0]):
        values = roll[p].copy()
        if gap_frames > 0:
            padded = np.pad(values, (1, 1), mode="constant")
            changes = np.diff(padded)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            for prev_end, next_start in zip(ends[:-1], starts[1:]):
                if 0 < next_start - prev_end <= gap_frames:
                    values[prev_end:next_start] = 1.0
        padded = np.pad(values, (1, 1), mode="constant")
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= min_frames:
                out[p, s:e] = 1.0
    return out


def _match_notes(
    pred: list[Note],
    target: list[Note],
    onset_tolerance: float,
    require_offset: bool = False,
    offset_tolerance: float = 0.05,
) -> tuple[int, int, int]:
    matched_target: set[int] = set()
    true_pos = 0
    pred_sorted = sorted(enumerate(pred), key=lambda item: item[1].onset)
    for _, pn in pred_sorted:
        candidates = []
        for j, tn in enumerate(target):
            if j in matched_target or pn.pitch != tn.pitch:
                continue
            onset_err = abs(pn.onset - tn.onset)
            if onset_err > onset_tolerance:
                continue
            if require_offset:
                tol = max(offset_tolerance, 0.2 * tn.duration)
                if abs(pn.offset - tn.offset) > tol:
                    continue
            candidates.append((onset_err, j))
        if candidates:
            _, best = min(candidates, key=lambda x: x[0])
            matched_target.add(best)
            true_pos += 1
    fp = len(pred) - true_pos
    fn = len(target) - true_pos
    return true_pos, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def note_metrics(
    pred_active: np.ndarray,
    target_active: np.ndarray,
    frame_hz: int,
    threshold: float = 0.5,
    onset_tolerance: float = 0.05,
    short_fragment_seconds: float = 0.08,
) -> dict[str, float]:
    pred = extract_notes(pred_active, frame_hz=frame_hz, threshold=threshold)
    target = extract_notes(target_active, frame_hz=frame_hz, threshold=0.5)
    onset_tp, onset_fp, onset_fn = _match_notes(pred, target, onset_tolerance)
    onset = prf(onset_tp, onset_fp, onset_fn)
    off_tp, off_fp, off_fn = _match_notes(pred, target, onset_tolerance, require_offset=True)
    onset_offset = prf(off_tp, off_fp, off_fn)
    minutes = max(1e-9, target_active.shape[1] / frame_hz / 60.0)
    short = sum(1 for n in pred if n.duration < short_fragment_seconds)
    out = {
        "notes_pred": float(len(pred)),
        "notes_target": float(len(target)),
        "onset_tp": float(onset_tp),
        "onset_fp": float(onset_fp),
        "onset_fn": float(onset_fn),
        "onset_precision": onset["precision"],
        "onset_recall": onset["recall"],
        "onset_f1": onset["f1"],
        "onset_offset_f1": onset_offset["f1"],
        "false_positives_per_min": onset_fp / minutes,
        "short_fragments_per_min": short / minutes,
    }
    return out


def frame_metrics(pred_active: np.ndarray, target_active: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = binarize(pred_active, threshold=threshold).astype(bool)
    target = binarize(target_active, threshold=0.5).astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    out = prf(tp, fp, fn)
    out.update({"frame_tp": float(tp), "frame_fp": float(fp), "frame_fn": float(fn)})
    return {f"frame_{k}" if k in {"precision", "recall", "f1"} else k: v for k, v in out.items()}
