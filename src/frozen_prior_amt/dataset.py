from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import AudioConfig, Paths
from .features import load_features
from .maestro import first_note_window_start, load_roll


@dataclass
class CachedPiece:
    piece_id: str
    split: str
    x: np.ndarray | None
    active: np.ndarray
    onset: np.ndarray

    @property
    def frames(self) -> int:
        if self.x is None:
            return self.active.shape[-1]
        return min(self.x.shape[-1], self.active.shape[-1])


def load_cached_pieces(paths: Paths, rows: pd.DataFrame, include_audio: bool) -> list[CachedPiece]:
    pieces: list[CachedPiece] = []
    for _, row in rows.iterrows():
        roll = load_roll(paths, row.piece_id)
        feats = load_features(paths, row.piece_id) if include_audio else None
        pieces.append(
            CachedPiece(
                piece_id=row.piece_id,
                split=row.split,
                x=feats["x"] if feats is not None else None,
                active=roll["active"].astype(np.float32),
                onset=roll["onset"].astype(np.float32),
            )
        )
    return pieces


class RandomWindowDataset:
    def __init__(
        self,
        pieces: list[CachedPiece],
        cfg: AudioConfig,
        num_samples: int,
        include_audio: bool,
        require_notes: bool = True,
        seed: int = 0,
    ) -> None:
        self.pieces = pieces
        self.cfg = cfg
        self.num_samples = num_samples
        self.include_audio = include_audio
        self.require_notes = require_notes
        self.rng = np.random.default_rng(seed)
        self.window_frames = cfg.frames_per_window
        self._gpu_cache: dict[str, torch.Tensor | bool | int] | None = None
        self._note_candidates: list[np.ndarray] = []
        self._max_starts: list[int] = []
        for piece in self.pieces:
            max_start = max(0, piece.frames - self.window_frames)
            self._max_starts.append(max_start)
            active_curve = piece.active[:, : piece.frames].sum(axis=0)
            candidates = np.where(active_curve > 0)[0]
            self._note_candidates.append(candidates[candidates <= max_start].astype(np.int64, copy=False))

    def __len__(self) -> int:
        return self.num_samples

    def _choose_start(self, piece_index: int) -> int:
        max_start = self._max_starts[piece_index]
        if not self.require_notes:
            return int(self.rng.integers(0, max_start + 1)) if max_start else 0
        candidates = self._note_candidates[piece_index]
        if len(candidates) == 0:
            return int(self.rng.integers(0, max_start + 1)) if max_start else 0
        center = int(self.rng.choice(candidates))
        jitter = int(self.rng.integers(-self.window_frames // 3, self.window_frames // 6 + 1))
        return int(np.clip(center + jitter, 0, max_start))

    def __getitem__(self, _: int) -> dict[str, Any]:
        piece_index = int(self.rng.integers(0, len(self.pieces)))
        piece = self.pieces[piece_index]
        start = self._choose_start(piece_index)
        end = start + self.window_frames
        y = piece.active[:, start:end]
        item: dict[str, Any] = {
            "y": y.astype(np.float32),
            "piece_id": piece.piece_id,
            "start_frame": start,
        }
        if self.include_audio:
            assert piece.x is not None
            item["x"] = piece.x[:, :, start:end].astype(np.float32)
        return item

    def enable_gpu_cache(self, device: torch.device, max_gb: float = 24.0) -> bool:
        if device.type != "cuda":
            return False
        if not self.pieces:
            return False
        frames = [piece.frames for piece in self.pieces]
        max_frames = max(frames)
        max_candidates = max((len(c) for c in self._note_candidates), default=0)
        float_bytes = 4
        int_bytes = 8
        estimated = len(self.pieces) * 88 * max_frames * float_bytes
        if self.include_audio:
            first_x = next(piece.x for piece in self.pieces if piece.x is not None)
            estimated += len(self.pieces) * first_x.shape[0] * first_x.shape[1] * max_frames * float_bytes
        if max_candidates > 0:
            estimated += len(self.pieces) * max_candidates * int_bytes
        if estimated / 1024**3 > max_gb:
            print(
                f"GPU cache skipped: estimated {estimated / 1024**3:.1f} GB exceeds "
                f"--gpu-cache-gb={max_gb:.1f}"
            )
            return False

        y = torch.zeros((len(self.pieces), 88, max_frames), dtype=torch.float32, device=device)
        x = None
        if self.include_audio:
            first_x = next(piece.x for piece in self.pieces if piece.x is not None)
            x = torch.zeros(
                (len(self.pieces), first_x.shape[0], first_x.shape[1], max_frames),
                dtype=torch.float32,
                device=device,
            )
        for i, piece in enumerate(self.pieces):
            frames_i = piece.frames
            y[i, :, :frames_i] = torch.as_tensor(piece.active[:, :frames_i], dtype=torch.float32, device=device)
            if x is not None:
                assert piece.x is not None
                x[i, :, :, :frames_i] = torch.as_tensor(piece.x[:, :, :frames_i], dtype=torch.float32, device=device)

        max_starts = torch.as_tensor(self._max_starts, dtype=torch.long, device=device)
        if max_candidates > 0:
            note_candidates = torch.zeros((len(self.pieces), max_candidates), dtype=torch.long, device=device)
            note_lengths = torch.zeros((len(self.pieces),), dtype=torch.long, device=device)
            for i, candidates in enumerate(self._note_candidates):
                if len(candidates) == 0:
                    continue
                note_lengths[i] = len(candidates)
                note_candidates[i, : len(candidates)] = torch.as_tensor(candidates, dtype=torch.long, device=device)
        else:
            note_candidates = torch.zeros((len(self.pieces), 1), dtype=torch.long, device=device)
            note_lengths = torch.zeros((len(self.pieces),), dtype=torch.long, device=device)

        self._gpu_cache = {
            "y": y,
            "x": x if x is not None else torch.empty(0, device=device),
            "has_x": x is not None,
            "max_starts": max_starts,
            "note_candidates": note_candidates,
            "note_lengths": note_lengths,
            "require_notes": self.require_notes,
            "max_frames": max_frames,
        }
        print(
            f"GPU cache enabled: {len(self.pieces)} pieces, max_frames={max_frames}, "
            f"estimated={estimated / 1024**3:.2f} GB"
        )
        return True

    def _sample_batch_gpu(self, batch_size: int) -> dict[str, Any]:
        assert self._gpu_cache is not None
        y_bank = self._gpu_cache["y"]
        assert isinstance(y_bank, torch.Tensor)
        device = y_bank.device
        pieces = y_bank.shape[0]
        piece_indices = torch.randint(0, pieces, (batch_size,), device=device)
        max_starts = self._gpu_cache["max_starts"]
        assert isinstance(max_starts, torch.Tensor)
        selected_max = max_starts[piece_indices]
        random_starts = (torch.rand(batch_size, device=device) * (selected_max + 1).float()).long()

        starts = random_starts
        note_lengths = self._gpu_cache["note_lengths"]
        note_candidates = self._gpu_cache["note_candidates"]
        if bool(self._gpu_cache["require_notes"]) and isinstance(note_lengths, torch.Tensor) and isinstance(note_candidates, torch.Tensor):
            selected_lengths = note_lengths[piece_indices]
            has_notes = selected_lengths > 0
            safe_lengths = selected_lengths.clamp_min(1)
            offsets = (torch.rand(batch_size, device=device) * safe_lengths.float()).long()
            centers = note_candidates[piece_indices, offsets]
            jitter = torch.randint(
                -self.window_frames // 3,
                self.window_frames // 6 + 1,
                (batch_size,),
                device=device,
            )
            note_starts = (centers + jitter).clamp_min(0)
            note_starts = torch.minimum(note_starts, selected_max)
            starts = torch.where(has_notes, note_starts, random_starts)

        time = starts[:, None] + torch.arange(self.window_frames, device=device)[None, :]
        keys = torch.arange(y_bank.shape[1], device=device)
        y = y_bank[piece_indices[:, None, None], keys[None, :, None], time[:, None, :]]
        batch: dict[str, Any] = {"y": y, "piece_index": piece_indices, "start_frame": starts}

        if bool(self._gpu_cache["has_x"]):
            x_bank = self._gpu_cache["x"]
            assert isinstance(x_bank, torch.Tensor)
            channels = torch.arange(x_bank.shape[1], device=device)
            bins = torch.arange(x_bank.shape[2], device=device)
            x = x_bank[
                piece_indices[:, None, None, None],
                channels[None, :, None, None],
                bins[None, None, :, None],
                time[:, None, None, :],
            ]
            batch["x"] = x.contiguous(memory_format=torch.channels_last)
        return batch

    def sample_batch(self, batch_size: int, pin_memory: bool = False) -> dict[str, Any]:
        if self._gpu_cache is not None:
            return self._sample_batch_gpu(batch_size)
        piece_indices = self.rng.integers(0, len(self.pieces), size=batch_size)
        y = np.empty((batch_size, 88, self.window_frames), dtype=np.float32)
        x = None
        if self.include_audio:
            first_x = next(piece.x for piece in self.pieces if piece.x is not None)
            x = np.empty((batch_size, first_x.shape[0], first_x.shape[1], self.window_frames), dtype=np.float32)
        piece_ids: list[str] = []
        starts = np.empty(batch_size, dtype=np.int64)

        for i, piece_index in enumerate(piece_indices):
            piece = self.pieces[int(piece_index)]
            max_start = self._max_starts[int(piece_index)]
            if self.require_notes and len(self._note_candidates[int(piece_index)]) > 0:
                center = int(self.rng.choice(self._note_candidates[int(piece_index)]))
                jitter = int(self.rng.integers(-self.window_frames // 3, self.window_frames // 6 + 1))
                start = int(np.clip(center + jitter, 0, max_start))
            else:
                start = int(self.rng.integers(0, max_start + 1)) if max_start else 0
            end = start + self.window_frames
            y[i] = piece.active[:, start:end]
            if x is not None:
                assert piece.x is not None
                x[i] = piece.x[:, :, start:end]
            piece_ids.append(piece.piece_id)
            starts[i] = start

        batch: dict[str, Any] = {
            "y": torch.from_numpy(y),
            "piece_id": piece_ids,
            "start_frame": starts,
        }
        if x is not None:
            batch["x"] = torch.from_numpy(x)
        if pin_memory:
            batch["y"] = batch["y"].pin_memory()
            if x is not None:
                batch["x"] = batch["x"].pin_memory()
        return batch


class OneWindowDataset:
    def __init__(
        self,
        piece: CachedPiece,
        cfg: AudioConfig,
        include_audio: bool,
        start_frame: int | None = None,
        repeats: int = 512,
    ) -> None:
        self.piece = piece
        self.cfg = cfg
        self.include_audio = include_audio
        self.repeats = repeats
        self.window_frames = cfg.frames_per_window
        if start_frame is None:
            start_frame = first_note_window_start(piece.active, cfg.frame_hz, self.window_frames)
        self.start_frame = int(start_frame)

    def __len__(self) -> int:
        return self.repeats

    def __getitem__(self, _: int) -> dict[str, Any]:
        start = self.start_frame
        end = start + self.window_frames
        item: dict[str, Any] = {
            "y": self.piece.active[:, start:end].astype(np.float32),
            "piece_id": self.piece.piece_id,
            "start_frame": start,
        }
        if self.include_audio:
            assert self.piece.x is not None
            item["x"] = self.piece.x[:, :, start:end].astype(np.float32)
        return item


class FixedWindowDataset:
    def __init__(
        self,
        pieces: list[CachedPiece],
        cfg: AudioConfig,
        include_audio: bool,
        windows_per_piece: int = 3,
    ) -> None:
        self.items: list[tuple[CachedPiece, int]] = []
        self.cfg = cfg
        self.include_audio = include_audio
        w = cfg.frames_per_window
        for piece in pieces:
            first = first_note_window_start(piece.active, cfg.frame_hz, w)
            max_start = max(0, piece.frames - w)
            starts = [first]
            if windows_per_piece > 1 and max_start > 0:
                grid = np.linspace(0, max_start, windows_per_piece, dtype=int).tolist()
                starts.extend(grid)
            for start in sorted(set(int(np.clip(s, 0, max_start)) for s in starts))[:windows_per_piece]:
                self.items.append((piece, start))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        piece, start = self.items[index]
        end = start + self.cfg.frames_per_window
        item: dict[str, Any] = {
            "y": piece.active[:, start:end].astype(np.float32),
            "piece_id": piece.piece_id,
            "start_frame": start,
        }
        if self.include_audio:
            assert piece.x is not None
            item["x"] = piece.x[:, :, start:end].astype(np.float32)
        return item
