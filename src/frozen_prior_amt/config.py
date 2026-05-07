from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MAESTRO_VERSION = "v3.0.0"
MAESTRO_ROOT_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0"
MAESTRO_CSV_URL = f"{MAESTRO_ROOT_URL}/maestro-v3.0.0.csv"
MAESTRO_MIDI_ZIP_URL = f"{MAESTRO_ROOT_URL}/maestro-v3.0.0-midi.zip"
MAESTRO_FULL_ZIP_URL = f"{MAESTRO_ROOT_URL}/maestro-v3.0.0.zip"
MAESTRO_ZIP_PREFIX = "maestro-v3.0.0"

MIDI_MIN = 21
MIDI_MAX = 108
N_KEYS = MIDI_MAX - MIDI_MIN + 1


@dataclass(frozen=True)
class Paths:
    root: Path = Path(".")
    raw: Path = Path("data/raw/maestro-v3.0.0")
    cache: Path = Path("data/cache")
    runs: Path = Path("runs")
    plots: Path = Path("plots")
    artifacts: Path = Path("artifacts")

    @classmethod
    def from_root(cls, root: str | Path = ".") -> "Paths":
        root_path = Path(root)
        return cls(
            root=root_path,
            raw=root_path / "data/raw/maestro-v3.0.0",
            cache=root_path / "data/cache",
            runs=root_path / "runs",
            plots=root_path / "plots",
            artifacts=root_path / "artifacts",
        )

    def ensure(self) -> None:
        for path in [self.raw, self.cache, self.runs, self.plots, self.artifacts]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    frame_hz: int = 50
    window_seconds: float = 4.0
    n_mels: int = 128
    n_cqt_bins: int = 88
    cqt_bins_per_octave: int = 12
    fmin_midi: int = MIDI_MIN
    n_fft: int = 1024

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate / self.frame_hz))

    @property
    def frames_per_window(self) -> int:
        return int(round(self.window_seconds * self.frame_hz))


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    steps: int = 500
    lr: float = 2e-3
    weight_decay: float = 0.0
    seed: int = 7
    hidden: int = 48
    num_workers: int = 0
    positive_weight: float = 8.0
    gpu_cache: bool = False
    gpu_cache_gb: float = 24.0
    compile_model: bool = False


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 100
    sample_steps: int = 24
    prediction_type: str = "x0"
    lr: float = 2e-4
    batch_size: int = 16
    steps: int = 1000
    hidden: int = 48
    seed: int = 11
    num_workers: int = 0
    gpu_cache: bool = False
    gpu_cache_gb: float = 24.0
    compile_model: bool = False
