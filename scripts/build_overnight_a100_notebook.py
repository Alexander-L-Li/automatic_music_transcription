#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/overnight_a100_resumable.ipynb"
INCLUDE = [
    "pyproject.toml",
    "README.md",
    "requirements-colab.txt",
    "src",
]


def build_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in INCLUDE:
            path = ROOT / rel
            tf.add(path, arcname=f"frozen_prior_amt/{rel}")
    return buf.getvalue()


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def main() -> None:
    blob = base64.b64encode(build_tarball()).decode("ascii")
    cells = [
        markdown_cell(
            """
# Frozen-Prior AMT Overnight A100 Run

Run this in a fresh Colab notebook with an A100 GPU and High-RAM enabled.

The seeded settings are fixed from the previous validation rescue:

- `prior_scale = 0.25`
- `start_frac = 0.75`
- `threshold = 0.4`

The notebook is split so completed stages can be reused. If a later cell fails, rerun the setup/import/helper cells, then continue from the failed stage. Training cells write pointer files under `runs/overnight_seeded_confirmation/`.
            """
        ),
        code_cell(
            f'''
import base64
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

RESET_PROJECT = False  # set True only if you intentionally want to wipe /content/frozen_prior_amt

_FPAMT_BLOB = """{blob}"""
root = Path("/content/frozen_prior_amt")
if RESET_PROJECT and root.exists():
    import shutil
    shutil.rmtree(root)

with tarfile.open(fileobj=io.BytesIO(base64.b64decode(_FPAMT_BLOB)), mode="r:gz") as tf:
    tf.extractall("/content", filter="data")

os.chdir(root)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements-colab.txt")])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(root)])
print("Project ready at", root)
            '''
        ),
        code_cell(
            '''
from pathlib import Path
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

try:
    import pretty_midi
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pretty_midi"])
    import pretty_midi

root = Path("/content/frozen_prior_amt")
os.chdir(root)

for name in list(sys.modules):
    if name == "frozen_prior_amt" or name.startswith("frozen_prior_amt."):
        del sys.modules[name]
sys.path.insert(0, str(root / "src"))

from frozen_prior_amt.config import AudioConfig, DiffusionConfig, MIDI_MIN, Paths
from frozen_prior_amt.dataset import FixedWindowDataset, load_cached_pieces
from frozen_prior_amt.diffusion import DiffusionSchedule, diffusion_to_roll, q_sample, sample_ddim
from frozen_prior_amt.features import cache_feature_set
from frozen_prior_amt.maestro import cache_rolls, download_maestro_files, fetch_metadata, write_subset
from frozen_prior_amt.metrics import frame_metrics, note_metrics
from frozen_prior_amt.models import AudioControlBranch, BaselineCNN, ControlledDenoiser, MidiDenoiser
from frozen_prior_amt.plotting import plot_prediction_grid
from frozen_prior_amt.training import configure_torch_for_device, device_auto

SEED = 20260506
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = device_auto()
configure_torch_for_device(device)
if device.type != "cuda":
    raise RuntimeError("This experiment is intended for an A100 GPU runtime.")
print("device:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)

AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("amp dtype:", AMP_DTYPE)

def autocast():
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)

paths = Paths.from_root(".")
paths.ensure()
audio_cfg = AudioConfig()

SUBSET_NAME = "overnight64_seed20260506_gpu_v1"
PRIOR_SUBSET_NAME = "overnight_midi512_seed20260506_gpu_v1"
PIECES_PER_SPLIT = 64
PRIOR_TRAIN_PIECES = 512
WINDOWS_PER_PIECE = 4

SEEDED_PRIOR_SCALE = 0.25
SEEDED_START_FRAC = 0.75
SEEDED_THRESHOLD = 0.4

BASELINE_STEPS = 12000
BASELINE_BATCH = 64
BASELINE_HIDDEN = 128
BASELINE_LR = 2e-3
BASELINE_POS_WEIGHT = 4.0

PRIOR_STEPS = 20000
PRIOR_BATCH = 64
PRIOR_HIDDEN = 128
PRIOR_LR = 2e-4

CONTROL_STEPS = 15000
CONTROL_BATCH = 64
CONTROL_HIDDEN = 128
CONTROL_LR = 2e-4

DIFF_TIMESTEPS = 120
DIFF_SAMPLE_STEPS = 32
EVAL_BATCH = 16
GPU_CACHE_GB = 32.0
USE_TORCH_COMPILE = True
TRAIN_ACCEL_FLAGS = ["--gpu-cache", "--gpu-cache-gb", str(GPU_CACHE_GB)]
if USE_TORCH_COMPILE:
    TRAIN_ACCEL_FLAGS.append("--compile")

EXP_DIR = Path("runs/overnight_seeded_confirmation")
EXP_DIR.mkdir(parents=True, exist_ok=True)

print("train flags:", TRAIN_ACCEL_FLAGS)
            '''
        ),
        code_cell(
            '''
def stratified_sample_split(metadata, split, n, seed, min_duration=30.0, max_duration=420.0, bins=6):
    part = metadata[
        (metadata["split"] == split)
        & (metadata["duration"] >= min_duration)
        & (metadata["duration"] <= max_duration)
    ].copy()
    if len(part) < n:
        part = metadata[
            (metadata["split"] == split)
            & (metadata["duration"] >= 20.0)
            & (metadata["duration"] <= max_duration)
        ].copy()
    if len(part) < n:
        raise RuntimeError(f"not enough rows for {split}: need {n}, have {len(part)}")

    q = min(bins, len(part))
    part["duration_bin"] = pd.qcut(part["duration"], q=q, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed + {"train": 0, "validation": 10_000, "test": 20_000}[split])
    chosen = []
    per_bin = int(math.ceil(n / max(1, part["duration_bin"].nunique())))
    for _, group in part.groupby("duration_bin", sort=True):
        group = group.sort_values(["canonical_composer", "canonical_title", "row_id"])
        take = min(per_bin, len(group))
        chosen.append(group.iloc[rng.choice(len(group), size=take, replace=False)])
    out = pd.concat(chosen, ignore_index=True).drop_duplicates("row_id")
    if len(out) < n:
        remaining = part[~part["row_id"].isin(out["row_id"])]
        extra = remaining.iloc[rng.choice(len(remaining), size=n - len(out), replace=False)]
        out = pd.concat([out, extra], ignore_index=True)
    elif len(out) > n:
        out = out.iloc[rng.choice(len(out), size=n, replace=False)]
    out = out.sort_values(["split", "duration", "canonical_composer", "canonical_title", "row_id"]).drop(columns=["duration_bin"], errors="ignore")
    return out.reset_index(drop=True)

metadata = fetch_metadata(paths)
subset_path = paths.cache / f"{SUBSET_NAME}.csv"
prior_subset_path = paths.cache / f"{PRIOR_SUBSET_NAME}.csv"

if not subset_path.exists():
    subset = pd.concat(
        [
            stratified_sample_split(metadata, "train", PIECES_PER_SPLIT, SEED),
            stratified_sample_split(metadata, "validation", PIECES_PER_SPLIT, SEED),
            stratified_sample_split(metadata, "test", PIECES_PER_SPLIT, SEED),
        ],
        ignore_index=True,
    )
    write_subset(paths, subset, name=SUBSET_NAME)
else:
    subset = pd.read_csv(subset_path)

if not prior_subset_path.exists():
    prior_subset = stratified_sample_split(metadata, "train", PRIOR_TRAIN_PIECES, SEED + 1234, max_duration=540.0, bins=8)
    write_subset(paths, prior_subset, name=PRIOR_SUBSET_NAME)
else:
    prior_subset = pd.read_csv(prior_subset_path)

print("\\n=== audio/control subset ===")
print(subset.groupby("split")["duration"].agg(["count", "min", "median", "max"]).to_string())
print(subset[["split", "duration", "canonical_composer", "canonical_title"]].head(12).to_string(index=False))
print("\\n=== MIDI prior train subset ===")
print(prior_subset.groupby("split")["duration"].agg(["count", "min", "median", "max"]).to_string())
            '''
        ),
        code_cell(
            '''
def run(cmd):
    print("\\n==>", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail = []
    progress_tail = ""
    last_progress_print = 0.0
    assert proc.stdout is not None
    for line in proc.stdout:
        is_progress = "%|" in line or line.lstrip().startswith(("train ", "eval ", "fetch ", "cache "))
        if is_progress:
            progress_tail = line
            now = time.monotonic()
            if now - last_progress_print >= 10.0:
                print(line, end="")
                last_progress_print = now
        else:
            if progress_tail:
                print(progress_tail, end="")
                progress_tail = ""
                last_progress_print = time.monotonic()
            print(line, end="")
        tail.append(line)
        if len(tail) > 300:
            tail.pop(0)
    code = proc.wait()
    if progress_tail:
        print(progress_tail, end="")
    if code != 0:
        raise RuntimeError(
            "command failed with exit code "
            + str(code)
            + ": "
            + " ".join(str(c) for c in cmd)
            + "\\n--- child output tail ---\\n"
            + "".join(tail)
        )

def latest(pattern, since=0.0):
    matches = [p for p in Path("runs").glob(pattern) if p.stat().st_mtime >= since]
    matches = sorted(matches, key=lambda p: p.stat().st_mtime)
    if not matches:
        matches = sorted(Path("runs").glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise RuntimeError(f"no files matched {pattern}")
    return matches[-1]

def write_pointer(name, path):
    ptr = EXP_DIR / f"{name}_checkpoint.txt"
    ptr.write_text(str(path), encoding="utf-8")
    print(f"{name} checkpoint:", path)

def read_pointer(name):
    ptr = EXP_DIR / f"{name}_checkpoint.txt"
    if not ptr.exists():
        return None
    path = Path(ptr.read_text(encoding="utf-8").strip())
    if path.exists():
        print(f"reusing {name} checkpoint:", path)
        return path
    print(f"ignoring stale {name} pointer:", path)
    return None

py = [sys.executable, "-m", "frozen_prior_amt.cli"]
            '''
        ),
        code_cell(
            '''
# Preflight: cache a tiny audio/alignment sanity sample from the prewritten subset.
# This is safe to rerun.
run(py + ["audio-alignment", "--subset-name", SUBSET_NAME, "--pieces-per-split", str(PIECES_PER_SPLIT), "--max-pieces", "2"])

roundtrip_subset = pd.read_csv(subset_path)
if set(roundtrip_subset["row_id"].tolist()) != set(subset["row_id"].tolist()):
    raise RuntimeError(f"{SUBSET_NAME}.csv changed after preflight caching; refusing to train on the wrong subset")
            '''
        ),
        code_cell(
            '''
FORCE_BASELINE = False
baseline_ckpt = None if FORCE_BASELINE else read_pointer("baseline")
if baseline_ckpt is None:
    stage_started = time.time()
    run(py + [
        "train-baseline",
        "--subset-name", SUBSET_NAME,
        "--pieces-per-split", str(PIECES_PER_SPLIT),
        "--steps", str(BASELINE_STEPS),
        "--batch-size", str(BASELINE_BATCH),
        "--hidden", str(BASELINE_HIDDEN),
        "--lr", str(BASELINE_LR),
        "--val-windows", str(WINDOWS_PER_PIECE),
        "--positive-weight", str(BASELINE_POS_WEIGHT),
        "--seed", str(SEED),
    ] + TRAIN_ACCEL_FLAGS)
    baseline_ckpt = latest("baseline-*/baseline.pt", since=stage_started)
    write_pointer("baseline", baseline_ckpt)
            '''
        ),
        code_cell(
            '''
FORCE_PRIOR = False
prior_ckpt = None if FORCE_PRIOR else read_pointer("prior")
if prior_ckpt is None:
    stage_started = time.time()
    run(py + [
        "train-prior",
        "--subset-name", PRIOR_SUBSET_NAME,
        "--pieces-per-split", str(PRIOR_TRAIN_PIECES),
        "--steps", str(PRIOR_STEPS),
        "--batch-size", str(PRIOR_BATCH),
        "--hidden", str(PRIOR_HIDDEN),
        "--lr", str(PRIOR_LR),
        "--timesteps", str(DIFF_TIMESTEPS),
        "--sample-steps", str(DIFF_SAMPLE_STEPS),
        "--prediction-type", "x0",
        "--seed", str(SEED + 1),
    ] + TRAIN_ACCEL_FLAGS)
    prior_ckpt = latest("midi-prior-*/midi_prior.pt", since=stage_started)
    write_pointer("prior", prior_ckpt)
            '''
        ),
        code_cell(
            '''
FORCE_CONTROL = False
baseline_ckpt = read_pointer("baseline")
prior_ckpt = read_pointer("prior")
if baseline_ckpt is None or prior_ckpt is None:
    raise RuntimeError("Need baseline and prior checkpoints before training control.")

control_ckpt = None if FORCE_CONTROL else read_pointer("control")
if control_ckpt is None:
    stage_started = time.time()
    run(py + [
        "train-control",
        "--subset-name", SUBSET_NAME,
        "--pieces-per-split", str(PIECES_PER_SPLIT),
        "--steps", str(CONTROL_STEPS),
        "--batch-size", str(CONTROL_BATCH),
        "--hidden", str(CONTROL_HIDDEN),
        "--lr", str(CONTROL_LR),
        "--timesteps", str(DIFF_TIMESTEPS),
        "--sample-steps", str(DIFF_SAMPLE_STEPS),
        "--prediction-type", "x0",
        "--prior-checkpoint", str(prior_ckpt),
        "--baseline-checkpoint", str(baseline_ckpt),
        "--baseline-hidden", str(BASELINE_HIDDEN),
        "--val-windows", str(WINDOWS_PER_PIECE),
        "--seed", str(SEED + 2),
    ] + TRAIN_ACCEL_FLAGS)
    control_ckpt = latest("control-*/control_branch.pt", since=stage_started)
    write_pointer("control", control_ckpt)
            '''
        ),
        code_cell(
            '''
def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

baseline_ckpt = read_pointer("baseline")
prior_ckpt = read_pointer("prior")
control_ckpt = read_pointer("control")
if baseline_ckpt is None or prior_ckpt is None or control_ckpt is None:
    raise RuntimeError("Missing one or more checkpoints.")

baseline_payload = torch_load(baseline_ckpt)
baseline_threshold = float(baseline_payload.get("metrics", {}).get("threshold", 0.4))
baseline = BaselineCNN(hidden=BASELINE_HIDDEN).to(device).eval()
baseline.load_state_dict(baseline_payload["model"])

control_payload = torch_load(control_ckpt)
diff_payload = control_payload["diff_cfg"]
diff_cfg = DiffusionConfig(**{k: diff_payload[k] for k in DiffusionConfig.__dataclass_fields__ if k in diff_payload})
pure_control_threshold = float(control_payload.get("metrics", {}).get("threshold", 0.8))

prior = MidiDenoiser(hidden=diff_cfg.hidden).to(device).eval()
prior.load_state_dict(torch_load(prior_ckpt)["model"])
for p in prior.parameters():
    p.requires_grad_(False)

control = AudioControlBranch(hidden=diff_cfg.hidden).to(device).eval()
control.load_state_dict(control_payload["control"])
pure_control = ControlledDenoiser(prior, control).to(device).eval()

class ScaledControl(nn.Module):
    def __init__(self, prior, control, prior_scale: float):
        super().__init__()
        self.prior = prior
        self.control = control
        self.prior_scale = prior_scale

    def forward(self, x_t, t, audio):
        with torch.no_grad():
            base = self.prior(x_t, t)
        return self.prior_scale * base + self.control(x_t, t, audio)

seeded_control = ScaledControl(prior, control, SEEDED_PRIOR_SCALE).to(device).eval()
schedule = DiffusionSchedule(diff_cfg.timesteps).to(device)
seeded_start_t = max(1, min(diff_cfg.timesteps - 1, int(round((diff_cfg.timesteps - 1) * SEEDED_START_FRAC))))

test_rows = subset[subset["split"] == "test"].copy()
download_maestro_files(paths, test_rows, include_audio=True)
cache_rolls(paths, test_rows, audio_cfg)
cache_feature_set(paths, test_rows, audio_cfg)
test_pieces = load_cached_pieces(paths, test_rows, include_audio=True)
test_ds = FixedWindowDataset(test_pieces, audio_cfg, include_audio=True, windows_per_piece=WINDOWS_PER_PIECE)
print("locked overnight test windows:", len(test_ds))
            '''
        ),
        code_cell(
            '''
def batch_to_device(batch):
    return {
        k: (v.to(device=device, dtype=torch.float32, non_blocking=True) if k in {"x", "y"} else v)
        for k, v in batch.items()
    }

def mean_metrics(rows):
    keys = sorted({k for row in rows for k in row})
    return {k: float(np.mean([float(row[k]) for row in rows])) for k in keys}

@torch.no_grad()
def sample_seeded(model, baseline_probs, audio):
    x0_seed = baseline_probs.unsqueeze(1).mul(2.0).sub(1.0)
    t0 = torch.full((x0_seed.shape[0],), seeded_start_t, device=device, dtype=torch.long)
    x = q_sample(x0_seed, t0, schedule, noise=torch.randn_like(x0_seed))
    steps = torch.linspace(seeded_start_t, 0, DIFF_SAMPLE_STEPS, device=device).long()
    for i, t_scalar in enumerate(steps):
        t = torch.full((x.shape[0],), int(t_scalar.item()), device=device, dtype=torch.long)
        pred = model(x, t, audio)
        alpha_bar = schedule.alpha_bars[t].view(-1, 1, 1, 1)
        if diff_cfg.prediction_type == "x0":
            x0 = pred.clamp(-1.0, 1.0)
            eps = (x - torch.sqrt(alpha_bar) * x0) / torch.sqrt(1.0 - alpha_bar).clamp_min(1e-6)
        else:
            eps = pred
            x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        x0 = x0.clamp(-1.5, 1.5)
        if i == len(steps) - 1:
            x = x0
            break
        t_prev = int(steps[i + 1].item())
        alpha_bar_prev = schedule.alpha_bars[t_prev].view(1, 1, 1, 1)
        x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps
    return diffusion_to_roll(x)

def metric_row(pred, target, threshold):
    return {
        **frame_metrics(pred, target, threshold=threshold),
        **note_metrics(pred, target, audio_cfg.frame_hz, threshold=threshold),
    }

@torch.no_grad()
def run_locked_overnight_test():
    loader = torch.utils.data.DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False, pin_memory=True)
    rows = {"baseline": [], "pure_control": [], "seeded_refinement": []}
    saved_examples = []
    for batch_index, batch in enumerate(tqdm(loader, desc="locked overnight test eval", mininterval=10.0, maxinterval=30.0)):
        batch = batch_to_device(batch)
        with autocast():
            baseline_probs = torch.sigmoid(baseline(batch["x"])).float()
            pure_pred = sample_ddim(
                pure_control,
                (batch["y"].shape[0], 1, 88, audio_cfg.frames_per_window),
                schedule,
                DIFF_SAMPLE_STEPS,
                device,
                audio=batch["x"],
                prediction_type=diff_cfg.prediction_type,
            ).float()
            seeded_pred = sample_seeded(seeded_control, baseline_probs, batch["x"]).float()

        y_np = batch["y"].float().cpu().numpy()
        base_np = baseline_probs.cpu().numpy()
        pure_np = pure_pred.cpu().numpy()
        seeded_np = seeded_pred.cpu().numpy()
        for i in range(len(y_np)):
            rows["baseline"].append(metric_row(base_np[i], y_np[i], baseline_threshold))
            rows["pure_control"].append(metric_row(pure_np[i], y_np[i], pure_control_threshold))
            rows["seeded_refinement"].append(metric_row(seeded_np[i], y_np[i], SEEDED_THRESHOLD))
            if len(saved_examples) < 12:
                saved_examples.append({
                    "global_index": batch_index * EVAL_BATCH + i,
                    "piece_id": batch["piece_id"][i],
                    "start_frame": int(batch["start_frame"][i]),
                    "target": y_np[i],
                    "baseline": base_np[i],
                    "pure_control": pure_np[i],
                    "seeded": seeded_np[i],
                })

    summary_rows = []
    for stage, stage_rows in rows.items():
        threshold = baseline_threshold if stage == "baseline" else pure_control_threshold if stage == "pure_control" else SEEDED_THRESHOLD
        summary_rows.append({"stage": stage, "threshold": threshold, "windows": len(stage_rows), **mean_metrics(stage_rows)})
    return pd.DataFrame(summary_rows).sort_values("onset_f1", ascending=False), saved_examples

summary, examples = run_locked_overnight_test()
summary_path = EXP_DIR / "overnight_locked_test_metrics.tsv"
summary.to_csv(summary_path, sep="\\t", index=False)

config = {
    "seed": SEED,
    "subset_name": SUBSET_NAME,
    "prior_subset_name": PRIOR_SUBSET_NAME,
    "pieces_per_split": PIECES_PER_SPLIT,
    "prior_train_pieces": PRIOR_TRAIN_PIECES,
    "windows_per_piece": WINDOWS_PER_PIECE,
    "baseline_checkpoint": str(baseline_ckpt),
    "prior_checkpoint": str(prior_ckpt),
    "control_checkpoint": str(control_ckpt),
    "baseline_threshold": baseline_threshold,
    "pure_control_threshold": pure_control_threshold,
    "seeded_prior_scale": SEEDED_PRIOR_SCALE,
    "seeded_start_frac": SEEDED_START_FRAC,
    "seeded_start_t": seeded_start_t,
    "seeded_threshold": SEEDED_THRESHOLD,
    "diff_sample_steps": DIFF_SAMPLE_STEPS,
    "a100_profile": {
        "baseline_steps": BASELINE_STEPS,
        "baseline_batch": BASELINE_BATCH,
        "baseline_hidden": BASELINE_HIDDEN,
        "prior_steps": PRIOR_STEPS,
        "prior_batch": PRIOR_BATCH,
        "prior_hidden": PRIOR_HIDDEN,
        "control_steps": CONTROL_STEPS,
        "control_batch": CONTROL_BATCH,
        "control_hidden": CONTROL_HIDDEN,
        "timesteps": DIFF_TIMESTEPS,
        "gpu_cache_gb": GPU_CACHE_GB,
        "torch_compile": USE_TORCH_COMPILE,
    },
    "test_piece_ids": test_rows["piece_id"].tolist(),
}
(EXP_DIR / "overnight_locked_test_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

print("\\n=== overnight locked test metrics ===")
print(summary[[
    "stage", "threshold", "windows", "onset_f1", "onset_precision", "onset_recall",
    "onset_offset_f1", "false_positives_per_min", "short_fragments_per_min",
    "frame_f1", "notes_pred", "notes_target",
]].to_string(index=False))
print("\\nwrote", summary_path)
            '''
        ),
        code_cell(
            '''
def roll_to_midi(roll, out_path, threshold=0.4, frame_hz=50, min_duration=0.06):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))
    active = roll >= threshold
    min_frames = max(1, int(round(min_duration * frame_hz)))
    for key in range(active.shape[0]):
        arr = active[key]
        idx = 0
        while idx < len(arr):
            if not arr[idx]:
                idx += 1
                continue
            start = idx
            while idx < len(arr) and arr[idx]:
                idx += 1
            end = idx
            if end - start >= min_frames:
                inst.notes.append(pretty_midi.Note(
                    velocity=80,
                    pitch=MIDI_MIN + key,
                    start=start / frame_hz,
                    end=end / frame_hz,
                ))
    pm.instruments.append(inst)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))

plot_dir = Path("plots/predictions/overnight_seeded_confirmation")
midi_dir = EXP_DIR / "midi"
plot_dir.mkdir(parents=True, exist_ok=True)
midi_dir.mkdir(parents=True, exist_ok=True)

for ex in examples:
    stem = f"{ex['global_index']:03d}_{ex['piece_id']}_frame{ex['start_frame']}"
    plot_prediction_grid(
        {
            "ground truth": ex["target"],
            "baseline": ex["baseline"],
            "pure control": ex["pure_control"],
            "seeded refinement": ex["seeded"],
        },
        audio_cfg.frame_hz,
        plot_dir / f"{stem}.png",
    )
    roll_to_midi(ex["target"], midi_dir / f"{stem}_ground_truth.mid", threshold=0.5, frame_hz=audio_cfg.frame_hz)
    roll_to_midi(ex["baseline"], midi_dir / f"{stem}_baseline.mid", threshold=baseline_threshold, frame_hz=audio_cfg.frame_hz)
    roll_to_midi(ex["pure_control"], midi_dir / f"{stem}_pure_control.mid", threshold=pure_control_threshold, frame_hz=audio_cfg.frame_hz)
    roll_to_midi(ex["seeded"], midi_dir / f"{stem}_seeded_refinement.mid", threshold=SEEDED_THRESHOLD, frame_hz=audio_cfg.frame_hz)

plots_zip = Path("/content/frozen_prior_amt_overnight64_plots.zip")
runs_tar = Path("/content/frozen_prior_amt_overnight64_metrics_and_midi.tar.gz")
for p in [plots_zip, runs_tar]:
    if p.exists():
        p.unlink()
shutil.make_archive(str(plots_zip.with_suffix("")), "zip", root_dir=root, base_dir="plots")
with tarfile.open(runs_tar, "w:gz") as tf:
    for rel in [
        "runs/results.tsv",
        "runs/overnight_seeded_confirmation",
        f"data/cache/{SUBSET_NAME}.csv",
        f"data/cache/{SUBSET_NAME}.manifest.json",
        f"data/cache/{PRIOR_SUBSET_NAME}.csv",
        f"data/cache/{PRIOR_SUBSET_NAME}.manifest.json",
    ]:
        path = root / rel
        if path.exists():
            tf.add(path, arcname=rel)

print("packaged", plots_zip)
print("packaged", runs_tar)
try:
    from google.colab import files
    files.download(str(plots_zip))
    files.download(str(runs_tar))
except Exception as exc:
    print("Automatic download did not start:", exc)
    print("Download these from the Colab file browser:")
    print(plots_zip)
    print(runs_tar)
            '''
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
