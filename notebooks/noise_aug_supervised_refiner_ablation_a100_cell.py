# Noise-augmented supervised refiner ablation for the post-overnight checkpoint set.
#
# Run this as one Colab A100 cell after the repo payload has been unpacked at
# /content/frozen_prior_amt. It trains a one-shot supervised refiner with
# diffusion-schedule noise applied to the baseline seed during training, then
# re-runs the plain supervised, frozen-prior, and no-prior checkpoints through
# the same evaluator. It writes the result table/CIs/figure/interpretation and
# saves a reload-and-eval snippet for the new checkpoint.

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/content/frozen_prior_amt")
if not ROOT.exists():
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
for name in list(sys.modules):
    if name == "frozen_prior_amt" or name.startswith("frozen_prior_amt."):
        del sys.modules[name]

ARCHIVE = ROOT / "newest_run/post_overnight/archives/prior_ablation_20260506_182141_FULL_WITH_CHECKPOINTS.tar.gz"
EXTRACTED = ROOT / "newest_run/post_overnight/extracted"
RUN_NAME = "noise_aug_supervised_refiner_ablation_20260506"
OUT_DIR = ROOT / "runs" / RUN_NAME
FIG_DIR = ROOT / "plots/predictions" / RUN_NAME
ARTIFACT_DIR = ROOT / "artifacts" / RUN_NAME
PLAIN_SUPERVISED_RUN = ROOT / "runs/supervised_refiner_ablation_20260506"

SUBSET_NAME = "clever_sprint_audio10_seed20260508"
SOURCE_RUN = EXTRACTED / "runs/clever_sprint_20260506_172222"
FOLLOWUP_RUN = EXTRACTED / "runs/thirty_min_followup_20260506_181000"
PRIOR_ABLATION_RUN = EXTRACTED / "runs/prior_ablation_20260506_182141"

SEED = 20260538
TRAIN_STEPS = 700
TRAIN_BATCH = 96
HIDDEN = 96
LR = 2e-4
POSITIVE_WEIGHT = 4.0
GPU_CACHE_GB = 28.0
WINDOWS_PER_PIECE = 4
EVAL_BATCH = 16
DIFF_SAMPLE_STEPS = 24
SEEDED_START_FRAC = 0.75
SEEDED_PRIOR_SCALE = 1.0
BOOTSTRAP_RESAMPLES = 1000
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9)


if not ARCHIVE.exists():
    raise FileNotFoundError(f"Missing checkpoint archive: {ARCHIVE}")

if not (EXTRACTED / "runs").exists():
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        tf.extractall(EXTRACTED)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)], check=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from frozen_prior_amt.config import AudioConfig, DiffusionConfig, Paths, TrainConfig
from frozen_prior_amt.dataset import FixedWindowDataset, RandomWindowDataset, load_cached_pieces
from frozen_prior_amt.diffusion import DiffusionSchedule, diffusion_to_roll, q_sample
from frozen_prior_amt.features import cache_feature_set
from frozen_prior_amt.maestro import cache_rolls, download_maestro_files
from frozen_prior_amt.metrics import frame_metrics, note_metrics
from frozen_prior_amt.models import AudioControlBranch, BaselineCNN, MidiDenoiser, SupervisedRefiner
from frozen_prior_amt.training import configure_torch_for_device, train_noise_augmented_supervised_refiner


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float, np.floating)):
                acc[k].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def metric_row(pred: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        **frame_metrics(pred, target, threshold=threshold),
        **note_metrics(pred, target, audio_cfg.frame_hz, threshold=threshold),
    }


def batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    out = {}
    for k, v in batch.items():
        if k in {"x", "y"}:
            t = v.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            if t.ndim == 4 and device.type == "cuda":
                t = t.contiguous(memory_format=torch.channels_last)
            out[k] = t
        else:
            out[k] = v
    return out


def maybe_compile(model: nn.Module, label: str) -> nn.Module:
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print(f"torch.compile enabled for {label}")
        return compiled
    except Exception as exc:
        print(f"torch.compile skipped for {label}: {exc}")
        return model


def loader_for(dataset, batch_size: int):
    workers = min(4, max(0, (os.cpu_count() or 2) - 2))
    kwargs = {}
    if workers:
        kwargs = {"num_workers": workers, "persistent_workers": True, "prefetch_factor": 4}
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        **kwargs,
    )


class ScaledControl(nn.Module):
    def __init__(self, prior: MidiDenoiser, control: AudioControlBranch, prior_scale: float):
        super().__init__()
        self.prior = prior
        self.control = control
        self.prior_scale = prior_scale

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.prior(x_t, t)
        return self.prior_scale * base + self.control(x_t, t, audio)


class NoPriorControl(nn.Module):
    def __init__(self, control: AudioControlBranch):
        super().__init__()
        self.control = control

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.control(x_t, t, audio)


@torch.no_grad()
def sample_seeded(model: nn.Module, baseline_probs: torch.Tensor, audio: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    x0_seed = baseline_probs.unsqueeze(1).mul(2.0).sub(1.0)
    t0 = torch.full((x0_seed.shape[0],), seeded_start_t, device=device, dtype=torch.long)
    x = q_sample(x0_seed, t0, schedule, noise=noise)
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
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


@torch.no_grad()
def collect_predictions(dataset, split_name: str) -> list[dict[str, object]]:
    torch.manual_seed(SEED + (17 if split_name == "validation" else 29))
    records: list[dict[str, object]] = []
    for batch in tqdm(loader_for(dataset, EVAL_BATCH), desc=f"{split_name} shared eval", mininterval=10.0, maxinterval=30.0):
        batch = batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            baseline_probs = torch.sigmoid(baseline_eval(batch["x"])).float()
            supervised_probs = torch.sigmoid(plain_refiner_eval(baseline_probs, batch["x"])).float()
            noise_aug_supervised_probs = torch.sigmoid(noise_aug_refiner_eval(baseline_probs, batch["x"])).float()
            shared_noise = torch.randn(
                (baseline_probs.shape[0], 1, baseline_probs.shape[1], baseline_probs.shape[2]),
                device=device,
                dtype=baseline_probs.dtype,
            )
            no_prior_probs = sample_seeded(no_prior_eval, baseline_probs, batch["x"], shared_noise).float()
            frozen_prior_probs = sample_seeded(frozen_prior_eval, baseline_probs, batch["x"], shared_noise).float()

        y_np = batch["y"].detach().float().cpu().numpy()
        preds_np = {
            "baseline_raw": baseline_probs.detach().cpu().numpy(),
            "supervised_refiner": supervised_probs.detach().cpu().numpy(),
            "noise_aug_supervised_refiner": noise_aug_supervised_probs.detach().cpu().numpy(),
            "no_prior_distmatched": no_prior_probs.detach().cpu().numpy(),
            "frozen_prior_distmatched": frozen_prior_probs.detach().cpu().numpy(),
        }
        piece_ids = batch["piece_id"]
        starts = batch["start_frame"]
        for i in range(len(y_np)):
            records.append(
                {
                    "piece_id": piece_ids[i],
                    "start_frame": int(starts[i]),
                    "target": y_np[i],
                    "preds": {name: value[i] for name, value in preds_np.items()},
                }
            )
    return records


def threshold_sweep(records: list[dict[str, object]], split_name: str) -> pd.DataFrame:
    rows = []
    for variant in [
        "baseline_raw",
        "supervised_refiner",
        "noise_aug_supervised_refiner",
        "no_prior_distmatched",
        "frozen_prior_distmatched",
    ]:
        for threshold in THRESHOLDS:
            metric_rows = [
                metric_row(record["preds"][variant], record["target"], threshold)  # type: ignore[index]
                for record in records
            ]
            rows.append(
                {
                    "split": split_name,
                    "variant": variant,
                    "threshold": threshold,
                    "windows": len(metric_rows),
                    **mean_metrics(metric_rows),
                }
            )
    return pd.DataFrame(rows)


def selected_results(records: list[dict[str, object]], selected_thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    window_rows = []
    for variant, threshold in selected_thresholds.items():
        metric_rows = []
        for record in records:
            row = metric_row(record["preds"][variant], record["target"], threshold)  # type: ignore[index]
            row.update(
                {
                    "variant": variant,
                    "threshold": threshold,
                    "piece_id": record["piece_id"],
                    "start_frame": record["start_frame"],
                }
            )
            metric_rows.append(row)
            window_rows.append(row)
        summary_rows.append({"variant": variant, "threshold": threshold, "windows": len(metric_rows), **mean_metrics(metric_rows)})
    return pd.DataFrame(summary_rows), pd.DataFrame(window_rows)


def bootstrap_deltas(window_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(SEED)
    piece_ids = sorted(window_df["piece_id"].unique().tolist())
    by_piece = (
        window_df.groupby(["variant", "piece_id"], as_index=False)["onset_f1"]
        .mean()
        .pivot(index="piece_id", columns="variant", values="onset_f1")
    )
    pairs = {
        "noise_aug_supervised_minus_supervised_refiner": ("noise_aug_supervised_refiner", "supervised_refiner"),
        "noise_aug_supervised_minus_no_prior_distmatched": ("noise_aug_supervised_refiner", "no_prior_distmatched"),
        "noise_aug_supervised_minus_frozen_prior_distmatched": ("noise_aug_supervised_refiner", "frozen_prior_distmatched"),
    }
    out = {}
    for name, (a, b) in pairs.items():
        samples = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            chosen = rng.choice(piece_ids, size=len(piece_ids), replace=True)
            samples.append(float((by_piece.loc[chosen, a] - by_piece.loc[chosen, b]).mean()))
        arr = np.asarray(samples, dtype=np.float64)
        out[name] = {
            "median": float(np.median(arr)),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }
    return out


def interpretation(results: pd.DataFrame, cis: dict[str, dict[str, float]]) -> str:
    vals = {row.variant: row for row in results.itertuples(index=False)}
    base = vals["baseline_raw"]
    sup = vals["supervised_refiner"]
    aug = vals["noise_aug_supervised_refiner"]
    nop = vals["no_prior_distmatched"]
    frozen = vals["frozen_prior_distmatched"]
    d_plain = cis["noise_aug_supervised_minus_supervised_refiner"]
    d_nop = cis["noise_aug_supervised_minus_no_prior_distmatched"]
    d_frozen = cis["noise_aug_supervised_minus_frozen_prior_distmatched"]
    closes_nop = abs(aug.onset_f1 - nop.onset_f1) <= 0.05 and d_nop["ci_low"] <= 0.0 <= d_nop["ci_high"]
    closes_frozen = abs(aug.onset_f1 - frozen.onset_f1) <= 0.05 and d_frozen["ci_low"] <= 0.0 <= d_frozen["ci_high"]
    lines = [
        f"The raw baseline reached onset F1 {base.onset_f1:.3f} at threshold {base.threshold:.2f}.",
        f"The plain supervised refiner reached {sup.onset_f1:.3f} at threshold {sup.threshold:.2f}.",
        f"The noise-augmented supervised refiner reached {aug.onset_f1:.3f} at threshold {aug.threshold:.2f}.",
        f"The no-prior and frozen-prior diffusion refiners reached {nop.onset_f1:.3f} and {frozen.onset_f1:.3f}, respectively.",
        f"Noise-augmented supervised minus plain supervised is {aug.onset_f1 - sup.onset_f1:+.3f} by the table, with bootstrap median {d_plain['median']:+.3f} and 95% CI [{d_plain['ci_low']:+.3f}, {d_plain['ci_high']:+.3f}].",
        f"Noise-augmented supervised minus no-prior diffusion is {aug.onset_f1 - nop.onset_f1:+.3f} by the table, with bootstrap median {d_nop['median']:+.3f} and 95% CI [{d_nop['ci_low']:+.3f}, {d_nop['ci_high']:+.3f}].",
        f"Noise-augmented supervised minus frozen-prior diffusion is {aug.onset_f1 - frozen.onset_f1:+.3f} by the table, with bootstrap median {d_frozen['median']:+.3f} and 95% CI [{d_frozen['ci_low']:+.3f}, {d_frozen['ci_high']:+.3f}].",
    ]
    if closes_nop and closes_frozen:
        lines.append("The noise-augmented supervised refiner closes most of the gap to diffusion, so the diffusion advantage appears substantially driven by implicit data augmentation and the paper should reframe around noise-augmented refinement with diffusion as one instantiation.")
    elif d_nop["ci_high"] < 0 and d_frozen["ci_high"] < 0:
        lines.append("The noise-augmented supervised refiner remains clearly below both diffusion rows, so iterative denoising at inference is doing real work beyond augmentation and the diffusion framing is correct for this ablation.")
    else:
        lines.append("The noise-augmented supervised refiner lands between the plain supervised and diffusion rows or has intervals that are not decisive, so this run should not pick a hard side without more seeds or a larger eval set.")
    drift = "down toward the diffusion thresholds" if aug.threshold < sup.threshold else "near or above the plain supervised threshold"
    lines.append(f"Its selected threshold is {aug.threshold:.2f}, which is {drift} relative to plain supervised {sup.threshold:.2f} and diffusion {nop.threshold:.2f}/{frozen.threshold:.2f}.")
    lines.append(f"False positives/min are noise-aug supervised {aug.false_positives_per_min:.1f}, no-prior diffusion {nop.false_positives_per_min:.1f}, and frozen-prior diffusion {frozen.false_positives_per_min:.1f}.")
    lines.append(f"Short fragments/min are noise-aug supervised {aug.short_fragments_per_min:.1f}, plain supervised {sup.short_fragments_per_min:.1f}, no-prior diffusion {nop.short_fragments_per_min:.1f}, and frozen-prior diffusion {frozen.short_fragments_per_min:.1f}.")
    return " ".join(lines)


def write_reload_snippet(path: Path, selected_threshold: float) -> None:
    snippet = f'''
from pathlib import Path
import torch
import pandas as pd
from frozen_prior_amt.config import AudioConfig, Paths
from frozen_prior_amt.dataset import FixedWindowDataset, load_cached_pieces
from frozen_prior_amt.features import cache_feature_set
from frozen_prior_amt.maestro import cache_rolls, download_maestro_files
from frozen_prior_amt.metrics import frame_metrics, note_metrics
from frozen_prior_amt.models import BaselineCNN, SupervisedRefiner
from frozen_prior_amt.training import configure_torch_for_device

ROOT = Path("/content/frozen_prior_amt")
paths = Paths.from_root(ROOT)
audio_cfg = AudioConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configure_torch_for_device(device)
subset = pd.read_csv(paths.cache / "{SUBSET_NAME}.csv")
test_rows = subset[subset["split"] == "test"].copy()
download_maestro_files(paths, test_rows, include_audio=True)
cache_rolls(paths, test_rows, audio_cfg)
cache_feature_set(paths, test_rows, audio_cfg)
test_ds = FixedWindowDataset(load_cached_pieces(paths, test_rows, include_audio=True), audio_cfg, include_audio=True, windows_per_piece={WINDOWS_PER_PIECE})

baseline_payload = torch.load(ROOT / "newest_run/post_overnight/extracted/runs/clever_sprint_20260506_172222/baseline/baseline.pt", map_location="cpu", weights_only=False)
baseline = BaselineCNN(hidden={HIDDEN}).to(device).eval()
baseline.load_state_dict(baseline_payload["model"])
refiner_payload = torch.load(ROOT / "runs/{RUN_NAME}/noise_aug_supervised_refiner.pt", map_location="cpu", weights_only=False)
refiner = SupervisedRefiner(hidden={HIDDEN}).to(device).eval()
refiner.load_state_dict(refiner_payload["model"])

rows = []
loader = torch.utils.data.DataLoader(test_ds, batch_size={EVAL_BATCH}, shuffle=False, pin_memory=device.type == "cuda")
with torch.no_grad():
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        y = batch["y"].numpy()
        seed = torch.sigmoid(baseline(x)).float()
        pred = torch.sigmoid(refiner(seed, x)).float().cpu().numpy()
        for pred_i, y_i in zip(pred, y):
            rows.append({{**frame_metrics(pred_i, y_i, threshold={selected_threshold}), **note_metrics(pred_i, y_i, audio_cfg.frame_hz, threshold={selected_threshold})}})
print({{k: sum(float(r[k]) for r in rows) / len(rows) for k in rows[0] if isinstance(rows[0][k], (int, float))}})
'''
    path.write_text(snippet.strip() + "\n", encoding="utf-8")


start_time = time.time()
for path in [OUT_DIR, FIG_DIR, ARTIFACT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

paths = Paths.from_root(ROOT)
paths.ensure()
subset_src = EXTRACTED / f"data/cache/{SUBSET_NAME}.csv"
subset_dst = paths.cache / f"{SUBSET_NAME}.csv"
if subset_src.exists() and not subset_dst.exists():
    subset_dst.parent.mkdir(parents=True, exist_ok=True)
    subset_dst.write_bytes(subset_src.read_bytes())
subset = pd.read_csv(subset_dst)

audio_cfg = AudioConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configure_torch_for_device(device)
if device.type != "cuda":
    raise RuntimeError("This ablation is intended for the Colab A100 runtime.")
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("device:", torch.cuda.get_device_name(0), "amp:", amp_dtype)

train_rows = subset[subset["split"] == "train"].copy()
val_rows = subset[subset["split"] == "validation"].copy()
test_rows = subset[subset["split"] == "test"].copy()
needed_rows = pd.concat([train_rows, val_rows, test_rows], ignore_index=True)
download_maestro_files(paths, needed_rows, include_audio=True)
cache_rolls(paths, needed_rows, audio_cfg)
cache_feature_set(paths, needed_rows, audio_cfg)

train_pieces = load_cached_pieces(paths, train_rows, include_audio=True)
val_pieces = load_cached_pieces(paths, val_rows, include_audio=True)
test_pieces = load_cached_pieces(paths, test_rows, include_audio=True)
train_ds = RandomWindowDataset(train_pieces, audio_cfg, num_samples=TRAIN_STEPS * TRAIN_BATCH, include_audio=True, seed=SEED)
val_ds = FixedWindowDataset(val_pieces, audio_cfg, include_audio=True, windows_per_piece=WINDOWS_PER_PIECE)
test_ds = FixedWindowDataset(test_pieces, audio_cfg, include_audio=True, windows_per_piece=WINDOWS_PER_PIECE)

baseline_payload = torch_load(SOURCE_RUN / "baseline/baseline.pt")
baseline = BaselineCNN(hidden=HIDDEN).to(device).eval()
baseline.load_state_dict(baseline_payload["model"])
for param in baseline.parameters():
    param.requires_grad_(False)

plain_supervised_ckpt = PLAIN_SUPERVISED_RUN / "supervised_refiner.pt"
if not plain_supervised_ckpt.exists():
    raise FileNotFoundError(f"Missing yesterday's plain supervised checkpoint: {plain_supervised_ckpt}")
plain_refiner_payload = torch_load(plain_supervised_ckpt)
plain_refiner = SupervisedRefiner(hidden=HIDDEN).to(device).eval()
plain_refiner.load_state_dict(plain_refiner_payload["model"])

prior_payload = torch_load(SOURCE_RUN / "midi_prior/midi_prior.pt")
diff_payload = prior_payload["diff_cfg"]
diff_cfg = DiffusionConfig(**{k: diff_payload[k] for k in DiffusionConfig.__dataclass_fields__ if k in diff_payload})
diff_cfg = DiffusionConfig(**{**diff_cfg.__dict__, "sample_steps": DIFF_SAMPLE_STEPS})
schedule = DiffusionSchedule(diff_cfg.timesteps).to(device)
seeded_start_t = max(1, min(diff_cfg.timesteps - 1, int(round((diff_cfg.timesteps - 1) * SEEDED_START_FRAC))))

noise_aug_ckpt = OUT_DIR / "noise_aug_supervised_refiner.pt"
if noise_aug_ckpt.exists():
    print("reusing noise-augmented supervised refiner checkpoint:", noise_aug_ckpt)
    noise_aug_payload = torch_load(noise_aug_ckpt)
    noise_aug_refiner = SupervisedRefiner(hidden=HIDDEN).to(device).eval()
    noise_aug_refiner.load_state_dict(noise_aug_payload["model"])
else:
    train_cfg = TrainConfig(
        steps=TRAIN_STEPS,
        batch_size=TRAIN_BATCH,
        lr=LR,
        hidden=HIDDEN,
        positive_weight=POSITIVE_WEIGHT,
        seed=SEED,
        num_workers=0,
        gpu_cache=True,
        gpu_cache_gb=GPU_CACHE_GB,
        compile_model=True,
    )
    noise_aug_refiner, noise_aug_metrics = train_noise_augmented_supervised_refiner(
        baseline,
        train_ds,
        val_ds,
        audio_cfg,
        train_cfg,
        diff_cfg,
        OUT_DIR,
        device=device,
    )
    print("noise-augmented supervised validation metrics:", noise_aug_metrics)

prior = MidiDenoiser(hidden=HIDDEN).to(device).eval()
prior.load_state_dict(prior_payload["model"])
for param in prior.parameters():
    param.requires_grad_(False)

frozen_control_payload = torch_load(FOLLOWUP_RUN / "distmatched_control_branch.pt")
frozen_control = AudioControlBranch(hidden=HIDDEN).to(device).eval()
frozen_control.load_state_dict(frozen_control_payload["control"])
frozen_prior_model = ScaledControl(prior, frozen_control, SEEDED_PRIOR_SCALE).to(device).eval()

no_prior_payload = torch_load(PRIOR_ABLATION_RUN / "noprior_distmatched_control_branch.pt")
no_prior_control = AudioControlBranch(hidden=HIDDEN).to(device).eval()
no_prior_control.load_state_dict(no_prior_payload["control"])
no_prior_model = NoPriorControl(no_prior_control).to(device).eval()

baseline_eval = maybe_compile(baseline, "baseline eval")
plain_refiner_eval = maybe_compile(plain_refiner, "plain supervised refiner eval")
noise_aug_refiner_eval = maybe_compile(noise_aug_refiner, "noise-augmented supervised refiner eval")
frozen_prior_eval = maybe_compile(frozen_prior_model, "frozen-prior diffusion eval")
no_prior_eval = maybe_compile(no_prior_model, "no-prior diffusion eval")

val_records = collect_predictions(val_ds, "validation")
test_records = collect_predictions(test_ds, "locked test")
val_sweep = threshold_sweep(val_records, "validation")
test_sweep = threshold_sweep(test_records, "locked_test")
selected_thresholds = (
    val_sweep.sort_values(["variant", "onset_f1"], ascending=[True, False])
    .groupby("variant", as_index=False)
    .first()
    .set_index("variant")["threshold"]
    .to_dict()
)

results, window_metrics = selected_results(test_records, selected_thresholds)
variant_order = [
    "baseline_raw",
    "supervised_refiner",
    "noise_aug_supervised_refiner",
    "no_prior_distmatched",
    "frozen_prior_distmatched",
]
results["variant"] = pd.Categorical(results["variant"], categories=variant_order, ordered=True)
results = results.sort_values("variant").reset_index(drop=True)
cis = bootstrap_deltas(window_metrics)
interp = interpretation(results, cis)

val_sweep.to_csv(ARTIFACT_DIR / "validation_threshold_sweep.tsv", sep="\t", index=False)
test_sweep.to_csv(ARTIFACT_DIR / "test_threshold_sweep.tsv", sep="\t", index=False)
results.to_csv(ARTIFACT_DIR / "results_table.tsv", sep="\t", index=False)
window_metrics.to_csv(ARTIFACT_DIR / "test_window_metrics.tsv", sep="\t", index=False)
(ARTIFACT_DIR / "bootstrap_cis.json").write_text(json.dumps(cis, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(7.2, 4.8))
for variant in variant_order:
    sub = test_sweep[test_sweep["variant"] == variant].sort_values("false_positives_per_min")
    ax.plot(sub["false_positives_per_min"], sub["onset_f1"], marker="o", linewidth=1.8, label=variant.replace("_", " "))
ax.set_xlabel("False positives per minute")
ax.set_ylabel("Onset F1")
ax.set_title("Locked-test F1 vs false-positive budget")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
figure_path = ARTIFACT_DIR / "f1_vs_fp_budget.png"
fig.savefig(figure_path, dpi=180)
plt.show()

write_reload_snippet(
    ARTIFACT_DIR / "reload_noise_aug_supervised_refiner_eval.py",
    float(selected_thresholds["noise_aug_supervised_refiner"]),
)

report = [
    "# Noise-Augmented Supervised Refiner Ablation",
    "",
    "## Locked Test Results",
    "",
    results[[
        "variant",
        "threshold",
        "windows",
        "onset_f1",
        "onset_precision",
        "onset_recall",
        "false_positives_per_min",
        "short_fragments_per_min",
    ]].to_markdown(index=False),
    "",
    "## Bootstrap 95% CIs",
    "",
]
for name, vals in cis.items():
    report.append(f"- `{name}`: median {vals['median']:+.3f}, 95% CI [{vals['ci_low']:+.3f}, {vals['ci_high']:+.3f}]")
report.extend(["", "## Interpretation", "", interp, "", f"Figure: `{figure_path}`"])
(ARTIFACT_DIR / "noise_aug_supervised_refiner_ablation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

manifest = {
    "run_name": RUN_NAME,
    "subset_name": SUBSET_NAME,
    "source_checkpoint_run": str(SOURCE_RUN.relative_to(ROOT)),
    "frozen_prior_checkpoint": str((FOLLOWUP_RUN / "distmatched_control_branch.pt").relative_to(ROOT)),
    "no_prior_checkpoint": str((PRIOR_ABLATION_RUN / "noprior_distmatched_control_branch.pt").relative_to(ROOT)),
    "plain_supervised_checkpoint": str(plain_supervised_ckpt.relative_to(ROOT)),
    "noise_aug_supervised_checkpoint": str(noise_aug_ckpt.relative_to(ROOT)),
    "threshold_selection": "validation onset_f1, locked test after selection",
    "bootstrap": "1000 resamples over test piece ids",
    "train": {
        "steps": TRAIN_STEPS,
        "batch_size": TRAIN_BATCH,
        "hidden": HIDDEN,
        "lr": LR,
        "positive_weight": POSITIVE_WEIGHT,
        "bf16": torch.cuda.is_bf16_supported(),
        "tf32": True,
        "torch_compile": True,
        "gpu_cache_gb": GPU_CACHE_GB,
        "only_change_from_plain_supervised": "diffusion-schedule noise on baseline seed during training",
        "noise_t_range": f"uniform integer [1, {diff_cfg.timesteps - 1}]",
        "timestep_conditioning": False,
        "clean_input_mixture": False,
    },
    "diffusion_eval": {
        "sample_steps": DIFF_SAMPLE_STEPS,
        "start_frac": SEEDED_START_FRAC,
        "start_t": seeded_start_t,
        "prior_scale": SEEDED_PRIOR_SCALE,
    },
    "selected_thresholds": {k: float(v) for k, v in selected_thresholds.items()},
    "elapsed_minutes": (time.time() - start_time) / 60.0,
}
(ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

bundle_path = ARTIFACT_DIR / f"{RUN_NAME}_artifacts.tar.gz"
with tarfile.open(bundle_path, "w:gz") as tf:
    for path in [
        ARTIFACT_DIR / "results_table.tsv",
        ARTIFACT_DIR / "validation_threshold_sweep.tsv",
        ARTIFACT_DIR / "test_threshold_sweep.tsv",
        ARTIFACT_DIR / "test_window_metrics.tsv",
        ARTIFACT_DIR / "bootstrap_cis.json",
        ARTIFACT_DIR / "f1_vs_fp_budget.png",
        ARTIFACT_DIR / "reload_noise_aug_supervised_refiner_eval.py",
        ARTIFACT_DIR / "noise_aug_supervised_refiner_ablation_report.md",
        ARTIFACT_DIR / "manifest.json",
        noise_aug_ckpt,
    ]:
        tf.add(path, arcname=path.relative_to(ROOT))

print("\n=== locked test result table ===")
print(results[[
    "variant",
    "threshold",
    "windows",
    "onset_f1",
    "onset_precision",
    "onset_recall",
    "false_positives_per_min",
    "short_fragments_per_min",
]].to_string(index=False))
print("\n=== bootstrap CIs ===")
print(json.dumps(cis, indent=2))
print("\n=== interpretation ===")
print(interp)
print("\nwrote artifacts:", ARTIFACT_DIR)
print("wrote bundle:", bundle_path)
print(f"elapsed minutes: {(time.time() - start_time) / 60.0:.1f}")
