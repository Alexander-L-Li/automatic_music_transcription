from __future__ import annotations

from contextlib import nullcontext
from collections import defaultdict
import importlib
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import AudioConfig, DiffusionConfig, TrainConfig
from .diffusion import DiffusionSchedule, q_sample, roll_to_diffusion_target, sample_ddim
from .metrics import frame_metrics, note_metrics
from .models import AudioControlBranch, BaselineCNN, ControlledDenoiser, MidiDenoiser, SupervisedRefiner
from .plotting import plot_prediction_grid
from .utils import set_seed


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch_for_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        dynamo = importlib.import_module("torch._dynamo")
        dynamo.config.suppress_errors = True
    except Exception:
        pass


def accelerator_summary(device: torch.device | None = None) -> dict[str, str]:
    device = device or device_auto()
    summary = {"device": str(device), "amp": "off", "tf32": "off"}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        summary.update(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": f"{props.major}.{props.minor}",
                "memory_gb": f"{props.total_memory / 1024**3:.1f}",
                "amp": dtype,
                "tf32": "on",
            }
        )
    return summary


def _amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _autocast(device: torch.device):
    dtype = _amp_dtype(device)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _channels_last_if_cuda(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda" and x.ndim == 4:
        if x.is_contiguous(memory_format=torch.channels_last):
            return x
        return x.contiguous(memory_format=torch.channels_last)
    return x


def _model_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def _make_scaler(device: torch.device):
    enabled = device.type == "cuda" and _amp_dtype(device) == torch.float16
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def _make_adamw(params: Any, lr: float, weight_decay: float = 0.0, device: torch.device | None = None) -> torch.optim.Optimizer:
    if device is not None and device.type == "cuda":
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
        except TypeError:
            pass
        except RuntimeError:
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _maybe_enable_gpu_cache(dataset: Any, device: torch.device, enabled: bool, max_gb: float) -> None:
    if not enabled or device.type != "cuda":
        return
    enable = getattr(dataset, "enable_gpu_cache", None)
    if callable(enable):
        enable(device, max_gb=max_gb)


def _maybe_compile_model(model: nn.Module, device: torch.device, enabled: bool) -> nn.Module:
    if not enabled or device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled")
        return compiled
    except Exception as exc:
        print(f"torch.compile skipped: {exc}")
        return model


def _backward_step(loss: torch.Tensor, opt: torch.optim.Optimizer, scaler: Any | None) -> None:
    opt.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        opt.step()
        return
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


def _loader(
    dataset: Any,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs.update(
            {
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 4,
            }
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=device.type == "cuda",
        **kwargs,
    )


def _progress(iterable: Any, desc: str) -> tqdm:
    return tqdm(iterable, desc=desc, mininterval=10.0, maxinterval=30.0, smoothing=0.05)


def _maybe_set_loss(pbar: tqdm, last_loss: float, last_log_time: float, force: bool = False) -> float:
    now = time.monotonic()
    if force or now - last_log_time >= 10.0:
        pbar.set_postfix(loss=f"{last_loss:.4f}", refresh=True)
        return now
    return last_log_time


def _training_batch(
    dataset: Any,
    batch_size: int,
    iterator: Any,
    loader: DataLoader | None,
    device: torch.device,
) -> tuple[dict[str, Any], Any]:
    sample_batch = getattr(dataset, "sample_batch", None)
    if callable(sample_batch):
        return sample_batch(batch_size, pin_memory=False), iterator
    assert loader is not None
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            acc[k].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if k in {"x", "y"}:
            t = v.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            out[k] = _channels_last_if_cuda(t, device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_baseline(
    model: BaselineCNN,
    dataset: Any,
    audio_cfg: AudioConfig,
    device: torch.device,
    batch_size: int = 8,
    threshold: float = 0.5,
) -> dict[str, float]:
    configure_torch_for_device(device)
    model.eval()
    loader = _loader(dataset, batch_size=batch_size, device=device, shuffle=False)
    rows: list[dict[str, float]] = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        with _autocast(device):
            probs = torch.sigmoid(model(batch["x"])).float().detach().cpu().numpy()
        target = batch["y"].detach().cpu().numpy()
        for pred_i, y_i in zip(probs, target):
            rows.append(
                {
                    **frame_metrics(pred_i, y_i, threshold=threshold),
                    **note_metrics(pred_i, y_i, audio_cfg.frame_hz, threshold=threshold),
                }
            )
    return _mean_metrics(rows)


@torch.no_grad()
def evaluate_baseline_sweep(
    model: BaselineCNN,
    dataset: Any,
    audio_cfg: AudioConfig,
    device: torch.device,
    batch_size: int = 8,
    thresholds: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = evaluate_baseline(model, dataset, audio_cfg, device, batch_size=batch_size, threshold=threshold)
        metrics["threshold"] = threshold
        if best is None or metrics.get("onset_f1", 0.0) > best.get("onset_f1", 0.0):
            best = metrics
    assert best is not None
    return best


def train_baseline(
    train_dataset: Any,
    val_dataset: Any,
    audio_cfg: AudioConfig,
    train_cfg: TrainConfig,
    out_dir: Path,
    device: torch.device | None = None,
) -> tuple[BaselineCNN, dict[str, float]]:
    set_seed(train_cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or device_auto()
    configure_torch_for_device(device)
    _maybe_enable_gpu_cache(train_dataset, device, train_cfg.gpu_cache, train_cfg.gpu_cache_gb)
    model = _model_to_device(BaselineCNN(hidden=train_cfg.hidden), device)
    model_for_train = _maybe_compile_model(model, device, train_cfg.compile_model)
    opt = _make_adamw(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, device=device)
    scaler = _make_scaler(device)
    pos_weight = torch.tensor(train_cfg.positive_weight, device=device)
    use_fast_sampler = callable(getattr(train_dataset, "sample_batch", None))
    loader = None if use_fast_sampler else _loader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        device=device,
        shuffle=True,
        drop_last=True,
        num_workers=train_cfg.num_workers,
    )
    iterator = None if loader is None else iter(loader)
    pbar = _progress(range(train_cfg.steps), desc="train baseline")
    last_loss = 0.0
    last_log_time = 0.0
    for step in pbar:
        batch, iterator = _training_batch(train_dataset, train_cfg.batch_size, iterator, loader, device)
        batch = _batch_to_device(batch, device)
        with _autocast(device):
            logits = model_for_train(batch["x"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"], pos_weight=pos_weight)
        _backward_step(loss, opt, scaler)
        last_loss = float(loss.detach().cpu())
        last_log_time = _maybe_set_loss(pbar, last_loss, last_log_time, force=step == train_cfg.steps - 1)
    metrics = evaluate_baseline_sweep(model, val_dataset, audio_cfg, device, batch_size=min(8, train_cfg.batch_size))
    metrics["train_loss"] = last_loss
    torch.save({"model": model.state_dict(), "train_cfg": train_cfg.__dict__, "metrics": metrics}, out_dir / "baseline.pt")
    return model, metrics


@torch.no_grad()
def evaluate_supervised_refiner(
    baseline: BaselineCNN,
    refiner: SupervisedRefiner,
    dataset: Any,
    audio_cfg: AudioConfig,
    device: torch.device,
    batch_size: int = 8,
    threshold: float = 0.5,
) -> dict[str, float]:
    configure_torch_for_device(device)
    baseline.eval()
    refiner.eval()
    loader = _loader(dataset, batch_size=batch_size, device=device, shuffle=False)
    rows: list[dict[str, float]] = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        with _autocast(device):
            seed = torch.sigmoid(baseline(batch["x"])).float()
            probs = torch.sigmoid(refiner(seed, batch["x"])).float().detach().cpu().numpy()
        target = batch["y"].detach().cpu().numpy()
        for pred_i, y_i in zip(probs, target):
            rows.append(
                {
                    **frame_metrics(pred_i, y_i, threshold=threshold),
                    **note_metrics(pred_i, y_i, audio_cfg.frame_hz, threshold=threshold),
                }
            )
    return _mean_metrics(rows)


@torch.no_grad()
def evaluate_supervised_refiner_sweep(
    baseline: BaselineCNN,
    refiner: SupervisedRefiner,
    dataset: Any,
    audio_cfg: AudioConfig,
    device: torch.device,
    batch_size: int = 8,
    thresholds: tuple[float, ...] = (
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.8,
        0.9,
    ),
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = evaluate_supervised_refiner(
            baseline,
            refiner,
            dataset,
            audio_cfg,
            device,
            batch_size=batch_size,
            threshold=threshold,
        )
        metrics["threshold"] = threshold
        if best is None or metrics.get("onset_f1", 0.0) > best.get("onset_f1", 0.0):
            best = metrics
    assert best is not None
    return best


def train_supervised_refiner(
    baseline: BaselineCNN,
    train_dataset: Any,
    val_dataset: Any,
    audio_cfg: AudioConfig,
    train_cfg: TrainConfig,
    out_dir: Path,
    device: torch.device | None = None,
) -> tuple[SupervisedRefiner, dict[str, float]]:
    set_seed(train_cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or device_auto()
    configure_torch_for_device(device)
    _maybe_enable_gpu_cache(train_dataset, device, train_cfg.gpu_cache, train_cfg.gpu_cache_gb)
    baseline = _model_to_device(baseline, device).eval()
    for param in baseline.parameters():
        param.requires_grad_(False)
    model = _model_to_device(SupervisedRefiner(hidden=train_cfg.hidden), device)
    model_for_train = _maybe_compile_model(model, device, train_cfg.compile_model)
    opt = _make_adamw(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, device=device)
    scaler = _make_scaler(device)
    pos_weight = torch.tensor(train_cfg.positive_weight, device=device)
    use_fast_sampler = callable(getattr(train_dataset, "sample_batch", None))
    loader = None if use_fast_sampler else _loader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        device=device,
        shuffle=True,
        drop_last=True,
        num_workers=train_cfg.num_workers,
    )
    iterator = None if loader is None else iter(loader)
    pbar = _progress(range(train_cfg.steps), desc="train supervised refiner")
    last_loss = 0.0
    last_log_time = 0.0
    for step in pbar:
        batch, iterator = _training_batch(train_dataset, train_cfg.batch_size, iterator, loader, device)
        batch = _batch_to_device(batch, device)
        with torch.no_grad():
            with _autocast(device):
                seed = torch.sigmoid(baseline(batch["x"])).float()
        with _autocast(device):
            logits = model_for_train(seed, batch["x"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"], pos_weight=pos_weight)
        _backward_step(loss, opt, scaler)
        last_loss = float(loss.detach().cpu())
        last_log_time = _maybe_set_loss(pbar, last_loss, last_log_time, force=step == train_cfg.steps - 1)
    metrics = evaluate_supervised_refiner_sweep(
        baseline,
        model,
        val_dataset,
        audio_cfg,
        device,
        batch_size=min(8, train_cfg.batch_size),
    )
    metrics["refiner_train_loss"] = last_loss
    torch.save(
        {
            "model": model.state_dict(),
            "train_cfg": train_cfg.__dict__,
            "metrics": metrics,
            "baseline_seeded": True,
        },
        out_dir / "supervised_refiner.pt",
    )
    return model, metrics


def train_noise_augmented_supervised_refiner(
    baseline: BaselineCNN,
    train_dataset: Any,
    val_dataset: Any,
    audio_cfg: AudioConfig,
    train_cfg: TrainConfig,
    diff_cfg: DiffusionConfig,
    out_dir: Path,
    device: torch.device | None = None,
) -> tuple[SupervisedRefiner, dict[str, float]]:
    """Train the one-shot refiner with diffusion-schedule noise on the baseline seed."""

    set_seed(train_cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or device_auto()
    configure_torch_for_device(device)
    _maybe_enable_gpu_cache(train_dataset, device, train_cfg.gpu_cache, train_cfg.gpu_cache_gb)
    baseline = _model_to_device(baseline, device).eval()
    for param in baseline.parameters():
        param.requires_grad_(False)
    model = _model_to_device(SupervisedRefiner(hidden=train_cfg.hidden), device)
    model_for_train = _maybe_compile_model(model, device, train_cfg.compile_model)
    schedule = DiffusionSchedule(diff_cfg.timesteps).to(device)
    opt = _make_adamw(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, device=device)
    scaler = _make_scaler(device)
    pos_weight = torch.tensor(train_cfg.positive_weight, device=device)
    use_fast_sampler = callable(getattr(train_dataset, "sample_batch", None))
    loader = None if use_fast_sampler else _loader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        device=device,
        shuffle=True,
        drop_last=True,
        num_workers=train_cfg.num_workers,
    )
    iterator = None if loader is None else iter(loader)
    pbar = _progress(range(train_cfg.steps), desc="train noise-aug supervised refiner")
    last_loss = 0.0
    last_log_time = 0.0
    for step in pbar:
        batch, iterator = _training_batch(train_dataset, train_cfg.batch_size, iterator, loader, device)
        batch = _batch_to_device(batch, device)
        with torch.no_grad():
            with _autocast(device):
                seed = torch.sigmoid(baseline(batch["x"])).float()
            seed_x0 = seed.unsqueeze(1).mul(2.0).sub(1.0)
            t = torch.randint(1, schedule.timesteps, (seed_x0.shape[0],), device=device)
            noisy_seed = q_sample(seed_x0, t, schedule, torch.randn_like(seed_x0))
            noisy_seed = _channels_last_if_cuda(noisy_seed.add(1.0).mul(0.5), device)
        with _autocast(device):
            logits = model_for_train(noisy_seed, batch["x"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"], pos_weight=pos_weight)
        _backward_step(loss, opt, scaler)
        last_loss = float(loss.detach().cpu())
        last_log_time = _maybe_set_loss(pbar, last_loss, last_log_time, force=step == train_cfg.steps - 1)
    metrics = evaluate_supervised_refiner_sweep(
        baseline,
        model,
        val_dataset,
        audio_cfg,
        device,
        batch_size=min(8, train_cfg.batch_size),
    )
    metrics["refiner_train_loss"] = last_loss
    torch.save(
        {
            "model": model.state_dict(),
            "train_cfg": train_cfg.__dict__,
            "diff_cfg": diff_cfg.__dict__,
            "metrics": metrics,
            "baseline_seeded": True,
            "train_time_noise": {
                "schedule": "diffusion q_sample",
                "source": "baseline seed",
                "t_low_inclusive": 1,
                "t_high_exclusive": schedule.timesteps,
                "noisy_input_rescaled_to_roll_space": True,
                "timestep_conditioning": False,
                "clean_input_mixture": False,
            },
        },
        out_dir / "noise_aug_supervised_refiner.pt",
    )
    return model, metrics


def save_baseline_predictions(
    model: BaselineCNN,
    dataset: Any,
    audio_cfg: AudioConfig,
    out_path: Path,
    device: torch.device | None = None,
) -> None:
    device = device or device_auto()
    configure_torch_for_device(device)
    model.eval().to(device)
    item = dataset[0]
    x = torch.from_numpy(item["x"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        with _autocast(device):
            pred = torch.sigmoid(model(x))[0].float().detach().cpu().numpy()
    plot_prediction_grid({"ground truth": item["y"], "baseline": pred}, audio_cfg.frame_hz, out_path)


def train_midi_prior(
    train_dataset: Any,
    audio_cfg: AudioConfig,
    diff_cfg: DiffusionConfig,
    out_dir: Path,
    device: torch.device | None = None,
) -> tuple[MidiDenoiser, dict[str, float]]:
    set_seed(diff_cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or device_auto()
    configure_torch_for_device(device)
    _maybe_enable_gpu_cache(train_dataset, device, diff_cfg.gpu_cache, diff_cfg.gpu_cache_gb)
    model = _model_to_device(MidiDenoiser(hidden=diff_cfg.hidden), device)
    model_for_train = _maybe_compile_model(model, device, diff_cfg.compile_model)
    schedule = DiffusionSchedule(diff_cfg.timesteps).to(device)
    opt = _make_adamw(model.parameters(), lr=diff_cfg.lr, device=device)
    scaler = _make_scaler(device)
    use_fast_sampler = callable(getattr(train_dataset, "sample_batch", None))
    loader = None if use_fast_sampler else _loader(
        train_dataset,
        batch_size=diff_cfg.batch_size,
        device=device,
        shuffle=True,
        drop_last=True,
        num_workers=diff_cfg.num_workers,
    )
    iterator = None if loader is None else iter(loader)
    pbar = _progress(range(diff_cfg.steps), desc="train MIDI prior")
    last_loss = 0.0
    last_log_time = 0.0
    for step in pbar:
        batch, iterator = _training_batch(train_dataset, diff_cfg.batch_size, iterator, loader, device)
        y = batch["y"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        x0 = roll_to_diffusion_target(y)
        t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        x_t = _channels_last_if_cuda(q_sample(x0, t, schedule, noise), device)
        with _autocast(device):
            pred = model_for_train(x_t, t)
            if diff_cfg.prediction_type == "epsilon":
                target = noise
            elif diff_cfg.prediction_type == "x0":
                target = x0
            else:
                raise ValueError(f"unknown prediction_type: {diff_cfg.prediction_type}")
            loss = F.mse_loss(pred, target)
        _backward_step(loss, opt, scaler)
        last_loss = float(loss.detach().cpu())
        last_log_time = _maybe_set_loss(pbar, last_loss, last_log_time, force=step == diff_cfg.steps - 1)
    metrics = {"prior_train_loss": last_loss}
    torch.save({"model": model.state_dict(), "diff_cfg": diff_cfg.__dict__, "metrics": metrics}, out_dir / "midi_prior.pt")
    return model, metrics


@torch.no_grad()
def save_prior_samples(
    model: MidiDenoiser,
    audio_cfg: AudioConfig,
    diff_cfg: DiffusionConfig,
    out_path: Path,
    device: torch.device | None = None,
    n: int = 3,
) -> None:
    device = device or device_auto()
    configure_torch_for_device(device)
    schedule = DiffusionSchedule(diff_cfg.timesteps).to(device)
    with _autocast(device):
        samples = sample_ddim(
            model.to(device),
            (n, 1, 88, audio_cfg.frames_per_window),
            schedule,
            diff_cfg.sample_steps,
            device,
            prediction_type=diff_cfg.prediction_type,
    ).float().detach().cpu().numpy()
    panels = {f"sample {i + 1}": samples[i] for i in range(n)}
    plot_prediction_grid(panels, audio_cfg.frame_hz, out_path, title="MIDI prior unconditional samples")


def train_control_branch(
    prior: MidiDenoiser,
    train_dataset: Any,
    val_dataset: Any,
    audio_cfg: AudioConfig,
    diff_cfg: DiffusionConfig,
    out_dir: Path,
    device: torch.device | None = None,
) -> tuple[ControlledDenoiser, dict[str, float]]:
    set_seed(diff_cfg.seed + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or device_auto()
    configure_torch_for_device(device)
    _maybe_enable_gpu_cache(train_dataset, device, diff_cfg.gpu_cache, diff_cfg.gpu_cache_gb)
    prior = _model_to_device(prior, device).eval()
    control = _model_to_device(AudioControlBranch(hidden=diff_cfg.hidden), device)
    model = _model_to_device(ControlledDenoiser(prior, control), device)
    model_for_train = _maybe_compile_model(model, device, diff_cfg.compile_model)
    schedule = DiffusionSchedule(diff_cfg.timesteps).to(device)
    opt = _make_adamw(control.parameters(), lr=diff_cfg.lr, device=device)
    scaler = _make_scaler(device)
    use_fast_sampler = callable(getattr(train_dataset, "sample_batch", None))
    loader = None if use_fast_sampler else _loader(
        train_dataset,
        batch_size=diff_cfg.batch_size,
        device=device,
        shuffle=True,
        drop_last=True,
        num_workers=diff_cfg.num_workers,
    )
    iterator = None if loader is None else iter(loader)
    pbar = _progress(range(diff_cfg.steps), desc="train frozen-prior control")
    last_loss = 0.0
    last_log_time = 0.0
    for step in pbar:
        batch, iterator = _training_batch(train_dataset, diff_cfg.batch_size, iterator, loader, device)
        batch = _batch_to_device(batch, device)
        x0 = roll_to_diffusion_target(batch["y"])
        t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        x_t = _channels_last_if_cuda(q_sample(x0, t, schedule, noise), device)
        with _autocast(device):
            pred = model_for_train(x_t, t, batch["x"])
            if diff_cfg.prediction_type == "epsilon":
                target = noise
            elif diff_cfg.prediction_type == "x0":
                target = x0
            else:
                raise ValueError(f"unknown prediction_type: {diff_cfg.prediction_type}")
            loss = F.mse_loss(pred, target)
        _backward_step(loss, opt, scaler)
        last_loss = float(loss.detach().cpu())
        last_log_time = _maybe_set_loss(pbar, last_loss, last_log_time, force=step == diff_cfg.steps - 1)
    metrics = evaluate_control(model, val_dataset, audio_cfg, diff_cfg, device, batch_size=min(8, diff_cfg.batch_size))
    metrics["control_train_loss"] = last_loss
    torch.save({"control": control.state_dict(), "diff_cfg": diff_cfg.__dict__, "metrics": metrics}, out_dir / "control_branch.pt")
    return model, metrics


@torch.no_grad()
def evaluate_control(
    model: ControlledDenoiser,
    dataset: Any,
    audio_cfg: AudioConfig,
    diff_cfg: DiffusionConfig,
    device: torch.device,
    batch_size: int = 1,
) -> dict[str, float]:
    configure_torch_for_device(device)
    model.eval()
    schedule = DiffusionSchedule(diff_cfg.timesteps).to(device)
    loader = _loader(dataset, batch_size=batch_size, device=device, shuffle=False)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in _progress(loader, desc="eval control"):
        batch = _batch_to_device(batch, device)
        with _autocast(device):
            pred = sample_ddim(
                model,
                (batch["y"].shape[0], 1, 88, audio_cfg.frames_per_window),
                schedule,
                diff_cfg.sample_steps,
                device,
                audio=batch["x"],
                prediction_type=diff_cfg.prediction_type,
            ).float().detach().cpu().numpy()
        preds.extend(list(pred))
        targets.extend(list(batch["y"].detach().cpu().numpy()))
    best: dict[str, float] | None = None
    for threshold in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        rows: list[dict[str, float]] = []
        for pred_i, y_i in zip(preds, targets):
            rows.append(
                {
                    **frame_metrics(pred_i, y_i, threshold=threshold),
                    **note_metrics(pred_i, y_i, audio_cfg.frame_hz, threshold=threshold),
                }
            )
        metrics = _mean_metrics(rows)
        metrics["threshold"] = threshold
        if best is None or metrics.get("onset_f1", 0.0) > best.get("onset_f1", 0.0):
            best = metrics
    assert best is not None
    return best


@torch.no_grad()
def save_control_predictions(
    model: ControlledDenoiser,
    baseline: BaselineCNN | None,
    dataset: Any,
    audio_cfg: AudioConfig,
    diff_cfg: DiffusionConfig,
    out_path: Path,
    device: torch.device | None = None,
) -> None:
    device = device or device_auto()
    configure_torch_for_device(device)
    item = dataset[0]
    audio = torch.from_numpy(item["x"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    schedule = DiffusionSchedule(diff_cfg.timesteps).to(device)
    with _autocast(device):
        control_pred = sample_ddim(
            model.to(device),
            (1, 1, 88, audio_cfg.frames_per_window),
            schedule,
            diff_cfg.sample_steps,
            device,
            audio=audio,
            prediction_type=diff_cfg.prediction_type,
    )[0].float().detach().cpu().numpy()
    panels = {"ground truth": item["y"]}
    if baseline is not None:
        with _autocast(device):
            base_pred = torch.sigmoid(baseline.to(device)(audio))[0].float().detach().cpu().numpy()
        panels["baseline"] = base_pred
    panels["frozen-prior control"] = control_pred
    plot_prediction_grid(panels, audio_cfg.frame_hz, out_path)
