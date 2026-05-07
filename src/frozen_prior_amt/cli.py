from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .config import AudioConfig, DiffusionConfig, Paths, TrainConfig
from .dataset import FixedWindowDataset, OneWindowDataset, RandomWindowDataset, load_cached_pieces
from .features import cache_feature_set, load_features
from .maestro import (
    cache_rolls,
    download_maestro_files,
    fetch_metadata,
    first_note_window_start,
    load_roll,
    save_run_manifest,
    select_subset,
    write_subset,
)
from .models import BaselineCNN, MidiDenoiser
from .plotting import plot_alignment, plot_piano_roll
from .training import (
    accelerator_summary,
    device_auto,
    save_baseline_predictions,
    save_control_predictions,
    save_prior_samples,
    train_baseline,
    train_control_branch,
    train_midi_prior,
    train_noise_augmented_supervised_refiner,
    train_supervised_refiner,
)
from .utils import append_tsv, now_id


def _paths(args: argparse.Namespace) -> Paths:
    paths = Paths.from_root(args.root)
    paths.ensure()
    return paths


def _audio_cfg(args: argparse.Namespace) -> AudioConfig:
    return AudioConfig(
        sample_rate=args.sample_rate,
        frame_hz=args.frame_hz,
        window_seconds=args.window_seconds,
    )


def _subset_path(paths: Paths, name: str) -> Path:
    return paths.cache / f"{name}.csv"


def _ensure_subset(paths: Paths, name: str, pieces_per_split: int) -> pd.DataFrame:
    path = _subset_path(paths, name)
    if path.exists():
        return pd.read_csv(path)
    metadata = fetch_metadata(paths)
    subset = select_subset(metadata, pieces_per_split=pieces_per_split)
    write_subset(paths, subset, name=name)
    return subset


def cmd_data_sanity(args: argparse.Namespace) -> None:
    paths = _paths(args)
    audio_cfg = _audio_cfg(args)
    metadata = fetch_metadata(paths, refresh=args.refresh)
    subset = select_subset(metadata, pieces_per_split=args.pieces_per_split)
    subset_path = write_subset(paths, subset, name=args.subset_name)
    print(f"wrote subset: {subset_path}")
    print(subset[["split", "duration", "canonical_composer", "canonical_title"]].to_string(index=False))
    download_maestro_files(paths, subset, include_audio=False, force=args.force_download)
    roll_paths = cache_rolls(paths, subset, audio_cfg, force=args.force_cache)
    for _, row in subset.head(args.max_plots).iterrows():
        roll = load_roll(paths, row.piece_id)
        out = paths.plots / "rolls" / f"{row.piece_id}.png"
        plot_piano_roll(
            roll["active"][:, : int(args.plot_seconds * audio_cfg.frame_hz)],
            audio_cfg.frame_hz,
            out,
            title=f"{row.split}: {row.canonical_composer} - {row.canonical_title}",
        )
        print(f"wrote piano-roll plot: {out}")
    print(f"cached {len(roll_paths)} piano rolls")


def cmd_audio_alignment(args: argparse.Namespace) -> None:
    paths = _paths(args)
    audio_cfg = _audio_cfg(args)
    subset = _ensure_subset(paths, args.subset_name, args.pieces_per_split)
    rows = subset.head(args.max_pieces).copy()
    download_maestro_files(paths, rows, include_audio=True, force=args.force_download)
    cache_rolls(paths, rows, audio_cfg, force=args.force_cache)
    cache_feature_set(paths, rows, audio_cfg, force=args.force_cache)
    for _, row in rows.iterrows():
        roll = load_roll(paths, row.piece_id)
        feats = load_features(paths, row.piece_id)
        out = paths.plots / "alignment" / f"{row.piece_id}.png"
        plot_alignment(
            feats["logmel"],
            roll["onset"],
            audio_cfg.frame_hz,
            out,
            title=f"{row.split}: {row.canonical_composer} - {row.canonical_title}",
        )
        print(f"wrote alignment plot: {out}")


def cmd_overfit_one(args: argparse.Namespace) -> None:
    paths = _paths(args)
    audio_cfg = _audio_cfg(args)
    subset = _ensure_subset(paths, args.subset_name, args.pieces_per_split)
    row = subset[subset["split"] == "train"].head(1)
    if row.empty:
        raise SystemExit("subset has no train piece")
    download_maestro_files(paths, row, include_audio=True, force=args.force_download)
    cache_rolls(paths, row, audio_cfg, force=args.force_cache)
    cache_feature_set(paths, row, audio_cfg, force=args.force_cache)
    piece = load_cached_pieces(paths, row, include_audio=True)[0]
    start = first_note_window_start(piece.active, audio_cfg.frame_hz, audio_cfg.frames_per_window)
    dataset = OneWindowDataset(piece, audio_cfg, include_audio=True, start_frame=start, repeats=max(args.steps * args.batch_size, 256))
    run_id = now_id("overfit-one")
    out_dir = paths.runs / run_id
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        num_workers=args.num_workers,
        positive_weight=args.positive_weight,
        gpu_cache=args.gpu_cache,
        gpu_cache_gb=args.gpu_cache_gb,
        compile_model=args.compile,
    )
    model, metrics = train_baseline(dataset, dataset, audio_cfg, train_cfg, out_dir)
    plot_path = paths.plots / "predictions" / f"{run_id}.png"
    save_baseline_predictions(model, dataset, audio_cfg, plot_path)
    save_run_manifest(out_dir / "manifest.json", row, audio_cfg, {"run_id": run_id, "command": "overfit-one"})
    append_tsv(paths.runs / "results.tsv", {"run_id": run_id, "stage": "overfit-one", **metrics})
    print(f"accelerator: {accelerator_summary(device_auto())}")
    print(f"metrics: {metrics}")
    print(f"wrote prediction plot: {plot_path}")
    print(f"wrote checkpoint: {out_dir / 'baseline.pt'}")


def cmd_train_baseline(args: argparse.Namespace) -> None:
    paths = _paths(args)
    audio_cfg = _audio_cfg(args)
    subset = _ensure_subset(paths, args.subset_name, args.pieces_per_split)
    train_rows = subset[subset["split"] == "train"]
    val_rows = subset[subset["split"] == "validation"]
    needed_rows = pd.concat([train_rows, val_rows], ignore_index=True)
    download_maestro_files(paths, needed_rows, include_audio=True, force=args.force_download)
    cache_rolls(paths, needed_rows, audio_cfg, force=args.force_cache)
    cache_feature_set(paths, needed_rows, audio_cfg, force=args.force_cache)
    train_pieces = load_cached_pieces(paths, train_rows, include_audio=True)
    val_pieces = load_cached_pieces(paths, val_rows, include_audio=True)
    train_ds = RandomWindowDataset(
        train_pieces,
        audio_cfg,
        num_samples=args.steps * args.batch_size,
        include_audio=True,
        seed=args.seed,
    )
    val_ds = FixedWindowDataset(val_pieces, audio_cfg, include_audio=True, windows_per_piece=args.val_windows)
    run_id = now_id("baseline")
    out_dir = paths.runs / run_id
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        positive_weight=args.positive_weight,
        num_workers=args.num_workers,
        seed=args.seed,
        gpu_cache=args.gpu_cache,
        gpu_cache_gb=args.gpu_cache_gb,
        compile_model=args.compile,
    )
    model, metrics = train_baseline(train_ds, val_ds, audio_cfg, train_cfg, out_dir)
    plot_path = paths.plots / "predictions" / f"{run_id}.png"
    save_baseline_predictions(model, val_ds, audio_cfg, plot_path)
    save_run_manifest(out_dir / "manifest.json", subset, audio_cfg, {"run_id": run_id, "command": "train-baseline"})
    append_tsv(paths.runs / "results.tsv", {"run_id": run_id, "stage": "baseline", **metrics})
    print(f"accelerator: {accelerator_summary(device_auto())}")
    print(f"metrics: {metrics}")
    print(f"wrote prediction plot: {plot_path}")
    print(f"wrote checkpoint: {out_dir / 'baseline.pt'}")


def cmd_train_prior(args: argparse.Namespace) -> None:
    paths = _paths(args)
    audio_cfg = _audio_cfg(args)
    subset = _ensure_subset(paths, args.subset_name, args.pieces_per_split)
    train_rows = subset[subset["split"] == "train"]
    download_maestro_files(paths, train_rows, include_audio=False, force=args.force_download)
    cache_rolls(paths, train_rows, audio_cfg, force=args.force_cache)
    train_pieces = load_cached_pieces(paths, train_rows, include_audio=False)
    train_ds = RandomWindowDataset(
        train_pieces,
        audio_cfg,
        num_samples=args.steps * args.batch_size,
        include_audio=False,
        seed=args.seed,
    )
    run_id = now_id("midi-prior")
    out_dir = paths.runs / run_id
    diff_cfg = DiffusionConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        timesteps=args.timesteps,
        sample_steps=args.sample_steps,
        prediction_type=args.prediction_type,
        num_workers=args.num_workers,
        seed=args.seed,
        gpu_cache=args.gpu_cache,
        gpu_cache_gb=args.gpu_cache_gb,
        compile_model=args.compile,
    )
    model, metrics = train_midi_prior(train_ds, audio_cfg, diff_cfg, out_dir)
    plot_path = paths.plots / "prior_samples" / f"{run_id}.png"
    save_prior_samples(model, audio_cfg, diff_cfg, plot_path)
    save_run_manifest(out_dir / "manifest.json", subset, audio_cfg, {"run_id": run_id, "command": "train-prior"})
    append_tsv(paths.runs / "results.tsv", {"run_id": run_id, "stage": "midi-prior", **metrics})
    print(f"accelerator: {accelerator_summary(device_auto())}")
    print(f"metrics: {metrics}")
    print(f"wrote prior sample plot: {plot_path}")
    print(f"wrote checkpoint: {out_dir / 'midi_prior.pt'}")


def _load_prior(path: Path, hidden: int) -> MidiDenoiser:
    ckpt = torch.load(path, map_location="cpu")
    model = MidiDenoiser(hidden=hidden)
    model.load_state_dict(ckpt["model"])
    return model


def _load_baseline(path: Path, hidden: int) -> BaselineCNN:
    ckpt = torch.load(path, map_location="cpu")
    model = BaselineCNN(hidden=hidden)
    model.load_state_dict(ckpt["model"])
    return model


def cmd_train_control(args: argparse.Namespace) -> None:
    paths = _paths(args)
    audio_cfg = _audio_cfg(args)
    subset = _ensure_subset(paths, args.subset_name, args.pieces_per_split)
    train_rows = subset[subset["split"] == "train"]
    val_rows = subset[subset["split"] == "validation"]
    needed_rows = pd.concat([train_rows, val_rows], ignore_index=True)
    download_maestro_files(paths, needed_rows, include_audio=True, force=args.force_download)
    cache_rolls(paths, needed_rows, audio_cfg, force=args.force_cache)
    cache_feature_set(paths, needed_rows, audio_cfg, force=args.force_cache)
    train_pieces = load_cached_pieces(paths, train_rows, include_audio=True)
    val_pieces = load_cached_pieces(paths, val_rows, include_audio=True)
    train_ds = RandomWindowDataset(
        train_pieces,
        audio_cfg,
        num_samples=args.steps * args.batch_size,
        include_audio=True,
        seed=args.seed,
    )
    val_ds = FixedWindowDataset(val_pieces, audio_cfg, include_audio=True, windows_per_piece=args.val_windows)
    diff_cfg = DiffusionConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        timesteps=args.timesteps,
        sample_steps=args.sample_steps,
        prediction_type=args.prediction_type,
        num_workers=args.num_workers,
        seed=args.seed,
        gpu_cache=args.gpu_cache,
        gpu_cache_gb=args.gpu_cache_gb,
        compile_model=args.compile,
    )
    prior = _load_prior(Path(args.prior_checkpoint), hidden=args.hidden)
    baseline = _load_baseline(Path(args.baseline_checkpoint), hidden=args.baseline_hidden) if args.baseline_checkpoint else None
    run_id = now_id("control")
    out_dir = paths.runs / run_id
    model, metrics = train_control_branch(prior, train_ds, val_ds, audio_cfg, diff_cfg, out_dir)
    plot_path = paths.plots / "predictions" / f"{run_id}.png"
    save_control_predictions(model, baseline, val_ds, audio_cfg, diff_cfg, plot_path)
    save_run_manifest(out_dir / "manifest.json", subset, audio_cfg, {"run_id": run_id, "command": "train-control"})
    append_tsv(paths.runs / "results.tsv", {"run_id": run_id, "stage": "control", **metrics})
    print(f"accelerator: {accelerator_summary(device_auto())}")
    print(f"metrics: {metrics}")
    print(f"wrote prediction plot: {plot_path}")
    print(f"wrote checkpoint: {out_dir / 'control_branch.pt'}")


def cmd_train_supervised_refiner(args: argparse.Namespace) -> None:
    paths = _paths(args)
    audio_cfg = _audio_cfg(args)
    subset = _ensure_subset(paths, args.subset_name, args.pieces_per_split)
    train_rows = subset[subset["split"] == "train"]
    val_rows = subset[subset["split"] == "validation"]
    needed_rows = pd.concat([train_rows, val_rows], ignore_index=True)
    download_maestro_files(paths, needed_rows, include_audio=True, force=args.force_download)
    cache_rolls(paths, needed_rows, audio_cfg, force=args.force_cache)
    cache_feature_set(paths, needed_rows, audio_cfg, force=args.force_cache)
    train_pieces = load_cached_pieces(paths, train_rows, include_audio=True)
    val_pieces = load_cached_pieces(paths, val_rows, include_audio=True)
    train_ds = RandomWindowDataset(
        train_pieces,
        audio_cfg,
        num_samples=args.steps * args.batch_size,
        include_audio=True,
        seed=args.seed,
    )
    val_ds = FixedWindowDataset(val_pieces, audio_cfg, include_audio=True, windows_per_piece=args.val_windows)
    baseline = _load_baseline(Path(args.baseline_checkpoint), hidden=args.baseline_hidden)
    run_id = now_id("supervised-refiner")
    out_dir = paths.runs / run_id
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        positive_weight=args.positive_weight,
        num_workers=args.num_workers,
        seed=args.seed,
        gpu_cache=args.gpu_cache,
        gpu_cache_gb=args.gpu_cache_gb,
        compile_model=args.compile,
    )
    _, metrics = train_supervised_refiner(baseline, train_ds, val_ds, audio_cfg, train_cfg, out_dir)
    save_run_manifest(out_dir / "manifest.json", subset, audio_cfg, {"run_id": run_id, "command": "train-supervised-refiner"})
    append_tsv(paths.runs / "results.tsv", {"run_id": run_id, "stage": "supervised-refiner", **metrics})
    print(f"accelerator: {accelerator_summary(device_auto())}")
    print(f"metrics: {metrics}")
    print(f"wrote checkpoint: {out_dir / 'supervised_refiner.pt'}")


def cmd_train_noise_augmented_supervised_refiner(args: argparse.Namespace) -> None:
    paths = _paths(args)
    audio_cfg = _audio_cfg(args)
    subset = _ensure_subset(paths, args.subset_name, args.pieces_per_split)
    train_rows = subset[subset["split"] == "train"]
    val_rows = subset[subset["split"] == "validation"]
    needed_rows = pd.concat([train_rows, val_rows], ignore_index=True)
    download_maestro_files(paths, needed_rows, include_audio=True, force=args.force_download)
    cache_rolls(paths, needed_rows, audio_cfg, force=args.force_cache)
    cache_feature_set(paths, needed_rows, audio_cfg, force=args.force_cache)
    train_pieces = load_cached_pieces(paths, train_rows, include_audio=True)
    val_pieces = load_cached_pieces(paths, val_rows, include_audio=True)
    train_ds = RandomWindowDataset(
        train_pieces,
        audio_cfg,
        num_samples=args.steps * args.batch_size,
        include_audio=True,
        seed=args.seed,
    )
    val_ds = FixedWindowDataset(val_pieces, audio_cfg, include_audio=True, windows_per_piece=args.val_windows)
    baseline = _load_baseline(Path(args.baseline_checkpoint), hidden=args.baseline_hidden)
    run_id = now_id("noise-aug-supervised-refiner")
    out_dir = paths.runs / run_id
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        positive_weight=args.positive_weight,
        num_workers=args.num_workers,
        seed=args.seed,
        gpu_cache=args.gpu_cache,
        gpu_cache_gb=args.gpu_cache_gb,
        compile_model=args.compile,
    )
    diff_cfg = DiffusionConfig(
        timesteps=args.timesteps,
        prediction_type=args.prediction_type,
        hidden=args.hidden,
        seed=args.seed,
    )
    _, metrics = train_noise_augmented_supervised_refiner(
        baseline,
        train_ds,
        val_ds,
        audio_cfg,
        train_cfg,
        diff_cfg,
        out_dir,
    )
    save_run_manifest(
        out_dir / "manifest.json",
        subset,
        audio_cfg,
        {"run_id": run_id, "command": "train-noise-aug-supervised-refiner"},
    )
    append_tsv(paths.runs / "results.tsv", {"run_id": run_id, "stage": "noise-aug-supervised-refiner", **metrics})
    print(f"accelerator: {accelerator_summary(device_auto())}")
    print(f"metrics: {metrics}")
    print(f"wrote checkpoint: {out_dir / 'noise_aug_supervised_refiner.pt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fpamt")
    parser.add_argument("--root", default=".", help="project root")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--frame-hz", type=int, default=50)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    sub = parser.add_subparsers(dest="command", required=True)

    def common_data(p: argparse.ArgumentParser) -> None:
        p.add_argument("--subset-name", default="tiny_subset")
        p.add_argument("--pieces-per-split", type=int, default=2)
        p.add_argument("--force-download", action="store_true")
        p.add_argument("--force-cache", action="store_true")

    p = sub.add_parser("data-sanity")
    common_data(p)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--max-plots", type=int, default=6)
    p.add_argument("--plot-seconds", type=float, default=20.0)
    p.set_defaults(func=cmd_data_sanity)

    p = sub.add_parser("audio-alignment")
    common_data(p)
    p.add_argument("--max-pieces", type=int, default=1)
    p.set_defaults(func=cmd_audio_alignment)

    def train_common(p: argparse.ArgumentParser) -> None:
        common_data(p)
        p.add_argument("--steps", type=int, default=500)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--lr", type=float, default=2e-3)
        p.add_argument("--hidden", type=int, default=48)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--num-workers", type=int, default=0)
        p.add_argument("--gpu-cache", action="store_true", help="pad selected training pieces onto CUDA and sample windows on GPU")
        p.add_argument("--gpu-cache-gb", type=float, default=24.0, help="skip GPU cache if estimated training cache exceeds this many GiB")
        p.add_argument("--compile", action="store_true", help="try torch.compile for the training forward pass")

    p = sub.add_parser("overfit-one")
    train_common(p)
    p.add_argument("--positive-weight", type=float, default=8.0)
    p.set_defaults(func=cmd_overfit_one)

    p = sub.add_parser("train-baseline")
    train_common(p)
    p.add_argument("--positive-weight", type=float, default=8.0)
    p.add_argument("--val-windows", type=int, default=3)
    p.set_defaults(func=cmd_train_baseline)

    def diffusion_common(p: argparse.ArgumentParser) -> None:
        common_data(p)
        p.add_argument("--steps", type=int, default=1000)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--hidden", type=int, default=48)
        p.add_argument("--timesteps", type=int, default=100)
        p.add_argument("--sample-steps", type=int, default=24)
        p.add_argument("--prediction-type", choices=["x0", "epsilon"], default="x0")
        p.add_argument("--seed", type=int, default=11)
        p.add_argument("--num-workers", type=int, default=0)
        p.add_argument("--gpu-cache", action="store_true", help="pad selected training pieces onto CUDA and sample windows on GPU")
        p.add_argument("--gpu-cache-gb", type=float, default=24.0, help="skip GPU cache if estimated training cache exceeds this many GiB")
        p.add_argument("--compile", action="store_true", help="try torch.compile for the training forward pass")

    p = sub.add_parser("train-prior")
    diffusion_common(p)
    p.set_defaults(func=cmd_train_prior)

    p = sub.add_parser("train-control")
    diffusion_common(p)
    p.add_argument("--prior-checkpoint", required=True)
    p.add_argument("--baseline-checkpoint")
    p.add_argument("--baseline-hidden", type=int, default=48)
    p.add_argument("--val-windows", type=int, default=3)
    p.set_defaults(func=cmd_train_control)

    p = sub.add_parser("train-supervised-refiner")
    train_common(p)
    p.set_defaults(lr=2e-4)
    p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--baseline-hidden", type=int, default=48)
    p.add_argument("--positive-weight", type=float, default=4.0)
    p.add_argument("--val-windows", type=int, default=3)
    p.set_defaults(func=cmd_train_supervised_refiner)

    p = sub.add_parser("train-noise-aug-supervised-refiner")
    train_common(p)
    p.set_defaults(lr=2e-4)
    p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--baseline-hidden", type=int, default=48)
    p.add_argument("--positive-weight", type=float, default=4.0)
    p.add_argument("--val-windows", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=120)
    p.add_argument("--prediction-type", choices=["x0", "epsilon"], default="x0")
    p.set_defaults(func=cmd_train_noise_augmented_supervised_refiner)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
