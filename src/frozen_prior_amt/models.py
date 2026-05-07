from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import N_KEYS


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class BaselineCNN(nn.Module):
    def __init__(self, in_channels: int = 2, hidden: int = 48, n_keys: int = N_KEYS) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.SiLU(),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, n_keys, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.mean(dim=2)
        return self.head(h)


class TimeBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        groups = max(1, min(8, channels // 8))
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class MidiDenoiser(nn.Module):
    def __init__(self, hidden: int = 48, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([TimeBlock(hidden, time_dim) for _ in range(6)])
        self.out = nn.Sequential(
            nn.GroupNorm(max(1, min(8, hidden // 8)), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))
        h = self.in_conv(x_t)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out(h)


class AudioControlBranch(nn.Module):
    def __init__(self, audio_channels: int = 2, hidden: int = 48, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(audio_channels, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.noisy_in = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([TimeBlock(hidden, time_dim) for _ in range(4)])
        self.zero = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.zero.weight)
        nn.init.zeros_(self.zero.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))
        a = self.audio_encoder(audio)
        a = F.interpolate(a, size=x_t.shape[-2:], mode="bilinear", align_corners=False)
        h = self.noisy_in(x_t) + a
        for block in self.blocks:
            h = block(h, t_emb)
        return self.zero(h)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = max(1, min(8, channels // 8))
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class SupervisedRefiner(nn.Module):
    """One-shot baseline-seeded refiner with the ControlNet branch capacity."""

    def __init__(self, audio_channels: int = 2, hidden: int = 48, n_keys: int = N_KEYS) -> None:
        super().__init__()
        self.n_keys = n_keys
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(audio_channels, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.seed_in = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualConvBlock(hidden) for _ in range(4)])
        self.out = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)

    def forward(self, baseline_roll: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if baseline_roll.ndim == 3:
            baseline_roll = baseline_roll.unsqueeze(1)
        a = self.audio_encoder(audio)
        a = F.interpolate(a, size=baseline_roll.shape[-2:], mode="bilinear", align_corners=False)
        h = self.seed_in(baseline_roll) + a
        for block in self.blocks:
            h = block(h)
        return self.out(h).squeeze(1)


class ControlledDenoiser(nn.Module):
    def __init__(self, prior: MidiDenoiser, control: AudioControlBranch) -> None:
        super().__init__()
        self.prior = prior
        self.control = control
        for param in self.prior.parameters():
            param.requires_grad_(False)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.prior(x_t, t)
        return base + self.control(x_t, t, audio)
