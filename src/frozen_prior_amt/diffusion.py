from __future__ import annotations

import torch
from torch import nn


class DiffusionSchedule:
    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


def roll_to_diffusion_target(y: torch.Tensor) -> torch.Tensor:
    return y.unsqueeze(1) * 2.0 - 1.0


def diffusion_to_roll(x: torch.Tensor) -> torch.Tensor:
    return ((x.squeeze(1) + 1.0) / 2.0).clamp(0.0, 1.0)


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule, noise: torch.Tensor | None = None) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar = schedule.alpha_bars[t].view(-1, 1, 1, 1)
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    shape: tuple[int, int, int, int],
    schedule: DiffusionSchedule,
    sample_steps: int,
    device: torch.device | str,
    audio: torch.Tensor | None = None,
    prediction_type: str = "x0",
) -> torch.Tensor:
    model.eval()
    x = torch.randn(shape, device=device)
    steps = torch.linspace(schedule.timesteps - 1, 0, sample_steps, device=device).long()
    for i, t_scalar in enumerate(steps):
        t = torch.full((shape[0],), int(t_scalar.item()), device=device, dtype=torch.long)
        if audio is None:
            pred = model(x, t)
        else:
            pred = model(x, t, audio)
        alpha_bar = schedule.alpha_bars[t].view(-1, 1, 1, 1)
        if prediction_type == "epsilon":
            eps = pred
            x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        elif prediction_type == "x0":
            x0 = pred.clamp(-1.0, 1.0)
            eps = (x - torch.sqrt(alpha_bar) * x0) / torch.sqrt(1.0 - alpha_bar).clamp_min(1e-6)
        else:
            raise ValueError(f"unknown prediction_type: {prediction_type}")
        x0 = x0.clamp(-1.5, 1.5)
        if i == len(steps) - 1:
            x = x0
            break
        t_prev = int(steps[i + 1].item())
        alpha_bar_prev = schedule.alpha_bars[t_prev].view(1, 1, 1, 1)
        x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps
    return diffusion_to_roll(x)
