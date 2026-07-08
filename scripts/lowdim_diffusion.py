# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small low-dimensional diffusion policy utilities for BallBasket-LowDim-v0.

This module intentionally keeps the model simple: an MLP denoises a short
sequence of future actions conditioned on a short history of low-dimensional
observations. It is a first training scaffold for validating the data pipeline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    """Set common pseudo-random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_device(device: str) -> torch.device:
    """Resolve a user-facing device string."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_normalizer(obs: np.ndarray, actions: np.ndarray, eps: float = 1.0e-6) -> dict[str, torch.Tensor]:
    """Compute mean/std statistics for observations and actions."""
    obs_mean = obs.mean(axis=0)
    obs_std = np.maximum(obs.std(axis=0), eps)
    action_mean = actions.mean(axis=0)
    action_std = np.maximum(actions.std(axis=0), eps)
    return {
        "obs_mean": torch.as_tensor(obs_mean, dtype=torch.float32),
        "obs_std": torch.as_tensor(obs_std, dtype=torch.float32),
        "action_mean": torch.as_tensor(action_mean, dtype=torch.float32),
        "action_std": torch.as_tensor(action_std, dtype=torch.float32),
    }


def normalizer_to_device(normalizer: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Move normalizer arrays/tensors to the target device."""
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in normalizer.items()}


def normalize_obs(obs: torch.Tensor, normalizer: dict[str, torch.Tensor]) -> torch.Tensor:
    """Normalize observations with broadcasting over optional horizon dimensions."""
    return (obs - normalizer["obs_mean"]) / normalizer["obs_std"]


def normalize_actions(actions: torch.Tensor, normalizer: dict[str, torch.Tensor]) -> torch.Tensor:
    """Normalize actions with broadcasting over optional horizon dimensions."""
    return (actions - normalizer["action_mean"]) / normalizer["action_std"]


def unnormalize_actions(actions: torch.Tensor, normalizer: dict[str, torch.Tensor]) -> torch.Tensor:
    """Map normalized actions back to environment action space."""
    return actions * normalizer["action_std"] + normalizer["action_mean"]


class LowDimSequenceDataset(Dataset):
    """Windowed low-dimensional demonstration dataset.

    Each item contains:
    - obs: ``obs_horizon`` observations ending at time t
    - actions: ``pred_horizon`` future actions starting at time t

    Windows are padded by repeating boundary samples inside each episode.
    """

    def __init__(self, dataset_path: str, obs_horizon: int, pred_horizon: int):
        super().__init__()
        if obs_horizon < 1:
            raise ValueError("obs_horizon must be >= 1.")
        if pred_horizon < 1:
            raise ValueError("pred_horizon must be >= 1.")

        with h5py.File(dataset_path, "r") as h5_file:
            self.obs = np.asarray(h5_file["data/obs"], dtype=np.float32)
            self.actions = np.asarray(h5_file["data/actions"], dtype=np.float32)
            self.episode_ends = np.asarray(h5_file["meta/episode_ends"], dtype=np.int64)

        if self.obs.ndim != 2:
            raise ValueError(f"Expected obs to be rank-2, got shape {self.obs.shape}.")
        if self.actions.ndim != 2:
            raise ValueError(f"Expected actions to be rank-2, got shape {self.actions.shape}.")
        if self.obs.shape[0] != self.actions.shape[0]:
            raise ValueError("obs and actions must have the same number of transitions.")
        if self.episode_ends.ndim != 1 or len(self.episode_ends) == 0:
            raise ValueError("meta/episode_ends must be a non-empty vector.")
        if self.episode_ends[-1] != self.obs.shape[0]:
            raise ValueError("The last episode end must equal the number of transitions.")

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.normalizer = compute_normalizer(self.obs, self.actions)
        self.indices: list[tuple[int, int, int]] = []

        episode_start = 0
        for episode_end in self.episode_ends:
            if episode_end <= episode_start:
                raise ValueError("episode_ends must be strictly increasing.")
            for step in range(episode_start, int(episode_end)):
                self.indices.append((episode_start, int(episode_end), step))
            episode_start = int(episode_end)

    @property
    def obs_dim(self) -> int:
        """Observation dimension."""
        return int(self.obs.shape[1])

    @property
    def action_dim(self) -> int:
        """Action dimension."""
        return int(self.actions.shape[1])

    def __len__(self) -> int:
        """Number of training windows."""
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one normalized training window."""
        episode_start, episode_end, step = self.indices[index]
        obs_indices = np.arange(step - self.obs_horizon + 1, step + 1)
        action_indices = np.arange(step, step + self.pred_horizon)
        obs_indices = np.clip(obs_indices, episode_start, episode_end - 1)
        action_indices = np.clip(action_indices, episode_start, episode_end - 1)

        obs = torch.as_tensor(self.obs[obs_indices], dtype=torch.float32)
        actions = torch.as_tensor(self.actions[action_indices], dtype=torch.float32)
        obs = normalize_obs(obs, self.normalizer)
        actions = normalize_actions(actions, self.normalizer)
        return {"obs": obs, "actions": actions}


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed integer diffusion timesteps."""
        device = timesteps.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
        return embeddings


class DenoisingMLP(nn.Module):
    """MLP noise predictor for low-dimensional action sequence diffusion."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.time_embedding = SinusoidalPosEmb(time_embed_dim)

        input_dim = pred_horizon * action_dim + obs_horizon * obs_dim + time_embed_dim
        output_dim = pred_horizon * action_dim

        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Mish())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        """Predict the noise added to an action sequence."""
        batch_size = noisy_actions.shape[0]
        action_flat = noisy_actions.reshape(batch_size, -1)
        obs_flat = obs_cond.reshape(batch_size, -1)
        time_emb = self.time_embedding(timesteps)
        output = self.net(torch.cat((action_flat, obs_flat, time_emb), dim=-1))
        return output.reshape(batch_size, self.pred_horizon, self.action_dim)


@dataclass
class DDPMScheduler:
    """Minimal DDPM scheduler for training and sampling."""

    num_train_timesteps: int = 100
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    device: torch.device | str = torch.device("cpu")

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat((torch.ones(1, device=self.device), self.alphas_cumprod[:-1]))
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, sample_shape: torch.Size) -> torch.Tensor:
        extracted = values.gather(0, timesteps)
        return extracted.reshape((timesteps.shape[0],) + (1,) * (len(sample_shape) - 1))

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0)."""
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, original_samples.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples.shape)
        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """One reverse diffusion step p(x_{t-1} | x_t)."""
        timesteps = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
        beta_t = self._extract(self.betas, timesteps, sample.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, sample.shape)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, timesteps, sample.shape)
        mean = sqrt_recip_alpha * (sample - beta_t * model_output / sqrt_one_minus)
        if timestep == 0:
            return mean
        variance = self._extract(self.posterior_variance, timesteps, sample.shape)
        return mean + torch.sqrt(variance) * torch.randn_like(sample)


@torch.inference_mode()
def sample_action_sequences(
    model: DenoisingMLP,
    scheduler: DDPMScheduler,
    obs_cond: torch.Tensor,
    action_dim: int,
    clip_sample: float | None = 2.0,
) -> torch.Tensor:
    """Sample normalized future action sequences conditioned on normalized observations."""
    model.eval()
    actions = torch.randn(
        (obs_cond.shape[0], model.pred_horizon, action_dim),
        device=obs_cond.device,
        dtype=obs_cond.dtype,
    )
    for timestep in reversed(range(scheduler.num_train_timesteps)):
        timesteps = torch.full((obs_cond.shape[0],), timestep, device=obs_cond.device, dtype=torch.long)
        noise_pred = model(actions, timesteps, obs_cond)
        actions = scheduler.step(noise_pred, timestep, actions)
        if clip_sample is not None:
            actions = torch.clamp(actions, -clip_sample, clip_sample)
    return actions


def checkpoint_config_from_args(args: Any, obs_dim: int, action_dim: int) -> dict[str, Any]:
    """Build the serializable training config stored in checkpoints."""
    return {
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "obs_horizon": int(args.obs_horizon),
        "pred_horizon": int(args.pred_horizon),
        "action_horizon": int(args.action_horizon),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "time_embed_dim": int(args.time_embed_dim),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "beta_start": float(args.beta_start),
        "beta_end": float(args.beta_end),
    }


def build_model_from_config(config: dict[str, Any]) -> DenoisingMLP:
    """Instantiate a denoising model from checkpoint config."""
    return DenoisingMLP(
        obs_dim=int(config["obs_dim"]),
        action_dim=int(config["action_dim"]),
        obs_horizon=int(config["obs_horizon"]),
        pred_horizon=int(config["pred_horizon"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        time_embed_dim=int(config["time_embed_dim"]),
    )
