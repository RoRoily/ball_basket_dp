# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual diffusion policy utilities for BallBasket-LowDim-v0.

This module keeps the first visual policy intentionally small: a compact CNN
encodes rendered RGB frames, then an MLP denoises a future action sequence
conditioned on the image history and optional low-dimensional state history.
"""

from __future__ import annotations

from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from lowdim_diffusion import (
    DDPMScheduler,
    SinusoidalPosEmb,
    compute_normalizer,
    normalize_actions,
    normalize_obs,
    unnormalize_actions,
)


def resize_rgb_image(image: np.ndarray, image_size: int) -> np.ndarray:
    """Resize an RGB image to a square uint8 image."""
    from PIL import Image

    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected an RGB image with rank 3, got shape {image.shape}.")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image with 3 channels, got shape {image.shape}.")
    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[0] == image_size and image.shape[1] == image_size:
        return image
    pil_image = Image.fromarray(image, mode="RGB")
    return np.asarray(pil_image.resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.uint8)


def render_rgb_frame(env, image_size: int) -> np.ndarray:
    """Render one RGB frame from a Gym/Isaac environment."""
    frame = env.render()
    if isinstance(frame, tuple):
        frame = frame[0]
    if isinstance(frame, list):
        frame = frame[0]
    return resize_rgb_image(np.asarray(frame), image_size)


class VisualSequenceDataset(Dataset):
    """Windowed RGB + low-dimensional demonstration dataset.

    Each item contains an image history, an observation history, and a future
    action sequence. Windows are padded by repeating boundary samples inside
    each episode, matching the low-dimensional dataset behavior.
    """

    def __init__(
        self,
        dataset_path: str,
        obs_horizon: int,
        pred_horizon: int,
        episode_indices: Sequence[int] | None = None,
        normalizer: dict[str, Any] | None = None,
        cache_images: bool = False,
    ):
        super().__init__()
        if obs_horizon < 1:
            raise ValueError("obs_horizon must be >= 1.")
        if pred_horizon < 1:
            raise ValueError("pred_horizon must be >= 1.")

        self.dataset_path = dataset_path
        self.cache_images = cache_images
        self._h5_file: h5py.File | None = None
        self._images_dataset = None

        with h5py.File(dataset_path, "r") as h5_file:
            if "data/images" not in h5_file:
                raise ValueError("Visual dataset must contain data/images.")
            self.obs = np.asarray(h5_file["data/obs"], dtype=np.float32)
            self.actions = np.asarray(h5_file["data/actions"], dtype=np.float32)
            self.episode_ends = np.asarray(h5_file["meta/episode_ends"], dtype=np.int64)
            self.image_shape = tuple(int(value) for value in h5_file["data/images"].shape[1:])
            self.images = np.asarray(h5_file["data/images"], dtype=np.uint8) if cache_images else None

        if len(self.image_shape) != 3 or self.image_shape[-1] != 3:
            raise ValueError(f"Expected images shaped (N, H, W, 3), got per-frame shape {self.image_shape}.")
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
        if self.images is not None and self.images.shape[0] != self.obs.shape[0]:
            raise ValueError("images and obs must have the same number of transitions.")

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.episode_ranges: list[tuple[int, int]] = []
        episode_start = 0
        for episode_end in self.episode_ends:
            if episode_end <= episode_start:
                raise ValueError("episode_ends must be strictly increasing.")
            self.episode_ranges.append((episode_start, int(episode_end)))
            episode_start = int(episode_end)

        if episode_indices is None:
            self.episode_indices = list(range(len(self.episode_ranges)))
        else:
            self.episode_indices = [int(index) for index in episode_indices]
            if len(self.episode_indices) == 0:
                raise ValueError("episode_indices must not be empty.")
            invalid_indices = [
                index for index in self.episode_indices if index < 0 or index >= len(self.episode_ranges)
            ]
            if invalid_indices:
                raise ValueError(f"Invalid episode indices: {invalid_indices}.")

        self.indices: list[tuple[int, int, int]] = []
        selected_transition_indices = []
        for episode_index in self.episode_indices:
            episode_start, episode_end = self.episode_ranges[episode_index]
            selected_transition_indices.extend(range(episode_start, episode_end))
            for step in range(episode_start, episode_end):
                self.indices.append((episode_start, int(episode_end), step))

        if normalizer is None:
            selected_transition_indices = np.asarray(selected_transition_indices, dtype=np.int64)
            self.normalizer = compute_normalizer(
                self.obs[selected_transition_indices], self.actions[selected_transition_indices]
            )
        else:
            self.normalizer = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in normalizer.items()}

    def __getstate__(self) -> dict[str, Any]:
        """Drop open HDF5 handles when DataLoader workers pickle the dataset."""
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_images_dataset"] = None
        return state

    def close(self) -> None:
        """Close the lazy HDF5 handle if it is open."""
        if self._h5_file is not None:
            self._h5_file.close()
        self._h5_file = None
        self._images_dataset = None

    def _image_dataset(self):
        if self.images is not None:
            return None
        if self._h5_file is None:
            self._h5_file = h5py.File(self.dataset_path, "r")
            self._images_dataset = self._h5_file["data/images"]
        return self._images_dataset

    def _load_images(self, indices: np.ndarray) -> np.ndarray:
        if self.images is not None:
            return self.images[indices]
        image_dataset = self._image_dataset()
        return np.stack([np.asarray(image_dataset[int(index)], dtype=np.uint8) for index in indices], axis=0)

    @property
    def num_episodes(self) -> int:
        """Number of episodes in the source dataset."""
        return len(self.episode_ranges)

    @property
    def num_selected_episodes(self) -> int:
        """Number of episodes used by this dataset view."""
        return len(self.episode_indices)

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
        """Return one normalized visual training window."""
        episode_start, episode_end, step = self.indices[index]
        obs_indices = np.arange(step - self.obs_horizon + 1, step + 1)
        action_indices = np.arange(step, step + self.pred_horizon)
        obs_indices = np.clip(obs_indices, episode_start, episode_end - 1)
        action_indices = np.clip(action_indices, episode_start, episode_end - 1)

        images_np = self._load_images(obs_indices)
        images = torch.as_tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        obs = torch.as_tensor(self.obs[obs_indices], dtype=torch.float32)
        actions = torch.as_tensor(self.actions[action_indices], dtype=torch.float32)
        obs = normalize_obs(obs, self.normalizer)
        actions = normalize_actions(actions, self.normalizer)
        return {"images": images, "obs": obs, "actions": actions}


class VisualEncoder(nn.Module):
    """Small convolutional image encoder."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.Mish(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images shaped (B, 3, H, W)."""
        return self.net(images)


class VisualDenoisingMLP(nn.Module):
    """MLP noise predictor conditioned on RGB image and optional lowdim history."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        image_feature_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 4,
        time_embed_dim: int = 64,
        use_lowdim_obs: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.image_feature_dim = image_feature_dim
        self.use_lowdim_obs = use_lowdim_obs
        self.image_encoder = VisualEncoder(image_feature_dim)
        self.time_embedding = SinusoidalPosEmb(time_embed_dim)

        obs_condition_dim = obs_horizon * obs_dim if use_lowdim_obs else 0
        input_dim = (
            pred_horizon * action_dim
            + obs_horizon * image_feature_dim
            + obs_condition_dim
            + time_embed_dim
        )
        output_dim = pred_horizon * action_dim

        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Mish())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        image_cond: torch.Tensor,
        obs_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise added to an action sequence."""
        batch_size = noisy_actions.shape[0]
        image_batch = image_cond.reshape(batch_size * self.obs_horizon, *image_cond.shape[2:])
        image_features = self.image_encoder(image_batch).reshape(batch_size, self.obs_horizon, -1)
        pieces = [
            noisy_actions.reshape(batch_size, -1),
            image_features.reshape(batch_size, -1),
            self.time_embedding(timesteps),
        ]
        if self.use_lowdim_obs:
            if obs_cond is None:
                raise ValueError("obs_cond is required when use_lowdim_obs=True.")
            pieces.insert(2, obs_cond.reshape(batch_size, -1))
        output = self.net(torch.cat(pieces, dim=-1))
        return output.reshape(batch_size, self.pred_horizon, self.action_dim)


@torch.inference_mode()
def sample_visual_action_sequences(
    model: VisualDenoisingMLP,
    scheduler: DDPMScheduler,
    image_cond: torch.Tensor,
    obs_cond: torch.Tensor | None,
    action_dim: int,
    clip_sample: float | None = 2.0,
) -> torch.Tensor:
    """Sample normalized future action sequences from a visual diffusion model."""
    model.eval()
    actions = torch.randn(
        (image_cond.shape[0], model.pred_horizon, action_dim),
        device=image_cond.device,
        dtype=image_cond.dtype,
    )
    for timestep in reversed(range(scheduler.num_train_timesteps)):
        timesteps = torch.full((image_cond.shape[0],), timestep, device=image_cond.device, dtype=torch.long)
        noise_pred = model(actions, timesteps, image_cond, obs_cond)
        actions = scheduler.step(noise_pred, timestep, actions)
        if clip_sample is not None:
            actions = torch.clamp(actions, -clip_sample, clip_sample)
    return actions


def visual_checkpoint_config_from_args(args: Any, dataset: VisualSequenceDataset) -> dict[str, Any]:
    """Build the serializable config stored in visual policy checkpoints."""
    image_height, image_width, image_channels = dataset.image_shape
    return {
        "obs_dim": int(dataset.obs_dim),
        "action_dim": int(dataset.action_dim),
        "obs_horizon": int(args.obs_horizon),
        "pred_horizon": int(args.pred_horizon),
        "action_horizon": int(args.action_horizon),
        "image_height": int(image_height),
        "image_width": int(image_width),
        "image_channels": int(image_channels),
        "image_feature_dim": int(args.image_feature_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "time_embed_dim": int(args.time_embed_dim),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "beta_start": float(args.beta_start),
        "beta_end": float(args.beta_end),
        "val_ratio": float(getattr(args, "val_ratio", 0.0)),
        "use_lowdim_obs": bool(args.use_lowdim_obs),
    }


def build_visual_model_from_config(config: dict[str, Any]) -> VisualDenoisingMLP:
    """Instantiate a visual denoising model from checkpoint config."""
    return VisualDenoisingMLP(
        obs_dim=int(config["obs_dim"]),
        action_dim=int(config["action_dim"]),
        obs_horizon=int(config["obs_horizon"]),
        pred_horizon=int(config["pred_horizon"]),
        image_feature_dim=int(config["image_feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        time_embed_dim=int(config["time_embed_dim"]),
        use_lowdim_obs=bool(config.get("use_lowdim_obs", True)),
    )


def prepare_eval_frame(frame: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    """Resize and convert one rendered frame to a model input tensor."""
    frame = resize_rgb_image(frame, image_size)
    return torch.as_tensor(frame, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0


def denormalize_visual_actions(actions: torch.Tensor, normalizer: dict[str, torch.Tensor]) -> torch.Tensor:
    """Map visual policy action samples back to environment action space."""
    return unnormalize_actions(actions, normalizer)
