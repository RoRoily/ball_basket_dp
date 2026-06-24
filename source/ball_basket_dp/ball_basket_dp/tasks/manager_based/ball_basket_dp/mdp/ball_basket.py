# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _basket_center_tensor(
    env: ManagerBasedRLEnv, basket_center: tuple[float, float, float], num_dims: int = 3
) -> torch.Tensor:
    """Return a per-environment basket center tensor in each environment frame."""
    center = torch.tensor(basket_center[:num_dims], device=env.device, dtype=torch.float32)
    return center.unsqueeze(0).repeat(env.num_envs, 1)


def ball_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Ball position in each environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def ball_linear_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Ball linear velocity in world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def basket_position(
    env: ManagerBasedRLEnv, basket_center: tuple[float, float, float] = (0.75, 0.0, 0.01)
) -> torch.Tensor:
    """Fixed basket target position in each environment frame."""
    return _basket_center_tensor(env, basket_center)


def ball_to_basket_vector(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    basket_center: tuple[float, float, float] = (0.75, 0.0, 0.01),
) -> torch.Tensor:
    """Vector from ball center to basket center in each environment frame."""
    return _basket_center_tensor(env, basket_center) - ball_position(env, asset_cfg)


def ball_to_basket_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    basket_center: tuple[float, float, float] = (0.75, 0.0, 0.01),
) -> torch.Tensor:
    """Euclidean distance from ball center to basket center."""
    return torch.linalg.norm(ball_to_basket_vector(env, asset_cfg, basket_center), dim=1)


def ball_in_basket(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    basket_center: tuple[float, float, float] = (0.75, 0.0, 0.01),
    basket_radius: float = 0.18,
    z_bounds: tuple[float, float] = (0.0, 0.20),
) -> torch.Tensor:
    """Whether the ball center lies inside the simple cylindrical basket region."""
    ball_pos = ball_position(env, asset_cfg)
    basket_xy = _basket_center_tensor(env, basket_center, num_dims=2)
    radial_distance = torch.linalg.norm(ball_pos[:, :2] - basket_xy, dim=1)
    z_ok = torch.logical_and(ball_pos[:, 2] >= z_bounds[0], ball_pos[:, 2] <= z_bounds[1])
    return torch.logical_and(radial_distance <= basket_radius, z_ok)


def ball_in_basket_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    basket_center: tuple[float, float, float] = (0.75, 0.0, 0.01),
    basket_radius: float = 0.18,
    z_bounds: tuple[float, float] = (0.0, 0.20),
) -> torch.Tensor:
    """Sparse success reward for the V0 geometric basket check."""
    return ball_in_basket(env, asset_cfg, basket_center, basket_radius, z_bounds).float()
