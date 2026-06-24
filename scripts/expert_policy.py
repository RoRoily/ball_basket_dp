# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scripted expert utilities for the ball-basket task."""

from __future__ import annotations

import torch

# Observation layout produced by BallBasket-LowDim-v0.
EE_POS_SLICE = slice(18, 21)
EE_VEL_SLICE = slice(25, 28)
BALL_POS_SLICE = slice(28, 31)
BALL_VEL_SLICE = slice(31, 34)
BASKET_POS_SLICE = slice(34, 37)


def policy_obs(obs) -> torch.Tensor:
    """Extract the concatenated policy observation tensor from Gym/Isaac observations."""
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs["policy"] if isinstance(obs, dict) else obs


def basket_reachable(basket_pos: torch.Tensor, reachable_xy_radius: float) -> bool:
    """Whether the basket target is close enough for a direct drop expert."""
    return bool(torch.linalg.norm(basket_pos[0, :2]).item() <= reachable_xy_radius)


def plan_mode(obs: torch.Tensor, requested_mode: str, reachable_xy_radius: float) -> str:
    """Resolve auto/drop/throw into the actual expert plan."""
    if requested_mode != "auto":
        return requested_mode
    return "drop" if basket_reachable(obs[:, BASKET_POS_SLICE], reachable_xy_radius) else "throw"


def stage_at_step(step: int, plan: str) -> str:
    """Return the current finite-state expert stage."""
    if step < 90:
        return "above_ball"
    if step < 150:
        return "descend"
    if step < 190:
        return "close"
    if step < 250:
        return "lift"
    if plan == "throw":
        if step < 310:
            return "windup"
        if step < 350:
            return "swing"
        if step < 370:
            return "release_throw"
    else:
        if step < 340:
            return "above_basket"
        if step < 380:
            return "release_drop"
    return "hold"


def basket_direction(basket_pos: torch.Tensor) -> torch.Tensor:
    """Unit vector in xy from robot base toward the basket."""
    direction = basket_pos[:, :2].clone()
    norm = torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    return direction / norm


def throw_release_xy(basket_pos: torch.Tensor, reachable_xy_radius: float) -> torch.Tensor:
    """Plan a reachable xy release point along the ray to the basket."""
    direction = basket_direction(basket_pos)
    release_radius = min(reachable_xy_radius, max(0.25, reachable_xy_radius - 0.04))
    return direction * release_radius


def target_for_stage(
    stage: str, ball_pos: torch.Tensor, basket_pos: torch.Tensor, reachable_xy_radius: float
) -> torch.Tensor:
    """End-effector target for the current stage in each environment frame."""
    if stage == "above_ball":
        return ball_pos + torch.tensor((0.0, 0.0, 0.28), device=ball_pos.device)
    if stage in ("descend", "close"):
        return ball_pos + torch.tensor((0.0, 0.0, 0.09), device=ball_pos.device)
    if stage == "lift":
        return ball_pos + torch.tensor((0.0, 0.0, 0.34), device=ball_pos.device)
    if stage == "windup":
        direction = basket_direction(basket_pos)
        target = torch.zeros_like(ball_pos)
        target[:, :2] = throw_release_xy(basket_pos, reachable_xy_radius) - 0.18 * direction
        target[:, 2] = 0.34
        return target
    if stage == "swing":
        target = torch.zeros_like(ball_pos)
        target[:, :2] = throw_release_xy(basket_pos, reachable_xy_radius) + 0.10 * basket_direction(basket_pos)
        target[:, 2] = 0.26
        return target
    if stage == "release_throw":
        target = torch.zeros_like(ball_pos)
        target[:, :2] = throw_release_xy(basket_pos, reachable_xy_radius) + 0.13 * basket_direction(basket_pos)
        target[:, 2] = 0.24
        return target
    if stage == "above_basket":
        return basket_pos + torch.tensor((0.0, 0.0, 0.34), device=basket_pos.device)
    if stage == "release_drop":
        return basket_pos + torch.tensor((0.0, 0.0, 0.22), device=basket_pos.device)
    return basket_pos + torch.tensor((0.0, 0.0, 0.26), device=basket_pos.device)


def expert_action(
    obs: torch.Tensor, step: int, plan: str, position_scale: float, reachable_xy_radius: float
) -> torch.Tensor:
    """Compute dx/dy/dz + gripper action for the scripted expert."""
    stage = stage_at_step(step, plan)
    ee_pos = obs[:, EE_POS_SLICE]
    ball_pos = obs[:, BALL_POS_SLICE]
    basket_pos = obs[:, BASKET_POS_SLICE]

    target = target_for_stage(stage, ball_pos, basket_pos, reachable_xy_radius)
    arm_cmd = torch.clamp((target - ee_pos) / position_scale, min=-1.0, max=1.0)

    gripper_cmd = torch.ones((obs.shape[0], 1), device=obs.device)
    if stage in ("close", "lift", "above_basket", "windup", "swing"):
        gripper_cmd.fill_(-1.0)

    return torch.cat((arm_cmd, gripper_cmd), dim=-1)


def write_ball_state(env, pos_env: torch.Tensor, lin_vel: torch.Tensor) -> None:
    """Write ball pose and velocity to the simulator."""
    scene = env.unwrapped.scene
    ball = scene["ball"]
    pos_w = pos_env + scene.env_origins
    quat_w = torch.zeros((pos_w.shape[0], 4), device=pos_w.device)
    quat_w[:, 0] = 1.0
    ang_vel = torch.zeros_like(lin_vel)
    ball.write_root_pose_to_sim(torch.cat((pos_w, quat_w), dim=-1))
    ball.write_root_velocity_to_sim(torch.cat((lin_vel, ang_vel), dim=-1))


def grasp_candidate_mask(obs: torch.Tensor, stage: str, grasp_distance: float) -> torch.Tensor:
    """Only allow virtual attach when the hand is physically close to the ball."""
    if stage != "close":
        return torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
    ee_pos = obs[:, EE_POS_SLICE]
    ball_pos = obs[:, BALL_POS_SLICE]
    return torch.linalg.norm(ball_pos - ee_pos, dim=1) <= grasp_distance


def carry_ball_with_hand(env, obs: torch.Tensor, grasped: torch.Tensor, grasp_offset_z: float) -> None:
    """Move grasped balls with the end-effector."""
    if not bool(grasped.any().item()):
        return
    ee_pos = obs[:, EE_POS_SLICE]
    ee_vel = obs[:, EE_VEL_SLICE]
    ball_pos = obs[:, BALL_POS_SLICE]
    ball_vel = obs[:, BALL_VEL_SLICE]
    grasp_offset = torch.tensor((0.0, 0.0, grasp_offset_z), device=obs.device).unsqueeze(0)
    target_pos = torch.where(grasped[:, None], ee_pos + grasp_offset, ball_pos)
    target_vel = torch.where(grasped[:, None], ee_vel, ball_vel)
    write_ball_state(env, target_pos, target_vel)


def release_throw_ball(
    env,
    obs: torch.Tensor,
    grasped: torch.Tensor,
    throw_time: float,
    throw_speed_scale: float,
    grasp_offset_z: float,
) -> None:
    """Release grasped balls with a ballistic velocity toward the basket."""
    if not bool(grasped.any().item()):
        return
    ee_pos = obs[:, EE_POS_SLICE]
    ball_pos = obs[:, BALL_POS_SLICE]
    ball_vel = obs[:, BALL_VEL_SLICE]
    basket_pos = obs[:, BASKET_POS_SLICE]
    release_pos = ee_pos + torch.tensor((0.0, 0.0, grasp_offset_z), device=obs.device).unsqueeze(0)
    target_pos = basket_pos + torch.tensor((0.0, 0.0, 0.08), device=obs.device).unsqueeze(0)
    gravity = torch.tensor((0.0, 0.0, -9.81), device=obs.device).unsqueeze(0)
    velocity = (target_pos - release_pos - 0.5 * gravity * throw_time * throw_time) / throw_time
    target_pos = torch.where(grasped[:, None], release_pos, ball_pos)
    target_vel = torch.where(grasped[:, None], velocity * throw_speed_scale, ball_vel)
    write_ball_state(env, target_pos, target_vel)
