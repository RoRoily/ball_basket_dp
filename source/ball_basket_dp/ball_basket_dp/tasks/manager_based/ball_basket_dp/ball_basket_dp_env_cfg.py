# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp


##
# Scene definition
##


@configclass
class BallBasketDpSceneCfg(InteractiveSceneCfg):
    """Scene for the first headless ball-basket task.

    This V0 intentionally has no robot. It lets us validate rigid object spawning,
    reset randomization, low-dimensional observations, and success logic before
    adding Franka control.
    """

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # task object
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.04), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_linear_velocity=10.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.5,
                restitution=0.2,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.18, 0.12)),
        ),
    )

    # fixed target marker. Success is computed geometrically around this center.
    basket_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketTarget",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 0.0, 0.01), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.18,
            height=0.02,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.95)),
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # No actions in V0. The scripted-expert/Franka action terms come after the
    # object reset, observation, and success machinery is verified.
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        ball_pos = ObsTerm(func=mdp.ball_position, params={"asset_cfg": SceneEntityCfg("ball")})
        ball_vel = ObsTerm(func=mdp.ball_linear_velocity, params={"asset_cfg": SceneEntityCfg("ball")})
        basket_pos = ObsTerm(func=mdp.basket_position, params={"basket_center": (0.75, 0.0, 0.01)})
        ball_to_basket = ObsTerm(
            func=mdp.ball_to_basket_vector,
            params={"asset_cfg": SceneEntityCfg("ball"), "basket_center": (0.75, 0.0, 0.01)},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": (-0.25, 0.25),
                "y": (-0.35, 0.35),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    distance_to_basket = RewTerm(
        func=mdp.ball_to_basket_distance,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("ball"), "basket_center": (0.75, 0.0, 0.01)},
    )
    in_basket = RewTerm(
        func=mdp.ball_in_basket_reward,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "basket_center": (0.75, 0.0, 0.01),
            "basket_radius": 0.18,
            "z_bounds": (0.0, 0.20),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Task success
    ball_in_basket = DoneTerm(
        func=mdp.ball_in_basket,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "basket_center": (0.75, 0.0, 0.01),
            "basket_radius": 0.18,
            "z_bounds": (0.0, 0.20),
        },
    )


##
# Environment configuration
##


@configclass
class BallBasketDpEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BallBasketDpSceneCfg = BallBasketDpSceneCfg(num_envs=256, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 4
        # viewer settings
        self.viewer.eye = (2.0, -2.0, 1.5)
        self.viewer.lookat = (0.45, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
