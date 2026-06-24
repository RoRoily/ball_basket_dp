# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
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

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

BASKET_CENTER = (0.75, 0.0, 0.01)
BALL_DEFAULT_POS = (0.35, 0.0, 0.04)
BALL_RADIUS = 0.04
BASKET_RADIUS = 0.18
BASKET_Z_BOUNDS = (0.0, 0.20)


##
# Scene definition
##


@configclass
class BallBasketDpSceneCfg(InteractiveSceneCfg):
    """Scene for the headless low-dimensional ball-basket task.

    This version adds Franka joint control while keeping the task low-dimensional.
    It is still a scaffolding environment: the scripted grasp/drop expert comes
    after robot actions, observations, and object reset are verified together.
    """

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # task object
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_DEFAULT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=BASKET_CENTER, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=BASKET_RADIUS,
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

    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        body_name="panda_hand",
        joint_names=["panda_joint.*"],
        scale=(0.04, 0.04, 0.04),
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls"),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*", "panda_finger_joint.*"])},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*", "panda_finger_joint.*"])},
        )
        end_effector_pose = ObsTerm(
            func=mdp.body_pose_in_env,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )
        end_effector_vel = ObsTerm(
            func=mdp.body_linear_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )
        ball_pos = ObsTerm(func=mdp.ball_position, params={"asset_cfg": SceneEntityCfg("ball")})
        ball_vel = ObsTerm(func=mdp.ball_linear_velocity, params={"asset_cfg": SceneEntityCfg("ball")})
        basket_pos = ObsTerm(func=mdp.basket_position, params={"basket_center": BASKET_CENTER})
        ball_to_basket = ObsTerm(
            func=mdp.ball_to_basket_vector,
            params={"asset_cfg": SceneEntityCfg("ball"), "basket_center": BASKET_CENTER},
        )
        ee_to_ball = ObsTerm(
            func=mdp.end_effector_to_ball_vector,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
                "ball_cfg": SceneEntityCfg("ball"),
            },
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
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*", "panda_finger_joint.*"]),
            "position_range": (-0.01, 0.01),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": (-0.20, 0.12),
                "y": (-0.30, 0.30),
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
        params={"asset_cfg": SceneEntityCfg("ball"), "basket_center": BASKET_CENTER},
    )
    in_basket = RewTerm(
        func=mdp.ball_in_basket_reward,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "basket_center": BASKET_CENTER,
            "basket_radius": BASKET_RADIUS,
            "z_bounds": BASKET_Z_BOUNDS,
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
            "basket_center": BASKET_CENTER,
            "basket_radius": BASKET_RADIUS,
            "z_bounds": BASKET_Z_BOUNDS,
        },
    )


##
# Environment configuration
##


@configclass
class BallBasketDpEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BallBasketDpSceneCfg = BallBasketDpSceneCfg(num_envs=128, env_spacing=2.5)
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
        self.episode_length_s = 6
        # viewer settings
        self.viewer.eye = (2.0, -2.0, 1.5)
        self.viewer.lookat = (0.45, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
