"""Unitree Go2 articulation imported from the local URDF.

This mirrors the Go2W sim2sim actuator setup with the wheel joints removed.
Keeping the physics, initial pose, and actuator values visible here makes
sim2sim tuning easier.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_GO2_URDF = _DATA_DIR / "Robots" / "unitree" / "go2_description" / "urdf" / "go2_description.urdf"


GO2_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


UNITREE_GO2_URDF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_GO2_URDF),
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=40.0,
            stiffness=50.0,
            damping=0.8,
            friction=0.0,
        ),
        "thigh": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=40.0,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
        ),
        "calf": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=40.0,
            stiffness=50.0,
            damping=1.2,
            friction=0.0,
        ),
    }
)
"""Go2 with Go2W-style position-controlled legs split by joint group."""
