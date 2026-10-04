"""Shared robot profiles for sim2sim actuator/contact fitting scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MGLF_LAB_ROOT = Path("/home/mglf/rc/MglfLab")
MGLF_SAR_ROOT = Path("/home/mglf/rc/Mglf_sar")


GO2_LEG_JOINT_NAMES = [
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

GO2W_WHEEL_JOINT_NAMES = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]


@dataclass(frozen=True)
class Sim2SimProfile:
    name: str
    robot: str
    isaac_task: str
    joint_names: tuple[str, ...]
    action_scales: tuple[float, ...]
    direct_joint_targets: bool = False

    @property
    def actuator_mujoco_log(self) -> str:
        return str(MGLF_SAR_ROOT / "logs" / "sim2sim" / f"{self.robot}_actuator_mujoco.csv")

    @property
    def actuator_isaac_log(self) -> str:
        return str(MGLF_LAB_ROOT / "logs" / "sim2sim" / f"{self.robot}_actuator_isaac.csv")

    @property
    def contact_mujoco_log(self) -> str:
        return str(MGLF_SAR_ROOT / "logs" / "sim2sim" / f"{self.robot}_contact_mujoco.csv")

    @property
    def contact_isaac_log(self) -> str:
        return str(MGLF_LAB_ROOT / "logs" / "sim2sim" / f"{self.robot}_contact_isaac.csv")


PROFILES = {
    "go2": Sim2SimProfile(
        name="go2",
        robot="go2",
        isaac_task="Go2-Rough-Teleop-v0",
        joint_names=tuple(GO2_LEG_JOINT_NAMES),
        action_scales=(0.25,) * len(GO2_LEG_JOINT_NAMES),
        direct_joint_targets=False,
    ),
    "go2w": Sim2SimProfile(
        name="go2w",
        robot="go2w",
        isaac_task="Go2W-Flat-Handstand-Back-v0",
        joint_names=tuple(GO2_LEG_JOINT_NAMES + GO2W_WHEEL_JOINT_NAMES),
        action_scales=tuple(0.125 if "hip_joint" in name else 0.25 for name in GO2_LEG_JOINT_NAMES)
        + (2.7,) * len(GO2W_WHEEL_JOINT_NAMES),
        direct_joint_targets=False,
    ),
}


def profile_names() -> list[str]:
    return sorted(PROFILES)


def get_profile(name: str) -> Sim2SimProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise SystemExit(f"Unknown sim2sim profile '{name}'. Available: {', '.join(profile_names())}") from exc


def short_joint_labels(joint_names: list[str] | tuple[str, ...]) -> list[str]:
    labels = []
    for name in joint_names:
        label = name
        for suffix in ("_joint", "_foot"):
            label = label.replace(suffix, "")
        label = label.replace("foot", "wheel")
        labels.append(label)
    return labels
