"""Replay the MuJoCo actuator-identification signal in Isaac Sim."""

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher
from sim2sim_profiles import get_profile, profile_names

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", choices=profile_names(), default="go2w")
parser.add_argument("--task", default=None)
parser.add_argument("--mujoco_log", default=None)
parser.add_argument("--output", default=None)
parser.add_argument("--steps", type=int, default=0, help="0 means use the number of rows in --mujoco_log.")
parser.add_argument(
    "--via_action_manager",
    action="store_true",
    default=None,
    help="Replay through env.step(actions), so JointPositionActionCfg/JointVelocityActionCfg scales are used.",
)
parser.add_argument(
    "--direct_joint_targets",
    dest="via_action_manager",
    action="store_false",
    help="Bypass the action manager and send joint position/velocity targets directly.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import mglf_lab  # noqa: F401


PROFILE = get_profile(args_cli.profile)
TASK = args_cli.task or PROFILE.isaac_task
MUJOCO_LOG = args_cli.mujoco_log or PROFILE.actuator_mujoco_log
OUTPUT = args_cli.output or PROFILE.actuator_isaac_log
VIA_ACTION_MANAGER = (not PROFILE.direct_joint_targets) if args_cli.via_action_manager is None else args_cli.via_action_manager


def read_mujoco_targets(path):
    with open(path, "r", newline="") as f:
        first = f.readline()
        lines = f.readlines() if first.startswith("#") else [first] + f.readlines()
    reader = csv.DictReader(lines)
    rows = list(reader)
    if not rows:
        raise RuntimeError(f"No target rows in {path}")

    def collect(prefix):
        names = sorted(
            [name for name in reader.fieldnames if name.startswith(prefix + "_")],
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        return torch.tensor([[float(row[name]) for name in names] for row in rows], dtype=torch.float32)

    times = torch.tensor([float(row["time"]) for row in rows], dtype=torch.float32)
    return times, collect("target_pos"), collect("target_vel")


def write_header(writer, num_dofs):
    header = ["step", "time"]
    header += [f"target_pos_{i}" for i in range(num_dofs)]
    header += [f"target_vel_{i}" for i in range(num_dofs)]
    header += [f"joint_pos_{i}" for i in range(num_dofs)]
    header += [f"joint_vel_{i}" for i in range(num_dofs)]
    header += [f"tau_est_{i}" for i in range(num_dofs)]
    header += [f"root_quat_w_{i}" for i in range(4)]
    header += [f"root_ang_vel_b_{i}" for i in range(3)]
    writer.writerow(header)


def build_action_from_targets(env, target_pos, target_vel, isaac_joint_names, profile, num_dofs):
    action_dim = env.unwrapped.action_manager.total_action_dim
    action = torch.zeros((1, action_dim), dtype=torch.float32, device=env.unwrapped.device)

    leg_names = [name for name in profile.joint_names[:num_dofs] if "foot_joint" not in name]
    default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0]

    for action_id, joint_name in enumerate(leg_names):
        if action_id >= action_dim:
            return action
        mujoco_id = profile.joint_names.index(joint_name)
        isaac_id = isaac_joint_names.index(joint_name)
        scale = profile.action_scales[mujoco_id]
        action[0, action_id] = (target_pos[0, mujoco_id] - default_joint_pos[isaac_id]) / scale

    wheel_action_names = [name for name in profile.joint_names[:num_dofs] if "foot_joint" in name]
    action_cursor = len(leg_names)
    for joint_name in wheel_action_names:
        if action_cursor >= action_dim:
            break
        mujoco_id = profile.joint_names.index(joint_name)
        action[0, action_cursor] = target_vel[0, mujoco_id] / profile.action_scales[mujoco_id]
        action_cursor += 1

    return action


def main():
    target_time, target_pos, target_vel = read_mujoco_targets(MUJOCO_LOG)

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.75)
    env_cfg.episode_length_s = max(float(getattr(env_cfg, "episode_length_s", 0.0)), 1.0e6)
    env = gym.make(TASK, cfg=env_cfg)
    obs, _ = env.reset()
    del obs

    robot = env.unwrapped.scene["robot"]
    num_dofs = min(target_pos.shape[1], robot.data.joint_pos.shape[1])
    isaac_joint_names = list(robot.data.joint_names)
    isaac_joint_ids = [isaac_joint_names.index(name) for name in PROFILE.joint_names[:num_dofs]]
    root_pose = torch.tensor([[0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=env.unwrapped.device)
    root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=env.unwrapped.device)
    output = Path(OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dt = env.unwrapped.step_dt if VIA_ACTION_MANAGER else env.unwrapped.physics_dt
    default_steps = int(torch.floor((target_time[-1] - target_time[0]) / log_dt).item()) + 1
    steps = default_steps if args_cli.steps <= 0 else min(args_cli.steps, default_steps)

    with output.open("w", newline="") as f:
        f.write(
            f"# source=isaacsim,profile={PROFILE.name},task={TASK},mujoco_log={MUJOCO_LOG},dt={log_dt},num_of_dofs={num_dofs}\n"
        )
        writer = csv.writer(f)
        write_header(writer, num_dofs)

        for step in range(steps):
            replay_time = float(step * log_dt)
            source_step = int(torch.argmin(torch.abs(target_time - replay_time)).item())
            q_target = target_pos[source_step : source_step + 1, :num_dofs].to(env.unwrapped.device)
            dq_target = target_vel[source_step : source_step + 1, :num_dofs].to(env.unwrapped.device)
            if not VIA_ACTION_MANAGER:
                dq_target = torch.zeros_like(dq_target)

            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)
            if VIA_ACTION_MANAGER:
                action = build_action_from_targets(env, q_target, dq_target, isaac_joint_names, PROFILE, num_dofs)
                env.step(action)
            else:
                if step == 0:
                    robot.write_joint_state_to_sim(q_target, torch.zeros_like(dq_target), joint_ids=isaac_joint_ids)
                robot.set_joint_position_target(q_target, joint_ids=isaac_joint_ids)
                robot.set_joint_velocity_target(dq_target, joint_ids=isaac_joint_ids)
                env.unwrapped.scene.write_data_to_sim()
                env.unwrapped.sim.step()
                env.unwrapped.scene.update(env.unwrapped.physics_dt)
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)
            env.unwrapped.scene.write_data_to_sim()

            row = [step, replay_time]
            row += q_target[0].detach().cpu().tolist()
            row += dq_target[0].detach().cpu().tolist()
            row += robot.data.joint_pos[0, isaac_joint_ids].detach().cpu().tolist()
            row += robot.data.joint_vel[0, isaac_joint_ids].detach().cpu().tolist()
            row += robot.data.applied_torque[0, isaac_joint_ids].detach().cpu().tolist()
            row += robot.data.root_quat_w[0].detach().cpu().tolist()
            row += robot.data.root_ang_vel_b[0].detach().cpu().tolist()
            writer.writerow(row)

    env.close()
    print(f"[INFO] Saved Isaac actuator replay to: {output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
