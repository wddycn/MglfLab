"""Compare MuJoCo actuator test data against Isaac replay data."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim2sim_profiles import get_profile, profile_names, short_joint_labels

DEFAULT_PROFILE = "go2w"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=profile_names(), default=DEFAULT_PROFILE)
    parser.add_argument("--mujoco", default=None)
    parser.add_argument("--isaac", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=0)
    args = parser.parse_args()
    profile = get_profile(args.profile)
    args.mujoco = args.mujoco or profile.actuator_mujoco_log
    args.isaac = args.isaac or profile.actuator_isaac_log
    args.out_dir = args.out_dir or f"/home/mglf/rc/MglfLab/logs/sim2sim/{profile.robot}_actuator_compare"
    args.joint_labels = short_joint_labels(profile.joint_names)
    return args


def load_csv(path):
    with open(path, "r", newline="") as f:
        first = f.readline()
        lines = f.readlines() if first.startswith("#") else [first] + f.readlines()
    reader = csv.DictReader(lines)
    rows = list(reader)

    def collect(prefix):
        names = sorted(
            [name for name in reader.fieldnames if name.startswith(prefix + "_")],
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)

    return {
        "time": np.asarray([float(row["time"]) for row in rows], dtype=np.float64),
        "target_pos": collect("target_pos"),
        "target_vel": collect("target_vel"),
        "joint_pos": collect("joint_pos"),
        "joint_vel": collect("joint_vel"),
        "tau_est": collect("tau_est"),
        "root_quat_w": collect("root_quat_w"),
        "root_ang_vel_b": collect("root_ang_vel_b"),
    }


def align_by_time(mujoco_time, mujoco_values, isaac_time, isaac_values, max_steps):
    valid = isaac_time <= mujoco_time[-1]
    if max_steps > 0:
        selected = np.flatnonzero(valid)[:max_steps]
        valid = np.zeros_like(valid, dtype=bool)
        valid[selected] = True
    t = isaac_time[valid]
    dims = min(mujoco_values.shape[1], isaac_values.shape[1])
    mujoco_aligned = np.column_stack(
        [np.interp(t, mujoco_time, mujoco_values[:, dim]) for dim in range(dims)]
    )
    return t, mujoco_aligned, isaac_values[valid, :dims]


def stats(a, b):
    diff = b - a
    return np.mean(np.abs(diff)), np.sqrt(np.mean(diff * diff)), np.max(np.abs(diff))


def plot(out_dir, key, time, mujoco, isaac, joint_labels, max_dims=16):
    dims = min(mujoco.shape[1], max_dims)
    fig, axes = plt.subplots(dims, 1, figsize=(12, max(4, 2.0 * dims)), sharex=True)
    if dims == 1:
        axes = [axes]
    for i in range(dims):
        axes[i].plot(time, mujoco[:, i], label="MuJoCo", linewidth=1.1)
        axes[i].plot(time, isaac[:, i], label="Isaac", linewidth=1.0, alpha=0.85)
        label = joint_labels[i] if i < len(joint_labels) else str(i)
        axes[i].set_ylabel(label)
        axes[i].grid(True, alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(key)
    fig.tight_layout()
    fig.savefig(out_dir / f"{key}.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mujoco = load_csv(args.mujoco)
    isaac = load_csv(args.isaac)

    lines = [f"profile: {args.profile}", f"MuJoCo: {args.mujoco}", f"Isaac:  {args.isaac}", "", "Isaac - MuJoCo errors:"]
    lines.append("Joint index order:")
    for i, name in enumerate(args.joint_labels):
        lines.append(f"  {i:02d}: {name}")
    lines.append("")
    for key in ["joint_pos", "joint_vel", "tau_est", "root_quat_w", "root_ang_vel_b"]:
        time, a, b = align_by_time(mujoco["time"], mujoco[key], isaac["time"], isaac[key], args.max_steps)
        mae, rmse, max_abs = stats(a, b)
        lines.append(f"{key}: shape={a.shape}, mae={mae:.6g}, rmse={rmse:.6g}, max_abs={max_abs:.6g}")
        plot(out_dir, key, time, a, b, args.joint_labels)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved actuator comparison to: {out_dir}")


if __name__ == "__main__":
    main()
