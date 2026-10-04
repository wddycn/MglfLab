"""Compare MuJoCo and Isaac contact-test logs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim2sim_profiles import get_profile, profile_names


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=profile_names(), default="go2w")
    parser.add_argument("--mujoco", default=None)
    parser.add_argument("--isaac", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--skip_steps", type=int, default=100)
    args = parser.parse_args()
    profile = get_profile(args.profile)
    args.mujoco = args.mujoco or profile.contact_mujoco_log
    args.isaac = args.isaac or profile.contact_isaac_log
    args.out_dir = args.out_dir or f"/home/mglf/rc/MglfLab/logs/sim2sim/{profile.robot}_contact_compare"
    return args


def load_csv(path):
    with open(path, "r", newline="") as f:
        first = f.readline()
        lines = f.readlines() if first.startswith("#") else [first] + f.readlines()
    reader = csv.DictReader(lines)
    rows = list(reader)

    def collect(prefix):
        names = sorted([n for n in reader.fieldnames if n.startswith(prefix + "_")], key=lambda n: int(n.rsplit("_", 1)[1]))
        if not names:
            return None
        return np.asarray([[float(row[n]) for n in names] for row in rows], dtype=np.float64)

    data = {key: collect(key) for key in ["joint_vel", "root_quat_w", "root_ang_vel_b"]}
    data["contact_count"] = np.asarray([float(row["contact_count"]) for row in rows], dtype=np.float64)[:, None]
    data["total_contact_force_norm"] = np.asarray([float(row["total_contact_force_norm"]) for row in rows], dtype=np.float64)[:, None]
    add_orientation_metrics(data)
    return data


def normalize_quat_wxyz(quat):
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    return quat / np.maximum(norm, 1e-12)


def quat_conj(quat):
    out = quat.copy()
    out[:, 1:] *= -1.0
    return out


def quat_mul(a, b):
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def quat_rotate_inverse(quat, vec):
    vec_quat = np.zeros((quat.shape[0], 4), dtype=np.float64)
    vec_quat[:, 1:] = vec[None, :]
    return quat_mul(quat_mul(quat_conj(quat), vec_quat), quat)[:, 1:]


def roll_pitch_yaw_from_quat(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=1)


def yaw_quat(yaw):
    half = 0.5 * yaw
    quat = np.zeros((yaw.shape[0], 4), dtype=np.float64)
    quat[:, 0] = np.cos(half)
    quat[:, 3] = np.sin(half)
    return quat


def add_orientation_metrics(data):
    quat = data.get("root_quat_w")
    if quat is None:
        return
    quat = normalize_quat_wxyz(quat)
    rpy = roll_pitch_yaw_from_quat(quat)
    gravity_w = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    yaw_only = yaw_quat(rpy[:, 2])
    yaw_free = normalize_quat_wxyz(quat_mul(quat_conj(yaw_only), quat))

    data["root_quat_w"] = quat
    data["projected_gravity_b"] = quat_rotate_inverse(quat, gravity_w)
    data["roll_pitch"] = rpy[:, :2]
    data["yaw_normalized_quat_w"] = yaw_free


def common(a, b, skip):
    n = min(a.shape[0], b.shape[0])
    d = min(a.shape[1], b.shape[1])
    return a[skip:n, :d], b[skip:n, :d]


def plot(out_dir, key, mujoco, isaac):
    dims = mujoco.shape[1]
    fig, axes = plt.subplots(dims, 1, figsize=(12, max(3, 2 * dims)), sharex=True)
    if dims == 1:
        axes = [axes]
    t = np.arange(mujoco.shape[0])
    for i in range(dims):
        axes[i].plot(t, mujoco[:, i], label="MuJoCo", linewidth=1.1)
        axes[i].plot(t, isaac[:, i], label="Isaac", linewidth=1.0, alpha=0.85)
        axes[i].grid(True, alpha=0.25)
        axes[i].set_ylabel(str(i))
    axes[0].legend()
    axes[-1].set_xlabel("step")
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

    lines = [f"profile: {args.profile}", f"MuJoCo: {args.mujoco}", f"Isaac:  {args.isaac}", f"skip_steps: {args.skip_steps}", "", "Isaac - MuJoCo errors:"]
    keys = [
        "joint_vel",
        "projected_gravity_b",
        "roll_pitch",
        "yaw_normalized_quat_w",
        "root_quat_w",
        "root_ang_vel_b",
        "contact_count",
        "total_contact_force_norm",
    ]
    for key in keys:
        a, b = common(mujoco[key], isaac[key], args.skip_steps)
        diff = b - a
        lines.append(f"{key}: shape={a.shape}, mae={np.mean(np.abs(diff)):.6g}, rmse={np.sqrt(np.mean(diff*diff)):.6g}, max_abs={np.max(np.abs(diff)):.6g}")
        plot(out_dir, key, a, b)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved contact comparison to: {out_dir}")


if __name__ == "__main__":
    main()
