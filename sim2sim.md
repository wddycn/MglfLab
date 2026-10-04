# 多机器人 Sim2Sim 拟合流程

当前脚本通过 `--profile` 选择机器人配置。已内置：

| profile | MuJoCo robot | Isaac task | 说明 |
| --- | --- | --- | --- |
| `go2w` | `go2w` | `Go2W-Flat-Handstand-Back-v0` | 默认，保留原 Go2W handstand 拟合流程 |
| `go2` | `go2` | `Go2-Rough-Teleop-v0` | Go2 rough teleop，默认 direct joint target replay |

共享配置在：

```bash
/home/mglf/rc/MglfLab/scripts/sim2sim_profiles.py
```

以后新增机器人时，优先在这里加 profile，不要在 replay/compare 脚本里继续写死任务名、日志名和关节名。

## Go2W 示例

### 1. MuJoCo actuator test

```bash
cd /home/mglf/rc/Mglf_sar
./scripts/run_mujoco_actuator_test.sh go2w scene
```

### 2. Isaac actuator replay

```bash
cd /home/mglf/rc/MglfLab
python scripts/replay_actuator_isaac.py --profile go2w
```

### 3. 比较 actuator q/dq/tau

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/compare_actuator.py --profile go2w
```

### 4. MuJoCo contact test

```bash
cd /home/mglf/rc/Mglf_sar
./scripts/run_mujoco_contact_test.sh go2w scene
```

### 5. Isaac contact replay

```bash
cd /home/mglf/rc/MglfLab
python scripts/replay_contact_isaac.py --profile go2w
```

### 6. 比较 contact/base/joint

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/compare_contact.py --profile go2w
```

## Go2 示例

### 1. MuJoCo actuator test

```bash
cd /home/mglf/rc/Mglf_sar
./scripts/run_mujoco_actuator_test.sh go2 scene
```

### 2. Isaac actuator replay

```bash
cd /home/mglf/rc/MglfLab
python scripts/replay_actuator_isaac.py --profile go2
```

`go2` profile 默认使用 direct joint target replay。也可以显式写：

```bash
python scripts/replay_actuator_isaac.py --profile go2 --direct_joint_targets
```

### 3. 比较 actuator

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/compare_actuator.py --profile go2
```

### 4. MuJoCo contact test

```bash
cd /home/mglf/rc/Mglf_sar
./scripts/run_mujoco_contact_test.sh go2 scene
```

### 5. Isaac contact replay 和比较

```bash
cd /home/mglf/rc/MglfLab
python scripts/replay_contact_isaac.py --profile go2
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/compare_contact.py --profile go2
```

## 可覆盖参数

MuJoCo 侧：

```bash
ROBOT=go2 SCENE=scene_terrain ./scripts/run_mujoco_actuator_test.sh
ROBOT=go2 SCENE=scene ./scripts/run_mujoco_contact_test.sh
```

或者使用位置参数：

```bash
./scripts/run_mujoco_actuator_test.sh go2 scene
./scripts/run_mujoco_contact_test.sh go2w scene
```

Isaac 侧：

```bash
python scripts/replay_actuator_isaac.py \
  --profile go2 \
  --task Go2-Rough-Teleop-v0 \
  --mujoco_log /path/to/mujoco.csv \
  --output /path/to/isaac.csv
```

compare 脚本也可以覆盖路径：

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/compare_actuator.py \
  --profile go2 \
  --mujoco /path/to/mujoco.csv \
  --isaac /path/to/isaac.csv
```

## 调参方向

重点看对应机器人 asset/env 里的：

```text
stiffness
damping
effort_limit_sim
velocity_limit_sim
friction
action scale
```

经验规则：

```text
Isaac 跟踪太慢/幅值偏小：提高 stiffness
Isaac 超调/振荡：提高 damping 或降低 stiffness
Isaac 速度峰值太小：提高 velocity_limit_sim 或降低 damping
Isaac 速度尖峰太大：提高 damping
Isaac torque 长期顶住：提高 effort_limit_sim 或降低目标幅值
Isaac 比 MuJoCo 更“粘”：降低 friction / damping
Isaac 比 MuJoCo 更“滑”：提高 damping / friction
```
