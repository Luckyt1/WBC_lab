# ELF3 PPO

基于 Isaac Lab 的 ELF3 31 关节机器人运动控制实验代码。策略用 PPO 控制腰部与双腿的 15 个关节，上肢 14 个关节跟随外部动作参考；训练中加入左右镜像对称性与身体状态、动力学辅助预测。

本仓库提供源码、配置、测试和离线评估工具。机器人模型、AMASS 动作数据和预训练权重需要另外准备。当前代码仍用于实验；训练奖励提高或单次仿真通过不等于真机效果已验证。

## 当前默认配方

入口配置：[configs/ppo_symmetry_dynamics.json](configs/ppo_symmetry_dynamics.json)。

| 项目 | 设置 |
| --- | --- |
| 任务 | `walk_elf3_tang_bxi_robust` |
| 初始化 | 随机初始化，seed 42 |
| 并行环境 / 训练更新数 | 4096 / 80,000 |
| 目标骨盆高度 | 固定 0.8 m |
| 指令采样 | 50% 站立、25% 前后走、12.5% 侧移、12.5% 转向 |
| 上肢参考 | 70% AMASS、30% 中立姿态 |
| 对称性 | 镜像数据增强 + 镜像损失，系数 0.1 |
| 辅助预测 | 当前骨盆速度、双脚接触；结合执行动作预测下一步关节速度、骨盆角速度 |
| 课程 | 按训练轮次推进；未启用评估门槛驱动的课程控制器 |
| 保存 / 评估 | 每 500 / 1000 轮 |

时间课程中，上肢幅度从 0.05 开始，在 5,000–20,000 轮逐步升至 1；推力扰动在 20,000 轮达到完整强度，重置速度扰动在 30,000 轮达到完整强度。此配置保留站立位置惩罚，不启用实验性的 BCD 抬脚奖励。

对称性约束比较原样本与镜像样本的策略输出，不要求左右腿同时做相同动作。辅助头仅在训练中提供监督，导出 ONNX 时仅包含动作网络。

## 安装与资源

完整步骤见 [安装说明](docs/installation.md)。训练环境参考 Python 3.11、Isaac Sim 5.0、Isaac Lab 2.2；本仓库内的 `rsl_rl/` 含 PPO 和辅助预测改动，需要使用这份源码。

在**单独准备的 Isaac Lab 环境**中，从仓库根目录执行：

```bash
python -m pip install -e ./rsl_rl --no-deps
python -m pip install -e '.[dev,viewer]'
cp .env.example .env
# 编辑 .env，填入机器人资产、动作数据、BXI 框架、离线评估 Python 路径
set -a
source .env
set +a
python scripts/check_resources.py --deployment
```

MuJoCo / ONNX 评估还需要安装说明中的 CPU 环境。`.env` 不会自动加载，打开新终端后需重新执行 `source` 部分。

## 训练

先检查命令；`--dry-run` 不会启动训练：

```bash
python scripts/train_current.py --output logs/my_run --dry-run
python scripts/train_current.py --output logs/my_run
```

`ELF3_DEPLOYMENT_PYTHON` 指定离线评估环境，也可用 `--deployment-python /path/to/python` 覆盖。每次新训练使用新的输出目录。

运行状态和评估汇总位于 `logs/my_run/state.json`、`logs/my_run/RESULTS.md`；模型及 `params/` 位于 `logs/walk_tang_bxi31_auxiliary_scratch/<run>/`。`state.json` 中的 `latest_checkpoint` 给出最近检查点路径。这里的“轮”指 PPO 更新次数。

## Play

将 `RUN_DIRECTORY_NAME` 替换为上述模型目录的名称，将检查点名称替换为实际文件名：

```bash
python legged_lab/scripts/play.py \
  --task walk_elf3_tang_bxi_robust --num_envs 1 \
  --experiment_name walk_tang_bxi31_auxiliary_scratch \
  --load_run RUN_DIRECTORY_NAME --checkpoint model_60000.pt \
  --pelvis_height 0.8 --command 0.3 0 0 0.8 0
```

`--command` 依次为前向速度、侧向速度、转向角速度、骨盆高度、腰部偏航角。观察站立时改为 `--command 0 0 0 0.8 0`。此命令用于仿真播放，不连接机器人。

## 导出与评估

导出时保留检查点同级的 `params/agent.yaml`、`params/env.yaml`；只复制 `.pt` 文件不足以复现配置。

```bash
"$ELF3_DEPLOYMENT_PYTHON" tools/export_walk_actor_onnx.py \
  --checkpoint logs/walk_tang_bxi31_auxiliary_scratch/RUN_DIRECTORY_NAME/model_60000.pt \
  --output output/policy.onnx

PYTHONPATH="$ELF3_BXI_FRAMEWORK_DIR:${PYTHONPATH:-}" \
  "$ELF3_DEPLOYMENT_PYTHON" tools/evaluate_walk_mujoco_standing.py \
  --mod deployment/com.bxi.walk_robust \
  --xml "${ELF3_MUJOCO_XML:-$ELF3_BXI_FRAMEWORK_DIR/data/mujoco_simulation/elf3.xml}" \
  --onnx output/policy.onnx --output output/standing.json
```

完整检查点评估需要 Isaac 环境和 CPU 评估环境；命令见 [代码与实验说明](docs/code-guide.md)。当前 ONNX 契约为 `obs[1,840] -> actions[1,15]`，包含 10 帧历史，控制频率 50 Hz。离线适配器读取模型中的高度元数据。

## 代码结构

```text
configs/                         当前可复用的训练配方
scripts/                         配方启动和外部资源检查入口
legged_lab/envs/elf3_tang/        环境、奖励、课程、上肢参考、对称性
legged_lab/assets/elf3_bxi_31/    固定关节参数与碰撞几何数据
rsl_rl/                          修改后的 PPO、网络、存储和 runner
tools/                           训练监督、评估、ONNX 导出、实验诊断
deployment/com.bxi.walk_robust/   显式加载 ONNX 的离线 MuJoCo 适配器
tests/                           CPU 回归测试
resources/                       本地外部资源目录，不提交 Git
```

历史消融和课程控制工具保留用于复现实验，其入口与当前默认配方的区别见 [代码说明](docs/code-guide.md)。

## 测试

在安装 PyTorch 的 CPU 环境中执行：

```bash
python -m pip install -r requirements/cpu-tests.txt
python -m pytest -q -p no:cacheprovider tests
```

测试涵盖辅助预测、镜像变换、奖励、课程、步态诊断与恢复训练检查。不要求启动 Isaac Sim；依赖外部 URDF 的检查在资源未提供时跳过。GitHub Actions 运行相同的 CPU 测试和配方命令检查；完整 Isaac 训练与 MuJoCo 仿真不在该 CI 中执行。

本次源码整理的本地检查结果见 [验证记录](docs/validation.md)。

## 上传 GitHub

本目录可作为独立仓库。`.gitignore` 已排除日志、检查点、数据集、机器人模型、缓存、虚拟环境和 `.env`。建立空的 GitHub 仓库后，在本目录执行（将远程地址替换为自己的）：

```bash
git init -b main
git add .
git status --short
git commit -m "Initial ELF3 PPO source release"
git remote add origin https://github.com/YOUR_USERNAME/elf3-ppo.git
git push -u origin main
```

先核对 `git status` 再提交；大模型可在单独的 Release 中管理。本次源码整理未上传远程仓库。

## 许可与来源

源码沿用 BSD-3-Clause；保留的第三方组件以自身许可和文件头为准。见 [LICENSE](LICENSE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。机器人资产、运行时框架和 AMASS 数据未包含在分发中，也不由本仓库的源码许可重新授权。
