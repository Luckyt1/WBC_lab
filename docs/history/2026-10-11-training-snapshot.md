# 2026-10-11 训练源码快照

本目录对应本次从 32,000 轮检查点继续训练的 IK 执行 v3 版本。45,000、50,000、55,000 和 56,000 轮检查点使用相同源码与配方，区别是学习到的权重和优化器状态。

训练已按用户要求停止：最后日志迭代 56,296，最后完整保存的检查点为 `model_56000.pt`。本次提取没有启动训练或部署模型。

## 版本与配置

| 项目 | 本次训练设置 |
| --- | --- |
| 任务 | `walk_elf3_tang_bxi_robust` |
| PPO 并行环境 / 随机种子 | 4096 / 42 |
| 观测 / 动作 | 840 维历史观测 / 15 维腰腿动作 |
| 骨盆高度 / 控制周期 | 0.8 m / 20 ms |
| 命令比例 | 50% 站立、25% 前后走、12.5% 侧移、12.5% 转向 |
| 训练项 | dynamics 辅助头、镜像增强与损失、BCD v1 |
| IK 轨迹库 | `ik_reach_hold_return_predictive_v3_servo_20261008` |
| 检查点 / 完整评估间隔 | 500 / 5000 次更新 |
| 续训起点 / 原定累计目标 | `model_32000.pt` / 80,000 次更新 |

检查点中的迭代编号从零开始，因此 `model_32000.pt` 对应 32,001 次已完成更新；本次启动计划新增 47,999 次更新。

此次没有加入新的抬腿奖励、脚底高度目标或摆腿课程。仓库保留旧步态实验工具，但默认配方不启用这些实验。

## 来源核对

已逐项校验本次监督程序归档的 218 个源码文件，来源工作区与启动时归档一致。独立源码沿用 2026-10-09 提取版本的路径配置、资源预检、测试样本和文档。原始输入文件也全部匹配，没有遗漏本次训练期间的代码修改。

- `SOURCE_MANIFEST.json`：提取前输入哈希，以及独立版本中 24 个路径、资源与接入适配文件的前后哈希。
- `provenance/training_source_sha256.json`：本次训练启动时归档的源码哈希。
- `provenance/training_snapshot.json`：训练配置、检查点对应关系、权重与保存配置的哈希。

这里的“同一版”指训练核心与配方一致；独立目录保留了便携路径适配，不宣称所有包装文件与原训练目录逐字节一致。

## 本机权重与数据位置

权重和训练日志仍在原工作区，未复制进源码目录。本次 45,000–56,000 轮检查点目录为：

```text
~/elf3_lab/elf3_lab/logs/walk_tang_bxi31_auxiliary_scratch/
  2026-10-10_18-19-24_ik_execution_v3_bcd_height080_dynamics_symmetry_stand50_resume32000_20261010_181911_4096_from32001/
```

该目录中 `model_45000.pt` 是前面讨论的候选，`model_56000.pt` 是本次最后保存的检查点。旁边的 `params/env.yaml` 和 `params/agent.yaml` 必须与权重一起保留；续训会校验配置和学习状态，不能为了改变路径随意重写这些文件。

本机完整 IK 轨迹库位于：

```text
~/elf3_lab/elf3_lab/data/ik_reach_hold_return_predictive_v3_servo_20261008/
```

机器人资产、BXI 框架、IK 上游文件、AMASS 数据和仿真环境按 [安装说明](installation.md) 配置。源码中的 `tests/fixtures/ik_banks/` 是测试样本，不能用于正式训练。

## 使用本目录

先按 README 和安装说明准备环境，复制 `.env.example` 并配置外部资源。使用原检查点续训时，资源环境变量应指向保存配方所使用的原资产和轨迹库。

下面命令只检查从 45,000 轮继续到累计 80,000 次更新的启动参数：

```bash
cd ~/github/WBC_lab
set -a
source .env
set +a

ELF3_CHECKPOINT="$HOME/elf3_lab/elf3_lab/logs/walk_tang_bxi31_auxiliary_scratch/2026-10-10_18-19-24_ik_execution_v3_bcd_height080_dynamics_symmetry_stand50_resume32000_20261010_181911_4096_from32001/model_45000.pt"
python scripts/train_current.py \
  --output logs/ik_v3_resume_45000 \
  --resume-checkpoint "$ELF3_CHECKPOINT" \
  --max-iterations 34999 \
  --dry-run
```

默认不传 `--resume-checkpoint` 时是从零训练。`--max-iterations` 始终表示本次新增更新数。这里只提供命令，当前训练保持停止。

## 45,000 轮的已知表现

原工作区仿真评估中，未见 IK 目标 84 次试验全部存活，81 次到位，79 次综合通过；困难案例 36 次全部存活，30 次综合通过。MuJoCo 静止与走停测量窗口的最大漂移分别为 1.31 cm、0.77 cm。

该候选没有解决低抬脚：0.3 m/s 指令下每步脚底峰值净空中位数约 0.82–0.87 cm，0.6 m/s 下约 1.69–1.76 cm。普通直行测试完成 60 秒，但测量窗口内航向变化约 −15.6°。受扰站立与部分组合行走转向验收仍未通过。

这是有限仿真覆盖的记录，45,000 轮候选未做新的真机验证；源码提取不改变这一结论。完整评估保留在原监督目录的 `evaluations/model_45000/`。
