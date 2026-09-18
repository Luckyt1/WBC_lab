# IsaacLab-Decoupled-WBC

WBC Lab 的核心训练子项目，基于 Isaac Lab / Isaac Sim、PyTorch 和 RSL-RL，为 ELF3 与 Unitree G1 提供解耦全身控制训练。项目通过腿腰策略配合独立的双臂动作，学习移动、平衡、身体高度与姿态控制。

**完整项目介绍、任务列表、环境安装及使用步骤见 [WBC Lab 使用指南](../README.md)。**

## 核心流程

1. 使用 PPO 训练教师策略，腿腰由策略控制，双臂回放 AMASS 动作或保持默认姿态。
2. 通过 BC → DAgger 蒸馏学生策略，学生使用去除足部接触通道后的历史观测。
3. 加载教师检查点或学生 TorchScript 模型，在仿真中进行交互或无界面回放。

## 目录导航

- `legged_lab/`：当前环境、奖励、任务配置、机器人模型及训练算法辅助代码。
- `dataset/g1/CMU/`：G1 与 ELF3 WBC 双臂运动需要的 AMASS 数据。
- `scripts/train_teacher.sh`、`scripts/train_student.sh`：教师训练与 BC → DAgger 蒸馏。
- `scripts/play_teacher.sh`、`scripts/play_student.sh`：训练后仿真回放。
- `tests/`、`tools/check_elf3_env.py`、`tools/check_elf3_asset.py`：训练逻辑和资产验证。

## 训练入口

从仓库根目录启动默认 ELF3 WBC 训练：

```bash
bash train.sh
```

若已激活环境并位于本项目目录：

```bash
WBC_TASK=elf3_wbc bash scripts/train_teacher.sh
```

项目内四个脚本的默认任务为 `elf3_flat`，使用其他任务时通过 `WBC_TASK` 指定。ELF3 模型包含 31 个可动关节，策略控制腿腰 15 个关节，双臂 14 个关节回放或保持，头部两个关节保持零位。G1 任务和全部已注册 ELF3 配置均保留。

已注册任务涵盖 ELF3 / G1 平地、G1 粗糙地形，以及 ELF3 WBC、下蹲、骨盆高度等实验。完整列表见 [任务注册入口](legged_lab/envs/__init__.py)；实验中的 `recovery` 表示训练修正，不代表跌倒起身。

## 数据与许可证

本仓库不包含历史模型、日志、实机部署程序或示例权重。学生蒸馏与回放需要先生成或另行提供兼容的教师检查点及对应 `params/` 配置。

源项目：`chrisyrniu/IsaacLab-Decoupled-WBC`，基准提交 `926756b492cb0f8cfbe1aae562e452efe3a11326`，包含本地训练修改。代码遵循 [BSD-3-Clause](LICENSE.txt)，数据来源及独立许可见 [数据说明](dataset/README.md)。
