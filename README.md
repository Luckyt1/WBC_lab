# WBC Lab：人形机器人解耦全身控制训练

WBC Lab 是一个基于 **Isaac Lab / Isaac Sim、PyTorch 和 RSL-RL** 的人形机器人仿真训练项目，面向 **ELF3 与 Unitree G1**，提供教师策略强化学习、学生策略蒸馏及仿真回放的完整流程。

项目围绕解耦全身控制（Decoupled Whole-Body Control，WBC）展开：双臂执行独立给定的动作，腿部和腰部策略学习跟踪移动、转向、身体高度与姿态指令，在上肢运动时维持平衡。适合研究上下肢协同、姿态控制、奖励设计、课程学习和策略蒸馏。

当前仓库保留训练源码、任务配置、机器人资产、部分 AMASS 动作数据及环境脚本。**不包含预训练权重、历史训练日志或实机部署程序**；首次使用需要从零训练，或自行提供兼容的检查点与运行配置。

## 控制与训练思路

1. **解耦控制**：策略输出腿腰 15 个关节的位置偏移，双臂 14 个关节由动作数据或默认姿态单独驱动。ELF3 共 31 个可动关节，剩余两个头部关节保持零位目标。
2. **教师训练**：通过 RSL-RL 的 PPO 算法在并行仿真环境中学习控制策略，结合指令课程、奖励函数与域随机化进行训练。
3. **学生蒸馏**：先进行行为克隆（BC），再进入 DAgger 阶段，在学生自身产生的状态上向教师学习。学生使用历史观测，并去除教师观测中的足部接触通道。
4. **仿真评估**：加载教师检查点或学生 TorchScript 模型回放，可使用图形界面交互或无界面定步数运行。

`elf3_wbc` 使用重定向到 G1 的 AMASS 双臂动作，再通过关节名称映射和肩部安装角补偿适配 ELF3。这里迁移的是**关节运动**，不保证不同臂长机器人之间的手部笛卡尔轨迹一致。具体映射与数据格式见 [动作数据说明](IsaacLab-Decoupled-WBC/dataset/README.md)。

## 支持的任务

任务统一注册在 [legged_lab/envs/__init__.py](IsaacLab-Decoupled-WBC/legged_lab/envs/__init__.py)，通过环境变量 `WBC_TASK` 选择。

| 任务 | 机器人 | 用途 |
| --- | --- | --- |
| `elf3_wbc` | ELF3 | 默认入口；AMASS 双臂运动下的移动、身体高度与姿态控制 |
| `elf3_flat` | ELF3 | 双臂保持默认姿态的平地基线 |
| `g1_flat` | G1 | G1 平地训练，回放 AMASS 双臂动作 |
| `g1_rough` | G1 | G1 粗糙地形训练，回放 AMASS 双臂动作 |
| `elf3_homie_squat` | ELF3 | HOMIE 风格下蹲实验，调整膝关节、高度奖励及低姿态下的站姿约束 |
| `elf3_homie_recovery` | ELF3 | 下蹲实验的修正配置，增加关节动作目标越界惩罚 |
| `elf3_pelvis_height` | ELF3 | 将高度跟踪参考从躯干改为骨盆的实验 |
| `elf3_pelvis_recovery` | ELF3 | 骨盆高度实验的修正配置，调整指令采样、奖励与 PPO 稳定性参数 |

后四项为实验配置；名称中的 `recovery` 指训练修正，不代表跌倒起身功能。不同机器人、关节顺序或高度参考的检查点不能直接混用。

## 项目结构

```text
WBC_lab/
├── README.md                         # 项目介绍与使用指南
├── setup_wbc_conda.sh                # 创建 Conda 环境
├── setup_wbc_training.sh             # 安装仿真与训练依赖
├── activate_wbc.sh                   # 激活环境并进入子项目
├── train.sh                          # ELF3 WBC 教师训练快捷入口
├── verify_wbc_training.sh            # 环境与训练链路验证
├── isaac_wbc_requirements.lock.txt   # 导出的 Python 包版本记录
├── wbc_*_constraints.txt             # 构建及训练依赖约束
└── IsaacLab-Decoupled-WBC/
    ├── legged_lab/
    │   ├── assets/                  # ELF3 / G1 模型及资产配置
    │   ├── envs/                    # 仿真环境、任务注册和机器人配置
    │   ├── mdp/                     # 指令生成与奖励函数
    │   ├── terrains/                # 地形和传感器相关配置
    │   ├── scripts/                 # 教师/学生训练与回放的 Python 入口
    │   └── utils/                   # 动作映射、观测、续训及诊断工具
    ├── scripts/                     # 四个训练/回放 Shell 入口
    ├── dataset/g1/CMU/              # 随仓库保留的 AMASS 动作子集
    ├── tests/                       # 关节布局、动作映射、训练逻辑等测试
    ├── tools/                      # ELF3 资产与仿真检查
    ├── pyproject.toml               # Python 包定义
    └── LICENSE.txt                 # 代码许可证
```

修改任务时，可从 [ELF3 配置](IsaacLab-Decoupled-WBC/legged_lab/envs/elf3/)、[G1 配置](IsaacLab-Decoupled-WBC/legged_lab/envs/g1/)和[奖励函数](IsaacLab-Decoupled-WBC/legged_lab/mdp/rewards.py)入手。

## 环境安装

安装脚本面向 **Linux x86_64**，训练和仿真需要 NVIDIA GPU 及兼容驱动，安装过程需要联网。脚本使用以下固定版本组合：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| PyTorch / torchvision | 2.7.0 / 0.22.0 |
| PyTorch CUDA 构建 | 12.8 |
| Isaac Sim | 5.0.0 |
| Isaac Lab | 2.2.0 |
| RSL-RL | 2.3.3 |

在仓库根目录执行：

```bash
bash setup_wbc_conda.sh
bash setup_wbc_training.sh
```

默认 Conda 路径为 `$HOME/miniconda3`，环境名为 `isaac_wbc`。若使用其他路径或环境名，**在安装前**设置：

```bash
export WBC_CONDA_ROOT=/path/to/miniconda3
export WBC_CONDA_ENV=isaac_wbc
```

本地项目以 editable 模式安装。详细包版本记录在 [isaac_wbc_requirements.lock.txt](isaac_wbc_requirements.lock.txt)，安装脚本会在成功后重新导出该文件。默认日志工具为 TensorBoard；安装脚本会移除环境内已有的 W&B，以避免其依赖与此版本 Isaac Sim 的约束冲突。

后续需要直接操作子项目时，在仓库根目录运行：

```bash
source ./activate_wbc.sh
```

此命令会激活环境、设置 `PYTHONPATH`，并将当前目录切换到 `IsaacLab-Decoupled-WBC/`。下面的示例分别注明了执行目录。

## 教师策略训练

在**仓库根目录**执行：

```bash
# 默认：elf3_wbc，1024 个并行环境，60000 轮，每 1000 轮保存
bash train.sh

# 自定义训练规模、轮数与实验名称
bash train.sh --num_envs=4096 --max_iterations=10000 --save_interval=500 --run_name=baseline

# 仅预览入口命令，不启动仿真；--dry-run 需放在首个参数位置
bash train.sh --dry-run --num_envs=4096

# 切换任务
WBC_TASK=elf3_flat bash train.sh
WBC_TASK=g1_flat bash train.sh
```

根据显存容量调整 `--num_envs`；可先用较小规模检查流程：

```bash
bash train.sh --num_envs=8 --max_iterations=5 --save_interval=1 --run_name=smoke
```

短运行只用于验证训练链路，不能用于判断策略是否已学会稳定控制。

根目录 `train.sh` 默认选择 `elf3_wbc`，子项目 `scripts/` 下四个入口默认选择 `elf3_flat`。直接调用子项目脚本时，应显式设置 `WBC_TASK`，并在训练、蒸馏和回放时保持一致。

### 续训与配置覆盖

在仓库根目录执行以下命令，首次创建固定名称的实验，之后重复执行会加载该目录的最新检查点：

```bash
bash train.sh --auto_resume --run_name=baseline --max_iterations=60000
```

此模式保存到 `IsaacLab-Decoupled-WBC/logs/elf3_wbc/baseline/`，`--max_iterations` 表示总目标轮数。普通训练则创建带时间戳的目录，如 `<时间戳>_baseline/`；同名后缀的普通训练不会被上述固定目录自动接续。

如需接续普通训练，可明确指定已有目录与检查点：

```bash
# 将示例目录和检查点替换为实际文件；这里的 1000 表示本次追加轮数
bash train.sh --resume --load_run=2026-09-18_18-00-00_baseline --checkpoint=model_1000.pt --max_iterations=1000
```

教师入口还支持重复传入 `--set KEY=VALUE` 覆盖配置；默认定位到环境配置，`agent.` 前缀定位到训练配置。例如：

```bash
bash train.sh --set 'commands.curriculum.enable=False'
```

续训时应沿用原实验的任务与配置；切换实验设计时使用新的运行名称。

## 学生蒸馏与仿真回放

先生成教师检查点，再从**仓库根目录**激活环境并选择实际的教师运行：

```bash
source ./activate_wbc.sh  # 后续命令在 IsaacLab-Decoupled-WBC/ 内执行
export WBC_TASK=elf3_wbc

# 查看已有教师检查点
find "logs/$WBC_TASK" -maxdepth 2 -name 'model_*.pt'

# 替换为实际目录名和检查点；baseline 对应上面的 --auto_resume 示例
export HTD_RUN=baseline
export HTD_CKPT=model_1000.pt

# 先检查教师效果；无图形界面时使用 --headless
bash scripts/play_teacher.sh --headless --max_steps=1000

# 执行 BC → DAgger 学生蒸馏
bash scripts/train_student.sh

# 学生训练完成并成功导出 JIT 模型后回放
bash scripts/play_student.sh --headless --max_steps=1000
```

学生脚本默认使用 1024 个环境，前 250000 步执行 BC，随后执行 DAgger 至总计 600000 步，每 5000 步保存一次，保留最近 3 个学生训练检查点。可通过 `--num_envs`、`--bc_steps`、`--max_steps` 等参数覆盖。

重复运行学生脚本会自动恢复同一教师来源下的最新学生检查点。蒸馏会读取教师保存的执行器参数和指令配置，并校验教师来源，需同时保留教师权重和 `params/`。当前蒸馏入口使用平面地形，不保留 `g1_rough` 的粗糙地形设置。

回放图形界面时去掉 `--headless`，可通过键盘调整控制指令，按键定义见 [keyboard.py](IsaacLab-Decoupled-WBC/legged_lab/utils/keyboard.py)。

## 训练产物与日志

产物位于 `IsaacLab-Decoupled-WBC/logs/<实验名>/<运行目录>/`。`elf3_wbc`、`elf3_flat`、`g1_flat` 和 `g1_rough` 默认以任务名作为实验名；四个 ELF3 实验任务默认继承 `elf3_wbc` 实验名。

为实验任务单独保存日志时，可在仓库根目录显式指定实验名：

```bash
WBC_TASK=elf3_pelvis_height bash train.sh --experiment_name=elf3_pelvis_height
```

Shell 蒸馏和回放脚本按 `logs/$WBC_TASK/` 查找文件，因此实验名应与 `WBC_TASK` 保持一致。使用这些实验任务时，还需在蒸馏和回放命令后追加 `--experiment_name="$WBC_TASK"`，使 Python 入口使用同一目录。

每个运行目录内的主要文件如下：

| 路径 | 内容 |
| --- | --- |
| `model_<轮数>.pt` | 教师训练检查点 |
| `params/env.yaml`、`params/agent.yaml` | 本次运行的环境及训练配置 |
| `events.out.tfevents.*` | 教师 TensorBoard 日志 |
| `student_checkpoints/student_step_<步数>.pt` | 学生训练检查点 |
| `student_checkpoints/student_policy_jit.pt` | 学生训练结束后成功导出的 TorchScript 模型，供学生回放脚本加载 |
| `student_checkpoints/teacher_source.txt` | 学生对应的教师来源记录 |
| `student_tensorboard/` | 学生 TensorBoard 日志 |

在已激活环境的**子项目目录**查看训练曲线：

```bash
tensorboard --logdir logs
```

## 验证与排查

在**仓库根目录**运行完整环境检查：

```bash
WBC_TASK=elf3_flat bash verify_wbc_training.sh
```

该脚本依次检查依赖一致性、CUDA 运算、单元测试、ELF3 仿真，以及短教师训练和学生蒸馏。显式设置 `elf3_flat` 可与脚本查找验证产物的路径保持一致。检查输出位于根目录 `environment_checks/`，短训练也会生成日志与检查点。

如只需运行已有单元测试，从仓库根目录执行：

```bash
source ./activate_wbc.sh
python -m pytest tests -q
```

| 问题 | 检查方向 |
| --- | --- |
| 找不到 Conda | 检查 `WBC_CONDA_ROOT`，或先执行 `setup_wbc_conda.sh` |
| CUDA 不可用 / 显存不足 | 检查 GPU 驱动与当前环境；显存不足时减小 `--num_envs` |
| 找不到教师检查点 | 检查 `WBC_TASK`、完整的 `HTD_RUN` 目录名和 `HTD_CKPT` 文件名 |
| 找不到学生 JIT 模型 | 确认学生训练已结束且导出成功；中途保存的训练检查点不等同于回放用 JIT 文件 |
| 提示关节布局或高度参考不匹配 | 使用检查点对应的机器人和原始任务，并保留配套 `params/` |
| AMASS 数据加载失败 | 确认 `dataset/g1/CMU/` 及其中的 `*_jpos.npy` 文件完整 |

## 数据、备份与来源

`dataset/g1/CMU/` 是重定向到 G1 的 AMASS CMU 动作子集，供 G1 任务与 ELF3 WBC 系列读取双臂运动；`elf3_flat` 保持默认双臂姿态。数据格式、ELF3 映射及时序选项见 [dataset/README.md](IsaacLab-Decoupled-WBC/dataset/README.md)，ELF3 资产说明见 [模型 README](IsaacLab-Decoupled-WBC/legged_lab/assets/elf3_dof31_current/README.md)。

仅备份基础训练代码时，可排除后续生成的 `logs/`、`environment_checks/`、`__pycache__/` 和 `*.egg-info/`。如需继续已有训练或回放策略，应额外保留对应的完整运行目录，尤其是权重、`params/` 与学生产物。

本项目基于 `chrisyrniu/IsaacLab-Decoupled-WBC` 的提交 `926756b492cb0f8cfbe1aae562e452efe3a11326`，包含 ELF3 适配及本地训练修改。代码采用 [BSD-3-Clause 许可证](IsaacLab-Decoupled-WBC/LICENSE.txt)；动作数据不受该代码许可证覆盖，使用与再分发需遵守原始数据集条款，详见[数据许可说明](IsaacLab-Decoupled-WBC/dataset/README.md#license-and-provenance)。
