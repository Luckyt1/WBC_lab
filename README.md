# WBC_lab · ELF3 / GL-165

ELF3 上肢 IK 扰动下的站立与行走 PPO 训练代码，从 `elf3_lab` 的 IK 执行 v3 提取。
策略控制腰腿 15 个关节，双臂执行程序化 IK 产生的“到达—保持—收回”轨迹。
训练目标是保持速度追踪、站立稳定和较小漂移；手臂作为外部扰动，不由 PPO 输出动作。

此目录是独立源码仓库。包含训练、轨迹生成、限速和防碰撞、ONNX 导出、MuJoCo 评估及 CPU 测试。
不包含训练日志、预训练权重、完整轨迹库、机器人网格或仿真器。
本仓库对应 IK 行走抗干扰版本，不包含另外的 `half/elf3_stand` 站立项目。

## 任务入口

| 机器人 | 训练入口 | 使用说明 |
| --- | --- | --- |
| GL-165 | `bash scripts/train_gl165.sh` | [资产准备、训练、预览与验证](docs/gl165.md) |
| ELF3 | `python scripts/train_current.py --output logs/my_run` | 下方 ELF3 配方与[安装说明](docs/installation.md) |

GL-165 的独立任务为 `walk_gl165_wbc`：策略控制腰腿，双臂执行连续 IK 扰动，训练行走与站立抗扰动。
不追踪腰部姿态或高度；30% 的手臂目标为侧伸动作，20% 为过头顶动作。
先按 [GL-165 使用说明](docs/gl165.md) 准备外部资产与轨迹库；路径可通过 `.env.example` 中的 `GL165_*` 变量配置。

## ELF3 提取记录：2026-10-11

此副本对应 10 月 10–11 日从 `model_32000.pt` 继续训练的 IK v3 版本。
训练按用户要求停止于日志迭代 56,296，最后已保存检查点为 56,000；讨论的 45,000 轮模型使用同一套源码与配方。
本次已核对训练归档中的 218 个源码文件，来源工作区无改动。
保留已有独立版本的外部资源路径适配，训练核心、奖励与 IK 执行逻辑保持一致。

训练记录、检查点对应关系和本机位置见 [本次训练快照](docs/training-snapshot.md)，
机器可读记录见 [provenance/training_snapshot.json](provenance/training_snapshot.json)。
源码不包含模型权重、完整轨迹库、训练日志或机器人网格；这些文件仍在原训练目录。

## ELF3 默认配方

[configs/ik_v3.json](configs/ik_v3.json) 保存以下启动配置：

| 项目 | 配置 |
| --- | --- |
| 任务 | `walk_elf3_tang_bxi_robust` |
| 策略输入 / 输出 | 10 × 84 = 840 维历史观测 / 15 维腰腿动作 |
| 控制周期 | 20 ms，50 Hz |
| 初始化 / 并行环境 | 从零训练，4096 环境 |
| 训练 / 保存 / 完整评估 | 80,000 次 PPO 更新 / 每 500 / 每 5,000 次 |
| 骨盆目标高度 | 固定 0.8 m |
| 指令比例 | 50% 站立，25% 前后走，12.5% 侧移，12.5% 转向 |
| 手臂 | 15% 中立，85% IK 任务；轨迹库有困难目标时，对其提高采样权重 |
| 对称性 / 辅助头 | 镜像增强与损失；当前状态和动作条件动力学预测 |
| BCD 奖励 | 启用 `bcd_v1`，见 [代码说明](docs/code-guide.md) |
| 手臂课程 | 从开始使用完整目标范围，无行走精度门槛驱动的课程升级 |
| 推扰 / 复位速度 | 保留原配方随训练轮数推进的时间课程 |

IK v3 保留实时腰部姿态补偿、实际到位判定、OBB 几何检查和 Ruckig 参考轨迹约束。
参考速度、加速度、jerk 上限分别为 6 rad/s、80 rad/s²、1000 rad/s³；这些是**指令轨迹**约束。
单回合上限 40 秒；失稳仍可提前终止。辅助预测头仅参与训练，导出的 ONNX 只含动作网络。

## 安装和数据

按 [安装说明](docs/installation.md) 准备训练环境、CPU 评估环境、IK 环境及机器人资产。
在单独准备的 Isaac 环境里安装此副本，不要覆盖正在训练的 editable 安装：

```bash
python -m pip install -e ./rsl_rl --no-deps
python -m pip install -e '.[dev,viewer]'
python tools/build_arm_ruckig.py
cp .env.example .env
# 编辑 .env，填入外部资源及 Python 的绝对路径
set -a
source .env
set +a
```

上游 IK 使用固定 commit；准备方法和从零生成 v1 → v2 → v3 轨迹库的命令见
[IK 数据说明](docs/ik-data.md)。已有相同版本轨迹库可直接设置 `ELF3_IK_BANK`。
小型 `tests/fixtures/ik_banks/` 仅用于单元测试，启动预检会拒绝将它作为训练数据。

IK 训练本身不读取 AMASS；默认监督程序的完整评估仍包含旧 AMASS 能力保留测试，因此也需准备该数据。

## 训练

从仓库根目录执行：

```bash
python scripts/check_resources.py --deployment --amass
python scripts/train_current.py --output logs/ik_v3_run --dry-run
python scripts/train_current.py --output logs/ik_v3_run
```

`--dry-run` 只打印命令。默认从零训练；恢复已有 IK 检查点时显式指定：

```bash
python scripts/train_current.py --output logs/ik_v3_continue \
  --resume-checkpoint /path/to/RUN/model_32000.pt --max-iterations 48000
```

`--max-iterations` 表示此次新增 PPO 更新数。检查点旁的 `params/` 和原轨迹库需保留；
续训会核对策略、优化器与配方，不应随意改写保存配置里的资产或轨迹库路径。
每次启动使用新输出目录。状态在 `logs/<run>/state.json`，汇总在 `RESULTS.md`，
实际检查点路径由 `latest_checkpoint` 给出。监督程序不会自动部署模型。

## Play 和导出

`RUN_DIRECTORY_NAME` 是检查点所在目录名：

```bash
python legged_lab/scripts/play.py \
  --task walk_elf3_tang_bxi_robust --num_envs 1 \
  --experiment_name walk_tang_bxi31_auxiliary_scratch \
  --load_run RUN_DIRECTORY_NAME --checkpoint model_32000.pt \
  --pelvis_height 0.8 --ik_arm_bank "$ELF3_IK_BANK" \
  --command 0 0 0 0.8 0
```

这是固定零速度命令下的 IK 手臂扰动播放。改成 `--command 0.3 0 0 0.8 0` 可观察前向行走。
该播放关闭推扰和观测噪声，用于看动作；完整评估应使用评估工具。

```bash
"$ELF3_DEPLOYMENT_PYTHON" tools/export_walk_actor_onnx.py \
  --checkpoint /path/to/RUN/model_32000.pt --output output/policy.onnx

"$ELF3_IK_PYTHON" tools/evaluate_ik_policy.py \
  --bank "$ELF3_IK_BANK" --onnx output/policy.onnx \
  --xml "$ELF3_MUJOCO_XML" --output output/ik_workspace
```

评估依赖外部 BXI Python 框架。独立站立漂移、走停漂移和行走评估命令见
[代码说明](docs/code-guide.md)。关注速度误差、漂移、存活率、滑步；腕部到位率作为扰动覆盖与执行诊断。

## 测试与目录

```bash
python -m pip install -r requirements/cpu-tests.txt
python tools/build_arm_ruckig.py
python -m pytest -q -p no:cacheprovider tests
```

```text
configs/                         IK v3 默认训练配方
scripts/                         启动、资源检查和上游 IK 准备
legged_lab/envs/elf3_tang/        环境、奖励、IK 轨迹重放、对称性
legged_lab/envs/elf3_tang/wbc/    腰部补偿、防碰撞、重力补偿、Ruckig 执行
legged_lab/envs/gl165/           GL-165 环境、奖励、步态、连续 IK 与检查点恢复
legged_lab/assets/gl165/         GL-165 关节名称、参数来源与 Isaac 资产配置
rsl_rl/                          PPO 与辅助预测头修改
tools/                           IK 生成/路径修复、训练监督、导出和评估
third_party/ruckig/               固定版本 MIT 原生轨迹生成源码
deployment/com.bxi.walk_robust/   显式加载 ONNX 的离线 MuJoCo 适配器
tests/                           CPU 回归测试及少量轨迹样本
resources/                       本机外部资源，Git 忽略
```

本地检查结果及未覆盖范围见 [验证记录](docs/validation.md)。采样防碰撞和仿真通过不等于真机验证。

## GitHub 仓库

此源码同步到 [Luckyt1/WBC_lab](https://github.com/Luckyt1/WBC_lab)。获取包含 GL-165 的 `jili165` 分支：

```bash
git clone --branch jili165 https://github.com/Luckyt1/WBC_lab.git
cd WBC_lab
```

原提交历史保留。GL-165 与 ELF3 使用各自的训练入口、机器人资产和轨迹库；旧版代码可从历史提交查阅。

`.gitignore` 排除模型、日志、完整数据、编译产物和 `.env`；提交前检查文件列表。
第三方来源和组件许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。外部资产和上游 IK 源码需自行准备。
