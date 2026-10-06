# 代码与实验说明

## 当前路径

| 功能 | 入口 |
| --- | --- |
| 当前配方 | `configs/ppo_symmetry_dynamics.json`、`scripts/train_current.py` |
| 训练监督、状态保存、定期评估 | `tools/run_walk_robust_training.py` |
| 环境步进、观测、上肢执行 | `legged_lab/envs/elf3_tang/elf3_env.py` |
| 基础与 robust 设置 | `walk_cfg.py`、`walk_bxi_cfg.py`、`walk_robust_cfg.py` |
| 时间课程 | `walk_robust_curriculum.py` |
| 站立采样与奖励 | `teleop_standing.py`、`standing_position*.py`、`rewards.py` |
| 上肢数据与保护 | `wbc/amass.py`、`wbc/elf3_arm_motion.py`、`wbc/arm_guard.py` |
| 镜像变换 | `symmetry.py` |
| 辅助监督标签 | `auxiliary.py` |
| PPO 与辅助损失 | `rsl_rl/rsl_rl/algorithms/ppo.py` |
| 辅助网络 | `rsl_rl/rsl_rl/modules/actor_auxiliary.py` |
| 覆盖率、速度、加速度诊断 | `arm_diagnostics.py`、`tools/evaluate_arm_speed.py` |

未带完整路径的环境文件均位于 `legged_lab/envs/elf3_tang/`。

## 动作与观测

策略输入为 10 帧、每帧 84 维，包含骨盆角速度、投影重力、5 维指令、腰腿关节状态、前一动作和双臂关节状态。Actor 输出 15 维腰腿动作；双臂参考由外部动作模块产生，头部保持设定姿态。真值线速度与接触状态可用于 critic 和训练监督，不是导出 actor 的额外传感器输入。

辅助预测分支复用 actor 特征。当前状态预测目标为骨盆速度与脚接触；动力学分支结合实际执行动作，预测下一步腰腿关节速度和骨盆角速度。训练损失采用分项处理与尺度归一化，部署导出仅保留动作网络。

## 历史实验

当前启动配置使用**随训练轮次变化的课程**。`tools/run_teleop_curriculum.py`、`tools/teleop_curriculum.py` 等性能门槛课程工具仍保留，但默认入口不调用这些控制器。

`liftoff_rewards.py` 与 `--liftoff_rewards` 是历史 BCD 抬脚奖励实验；默认配置未启用。`stand_cfg.py`、`focus_cfg.py`、`standing_refine_cfg.py` 及相关工具用于专项训练或续训，不应与默认从零配方混为一谈。部分继承链保留旧 ELF3 / TienKung 的源码定义，其资产未分发；本文档只覆盖 31 关节 ELF3 robust 任务。

对比实验需记录完整配置、随机种子、训练更新数、同一评估场景与参考动作覆盖率。仅比较总奖励或训练时长，不能判断站立漂移、单脚支撑与有效抬脚是否改善。

## 评估命令

以下完整评估在 Isaac Python 中执行，离线部分由参数指定的 CPU Python 运行：

```bash
python tools/evaluate_dynamics_checkpoint.py \
  --checkpoint logs/walk_tang_bxi31_auxiliary_scratch/RUN_DIRECTORY_NAME/model_60000.pt \
  --output output/eval_60000 \
  --deployment_python "$ELF3_DEPLOYMENT_PYTHON"
```

需要检查点同级的 `params/`、匹配的机器人资源与 AMASS 数据。输出包含 Isaac 速度跟踪、站立、步态、辅助预测诊断，以及 MuJoCo 走停/行走评估。完整评估会使用 GPU；避免与正在运行的训练争用显存。

单独查看已导出模型的离线 MuJoCo 播放：

```bash
PYTHONPATH="$ELF3_BXI_FRAMEWORK_DIR:${PYTHONPATH:-}" \
  "$ELF3_DEPLOYMENT_PYTHON" deployment/com.bxi.walk_robust/tools/validate_sim.py \
  --xml "${ELF3_MUJOCO_XML:-$ELF3_BXI_FRAMEWORK_DIR/data/mujoco_simulation/elf3.xml}" \
  --onnx output/policy.onnx --viewer --seconds 30
```

离线适配器要求显式给出 ONNX，不会隐式加载旧模型。高度由导出模型的 `height_command` 元数据恢复；第三方模型若没有该字段则沿用适配器默认值 0.8235 m，应先确认训练约定。

本仓库提供仿真与导出链路；`deployment/` 未包含 ROS 插件注册或真机运行入口。

## 整理与验证范围

发布整理集中于独立目录、配置入口、环境变量、文档与资源排除，以及消除离线工具对旧 ONNX 的隐式依赖。未修改当前配方的 PPO 算法、奖励系数、对称性计算或课程推进规则。

`SOURCE_MANIFEST.json` 记录整理前输入文件的 SHA-256，用于追溯来源；它不是修改后文件的完整性清单。`tests/fixtures/ppo_before_auxiliary.py` 为不带辅助头时的历史 PPO 回归基线，因此测试无需训练日志归档。

CPU 测试覆盖可独立运行的数值与配置逻辑。它们不能证明新机器的完整 Isaac 环境可用，也不能替代重训与真机验证。
