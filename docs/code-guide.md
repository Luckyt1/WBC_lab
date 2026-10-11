# 训练与评估代码

## 控制分工

`elf3_env.py` 将腰腿动作交给 PPO，将上肢参考交给独立执行器。
`ik_training.py` 校验轨迹库哈希、切换 IK 输入、固定完整手臂范围和回合预算。
`wbc/ik_arm_motion.py` 负责分组采样、各运动阶段、到位等待、超时与覆盖日志。
`wbc/arm_pose_tracker.py`、`arm_guard.py`、`predictive_arm.py`、`arm_gravity.py`
分别负责腰部姿态补偿、几何检查、参考轨迹约束和重力前馈。

策略观测包含手臂位置和速度，PPO 因此可以学习应对上肢运动。
`rsl_rl/rsl_rl/modules/actor_auxiliary.py` 从策略特征估计当前骨盆速度、双脚接触；
动力学分支额外输入当前执行动作，预测下一步腰腿关节速度和骨盆角速度。
辅助头的监督真值来自仿真，部署导出只保留动作网络。

`symmetry.py` 实现观测、动作和辅助标签镜像；不要求两腿同时做相同动作。
`liftoff_rewards.py` 的 BCD 配方将重心居中奖励限制在站立，膝距改为骨盆航向系的有符号横向距离，
行走时增加载荷转移奖励。这里保留这套既有实现，不以提取代码为由调整奖励。

## 评估

导出前保留检查点旁 `params/agent.yaml` 与 `params/env.yaml`。以 README 的方式生成 `output/policy.onnx` 后：

```bash
PYTHONPATH="$ELF3_BXI_FRAMEWORK_DIR:${PYTHONPATH:-}" \
  "$ELF3_DEPLOYMENT_PYTHON" tools/evaluate_walk_mujoco_standing.py \
  --mod deployment/com.bxi.walk_robust --xml "$ELF3_MUJOCO_XML" \
  --onnx output/policy.onnx --output output/standing.json

PYTHONPATH="$ELF3_BXI_FRAMEWORK_DIR:${PYTHONPATH:-}" \
  "$ELF3_DEPLOYMENT_PYTHON" tools/diagnose_walk_mujoco.py \
  --mod deployment/com.bxi.walk_robust --xml "$ELF3_MUJOCO_XML" \
  --onnx output/policy.onnx --output output/gait --include-mixed-yaw

"$ELF3_IK_PYTHON" tools/evaluate_ik_policy.py \
  --bank "$ELF3_IK_BANK" --onnx output/policy.onnx \
  --xml "$ELF3_MUJOCO_XML" --output output/ik_workspace --full
```

前两个入口测固定手臂时的漂移、走停和步态；第三个测 IK 扰动，分开报告未见目标与困难姿态回归。
单独验证速度追踪不能代表全工作空间手臂扰动已经稳定。`--smoke` 只运行一个 IK 场景，用于接线检查。

完整检查点评估由训练环境启动：

```bash
python tools/evaluate_dynamics_checkpoint.py \
  --checkpoint /path/to/RUN/model_32000.pt --output output/checkpoint_eval \
  --deployment_python "$ELF3_DEPLOYMENT_PYTHON" \
  --ik-bank "$ELF3_IK_BANK" --ik-python "$ELF3_IK_PYTHON"
```

其中的 Isaac 评估含旧 AMASS 测试，不是完整 IK 分布的替代品。
部署适配器依赖 BXI 框架的 `JointPolicy` 等接口；仓库没有包含 ROS 控制节点和预训练模型。

## 提取边界

保留 ELF3 Tang 任务族和其公共框架依赖，避免裁掉导入链。旧 AMASS、课程门槛和消融工具仍在源码中，
但默认 `scripts/train_current.py` 不启用门槛课程。
提取时只增加便携路径、资源准备/预检、启动配方、测试样本与文档；核心训练、奖励、IK v3 执行逻辑沿用来源项目。
`SOURCE_MANIFEST.json` 记录提取时输入文件哈希，便于核对来源；不是训练性能报告。
