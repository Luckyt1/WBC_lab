# IK 来源与轨迹库

上游数值 IK 固定为 `konodoki/com.bxi.upper_body_teleop` 的
`47087655a6c30f79175f3909ef5eb8131da74877`。
仓库只保存来源哈希和运行所需重力常数；上游 Python、YAML、URDF 通过本地准备脚本取得。

```bash
git clone https://github.com/konodoki/com.bxi.upper_body_teleop.git resources/upper_body_teleop
git -C resources/upper_body_teleop checkout 47087655a6c30f79175f3909ef5eb8131da74877
python scripts/prepare_ik_source.py --checkout resources/upper_body_teleop
```

准备脚本先校验四个文件的 SHA-256，再复制到 `ELF3_IK_SOURCE_DIR`，默认
`resources/elf3_ik_upstream`。输出目录必须不存在。运行时通过 AST 只加载数值 IK 和限速函数，
不启动 ROS 节点。外部文件依照上游条款使用，不由本项目重新授权。

## 从零生成

先按安装说明准备 `ELF3_IK_PYTHON`。以下输出目录均须不存在：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ELF3_IK_PYTHON" tools/generate_ik_arm_bank.py \
  --output resources/ik_bank_v1 --xml "$ELF3_MUJOCO_XML" \
  --train-targets 512 --test-targets 128 --workers 4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ELF3_IK_PYTHON" tools/repair_ik_arm_bank.py \
  --source resources/ik_bank_v1 --output resources/ik_bank_v2 \
  --xml "$ELF3_MUJOCO_XML" --workers 4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ELF3_IK_PYTHON" tools/plan_ik_arm_bank_v3.py \
  --source resources/ik_bank_v2 --output resources/ik_bank_v3 \
  --original-bank resources/ik_bank_v1 --xml "$ELF3_MUJOCO_XML" --workers 4
```

生成器使用独立随机种子分隔训练与测试目标，排除镜像重复。目标覆盖前伸、侧举、上举、
单臂、不对称和工作空间姿态。可选的 `--previous-targets TARGETS.json --previous-results RESULTS.json`
用于添加已有失败回归目标；必须成对提供。没有历史文件也能生成，困难目标不足时采样器使用实际可用类别。

初版缓存使用上一帧限速指令作为名义反馈。v2 验证去程和回程，检查几何、MuJoCo 自接触和终点误差；
v3 对失败路径增加有预算限制的 RRT-Connect 搜索，保留全部目标的训练/测试归属。
未找到路径的目标记录在 `rejected_targets`，不会偷偷计作通过。

新生成库不等于旧实验的原始库，特别是没有加入那批历史失败目标时。
复现旧实验需使用原库的 `manifest.json`、`train.npz`、`test.npz` 及校验哈希。
已完成 v3 库的训练重放不依赖旧库；生成和跨版本续训审计可能需要原始父库。

## v3 执行

- 单个任务包括准备、到达、保持、收回、静置。最长回合 40 秒。
- 腕部位置误差 ≤2 cm、姿态误差 ≤0.1 rad，连续满足 0.1 秒后开始保持；额外等待最多 5 秒，超时记录失败。
- 路径相对骨盆定义，通过有界 DLS 修正补偿实际腰部姿态。
- Ruckig 约束参考速度/加速度/jerk 为 6/80/1000；路径时间估计仍采用较慢的 2/4/20 参数。
- OBB 基础间距 3 mm，预测 20/80/200 ms 的几何状态，动态间距额外增加至多 5 mm；必要时有界制动或向外退让。
- 限速与防碰撞是有限采样的参考执行策略，不保证实测关节导数、连续时间无碰撞或真机安全。

重点日志：`IK/arrival_rate`、`IK/reach_timeouts`、`IK/completed_cycles`、
`IK/reference_*`、`ArmGuard/predictive_brake_fraction`、`ArmGuard/unresolved_prediction_fraction`、
`ArmMotion/actual_*`。到位率必须结合有效任务数量、保持环境数量和身体稳定性解释。
