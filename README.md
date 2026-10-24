# WBC_lab · ELF3 训练与快速部署

当前默认版本对应 **2026-10-23 启动的训练**：取消腰部 yaw / roll / pitch 指令，双侧脚踝左右最大 5 rad/s、前后最大 15 rad/s；保留整机转向、可变站高和 IK 手臂扰动。

## 本机快速使用

```bash
cd /home/cheng/elf3_lab/github/WBC_lab
./deploy.sh check
./deploy.sh train --dry-run
```

本机已经准备资源与 `.env`。检查通过后，可显式启动一个新任务：

```bash
./deploy.sh train --output logs/my_run --background
./deploy.sh status --output logs/my_run
tail -f logs/my_run/train.log
```

当前默认是 12,556 环境、从零训练 80,000 次 PPO 更新。原有训练仍在 `workspace/no_waist_pose_ankle_limits_20261023` 中运行；本次整理不接管它。使用本目录的 `train` 会启动新任务，请在 GPU 有空闲时执行。

小规模验证只运行 16 环境、2 次 PPO 更新，读取 PhysX 脚踝上限，并检查保存的模型数值：

```bash
./deploy.sh smoke
```

## 搬到另一台机器

推荐使用本机生成的 **含资源部署包** `dist/WBC_lab_resources_*.tar.gz`。包含源代码、机器人 USD/网格、MuJoCo 模型、BXI 离线框架、IK v3 轨迹和本轮评估使用的数据；不包含 Python 环境、仿真器、模型权重或训练日志。

在 Linux x86_64、Python 3.11、可用 NVIDIA 驱动的机器上：

```bash
sha256sum -c WBC_lab_resources_*.tar.gz.sha256
tar -xzf WBC_lab_resources_YYYYMMDD_HHMMSS.tar.gz
cd WBC_lab
./deploy.sh setup --python python3.11
./deploy.sh check --verify-bundle
./deploy.sh smoke
./deploy.sh train --output logs/my_run --num-envs 4096 --background
```

`setup` 首次联网安装 Isaac Sim 5.0.0、Isaac Lab 2.2.0、CUDA PyTorch 2.7.0，以及独立的 CPU 评估 / Pinocchio 环境，随后编译 Ruckig。首次下载仿真器和扩展较大，安装时间由网络和缓存决定。详细系统依赖、已有环境复用及 NVIDIA 首次启动说明见 [安装说明](docs/installation.md)。

资源路径默认相对于仓库，移动目录不需要逐个改路径。已创建的 Python 环境不要直接跨机器复制；在目标机器重新运行 `setup`。配置说明见 [.env.example](.env.example)。

## 当前训练配方

唯一默认入口是 [configs/current.json](configs/current.json)：

| 项目 | 当前设置 |
|---|---|
| 任务 | `walk_elf3_tang_bxi_robust` |
| 观测 / 动作 | 840 维历史观测 / 15 维腰腿动作 |
| 训练周期 | 50 Hz |
| 并行环境 / 随机种子 | 12,556 / 42 |
| PPO 更新 / 保存 / 完整评估 | 80,000 / 每 500 / 每 5,000 次 |
| 骨盆目标高度 | 0.4–0.8 m，前 20,000 次逐步展开 |
| 指令比例 | 站立 50%，前后走 25%，侧移 12.5%，整机转向 12.5% |
| 腰部姿态指令 | yaw / roll / pitch 关闭；腰部关节仍由策略控制 |
| 双侧 ankle_x（左右） | 速度绝对值上限 **5 rad/s** |
| 双侧 ankle_y（前后） | 速度绝对值上限 **15 rad/s** |
| 手臂与辅助训练 | IK v3；镜像对称性；动力学辅助头；BCD 奖励 |

脚踝上限通过 `ImplicitActuatorCfg.velocity_limit_sim` 写入 PhysX，训练开始前会回读全部环境的四个关节上限；不符合配置就报错。其他关节限速、PD 参数和力矩上限保持原配方。

`configs/ik_v3.json` 保留旧固定 0.8 m 配方供明确选择，已不是默认。高度、指令和训练来源见 [版本记录](docs/training-snapshot.md)。

## 常用命令

```bash
# 较小显存：减少并行环境，其余当前配方不变
./deploy.sh train --num-envs 4096 --output logs/run_4096 --background

# 前台运行；Ctrl+C 停止该任务及其评估子进程
./deploy.sh train --output logs/foreground

# 同一部署内续训：保留原检查点旁的 params/，指定新增更新次数
./deploy.sh train --resume-checkpoint /path/to/RUN/model_25000.pt \
  --max-iterations 55000 --output logs/continue --background

# 导出动作网络，并与 PyTorch 比较 32 组输入
./deploy.sh export --checkpoint /path/to/RUN/model_25000.pt --output output/policy.onnx

# CPU 回归测试
./deploy.sh test

# 重新生成可搬迁部署包及 SHA-256 校验文件
./deploy.sh pack --with-resources

# 纯源码包：不含外部资源
./deploy.sh pack
```

续训会验证原配方、策略、辅助头和两个优化器；不要把旧腰部 RPY 模型当作当前版本继续训练。跨目录迁移已有检查点时，原配置包含绝对资源路径，需额外做迁移核验；本部署包默认从零训练。

`state.json` 记录进度、检查点和评估结果，`train.log` 为训练日志，`supervisor.log` 为后台监督程序日志。训练完成仍会运行最终评估，不会自动部署到真机。

## 源码与资源

```text
configs/current.json             当前训练参数
scripts/ + deploy.sh             安装、检查、训练、状态、验证、打包
legged_lab/                      仿真环境、关节设置、奖励、IK 执行
rsl_rl/                          本项目修改的 PPO 与辅助预测头
tools/                           监督训练、评估、ONNX 导出和 IK 工具
deployment/com.bxi.walk_robust/   离线 MuJoCo 推理适配器
resources/                       本机训练资源，Git 忽略，可随部署包搬迁
requirements/                    三套环境的依赖与训练版本约束
provenance/                      当前来源与历史提取记录
tests/                           CPU 回归测试及小型测试数据
dist/                            自动生成的源码包 / 含资源包
```

Git 仓库只管理代码、说明和小型测试数据；`resources/`、`.env`、环境、日志和权重均被忽略。含资源包用于用户已有资源的本地迁移，与纯源码仓库区分。第三方信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

验证范围见 [验证记录](docs/validation.md)。训练代码可运行与策略效果合格是两件事，启动验证不代表已学会行走。
