# WBC 基础训练代码备份

此目录保留当前 ELF3 / G1 训练源码、任务配置、机器人 USD 模型、AMASS 动作数据、教师训练与学生蒸馏入口，以及安装和验证脚本。

已移除训练检查点与日志、诊断输出、历史实验报告和专用启动工具、旧 `elf3_lab` 工程、重复压缩包、构建缓存、示例权重、实机部署目录及嵌套 Git 历史。**本备份不含已训练权重，不能直接恢复原来的训练轮数。** 原有算法与任务配置源码保留，默认入口从零开始训练。

## 恢复环境

在解压后的本目录执行：

```bash
bash setup_wbc_conda.sh
bash setup_wbc_training.sh
```

安装脚本沿用本机原有版本：Python 3.11、PyTorch 2.7.0 / CUDA 12.8、Isaac Sim 5.0.0、Isaac Lab 2.2.0、RSL-RL 2.3.3。需要 NVIDIA GPU 及兼容驱动；安装过程需要联网。详细包版本保存在 `isaac_wbc_requirements.lock.txt`，本地项目由安装脚本以 editable 模式安装。

默认使用 `$HOME/miniconda3` 下的 `isaac_wbc` 环境。激活脚本也支持 PATH 中已有的 Conda；如安装位置或环境名不同，请先设置：

```bash
export WBC_CONDA_ROOT=/path/to/miniconda3
export WBC_CONDA_ENV=isaac_wbc
```

## 训练

```bash
# 默认：ELF3 WBC、1024 个环境、60000 轮，每 1000 轮保存
bash train.sh

# 调整规模和训练轮数
bash train.sh --num_envs=4096 --max_iterations=10000 --save_interval=500

# 只预览入口命令
bash train.sh --dry-run --num_envs=4096

# 同一个命名实验自动恢复自身最新检查点，max_iterations 为总目标轮数
bash train.sh --auto_resume --run_name=baseline --max_iterations=60000

# 双臂保持默认姿态的平地任务
WBC_TASK=elf3_flat bash train.sh
```

默认任务的结果保存在 `IsaacLab-Decoupled-WBC/logs/elf3_wbc/`。`elf3_wbc` 使用 AMASS 双臂运动，`elf3_flat` 保持双臂默认姿态；数据目录不能直接删除。其他已注册任务保留在 `legged_lab/envs/__init__.py`。

## 学生蒸馏和回放

先完成教师训练，再选择真实生成的目录和检查点：

```bash
source ./activate_wbc.sh  # 激活环境并进入 IsaacLab-Decoupled-WBC
export WBC_TASK=elf3_wbc
export HTD_RUN=baseline       # 替换为 logs/elf3_wbc/ 下的实际目录名
export HTD_CKPT=model_1000.pt # 替换为实际检查点文件名
bash scripts/train_student.sh
bash scripts/play_teacher.sh --headless --max_steps=1000
bash scripts/play_student.sh --headless --max_steps=1000
```

## 验证与备份

在本目录运行 `bash verify_wbc_training.sh` 可检查依赖、CUDA、单元测试、ELF3 仿真和短教师/学生训练。该脚本会生成新的 `environment_checks/` 和训练日志。

云端备份时保存本目录即可；后续产生的 `logs/`、`environment_checks/`、`__pycache__/`、`*.egg-info/` 等无需加入基础代码备份。`.gitignore` 已列出这些内容，但普通网盘上传不会自动应用 Git 忽略规则。

主项目基于 `chrisyrniu/IsaacLab-Decoupled-WBC` 的 `926756b492cb0f8cfbe1aae562e452efe3a11326` 提交并包含本地修改。代码许可证见 [LICENSE.txt](IsaacLab-Decoupled-WBC/LICENSE.txt)，动作数据来源与许可说明见 [dataset/README.md](IsaacLab-Decoupled-WBC/dataset/README.md)。
